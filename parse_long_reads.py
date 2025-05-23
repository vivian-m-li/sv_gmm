import ast
import csv
import re
import os
import argparse
import subprocess
import pysam
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from helper import get_sv_lookup, get_sv_stats_collapsed_df
from typing import List
from collections import defaultdict

SCRATCH_DIR = "/scratch/Users/vili4418/"


def parse_long_read_samples():
    file = "long_reads/raw_1kg_ont_vienna_hg38.txt"
    root = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1KG_ONT_VIENNA/hg38"
    df = pd.DataFrame(columns=["sample_id", "cram_file", "indexed_cram_file"])
    with open(file, "r") as f:
        soup = BeautifulSoup(f, "html.parser")
        table = soup.find("table")
        rows = table.find_all("tr")
        for row in rows:
            if ".cram" in row.text:
                columns = row.find_all("td")
                for col in columns:
                    if ".cram" in col.text:
                        pattern = r"([\S]+)\.hg38\.cram(.crai)?"
                        sample_id = re.search(pattern, col.text).group(1)
                        file_name = f"{root}/{col.text}"
                        if ".crai" in col.text:
                            df.loc[
                                df["sample_id"] == sample_id,
                                "indexed_cram_file",
                            ] = file_name
                        else:
                            df.loc[len(df)] = [sample_id, file_name, ""]
    df.to_csv("long_reads/long_read_samples.csv", index=False)


def read_cigars_from_file(bam_file: str, sv_deletion_size: int):
    try:
        bam = pysam.AlignmentFile(bam_file, "rb")
        bam.fetch()
    except ValueError as e:
        raise ValueError(
            f"Error opening/fetching BAM file {bam_file}: {e}"
        ) from e

    deletions = []
    for read in bam.fetch():
        if read.is_unmapped:
            continue
        cigar_string = read.cigarstring
        if not cigar_string or "D" not in cigar_string:
            continue

        for match in re.finditer(r"(\d+)D", cigar_string):
            deletion_size = int(match.group(1))
            # account for 500 bp of tolerance
            # set 25bp as threshold - smallest sv is 51bp
            if deletion_size >= max(25, sv_deletion_size - 500):
                ref_pos = read.reference_start
                cigar_tuples = read.cigartuples
                for op, length in cigar_tuples:
                    if op == 2 and length == deletion_size:  # 2 = deletion
                        deletions.append(
                            {
                                "start": ref_pos,
                                "stop": ref_pos + deletion_size,
                                "length": deletion_size,
                            }
                        )
                        break

                    if op in [0, 2, 3, 7, 8]:  # M, D, N, =, X
                        ref_pos += length

    return deletions


def remove_blank_lines_from_evidence():
    files = os.listdir("long_reads/evidence")
    for file in files:
        if ".csv" not in file:
            continue
        with open(os.path.join("long_reads/evidence", file), "r") as f:
            lines = f.readlines()
        with open(os.path.join("long_reads/evidence", file), "w") as f:
            for line in lines:
                row = line.split(",")
                if len(row) > 1:
                    f.write(line)


def write_samples_to_redo():
    file_dir = "/Users/vili4418/sv/eofiles"
    files = os.listdir(file_dir)
    samples_to_redo = defaultdict(list)
    for file in files:
        pattern = r"[\S]*[\/]*download_long_read_evidence_[\S]+\.out"
        if not re.search(pattern, file):
            continue
        with open(os.path.join(file_dir, file), "r") as f:
            for line in f.readlines():
                if line.startswith("Redo"):
                    pattern = r"Redo sample ([\S]+)-([\S]+)"
                    match = re.search(pattern, line)
                    sv_id = match.group(1)
                    sample_id = match.group(2)
                    samples_to_redo[sample_id].append(sv_id)
                

    with open("long_reads/redo_samples.txt", "w") as f:
        for sample_id, sv_ids in samples_to_redo.items():
            f.write(f"{sample_id}: {', '.join(sv_ids)}\n")


def remove_bam_file(file: str, *, scratch: bool = False):
    file_path = os.path.join("long_reads/reads", file)
    if scratch:
        file_path = os.path.join(SCRATCH_DIR, file_path)
    try:
        os.remove(file_path)
        os.remove(f"{file_path}.bai")
    except FileNotFoundError:
        return


def write_long_read_evidence(
    sv_id: str, sample_id: str, deletions: List
) -> bool:
    # assumes file is in home directory and not in scratch
    if len(deletions) == 0:
        return
    file_name = f"long_reads/evidence/{sv_id}.csv"
    with open(file_name, "a") as f:
        csv_writer = csv.writer(f)
        row = [sample_id]
        for deletion in deletions:
            row.extend([deletion["start"], deletion["stop"]])
        csv_writer.writerow(row)


def write_all_long_read_evidence():
    # rewrite bam files into CSVs to save memory
    sv_lookup = get_sv_lookup()
    files = os.listdir("long_reads/reads")
    for file in files:
        if ".bam" not in file or ".bai" in file:
            continue
        pattern = r"([\S]+)-([\S]+)\.bam"
        match = re.search(pattern, file)
        sv_id = match.group(1)
        sample_id = match.group(2)

        sv_row = sv_lookup[sv_lookup["id"] == sv_id]
        start = sv_row["start"].values[0]
        stop = sv_row["stop"].values[0]
        svlen = stop - start

        try:
            deletions = read_cigars_from_file(
                os.path.join("long_reads/reads", file), svlen
            )
        except (ValueError, OSError):
            remove_bam_file(file)
            continue

        write_long_read_evidence(sv_id, sample_id, deletions)
        remove_bam_file(file)


def get_processed_samples(sv_id: str):
    file_name = f"long_reads/evidence/{sv_id}.csv"
    if not os.path.exists(file_name):
        return {}

    sample_evidence = {}
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            sample_evidence[row[0]] = []
            for i in range(1, len(row), 2):
                sample_evidence[row[0]].append(
                    {
                        "start": int(row[i]),
                        "stop": int(row[i + 1]),
                        "length": int(row[i + 1]) - int(row[i]),
                    }
                )
    return sample_evidence


def get_sv_region(sv_id: str, tolerance: int):
    sv_lookup = get_sv_lookup()
    row = sv_lookup[sv_lookup["id"] == sv_id]
    start = row["start"].values[0] - tolerance
    stop = row["stop"].values[0] + tolerance
    region = f"chr{row['chr'].values[0]}:{start}-{stop}"
    sv_len = stop - start - 2 * tolerance
    return region, sv_len


def get_bam_file(
    sv_id: str,
    sample_id: str,
    *,
    region: str,
    cram_file: str,
    scratch: bool = False,
):
    output_file_name = f"{sv_id}-{sample_id}.bam"
    output_file = os.path.join("long_reads/reads", output_file_name)
    if scratch:
        output_file = os.path.join(SCRATCH_DIR, output_file)
    indexed_output_file = f"{output_file}.bai"

    # check that both the file and indexed file exist
    if not os.path.exists(output_file) and not os.path.exists(
        indexed_output_file
    ):
        subprocess.run(
            ["bash", "get_cigar.sh"] + [cram_file, region, output_file],
            capture_output=True,
            text=True,
        )

    return output_file


def get_long_read_svs(
    sv_id: str,
    samples: List[str],
    *,
    tolerance: int = 100,
    scratch: bool = False,
):
    region, sv_len = get_sv_region(sv_id, tolerance)
    long_reads = pd.read_csv("long_reads/long_read_samples.csv")
    deletions = get_processed_samples(sv_id)
    for sample_id in samples:
        if sample_id in deletions:
            continue

        row = long_reads[long_reads["sample_id"] == sample_id]
        if row.empty:
            print(f"Sample {sample_id} not found in long reads")
            continue
        cram_file = row["cram_file"].values[0]
        output_file = get_bam_file(
            sv_id,
            sample_id,
            region=region,
            cram_file=cram_file,
            scratch=scratch,
        )

        # possible ValueError
        deletions[sample_id] = read_cigars_from_file(output_file, sv_len)
        write_long_read_evidence(sv_id, sample_id, deletions[sample_id])
        remove_bam_file(output_file)

    return deletions


def get_all_long_reads():
    split_svs = pd.read_csv("1kgp/split_svs.csv")
    sv_ids = split_svs["sv_id"].unique()
    df = get_sv_stats_collapsed_df()
    df = df[df["id"].isin(sv_ids)]

    new_df = pd.DataFrame(
        columns=[
            "id",
            "sv_id",
            "chr",
            "sample_ids",
            "start",
            "stop",
            "length",
            "lr_start",
            "lr_stop",
            "lr_length",
            "start_diff",
            "stop_diff",
            "length_diff",
        ]
    )
    for _, row in df.iterrows():
        sv_id = row["id"]
        modes = ast.literal_eval(row["modes"])
        modes = sorted(modes, key=lambda x: x["start"])
        for i, mode in enumerate(modes):
            mode_id = f"{sv_id}_{i + 1}"
            all_sample_ids = mode["sample_ids"]
            deletions = get_long_read_svs(sv_id, all_sample_ids)

            starts = []
            stops = []
            lengths = []
            sample_ids = []
            for sample_id, sv in deletions.items():
                if len(sv) == 0:
                    continue
                sample_ids.append(sample_id)
                starts.append(np.mean([x["start"] for x in sv]))
                stops.append(np.mean([x["stop"] for x in sv]))
                lengths.append(np.mean([x["length"] for x in sv]))

            if len(starts) == 0:
                new_df.loc[len(new_df)] = [
                    mode_id,
                    sv_id,
                    row["chr"],
                    sample_ids,
                    mode["start"],
                    mode["end"],
                    mode["length"],
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
                continue

            start = np.mean(starts)
            stop = np.mean(stops)
            length = np.mean(lengths)
            start_diff = abs(start - mode["start"])
            stop_diff = abs(stop - mode["end"])
            length_diff = abs(length - mode["length"])

            new_df.loc[len(new_df)] = [
                mode_id,
                sv_id,
                row["chr"],
                sample_ids,
                mode["start"],
                mode["end"],
                mode["length"],
                start,
                stop,
                length,
                start_diff,
                stop_diff,
                length_diff,
            ]

    new_df.to_csv("long_reads/split_svs_lr.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare between two CIGAR strings to identify differences in SV deletions"
    )
    parser.add_argument("-s", type=str, help="First BAM/CRAM file")
    parser.add_argument(
        "-id", type=str, help="SV ID both samples originate from"
    )
    parser.add_argument(
        "-t",
        type=int,
        default=100,
        help="Tolerance",
    )

    args = parser.parse_args()
    if args.id is None and args.s is None:
        get_all_long_reads()
    else:
        if args.id is None:
            raise ValueError("SV ID is required")
        if args.s is None:
            raise ValueError("Sample ID is required")

        get_long_read_svs(
            args.id,
            [args.s],
            tolerance=args.t,
        )
