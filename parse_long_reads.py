import ast
import csv
import re
import os
import argparse
import subprocess
import pysam
import pandas as pd
from bs4 import BeautifulSoup
from helper import get_sv_lookup, get_sv_stats_collapsed_df
from typing import List, Dict, Tuple
from collections import defaultdict

SCRATCH_DIR = "/scratch/Users/vili4418/"

"""Standalone functions"""


def parse_long_read_samples():
    """Parse available 1kGP long read samples from the online data. Saves the available sample IDs and the link to their cram files."""
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


def remove_blank_lines_from_evidence():
    """Removes blank lines that are introduced when appending to CSV files from each SV evidence file."""
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
    """Parse fiji output file to find samples that need to be redone."""
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


"""File I/O helpers"""


def get_bam_file(
    sv_id: str,
    sample_id: str,
    *,
    region: str,
    cram_file: str,
    scratch: bool = False,
):
    """Download the bam file for a given region from the cram file if it doesn't already exist."""
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


def remove_bam_file(file: str, *, scratch: bool = False):
    """Deletes bam and bai files from scratch to save space"""
    file_path = os.path.join("long_reads/reads", file)
    if scratch:
        file_path = os.path.join(SCRATCH_DIR, file_path)
    try:
        os.remove(file_path)
        os.remove(f"{file_path}.bai")
    except FileNotFoundError:
        return


def write_sample_long_read_evidence(
    sv_id: str, sample_id: str, deletions: List
) -> bool:
    """Writes the long read evidence from one sample to a file for a given SV."""
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


"""Helper functions"""


def get_sv_region(sv_id: str, tolerance: int) -> Tuple[str, int]:
    """For a given SV, returns the frame (chr:start-stop) to extract reads from."""
    sv_lookup = get_sv_lookup()
    row = sv_lookup[sv_lookup["id"] == sv_id]
    start = row["start"].values[0] - tolerance
    stop = row["stop"].values[0] + tolerance
    region = f"chr{row['chr'].values[0]}:{start}-{stop}"
    sv_len = stop - start - 2 * tolerance
    return region, sv_len


def get_processed_samples(sv_id: str) -> Dict[str, List[dict]]:
    """For a given SV, returns the evidence (start, stop, and length of each long read for each sample) that has already been processed."""
    file_name = f"long_reads/evidence/{sv_id}.csv"
    if not os.path.exists(file_name):
        return {}

    sample_evidence = {}
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            sample_evidence[row[0]] = []  # row[0] is the sample id
            for i in range(1, len(row), 2):
                sample_evidence[row[0]].append(
                    {
                        "start": int(row[i]),
                        "stop": int(row[i + 1]),
                        "length": int(row[i + 1]) - int(row[i]),
                    }
                )
    return sample_evidence


"""Long read data processing functions"""


def filter_evidence(sv_id: str, start, end, svlen):
    """For an SV, filters out evidence from each sample that is not within some range of the SV region"""
    file = f"long_reads/evidence/{sv_id}.csv"
    with open(file, "r") as f:
        for row in f:
            evidence = []
            vals = row.split(",")
            sample_id = vals[0]
            vals = vals[1:]
            for e_start, e_end in zip(vals[::2], vals[1::2]):
                e_start, e_end = int(e_start), int(e_end)
                # evidence is within twice the svlen of the evidence
                if e_start >= start - svlen * 2 and e_end <= end + svlen * 2:
                    evidence.append(
                        {
                            "start": e_start,
                            "stop": e_end,
                            "length": e_end - e_start,
                        }
                    )
            write_sample_long_read_evidence(sv_id, sample_id, evidence)


def filter_evidence_all():
    "For all SVs, filter evidence to be within some range of the SV region (standalone function)"
    sv_lookup = get_sv_lookup()
    files = os.listdir("long_reads/evidence")
    for file in files:
        sv_id = file.split(".")[0]
        row = sv_lookup[sv_lookup["id"] == sv_id]
        start = row["start"].values[0]
        stop = row["stop"].values[0]
        svlen = stop - start
        filter_evidence(sv_id, start, stop, svlen)


def read_cigars_from_file(bam_file: str, sv_deletion_size: int):
    """For a given bam file (sample-specific, filtered by SV region), read the cigar strings to identify deletions that match the given SV size."""
    try:
        # open the sample's bam file (corresponding to an SV region), read the cigar strings that
        bam = pysam.AlignmentFile(bam_file, "rb")
        bam.fetch()
    except ValueError as e:
        # some files are corrupted?
        raise ValueError(
            f"Error opening/fetching BAM file {bam_file}: {e}"
        ) from e

    deletions = []
    for read in bam.fetch():
        # check if the read has been mapped to a reference sequence
        if read.is_unmapped:
            continue

        # a string representing how the read aligns to the reference sequence
        cigar_string = read.cigarstring

        # D indicates there's a deletion in this region
        if not cigar_string or "D" not in cigar_string:
            continue

        # look for all deletions in this cigar string
        for match in re.finditer(r"(\d+)D", cigar_string):
            # the digit in front of the D in the cigar string
            deletion_size = int(match.group(1))

            # TODO:M deletion size should be in a range
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


def write_all_long_read_evidence():
    """Search through all bam files to update the deletions for each SV/sample and write to evidence files."""
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

        write_sample_long_read_evidence(sv_id, sample_id, deletions)
        remove_bam_file(file)


def get_long_read_svs(
    sv_id: str,
    samples: List[str],
    *,
    tolerance: int = 100,
    scratch: bool = False,
) -> Dict[str, List[dict]]:
    """
    Data processing workflow for one SV using long read data.
    Process sequences for samples that have not been processed yet for a given SV.
    Downloads the bam file for the SV region from the cram file, extracts the cigar strings, and returns the deletions that match the SV size.
    Writes the evidence and removes the bam file.
    """

    region, sv_len = get_sv_region(sv_id, tolerance)
    long_reads = pd.read_csv("long_reads/long_read_samples.csv")

    # get samples that have been processed for this SV already
    deletions = get_processed_samples(sv_id)

    for sample_id in samples:
        # skip samples that have already been processed
        if sample_id in deletions:
            continue

        # process the sample
        row = long_reads[long_reads["sample_id"] == sample_id]
        if row.empty:
            print(f"Sample {sample_id} not found in long reads")
            continue
        cram_file = row["cram_file"].values[0]

        # get the bam file for this region from the cram
        output_file = get_bam_file(
            sv_id,
            sample_id,
            region=region,
            cram_file=cram_file,
            scratch=scratch,
        )

        # process the sample - get the deletions from the cigar string
        # possible ValueError
        deletions[sample_id] = read_cigars_from_file(output_file, sv_len)

        # write evidence to file and remove bam file
        write_sample_long_read_evidence(sv_id, sample_id, deletions[sample_id])
        remove_bam_file(output_file)

    return deletions


def get_all_long_reads():
    """Writes long read evidence for each SV."""

    # get SVs that were clustered into 1, 2 or 3 modes with low/medium/high confidence
    svs = pd.read_csv("1kgp/svs_n_modes.csv")
    svs = svs[svs["confidence"] != "inconclusive"]
    sv_ids = svs["sv_id"].unique()

    # get only svs that were clustered using short reads
    df = get_sv_stats_collapsed_df()
    df = df[df["id"].isin(sv_ids)]

    # get samples with long read data
    all_sample_ids = pd.read_csv("long_reads/long_read_samples.csv")[
        "sample_id"
    ].tolist()

    new_df = pd.DataFrame(
        columns=[
            "id",
            "chr",
            "num_modes_sr",
            "num_modes_lr",
            "num_samples_sr",
            "num_samples_lr",
            "sample_ids_sr",
            "sample_ids_lr",
        ]
    )
    for _, row in df.iterrows():
        sv_id = row["id"]
        deletions = get_long_read_svs(sv_id, all_sample_ids)
        sample_ids = list(deletions.keys())

        modes = ast.literal_eval(row["modes"])
        sample_ids_sr = []
        for mode in modes:
            sample_ids_sr.extend(mode["sample_ids"])

        new_df.loc[len(new_df)] = [
            sv_id,
            row["chr"],
            row["num_modes"],
            0,
            len(sample_ids_sr),
            len(sample_ids),
            sample_ids_sr,
            sample_ids,
        ]

    new_df.to_csv("long_reads/sr_lr_comparison.csv", index=False)


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
