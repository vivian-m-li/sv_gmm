import re
import os
import argparse
import subprocess
import pysam
import pandas as pd
from bs4 import BeautifulSoup
from helper import get_sv_lookup


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
    deletions = []
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        for read in bam.fetch():
            if read.is_unmapped:
                continue
            cigar_string = read.cigarstring
            if not cigar_string or "D" not in cigar_string:
                continue

            for match in re.finditer(r"(\d+)D", cigar_string):
                deletion_size = int(match.group(1))
                # account for 500 bp of tolerance
                # TODO: this won't work for small deletions
                if deletion_size >= sv_deletion_size - 500:

                    ref_pos = read.reference_start
                    cigar_tuples = read.cigartuples

                    for op, length in cigar_tuples:
                        if op == 2 and length == deletion_size:  # 2 = deletion
                            deletions.append(
                                {
                                    "start": ref_pos,
                                    "size": deletion_size,
                                }
                            )
                            break

                        if op in [0, 2, 3, 7, 8]:  # M, D, N, =, X
                            ref_pos += length

    return deletions


def compare_long_reads(sv_id: str, sample1: str, sample2: str, tolerance: int):
    # get sv region
    sv_lookup = get_sv_lookup()
    row = sv_lookup[sv_lookup["id"] == sv_id]
    start = row["start"].values[0] - tolerance
    stop = row["stop"].values[0] + tolerance
    region = f"chr{row['chr'].values[0]}:{start}-{stop}"
    sv_len = stop - start
    print(region, sv_len)

    # check both samples have long read files
    long_read_samples = pd.read_csv("long_reads/long_read_samples.csv")
    sample_rows = long_read_samples[
        long_read_samples["sample_id"].isin([sample1, sample2])
    ]
    if len(sample_rows) < 2:
        raise ValueError(
            "One or both samples does not have a corresponding long read file."
        )

    deletions = {}
    for sample_id in (sample1, sample2):
        output_file = f"long_reads/reads/{sv_id}-{sample_id}.bam"

        if not os.path.exists(output_file):
            cram_file = sample_rows[sample_rows["sample_id"] == sample_id][
                "cram_file"
            ].values[0]
            subprocess.run(
                ["bash", "get_cigar.sh"] + [cram_file, region, output_file],
                capture_output=True,
                text=True,
            )

        deletions[sample_id] = read_cigars_from_file(output_file, sv_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare between two CIGAR strings to identify differences in SV deletions"
    )
    parser.add_argument("-s1", type=str, help="First BAM/CRAM file")
    parser.add_argument("-s2", type=str, help="Second BAM/CRAM file")
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
    if args.id is None:
        raise ValueError("SV ID is required")
    if args.s1 is None or args.s2 is None:
        raise ValueError("Both samples are required")

    compare_long_reads(
        args.id,
        args.s1,
        args.s2,
        args.t,
    )
