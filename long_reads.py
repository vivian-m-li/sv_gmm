import re
import os
import argparse
import subprocess
import pysam
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
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


def read_cigars_from_file(bam_file):
    deletions = []
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        for read in bam.fetch():
            import pdb

            pdb.set_trace()
            if read.is_unmapped:
                continue
            cigar = read.cigarstring

    return deletions


def compare_long_reads(sv_id: str, sample1: str, sample2: str, tolerance: int):
    # get sv region
    sv_lookup = get_sv_lookup()
    row = sv_lookup[sv_lookup["id"] == sv_id]
    start = row["start"].values[0] - tolerance
    stop = row["stop"].values[0] + tolerance
    region = f"{row['chr'].values[0]}:{start}-{stop}"

    # check both samples have long read files
    long_read_samples = pd.read_csv("long_reads/long_read_samples.csv")
    sample_rows = long_read_samples[
        long_read_samples["sample_id"].isin([sample1, sample2])
    ]
    if len(sample_rows) < 2:
        raise ValueError(
            "One or both samples does not have a corresponding long read file."
        )

    cigar_strings = {}
    for sample_id in (sample1, sample2):
        output_file = f"long_reads/reads/{sv_id}-{sample_id}.sam"

        if not os.path.exists(output_file):
            cram_file = sample_rows[sample_rows["sample_id"] == sample_id][
                "cram_file"
            ].values[0]
            subprocess.run(
                ["bash", "get_cigar.sh"] + [cram_file, region, output_file],
                capture_output=True,
                text=True,
            )

        cigar_strings[sample_id] = read_cigars_from_file(output_file)

    # compare the cigar strings

    # summary = summarize_cigar(cigar)

    # print(f"CIGAR Summary:")
    # print(f"  Total operations: {summary['total_operations']}")
    # print(f"  Operation counts:")
    # for op, count in summary["operation_counts"].items():
    #     print(f"    {op}: {count}")

    # print(f"  Large indels (>= {args.min_sv_size} bp):")
    # for sv_type, pos, length in summary["large_indels"]:
    #     print(f"    {sv_type} at relative position {pos} with length {length}")

    # return

    # # Analyze from SAM/BAM files (simplified, would typically use pysam)
    # # This is just a demonstration of the concept
    # if args.file1 and args.file2 and args.region:
    #     print(
    #         f"Comparing CIGAR strings for region {args.region} between samples..."
    #     )
    #     print(
    #         "This would typically use pysam to extract records from BAM/CRAM files"
    #     )
    #     # Implementation would depend on having pysam to read BAM/CRAM files


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
    compare_long_reads(
        args.id,
        args.s1,
        args.s2,
        args.t,
    )
