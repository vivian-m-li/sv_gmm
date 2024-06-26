import os
import sys
import subprocess
import argparse
import pandas as pd

STIX_SCRIPT = "./query_stix.sh"
OUTPUT_FILE = "stix_output.txt"


def txt_to_df(filename: str):
    column_names = ["l_chr", "l_start", "l_end", "r_chr", "r_start", "r_end", "type"]
    df = pd.read_csv(filename, sep="\s+", names=column_names)
    return df


def parse_input(input: str) -> str:
    parts = input.split(":")
    if len(parts) != 2:
        print("Input string must be in the format 'chromosome:position'")
        sys.exit(1)

    chromosome = int(parts[0])
    position = int(parts[1])
    return f"{chromosome}:{position}-{position}"


def query_stix(l: str, r: str):
    result = subprocess.run(
        ["bash", STIX_SCRIPT] + [l, r, OUTPUT_FILE], capture_output=True, text=True
    )
    df = txt_to_df(OUTPUT_FILE)
    squiggle_data = df[["l_start", "r_end"]]
    # TODO: now that we've processed the df, pass into squiggle code


def main():
    parser = argparse.ArgumentParser(
        description="Queries structural variants in a specific region"
    )
    parser.add_argument(
        "-l",
        type=str,
        help="Left position of the structural variant, format=chromosome:position",
    )
    parser.add_argument(
        "-r",
        type=str,
        help="Right position of the structural variant, format=chromosome:position",
    )

    args = parser.parse_args()
    l = parse_input(args.l)
    r = parse_input(args.r)
    query_stix(l, r)

    os.remove(OUTPUT_FILE)


if __name__ == "__main__":
    main()
