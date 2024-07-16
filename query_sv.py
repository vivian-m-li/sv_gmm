import os
import sys
import subprocess
import argparse
import csv
import pandas as pd
import numpy as np
from viz import run_viz_gmm, run_gmm_l_r


STIX_SCRIPT = "./query_stix.sh"
FILE_DIR = "stix_output"
PROCESSED_FILE_DIR = "processed_stix_output"
PLOT_DIR = "plots"


def txt_to_df(filename: str):
    column_names = [
        "file_id",
        "l_chr",
        "l_start",
        "l_end",
        "r_chr",
        "r_start",
        "r_end",
        "type",
    ]
    df = pd.read_csv(filename, names=column_names, sep="\s+")
    return df


def giggle_format(chromosome: int, position: int):
    return f"{chromosome}:{position}-{position}"


def load_squiggle_data(filename: str):
    squiggle_data = []
    if os.path.isfile(filename):
        with open(filename, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                squiggle_data.append(np.array([float(x) for x in row]))
    return squiggle_data


def parse_input(input: str) -> str:
    parts = input.split(":")
    if len(parts) != 2:
        print("Input string must be in the format 'chromosome:position'")
        sys.exit(1)
    try:
        chromosome = int(parts[0])
        position = int(parts[1])
    except ValueError:
        print("Input string must be in the format 'chromosome:position'")
        sys.exit(1)
    return giggle_format(chromosome, position)


def query_stix(l: str, r: str, run_gmm: bool = True):
    for directory in [FILE_DIR, PROCESSED_FILE_DIR, PLOT_DIR]:
        if not os.path.exists(directory):
            os.mkdir(directory)

    file_name = f"{l}_{r}"
    output_file = f"{FILE_DIR}/{file_name}.txt"
    processed_output_file = f"{PROCESSED_FILE_DIR}/{file_name}.csv"

    if os.path.isfile(processed_output_file):
        squiggle_data = load_squiggle_data(processed_output_file)
    else:
        if not os.path.isfile(output_file):
            subprocess.run(
                ["bash", STIX_SCRIPT] + [l, r, output_file],
                capture_output=True,
                text=True,
            )
        df = txt_to_df(output_file)

        grouped = df.groupby("file_id")
        squiggle_data = []
        for _, group in grouped:
            l_starts = group["l_start"].tolist()
            r_ends = group["r_end"].tolist()
            sv_evidence = [item for pair in zip(l_starts, r_ends) for item in pair]
            squiggle_data.append(np.array(sv_evidence))

        with open(processed_output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(squiggle_data)

    if run_gmm:
        if len(squiggle_data) == 0:
            print("No structural variants found in this region.")
            return

        run_viz_gmm(
            squiggle_data,
            file_name=f"{PLOT_DIR}/{file_name}",
            L=int(l.split("-")[1]),
            R=int(r.split("-")[1]),
        )


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


if __name__ == "__main__":
    main()
