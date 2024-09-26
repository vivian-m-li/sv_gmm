import os
import sys
import subprocess
import argparse
import csv
import pandas as pd
import numpy as np
from viz import run_viz_gmm
from typing import List, Dict


STIX_SCRIPT = "./query_stix.sh"
FILE_DIR = "stix_output"
PROCESSED_FILE_DIR = "processed_stix_output"
PLOT_DIR = "plots"


def txt_to_df(filename: str):
    column_names = [
        "file_id",
        "sample_id",
        "l_chr",
        "l_start",
        "l_end",
        "r_chr",
        "r_start",
        "r_end",
        "type",
    ]
    df = pd.read_csv(filename, names=column_names, sep=r"\s+")
    df["sample_id"] = df["sample_id"].str.extract(r"/([^/]+)\.bed\.gz", expand=False)
    return df


def giggle_format(chromosome: str, position: int):
    return f"{chromosome.lower()}:{position}-{position}"


def reverse_giggle_format(l: str, r: str):
    chr = l.split(":")[0]
    start = int(l.split("-")[1])
    stop = int(r.split("-")[1])
    return chr, start, stop


def load_squiggle_data(filename: str):
    squiggle_data = {}
    if os.path.isfile(filename):
        with open(filename, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                squiggle_data[row[0]] = np.array([float(x) for x in row[1:]])
    return squiggle_data


def parse_input(input: str) -> str:
    parts = input.split(":")
    if len(parts) != 2:
        print("Input string must be in the format 'chromosome:position'")
        sys.exit(1)
    try:
        chromosome = parts[0]
        position = int(parts[1])
    except ValueError:
        print("Input string must be in the format 'chromosome:position'")
        sys.exit(1)
    return giggle_format(chromosome, position)


def get_reference_samples(
    df: pd.DataFrame,
    squiggle_data: Dict[str, np.ndarray[float]],
    chr: str,
    start: int,
    stop: int,
) -> List[str]:
    row = df[(df["chr"] == chr) & (df["start"] == start) & (df["stop"] == stop)]
    samples = [
        sample_id for sample_id in df.columns[11:-1] if sample_id in squiggle_data
    ]
    ref_samples = [col for col in samples if row.iloc[0][col] == "(0, 0)"]
    return ref_samples


def query_stix(l: str, r: str, run_gmm: bool = True, *, filter_reference: bool = True):
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
            # TODO: the stix query isn't getting anything for x/y chrs
            subprocess.run(
                ["bash", STIX_SCRIPT] + [l, r, output_file],
                capture_output=True,
                text=True,
            )
        df = txt_to_df(output_file)

        grouped = df.groupby("file_id")
        squiggle_data = {}
        processed_stix_output = []
        for _, group in grouped:
            l_starts = group["l_start"].tolist()
            r_ends = group["r_end"].tolist()
            sample_id = group["sample_id"].iloc[0]
            sv_evidence = [item for pair in zip(l_starts, r_ends) for item in pair]

            squiggle_data[sample_id] = np.array(sv_evidence)
            sv_evidence = [sample_id] + sv_evidence
            processed_stix_output.append(sv_evidence)

        with open(processed_output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(processed_stix_output)

    chr, start, stop = reverse_giggle_format(l, r)
    deletions_df = pd.read_csv("1000genomes/deletions_df.csv", low_memory=False)

    # remove samples queried by stix but missing in the 1000genomes columns
    sample_ids = set(deletions_df.columns[11:-1])
    missing_keys = set(squiggle_data.keys()) - sample_ids
    for key in missing_keys:
        squiggle_data.pop(key, None)

    if filter_reference:
        ref_samples = get_reference_samples(
            deletions_df, squiggle_data, chr, start, stop
        )
        for ref in ref_samples:
            squiggle_data.pop(ref, None)

    if run_gmm:
        if len(squiggle_data) == 0:
            # print("No structural variants found in this region.")
            return

        run_viz_gmm(
            squiggle_data,
            file_name=f"{PLOT_DIR}/{file_name}",
            chr=chr,
            L=start,
            R=stop,
            plot=True,
            plot_bokeh=False,
        )

    return squiggle_data


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
