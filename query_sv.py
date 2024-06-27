import os
import sys
import subprocess
import argparse
import pandas as pd
from viz import *

STIX_SCRIPT = "./query_stix.sh"
FILE_DIR = "stix_output"
PLOT_DIR = "plots"


def txt_to_df(filename: str):
    column_names = ["l_chr", "l_start", "l_end", "r_chr", "r_start", "r_end", "type"]
    df = pd.read_csv(filename, sep="\s+", names=column_names)
    return df


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

    return f"{chromosome}:{position}-{position}"


def query_stix(l: str, r: str):
    if not os.path.exists(FILE_DIR):
        os.mkdir(FILE_DIR)

    file_name=f"{l}_{r}"
    output_file = f"{FILE_DIR}/{file_name}.txt"
    
    if not os.path.isfile(output_file):
        subprocess.run(
            ["bash", STIX_SCRIPT] + [l, r, output_file], capture_output=True, text=True
        )
    df = txt_to_df(output_file)
    squiggle_data = df[["l_start", "r_end"]].to_numpy()

    if len(squiggle_data) == 0:
        print("No structural variants found in this region.")
        return
    
    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)

    L = int(l.split("-")[1])
    R = int(r.split("-")[1])
    bokeh_scatterplot(
        squiggle_data,
        file_name=f"{PLOT_DIR}/{file_name}",
        lower_bound=L - 1900,
        upper_bound=R + 1900,
        L=L,
        l=450,
        R=R,
    )
    # TODO: this function takes in a file
    mb = filter_and_plot_sequences_bokeh()
    intercepts = plot_fitted_lines_bokeh(mb, L=L, l=450, R=R)

    # os.remove(output_file)


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
