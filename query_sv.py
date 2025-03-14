import os
import sys
import shutil
import subprocess
import argparse
import csv
import pandas as pd
import numpy as np
from process_data import run_viz_gmm
from run_dirichlet import run_dirichlet
from helper import get_sample_ids
from typing import List, Dict

SCRATCH_DIR = "/scratch/Users/vili4418"
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
    df["sample_id"] = df["sample_id"].str.extract(
        r"/([^/]+)\.bed\.gz", expand=False
    )
    return df


def giggle_format(chromosome: str, position: int):
    return f"{chromosome.lower()}:{position}-{position}"


def reverse_giggle_format(l: str, r: str):  # noqa741
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
    squiggle_data: Dict[str, np.ndarray[float]],
    chr: str,
    start: int,
    stop: int,
    file_root: str = "1kgp",
) -> List[str]:
    df = pd.read_csv(f"{file_root}/deletions_by_chr/chr{chr}.csv")
    row = df[(df["start"] == start) & (df["stop"] == stop)]
    samples = squiggle_data.keys()
    ref_samples = [col for col in samples if row.iloc[0][col] == "(0, 0)"]
    return ref_samples


def query_stix_bash(
    l: int,
    r: int,
    output_dir: str,
    file_name: str,
    multi_files: bool,
    scratch: bool,
):
    bash_file = "query_stix_multifile.sh" if multi_files else "query_stix.sh"
    if multi_files:
        output_file = f"{output_dir}/partial_outputs/{file_name}"
    else:
        output_file = f"{output_dir}/{file_name}"

    subprocess.run(
        ["bash", bash_file] + [l, r, output_file, str(not scratch)],
        capture_output=True,
        text=True,
    )

    if multi_files:
        with open(f"{output_dir}/{file_name}.txt", "w") as out_file:
            for i in range(8):
                with open(
                    f"{output_dir}/partial_outputs/{file_name}_{i}.txt", "r"
                ) as partial_file:
                    out_file.write(
                        partial_file.read()
                    )  # write the partial file to the main file

                os.remove(
                    f"{output_dir}/partial_outputs/{file_name}_{i}.txt"
                )  # remove partial file


def write_processed_output(output_file: str, processed_output_file: str):
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

    return squiggle_data


def query_stix(
    l: str,
    r: str,
    run_gmm: bool = True,
    *,
    filter_reference: bool = True,
    single_trial: bool = True,
    plot: bool = True,
    reference_genome: str = "grch38",  # or grch37
    scratch: bool = False,
):
    # read/write files in scratch if flagged
    if scratch:
        output_file_dir = f"{SCRATCH_DIR}/{FILE_DIR}"
        processed_file_dir = f"{SCRATCH_DIR}/{PROCESSED_FILE_DIR}"
    else:
        output_file_dir = FILE_DIR
        processed_file_dir = PROCESSED_FILE_DIR
    plot_dir = PLOT_DIR

    for directory in [output_file_dir, processed_file_dir, plot_dir]:
        if not os.path.exists(directory):
            os.mkdir(directory)

    file_root = "1kgp"
    if reference_genome == "grch37":
        file_root = "low_cov_grch37"

    # set up the correct file paths in case scratch is True
    file_name = f"{l}_{r}"
    output_file = f"{output_file_dir}/{file_name}.txt"
    processed_output_file = f"{processed_file_dir}/{file_name}.csv"
    home_output_file = f"{FILE_DIR}/{file_name}.txt"
    home_processed_output_file = f"{PROCESSED_FILE_DIR}/{file_name}.csv"

    if reference_genome == "grch37" and not scratch:
        output_file = f"{file_root}/{output_file}"
        processed_output_file = f"{file_root}/{processed_output_file}"

    # check if this sv has already been queried for and processed in the home directory
    if os.path.isfile(home_processed_output_file):
        squiggle_data = load_squiggle_data(home_processed_output_file)
    else:
        # check if this sv has already been queried for in the home directory
        if not os.path.isfile(home_output_file):
            multi_files = reference_genome == "grch38"
            # Note: x/y chromosomes are ignored in the analysis and are not queried by the script
            query_stix_bash(
                l, r, output_file_dir, file_name, multi_files, scratch
            )
        squiggle_data = write_processed_output(
            output_file, processed_output_file
        )

        if scratch:
            # move files from scratch to home directory
            shutil.move(output_file, home_output_file)
            shutil.move(processed_output_file, home_processed_output_file)

    chr, start, stop = reverse_giggle_format(l, r)

    # remove samples queried by stix but missing in the 1000genomes columns
    sample_ids = get_sample_ids(file_root)
    missing_keys = set(squiggle_data.keys()) - sample_ids
    for key in missing_keys:
        squiggle_data.pop(key, None)

    if filter_reference:
        ref_samples = get_reference_samples(
            squiggle_data, chr, start, stop, file_root
        )
        for ref in ref_samples:
            squiggle_data.pop(ref, None)

    if run_gmm:
        if len(squiggle_data) == 0:
            # print("No structural variants found in this region.")
            return

        if single_trial:
            run_viz_gmm(
                squiggle_data,
                file_name=f"{PLOT_DIR}/{file_name}",
                chr=chr,
                L=start,
                R=stop,
                plot=plot,
                plot_bokeh=False,
            )
        else:
            run_dirichlet(
                squiggle_data,
                **{
                    "file_name": f"{PLOT_DIR}/{file_name}",
                    "chr": chr,
                    "L": start,
                    "R": stop,
                    "plot": plot,
                    "plot_bokeh": False,
                },
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
    parser.add_argument(
        "-p",
        type=bool,
        help="Plot the length and L coordinate of each sample",
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "-d",
        type=bool,
        help="Rerun the SV until >= 80% confident in the outcome",
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "-ref", type=str, help="Reference genome", default="grch38"
    )

    args = parser.parse_args()
    l = parse_input(args.l)  # noqa741
    r = parse_input(args.r)
    p = args.p
    d = args.d
    reference_genome = "grch38" if args.ref != "grch37" else args.ref
    query_stix(
        l,
        r,
        True,
        single_trial=not d,
        plot=p,
        reference_genome=reference_genome,
    )


if __name__ == "__main__":
    main()
