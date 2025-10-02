import os
import sys
import re
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

# Increase the field size limit to avoid triggering the error
csv.field_size_limit(sys.maxsize)

SCRATCH_DIR = "/scratch/Users/vili4418"
FILE_DIR = "stix_output"
PROCESSED_FILE_DIR = "processed_stix_output"
PLOT_DIR = "plots"


def txt_to_df(filename: str, long_reads: bool) -> pd.DataFrame:
    """Parses the raw stix output into a dataframe."""
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
    if long_reads: # format is slightly different
        df["sample_id"] = df["sample_id"].str.extract(r"^([^\.]+)")
    return df


def giggle_format(chromosome: str, position: int):
    return f"{chromosome.lower()}:{position}-{position}"


def reverse_giggle_format(l: str, r: str):  # noqa741
    chr = l.split(":")[0]
    start = int(l.split("-")[1])
    stop = int(r.split("-")[1])
    return chr, start, stop


def lookup_sv_position(sv_id: str):
    lookup = pd.read_csv("1kgp/sv_lookup.csv")
    row = lookup[lookup["id"] == sv_id]
    if row.empty:
        raise ValueError(f"SV ID {sv_id} not found in lookup table.")

    chr = str(row["chr"].values[0])
    start = int(row["start"].values[0])
    stop = int(row["stop"].values[0])
    return chr, start, stop


def load_squiggle_data(filename: str, rewrite_file: bool = False):
    squiggle_data = {}
    if not os.path.isfile(filename):
        return squiggle_data

    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 2:
                continue
            pattern = r"[\S]*([A-Z][A-Z]\d\d\d\d\d)"
            match = re.match(pattern, row[0])
            if match is None:
                raise ValueError("Invalid sample ID")
            sample_id = match.group(1)
            evidence = np.array([int(float(x)) for x in row[1:]])
            if len(evidence) > 0:
                squiggle_data[row[0]] = evidence

    # write squiggle data
    if rewrite_file:
        with open(filename, "w") as file:
            for sample_id, evidence in squiggle_data.items():
                evidence_str = ",".join(map(str, evidence))
                file.write(f"{sample_id},{evidence_str}\n")
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
    multi_files: bool, # if using grch38 reference, the STIX index and db are split into shards
    long_reads: bool,
    scratch: bool,
):
    bash_file = "query_stix.sh"
    if long_reads:
        bash_file = "query_stix_lr.sh"
    elif multi_files:
        bash_file = "query_stix_multifile.sh"

    if multi_files or long_reads:
        output_file = f"{output_dir}/partial_outputs/{file_name}"
    else:
        output_file = f"{output_dir}/{file_name}"

    subprocess.run(
        ["bash", bash_file] + [l, r, output_file, str(not scratch)],
        capture_output=True,
        text=True,
    )

    if multi_files or long_reads:
        # the query output from each shard was written to a different file, and we want to write them to the same file
        with open(f"{output_dir}/{file_name}.txt", "w") as out_file:
            n_shards = 23 if long_reads else 8
            for i in range(n_shards):
                with open(
                    f"{output_dir}/partial_outputs/{file_name}_{i}.txt", "r"
                ) as partial_file:
                    # write the partial file to the main file
                    out_file.write(partial_file.read())

                # remove partial file
                os.remove(f"{output_dir}/partial_outputs/{file_name}_{i}.txt")


def write_processed_output(
    output_file: str,
    processed_output_file: str,
    long_reads: bool,
    l_col: str = "l_start",
    r_col: str = "r_end",
) -> Dict[str, np.ndarray[float]]:
    """
    Parses the raw stix output (from the patched -g version) into pairs of coordinates for each sample. Each pair represents the start/stop of the deletion.
    For short reads, use l_start and r_end since the breakpoints are calculated from the clustering of paired-end reads.
    For long reads, use l_end and r_start to calculate the deletion. All long reads returned from stix are split reads, so the l_start and r_end coordinates capture the entire read length (not just the deletion).
    """
    df = txt_to_df(output_file, long_reads)

    grouped = df.groupby("file_id")
    squiggle_data = {}
    processed_stix_output = []
    for _, group in grouped:
        ls = group[l_col].tolist()
        rs = group[r_col].tolist()
        sample_id = group["sample_id"].iloc[0]
        sv_evidence = [item for pair in zip(ls, rs) for item in pair]

        squiggle_data[sample_id] = np.array(sv_evidence)
        sv_evidence = [sample_id] + sv_evidence
        processed_stix_output.append(sv_evidence)

    with open(processed_output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(processed_stix_output)

    return squiggle_data


def query_stix(
    *,
    l: str = "",
    r: str = "",
    sv_id: str = "",
    run_gmm: bool = True,
    filter_reference: bool = True,
    single_trial: bool = True,
    plot: bool = True,
    reference_genome: str = "grch38",  # or grch37
    long_reads: bool = False,
    scratch: bool = False,
):
    if sv_id == "" and (l == "" or r == ""):
        raise ValueError("Missing SV position or ID")

    if sv_id != "":
        chr, start, stop = lookup_sv_position(sv_id)
        l = giggle_format(chr, start)  # noqa741
        r = giggle_format(chr, stop)
    else:
        chr, start, stop = reverse_giggle_format(l, r)

    # Note: x/y chromosomes are ignored in the analysis and are not queried by the script
    if chr.lower() in ["x", "y"]:
        return {}

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
            query_stix_bash(
                l,
                r,
                output_file_dir,
                file_name,
                multi_files,
                long_reads,
                scratch,
            )
        squiggle_data = write_processed_output(
            output_file,
            processed_output_file,
            long_reads,
            l_col="l_end" if long_reads else "l_start",
            r_col="r_start" if long_reads else "r_end",
        )

        if scratch:
            # move files from scratch to home directory
            shutil.move(output_file, home_output_file)
            shutil.move(processed_output_file, home_processed_output_file)

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
                sv_id=sv_id,
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
        "-id",
        type=str,
        help="Structural variant ID from 1000 Genomes Project",
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
        "-s",
        type=bool,
        help="Use scratch directory for intermediate files",
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "-lr",
        type=bool,
        help="Cluster using long-read data instead of the default short-reads",
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "-ref", type=str, help="Reference genome", default="grch38"
    )

    args = parser.parse_args()

    l = parse_input(args.l) if args.l is not None else ""  # noqa741
    r = parse_input(args.r) if args.r is not None else ""
    sv_id = args.id or ""
    p = args.p
    d = args.d
    s = args.s
    lr = args.lr
    reference_genome = "grch38" if args.ref != "grch37" else args.ref
    query_stix(
        l=l,
        r=r,
        sv_id=sv_id,
        run_gmm=True,
        single_trial=not d,
        plot=p,
        reference_genome=reference_genome,
        long_reads=lr,
        scratch=s,
    )


if __name__ == "__main__":
    main()
