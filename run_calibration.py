import os
import argparse
import multiprocessing
import pandas as pd
import numpy as np
from timeout import break_after
from run_dirichlet import run_dirichlet
from write_sv_output import (
    get_raw_data,
    init_sv_stat_row,
    write_sv_stats,
    write_posterior_distributions,
    concat_multi_processed_sv_files,
    write_post_processed_files,
)
from typing import Set, Dict

FILE_DIR = "processed_svs_converge"
SCRATCH_FILE_DIR = os.path.join("/scratch/Users/vili4418", FILE_DIR)
OUTPUT_FILE_NAME = "sv_stats_converge.csv"


def run_dirichlet_inner(
    row: Dict,
    population_size: int,
    input_dir: str,
):
    sv_id = row["id"]
    reads, num_samples = get_raw_data(
        row, input_dir, filter_reference_samples=False
    )

    if reads.empty:
        # if no samples have any data supporting the SV then not included
        gmms, alphas, posterior_distributions = [(None, [])], [], []
    else:
        gmms, alphas, posterior_distributions = run_dirichlet(
            reads,
            **{
                "chr": row["chr"],
                "L": row["start"],
                "R": row["stop"],
                "plot": False,
                "stem": input_dir,
            },
        )

    for i, (gmm, evidence_by_mode) in enumerate(gmms):
        sv_stat = init_sv_stat_row(
            row,
            num_samples=num_samples,
            num_reference=num_samples - reads["sample_id"].nunique(),
        )
        write_sv_stats(
            sv_stat, gmm, evidence_by_mode, population_size, SCRATCH_FILE_DIR, i
        )

    if gmms[0][0] is not None:
        write_posterior_distributions(
            sv_id, alphas, posterior_distributions, SCRATCH_FILE_DIR
        )


def run_calibration_test(
    sv_df: pd.DataFrame,
    sv_subset: pd.DataFrame,
    *,
    d: int,
    r: float,
    input_dir: str,
    output_dir: str,
    sample_ids: Set[str],
):
    population_size = len(sample_ids)
    # TODO: parallelize this function
    for _, row in sv_subset.iterrows():
        # TODO: pass in d and r args to run_dirichlet_inner
        run_dirichlet_inner(row, population_size, input_dir)

    # TODO: update these filepaths to point to scratch
    concat_multi_processed_sv_files(FILE_DIR, OUTPUT_FILE_NAME, output_dir)
    write_post_processed_files(input_dir, output_dir)
    # TODO: move final output files to a new directory (id'ed by d and r) in output_dir
    # TODO: remove intermediate directory - processed_svs_converge

    # append to a master file with columns d, r, TP, FP, FN, TN
    # do analysis later on which combinations led to "partial correctness"


def download_stix_data(
    sv_subset: pd.DataFrame,
    input_dir: str,
):
    """Download stix data for all regions in the subset before running calibration tests."""
    # TODO: parallelize
    pass


def run_calibration(
    *,
    input_dir: str,
    output_dir: str,
    sv_regions_file: str,
    sv_lookup_file: str,
    sample_ids_file: str,
    d_min: int,
    d_max: int,
    d_step: int,
    r_min: float,
    r_max: float,
    r_step: float,
):
    sv_subset = pd.read_csv(os.path.jion(input_dir, sv_regions_file))
    sv_df = pd.read_csv(os.path.join(input_dir, sv_lookup_file))
    sample_ids = set()
    with open(sample_ids_file, "r") as f:
        for line in f:
            sample_ids.add(line.strip())

    # download stix data for all regions in the subset before running calibration tests
    download_stix_data(sv_subset, input_dir)

    for d in range(d_min, d_max + 1, d_step):
        for r in np.arange(r_min, r_max + r_step, r_step):
            print(f"Running calibration for d={d}, r={r}")
            run_calibration_test(
                sv_df,
                sv_subset,
                d=d,
                r=round(r, 2),
                input_dir=input_dir,
                output_dir=output_dir,
                sample_ids=sample_ids,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Run calibration test on a subset of structural variants"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory for incoming data",
        default="calibration",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for results",
        default="calibration/results",
    )
    parser.add_argument(
        "--regions",
        type=str,
        help="CSV file defining regions to consider",
        default="sv_subset.csv",
    )
    parser.add_argument(
        "--sv_lookup",
        type=str,
        help="CSV file with structural variants",
        default="filtered_dels.csv",
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        help="Txt file containing sample IDs with long read data available",
        default=None,
    )
    parser.add_argument(
        "--d_min",
        type=int,
        help="Minimum distance threshold to consider",
        default=50,
    )
    parser.add_argument(
        "--d_max",
        type=int,
        help="Maximum distance threshold to consider",
        default=550,
    )
    parser.add_argument(
        "--d_step",
        type=int,
        help="Step size for distance values",
        default=50,
    )
    parser.add_argument(
        "--r_min",
        type=float,
        help="Minimum reciprocal overlap threshold to consider",
        default=0.4,
    )
    parser.add_argument(
        "--r_max",
        type=float,
        help="Maximum reciprocal overlap threshold to consider",
        default=0.9,
    )
    parser.add_argument(
        "--r_step",
        type=float,
        help="Step size for reciprocal overlap values",
        default=0.05,
    )

    args = parser.parse_args()
    run_calibration(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sv_regions_file=args.regions,
        sv_lookup_file=args.sv_lookup,
        sample_ids_file=args.sample_ids,
        d_min=args.d_min,
        d_max=args.d_max,
        d_step=args.d_step,
        r_min=args.r_min,
        r_max=args.r_max,
        r_step=args.r_step,
    )


if __name__ == "__main__":
    main()
