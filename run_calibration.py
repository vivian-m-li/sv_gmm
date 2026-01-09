import os
import argparse
import multiprocessing
import shutil
import pandas as pd
import numpy as np
from run_dirichlet import run_dirichlet
from query_sv import giggle_format, query_stix_bash
from write_sv_output import (
    get_raw_data,
    init_sv_stat_row,
    write_sv_stats,
    write_posterior_distributions,
    concat_multi_processed_sv_files,
    write_post_processed_files,
)
from typing import Set, Dict

SLURM_CPUS = int(os.environ['SLURM_CPUS_ON_NODE'])
FILE_DIR = "calibration_outputs"
SCRATCH_FILE_DIR = os.path.join("/scratch/Users/vili4418", FILE_DIR)
OUTPUT_FILE_NAME = "sv_stats_converge.csv"


def calc_confusion_matrix(
    sv_subset: pd.DataFrame,
    svs_n_modes: pd.DataFrame,
) -> Dict[str, float]:
    """Calculate confusion matrix based on predicted vs actual number of SVs."""
    # remove rows that didn't run due to lack of data
    svs_n_modes = svs_n_modes[svs_n_modes["confidence"] != "inconclusive"]
    svs_n_modes.rename(
        columns={"sv_id": "id", "num_modes": "n_svs_predicted"}, inplace=True
    )
    merged = sv_subset.merge(svs_n_modes, on="id", how="right")

    TP = merged[
        (merged["n_svs_actual"] == 2) & (merged["n_svs_predicted"] == 2)
    ].shape[0]
    FP = (
        merged[
            (merged["n_svs_actual"] == 1) & (merged["n_svs_predicted"] == 2)
        ].shape[0]
        + merged[merged["n_svs_predicted"] == 3].shape[
            0
        ]  # this covers cases where we predicted 3 modes when actual is 1 or 2
    )
    FN = merged[
        (merged["n_svs_actual"] == 2) & (merged["n_svs_predicted"] == 1)
    ].shape[0]
    TN = merged[
        (merged["n_svs_actual"] == 1) & (merged["n_svs_predicted"] == 1)
    ].shape[0]

    n_svs = merged.shape[0]
    values = {"TP": TP, "FP": FP, "FN": FN, "TN": TN}
    values = {k: v / n_svs for k, v in values.items()}
    return values


def run_dirichlet_inner(
    row: Dict,
    population_size: int,
    input_dir: str,
    output_dir: str,
    d: int,
    r: float,
):
    # filter reference samples is set to false because we don't have references for regions, only SVs
    reads, num_samples = get_raw_data(
        row, input_dir, filter_reference_samples=False, print_messages=False,
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
                "d_threshold": d,
                "r_threshold": r,
                "synthetic_data": True,
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
            sv_stat, gmm, evidence_by_mode, population_size, output_dir, i
        )

    if gmms[0][0] is not None:
        write_posterior_distributions(
            row["id"], alphas, posterior_distributions, output_dir
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
    """
    Run a single calibration test for given distance and reciprocal overlap thresholds.
    TODO: do analysis later on which combinations led to "partial correctness"
    """
    population_size = len(sample_ids)
    processed_file_dir = os.path.join(
        SCRATCH_FILE_DIR, "d{}_r{:.2f}".format(d, r)
    )
    if not os.path.exists(processed_file_dir):
        os.makedirs(processed_file_dir)

    with multiprocessing.Manager():
        p = multiprocessing.Pool(SLURM_CPUS)
        args = []
        for _, row in sv_subset.iterrows():
            args.append(
                (row, population_size, input_dir, processed_file_dir, d, r)
            )

        p.starmap(run_dirichlet_inner, args)
        p.close()
        p.join()

    results_dir = os.path.join(output_dir, "d{}_r{:.2f}".format(d, r))
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    concat_multi_processed_sv_files(
        processed_file_dir, OUTPUT_FILE_NAME, results_dir
    )
    write_post_processed_files(input_dir, results_dir, sv_subset, True)
    shutil.rmtree(processed_file_dir)

    svs_n_modes = pd.read_csv(os.path.join(results_dir, "svs_n_modes.csv"))
    results = calc_confusion_matrix(sv_subset, svs_n_modes)

    final_output_file = os.path.join(output_dir, "results.csv")
    if not os.path.exists(final_output_file):
        with open(final_output_file, "w") as f:
            f.write("d,r,TP,FP,FN,TN\n")
    with open(final_output_file, "a") as f:
        f.write(
            "{},{},{},{},{},{}\n".format(
                d,
                r,
                results["TP"],
                results["FP"],
                results["FN"],
                results["TN"],
            )
        )


def download_stix_data_inner(
    row: pd.Series,
    output_dir: str,
):
    # check if the data for this region already exists
    l = giggle_format(str(row.chr), row.start)  # noqa:741
    r = giggle_format(str(row.chr), row.stop)
    filename = f"{l}_{r}"
    file = os.path.join(output_dir, f"{filename}.txt")
    if os.path.isfile(file):
        return

    stix_file = query_stix_bash(
        l,
        r,
        output_dir,
        filename,
        # hard-coded stix parameters for 1kg high coverage data on Vivian's fiji
        "/Users/vili4418/sv/stix/bin/stix",
        "/scratch/Shares/layer/stix/indices/1kg_high_coverage_vivian/shard",
        "/scratch/Shares/layer/stix/indices/1kg_high_coverage_vivian/shard",
        8,
    )
    print(f"Downloaded STIX data {stix_file}")


def download_stix_data(
    sv_subset: pd.DataFrame,
    input_dir: str,
):
    """Download stix data for all regions in the subset before running calibration tests."""
    output_dir = os.path.join(input_dir, "stix_output")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with multiprocessing.Manager():
        print(multiprocessing.cpu_count(), SLURM_CPUS)
        p = multiprocessing.Pool(SLURM_CPUS)
        args = []

        for _, row in sv_subset.iterrows():
            args.append((row, output_dir))

        p.starmap(download_stix_data_inner, args)
        p.close()
        p.join()


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
    """Run calibration tests over a grid of distance and reciprocal overlap thresholds."""
    sv_subset = pd.read_csv(os.path.join(input_dir, sv_regions_file))
    sv_subset["id"] = sv_subset.apply(
        lambda row: f"{giggle_format(str(row['chr']), row['start'])}_{giggle_format(str(row['chr']), row['stop'])}",
        axis=1,
    )
    sv_df = pd.read_csv(os.path.join(input_dir, sv_lookup_file))
    sample_ids = set()
    with open(os.path.join(input_dir, sample_ids_file), "r") as f:
        for line in f:
            sample_ids.add(line.strip())

    # check that all stix data has been downloaded for regions in sv_subset before running calibration tests
    download_stix_data(sv_subset, input_dir)

    # run calibration tests over grid of d and r values
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
        default="deletions.csv",
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        help="Txt file containing sample IDs with long read data available",
        default="sample_ids.txt",
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
