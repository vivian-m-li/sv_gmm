import os
import re
import argparse
import multiprocessing
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.transition_criterion import MinTrials
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.adapter.registry import Generators
from run_dirichlet import run_dirichlet
from query_sv import (
    giggle_format,
    query_stix_bash,
    get_query_region,
)
from helper import calc_pr_auc
from write_sv_output import (
    get_raw_data,
    init_sv_stat_row,
    write_sv_stats,
    write_posterior_distributions,
    concat_multi_processed_sv_files,
    write_post_processed_files,
)
from collections import defaultdict, Counter
from typing import Set, Dict

SLURM_CPUS = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
FILE_DIR = "calibration_outputs"
SCRATCH_FILE_DIR = os.path.join("/scratch/Users/vili4418", FILE_DIR)
OUTPUT_FILE_NAME = "sv_stats_converge.csv"


def all_consensus_svs(*, plot: bool = False):
    """Gets the distribution of mode predictions for each SV across all calibration runs."""
    if not os.path.exists("calibration/results/sv_results.csv"):
        sv_mode_counts = defaultdict(Counter)
        for run_dir in os.listdir("calibration/results"):
            if not os.path.isdir(os.path.join("calibration/results", run_dir)):
                continue
            svs_n_modes = pd.read_csv(
                os.path.join("calibration/results", run_dir, "svs_n_modes.csv")
            )
            svs_n_modes = svs_n_modes[
                svs_n_modes["confidence"] != "inconclusive"
            ]
            for i, row in svs_n_modes.iterrows():
                sv_mode_counts[row["sv_id"]][row["num_modes"]] += 1

        df = pd.DataFrame(
            columns=[
                "sv_id",
                "n_modes_1",
                "n_modes_2",
                "n_modes_3",
                "majority_outcome",
                "majority_percent",
                "n_models_run",
            ]
        )
        for sv_id, counts in sv_mode_counts.items():
            n_modes_1 = counts.get(1, 0)
            n_modes_2 = counts.get(2, 0)
            n_modes_3 = counts.get(3, 0)
            total = n_modes_1 + n_modes_2 + n_modes_3
            majority_outcome = np.argmax([n_modes_1, n_modes_2, n_modes_3]) + 1
            majority_count = max(n_modes_1, n_modes_2, n_modes_3)
            df.loc[len(df)] = [
                sv_id,
                n_modes_1,
                n_modes_2,
                n_modes_3,
                int(majority_outcome),
                majority_count / total,
                total,
            ]
        df.to_csv("calibration/results/sv_results.csv", index=False)
    else:
        df = pd.read_csv("calibration/results/sv_results.csv")

    if plot:
        # plot distribution of mode predictions
        plt.figure(figsize=(8, 6))
        plt.hist(df["majority_percent"].values, bins=20, range=(0, 1))
        plt.xlabel("% of Runs Agreeing on Majority Mode")
        plt.ylabel("Number of SVs")
        plt.show()


def assign_model_score(*, plot: bool = False):
    # assign scores based on how often models agreed with the consensus
    sv_results = pd.read_csv("calibration/results/sv_results.csv")
    df = pd.read_csv("calibration/results/results.csv")
    for i, row in df.iterrows():
        dir = f"d{int(row['d'])}_r{row['r']:.2f}_q{row['q']:.2f}"
        svs_n_modes = pd.read_csv(
            os.path.join("calibration/results", dir, "svs_n_modes.csv")
        )
        merged = svs_n_modes.merge(sv_results, on="sv_id", how="left")
        merged["correct"] = merged.apply(
            lambda row: int(row["num_modes"] == row["majority_outcome"]), axis=1
        )
        n_correct = sum(merged["correct"].values)
        n_run = svs_n_modes[svs_n_modes["confidence"] != "inconclusive"].shape[
            0
        ]
        df.loc[i, "n_run"] = n_run
        df.loc[i, "n_correct"] = n_correct
        df.loc[i, "model_score"] = n_correct / n_run
    df.to_csv("calibration/results/results.csv", index=False)

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        for i, (param, xlabel) in enumerate(
            zip(
                ["d", "r", "q"],
                [
                    "Distance at which penalty = 0",
                    "Reciprocal overlap at which penalty = 0",
                    "Required evidence overlap with original SV region",
                ],
            )
        ):
            df.boxplot(column="model_score", by=param, ax=axs[i])
            axs[i].set_xlabel(xlabel)
            axs[i].set_ylabel("Model Score")
            axs[i].set_title("")
            axs[i].set_ylim(0, 1)
            axs[i].grid(False)
        plt.suptitle("")
        plt.show()


def find_pareto_front():
    # construct tp/fp curve from results.csv files in calibration output directories
    results = pd.read_csv("calibration/results/results.csv")
    plt.figure(figsize=(8, 6))

    results = results.sort_values(by=["FP", "TP"], ascending=[True, False])
    plt.scatter(results["FP"], results["TP"])
    for i, row in results.iterrows():
        plt.text(
            row["FP"],
            row["TP"] - 0.01,
            f"d={int(row['d'])},r={row['r']:.2f},q={row['q']:.2f},p={row['p']:.2f}",
            fontsize=8,
            ha="center",
        )

    pareto_front = pd.DataFrame(columns=["FP", "TP", "d", "r", "q", "p"])
    max_tp = -1
    for i, row in results.iterrows():
        if row["TP"] > max_tp:
            pareto_front.loc[len(pareto_front)] = row
            max_tp = row["TP"]
    plt.plot(
        pareto_front["FP"],
        pareto_front["TP"],
        color="red",
        linewidth=2,
    )

    # find the pareto optimal point closest to (0, 1)
    pareto_front["dist"] = np.sqrt(
        pareto_front["FP"] ** 2 + (1 - pareto_front["TP"]) ** 2
    )
    best_point = pareto_front.loc[pareto_front["dist"].idxmin()]
    print(
        f"Best point on Pareto front: d={int(best_point['d'])}, r={best_point['r']:.2f}, q={best_point['q']:.2f}, p={int(best_point['p'])}, FPR={best_point['FP']:.4f}, TPR={best_point['TP']:.4f}"
    )

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()


def get_confusion_matrix(
    sv_subset: pd.DataFrame,
    svs_n_modes: pd.DataFrame,
    q: float,
) -> Dict[str, float]:
    """
    Calculate confusion matrix based on predicted vs actual number of SVs.
    The actual number of SVs are determined based on the long read data
    and are defined in the sv_subset.csv file.
    """
    # remove rows that didn't run due to lack of data
    svs_n_modes = svs_n_modes.rename(
        columns={"sv_id": "id", "num_modes": "n_svs_predicted"},
    )
    svs_n_modes = svs_n_modes[svs_n_modes["confidence"] != "inconclusive"]
    if "q" in sv_subset.columns:
        sv_subset_q = sv_subset[sv_subset["q"] == q]
    else:
        sv_subset_q = sv_subset
    merged = sv_subset_q.merge(svs_n_modes, on="id", how="right")

    TP = merged[
        (merged["n_svs_actual"] == 2) & (merged["n_svs_predicted"] >= 2)
    ].shape[0]
    FP = merged[
        (merged["n_svs_actual"] == 1) & (merged["n_svs_predicted"] >= 2)
    ].shape[
        0
    ]  # this covers cases where we predicted 3 modes when actual is 1 or 2
    FN = merged[
        (merged["n_svs_actual"] == 2) & (merged["n_svs_predicted"] == 1)
    ].shape[0]
    TN = merged[
        (merged["n_svs_actual"] == 1) & (merged["n_svs_predicted"] == 1)
    ].shape[0]

    values = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
    }  # keep raw values instead of proportions
    # n_svs = merged.shape[0]
    # values = {k: v / n_svs for k, v in values.items()}
    return values


def rewrite_calibration_results():
    """
    Rewrites the results.csv file with the confusion matrix from each run.
    Standalone function
    """
    sv_subset = pd.read_csv("calibration/sv_subset.csv")
    sv_subset["id"] = sv_subset.apply(
        lambda row: f"{giggle_format(str(row['chr']), row['start'])}_{giggle_format(str(row['chr']), row['stop'])}",
        axis=1,
    )

    runs = os.listdir("calibration/results")
    runs.remove("results.csv")
    results_df = pd.DataFrame(
        columns=["d", "r", "q", "p", "TP", "FP", "FN", "TN"]
    )
    for dirname in runs:
        pattern = r"d([\d]+)_r([0,1].[\d]+)_q([0,1].[\d]+)_p([\d]+)"
        match = re.match(pattern, dirname)
        d, r, q, p = (
            int(match.group(1)),
            float(match.group(2)),
            float(match.group(3)),
            int(match.group(4)),
        )
        svs_n_modes = pd.read_csv(
            os.path.join("calibration/results", dirname, "svs_n_modes.csv")
        )
        confusion_mat = get_confusion_matrix(sv_subset, svs_n_modes)
        results_df.loc[len(results_df)] = [
            d,
            r,
            q,
            p,
            confusion_mat["TP"],
            confusion_mat["FP"],
            confusion_mat["FN"],
            confusion_mat["TN"],
        ]
    results_df.sort_values(by=["q", "d", "r", "p"], inplace=True)
    results_df = results_df.astype({"d": int, "p": int})
    results_df.to_csv("calibration/results_new.csv", index=False)


def run_dirichlet_inner(
    row: dict,
    sample_ids: set[str],
    input_dir: str,
    output_dir: str,
    q: float,
    d: int,
    r: float,
    pen: int,
):
    # get reads from stix file and filter reference samples
    reads, num_samples = get_raw_data(
        row,
        input_dir,
        stix_file_dir=os.path.join(input_dir, f"stix_output_{q}"),
        filter_reference_samples=True,
        samples_to_keep=list(
            sample_ids
        ),  # keep only samples with long reads (used in genotyping)
        print_messages=False,
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
                "max_penalty": pen,
                "synthetic_data": True,
                "plot": False,
                "stem": input_dir,
            },
        )

    population_size = len(sample_ids)
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
    *,
    d: int,
    r: float,
    q: float,
    pen: int,
    input_dir: str,
    output_dir: str,
    sample_ids: Set[str],
):
    """
    Run a single calibration test for given distance and reciprocal overlap thresholds.
    TODO: do analysis later on which combinations led to "partial correctness"
    """
    print(f"Running calibration for d={d}, r={r}, q={q}, p={pen}")
    processed_file_dir = os.path.join(
        SCRATCH_FILE_DIR, "d{}_r{:.2f}_p{}".format(d, r, pen)
    )
    if not os.path.exists(processed_file_dir):
        os.makedirs(processed_file_dir)

    with multiprocessing.Manager():
        p = multiprocessing.Pool(SLURM_CPUS)
        args = []
        for _, row in sv_df.iterrows():
            args.append(
                (
                    row,
                    sample_ids,
                    input_dir,
                    processed_file_dir,
                    q,
                    d,
                    r,
                    pen,
                )
            )

        p.starmap(run_dirichlet_inner, args)
        p.close()
        p.join()

    results_dir = os.path.join(
        output_dir, "d{}_r{:.2f}_q{:.2f}_p{}".format(d, r, q, pen)
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    concat_multi_processed_sv_files(
        processed_file_dir, OUTPUT_FILE_NAME, results_dir
    )
    write_post_processed_files(input_dir, results_dir, sv_df[["id"]], True)
    shutil.rmtree(processed_file_dir)

    svs_n_modes = pd.read_csv(os.path.join(results_dir, "svs_n_modes.csv"))
    results = get_confusion_matrix(
        sv_df[["id", "n_svs_actual", "q"]], svs_n_modes, q
    )

    final_output_file = os.path.join(output_dir, "results.csv")
    if not os.path.exists(final_output_file):
        with open(final_output_file, "w") as f:
            f.write("d,r,q,p,TP,FP,FN,TN\n")
    with open(final_output_file, "a") as f:
        f.write(
            "{},{},{},{},{},{},{},{}\n".format(
                d,
                r,
                q,
                pen,
                results["TP"],
                results["FP"],
                results["FN"],
                results["TN"],
            )
        )

    return results


def snap_to_grid(value: float, min_val: float, step: float) -> float:
    """Snap a continuous value to the nearest grid point defined by min + k*step."""
    k = round((value - min_val) / step)
    return round(min_val + k * step, 10)


def run_calibration_bayesian_opt(
    sv_df: pd.DataFrame,
    sample_ids: Set[str],
    *,
    input_dir: str,
    output_dir: str,
    n_trials: int = 40,
    batch_size: int = 1,
    d_min: int,
    d_max: int,
    d_step: int,
    r_min: float,
    r_max: float,
    r_step: float,
    q_min: float,
    q_max: float,
    q_step: float,
    p_min: int,
    p_max: int,
    p_step: int,
):
    """
    Run calibration using Bayesian Optimization to find the parameter
    combination that maximizes PR-AUC.

    If calibration/results/results.csv already exists, those reults are fed into
    the BO model as prior observations before any new runs are launched.
    """

    # define a BO generation strategy that starts with the center node, then transitions to Sobol for 5 trials, then transitions to BoTorch for the rest of the trials
    generator_spec = GeneratorSpec(generator_enum=Generators.BOTORCH_MODULAR)
    botorch_node = GenerationNode(
        name="BoTorch",
        generator_specs=[generator_spec],
    )
    sobol_node = GenerationNode(
        name="Sobol",
        generator_specs=[
            GeneratorSpec(
                generator_enum=Generators.SOBOL,
                generator_kwargs={"seed": 42},
            ),
        ],
        transition_criteria=[
            # transition to BoTorch node once there are 15 trials on the experiment.
            MinTrials(
                threshold=15,
                transition_to=botorch_node.name,
                use_all_trials_in_exp=True,
            )
        ],
    )
    center_node = CenterGenerationNode(next_node_name=sobol_node.name)
    gs = GenerationStrategy(
        name="Center+Sobol+BoTorch",
        nodes=[center_node, sobol_node, botorch_node],
    )

    # define the search space
    ax_client = AxClient(generation_strategy=gs, verbose_logging=False)
    ax_client.create_experiment(
        name="calibration_experiment",
        parameters=[
            {
                "name": "d",
                "type": "range",
                "bounds": [d_min, d_max],
                "value_type": "int",
            },
            {
                "name": "r",
                "type": "range",
                "bounds": [r_min, r_max],
                "value_type": "float",
            },
            {
                "name": "q",
                "type": "range",
                "bounds": [q_min, q_max],
                "value_type": "float",
            },
            {
                "name": "p",
                "type": "range",
                "bounds": [p_min, p_max],
                "value_type": "int",
            },
        ],
        objectives={"pr_auc": ObjectiveProperties(minimize=False)},
    )

    # load pre-existing results
    # if BO has been searching in one space for a while, then we should subsample
    # the existing results so the model doesn't get stuck in one space
    results_file = os.path.join(output_dir, "results.csv")
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        for i, row in df.iterrows():
            params = {
                "d": row["d"],
                "r": round(row["r"], 2),
                "q": round(snap_to_grid(row["q"], q_min, q_step), 2),
                "p": row["p"],
            }
            score = calc_pr_auc(row["TP"], row["FP"], row["FN"])
            _, trial_index = ax_client.attach_trial(params)
            ax_client.complete_trial(
                trial_index=trial_index,
                raw_data={"pr_auc": (score, None)},
            )

    # BO loop
    for trial in range(n_trials):
        print(f"Running Bayesian Optimization Trial {trial + 1}/{n_trials}")

        # ask for the next candidate(s)
        if batch_size > 1:
            parameterizations, trial_index = ax_client.get_next_trials(
                max_trials=batch_size
            )
        else:
            parameterization, trial_index = ax_client.get_next_trial()
            parameterizations = {trial_index: parameterization}

        # evaluate each candidate
        for t_index, params in parameterizations.items():
            d = int(params["d"])
            r = round(params["r"], 2)
            q = round(snap_to_grid(params["q"], q_min, q_step), 2)
            p = int(params["p"])

            # this function is parallelized for each SV but runs one calibration test for the given parameters
            results = run_calibration_test(
                sv_df,
                d=d,
                r=r,
                q=q,
                pen=p,
                input_dir=input_dir,
                output_dir=output_dir,
                sample_ids=sample_ids,
            )

        for t_index, params in parameterizations.items():
            score = calc_pr_auc(results["TP"], results["FP"], results["FN"])
            print(f"Trial {t_index} F1 score = {score:.4f}")
            ax_client.complete_trial(
                trial_index=t_index,
                raw_data={"pr_auc": (score, None)},
            )

    # after BO loop, get the best parameters and run one final calibration test with those parameters to save the outputs
    best_params, metrics = ax_client.get_best_parameters()
    best_d = int(best_params["d"])
    best_r = round(best_params["r"], 2)
    best_q = round(snap_to_grid(best_params["q"], q_min, q_step), 2)
    best_p = int(best_params["p"])

    print(
        f"Best parameters found by Bayesian Optimization: d={best_d}, r={best_r}, q={best_q}, p={best_p}"
    )
    print(f"Best F1 score: {metrics[0]['pr_auc']:.4f}")

    best_result = pd.DataFrame(
        [
            {
                "d": best_d,
                "r": best_r,
                "q": best_q,
                "p": best_p,
                "pr_auc": metrics[0]["pr_auc"],
            }
        ]
    )
    best_result.to_csv(os.path.join(output_dir, "best_params.csv"), index=False)

    return best_params


def run_calibration_grid_search(
    sv_df: pd.DataFrame,
    sample_ids: Set[str],
    *,
    input_dir: str,
    output_dir: str,
    d_min: int,
    d_max: int,
    d_step: int,
    r_min: float,
    r_max: float,
    r_step: float,
    q_min: float,
    q_max: float,
    q_step: float,
    p_min: int,
    p_max: int,
    p_step: int,
):
    """Run calibration tests over a grid of distance and reciprocal overlap thresholds."""
    # run calibration tests over grid of d, r, and q values
    for q in np.arange(q_min, q_max + 0.01, q_step):
        for d in range(d_min, d_max + 1, d_step):
            for r in np.arange(r_min, r_max + 0.01, r_step):
                for pen in range(p_min, p_max + 1, p_step):
                    run_calibration_test(
                        sv_df,
                        d=d,
                        r=round(r, 2),
                        q=round(q, 2),
                        pen=pen,
                        input_dir=input_dir,
                        output_dir=output_dir,
                        sample_ids=sample_ids,
                    )


def download_stix_data_inner(
    row: pd.Series,
    output_dir: str,
    q: float,
):
    # check if the data for this region already exists
    query_region = get_query_region(
        f"{row.chr}:{row.start}",
        f"{row.chr}:{row.stop}",
        q,
    )
    filename = f"{query_region.file_name}.txt"
    file = os.path.join(output_dir, filename)
    if os.path.isfile(file):
        return

    stix_file = query_stix_bash(
        query_region,
        output_dir,
        # hard-coded stix parameters for 1kg high coverage data on Vivian's fiji
        "/Users/vili4418/sv/stix/bin/stix",
        "/scratch/Shares/layer/stix/indices/1kg_high_coverage_vivian/shard",
        "/scratch/Shares/layer/stix/indices/1kg_high_coverage_vivian/shard",
        8,
        True,
    )
    print(f"Downloaded STIX data {stix_file}")


def download_stix_data(
    sv_subset: pd.DataFrame,
    input_dir: str,
    q: float,
):
    """Download stix data for all regions in the subset before running calibration tests."""
    output_dir = os.path.join(input_dir, f"stix_output_{q}")

    # set up the correct file paths in case we need to query more SVs
    partial_outputs_dir = os.path.join(output_dir, "partial_outputs")
    if not os.path.exists(partial_outputs_dir):
        os.makedirs(partial_outputs_dir)

    with multiprocessing.Manager():
        p = multiprocessing.Pool(SLURM_CPUS)
        args = []

        for _, row in sv_subset.iterrows():
            args.append((row, output_dir, q))

        p.starmap(download_stix_data_inner, args)
        p.close()
        p.join()

    os.rmdir(partial_outputs_dir)


def run_calibration(
    sv_lookup_file: str,
    sample_ids_file: str,
    sv_regions_file: str,
    search_func: str,
    **kwargs,
):
    input_dir = kwargs["input_dir"]

    sv_subset = pd.read_csv(os.path.join(input_dir, sv_regions_file))
    sv_subset["id"] = sv_subset.apply(
        lambda row: f"{giggle_format(str(row['chr']), row['start'])}_{giggle_format(str(row['chr']), row['stop'])}",
        axis=1,
    )
    sv_df = pd.read_csv(os.path.join(input_dir, sv_lookup_file))
    sv_df.rename(columns={"id": "og_sv_id"}, inplace=True)
    merged_df = sv_subset.merge(sv_df, on=["chr", "start", "stop"], how="left")
    sample_ids = set()
    with open(os.path.join(input_dir, sample_ids_file), "r") as f:
        for line in f:
            sample_ids.add(line.strip())

    for q in np.arange(
        kwargs["q_min"], kwargs["q_max"] + 0.01, kwargs["q_step"]
    ):
        # check that all stix data has been downloaded for regions in sv_subset before running calibration tests
        download_stix_data(sv_subset, input_dir, round(q, 2))

    if search_func == "bayesian_optimization":
        run_calibration_bayesian_opt(merged_df, sample_ids, **kwargs)
    else:
        run_calibration_grid_search(merged_df, sample_ids, **kwargs)


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
        "--search_func",
        type=str,
        help="Search function to use for calibration (grid_search or bayesian_optimization)",
        default="grid_search",
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
    parser.add_argument(
        "--q_min",
        type=float,
        help="Minimum query overlap to consider",
        default=0.5,
    )
    parser.add_argument(
        "--q_max",
        type=float,
        help="Maximum query overlap to consider",
        default=1.0,
    )
    parser.add_argument(
        "--q_step",
        type=float,
        help="Step size for reciprocal overlap values",
        default=0.1,
    )
    parser.add_argument(
        "--p_min",
        type=int,
        help="Minimum penalty size to consider",
        default=100,
    )
    parser.add_argument(
        "--p_max",
        type=int,
        help="Maximum penalty size to consider",
        default=600,
    )
    parser.add_argument(
        "--p_step",
        type=int,
        help="Step size for penalty values",
        default=100,
    )

    args = parser.parse_args()
    print("Using the following arguments:", args, "\n")

    run_calibration(
        args.sv_lookup,
        args.sample_ids,
        args.regions,
        args.search_func,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        d_min=args.d_min,
        d_max=args.d_max,
        d_step=args.d_step,
        r_min=args.r_min,
        r_max=args.r_max,
        r_step=args.r_step,
        q_min=args.q_min,
        q_max=args.q_max,
        q_step=args.q_step,
        p_min=args.p_min,
        p_max=args.p_max,
        p_step=args.p_step,
    )


if __name__ == "__main__":
    main()
