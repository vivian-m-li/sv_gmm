import argparse
from collections import defaultdict, Counter
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.model.calibrate import calibrate, get_confusion_matrix
from src.utils.config_loader import load_config
from src.utils.model_helper import giggle_format


def rewrite_calibration_results(
    truth_set: str, results_dir: str, results_file: str
):
    """
    Rewrites the results.csv file with the confusion matrix from each run.
    Standalone function
    """
    sv_subset = pd.read_csv(truth_set)
    sv_subset["id"] = sv_subset.apply(
        lambda row: f"{giggle_format(str(row['chr']), row['start'])}_{giggle_format(str(row['chr']), row['stop'])}",
        axis=1,
    )

    runs = os.listdir(results_dir)
    runs.remove(os.path.basename(results_file))
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
            os.path.join(results_dir, dirname, "svs_n_modes.csv")
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
    results_df.to_csv(
        os.path.join(results_dir, "new_" + os.path.basename(results_file)),
        index=False,
    )


def all_consensus_svs(
    results_dir: str, results_per_sv_file: str, *, plot: bool = False
):
    """Gets the distribution of mode predictions for each SV across all calibration runs."""
    out_filename = os.path.join(results_dir, results_per_sv_file)
    if not os.path.exists(out_filename):
        sv_mode_counts = defaultdict(Counter)
        for run_dir in os.listdir(results_dir):
            if not os.path.isdir(os.path.join(results_dir, run_dir)):
                continue
            svs_n_modes = pd.read_csv(
                os.path.join(results_dir, run_dir, "svs_n_modes.csv")
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
        df.to_csv(out_filename, index=False)
    else:
        df = pd.read_csv(out_filename)

    if plot:
        # plot distribution of mode predictions
        plt.figure(figsize=(8, 6))
        plt.hist(df["majority_percent"].values, bins=20, range=(0, 1))
        plt.xlabel("% of Runs Agreeing on Majority Mode")
        plt.ylabel("Number of SVs")
        plt.show()


def assign_model_score(
    results_dir: str,
    results_file: str,
    results_per_sv_file: str,
    *,
    plot: bool = False,
):
    # assign scores based on how often models agreed with the consensus
    sv_results = pd.read_csv(os.path.join(results_dir, results_per_sv_file))
    df = pd.read_csv(os.path.join(results_dir, results_file))
    for i, row in df.iterrows():
        dir = f"d{int(row['d'])}_r{row['r']:.2f}_q{row['q']:.2f}"
        svs_n_modes = pd.read_csv(
            os.path.join(results_dir, dir, "svs_n_modes.csv")
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
    df.to_csv(os.path.join(results_dir, results_file), index=False)

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


def find_pareto_front(results_dir: str, results_file: str):
    # construct tp/fp curve from results.csv files in calibration output directories
    results = pd.read_csv(os.path.join(results_dir, results_file))
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


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate SPLIT parameters across a truth set."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to the TOML configuration file (default: config.toml)",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    calibrate(cfg)


if __name__ == "__main__":
    main()
