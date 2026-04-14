import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.model.split import split_sv
from src.utils.model_helper import giggle_format
from paper.data_processing.get_bam_files import get_bam_files

FIJI_PATH = "vili4418@fiji.colorado.edu:/Users/vili4418/sv/sv_gmm"


def sample_length_heatmap():
    best_results = pd.read_csv(
        "output/calibration/results/best_params.csv"
    ).loc[0]
    best_results_dir = "d{}_r{:.2f}_q{:.2f}_p{}".format(
        int(best_results["d"]),
        best_results["r"],
        best_results["q"],
        int(best_results["p"]),
    )
    merged = pd.read_csv(
        os.path.join(
            "output/calibration/results", best_results_dir, "merged.csv"
        )
    )

    sample_size_ranges = [
        (0, 20),
        (21, 100),
        (101, 200),
        (201, 500),
        (500, 1000),
    ]
    svlen_ranges = [
        (50, 100),
        (101, 200),
        (201, 500),
        (501, 1000),
        (1001, 30000),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    left_ax = axs[0]
    right_ax = axs[1]

    no_split = merged[merged["n_svs_actual"] == 1]
    to_split = merged[merged["n_svs_actual"] == 2]
    no_split_values = np.zeros((len(sample_size_ranges), len(svlen_ranges)))
    to_split_values = np.zeros((len(sample_size_ranges), len(svlen_ranges)))
    for ax, df, heatmap_values, true_val, false_val in (
        [left_ax, no_split, no_split_values, "tn", "fp"],
        [right_ax, to_split, to_split_values, "tp", "fn"],
    ):
        for i, (svlen_min, svlen_max) in enumerate(svlen_ranges):
            for j, (sample_size_min, sample_size_max) in enumerate(
                sample_size_ranges
            ):
                subset = df[
                    (df["num_samples_run"] >= sample_size_min)
                    & (df["num_samples_run"] <= sample_size_max)
                    & (df["svlen"] >= svlen_min)
                    & (df["svlen"] <= svlen_max)
                ]
                if subset.empty:
                    # print(
                    #     f"No samples with num_samples_run between {sample_size_min} and {sample_size_max} and svlen between {svlen_min} and {svlen_max}"
                    # )
                    continue
                n_true = subset[subset["label"] == true_val].shape[0]
                n_false = subset[subset["label"] == false_val].shape[0]
                heatmap_values[i, j] = (n_true - n_false) / (n_true + n_false)

                # add text to cell with TP/FP or TP/FN values
                text = f"{n_true}/{n_false}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                )

    left_im = left_ax.imshow(no_split_values, cmap="RdBu", vmin=-1, vmax=1)
    fig.colorbar(left_im, ax=left_ax)
    left_ax.set_title("n_svs_actual = 1", fontsize=16)

    right_im = right_ax.imshow(to_split_values, cmap="RdBu", vmin=-1, vmax=1)
    fig.colorbar(right_im, ax=right_ax)
    right_ax.set_title("n_svs_actual = 2", fontsize=16)

    for ax in axs:
        ax.set_xticks(np.arange(len(sample_size_ranges)))
        ax.set_yticks(np.arange(len(sample_size_ranges)))
        ax.set_xticklabels(
            [f"{min}-{max}" for min, max in sample_size_ranges], rotation=45
        )
        ax.set_yticklabels(
            [f"{min}-{max}" for min, max in svlen_ranges], rotation=45
        )
        ax.set_xlabel("Sample Size", fontsize=14)
        ax.set_ylabel("SV Length", fontsize=14)

    plt.tight_layout()
    plt.show()


def merge_dfs(dir):
    svs_classified = pd.read_csv(
        os.path.join("output/calibration/results", dir, "svs_n_modes.csv")
    )
    svs_classified.rename(
        columns={
            "num_modes": "n_svs_predicted",
            "num_modes_2": "n_svs_predicted_2",
        },
        inplace=True,
    )
    subset_run = pd.read_csv(
        "output/calibration/sv_subset_sr_lr_nonref_most_samples.csv"
    )
    subset_run["sv_id"] = subset_run.apply(
        lambda row: f"{giggle_format(row['chr'], row['start'])}_{giggle_format(row['chr'], row['stop'])}",
        axis=1,
    )
    svs_classified = pd.merge(
        svs_classified, subset_run, on="sv_id", how="left"
    )
    svs_classified["label"] = ""
    for i, row in svs_classified.iterrows():
        label = ""
        if row["n_svs_actual"] == 2 and row["n_svs_predicted"] >= 2:
            label = "tp"
        elif row["n_svs_actual"] == 1 and row["n_svs_predicted"] >= 2:
            label = "fp"
        elif row["n_svs_actual"] == 1 and row["n_svs_predicted"] == 1:
            label = "tn"
        elif row["n_svs_actual"] == 2 and row["n_svs_predicted"] == 1:
            label = "fn"
        svs_classified.at[i, "label"] = label

    sv_runs = pd.read_csv(
        os.path.join("output/calibration/results", dir, "most_common_split.csv")
    )[
        [
            "id",
            "svlen",
            "af",
            "num_samples",
            "num_samples_run",
            "num_gmm_runs",
            "modes",
        ]
    ]
    sv_runs.rename(columns={"id": "sv_id"}, inplace=True)
    df = pd.merge(svs_classified, sv_runs, on="sv_id")

    truth_set = pd.read_csv(
        "data/calibration/1k_sr_sv_non_ref-1k_sr_lr_gt_non_ref.bed.gz",
        delimiter="\t",
    )
    truth_set["chr"] = truth_set.apply(
        lambda row: row["#chrom"].strip("chr"), axis=1
    )
    truth_set["sv_id"] = truth_set.apply(
        lambda row: f"{giggle_format(row['chr'], row['start'])}_{giggle_format(row['chr'], row['end'])}",
        axis=1,
    )
    truth_set.rename(columns={"svid": "id"}, inplace=True)

    df = pd.merge(
        df,
        truth_set[["id", "sv_id", "jaccard", "sr_lr_non_ref", "sr_non_ref"]],
        on="sv_id",
        how="left",
    )
    df = df[
        [
            "id",
            "chr",
            "start",
            "stop",
            "svlen",
            "af",
            "sr_lr_non_ref",
            "sr_non_ref",
            "jaccard",
            "n_svs_actual",
            "n_svs_predicted",
            "n_svs_predicted_2",
            "label",
            "confidence",
            "ci_lower",
            "ci_upper",
            "num_samples",
            "num_samples_run",
            "num_gmm_runs",
            "modes",
        ]
    ]
    return df


def copy_result_files():
    results_file = "output/calibration/results/results.csv"
    if not os.path.exists(results_file):
        for file in [
            results_file,
            "output/calibration/results/best_params.csv",
        ]:
            subprocess.run(
                [
                    "scp",
                    f"{FIJI_PATH}/output/calibration/results/{os.path.basename(file)}",
                    file,
                ]
            )
    results = pd.read_csv(results_file)
    for _, row in results.iterrows():
        d = int(row["d"])
        r = f"{row['r']:.2f}"
        q = f"{row['q']:.2f}"
        p = int(row["p"])
        dir_name = os.path.join(
            "output/calibration/results", f"d{d}_r{r}_q{q}_p{p}"
        )
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            for filename in ["svs_n_modes.csv", "most_common_split.csv"]:
                subprocess.run(
                    [
                        "scp",
                        f"{FIJI_PATH}/{dir_name}/{filename}",
                        dir_name,
                    ]
                )


def add_evaluation_metrics():
    results = pd.read_csv("output/calibration/results/results.csv")
    results["accuracy"] = (results["TP"] + results["TN"]) / (
        results["TP"] + results["FP"] + results["TN"] + results["FN"]
    )
    results["precision"] = results["TP"] / (results["TP"] + results["FP"])
    results["recall"] = results["TP"] / (results["TP"] + results["FN"])
    results["f1"] = (2 * results["precision"] * results["recall"]) / (
        results["precision"] + results["recall"]
    )
    results.to_csv("output/calibration/results/results.csv", index=False)


def merge_dfs_all():
    results = pd.read_csv("output/calibration/results/results.csv")
    for _, row in results.iterrows():
        d = int(row["d"])
        r = f"{row['r']:.2f}"
        q = f"{row['q']:.2f}"
        p = int(row["p"])
        dir_name = os.path.join(
            "output/calibration/results", f"d{d}_r{r}_q{q}_p{p}"
        )
        merged_df_path = os.path.join(dir_name, "merged.csv")
        if not os.path.exists(merged_df_path):
            full_df = merge_dfs(f"d{d}_r{r}_q{q}_p{p}")
            full_df.to_csv(merged_df_path, index=False)


def print_class_distribution():
    best_results = pd.read_csv(
        "output/calibration/results/best_params.csv"
    ).loc[0]
    best_results_dir = "d{}_r{:.2f}_q{:.2f}_p{}".format(
        int(best_results["d"]),
        best_results["r"],
        best_results["q"],
        int(best_results["p"]),
    )
    merged = pd.read_csv(
        os.path.join(
            "output/calibration/results", best_results_dir, "merged.csv"
        )
    )
    print("Best params: d={}, r={}, q={}, p={}".format(*best_results), "\n")
    for label in ["tp", "fn", "tn", "fp"]:
        print(label)
        subset = merged[merged["label"] == label]
        print(f"n_svs={subset.shape[0]}")
        for col in [
            "jaccard",
            "num_samples_run",
            "num_gmm_runs",
            "svlen",
            "af",
        ]:
            print(
                col,
                f"mean={np.mean(subset[col]):.2f}",
                f"median={np.std(subset[col]):.2f}",
            )
        print("confidence:", subset["confidence"].value_counts())
        print("\n")


def copy_viz_files():
    viz_subset = "data/calibration/viz_subset.csv"
    if not os.path.exists(viz_subset):
        subprocess.run(
            [
                "scp",
                "vili4418@fiji.colorado.edu:/Users/vili4418/sv/sv_gmm/data/calibration/viz_subset.csv",
                viz_subset,
            ]
        )
    df = pd.read_csv(viz_subset)

    for _, row in df.iterrows():
        sv_id = row["id"]
        label = row["label"]
        sv_string = f"{giggle_format(row['chr'], row['start'])}_{giggle_format(row['chr'], row['stop'])}"

        print(f"Processing {sv_id}...")

        # copy cluster files and move them to appropriate directory
        cluster_dir = f"output/calibration/viz/clusters/{label}"
        os.makedirs(cluster_dir, exist_ok=True)

        cluster_file = f"{cluster_dir}/{sv_id}.png"
        if not os.path.exists(cluster_file):
            print("Copying clusters plot...")
            subprocess.run(
                [
                    "scp",
                    f"vili4418@fiji.colorado.edu:/Users/vili4418/sv/sv_gmm/output/calibration/plots/{sv_string}.png",
                    cluster_file,
                ],
                stdout=subprocess.DEVNULL,
            )

        # copy bam files
        samplot_dir = f"output/calibration/viz/samplots/{label}"
        os.makedirs(samplot_dir, exist_ok=True)

        bam_file_dir = "data/calibration/bam_files"
        if not os.path.exists(os.path.join(bam_file_dir, sv_id)):
            print("Getting bam files for samplot...")
            subprocess.run(
                [
                    "scp",
                    "-r",
                    f"vili4418@fiji.colorado.edu:/Users/vili4418/sv/sv_gmm/data/long_reads/bam_files/{sv_id}",
                    bam_file_dir,
                ],
                stdout=subprocess.DEVNULL,
            )

        # run samplot on the bam files
        samplot_sv_dir = os.path.join(
            "output/calibration/viz/samplots", label, sv_id
        )
        if not os.path.exists(samplot_sv_dir):
            print("Compiling samplot...")
            os.makedirs(samplot_sv_dir, exist_ok=True)

            bam_files = os.listdir(os.path.join(bam_file_dir, sv_id))
            sample_ids = [
                bam_file.split(".")[0]
                for bam_file in bam_files
                if bam_file.endswith(".bam")
            ]

            # rename bam files dir for batched samplot viz
            os.rename(
                os.path.join(bam_file_dir, sv_id),
                os.path.join(bam_file_dir, f"{sv_id}_temp"),
            )
            os.makedirs(os.path.join(bam_file_dir, sv_id), exist_ok=True)

            # batch samples in groups of 10 for easier visual validation
            n_samplots = len(sample_ids) // 10 + 1
            for i in range(n_samplots):
                sample_ids_subset = sample_ids[i * 10 : (i + 1) * 10]

                # copy subset of bam files for samplot viz
                for sample_id in sample_ids_subset:
                    for ext in [".bam", ".bam.bai"]:
                        filename = os.path.join(
                            bam_file_dir,
                            f"{sv_id}_temp",
                            f"{sample_id}{ext}",
                        )
                        if os.path.exists(filename):
                            os.rename(
                                filename,
                                os.path.join(
                                    bam_file_dir, sv_id, f"{sample_id}{ext}"
                                ),
                            )

                subprocess.run(
                    [
                        "bash",
                        os.path.join(
                            os.getcwd(),
                            "src/utils/bash/samplot_viz.sh",
                        ),
                    ]
                    + [
                        sv_id,
                        str(row["chr"]),
                        str(row["start"]),
                        str(row["stop"]),
                        bam_file_dir,
                        samplot_sv_dir,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                # rename the samplot file with the batch number
                samplot_output_file = os.path.join(
                    samplot_sv_dir, f"{sv_id}.png"
                )
                if os.path.exists(samplot_output_file):
                    batch_file_name = os.path.join(
                        samplot_sv_dir, f"{sv_id}_{i + 1}.png"
                    )
                    os.rename(samplot_output_file, batch_file_name)

                # copy files back to temp dir to prepare for next batch
                for file in os.listdir(os.path.join(bam_file_dir, sv_id)):
                    os.rename(
                        os.path.join(bam_file_dir, sv_id, file),
                        os.path.join(bam_file_dir, f"{sv_id}_temp", file),
                    )

            # remove the subset dir and rename temp dir back to original
            os.rmdir(os.path.join(bam_file_dir, sv_id))
            os.rename(
                os.path.join(bam_file_dir, f"{sv_id}_temp"),
                os.path.join(bam_file_dir, sv_id),
            )


def build_viz_subset():
    filename = "data/calibration/viz_subset.csv"

    best_results = pd.read_csv(
        "output/calibration/results/best_params.csv"
    ).loc[0]
    best_results_dir = "d{}_r{:.2f}_q{:.2f}_p{}".format(
        int(best_results["d"]),
        best_results["r"],
        best_results["q"],
        int(best_results["p"]),
    )
    if not os.path.exists(filename):
        merged_df_path = os.path.join(
            "output/calibration/results", best_results_dir, "merged.csv"
        )
        if not os.path.exists(merged_df_path):
            full_df = merge_dfs(best_results_dir)
            full_df.to_csv(merged_df_path, index=False)
        else:
            full_df = pd.read_csv(merged_df_path)
        df = pd.concat(
            [
                full_df[full_df["label"] == label].sample(n=10)
                for label in ["tp", "fp", "tn", "fn"]
            ]
        )
        df.to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)

    for _, row in df.iterrows():
        sv_id = row["id"]

        # download bam files - samplots will be generated locally
        if not os.path.exists(f"data/long_reads/bam_files/{sv_id}"):
            print(f"Getting bam files for {sv_id}...")
            get_bam_files(sv_id)

        # save the plot for one iteration of the gmm plot
        plot_file = f"{giggle_format(str(row['chr']), row['start'])}_{giggle_format(str(row['chr']), row['stop'])}.png"
        if not os.path.exists(
            os.path.join("output/calibration/plots", plot_file)
        ):
            split_sv(
                sv_id=sv_id,
                d_threshold=best_results["d"],
                r_threshold=best_results["r"],
                max_penalty=best_results["p"],
                input_dir="calibration",
                insert_size_file="data/calibration/insert_sizes.csv",
                sample_id_file="data/calibration/sample_ids.csv",
                stix_file_dir=f"stix_output_{best_results['q']}",
            )


if __name__ == "__main__":
    # copy_result_files()
    # add_evaluation_metrics()
    # merge_dfs_all()
    # build_viz_subset()
    # copy_viz_files()

    # print_class_distribution()
    sample_length_heatmap()
