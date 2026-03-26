import os
import shutil
import subprocess
import pandas as pd
from get_bam_files import get_bam_files
from query_sv import (
    giggle_format,
    query_stix,
)


def merge_dfs(dir):
    svs_classified = pd.read_csv(
        os.path.join("calibration/results", dir, "svs_n_modes.csv")
    )
    svs_classified.rename(
        columns={
            "num_modes": "n_svs_predicted",
            "num_modes_2": "n_svs_predicted_2",
        },
        inplace=True,
    )
    subset_run = pd.read_csv("calibration/sv_subset.csv")
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
        os.path.join("calibration/results", dir, "sv_stats_collapsed.csv")
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
        "calibration/1k_sr_sv_non_ref-1k_sr_lr_gt_non_ref.bed.gz",
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


def copy_viz_files():
    df = pd.read_csv("calibration/viz_subset.csv")
    # copy cluster files over
    subprocess.run(
        [
            "scp",
            "vili4418@fiji.colorado.edu:/Users/vili4418/sv/sv_gmm/calibration/plots/*",
            "calibration/plots/",
        ]
    )

    for _, row in df.iterrows():
        sv_id = row["id"]
        label = row["label"]
        sv_string = f"{giggle_format(row['chr'], row['start'])}_{giggle_format(row['chr'], row['stop'])}"

        print(f"Processing {sv_id}...")

        # copy cluster files and move them to appropriate directory
        cluster_dir = f"calibration/viz/clusters/{label}"
        if not os.path.exists(cluster_dir):
            os.mkdir(cluster_dir)
        cluster_file = f"calibration/plots/{sv_string}.png"
        shutil.move(
            cluster_file,
            f"calibration/viz/clusters/{label}/{sv_id}.png",
        )

        # copy bam files
        samplot_dir = f"calibration/viz/samplots/{label}"
        if not os.path.exists(samplot_dir):
            os.mkdir(samplot_dir)

        if not os.path.exists(f"calibration/bam_files/{sv_id}"):
            subprocess.run(
                [
                    "scp",
                    "-r",
                    f"vili4418@fiji.colorado.edu:/Users/vili4418/sv/sv_gmm/long_reads/bam_files/{sv_id}",
                    "calibration/bam_files",
                ]
            )

        # run samplot on the bam files
        if not os.path.exists(f"calibration/viz/samplots/{label}/{sv_id}.png"):
            print("Compiling samplot")
            subprocess.run(
                ["bash", "samplot_viz.sh"]
                + [
                    sv_id,
                    str(row["chr"]),
                    str(row["start"]),
                    str(row["stop"]),
                    "calibration/bam_files",
                    f"calibration/viz/samplots/{label}",
                ],
                capture_output=True,
                text=True,
            )


def build_viz_subset():
    filename = "calibration/viz_subset.csv"

    best_results = pd.read_csv("calibration/results/best_params.csv").loc[0]
    best_results_dir = "d{}_r{:.2f}_q{:.2f}_p{}".format(
        int(best_results["d"]),
        best_results["r"],
        best_results["q"],
        int(best_results["p"]),
    )
    if not os.path.exists(filename):
        full_df = merge_dfs(best_results_dir)
        full_df.to_csv("calibration/merged.csv", index=False)
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
        if not os.path.exists(f"long_reads/bam_files/{sv_id}"):
            print(f"Getting bam files for {sv_id}...")
            get_bam_files(sv_id)

        # save the plot for one iteration of the gmm plot
        plot_file = f"{giggle_format(str(row['chr']), row['start'])}_{giggle_format(str(row['chr']), row['stop'])}.png"
        if not os.path.exists(os.path.join("calibration/plots", plot_file)):
            query_stix(
                sv_id=sv_id,
                d_threshold=best_results["d"],
                r_threshold=best_results["r"],
                max_penalty=best_results["p"],
                input_dir="calibration",
                stix_file_dir=f"stix_output_{best_results['q']}",
            )


if __name__ == "__main__":
    build_viz_subset()
    copy_viz_files()
