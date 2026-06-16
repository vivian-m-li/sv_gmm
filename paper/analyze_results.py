import ast
import os
import subprocess

import numpy as np
import pandas as pd

from src.model.membership import check_group_membership
from src.utils.config_loader import load_config
from src.utils.constants import NONREF_GTS
from src.utils.helper import get_sample_ids
from src.utils.model_helper import get_insert_size_lookup


def merge_dfs(dir: str):
    svs_classified = pd.read_csv(os.path.join(dir, "svs_n_modes.csv"))
    svs_classified.rename(
        columns={
            "sv_id": "id",
            "num_modes": "n_svs_predicted",
            "num_modes_2": "n_svs_predicted_2",
        },
        inplace=True,
    )

    sample_ids = get_sample_ids("data/1kg/sample_ids.txt")

    collapsed = pd.read_csv(os.path.join(dir, "sv_stats_collapsed.csv"))
    collapsed = collapsed[
        ["id", "num_samples", "num_samples_run", "num_gmm_runs"]
    ]
    svs_classified = pd.merge(
        svs_classified,
        collapsed,
        on="id",
        how="left",
    )

    full_callset = pd.read_csv("data/1kg/1kg.subset.csv", low_memory=False)
    full_callset["nonref_samples"] = ""
    for i, row in full_callset.iterrows():
        sv_sample_ids = []
        for sample_id in sample_ids:
            gt = ast.literal_eval(row[sample_id])
            if gt in NONREF_GTS:
                sv_sample_ids.append(sample_id)
        full_callset.at[i, "nonref_samples"] = ",".join(sv_sample_ids)
    full_callset = full_callset[["id", "svlen", "af", "nonref_samples"]]

    merged = pd.merge(
        svs_classified,
        full_callset,
        on="id",
        how="left",
    )
    merged.to_csv(os.path.join(dir, "merged.csv"), index=False)


def print_split_distribution(dir: str):
    merged = pd.read_csv(os.path.join(dir, "merged.csv"))
    merged = merged[merged["confidence"] != "inconclusive"]
    n = merged.shape[0]
    for n_modes in [1, 2, 3]:
        print(f"n_modes={n_modes}")
        subset = merged[merged["n_svs_predicted"] == n_modes]
        n_svs = subset.shape[0]
        print(f"n_svs={n_svs}, ({n_svs / n:.2%})")
        for col in [
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
        for confidence in ["high", "medium", "low"]:
            n_confidence = subset[subset["confidence"] == confidence].shape[0]
            print(
                confidence,
                f"n_svs={n_confidence}, ({n_confidence / n_svs:.2%})",
            )

        print("\n")


def compare_all_svs(dir):
    all_svs = pd.read_csv("data/1kg/sv_lookup.csv")
    split_svs = pd.read_csv(os.path.join(dir, "merged.csv"))
    split_svs = split_svs[split_svs["confidence"] != "inconclusive"]

    not_split = all_svs[~all_svs["id"].isin(split_svs["id"])]

    for label, df in zip(["split", "not split"], [split_svs, not_split]):
        print(label)
        print(f"n_svs={df.shape[0]} ({df.shape[0] / all_svs.shape[0]:.2%})")
        for col in ["svlen", "af"]:
            print(
                col,
                f"mean={np.mean(df[col]):.2f}",
                f"median={np.std(df[col]):.2f}",
            )
        print("\n")


def df_to_bed(df, output_file):
    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            f.write(
                f"{row['chr']}\t{row['start']}\t{row['stop']}\t{row['id']}\n"
            )


def compare_sv_intersections(dir):
    if not os.path.exists("output/results/split_svs.bed"):
        # convert to bed files for bedtools
        merged = pd.read_csv(os.path.join(dir, "merged.csv"))
        merged = merged[merged["confidence"] != "inconclusive"]
        lookup = pd.read_csv("data/1kg/sv_lookup.csv")
        merged = pd.merge(
            merged, lookup[["id", "chr", "start", "stop"]], on="id", how="left"
        )

        split = merged[merged["n_svs_predicted"] > 1]
        not_split = merged[merged["n_svs_predicted"] == 1]

        for label, df in zip(["split", "not_split"], [split, not_split]):
            df_to_bed(df, f"output/results/{label}_svs.bed")

    # run bedtools intersect
    if not os.path.exists("output/results/split_intersect.bed"):
        cfg = load_config("config.toml")
        for label in ["split", "not_split"]:
            subprocess.run(
                [
                    "bash",
                    os.path.join(
                        os.getcwd(), "src/utils/bash/bed_intersect.sh"
                    ),
                ]
                + [
                    cfg["bedtools"]["bin"],
                    f"output/results/{label}_svs.bed",
                    "data/1kg/1kg.subset.bed",
                    f"output/results/{label}_intersect.bed",
                ],
            )

    # compare the number of SV intersections per SV in each group
    for label in ["split", "not_split"]:
        intersect = pd.read_csv(
            f"output/results/{label}_intersect.bed",
            sep="\t",
            header=None,
            names=[
                "chr",
                "start",
                "stop",
                "id",
                "chr_i",
                "start_i",
                "stop_i",
                "id_i",
            ],
        )
        # remove all rows where id == id_i (i.e. self-intersections)
        intersect = intersect[intersect["id"] != intersect["id_i"]]

        counts = intersect["id"].value_counts()
        print(label)
        print(
            f"mean intersections per sv: {np.mean(counts):.2f}, median: {np.median(counts)}"
        )
    # output

    # 0.5 overlap required
    # split: mean intersections per sv: 1.47, median: 1.0
    # not_split: mean intersections per sv: 1.21, median: 1.0

    # no minimum overlap required
    # split: mean intersections per sv: 2.03, median: 1.0
    # not_split: mean intersections per sv: 1.51, median: 1.0


def get_group_memberships(out_dir: str):
    results = pd.read_csv("output/results/most_common_split.csv")
    sample_ids = get_sample_ids("data/1kg/sample_ids.txt")
    insert_size_lookup = get_insert_size_lookup(
        "data/1kg", "insert_sizes.csv", 450, sample_ids
    )

    df = pd.DataFrame(
        columns=[
            "id",
            "cluster_idx",
            "n_members",
            "n_non_members",
            "sd_threshold",
        ]
    )
    for df_idx, sv_row in results.iterrows():
        modes = ast.literal_eval(sv_row["modes"])
        for i, mode in enumerate(modes):
            membership = check_group_membership(
                (sv_row["chr"], mode["start"], mode["end"]),
                mode["sample_ids"],
                insert_size_lookup,
                threshold=2.0,
            )
            n_members = sum(membership.values())
            n_non_members = len(membership) - n_members
            df.loc[df_idx] = [
                sv_row["id"],
                i + 1,
                n_members,
                n_non_members,
                2.0,
            ]
        print(f"Processed {df_idx + 1}/{results.shape[0]} SVs", end="\r")

    file = "membership.csv"
    df.to_csv(os.path.join(out_dir, file), index=False)


if __name__ == "__main__":
    dir = "output/results"
    # merge_dfs(dir)
    # print_split_distribution(dir)
    # compare_all_svs(dir)
    # compare_sv_intersections(dir)
    get_group_memberships(dir)
