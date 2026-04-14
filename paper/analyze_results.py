import ast
import os

import numpy as np
import pandas as pd

from src.utils.constants import NONREF_GTS
from src.utils.helper import get_sample_ids


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


if __name__ == "__main__":
    dir = "output/results"
    # merge_dfs(dir)
    # print_split_distribution(dir)
    compare_all_svs(dir)
