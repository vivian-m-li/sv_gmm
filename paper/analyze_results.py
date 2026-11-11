import ast
from collections import defaultdict
import csv
import os
import subprocess

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from paper.helper import get_ancestry_lookup
from src.model.membership import check_group_membership
from src.utils.config_loader import load_config
from src.utils.constants import NONREF_GTS, SUBPOPULATIONS, SUPERPOPULATIONS
from src.utils.helper import get_sample_ids, get_most_common_split_df
from src.utils.model_helper import get_insert_size_lookup, reciprocal_overlap


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

    collapsed = pd.read_csv(os.path.join(dir, "most_common_split.csv"))
    collapsed = collapsed[
        [
            "id",
            "num_samples",
            "num_samples_run",
            "num_gmm_runs",
            "n_split_samples",
            "n_paired_samples",
        ]
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
    """Deprecated due to changes in confidence scoring."""
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
        if "sv_id" not in df.columns:
            df["sv_id"] = df["id"].astype(str)
        if "stop" not in df.columns:
            df["stop"] = df["end"]
        for _, row in df.iterrows():
            f.write(
                f"{row['chr']}\t{row['start']}\t{row['stop']}\t{row['sv_id']}\t{row['id']}\n"
            )


def bed_to_df(bed_file: str, remove_dupes: bool = False) -> pd.DataFrame:
    """Convert a BED file to a pandas DataFrame."""
    df = pd.read_csv(
        bed_file,
        delimiter="\t",
        names=[
            "sv1_chr",
            "sv1_l",
            "sv1_r",
            "sv1_id",
            "sv1_id_dup",
            "sv2_chr",
            "sv2_l",
            "sv2_r",
            "sv2_id",
            "sv2_svid",
        ],
    )
    if remove_dupes:
        df = df[df["sv1_id"] != df["sv2_id"]]

    df["r"] = df.apply(
        lambda row: reciprocal_overlap(
            (row["sv1_l"], row["sv1_r"]), (row["sv2_l"], row["sv2_r"])
        ),
        axis=1,
    )
    return df


def get_overlapping_clustered_svs(cfg: dict):
    """Checks for overlaps between clustered SVs and the original SV breakpoints."""
    overlap_threshold = 0.5
    if not os.path.exists("output/results/sv_lookup_intersect.csv"):
        lookup = pd.read_csv("data/1kg/sv_lookup.csv")
        lookup["sv_id"] = lookup["id"].astype(str)
        lookup.rename(columns={"stop": "end"}, inplace=True)
        bed_file = "data/1kg/sv_lookup.bed"
        df_to_bed(lookup, bed_file)
        consensus_svs_file = "output/results/consensus_svs.bed"
        if not os.path.exists(consensus_svs_file):
            df_to_bed(
                pd.read_csv("output/results/consensus_svs.csv"),
                consensus_svs_file,
            )

        output_bed_file = "output/results/sv_lookup_intersect.bed"
        bedtools_bin = cfg["bedtools"]["bin"]
        subprocess.run(
            ["bash", "src/utils/bash/bed_intersect.sh"]
            + [
                bedtools_bin,
                bed_file,
                consensus_svs_file,
                output_bed_file,
            ],
            capture_output=True,
            text=True,
        )

        df = bed_to_df(output_bed_file)
        df.to_csv("output/results/sv_lookup_intersect.csv", index=False)
    else:
        df = pd.read_csv("output/results/sv_lookup_intersect.csv")

    overlaps = df[
        (df["r"] >= overlap_threshold) & (df["sv1_id"] != df["sv2_id"])
    ]

    # this gets all overlapping SVs - however, we want to check whether the original SVs were overlapping as well (pre-clustering)
    # print(
    #     overlaps[["sv1_id", "sv1_l", "sv1_r", "sv2_id", "sv2_l", "sv2_r", "r"]]
    # )

    og_overlaps = pd.read_csv("data/1kg/og_svs_intersect.csv")
    og_overlaps = og_overlaps[og_overlaps["r"] >= overlap_threshold]
    # only need sv ids to check for overlaps
    og_overlaps = og_overlaps[["sv1_id", "sv2_id"]]

    # merge new overlaps with og_overlaps
    new_overlaps = overlaps.merge(
        og_overlaps,
        left_on=["sv1_id", "sv2_id"],
        right_on=["sv1_id", "sv2_id"],
        how="left",
        indicator=True,
    )
    # filter for entries in overlaps that are not in og_overlaps
    new_overlaps = new_overlaps[new_overlaps["_merge"] == "left_only"]

    # there are 22 overlapping SVs (>= 0.7 reciprocal overlap)
    print(new_overlaps.shape[0], new_overlaps)


def compare_sv_intersections(dir):
    if not os.path.exists("output/results/split_svs.bed"):
        # convert to bed files for bedtools
        merged = pd.read_csv("output/results/merged.csv")
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
    """Get the number of members and non-members for each SV mode using insert size distributions."""
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


def logistic_regression():
    merged = pd.read_csv("output/results/merged.csv")

    # only include SVs that were clustered by SPLIT
    merged = merged[merged["num_samples_run"] > 10]

    merged["is_split"] = (merged["n_svs_predicted"] > 1).astype(int)
    merged["clustering_type"] = "mixed"
    merged.loc[
        ((merged["n_split_samples"] > 0) & (merged["n_paired_samples"] == 0)),
        "clustering_type",
    ] = "split"
    merged.loc[
        ((merged["n_split_samples"] == 0) & (merged["n_paired_samples"] > 0)),
        "clustering_type",
    ] = "paired"

    # one-hot encode clustering type
    clustering_dummies = pd.get_dummies(
        merged["clustering_type"], prefix="clustering"
    )

    # features and target variable
    X = pd.concat(
        [merged[["af", "num_samples", "svlen"]], clustering_dummies], axis=1
    )
    y = merged["is_split"]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # predictions and evaluation
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("Feature coefficients:")
    for feature, coef in zip(X.columns, model.coef_[0]):
        print(f"{feature}: {coef:.4f}")


def sample_sv_comparison(
    sample_sv_lookup: pd.DataFrame, sample_ids: list[str], split_svs: set[str]
):
    # aggregate the dataframe to have one row per sample to reduce # of operations in the nested loop below
    sample_sv_lookup = (
        sample_sv_lookup.groupby("sample_id")
        .agg(
            original_sv_ids=("original_sv_id", lambda x: list(x.unique())),
            sv_ids=("sv_id", lambda x: list(x.unique())),
        )
        .reset_index()
    )
    sample_lookup, population_lookup = get_ancestry_lookup()

    results = {}
    for s1_idx, s1 in enumerate(sample_ids):
        s1_population = sample_lookup[s1]
        s1_superpopulation = population_lookup[s1_population]

        s1_svs_df = sample_sv_lookup[sample_sv_lookup["sample_id"] == s1]
        s1_original_svs = set(s1_svs_df["original_sv_ids"].values[0])
        s1_svs = set(s1_svs_df["sv_ids"].values[0])
        for s2_idx, s2 in enumerate(sample_ids[s1_idx + 1 :]):
            print(
                f"Processing s1 {s1_idx + 1}/{len(sample_ids)}, s2 {s2_idx + 1}/{len(sample_ids[s1_idx + 1 :])}",
                end="\r",
            )
            s2_population = sample_lookup[s2]
            s2_superpopulation = population_lookup[s2_population]

            s2_svs_df = sample_sv_lookup[sample_sv_lookup["sample_id"] == s2]
            s2_original_svs = set(s2_svs_df["original_sv_ids"].values[0])
            s2_svs = set(s2_svs_df["sv_ids"].values[0])

            # we only want to count SVs that originate from the same original SV
            shared_svs = s1_original_svs.intersection(s2_original_svs)

            # now we need to count how many of the new SVs (only those that are shared) are shared between the two samples
            s1_new_svs = set(
                sv for sv in s1_svs if sv.rsplit("_", 1)[0] in shared_svs
            )
            s2_new_svs = set(
                sv for sv in s2_svs if sv.rsplit("_", 1)[0] in shared_svs
            )
            n_shared_svs = len(s1_new_svs.intersection(s2_new_svs))
            n_all_svs = len(shared_svs)

            # now count only SVs that are split
            shared_split_svs = shared_svs.intersection(split_svs)
            s1_split_new_svs = set(
                sv
                for sv in s1_new_svs
                if sv.rsplit("_", 1)[0] in shared_split_svs
            )
            s2_split_new_svs = set(
                sv
                for sv in s2_new_svs
                if sv.rsplit("_", 1)[0] in shared_split_svs
            )
            n_shared_split_svs = len(
                s1_split_new_svs.intersection(s2_split_new_svs)
            )
            n_all_split_svs = len(shared_split_svs)

            results[(s1, s2)] = {
                "n_shared_svs": n_shared_svs,
                "n_all_svs": n_all_svs,
                "fraction_shared_svs": n_shared_svs / n_all_svs,
                "n_shared_split_svs": n_shared_split_svs,
                "n_all_split_svs": n_all_split_svs,
                "fraction_shared_split_svs": n_shared_split_svs
                / n_all_split_svs,
                "s1_population": s1_population,
                "s2_population": s2_population,
                "s1_superpopulation": s1_superpopulation,
                "s2_superpopulation": s2_superpopulation,
            }

    return results


def generate_ancestry_null_distribution(n_distributions: int = 1000):
    """Generate a null distribution of the likelihood of two (super)populations co-occurring in the same mode by randomly shuffling SVs between samples."""
    lookup = pd.read_csv("output/results/sample_ancestry_lookup.csv")
    sample_ids = list(lookup["sample_id"].unique())
    splits = get_most_common_split_df()
    split_svs = set(splits[splits["num_modes"] > 1]["id"].values)

    columns = [
        f"{pop1},{pop2}" for pop1 in SUBPOPULATIONS for pop2 in SUBPOPULATIONS
    ] + [
        f"{superpop1},{superpop2}"
        for superpop1 in SUPERPOPULATIONS
        for superpop2 in SUPERPOPULATIONS
    ]

    with (
        open(
            "output/results/ancestry_null_distributions_all_svs.csv",
            "w",
            newline="",
        ) as all_svs_file,
        open(
            "output/results/ancestry_null_distributions_split_svs.csv",
            "w",
            newline="",
        ) as split_svs_file,
    ):
        all_svs_writer = csv.writer(all_svs_file)
        split_svs_writer = csv.writer(split_svs_file)

        all_svs_writer.writerow(columns)
        split_svs_writer.writerow(columns)

        files = os.listdir("output/results/ancestry_null_distributions")
        for i, file in enumerate(files):
            print(f"file {i + 1}/{len(files)}")
            assignments = pd.read_csv(
                f"output/results/ancestry_null_distributions/{file}"
            )
            comparisons = sample_sv_comparison(
                assignments, sample_ids, split_svs
            )
            pop_comparisons = defaultdict(list)
            superpop_comparisons = defaultdict(list)
            pop_comparisons_split = defaultdict(list)
            superpop_comparisons_split = defaultdict(list)

            for result in comparisons.values():
                pop1 = result["s1_population"]
                pop2 = result["s2_population"]
                superpop1 = result["s1_superpopulation"]
                superpop2 = result["s2_superpopulation"]
                pop_comparisons[f"{pop1},{pop2}"].append(
                    result["fraction_shared_svs"]
                )
                superpop_comparisons[f"{superpop1},{superpop2}"].append(
                    result["fraction_shared_svs"]
                )
                pop_comparisons_split[f"{pop1},{pop2}"].append(
                    result["fraction_shared_split_svs"]
                )
                superpop_comparisons_split[f"{superpop1},{superpop2}"].append(
                    result["fraction_shared_split_svs"]
                )

            all_svs_writer.writerow(
                [
                    np.mean(pop_comparisons[f"{pop1},{pop2}"])
                    for pop1 in SUBPOPULATIONS
                    for pop2 in SUBPOPULATIONS
                ]
                + [
                    np.mean(superpop_comparisons[f"{superpop1},{superpop2}"])
                    for superpop1 in SUPERPOPULATIONS
                    for superpop2 in SUPERPOPULATIONS
                ]
            )
            split_svs_writer.writerow(
                [
                    np.mean(pop_comparisons_split[f"{pop1},{pop2}"])
                    for pop1 in SUBPOPULATIONS
                    for pop2 in SUBPOPULATIONS
                ]
                + [
                    np.mean(
                        superpop_comparisons_split[f"{superpop1},{superpop2}"]
                    )
                    for superpop1 in SUPERPOPULATIONS
                    for superpop2 in SUPERPOPULATIONS
                ]
            )
            all_svs_file.flush()
            split_svs_file.flush()

    with (
        open(
            "output/results/ancestry_null_distributions_all_svs.csv",
            "a",
            newline="",
        ) as all_svs_file,
        open(
            "output/results/ancestry_null_distributions_split_svs.csv",
            "a",
            newline="",
        ) as split_svs_file,
    ):
        for i in range(n_distributions):
            print(f"Generating null distribution {i + 1}/{n_distributions}")
            assignments = lookup.copy()
            assignments["sv_id"] = np.random.permutation(
                assignments["sv_id"].values
            )
            comparisons = sample_sv_comparison(
                assignments, sample_ids, split_svs
            )
            pop_comparisons = defaultdict(list)
            superpop_comparisons = defaultdict(list)
            pop_comparisons_split = defaultdict(list)
            superpop_comparisons_split = defaultdict(list)

            for result in comparisons.values():
                pop1 = result["s1_population"]
                pop2 = result["s2_population"]
                superpop1 = result["s1_superpopulation"]
                superpop2 = result["s2_superpopulation"]
                pop_comparisons[f"{pop1},{pop2}"].append(
                    result["fraction_shared_svs"]
                )
                superpop_comparisons[f"{superpop1},{superpop2}"].append(
                    result["fraction_shared_svs"]
                )
                pop_comparisons_split[f"{pop1},{pop2}"].append(
                    result["fraction_shared_split_svs"]
                )
                superpop_comparisons_split[f"{superpop1},{superpop2}"].append(
                    result["fraction_shared_split_svs"]
                )

            all_svs_writer.writerow(
                [
                    np.mean(pop_comparisons[f"{pop1},{pop2}"])
                    for pop1 in SUBPOPULATIONS
                    for pop2 in SUBPOPULATIONS
                ]
                + [
                    np.mean(superpop_comparisons[f"{superpop1},{superpop2}"])
                    for superpop1 in SUPERPOPULATIONS
                    for superpop2 in SUPERPOPULATIONS
                ]
            )
            split_svs_writer.writerow(
                [
                    np.mean(pop_comparisons_split[f"{pop1},{pop2}"])
                    for pop1 in SUBPOPULATIONS
                    for pop2 in SUBPOPULATIONS
                ]
                + [
                    np.mean(
                        superpop_comparisons_split[f"{superpop1},{superpop2}"]
                    )
                    for superpop1 in SUPERPOPULATIONS
                    for superpop2 in SUPERPOPULATIONS
                ]
            )


def compare_sample_ancestries():
    """Generate a csv file with the fraction of shared SVs between all pairs of samples, along with their populations and superpopulations."""

    sample_ancestry_path = "output/results/sample_ancestry_lookup.csv"
    sample_sv_lookup = None
    if os.path.exists(sample_ancestry_path):
        sample_sv_lookup = pd.read_csv(sample_ancestry_path)
        processed_svs = set(sample_sv_lookup["sv_id"].values)
    else:
        processed_svs = set()

    splits = get_most_common_split_df()
    sample_lookup, population_lookup = get_ancestry_lookup()
    if sample_sv_lookup is None or len(processed_svs) < splits.shape[0]:
        # updated to csvwriter instead of pandas for efficiency
        with open(sample_ancestry_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if sample_sv_lookup is None or sample_sv_lookup.empty:
                writer.writerow(
                    [
                        "original_sv_id",
                        "sv_id",
                        "sample_id",
                        "population",
                        "superpopulation",
                    ]
                )
            for i, row in splits.iterrows():
                print(f"Processing row {i + 1}/{splits.shape[0]}", end="\r")
                sv_id = row["id"]
                if sv_id in processed_svs:
                    continue

                modes = ast.literal_eval(row["modes"])
                for mode_idx, mode in enumerate(modes):
                    for sample_id in mode["sample_ids"]:
                        population = sample_lookup[sample_id]
                        superpopulation = population_lookup[population]
                        writer.writerow(
                            [
                                sv_id,
                                f"{sv_id}_{mode_idx + 1}",
                                sample_id,
                                population,
                                superpopulation,
                            ]
                        )
                if i % 1000 == 0:
                    csvfile.flush()
        sample_sv_lookup = pd.read_csv(sample_ancestry_path)

    split_svs = set(splits[splits["num_modes"] > 1]["id"].values)
    sample_ids = list(sample_sv_lookup["sample_id"].unique())

    df_path = "output/results/sample_ancestry_comparison.csv"
    write_header = not os.path.exists(df_path)

    with open(df_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(
                [
                    "sample_1",
                    "sample_2",
                    "num_shared_svs",
                    "num_total_svs",
                    "fraction_shared_svs",
                    "num_shared_split_svs",
                    "num_total_split_svs",
                    "fraction_shared_split_svs",
                    "sample_1_population",
                    "sample_2_population",
                    "sample_1_superpopulation",
                    "sample_2_superpopulation",
                ]
            )

        comparisons = sample_sv_comparison(
            sample_sv_lookup, sample_ids, split_svs
        )
        for (s1, s2), result in comparisons.items():
            writer.writerow(
                [
                    s1,
                    s2,
                    result["n_shared_svs"],
                    result["n_all_svs"],
                    result["n_shared_svs"] / result["n_all_svs"],
                    result["n_shared_split_svs"],
                    result["n_all_split_svs"],
                    (
                        result["n_shared_split_svs"] / result["n_all_split_svs"]
                        if result["n_all_split_svs"] > 0
                        else 0
                    ),
                    result["s1_population"],
                    result["s2_population"],
                    result["s1_superpopulation"],
                    result["s2_superpopulation"],
                ]
            )
        csvfile.flush()


def split_vs_dominant():
    """Analyze split clusters vs. dominant clusters"""
    df = pd.read_csv("output/results/most_common_split.csv")
    df = df[df["num_samples_run"] > 10]
    for _, row in df.iterrows():
        sv_coords = row["start", row["stop"]]
        # get the dominant cluster and the split clusters
        # the dominant cluster here is defined by the cluster with the most samples
        modes = ast.literal_eval(row["modes"])
        dominant_cluster_idx = max(
            range(len(modes)), key=lambda i: len(modes[i]["sample_ids"])
        )
        dominant_cluster = modes[dominant_cluster_idx]
        overlaps = [
            reciprocal_overlap((m["start"], m["stop"]), sv_coords)
            for m in modes
        ]

        # compare the dominant cluster to the split clusters
        # is the dominant cluster bigger or smaller than the original SV?
        # are the split clusters bigger or smaller than the original SV?
        # calculate relative length ratio

        # Fraction of secondary clusters that are shorter than the dominant cluster.
        # Fraction that are longer.
        # Median and interquartile range of the length difference and length ratio.
        # Results stratified by two-cluster versus three-cluster regions.


def outlier_analysis():
    pass


def generate_genotyping_list():
    df = pd.DataFrame(columns=["id", "chr", "start", "end"])

    orig_svs = pd.read_csv("data/1kg/sv_lookup.csv")
    for i, row in orig_svs.iterrows():
        print(f"Processing row {i + 1}/{orig_svs.shape[0]}", end="\r")
        df.loc[len(df)] = [
            f"{row['id']}_0",
            row["chr"],
            row["start"],
            row["stop"],
        ]

    split_svs = pd.read_csv("output/results/most_common_split.csv")
    for i, row in split_svs.iterrows():
        print(f"Processing row {i + 1}/{split_svs.shape[0]}", end="\r")
        modes = ast.literal_eval(row["modes"])
        for i, mode in enumerate(modes):
            df.loc[len(df)] = [
                f"{row['id']}_{i + 1}",
                row["chr"],
                mode["start"],
                mode["end"],
            ]

    df.to_csv("output/results/original_and_split_svs.csv", index=False)


def genotyping_analysis():
    pass


if __name__ == "__main__":
    dir = "output/results"
    cfg = load_config()
    get_overlapping_clustered_svs(cfg)
