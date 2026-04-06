import os
import subprocess
import argparse
import pandas as pd
from query_sv import load_vcf, giggle_format
from helper import stix_output_to_df
from typing import Optional, Set, Tuple

"""
The purpose of this script is to pick a subset of structural variants (SVs) from a larger VCF file to be used for the calibration test.
Example usage: python3 pick_calibration_subset.py --sv_lookup final-vcf.unphased.vcf.gz --bedtools_path /Users/vili4418/sv/bedtools/bin/bedtools -n 192
"""

Q_VALS = [0.6, 0.7, 0.8, 0.9, 1.0]


def pick_pairs(
    df: pd.DataFrame, overlaps: pd.DataFrame, *, n_pairs: int, input_dir: str
) -> Set[Tuple]:
    """Pick n_pairs of SVs with >= 30% reciprocal overlap. Ensure no SV is used more than once."""
    seen = set()
    pairs = set()

    # pick a random row in overlaps
    while len(pairs) < n_pairs:
        row = overlaps.sample(n=1).iloc[0]
        sv1_id = row["sv1_id"]
        sv2_id = row["sv2_id"]

        # ensure we don't pick the same sv twice
        if sv1_id in seen or sv2_id in seen:
            continue
        seen.add(sv1_id)
        seen.add(sv2_id)
        pairs.add((sv1_id, sv2_id))

    out_df = pd.DataFrame(
        columns=[
            "sv1_id",
            "sv2_id",
            "chr",
            "sv1_start",
            "sv1_stop",
            "sv1_len",
            "sv2_start",
            "sv2_stop",
            "sv2_len",
            "r",
        ]
    )
    for sv1, sv2 in pairs:
        sv1_row = df[df["id"] == sv1].iloc[0]
        sv2_row = df[df["id"] == sv2].iloc[0]
        out_df.loc[len(out_df)] = [
            sv1,
            sv2,
            sv1_row["chr"],
            sv1_row["start"],
            sv1_row["stop"],
            sv1_row["svlen"],
            sv2_row["start"],
            sv2_row["stop"],
            sv2_row["svlen"],
            overlaps[(overlaps["sv1_id"] == sv1) & (overlaps["sv2_id"] == sv2)][
                "r"
            ].values[0],
        ]

    out_file = os.path.join(input_dir, "pairs.csv")
    out_df.to_csv(out_file, index=False)
    print(f"Wrote pairs of SVs for calibration test to {out_file}")
    return pairs


def pick_svs(
    df: pd.DataFrame, pairs: Set[Tuple], *, n_svs: int, input_dir: str
) -> pd.DataFrame:
    """Pick n_svs single SVs that are not in the pairs set."""
    seen = set([sv for pair in pairs for sv in pair])
    svs = set()
    while len(svs) < n_svs:
        row = df.sample(n=1).iloc[0]
        sv_id = row["id"]

        # don't pick svs already in pairs or already picked
        if sv_id in svs or sv_id in seen:
            continue
        svs.add(sv_id)

    out_df = pd.DataFrame(columns=["sv_id", "chr", "start", "stop", "svlen"])
    for sv in svs:
        sv_row = df[df["id"] == sv].iloc[0]
        out_df.loc[len(out_df)] = [
            sv,
            sv_row["chr"],
            sv_row["start"],
            sv_row["stop"],
            sv_row["svlen"],
        ]

    out_file = os.path.join(input_dir, "singles.csv")
    out_df.to_csv(out_file, index=False)
    print(f"Wrote single SVs for calibration test to {out_file}")

    return svs


def run_bedtools_intersect(
    df: pd.DataFrame, *, bedtools_path: str, input_dir: str, out_file: str
) -> pd.DataFrame:
    """Run bedtools intersect to find SVs with >= 30% reciprocal overlap."""
    # convert csv to bed file
    bed_file = os.path.join(input_dir, "deletions.bed")
    with open(bed_file, "w") as bed_f:
        for _, row in df.iterrows():
            bed_f.write(
                f"{row['chr']}\t{row['start']}\t{row['stop']}\t{row['id']}\n"
            )

    # run bedtools to get overlaps
    # keep overlaps with >= 30% reciprocal overlap
    intersect_file = os.path.join(input_dir, "intersect.bed")
    subprocess.run(
        [
            bedtools_path,
            "intersect",
            "-a",
            bed_file,
            "-b",
            bed_file,
            "-wao",
            "-f",
            "0.3",
            "-r",
        ],
        stdout=open(intersect_file, "w"),
        check=True,
    )

    # convert bed back to csv and remove self-overlaps
    overlaps_df = pd.DataFrame(columns=["sv1_id", "sv2_id", "r"])
    for line in open(intersect_file, "r"):
        fields = line.strip().split("\t")
        sv1_id = fields[3]
        sv2_id = fields[7]
        if sv1_id == sv2_id:
            continue

        overlap_len = int(fields[8])
        sv1_row = df[df["id"] == sv1_id].iloc[0]
        sv2_row = df[df["id"] == sv2_id].iloc[0]
        r = min(overlap_len / sv1_row["svlen"], overlap_len / sv2_row["svlen"])
        overlaps_df.loc[len(overlaps_df)] = [sv1_id, sv2_id, r]

    overlaps_df.to_csv(out_file, index=False)
    print(f"Wrote overlaps to {out_file}")
    return overlaps_df


def filter_svs(
    df: pd.DataFrame, input_dir: str, sample_ids_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Filters the initial list of deletions.
    1) Filters SVs based on samples with LR data available.
    2) Filters SVs with > 10 samples having the SV.
    """
    df_filtered = df.copy()

    if sample_ids_file is not None and os.path.exists(
        os.path.join(input_dir, sample_ids_file)
    ):
        sample_ids = set()
        with open(os.path.join(input_dir, sample_ids_file), "r") as f:
            for line in f:
                sample_ids.add(line.strip())
        all_cols = set(df_filtered.columns[11:-1])
        cols_to_drop = all_cols - sample_ids
        df_filtered.drop(columns=cols_to_drop, inplace=True)

    df_filtered["n_w_allele"] = 0
    for index, row in df_filtered.iterrows():
        n_w_allele = 0
        for col in df_filtered.columns[11:-2]:
            if row[col] != "(0, 0)":
                n_w_allele += 1
        df_filtered.at[index, "n_w_allele"] = int(n_w_allele)
    df_filtered = df_filtered[df_filtered["n_w_allele"] > 10]
    df_filtered.reset_index(drop=True, inplace=True)

    print(f"Number of SVs after filtering: {df_filtered.shape[0]}")
    df_filtered.to_csv(
        os.path.join(input_dir, "filtered_dels.csv"), index=False
    )
    return df_filtered


def subset_svs(
    *,
    input_dir: str,
    sv_lookup_file: str,
    filtered: bool,
    sample_ids_file: Optional[str],
    bedtools_path: str,
    n_svs: int,
) -> None:
    """
    Picks a subset of SVs for the calibration test. The subset consists of n_svs single SVs and n_svs pairs of SVs with >= 30% reciprocal overlap.
     - If the input SV lookup file is not already filtered, it filters the SVs based on samples with long read data available and number of samples with the SV.
     - It then runs bedtools intersect to find pairs of SVs with >= 30% reciprocal overlap.
     - Finally, it picks n_svs single SVs that are not in the pairs set and n_svs pairs of SVs for the calibration test and writes them to separate csv files.
     - It also writes a combined csv file with all the picked SVs and their coordinates.
    """
    # read the sv lookup file
    # if it is a vcf, load it as a dataframe
    if sv_lookup_file.endswith(".vcf") or sv_lookup_file.endswith(".vcf.gz"):
        df = load_vcf(input_dir, sv_lookup_file)
    else:
        df = pd.read_csv(os.path.join(input_dir, sv_lookup_file))

    # filter the dataset if not already filtered
    if not filtered:
        df = filter_svs(df, input_dir, sample_ids_file)

    # run bedtools to get SVs with >= 30% reciprocal overlap
    overlaps_file = os.path.join(input_dir, "overlaps.csv")
    if not os.path.exists(overlaps_file):
        overlaps = run_bedtools_intersect(
            df,
            bedtools_path=bedtools_path,
            input_dir=input_dir,
            out_file=overlaps_file,
        )
    else:
        overlaps = pd.read_csv(overlaps_file)

    pairs = pick_pairs(
        df, overlaps, n_pairs=int(n_svs / 2), input_dir=input_dir
    )
    svs = pick_svs(df, pairs, n_svs=int(n_svs / 2), input_dir=input_dir)

    subset_df = pd.DataFrame(columns=["chr", "start", "stop", "n_svs_actual"])
    for sv in svs:
        row = df[df["id"] == sv].iloc[0]
        subset_df.loc[len(subset_df)] = [
            row["chr"],
            row["start"],
            row["stop"],
            1,
        ]
    for sv1, sv2 in pairs:
        row1 = df[df["id"] == sv1].iloc[0]
        row2 = df[df["id"] == sv2].iloc[0]
        subset_df.loc[len(subset_df)] = [
            row1["chr"],
            # get outer bounds of the two SVs
            min(row1["start"], row2["start"]),
            max(row1["stop"], row2["stop"]),
            2,
        ]
    subset_path = os.path.join(input_dir, "sv_subset.csv")
    subset_df.to_csv(subset_path, index=False)
    print(f"Wrote subset of SVs to {subset_path}")


def get_samples_by_q(q: float):
    """
    Gets the samples that have STIX data available for each query range (q) value.
    Writes the results to a csv file with columns: sv_id, sample_ids_{q} for each q value.
    """
    filename = "calibration/samples_by_q.csv"
    if not os.path.exists(filename):
        deletions = pd.read_csv("calibration/deletions.csv")
        subset = pd.read_csv("calibration/sv_subset.csv")
        all_sample_ids = set()
        with open("calibration/lr_sample_ids.txt", "r") as f:
            for line in f:
                all_sample_ids.add(line.strip())

        deletions = pd.merge(
            deletions,
            subset[["chr", "start", "stop"]],
            on=["chr", "start", "stop"],
            how="right",
        )
        df = pd.DataFrame(
            columns=["sv_id", "nonref_samples"]
            + [f"sample_ids_q{q}" for q in Q_VALS]
        )
        for _, row in deletions.iterrows():
            sv_id = row["id"]
            start = giggle_format(str(row["chr"]), row["start"])
            stop = giggle_format(str(row["chr"]), row["stop"])
            sample_ids_by_q = {}

            # get reference samples
            reference_samples = [
                col for col in all_sample_ids if row[col] == "(0, 0)"
            ]
            nonref_samples = ",".join(
                list(all_sample_ids - set(reference_samples))
            )
            for q in Q_VALS:
                reads = stix_output_to_df(
                    f"calibration/stix_output_{q}/{start}_{stop}.txt"
                )
                reads = reads[reads["sample_id"].isin(all_sample_ids)]

                # filter out reference samples
                reads = reads[~reads["sample_id"].isin(reference_samples)]

                if reads.shape[0] == 0:
                    sample_ids_by_q[q] = ""
                    continue

                # filter out samples with 0 or 1 pairs of points
                sample_counts = reads["sample_id"].value_counts()
                valid_samples = sample_counts[sample_counts > 1].index.tolist()
                reads = reads[reads["sample_id"].isin(valid_samples)]
                sample_ids_by_q[q] = ",".join(
                    reads["sample_id"].unique().tolist()
                )
            df.loc[len(df)] = [sv_id, nonref_samples] + [
                sample_ids_by_q[q] for q in Q_VALS
            ]
        df.to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)

    subset_q = df[["sv_id", f"sample_ids_q{q}"]]
    subset_q = subset_q.rename(columns={f"sample_ids_q{q}": "sample_ids"})
    subset_q = subset_q.astype({"sample_ids": str})
    return subset_q


def recalculate_jaccard(geno_df: pd.DataFrame, q: float):
    """
    Recalculates the jaccard index for each SV using only the samples that are:
    1) non-ref in the SR callset,
    2) have long read data available, and
    3) have enough short read data (> 1 pair of points)
    """
    samples_by_q = get_samples_by_q(q)
    df = geno_df.copy()
    for i, row in df.iterrows():
        sv_row = samples_by_q[samples_by_q["sv_id"] == row["id"]]
        if sv_row.empty:  # SV not included in the subset
            continue

        sr_nonref_sample_str = sv_row["sample_ids"].values[0]
        if len(sr_nonref_sample_str) == 0:
            df.at[i, "jaccard"] = 0.0
            continue

        sr_nonref_samples = set(sr_nonref_sample_str.split(","))

        # nonref_samples are samples where the LR data matched the SV in the SR callset
        lr_nonref_samples = set(row["sr_lr_non_ref"].split(","))

        # only keep LR samples that are nonref AND have STIX data in the SR callset before recalculating the jaccard index
        to_remove = lr_nonref_samples - sr_nonref_samples
        lr_reduced_samples = lr_nonref_samples - to_remove

        # for logging purposes
        # prev_jaccard = len(
        #     sr_nonref_samples.intersection(lr_nonref_samples)
        # ) / len(sr_nonref_samples.union(lr_nonref_samples))

        jaccard = len(sr_nonref_samples.intersection(lr_reduced_samples)) / len(
            sr_nonref_samples.union(lr_reduced_samples)
        )
        # print(
        #     row["id"],
        #     f"jaccard {prev_jaccard} -> {jaccard}",
        #     f"removed {len(to_remove)} samples",
        # )
        df.at[i, "jaccard"] = jaccard
    return df


def subset_svs_from_jaccard(
    *,
    input_dir: str,
    sv_lookup_file: str,
    genotyping_results_file: str,
    n_svs: int,
    filter_lr_samples_only: bool = False,
    keep_svs_from: str | None = None,
    subset_per_q: bool = False,
):
    """
    Picks a subset of SVs for the calibration test based on Jaccard similarity of the genotyping results.

    genotyping_results_file is a csv file with mandatory columns: svid, jaccard

    """
    intermediate_df_path = "calibration/merged_sv_lookup.csv"
    if not os.path.exists(intermediate_df_path):
        # read the sv lookup file
        # if it is a vcf, load it as a dataframe
        if sv_lookup_file.endswith(".vcf") or sv_lookup_file.endswith(
            ".vcf.gz"
        ):
            df = load_vcf(input_dir, sv_lookup_file)
        else:
            df = pd.read_csv(os.path.join(input_dir, sv_lookup_file))

        # merge the file with the genotyping results file on sv id
        geno_df = pd.read_csv(
            os.path.join(input_dir, genotyping_results_file), delimiter="\t"
        )
        geno_df.rename(columns={"svid": "id"}, inplace=True)

        # filter out SVs with no matching samples between genotyping results and original SV
        geno_df = geno_df[geno_df["jaccard"] > 0]

        # merge dfs to get coordinates for the SVs in the genotyping results file
        df = geno_df.merge(
            df,
            on="id",
            how="left",
            suffixes=("", "_y"),
        )

        print(
            f"Number of SVs after merging with genotyping results: {df.shape[0]}"
        )

        # filter out SVs with <= 10 non-ref (from the SR genotype) samples in the LR set
        if filter_lr_samples_only:
            all_lr_samples = set(
                pd.read_csv("long_reads/long_read_samples.csv")[
                    "sample_id"
                ].tolist()
            )
            keep = set()
            for _, row in df.iterrows():
                non_ref_samples = []
                for sample_id in all_lr_samples:
                    is_nonref = row[sample_id] != "(0, 0)"
                    if is_nonref:
                        non_ref_samples.append(sample_id)
                if len(non_ref_samples) > 10:
                    keep.add(row["id"])
            df = df[df["id"].isin(keep)]
            print(
                f"Number of SVs after filtering for LR samples: {df.shape[0]}"
            )

        df = df[
            [
                "id",
                "chr",
                "start",
                "stop",
                "jaccard",
                "size_diff",
                "sr_lr_non_ref",
                "sr_non_ref",
            ]
        ]
        df.to_csv("calibration/merged_sv_lookup.csv", index=False)
    else:
        df = pd.read_csv(intermediate_df_path)

    subset_df = pd.DataFrame(columns=["chr", "start", "stop", "n_svs_actual"])

    # keep all SVs in the previous subset that still match the criteria to allow
    # us to reuse STIX data that had been previously downloaded
    rows_to_drop = set()
    if keep_svs_from is not None:
        prev_subset = pd.read_csv(os.path.join(input_dir, keep_svs_from))
        for _, row in prev_subset.iterrows():
            sv_row = df[
                (df["chr"] == row["chr"])
                & (df["start"] == row["start"])
                & (df["stop"] == row["stop"])
            ]

            # if a row exists, add it to the subset df and remove it from the main df so we don't pick it again
            if sv_row.shape[0] > 0:
                subset_df.loc[len(subset_df)] = [
                    row["chr"],
                    row["start"],
                    row["stop"],
                    row["n_svs_actual"],
                ]
                rows_to_drop.add(sv_row.index.values[0])
        print("Number of SVs kept from previous subset: ", subset_df.shape[0])

    unseen_svs = df[~df.index.isin(rows_to_drop)]
    # pick up to n_svs/2 SVs with jaccard index of 1 and n_svs/2 SVs with jaccard index < 1
    svs_one_mode = unseen_svs[unseen_svs["jaccard"] == 1]
    n_svs_one_mode = min(
        int(n_svs / 2) - subset_df[subset_df["n_svs_actual"] == 1].shape[0],
        svs_one_mode.shape[0],
    )

    svs_one_mode = svs_one_mode.sample(n=n_svs_one_mode, random_state=42)
    n_svs_multi_mode = (
        subset_df[subset_df["n_svs_actual"] == 1].shape[0]
        + n_svs_one_mode
        - subset_df[subset_df["n_svs_actual"] == 2].shape[0]
    )
    svs_multi_modes = unseen_svs[unseen_svs["jaccard"] < 1].sample(
        n=n_svs_multi_mode, random_state=42
    )

    print(
        f"Picking {n_svs_one_mode} SVs with jaccard index of 1 and {n_svs_multi_mode} SVs with jaccard index < 1"
    )

    # if subset_per_q is True, then we need to recalculate the jaccard index for
    # each SV, since different query regions can lead to more or less STIX data pulled
    # use the already-picked SVs as the subset from 1kg and add a q column to
    # keep track of the "actual" number of SVs per query region (based on
    # available data)
    if subset_per_q:
        selected = pd.concat([subset_df, svs_one_mode, svs_multi_modes])
        final_subset = pd.DataFrame(
            columns=["chr", "start", "stop", "n_svs_actual", "q"]
        )
        for q in Q_VALS:
            geno_sample_splits = recalculate_jaccard(df, q)
            for _, row in selected.iterrows():
                geno_row = geno_sample_splits[
                    (geno_sample_splits["chr"] == row["chr"])
                    & (geno_sample_splits["start"] == row["start"])
                    & (geno_sample_splits["stop"] == row["stop"])
                ]
                updated_jaccard = geno_row["jaccard"].values[0]
                n_svs_actual = 1
                if updated_jaccard > 0 and updated_jaccard < 1:
                    n_svs_actual = 2
                final_subset.loc[len(final_subset)] = [
                    row["chr"],
                    row["start"],
                    row["stop"],
                    n_svs_actual,
                    q,
                ]
    else:
        for sv_df, n_svs_actual in zip([svs_one_mode, svs_multi_modes], (1, 2)):
            for _, row in sv_df.iterrows():
                subset_df.loc[len(subset_df)] = [
                    row["chr"],
                    row["start"],
                    row["stop"],
                    n_svs_actual,
                ]
        final_subset = subset_df

    out_filename = (
        "sv_subset_sr_lr_nonref" if filter_lr_samples_only else "sv_subset"
    )
    subset_path = os.path.join(input_dir, f"{out_filename}.csv")
    final_subset = final_subset.astype(
        {"chr": int, "start": int, "stop": int, "n_svs_actual": int}
    )
    final_subset.to_csv(subset_path, index=False)
    print(f"Wrote subset of SVs to {subset_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pick a subset of structural variants from a VCF file to be used for the calibration test"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory for incoming data",
        default="calibration",
    )
    parser.add_argument(
        "--sv_lookup",
        type=str,
        help="VCF or CSV file with structural variants",
        default="deletions.csv",
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        help="Txt file containing sample IDs with long read data available",
        default="sample_ids.txt",
    )
    parser.add_argument(
        "-f",
        type=bool,
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--bedtools_path",
        type=str,
        help="Path to bedtools executable",
        default="bedtools",
    )
    parser.add_argument(
        "-n",
        type=int,
        help="Number of single SVs to pick for calibration test",
        default=1000,
    )

    args = parser.parse_args()
    subset_svs(
        input_dir=args.input_dir,
        sv_lookup_file=args.sv_lookup,
        filtered=args.f,
        sample_ids_file=args.sample_ids,
        bedtools_path=args.bedtools_path,
        n_svs=args.n,
    )


if __name__ == "__main__":
    # main()
    subset_svs_from_jaccard(
        input_dir="calibration",
        sv_lookup_file="deletions.csv",
        genotyping_results_file="1k_sr_sv_non_ref-1k_sr_lr_gt_non_ref.bed.gz",
        n_svs=1000,
        filter_lr_samples_only=True,
        keep_svs_from="sv_subset.csv",
        subset_per_q=True,
    )
