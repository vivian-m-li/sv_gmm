import os
import argparse
import pandas as pd
from query_sv import load_vcf
from typing import Optional, Set, Tuple

"""The purpose of this script is to pick a subset of structural variants (SVs) from a larger VCF file to be used for the calibration test."""


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
        sv1_row = df[df["sv_id"] == sv1].iloc[0]
        sv2_row = df[df["sv_id"] == sv2].iloc[0]
        out_df[len(out_df)] = [
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
        sv_id = row["sv_id"]

        # don't pick svs already in pairs or already picked
        if sv_id in svs or sv_id in seen:
            continue
        svs.add(sv_id)

    out_df = pd.DataFrame(columns=["sv_id", "chr", "start", "stop", "svlen"])
    for sv in svs:
        sv_row = df[df["sv_id"] == sv].iloc[0]
        out_df[len(out_df)] = [
            sv,
            sv_row["chr"],
            sv_row["start"],
            sv_row["stop"],
            sv_row["svlen"],
        ]
    out_file = os.path.join(input_dir, "singles.csv")
    out_df.to_csv(out_file, index=False)
    print(f"Wrote single SVs for calibration test to {out_file}")
    pass


def filter_svs(
    df: pd.DataFrame, input_dir: str, sample_ids_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Filters the initial list of deletions.
    1) Filters SVs based on samples with LR data available.
    2) Filters SVs with > 10 samples having the SV.
    """
    df_filtered = df.copy()

    if sample_ids_file is not None:
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
    samtools_path: str,
    n_svs: int,
):
    # read the sv lookup file
    # if it is a vcf, load it as a dataframe
    if sv_lookup_file.endswith(".vcf") or sv_lookup_file.endswith(".vcf.gz"):
        df = load_vcf(input_dir, sv_lookup_file)
    else:
        df = pd.read_csv(os.path.join(input_dir, sv_lookup_file))

    # filter the dataset if not already filtered
    if not filtered:
        df = filter_svs(df, input_dir, sample_ids_file)

    # run samtools to get SVs with >= 30% reciprocal overlap
    if not os.path.exists(os.path.join(input_dir, "overlaps.csv")):
        # run samtools to generate this file
        # might need to convert csv back into vcf first
        # overlapping SVs will have cols sv1_id, sv2_id, reciprocal_overlap
        overlaps = ""
        pass
    else:
        overlaps = pd.read_csv(os.path.join(input_dir, "overlaps.csv"))

    pairs = pick_pairs(
        df, overlaps, n_pairs=int(n_svs / 2), input_dir=input_dir
    )
    svs = pick_svs(df, pairs, n_svs=int(n_svs / 2), input_dir=input_dir)

    # write the subset of SVs to a new file
    # dataframe with cols chr, start, stop, n_svs_actual
    print("Wrote subset of SVs to subset_svs.csv")


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
        default=None,
    )
    parser.add_argument(
        "-f",
        type=bool,
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--samtools_path",
        type=str,
        help="Path to samtools executable",
        default="samtools",
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
        samtools_path=args.samtools_path,
        n_svs=args.n,
    )


if __name__ == "__main__":
    main()
