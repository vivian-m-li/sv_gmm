import os
import ast
import subprocess
import pandas as pd
import numpy as np
from collections import Counter
from gmm_types import Evidence, SVStat
from typing import List, Optional, Tuple

PROCESSED_SVS_DIR = "processed_svs"

"""Get commonly used dataframes"""


def get_deletions_df(stem: str = "1kgp"):
    """Returns a dataframe of all SVs and genotypes for each sample."""
    file = os.path.join(stem, "deletions.csv")
    return pd.read_csv(file, low_memory=False)


def get_sv_stats_df(stem: str = "1kgp"):
    """DEPRECATED: use get_sv_stats_collapsed_df instead. Used for pre-dirichlet analysis."""
    file = os.path.join(stem, "sv_stats.csv")
    return pd.read_csv(file)


def get_sv_stats_converge_df(stem: str = "1kgp"):
    """Returns a dataframe of all SVs and GMM results for each run of the SV."""
    file = os.path.join(stem, "sv_stats_converge.csv")
    return pd.read_csv(file, low_memory=False)


def get_sv_stats_collapsed_df(stem: str = "1kgp"):
    """Returns a dataframe of all SVs and GMM results after collapsing to consensus results."""
    file = os.path.join(stem, "sv_stats_collapsed.csv")
    return pd.read_csv(file)


def get_sample_ids(file_root: str = "1kgp"):
    """Read and return the sample ids from a file."""
    sample_ids = set()
    with open(f"{file_root}/sample_ids.txt", "r") as f:
        for line in f:
            sample_ids.add(line.strip())
    return sample_ids


def get_sv_lookup(stem: str = "1kgp"):
    """Get a dataframe mapping sv_id to chr, start, stop."""
    file = os.path.join(stem, "sv_lookup.csv")
    return pd.read_csv(file)


def get_sv_chr(sv_id: str):
    """Get the chromosome, start, and stop for a given sv_id."""
    df = get_sv_lookup()
    row = df[df["id"] == sv_id]
    chr, start, stop = (
        row["chr"].values[0],
        row["start"].values[0],
        row["stop"].values[0],
    )
    print(sv_id, chr, start, stop, flush=True)
    return chr, start, stop


def get_svlen(evidence_by_mode: List[List[Evidence]]) -> List[List[SVStat]]:
    """Calculates the mean length/start/stop for each mode of an SV."""
    all_stats = []
    for mode in evidence_by_mode:
        stats = []
        for evidence in mode:
            lengths = [
                np.mean(paired_end) - np.mean(paired_end)
                for paired_end in evidence.paired_ends
            ]
            stats.append(
                SVStat(
                    length=np.mean(lengths) - evidence.mean_insert_size,
                    start=np.mean(
                        [paired_end[0] for paired_end in evidence.paired_ends]
                    ),
                    end=np.mean(
                        [paired_end[1] for paired_end in evidence.paired_ends]
                    ),
                )
            )
        all_stats.append(stats)
    return all_stats


def remove_gatk_rows():
    """Removes synthetic tests run with GATK from each results file."""
    files = os.listdir("synthetic_data")
    for file in files:
        if not file.startswith("results") or not file.endswith(".csv"):
            continue
        df = pd.read_csv(f"synthetic_data/{file}")
        df = df[df["gmm_model"] != "gatk"]
        df.to_csv(f"synthetic_data/{file}", index=False)


def calc_af(n_homozygous, n_heterozygous, population_size):
    """Calculate allele frequency from number of homozygous and heterozygous individuals."""
    return ((n_homozygous * 2) + n_heterozygous) / (population_size * 2)


def stix_output_to_df(
    filename: str, *, write_empty_file: bool = False
) -> pd.DataFrame:
    """
    Parses the raw stix output into a dataframe.
    Be aware that file_id is not a unique identifier if there are multiple shards for an index.
    """
    column_names = [
        "file_id",
        "sample_id",
        "l_chr",
        "l_start",
        "l_end",
        "r_chr",
        "r_start",
        "r_end",
        "type",
    ]
    # check if file is empty
    if write_empty_file or os.stat(filename).st_size == 0:
        return pd.DataFrame(columns=column_names)

    df = pd.read_csv(filename, names=column_names, sep=r"\s+")
    df["sample_id"] = df["sample_id"].str.extract(
        r".*([A-Z]{2}\d{5}).*", expand=False
    )
    return df


def df_to_bed(
    *,
    out_file: str,
    in_file: Optional[str] = None,
    in_df: Optional[pd.DataFrame] = None,
):
    """Converts a csv to a bed file to be used with bedtools. Requires the dataframe to have columns: chr, start, end, sv_id, id."""
    assert in_file or in_df is not None, "Must provide either in_file or in_df"
    df = in_df if in_df is not None else pd.read_csv(in_file)
    with open(out_file, "w") as outfile:
        for _, row in df.iterrows():
            outfile.write(
                f"{row['chr']}\t{row['start']}\t{row['end']}\t{row['sv_id']}\t{row['id']}\n"
            )


def find_missing_sample_ids():
    """Gets sample ids that are in the STIX database but are not in the original deletions VCF file"""
    sample_ids = set()
    for file in os.listdir("stix_output"):
        df = stix_output_to_df(f"stix_output/{file}")
        sample_ids = sample_ids | set(df["sample_id"].tolist())
    deletions_df = get_deletions_df()
    missing = sample_ids - set(deletions_df.columns[11:-1])

    return missing


def find_missing_processed_svs():
    """
    DEPRECATED: new processed_svs_dir is processed_svs_converge
    Gets SVs that are in the original deletions VCF that have not yet been run through the pipeline.
    """
    processed_sv_ids = set(
        [file.strip(".csv") for file in os.listdir(PROCESSED_SVS_DIR)]
    )
    deletions_df = get_deletions_df()
    missing = set(deletions_df["id"]) - processed_sv_ids
    return missing


def get_num_intersecting_genes():
    """Gets number of genes that intersect with at least 1 SV, and number of SVs that each gene intersects with."""
    df = pd.read_csv(
        "low_cov_grch37/intersect_num_overlap.csv", header=None, delimiter="\t"
    )
    num_intersections = df.iloc[:, 5]
    print(f"Total number of genes: {len(num_intersections)}")

    # Number of SV that each gene intersects with
    # 23.36% of genes intersect with at least 1 SV
    num_intersections_filtered = num_intersections[num_intersections > 0]
    print(num_intersections_filtered.describe())


def get_num_new_svs():
    """Gets number of new SVs created by splitting multi-modal SVs into multiple single-modal SVs."""
    df = get_sv_stats_df()
    df = df[df["num_samples"] > 0]
    mode_data = [df[df["num_modes"] == i + 1] for i in range(3)]
    num_two_modes = len(mode_data[1])
    num_three_modes = len(mode_data[2])
    print(f"Number of new SVs: {num_two_modes + (num_three_modes * 2)}")


def get_sample_sequencing_centers():
    """Get a dataframe mapping sample_id to sequencing center(s)."""
    # data obtained from https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/20130502.phase3.sequence.index
    df = pd.read_csv(
        "1kgp/20130502.phase3.sequence.index", sep="\t", low_memory=False
    )
    df["CENTER_NAME"] = df["CENTER_NAME"].str.upper()
    df = df[["SAMPLE_NAME", "CENTER_NAME"]].drop_duplicates()
    df = df.groupby("SAMPLE_NAME")["CENTER_NAME"].apply(list).reset_index()
    return df


"""Handle varying insert sizes"""


def get_insert_sizes(get_files: bool = False):
    """Get mean insert sizes for all samples in 1kgp. If get_files is True, will download mapped files from BAS files."""
    if get_files:
        samples_df = pd.read_csv("1kgp/bam_bas_files.tsv", sep="\t")
        pattern = r"data\/.*\/alignment\/.*\.mapped\.ILLUMINA\.bwa\..+\.low_coverage\.\d+\.bam\.bas"
        bas_files_df = samples_df[samples_df["BAS FILE"].str.fullmatch(pattern)]
        bas_files_df["sample_id"] = bas_files_df["BAS FILE"].str.extract(
            r"\/(.+)\/alignment"
        )

        for _, row in bas_files_df.iterrows():
            bas_file = row["BAS FILE"]
            sample_id = row["sample_id"]
            if f"{sample_id}.tsv" in os.listdir("1kgp/mapped_files"):
                continue
            subprocess.run(
                ["bash", "get_mapped_files.sh"] + [sample_id, bas_file],
                capture_output=True,
                text=True,
            )

    df = pd.DataFrame(columns=["sample_id", "mean_insert_size"])
    mapped_files = os.listdir("1kgp/mapped_files")
    for i, file in enumerate(mapped_files):
        sample_id = file.strip(".tsv")
        try:
            file_df = pd.read_csv(f"1kgp/mapped_files/{file}", sep="\t")
        except Exception:
            # alignment files don't exist for samples HG00361, HG00844, NA18555.tsv
            print(file)
        mean_insert_size = int(np.mean(file_df["mean_insert_size"]))
        df.loc[i] = [sample_id, mean_insert_size]
    df.to_csv("1kgp/insert_sizes.csv", index=False)


def get_mean_coverage():
    """Get mean coverage for all samples in 1kgp."""
    df = get_sv_stats_df()
    df = df[df["num_modes"] > 1]
    coverage_df = pd.DataFrame(columns=["sv_id", "num_samples", "coverage"])
    for _, sv in df.iterrows():
        modes = ast.literal_eval(sv["modes"])
        chr = sv["chr"]
        L = sv["start"]
        R = sv["stop"]
        reads = stix_output_to_df(f"stix_output/{chr}:{L}_{chr}:{R}.txt")
        for mode in modes:
            coverage = [
                reads[reads["sample_id"] == sample_id].shape[0]
                for sample_id in mode["sample_ids"]
            ]

            coverage_df.loc[len(coverage_df)] = [
                sv["id"],
                mode["num_samples"],
                coverage,
            ]

    coverage_df.to_csv("1kgp/coverage.csv", index=False)


def get_insert_size_diff():
    """Get the mean difference between insert sizes obtained from mapped files and insert sizes scraped from metadata."""
    df = pd.read_csv(
        "1kgp/insert_sizes_scraped.csv",
        dtype={"sample_id": str, "mean_insert_size": int},
    )
    df2 = pd.read_csv(
        "1kgp/insert_sizes.csv",
        dtype={"sample_id": str, "mean_insert_size": float},
    )

    df2 = df2.rename(columns={"mean_insert_size": "true_mean_insert_size"})
    # join df2 to df
    df = df.merge(
        df2,
        on="sample_id",
        how="inner",
    )
    df["diff"] = abs(df["mean_insert_size"] - df["true_mean_insert_size"])
    mean_diff = df["diff"].mean()
    print(mean_diff)
    df.to_csv("1kgp/insert_size_compare.csv", index=False)


"""Dirichlet helpers"""


def calculate_posteriors(alpha):
    """Calculate the posterior probabilities and variances for each mode given alpha parameters."""
    p = alpha / np.sum(alpha)
    sum_alpha_post = np.sum(alpha)
    var = (alpha * (sum_alpha_post - alpha)) / (
        sum_alpha_post**2 * (sum_alpha_post + 1)
    )
    return p, var


def calculate_ci(p, var, n):
    """Calculate the 95% confidence interval for the difference in means between the two most probable modes."""
    # Calculate the difference in means between the two most probable modes
    posterior_mu_sorted_indices = np.argsort(p)
    posterior_mu_sorted = p[posterior_mu_sorted_indices]
    diff_in_means = posterior_mu_sorted[-1] - posterior_mu_sorted[-2]

    # Calculate the confidence interval for our difference in means
    diff_var = (var[posterior_mu_sorted_indices[-1]]) / n + (
        var[[posterior_mu_sorted_indices[-2]]]
    ) / n
    confidence = 1.96 * np.sqrt(diff_var)
    ci = [diff_in_means - confidence, diff_in_means + confidence]

    return ci


def calculate_posteriors_from_trials(outcomes):
    """Calculate the posterior probabilities and variances for each mode given a list of outcomes."""
    counts = Counter(outcomes)
    alpha = np.array([1, 1, 1]) + np.array([counts[1], counts[2], counts[3]])
    return calculate_posteriors(alpha)


def reciprocal_overlap(sv1: Tuple[int, int], sv2: Tuple[int, int]) -> float:
    """
    Calculates the reciprocal overlap between two structural variants.
    r = min(% overlap sv 1, % overlap sv 2)
    """
    start1, end1 = sv1
    start2, end2 = sv2
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_start >= overlap_end:
        return 0.0
    overlap_length = overlap_end - overlap_start
    sv1_length = end1 - start1
    sv2_length = end2 - start2
    return min(overlap_length / sv1_length, overlap_length / sv2_length)
