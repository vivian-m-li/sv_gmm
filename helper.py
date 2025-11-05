import os
import re
import ast
import subprocess
import pandas as pd
import numpy as np
import csv
from scipy.spatial.distance import braycurtis
from collections import defaultdict, Counter
from gmm_types import Evidence, SVStat, SVInfoGMM, SUPERPOPULATIONS
from dataclasses import fields
from typing import List, Optional, Tuple

PROCESSED_STIX_DIR = "processed_stix_output"
PROCESSED_SVS_DIR = "processed_svs"

"""Get commonly used dataframes"""


def get_deletions_df(stem: str = "1kgp"):
    """Returns a dataframe of all SVs and genotypes for each sample."""
    return pd.read_csv(f"{stem}/deletions_df.csv", low_memory=False)


def get_sv_stats_df(stem: str = "1kgp"):
    """DEPRECATED: use get_sv_stats_collapsed_df instead. Used for pre-dirichlet analysis."""
    return pd.read_csv(f"{stem}/sv_stats.csv")


def get_sv_stats_converge_df(stem: str = "1kgp"):
    """Returns a dataframe of all SVs and GMM results for each run of the SV."""
    return pd.read_csv(f"{stem}/sv_stats_converge.csv", low_memory=False)


def get_sv_stats_collapsed_df(stem: str = "1kgp"):
    """Returns a dataframe of all SVs and GMM results after collapsing to consensus results."""
    return pd.read_csv(f"{stem}/sv_stats_collapsed.csv")


def get_sample_ids(file_root: str = "1kgp"):
    """Read and return the sample ids from a file."""
    sample_ids = set()
    with open(f"{file_root}/sample_ids.txt", "r") as f:
        for line in f:
            sample_ids.add(line.strip())
    return sample_ids


def get_sv_lookup(stem: str = "1kgp"):
    """Get a dataframe mapping sv_id to chr, start, stop."""
    return pd.read_csv(f"{stem}/sv_lookup.csv")


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
    """Gets sample ids that are in the STIX databse but are not in the original deletions VCF file"""
    sample_ids = set()
    for file in os.listdir(PROCESSED_STIX_DIR):
        with open(f"{PROCESSED_STIX_DIR}/{file}") as f:
            for line in f:
                sample_id = line.strip().split(",")[0]
                if sample_id[0].isalpha():
                    sample_ids.add(sample_id)

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


def extract_data_from_deletions_df():
    """Extracts sample ids and splits deletions into separate files by chromosome. Makes SV lookup more efficient if the SV chromosome is known."""
    deletions_df = get_deletions_df()

    sample_ids = set(deletions_df.columns[11:-1])
    with open("1kgp/sample_ids.txt", "w") as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")

    # split the deletions into separate files by chromosome
    # another option is to create a lookup for row index in deletions_df by chr, start, stop, sample_id and
    os.mkdir("1kgp/deletions_by_chr")
    for i in range(1, 23):
        chr_df = deletions_df[deletions_df["chr"] == i]
        chr_df.to_csv(f"1kgp/deletions_by_chr/chr{i}.csv", index=False)


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
        file = f"{PROCESSED_STIX_DIR}/{chr}:{L}-{L}_{chr}:{R}-{R}.csv"
        squiggle_data = {}
        with open(file, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                squiggle_data[row[0]] = np.array([float(x) for x in row[1:]])
        for mode in modes:
            coverage = [
                int(len(squiggle_data[sample_id]) / 2)
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


"""
Handle errors that may arise when running the pipeline on all SVs
"""


def get_unprocessed_svs():
    """Get SVs that have not yet been processed."""
    df = get_deletions_df()
    svs = set(
        [
            (row["chr"].lower(), str(row["start"]), str(row["stop"]))
            for _, row in df.iterrows()
        ]
    )

    processed_files = os.listdir("processed_stix_output")
    pattern = r"([\w]+):(\d+)-\d+_[\w]+:(\d+)-\d+.csv"
    processed_svs = set(
        [re.search(pattern, file).groups for file in processed_files]
    )

    unprocessed_svs = svs - processed_svs
    with open("1kgp/unprocessed_svs.txt", "w") as f:
        for chr, start, stop in unprocessed_svs:
            f.write(f"{chr.upper()},{start},{stop}\n")


def get_med_low_confidence_svs():
    """Get SVs that have medium or low confidence."""
    df = get_sv_stats_converge_df()
    grouped = df.groupby("id")
    # get how many times we saw each id in df
    counts = grouped.size()
    counts.sort_values(ascending=False, inplace=True)
    with open("1kgp/med_low_confidence_svs.txt", "w") as f:
        for sv_id in counts[counts > 8].index:
            f.write(f"{sv_id}\n")


"""
Analyze SVs that have been run
"""


def write_sv_stats_collapsed(stem: str = "1kgp"):
    """Collapse sv_stats_converge.csv to sv_stats_collapsed.csv by picking the most common result for each SV."""
    df = get_sv_stats_converge_df(stem)
    svs_n_modes = pd.read_csv(f"{stem}/svs_n_modes.csv")
    svs_n_modes.rename(
        columns={"num_modes": "consensus_num_modes"}, inplace=True
    )
    df = df.merge(svs_n_modes, left_on="id", right_on="sv_id")
    df.loc[df["num_modes"] == 0, "num_modes"] = (
        1  # set num_modes = 1 where it is 0
    )
    sv_ids = df["id"].unique()
    with open(f"{stem}/sv_stats_collapsed.csv", mode="w", newline="") as out:
        fieldnames = [field.name for field in fields(SVInfoGMM)]
        csv_writer = csv.DictWriter(out, fieldnames=fieldnames)
        csv_writer.writeheader()
        for sv_id in sv_ids:
            rows = df[df["id"] == sv_id]
            if len(rows) < 1:
                continue

            consensus_num_modes = rows["consensus_num_modes"].values[0]
            rows = rows[rows["num_modes"] == consensus_num_modes]
            samples = Counter()
            samples_by_row = {}
            for i, row in rows.iterrows():
                modes = sorted(
                    ast.literal_eval(row["modes"]), key=lambda x: x["start"]
                )
                sample_ids = [
                    ",".join(sorted(mode["sample_ids"])) for mode in modes
                ]
                samples_joined = ";".join(sample_ids)
                samples.update([samples_joined])
                samples_by_row[samples_joined] = i

            most_common = samples.most_common(1)[0][0]
            row = rows.loc[samples_by_row[most_common]].copy()
            row = row.drop(
                ["consensus_num_modes", "num_modes_2", "sv_id", "confidence"]
            )
            csv_writer.writerow(row.to_dict())


def write_ancestry_dissimilarity(stem: str = "1kgp"):
    """Calculate the dissimilarity in ancestry between modes for each SV."""
    df = get_sv_stats_collapsed_df()
    confidence = pd.read_csv(f"{stem}/svs_n_modes.csv")
    confidence.rename(
        columns={"num_modes": "consensus_num_modes"}, inplace=True
    )
    df = df.merge(
        confidence,
        left_on="id",
        right_on="sv_id",
    )
    df = df[(df["confidence"] != "low") & (df["consensus_num_modes"] > 1)]
    ancestry_df = pd.read_csv("1kgp/ancestry.tsv", delimiter="\t")

    results_df = pd.DataFrame(
        columns=["chr", "start", "stop", "num_samples", "dissimilarity"]
    )
    for i, row in df.iterrows():
        ancestry = []
        modes = ast.literal_eval(row["modes"])
        for mode in modes:
            sample_ids = mode["sample_ids"]
            ancestry_counter = {anc: 0 for anc in SUPERPOPULATIONS}
            for sample_id in sample_ids:
                ancestry_row = ancestry_df[
                    ancestry_df["Sample name"] == sample_id
                ]
                superpopulation = (
                    ancestry_row["Superpopulation code"].values[0].split(",")[0]
                )
                ancestry_counter[superpopulation] += 1
            ancestry_counter = {
                k: v / len(sample_ids) for k, v in ancestry_counter.items()
            }
            ancestry.append([v for v in ancestry_counter.values()])

        dissimilarities = []
        for i in range(row["num_modes"] - 1):
            for j in range(i + 1, row["num_modes"]):
                dissimilarity = braycurtis(ancestry[i], ancestry[j])
                dissimilarities.append(dissimilarity)

        results_df.loc[len(results_df)] = [
            row["chr"],
            row["start"],
            row["stop"],
            row["num_samples"],
            np.mean(dissimilarities),
        ]

    # explicitly cast columns to correct types - everything was converting to float because of dissimilarity
    results_df["chr"] = results_df["chr"].astype(int)
    results_df["start"] = results_df["start"].astype(int)
    results_df["stop"] = results_df["stop"].astype(int)
    results_df["num_samples"] = results_df["num_samples"].astype(int)

    results_df = results_df.sort_values(by="dissimilarity", ascending=False)
    results_df.to_csv(f"{stem}/ancestry_dissimilarity.csv", index=False)


def get_n_modes(stem: str = "1kgp"):
    """Get the number of modes and confidence for each SV."""
    sv_df = pd.DataFrame(
        columns=["sv_id", "num_modes", "confidence", "num_modes_2"]
    )

    deletions_df = get_deletions_df()
    deletions_df = deletions_df[deletions_df["num_samples"] > 0]
    svs = deletions_df["id"].unique()
    df = get_sv_stats_converge_df(stem)
    for sv_id in svs:
        rows = df[df["id"] == sv_id]

        # if the sv_id didn't run (due to lack of evidence in long read analysis), skip the row
        if len(rows) == 0:
            continue

        # if the GMM didn't run at all on a sample
        # or GMM defaulted to 1 mode because there were too few samples
        # label it as 1 mode inconclusively
        if (len(rows) == 1 and rows["num_iterations"].values[0] == 0) or rows[
            "num_samples"
        ].values[0] < 10:
            sv_df.loc[len(sv_df)] = [sv_id, 1, "inconclusive", np.nan]
            continue

        outcomes = rows["num_modes"].values
        counter = Counter(outcomes)
        most_common = counter.most_common(2)
        num_modes = max(1, most_common[0][0])
        num_modes_2 = int(most_common[1][0]) if len(counter) > 1 else np.nan

        p, var = calculate_posteriors_from_trials(outcomes)
        ci = calculate_ci(p, var, len(outcomes))
        new_row = [sv_id, num_modes]
        if ci[0] >= 0.6:
            new_row.append("high")
        elif ci[0] >= 0.3 and ci[1] < 0.6:
            new_row.append("medium")
        else:
            new_row.append("low")
        new_row.append(num_modes_2)

        sv_df.loc[len(sv_df)] = new_row
    sv_df.to_csv(f"{stem}/svs_n_modes.csv", index=False)


def get_sv_outliers(sv_rows, threshold: float):
    """Get outlier samples for a given SV."""
    n = len(sv_rows)
    outlier_counts = defaultdict(lambda: 0)
    for i, row in sv_rows.iterrows():
        modes = ast.literal_eval(row["modes"])
        for mode in modes:
            if mode["num_samples"] == 1:
                outlier_sample = mode["sample_ids"][0]
                outlier_counts[outlier_sample] += 1

    confident_outliers = []
    for outlier, count in outlier_counts.items():
        # this threshold is important for determining when we can confidently say something is an outlier
        if count / n > threshold:
            confident_outliers.append(outlier)

    return confident_outliers


def get_outliers(stem: str = "1kgp", threshold: float = 0.9):
    """Get outlier samples for each SV and write to a file."""
    n_modes_df = pd.read_csv(f"{stem}/svs_n_modes.csv")
    n_modes_df = n_modes_df[n_modes_df["num_modes"] > 1]
    df = get_sv_stats_converge_df()
    with open(f"{stem}/outliers.csv", "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["sv_id", "sample_ids"])
        for _, row in n_modes_df.iterrows():
            sv_rows = df[df["id"] == row["sv_id"]]
            outliers = get_sv_outliers(sv_rows, threshold)
            if len(outliers) > 0:
                csv_writer.writerow([row["sv_id"], ",".join(outliers)])


def get_consensus_svs(stem: str = "1kgp"):
    """Get consensus SVs by averaging the start/stop/length of each mode across all runs of the SV."""
    df = get_sv_stats_converge_df()
    svs_n_modes = pd.read_csv(f"{stem}/svs_n_modes.csv")
    sv_ids = svs_n_modes["sv_id"].unique()
    consensus_df = pd.DataFrame(
        columns=[
            "id",
            "sv_id",
            "chr",
            "start",
            "start_sd",
            "end",
            "end_sd",
            "length",
            "length_sd",
            "num_samples",
            "num_samples_std",
        ]
    )
    for sv_id in sv_ids:
        sv_rows = df[df["id"] == sv_id]
        chr = sv_rows["chr"].values[0]
        modes_count = Counter(sv_rows["num_modes"])
        num_modes = max(modes_count, key=modes_count.get)
        sv_rows = sv_rows[sv_rows["num_modes"] == num_modes]
        all_mode_stats = [[] for _ in range(num_modes)]
        for _, row in sv_rows.iterrows():
            modes = ast.literal_eval(row["modes"])
            modes = sorted(modes, key=lambda x: x["start"])
            for i, mode in enumerate(modes):
                all_mode_stats[i].append(
                    {
                        "length": mode["length"],
                        "start": mode["start"],
                        "end": mode["end"],
                        "num_samples": mode["num_samples"],
                    }
                )

        # write mean and sd start/stop/length for each mode
        for i, mode in enumerate(all_mode_stats):
            row = [f"{sv_id}_{i + 1}", sv_id, chr]
            for key in ["start", "end", "length", "num_samples"]:
                values = [mode_stat[key] for mode_stat in mode]
                row.append(int(np.mean(values)))
                row.append(np.std(values))
            consensus_df.loc[len(consensus_df)] = row
    consensus_df.to_csv(f"{stem}/consensus_svs.csv", index=False)


def get_new_gene_intersections():
    """Get new gene intersections created by splitting multi-modal SVs into multiple single-modal SVs."""
    og_intersections = set()  # (sv_id, gene_id)
    with open("1kgp/original_gene_intersections.bed", "r") as f:
        for line in f:
            row = line.strip().split("\t")
            sv_id, gene_id = row[3], row[7]
            og_intersections.add((sv_id, gene_id))

    df_to_bed(
        in_file="1kgp/consensus_svs.csv", out_file="1kgp/consensus_svs.bed"
    )
    subprocess.run(
        ["bash", "bed_intersect.sh"]
        + [  # noqa503
            "1kgp/consensus_svs.bed",
            "1kgp/genes.bed",
            "1kgp/consensus_gene_intersections.bed",
        ],
        capture_output=True,
        text=True,
    )

    split_intersections = set()
    sv_split_lookup = {}
    with open("1kgp/consensus_gene_intersections.bed", "r") as f:
        for line in f:
            row = line.strip().split("\t")
            sv_id, sv_split_id, gene_id = row[3], row[4], row[8]
            split_intersections.add((sv_id, gene_id))
            sv_split_lookup[(sv_id, gene_id)] = sv_split_id

    new_intersections = split_intersections - og_intersections
    with open("1kgp/new_gene_intersections.bed", "w") as f:
        for sv_id, gene_id in new_intersections:
            sv_split_id = sv_split_lookup[(sv_id, gene_id)]
            f.write(f"{sv_split_id}\t{gene_id}\n")


def outlier_gene_intersections():
    """Get SVs that are both outliers and have new gene intersections."""
    svs_with_new_genes = set()
    with open("1kgp/new_gene_intersections.bed", "r") as f:
        for line in f:
            sv_id = line.strip().split()[0]
            sv_id = "_".join(sv_id.split("_")[0:2])
            svs_with_new_genes.add(sv_id)

    svs_with_outliers = set()
    with open("1kgp/outliers.txt", "r") as f:
        for line in f:
            sv_id = line.strip().split()[0]
            svs_with_outliers.add(sv_id)

    print(svs_with_outliers.intersection(svs_with_new_genes))


def high_confidence_gene_intersections():
    """Get SVs that are both high confidence and have new gene intersections."""
    svs_with_new_genes = set()
    with open("1kgp/new_gene_intersections.bed", "r") as f:
        for line in f:
            sv_id = line.strip().split()[0]
            sv_id = "_".join(sv_id.split("_")[0:2])
            svs_with_new_genes.add(sv_id)

    df = pd.read_csv("1kgp/svs_n_modes.csv")
    high_confidence_svs = set(
        df[(df["confidence"] == "high") & (df["num_modes"] >= 1)]["sv_id"]
    )
    print("n high confidence SVs:", len(high_confidence_svs))
    print(
        "n high confidence SVs with new gene intersections:",
        len(high_confidence_svs.intersection(svs_with_new_genes)),
    )


def recalculate_afs():
    """Recalculate allele frequencies based on genotypes in deletions_df and compare to allele frequencies in sv_stats_collapsed_df."""
    df = get_deletions_df().head(1000)
    results_df = get_sv_stats_collapsed_df()
    sample_ids = get_sample_ids()
    for _, row in df.iterrows():
        gt_samples = set()
        for sample_id in sample_ids:
            gt = ast.literal_eval(row[sample_id])
            gt = tuple([0 if g is None else g for g in gt])  # convert None to 0
            gt_sum = sum(gt)
            if gt_sum > 0:
                gt_samples.add(sample_id)

        results_row = results_df[results_df["id"] == row["id"]]
        if results_row.empty and len(gt_samples) == 0:
            continue

        model_samples = set()
        if not results_row.empty:
            modes = ast.literal_eval(results_row["modes"])
            if len(modes) == 0:
                continue
            for mode in modes:
                for sample_id in mode["sample_ids"]:
                    model_samples.add(sample_id)

        print(
            f"SV {row['id']}: n samples with alleles={len(gt_samples)}, n samples through model={len(model_samples)}"
        )


def compare_short_long_reads():
    """Compare the number of modes and confidence between short read and long read analyses."""
    sr_df = pd.read_csv("1kgp/svs_n_modes.csv")
    sr_df = sr_df[sr_df["confidence"] != "inconclusive"]
    sr_df = sr_df.rename(
        columns={
            "num_modes": "sr_num_modes",
            "confidence": "sr_confidence",
            "num_modes_2": "sr_num_modes_2",
        }
    )

    lr_df = pd.read_csv("long_reads/svs_n_modes.csv")
    lr_df = lr_df[lr_df["confidence"] != "inconclusive"]
    lr_df = lr_df.rename(
        columns={
            "num_modes": "lr_num_modes",
            "confidence": "lr_confidence",
            "num_modes_2": "lr_num_modes_2",
        }
    )

    merged = lr_df.merge(sr_df, on="sv_id", how="inner")
    merged["consensus"] = merged.apply(
        lambda row: (
            row["lr_num_modes"]
            if row["lr_num_modes"] == row["sr_num_modes"]
            else np.nan
        ),
        axis=1,
    )
    # consensus_df = merged[~merged["consensus"].isna()] # get all rows where the consensus is not NaN
    merged.to_csv("1kgp/sr_lr_merged.csv", index=False)


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


def get_overlapping_clustered_svs():
    """Checks for overlaps between clustered SVs and the original SV breakpoints."""

    if not os.path.exists("1kgp/sv_lookup_intersect.csv"):
        lookup = get_sv_lookup()
        lookup["sv_id"] = lookup["id"].astype(str)
        lookup.rename(columns={"stop": "end"}, inplace=True)
        bed_file = "1kgp/sv_lookup.bed"
        df_to_bed(in_df=lookup, out_file=bed_file)

        output_bed_file = "1kgp/sv_lookup_intersect.bed"
        subprocess.run(
            ["bash", "bed_intersect.sh"]
            + [
                bed_file,
                "1kgp/consensus_svs.bed",
                output_bed_file,
            ],
            capture_output=True,
            text=True,
        )

        df = bed_to_df(output_bed_file)
        df.to_csv("1kgp/sv_lookup_intersect.csv", index=False)
    else:
        df = pd.read_csv("1kgp/sv_lookup_intersect.csv")

    overlaps = df[(df["r"] >= 0.25) & (df["sv1_id"] != df["sv2_id"])]

    # this gets all overlapping SVs - however, we want to check whether the original SVs were overlapping as well (pre-clustering)
    # print(
    #     overlaps[["sv1_id", "sv1_l", "sv1_r", "sv2_id", "sv2_l", "sv2_r", "r"]]
    # )

    og_overlaps = pd.read_csv("1kgp/og_svs_intersect.csv")
    og_overlaps = og_overlaps[og_overlaps["r"] >= 0.25]
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

    # TODO: sv2 coordinates do not take into account the read length (i.e. SV is 450 bp longer than reference)
    print(new_overlaps)


def get_bam_files(sv_id: str):
    """Get BAM files for all samples that have the given SV."""
    # get samples with the given sv_id
    df = pd.read_csv("long_reads/sample_sv_lookup.csv")
    df = df[df["sv_id"] == sv_id]
    long_read_samples = pd.read_csv("long_reads/long_read_samples.csv")
    df = df.merge(long_read_samples, on="sample_id")

    # get bam files
    file_dir = f"long_reads/bam_files/{sv_id}"
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    chr, start, stop = get_sv_chr(sv_id)
    region = f"chr{chr}:{start}-{stop}"
    for _, row in df.iterrows():
        output_file = f"{file_dir}/{row['sample_id']}.bam"
        subprocess.run(
            ["bash", "get_cigar.sh"] + [row["cram_file"], region, output_file],
            capture_output=True,
            text=True,
        )

    # remove all indexed cram files
    files = os.listdir()
    for file in files:
        if file.endswith(".cram.crai"):
            os.remove(file)


def get_all_bam_files():
    """Get BAM files for all samples/SV regions for SVs with 2+ modes."""
    df = pd.read_csv("1kgp/svs_n_modes.csv")
    df = df[df["num_modes"] > 1]
    sv_ids = df["sv_id"].unique()
    for sv_id in sv_ids:
        get_bam_files(sv_id)


def write_samplot_files():
    """Write samplot images for all SVs with 2+ modes."""
    sv_ids = os.listdir("long_reads/bam_files")
    for sv_id in sv_ids:
        chr, start, stop = get_sv_chr(sv_id)
        subprocess.run(
            ["bash", "samplot_viz.sh"] + [sv_id, chr, str(start), str(stop)],
            capture_output=True,
            text=True,
        )


def write_post_processed_files(stem: str = "1kgp"):
    """Write all post-processed files after running the dirichlet process with short or long reads."""
    get_n_modes(stem)
    print("wrote svs_n_modes.csv")
    if stem == "long_reads":
        compare_short_long_reads()
        print("wrote sr_lr_merged.csv")
    get_consensus_svs(stem)
    print("wrote consensus_svs.csv")
    write_sv_stats_collapsed(stem)
    print("wrote sv_stats_collapsed.csv")
    get_outliers(stem)
    print("wrote outliers.csv")
    write_ancestry_dissimilarity(stem)
    print("wrote ancestry_dissimilarity.csv")
    # get_new_gene_intersections() # need bedtools to run this


if __name__ == "__main__":
    # write_post_processed_files("long_reads")
    # remove_gatk_rows()
    write_samplot_files()
