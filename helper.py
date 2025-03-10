import os
import ast
import subprocess
import pandas as pd
import numpy as np
import csv
from scipy.spatial.distance import braycurtis
from collections import defaultdict, Counter
from gmm_types import Evidence, SVStat
from typing import List

PROCESSED_STIX_DIR = "processed_stix_output"
PROCESSED_SVS_DIR = "processed_svs"


def get_deletions_df(stem: str = "1kgp"):
    return pd.read_csv(f"{stem}/deletions_df.csv", low_memory=False)


def get_sv_stats_df(stem: str = "1kgp"):
    return pd.read_csv(f"{stem}/sv_stats.csv")


def get_sv_stats_converge_df(stem: str = "1kgp"):
    return pd.read_csv(f"{stem}/sv_stats_converge.csv", low_memory=False)


def get_sample_ids(file_root: str = "1kgp"):
    sample_ids = set()
    with open(f"{file_root}/sample_ids.txt", "r") as f:
        for line in f:
            sample_ids.add(line.strip())
    return sample_ids


def get_svlen(evidence_by_mode: List[List[Evidence]]) -> List[List[SVStat]]:
    all_stats = []
    for mode in evidence_by_mode:
        stats = []
        for evidence in mode:
            lengths = [
                max(paired_end) - min(paired_end)
                for paired_end in evidence.paired_ends
            ]
            stats.append(
                SVStat(
                    length=np.mean(lengths) - evidence.mean_insert_size,
                    start=max(
                        [paired_end[0] for paired_end in evidence.paired_ends]
                    ),
                    end=min(
                        [paired_end[1] for paired_end in evidence.paired_ends]
                    ),
                )
            )
        all_stats.append(stats)
    return all_stats


def find_missing_sample_ids():
    sample_ids = set()
    for file in os.listdir(PROCESSED_STIX_DIR):
        with open(f"{PROCESSED_STIX_DIR}/{file}") as f:
            for line in f:
                sample_id = line.strip().split(",")[0]
                if sample_id[0].isalpha():
                    sample_ids.add(sample_id)

    # print(len(sample_ids))  # 2535
    deletions_df = get_deletions_df()
    missing = sample_ids - set(deletions_df.columns[11:-1])

    # print(len(missing))  # 31
    # print(missing)
    # {'HG00702', 'NA20898', 'HG00124', 'NA20336', 'NA19675', 'HG03715', 'HG02024', 'NA19311', 'NA19685', 'HG02363', 'NA20871', 'NA20322', 'HG00501', 'nan', 'NA19240', 'HG01983', 'NA19985', 'HG02381', 'HG02388', 'NA20341', 'HG02377', 'NA20526', 'HG00635', 'NA19313', 'HG02387', 'NA19660', 'HG00733', 'NA20893', 'HG03948', 'HG02372', 'HG02046', 'NA20344'}

    return missing


def find_missing_processed_svs():
    processed_sv_ids = set(
        [file.strip(".csv") for file in os.listdir(PROCESSED_SVS_DIR)]
    )
    deletions_df = get_deletions_df()
    missing = set(deletions_df["id"]) - processed_sv_ids
    print(len(missing))
    print(missing)
    return missing


def get_ambiguous_svs():
    df = pd.read_csv("1kgp/sv_stats_merged.csv")
    unique_svs = df["id"].unique()
    rerun_sv_ids = []
    for sv_id in unique_svs:
        sv_df = df[df["id"] == sv_id]
        num_modes = sv_df["num_modes"].unique()
        if len(num_modes) > 1:
            rerun_sv_ids.append(sv_id)

    with open("1kgp/svs_to_rerun.txt", "w") as f:
        for sv_id in rerun_sv_ids:
            f.write(f"{sv_id}\n")


def get_num_intersecting_genes():
    df = pd.read_csv(
        "1kgp/intersect_num_overlap.csv", header=None, delimiter="\t"
    )
    num_intersections = df.iloc[:, 5]
    print(f"Total number of genes: {len(num_intersections)}")

    # Number of SV that each gene intersects with
    # 23.36% of genes intersect with at least 1 SV
    num_intersections_filtered = num_intersections[num_intersections > 0]
    print(num_intersections_filtered.describe())


def get_num_new_svs():
    df = get_sv_stats_df()
    df = df[df["num_samples"] > 0]
    mode_data = [df[df["num_modes"] == i + 1] for i in range(3)]
    num_two_modes = len(mode_data[1])
    num_three_modes = len(mode_data[2])
    print(f"Number of new SVs: {num_two_modes + (num_three_modes * 2)}")


def get_sample_sequencing_centers():
    # data obtained from https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/20130502.phase3.sequence.index
    df = pd.read_csv(
        "1kgp/20130502.phase3.sequence.index", sep="\t", low_memory=False
    )
    df["CENTER_NAME"] = df["CENTER_NAME"].str.upper()
    df = df[["SAMPLE_NAME", "CENTER_NAME"]].drop_duplicates()
    df = df.groupby("SAMPLE_NAME")["CENTER_NAME"].apply(list).reset_index()
    return df


def get_insert_sizes(get_files: bool = False):
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


def write_ancestry_dissimilarity():
    df = get_sv_stats_df()
    ancestry_df = pd.read_csv("1kgp/ancestry.tsv", delimiter="\t")
    df = df[df["num_modes"] == 2]

    results_df = pd.DataFrame(
        columns=["chr", "start", "stop", "num_samples", "dissimilarity"]
    )
    for i, row in df.iterrows():
        ancestry = []
        row["num_samples"]
        modes = ast.literal_eval(
            row["modes"]
        )  # currently a str type, parse into a list
        for mode in modes:
            sample_ids = mode["sample_ids"]
            ancestry_comp = {
                anc: 0 for anc in ["SAS", "EAS", "EUR", "AMR", "AFR"]
            }
            for sample_id in sample_ids:
                ancestry_row = ancestry_df[
                    ancestry_df["Sample name"] == sample_id
                ]
                superpopulation = (
                    ancestry_row["Superpopulation code"].values[0].split(",")[0]
                )
                ancestry_comp[superpopulation] += 1
            ancestry_comp = {
                k: v / len(sample_ids) for k, v in ancestry_comp.items()
            }
            ancestry.append([v for v in ancestry_comp.values()])

        dissimilarity = braycurtis(ancestry[0], ancestry[1])
        results_df.loc[i] = [
            row["chr"],
            row["start"],
            row["stop"],
            row["num_samples"],
            dissimilarity,
        ]

    results_df = results_df.sort_values(by="dissimilarity", ascending=False)
    results_df.to_csv("1kgp/ancestry_dissimilarity.csv")


def extract_data_from_deletions_df():
    deletions_df = get_deletions_df()

    sample_ids = set(deletions_df.columns[11:-1])
    with open("1kgp/sample_ids.txt", "w") as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")

    # split the deletions into separate files by chromosome
    # another option is to create a lookup for row index in deletions_df by chr, start, stop, sample_id and
    os.mkdir("1kgp/deletions_by_chr")
    for i in range(1, 23):
        chr_df = deletions_df[deletions_df["chr"] == f"{i}"]
        chr_df.to_csv(f"1kgp/deletions_by_chr/chr{i}.csv", index=False)


def get_insert_size_diff():
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


def calculate_posteriors(alpha):
    p = alpha / np.sum(alpha)
    sum_alpha_post = np.sum(alpha)
    var = (alpha * (sum_alpha_post - alpha)) / (
        sum_alpha_post**2 * (sum_alpha_post + 1)
    )
    return p, var


def calculate_ci(p, var, n):
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
    counts = Counter(outcomes)
    alpha = np.array([1, 1, 1]) + np.array([counts[1], counts[2], counts[3]])
    return calculate_posteriors(alpha)


def get_n_modes():
    sv_df = pd.DataFrame(columns=["sv_id", "num_modes", "confidence"])
    df = get_sv_stats_converge_df()
    svs = df["id"].unique()
    for sv_id in svs:
        outcomes = df[df["id"] == sv_id]["num_modes"].values
        counter = Counter(outcomes)
        num_modes = max(counter, key=counter.get)
        num_modes = 1 if num_modes == 0 else num_modes
        p, var = calculate_posteriors_from_trials(outcomes)
        ci = calculate_ci(p, var, len(outcomes))
        new_row = [sv_id, num_modes]
        if ci[0] >= 0.6:
            new_row.append("high")
        elif ci[0] >= 0.3 and ci[1] < 0.6:
            new_row.append("medium")
        else:
            new_row.append("low")
        sv_df.loc[len(sv_df)] = new_row
    sv_df.to_csv("1kgp/svs_n_modes.csv", index=False)


def get_outliers(sv_rows):
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
        if count / n > 0.9:
            confident_outliers.append(outlier)

    return confident_outliers


def get_consensus_svs():
    df = get_sv_stats_converge_df()
    sv_ids = df["id"].unique()
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
    consensus_df.to_csv("1kgp/consensus_svs.csv", index=False)


def df_to_bed(in_file: str, out_file: str):
    df = pd.read_csv(in_file)
    with open(out_file, "w") as outfile:
        for _, row in df.iterrows():
            outfile.write(
                f"{row['chr']}\t{row['start']}\t{row['end']}\t{row['sv_id']}\t{row['id']}\n"
            )


def get_new_gene_intersections():
    og_intersections = set()  # (sv_id, gene_id)
    with open("1kgp/original_gene_intersections.bed", "r") as f:
        for line in f:
            row = line.strip().split("\t")
            sv_id, gene_id = row[3], row[7]
            og_intersections.add((sv_id, gene_id))

    df_to_bed("1kgp/consensus_svs.csv", "1kgp/consensus_svs.bed")
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


def get_sv_chr(sv_id: str):
    df = get_deletions_df()
    row = df[df["id"] == sv_id]
    chr, start, stop = (
        row["chr"].values[0],
        row["start"].values[0],
        row["stop"].values[0],
    )
    print(chr, start, stop)


if __name__ == "__main__":
    get_sv_chr("HGSV_120216")
