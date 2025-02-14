import os
import ast
import subprocess
import pandas as pd
import csv
from scipy.spatial.distance import braycurtis
from dataclasses import fields
from gmm_types import *

PROCESSED_STIX_DIR = "processed_stix_output"
PROCESSED_SVS_DIR = "processed_svs"


def find_missing_sample_ids():
    sample_ids = set()
    for file in os.listdir(PROCESSED_STIX_DIR):
        with open(f"{PROCESSED_STIX_DIR}/{file}") as f:
            for line in f:
                sample_id = line.strip().split(",")[0]
                if sample_id[0].isalpha():
                    sample_ids.add(sample_id)

    # print(len(sample_ids))  # 2535
    deletions_df = pd.read_csv("1000genomes/deletions_df.csv", low_memory=False)
    missing = sample_ids - set(deletions_df.columns[11:-1])

    # print(len(missing))  # 31
    # print(missing)
    # {'HG00702', 'NA20898', 'HG00124', 'NA20336', 'NA19675', 'HG03715', 'HG02024', 'NA19311', 'NA19685', 'HG02363', 'NA20871', 'NA20322', 'HG00501', 'nan', 'NA19240', 'HG01983', 'NA19985', 'HG02381', 'HG02388', 'NA20341', 'HG02377', 'NA20526', 'HG00635', 'NA19313', 'HG02387', 'NA19660', 'HG00733', 'NA20893', 'HG03948', 'HG02372', 'HG02046', 'NA20344'}

    return missing


def find_missing_processed_svs():
    processed_sv_ids = set(
        [file.strip(".csv") for file in os.listdir(PROCESSED_SVS_DIR)]
    )
    deletions_df = pd.read_csv("1000genomes/deletions_df.csv", low_memory=False)
    missing = set(deletions_df["id"]) - processed_sv_ids
    print(len(missing))
    print(missing)
    return missing


def concat_processed_sv_files():
    with open("1000genomes/sv_stats.csv", mode="w", newline="") as out:
        fieldnames = [field.name for field in fields(SVInfoGMM)]
        csv_writer = csv.DictWriter(out, fieldnames=fieldnames)
        csv_writer.writeheader()
        for file in os.listdir(PROCESSED_SVS_DIR):
            with open(f"{PROCESSED_SVS_DIR}/{file}") as f:
                for line in f:
                    out.write(line)


def concat_multi_processed_sv_files():
    with open("1000genomes/sv_stats_merged.csv", mode="w", newline="") as out:
        fieldnames = [field.name for field in fields(SVInfoGMM)]
        csv_writer = csv.DictWriter(out, fieldnames=fieldnames)
        csv_writer.writeheader()
        for file in os.listdir(PROCESSED_SVS_DIR):
            if "iteration" not in file:
                continue
            with open(f"{PROCESSED_SVS_DIR}/{file}") as f:
                for line in f:
                    out.write(line)


def get_ambiguous_svs():
    df = pd.read_csv("1000genomes/sv_stats_merged.csv")
    unique_svs = df["id"].unique()
    rerun_sv_ids = []
    for sv_id in unique_svs:
        sv_df = df[df["id"] == sv_id]
        num_modes = sv_df["num_modes"].unique()
        if len(num_modes) > 1:
            rerun_sv_ids.append(sv_id)

    with open("1000genomes/svs_to_rerun.txt", "w") as f:
        for sv_id in rerun_sv_ids:
            f.write(f"{sv_id}\n")


def get_num_intersecting_genes():
    df = pd.read_csv(
        "1000genomes/intersect_num_overlap.csv", header=None, delimiter="\t"
    )
    num_intersections = df.iloc[:, 5]
    print(f"Total number of genes: {len(num_intersections)}")

    # Number of SV that each gene intersects with
    # 23.36% of genes intersect with at least 1 SV
    num_intersections_filtered = num_intersections[num_intersections > 0]
    print(num_intersections_filtered.describe())


def get_num_new_svs():
    df = pd.read_csv("1000genomes/sv_stats.csv")
    df = df[df["num_samples"] > 0]
    mode_data = [df[df["num_modes"] == i + 1] for i in range(3)]
    num_two_modes = len(mode_data[1])
    num_three_modes = len(mode_data[2])
    print(f"Number of new SVs: {num_two_modes + (num_three_modes * 2)}")


def get_sample_sequencing_centers():
    # data obtained from https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/20130502.phase3.sequence.index
    df = pd.read_csv(
        "1000genomes/20130502.phase3.sequence.index", sep="\t", low_memory=False
    )
    df["CENTER_NAME"] = df["CENTER_NAME"].str.upper()
    df = df[["SAMPLE_NAME", "CENTER_NAME"]].drop_duplicates()
    df = df.groupby("SAMPLE_NAME")["CENTER_NAME"].apply(list).reset_index()
    return df


def get_insert_sizes(get_files: bool = False):
    if get_files:
        samples_df = pd.read_csv("1000genomes/bam_bas_files.tsv", sep="\t")
        pattern = r"data\/.*\/alignment\/.*\.mapped\.ILLUMINA\.bwa\..+\.low_coverage\.\d+\.bam\.bas"
        bas_files_df = samples_df[samples_df["BAS FILE"].str.fullmatch(pattern)]
        bas_files_df["sample_id"] = bas_files_df["BAS FILE"].str.extract(
            r"\/(.+)\/alignment"
        )

        for _, row in bas_files_df.iterrows():
            bas_file = row["BAS FILE"]
            sample_id = row["sample_id"]
            if f"{sample_id}.tsv" in os.listdir("1000genomes/mapped_files"):
                continue
            subprocess.run(
                ["bash", "get_mapped_files.sh"] + [sample_id, bas_file],
                capture_output=True,
                text=True,
            )

    df = pd.DataFrame(columns=["sample_id", "mean_insert_size"])
    mapped_files = os.listdir("1000genomes/mapped_files")
    for i, file in enumerate(mapped_files):
        sample_id = file.strip(".tsv")
        try:
            file_df = pd.read_csv(f"1000genomes/mapped_files/{file}", sep="\t")
        except Exception:
            # alignment files don't exist for samples HG00361, HG00844, NA18555.tsv
            print(file)
        mean_insert_size = int(np.mean(file_df["mean_insert_size"]))
        df.loc[i] = [sample_id, mean_insert_size]
    df.to_csv("1000genomes/insert_sizes.csv", index=False)


def get_mean_coverage():
    df = pd.read_csv("1000genomes/sv_stats.csv")
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

    coverage_df.to_csv("1000genomes/coverage.csv", index=False)


def write_ancestry_dissimilarity():
    df = pd.read_csv("1000genomes/sv_stats.csv")
    ancestry_df = pd.read_csv("1000genomes/ancestry.tsv", delimiter="\t")
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
            ancestry_comp = {anc: 0 for anc in ["SAS", "EAS", "EUR", "AMR", "AFR"]}
            for sample_id in sample_ids:
                ancestry_row = ancestry_df[ancestry_df["Sample name"] == sample_id]
                superpopulation = (
                    ancestry_row["Superpopulation code"].values[0].split(",")[0]
                )
                ancestry_comp[superpopulation] += 1
            ancestry_comp = {k: v / len(sample_ids) for k, v in ancestry_comp.items()}
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
    results_df.to_csv("1000genomes/ancestry_dissimilarity.csv")


def extract_data_from_deletions_df():
    deletions_df = pd.read_csv("1000genomes/deletions_df.csv", low_memory=False)

    sample_ids = set(deletions_df.columns[11:-1])
    with open("1000genomes/sample_ids.txt", "w") as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")

    # split the deletions into separate files by chromosome
    # another option is to create a lookup for row index in deletions_df by chr, start, stop, sample_id and
    os.mkdir("1000genomes/deletions_by_chr")
    for i in range(1, 23):
        chr_df = deletions_df[deletions_df["chr"] == f"{i}"]
        chr_df.to_csv(f"1000genomes/deletions_by_chr/chr{i}.csv", index=False)


def get_insert_size_diff():
    df = pd.read_csv(
        "1000genomes/insert_sizes_scraped.csv",
        dtype={"sample_id": str, "mean_insert_size": int},
    )
    df2 = pd.read_csv(
        "1000genomes/insert_sizes.csv",
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
    df.to_csv("1000genomes/insert_size_compare.csv", index=False)


if __name__ == "__main__":
    concat_processed_sv_files()
