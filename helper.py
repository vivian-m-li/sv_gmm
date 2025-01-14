import os
import subprocess
import pandas as pd
import csv
from dataclasses import fields
from gmm_types import *


def find_missing_sample_ids():
    sample_ids = set()
    for file in os.listdir("processed_stix_output"):
        with open(f"processed_stix_output/{file}") as f:
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
    processed_sv_ids = set([file.strip(".csv") for file in os.listdir("processed_svs")])
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
        for file in os.listdir("processed_svs"):
            with open(f"processed_svs/{file}") as f:
                for line in f:
                    out.write(line)


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


if __name__ == "__main__":
    get_insert_sizes()
