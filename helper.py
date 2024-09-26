import os
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


if __name__ == "__main__":
    concat_processed_sv_files()
