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
    # {'BI_GS_CNV_3_179850285_179863242', 'SI_BD_3777', 'BI_GS_CNV_3_178852217_178860500', 'UW_VH_22869', 'BI_GS_DEL1_B2_P0674_74', 'YL_CN_FIN_274', 'DEL_pindel_1945', 'BI_GS_DEL1_B1_P0212_884', 'DEL_pindel_1947', 'BI_GS_DEL1_B1_P0674_634', 'UW_VH_12590', 'BI_GS_DEL1_B4_P0674_136', 'DEL_pindel_10330', 'SI_BD_3779', 'BI_GS_DEL1_B5_P0673_746', 'UW_VH_4766', 'BI_GS_DEL1_B2_P0213_305', 'SI_BD_3772', 'BI_GS_DEL1_B5_P0213_119', 'BI_GS_DEL1_B1_P0213_274', 'BI_GS_DEL1_B2_P0212_692', 'BI_GS_CNV_1_212324848_212327847', 'DEL_pindel_10325', 'BI_GS_DEL1_B5_P0214_356', 'SI_BD_3781', 'BI_GS_DEL1_B3_P0674_10', 'BI_GS_DEL1_B1_P0213_330', 'BI_GS_DEL1_B5_P0674_126', 'UW_VH_20586', 'BI_GS_DEL1_B2_P0213_52', 'BI_GS_DEL1_B3_P0214_4', 'SI_BD_3769', 'BI_GS_DEL1_B4_P0213_63', 'YL_CN_PEL_289', 'YL_CN_ACB_927', 'DUP_uwash_chr3_179765681_179813685', 'BI_GS_DEL1_B5_P0674_428', 'UW_VH_12673', 'UW_VH_10095', 'BI_GS_DEL1_B5_P0674_770', 'YL_CN_CDX_702', 'BI_GS_DEL1_B4_P0674_47', 'UW_VH_6151', 'BI_GS_DEL1_B1_P0673_728', 'UW_VH_6797'}
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
