import multiprocessing
import os
import re

import pandas as pd
import pysam

from query_sv import (
    get_query_region,
    query_stix_bash,
    load_vcf,
    process_input_files,
)
from src.utils.helper import get_sample_ids
from src.utils.timeout import break_after

INPUT_DIR = "1kgp"


def prune_genes_bed():
    with open(f"{INPUT_DIR}/grch38.genes.bed", "r") as infile, open(
        f"{INPUT_DIR}/genes.bed", "w"
    ) as outfile:
        for line in infile:
            fields = line.strip().split("\t")
            chrom, start, stop, annotations = (
                fields[0],
                fields[1],
                fields[2],
                fields[6],
            )
            match = re.search(r'gene_name "([^"]+)"', annotations)
            gene_name = match.group(1)
            outfile.write(f"{chrom}\t{start}\t{stop}\t{gene_name}\n")


def vcf_to_bed():
    vcf_in = pysam.VariantFile(f"{INPUT_DIR}/1kg.subset.vcf.gz")
    with open(f"{INPUT_DIR}/deletions.bed", "w") as f:
        for record in vcf_in.fetch():
            info = dict(record.info)
            if info["SVTYPE"] != "DEL":
                continue
            f.write(
                f"{record.chrom.strip('chr')}\t{record.start}\t{record.stop}\t{record.id}\n"
            )


def write_cipos():
    vcf_in = pysam.VariantFile(f"{INPUT_DIR}/1kg.subset.vcf.gz")
    df = pd.DataFrame(columns=["id", "cipos", "ciend"])
    # a record will have all 4 fields cipos, cipos95, ciend, and ciend95 or none of the above
    n_missing = 0
    for record in vcf_in.fetch():
        info = dict(record.info)
        chr = record.chrom.strip("chr")
        if info["SVTYPE"] != "DEL" or chr in ["X", "Y"]:
            continue
        if "CIPOS" in record.info:
            df.loc[len(df)] = [
                record.id,
                record.info["CIPOS"],
                record.info["CIEND"],
            ]
        else:
            n_missing += 1

    print(f"Number of records without CIPOS: {n_missing}")  # 74886 records
    df.to_csv(f"{INPUT_DIR}/cipos.csv", index=False)  # 11522 rows


def query_stix_sv(row: pd.Series):
    query_region = get_query_region(
        f"{row['chr']}:{row['start']}", f"{row['chr']}:{row['stop']}"
    )
    _ = query_stix_bash(
        query_region,
        "/scratch/Users/vili4418/stix_output",
        "/Users/vili4418/sv/stix/bin/stix",
        "/scratch/Shares/layer/stix/indices/1kg_high_coverage_vivian/shard",
        "/scratch/Shares/layer/stix/indices/1kg_high_coverage_vivian/shard",
        8,
        True,
    )


@break_after(hours=166, minutes=0)  # break before the job is cancelled
def query_stix_all():
    """Query all SVs in deletions.csv using STIX and update num_samples column."""
    filename = f"{INPUT_DIR}/deletions.csv"
    process_input_files(INPUT_DIR, "deletions.csv", "insert_sizes.csv")

    df = pd.read_csv(filename, low_memory=False)
    df["num_samples"] = 0
    sample_ids = get_sample_ids(INPUT_DIR)
    for index, row in df.iterrows():
        n_w_allele = 0
        for sample_id in sample_ids:
            if row[sample_id] != "(0, 0)":
                n_w_allele += 1
        df.at[index, "num_samples"] = int(n_w_allele)

    with multiprocessing.Manager():
        p = multiprocessing.Pool(multiprocessing.cpu_count())
        args = []
        for _, row in df.iterrows():
            if row["num_samples"] <= 10:
                # skip rows with too few samples
                continue
            args.append((row.to_dict(),))

        print(f"Querying stix for {len(args)} SVs...")
        p.starmap(query_stix_sv, args)
        p.close()
        p.join()
    df.to_csv(filename, index=False)


def main():
    if not os.path.isfile(f"{INPUT_DIR}/deletions.csv"):
        load_vcf(INPUT_DIR, "1kg.subset.vcf.gz")
    query_stix_all()


if __name__ == "__main__":
    main()
