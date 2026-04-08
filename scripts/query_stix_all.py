import argparse
import multiprocessing
import os
import re

import pandas as pd
import pysam

from src.data.query_stix import query_stix
from src.utils.config_loader import load_config
from src.utils.helper import get_sample_ids
from src.utils.model_helper import (
    get_query_region,
    load_vcf,
    process_input_files,
    giggle_format,
)
from src.utils.timeout import break_after

INPUT_DIR = "1kgp"


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


def query_stix_sv(cfg: dict, row: pd.Series):
    query_region = get_query_region(
        giggle_format(row["chr"], row["start"]),
        giggle_format(row["chr"], row["stop"]),
        cfg["query"]["read_overlap"],
    )
    _ = query_stix(
        query_region,
        cfg["paths"]["stix_output_dir"],
        cfg["stix"]["bin"],
        cfg["stix"]["index"],
        cfg["stix"]["database"],
        cfg["stix"]["num_shards"],
        True,
    )


@break_after(hours=166, minutes=0)  # break before the job is cancelled
def query_stix_all(cfg: dict, sv_lookup_file: str):
    """Query all SVs in deletions.csv using STIX and update num_samples column."""
    input_dir = cfg["input_dir"]

    filename = os.path.join(input_dir, sv_lookup_file)
    _ = process_input_files(
        input_dir,
        sv_lookup_file,
        cfg["input_files"].get("sample_id_file"),
        cfg["input_files"].get("insert_size_file"),
        cfg["input_files"].get("default_insert_size"),
    )

    df = pd.read_csv(filename, low_memory=False)
    df["num_samples"] = 0
    sample_ids = get_sample_ids(INPUT_DIR)
    for index, row in df.iterrows():
        n_nonref = 0
        for sample_id in sample_ids:
            if row[sample_id] != "(0, 0)":
                n_nonref += 1
        df.at[index, "num_samples"] = int(n_nonref)

    with multiprocessing.Manager():
        p = multiprocessing.Pool(multiprocessing.cpu_count())
        args = []
        for _, row in df.iterrows():
            if row["num_samples"] <= 10:
                # skip rows with too few samples
                continue
            args.append(
                (
                    cfg,
                    row.to_dict(),
                )
            )

        print(f"Querying stix for {len(args)} SVs...")
        p.starmap(query_stix_sv, args)
        p.close()
        p.join()
    df.to_csv(filename, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Queries all structural variants in a callset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a TOML config file. CLI flags override config values.",
    )
    args = parser.parse_args()

    config_path = args.config or (
        "config.toml" if os.path.isfile("config.toml") else None
    )
    cfg = load_config(config_path)

    input_dir = cfg["paths"]["input_dir"]
    sv_lookup_file = cfg["input_files"]["sv_lookup_file"]

    if sv_lookup_file.endswith(".vcf") or sv_lookup_file.endswith(".vcf.gz"):
        _, sv_lookup_file = load_vcf(input_dir, sv_lookup_file)

    query_stix_all(cfg, sv_lookup_file)


if __name__ == "__main__":
    main()
