import re
import os
import pandas as pd
import pysam
import multiprocessing
from typing import Dict
from query_sv import giggle_format, query_stix, PROCESSED_FILE_DIR
from helper import calc_af
from process_data import (
    get_intercepts,
    get_insert_size_lookup,
)
from timeout import break_after

FILE_DIR = "1kgp"


def prune_genes_bed():
    with open(f"{FILE_DIR}/grch38.genes.bed", "r") as infile, open(
        f"{FILE_DIR}/genes.bed", "w"
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
    vcf_in = pysam.VariantFile(f"{FILE_DIR}/1kg.subset.vcf.gz")
    with open(f"{FILE_DIR}/deletions.bed", "w") as f:
        for record in vcf_in.fetch():
            info = dict(record.info)
            if info["SVTYPE"] != "DEL":
                continue
            f.write(
                f"{record.chrom.strip('chr')}\t{record.start}\t{record.stop}\t{record.id}\n"
            )


def write_cipos():
    vcf_in = pysam.VariantFile(f"{FILE_DIR}/1kg.subset.vcf.gz")
    df = pd.DataFrame(columns=["id", "cipos", "ciend"])
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
    df.to_csv(f"{FILE_DIR}/cipos.csv", index=False)  # 11522 rows


def load_vcf():
    vcf_in = pysam.VariantFile(f"{FILE_DIR}/1kg.subset.vcf.gz")
    header = [
        "id",
        "chr",
        "start",
        "stop",
        "svlen",
        "ref",
        "alt",
        "qual",
        "filter",
        "af",
        "info",
    ] + list(vcf_in.header.samples)
    data = []
    n_removed = 0
    for record in vcf_in.fetch():
        info = dict(record.info)
        chr = record.chrom.strip("chr")
        if info["SVTYPE"] != "DEL" or chr in ["X", "Y"]:
            continue
        row = [
            record.id,
            chr,
            record.start,
            record.stop,
            record.rlen,
            record.ref,
            ",".join([str(alt) for alt in record.alts]),
            record.qual,
            record.filter.keys(),
            info["AF"],  # placeholder until it's recalculated below
            info,
        ]
        n_homozygous = 0
        n_heterozygous = 0
        for sample in record.samples:
            gt = record.samples[sample]["GT"]
            gt = tuple([0 if g is None else g for g in gt])  # convert None to 0
            gt_sum = sum(gt)
            if gt_sum == 1:
                n_heterozygous += 1
            elif gt_sum == 2:
                n_homozygous += 1
            row.append(gt)

        af = calc_af(n_homozygous, n_heterozygous, len(record.samples))
        row[9] = af

        # only keep the rows where at least one sample has an allele for the SV (i.e. a 1 in their GT)
        # removed 1760 rows without genotypes
        if n_homozygous > 0 or n_heterozygous > 0:
            data.append(row)

        else:
            n_removed += 1

    print(f"Removed {n_removed} rows without genotypes")
    df = pd.DataFrame(data, columns=header)
    df["num_samples"] = 0
    df.to_csv(f"{FILE_DIR}/deletions_df.csv", index=False)


def get_num_samples(row_index: int, row, lookup: Dict[int, int]):
    start = giggle_format(str(row.chr), row.start)
    end = giggle_format(str(row.chr), row.stop)
    squiggle_data = query_stix(
        l=start, r=end, run_gmm=False, plot=False, scratch=True
    )
    if len(squiggle_data) > 0:
        insert_size_lookup = get_insert_size_lookup()
        intercepts, _ = get_intercepts(
            squiggle_data,
            file_name=None,
            L=row.start,
            R=row.stop,
            insert_size_lookup=insert_size_lookup,
        )
        lookup[row_index] = len(intercepts)


@break_after(hours=23, minutes=55)  # break before the job is cancelled
def get_num_sv():
    filename = f"{FILE_DIR}/deletions_df.csv"
    df = pd.read_csv(filename, low_memory=False)

    processed_files = os.listdir(PROCESSED_FILE_DIR)
    processed_svs = []
    pattern = r"([\w]+):(\d+)-\d+_[\w]+:(\d+)-\d+.csv"

    for file in processed_files:
        match = re.search(pattern, file)
        chr, start, stop = match.groups()
        processed_svs.append((chr, int(start), int(stop)))

    with multiprocessing.Manager() as manager:
        p = multiprocessing.Pool(multiprocessing.cpu_count())
        lookup = manager.dict()
        args = []
        for i, row in df.iterrows():
            if (str(row.chr), row.start, row.stop) in processed_svs:
                continue
            args.append((i, row, lookup))

        p.starmap(get_num_samples, args)
        p.close()
        p.join()

        # this only gets updated if all SVs are processed before timeout
        for row_index, num_samples in lookup.items():
            df.loc[row_index, "num_samples"] = num_samples
    df.to_csv(filename, index=False)


def main():
    if not os.path.isfile(f"{FILE_DIR}/deletions_df.csv"):
        load_vcf()
    # get_num_sv()


if __name__ == "__main__":
    main()
