import re
import os
import pandas as pd
import pysam
import multiprocessing
from typing import Dict
from query_sv import giggle_format, query_stix, PROCESSED_FILE_DIR
from process_data import (
    get_intercepts,
    get_insert_size_lookup,
)

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
    for record in vcf_in.fetch():
        info = dict(record.info)
        if info["SVTYPE"] != "DEL":
            continue
        row = [
            record.id,
            record.chrom.strip("chr"),
            record.start,
            record.stop,
            record.rlen,
            record.ref,
            ",".join([str(alt) for alt in record.alts]),
            record.qual,
            record.filter.keys(),
            info["AF"],
            info,
        ]
        for sample in record.samples:
            row.append(record.samples[sample]["GT"])
        data.append(row)

    df = pd.DataFrame(data, columns=header)
    df["num_samples"] = 0
    df.to_csv(f"{FILE_DIR}/deletions_df.csv", index=False)


def get_num_samples(row_index: int, row, lookup: Dict[int, int]):
    start = giggle_format(str(row.chr), row.start)
    end = giggle_format(str(row.chr), row.stop)
    squiggle_data = query_stix(start, end, False, plot=False)
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


def get_num_sv():
    filename = f"{FILE_DIR}/deletions_df.csv"
    df = pd.read_csv(filename, low_memory=False)

    # only get the first or second half of the df to split into multiple jobs
    df = df.iloc[: len(df) // 2]

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

        for row_index, num_samples in lookup.items():
            df.loc[row_index, "num_samples"] = num_samples
    df.to_csv(filename, index=False)


def main():
    if not os.path.isfile(f"{FILE_DIR}/deletions_df.csv"):
        load_vcf()
    get_num_sv()


if __name__ == "__main__":
    main()
