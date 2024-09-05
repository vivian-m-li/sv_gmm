import pandas as pd
import pysam
import multiprocessing
from query_sv import *
from viz import *

CHRS = [str(i) for i in range(1, 23)] + ["X", "Y"]
FILE_DIR = "1000genomes"


def load_vcf():
    vcf_in = pysam.VariantFile(
        f"{FILE_DIR}/ALL.wgs.integrated_sv_map_v2.20130502.svs.genotypes.deletions.vcf.gz"
    )
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
        row = [
            record.id,
            record.chrom,
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
    squiggle_data = query_stix(start, end, False)
    if len(squiggle_data) > 0:
        intercepts, _ = get_intercepts(
            squiggle_data,
            file_name=None,
            L=row.start,
            R=row.stop,
        )
        lookup[row_index] = len(intercepts)


def get_num_sv():
    filename = f"{FILE_DIR}/deletions_df.csv"
    df = pd.read_csv(filename)
    with multiprocessing.Manager() as manager:
        p = multiprocessing.Pool(multiprocessing.cpu_count())
        lookup = manager.dict()
        args = []
        for i, row in df.iterrows():
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