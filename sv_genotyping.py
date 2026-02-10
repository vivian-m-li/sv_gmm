import os
import pysam
import subprocess
import ast
import pandas as pd
from typing import Set

"""
This script takes the original SR SVs and the LR SVs, builds an overlap set using bedtools intersect, and then uses the results to build a lookup file that maps the original SR SV IDs to the new split SV IDs, along with the coordinates and non-ref sample IDs for each split SV. This lookup file can then be used to assign genotypes to the split SVs based on the genotypes of the original SR SVs.
"""

FILE_DIR = "sv_genotyping"
GTs = [(0, 1), (1, 0), (1, 1)]


def vcf_to_bed(in_file: str, out_file: str, sample_ids: Set[int]) -> None:
    """Converts a vcf to a bed file to be used with bedtools intersect. Writes out columns id, chr, start, end, non-ref sample_ids."""
    vcf_in = pysam.VariantFile(in_file)
    with open(out_file, "w") as outfile:
        for record in vcf_in.fetch():
            record_samples = [
                s_id
                for s_id, s_gt in record.samples.items()
                if s_id in sample_ids and s_gt["GT"] in GTs
            ]
            outfile.write(
                f"{record.id}\t{record.chrom}\t{record.start}\t{record.stop}\t{','.join(record_samples)}\n"
            )
    vcf_in.close()


def build_sr_lr_overlap_set(
    *, sr_vcf: str, lr_vcf: str, bedtools_path: str, output_file: str
) -> None:
    sr_bed = os.path.join(
        FILE_DIR, sr_vcf.split("/")[-1].replace(".vcf.gz", ".bed")
    )
    lr_bed = os.path.join(
        FILE_DIR, lr_vcf.split("/")[-1].replace(".vcf.gz", ".bed")
    )

    if not os.path.exists(sr_bed) and not os.path.exists(lr_bed):
        print("Converting VCFs to BED files...")

        # only keep the sample IDs that are in both VCFs
        sample_ids = set(pysam.VariantFile(sr_vcf).header.samples).intersection(
            set(pysam.VariantFile(lr_vcf).header.samples)
        )
        if not os.path.exists(sr_bed):
            vcf_to_bed(sr_vcf, sr_bed, sample_ids)
        if not os.path.exists(lr_bed):
            vcf_to_bed(lr_vcf, lr_bed, sample_ids)

    intersect_file = os.path.join(FILE_DIR, "sr_lr_intersect.bed")
    subprocess.run(
        [bedtools_path, "intersect", "-a", sr_bed, "-b", lr_bed, "-wao"],
        stdout=open(intersect_file, "w"),
        check=True,
    )

    intersect_df = pd.read_csv(
        intersect_file,
        sep="\t",
        names=[
            "sr_sv_id",
            "sr_chr",
            "sr_start",
            "sr_stop",
            "sr_sample_ids",
            "lr_sv_id",
            "lr_chr",
            "lr_start",
            "lr_stop",
            "lr_sample_ids",
            "n_bp_overlap",
        ],
    )

    out_df = pd.DataFrame(
        columns=[
            "sr_sv_id",
            "lr_sv_id",
            "reciprocal_overlap",
            "sample_overlap",
            "sample_ids",
        ]
    )
    for _, row in intersect_df.iterrows():
        if row["lr_sv_id"] == ".":
            out_df.loc[len(out_df)] = [row["sr_sv_id"], None, 0, 0, ""]
            continue
        overlap = row["n_bp_overlap"] / max(
            row["sr_stop"] - row["sr_start"], row["lr_stop"] - row["lr_start"]
        )
        sr_samples = set(row["sr_sample_ids"].split(","))
        lr_samples = set(row["lr_sample_ids"].split(","))
        sample_overlap = len(sr_samples.intersection(lr_samples)) / len(
            sr_samples.union(lr_samples)
        )
        out_df.loc[len(out_df)] = [
            row["sr_sv_id"],
            row["lr_sv_id"],
            overlap,
            sample_overlap,
            ",".join(sr_samples.intersection(lr_samples)),
        ]
    out_df.to_csv(os.path.join(FILE_DIR, output_file), index=False)


def convert_results_to_vcf(out_file: str):
    """Takes the results from SPLIT and converts them to a VCF file with the same format as the original SVs, but with the new SV IDs and coordinates. The non-ref sample IDs should reflect the mode that the sample was assigned to. Also builds a lookup file that maps the original SV IDs to the new split SV IDs."""
    collapsed = pd.read_csv("results/sv_stats_collapsed.csv")

    # initialize a new vcf and copy headers from the original vcf, keeping ID, CHROM, POS, END, and sample columns
    template_vcf = pysam.VariantFile("1kgp/1kg.subset.vcf.gz")
    expanded_vcf = pysam.VariantFile(
        os.path.join(FILE_DIR, out_file), "w", header=template_vcf.header
    )

    lookup_df = pd.DataFrame(
        columns=[
            "original_sv_id",
            "split_sv_id",
            "chr",
            "start",
            "stop",
            "sample_ids",
        ]
    )

    for _, row in collapsed.iterrows():
        modes = ast.literal_eval(row["modes"])
        # sort modes from most to least reciprocal overlap with the original SV
        modes = sorted(modes, key=lambda x: x[1], reverse=True)
        for i, mode in enumerate(modes):
            # write a new row in the vcf with the new SV ID, coordinates, and non-ref sample IDs
            new_record = expanded_vcf.new_record()
            expanded_vcf.id = f"{row['id']}_{i}"
            expanded_vcf.chrom = row["chr"]
            expanded_vcf.pos = mode["start"]
            expanded_vcf.stop = mode["stop"]
            for sample in mode["sample_ids"]:
                new_record.samples[sample] = (1, 1)
            expanded_vcf.write(new_record)

            lookup_df.loc[len(lookup_df)] = [
                expanded_vcf.id,
                expanded_vcf.id,
                expanded_vcf.chrom,
                expanded_vcf.pos,
                expanded_vcf.stop,
                ",".join(mode["sample_ids"]),
            ]
    expanded_vcf.close()


def main():
    # build_sr_lr_overlap_set once for the original SR SVs, then for the LR SVs
    lr_vcf = "long_reads/final-vcf.unphased.vcf.gz"
    expanded_sr_vcf = "results/1kg_expanded_vcf.gz"
    if not os.path.exists(expanded_sr_vcf):
        convert_results_to_vcf(expanded_sr_vcf)
    for sr_vcf, output_file in zip(
        ["1kgp/1kg.subset.vcf.gz", expanded_sr_vcf],
        ["original_sr", "expanded_sr"],
    ):
        build_sr_lr_overlap_set(
            sr_vcf=sr_vcf,
            lr_vcf=lr_vcf,
            bedtools_path="/Users/vili4418/sv/bedtools/bin/bedtools",
            output_file=f"{output_file}_lr_overlaps.csv",
        )


if __name__ == "__main__":
    main()
