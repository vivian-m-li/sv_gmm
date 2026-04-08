"""
This script takes the original SR SVs and the LR SVs, builds an overlap set using bedtools intersect, and then uses the results to build a lookup file that maps the original SR SV IDs to the new split SV IDs, along with the coordinates and non-ref sample IDs for each split SV. This lookup file can then be used to assign genotypes to the split SVs based on the genotypes of the original SR SVs.
"""

import ast
import os
import re
import subprocess

import pandas as pd
import pysam

from src.utils.model_helper import reciprocal_overlap

FILE_DIR = "sv_genotyping"
GTs = [(0, 1), (1, 0), (1, 1)]


def vcf_to_csv(
    in_file: str,
    out_file: str,
    sv_ids: set[str] | None,
    sample_ids: set[str] | None,
) -> None:
    """
    Converts a vcf to a csv with fields id, chr, start, end, sample_ids.
    If sv_ids is provided, only keeps SVs in sv_ids.
    If sample_ids is provided, only keeps SVs with at least one sample in sample_ids.
    """
    df = pd.DataFrame(columns=["id", "chr", "start", "end", "sample_ids"])
    vcf_in = pysam.VariantFile(in_file)
    for record in vcf_in.fetch():
        for record in vcf_in.fetch():
            if sv_ids is not None and record.id not in sv_ids:
                continue

            info = dict(record.info)
            sv_type = info.get("SVTYPE")
            if sv_type is None:
                pattern = r"^chr[^-]+-\d+-([A-Z]+)->"
                match = re.match(pattern, record.id)
                sv_type = match.group(1)

            # only keep deletion records
            if sv_type != "DEL" or chr in ["X", "Y"]:
                continue

            record_samples = [
                s_id
                for s_id, s_gt in record.samples.items()
                if s_id in sample_ids and s_gt["GT"] in GTs
            ]
            df.loc[len(df)] = [
                record.id,
                record.chrom,
                record.start,
                record.stop,
                ",".join(record_samples),
            ]
    vcf_in.close()
    df.to_csv(out_file, index=False)


def vcf_to_csv_filtered(in_file: str, out_file: str):
    """
    Loads the SVs that were run through SPLIT and the samples that have long
    reads, and filters the VCF to only keep those SVs and samples before
    converting to CSV.
    """
    svs_split_df = pd.read_csv("results/svs_n_modes.csv")
    svs_split_df = svs_split_df[svs_split_df["confidence"] != "inconclusive"]
    svs_run = set(svs_split_df["sv_id"].values)

    lr_samples_df = pd.read_csv("long_reads/long_read_samples.csv")
    lr_samples = set(lr_samples_df["sample_id"].values)

    vcf_to_csv(in_file, out_file, sv_ids=svs_run, sample_ids=lr_samples)


def vcf_to_bed(in_file: str, out_file: str, sample_ids: set[int]) -> None:
    """Converts a vcf to a bed file to be used with bedtools intersect. Writes out columns chr, start, end, id"""

    print(f"Converting {in_file} to bed...", flush=True)
    vcf_in = pysam.VariantFile(in_file)
    sample_lookup = pd.DataFrame(columns=["sv_id", "sample_ids"])
    with open(out_file, "w") as outfile:
        for record in vcf_in.fetch():
            info = dict(record.info)
            sv_type = info.get("SVTYPE")
            if sv_type is None:
                pattern = r"^chr[^-]+-\d+-([A-Z]+)->"
                match = re.match(pattern, record.id)
                sv_type = match.group(1)

            # only keep deletion records
            if sv_type != "DEL" or chr in ["X", "Y"]:
                continue

            outfile.write(
                f"{record.chrom}\t{record.start}\t{record.stop}\t{record.id}\n"
            )
            record_samples = [
                s_id
                for s_id, s_gt in record.samples.items()
                if s_id in sample_ids and s_gt["GT"] in GTs
            ]
            sample_lookup.loc[len(sample_lookup)] = [record.id, record_samples]
    vcf_in.close()
    sample_lookup.to_csv(
        out_file.replace(".bed", "_sample_lookup.csv"), index=False
    )


def build_sr_lr_overlap_set(
    *, sr_vcf: str, lr_vcf: str, bedtools_path: str, output_file: str
) -> None:
    print(
        f"Building SR/LR overlap set for {sr_vcf} and {lr_vcf}...", flush=True
    )
    sr_root = ".".join(sr_vcf.split("/")[-1].strip(".gz").split(".")[:-1])
    lr_root = ".".join(lr_vcf.split("/")[-1].strip(".gz").split(".")[:-1])
    sr_bed = os.path.join(FILE_DIR, f"{sr_root}.bed")
    lr_bed = os.path.join(FILE_DIR, f"{lr_root}.bed")

    if not os.path.exists(sr_bed) or not os.path.exists(lr_bed):
        print("Converting VCFs to BED files...", flush=True)

        # only keep the sample IDs that are in both VCFs
        sample_ids = set(pysam.VariantFile(sr_vcf).header.samples).intersection(
            set(pysam.VariantFile(lr_vcf).header.samples)
        )
        for vcf_file, bed_file in zip([sr_vcf, lr_vcf], [sr_bed, lr_bed]):
            if not os.path.exists(bed_file):
                try:
                    vcf_to_bed(vcf_file, bed_file, sample_ids)
                except Exception:
                    os.remove(bed_file)
                    raise Exception(f"Failed to write {bed_file}.")

    print("Running bedtools intersect...", flush=True)
    intersect_file = os.path.join(
        FILE_DIR,
        f"{sr_root}_{lr_root}_intersect.bed",
    )
    subprocess.run(
        [bedtools_path, "intersect", "-a", sr_bed, "-b", lr_bed, "-wao"],
        stdout=open(intersect_file, "w"),
        check=True,
    )

    intersect_df = pd.read_csv(
        intersect_file,
        sep="\t",
        names=[
            "sr_chr",
            "sr_start",
            "sr_stop",
            "sr_sv_id",
            "lr_chr",
            "lr_start",
            "lr_stop",
            "lr_sv_id",
            "n_bp_overlap",
        ],
    )

    # join with sample ID lookup by SV for both SR and LR callsets
    sr_sample_lookup = pd.read_csv(
        os.path.join(FILE_DIR, f"{sr_root}_sample_lookup.csv")
    )
    sr_sample_lookup = sr_sample_lookup.rename(
        columns={"sv_id": "sr_sv_id", "sample_ids": "sr_sample_ids"}
    )
    lr_sample_lookup = pd.read_csv(
        os.path.join(FILE_DIR, f"{lr_root}_sample_lookup.csv")
    )
    lr_sample_lookup = lr_sample_lookup.rename(
        columns={"sv_id": "lr_sv_id", "sample_ids": "lr_sample_ids"}
    )
    intersect_df = intersect_df.merge(sr_sample_lookup, on="sr_sv_id")
    intersect_df = intersect_df.merge(lr_sample_lookup, on="lr_sv_id")

    out_df = pd.DataFrame(
        columns=[
            "sr_sv_id",
            "lr_sv_id",
            "reciprocal_overlap",
            "sample_overlap",
            "sample_ids",
        ]
    )

    print(f"Writing {output_file}...", flush=True)
    for _, row in intersect_df.iterrows():
        if row["lr_sv_id"] == ".":
            out_df.loc[len(out_df)] = [row["sr_sv_id"], None, 0, 0, ""]
            continue
        overlap = row["n_bp_overlap"] / max(
            row["sr_stop"] - row["sr_start"], row["lr_stop"] - row["lr_start"]
        )
        sr_samples = set(ast.literal_eval(row["sr_sample_ids"]))
        lr_samples = set(ast.literal_eval(row["lr_sample_ids"]))
        try:
            sample_overlap = len(sr_samples.intersection(lr_samples)) / len(
                sr_samples.union(lr_samples)
            )
        except ZeroDivisionError:
            sample_overlap = 0
        out_df.loc[len(out_df)] = [
            row["sr_sv_id"],
            row["lr_sv_id"],
            overlap,
            sample_overlap,
            ",".join(sr_samples.intersection(lr_samples)),
        ]
    out_df.to_csv(os.path.join(FILE_DIR, output_file), index=False)
    print(f"Done writing {output_file}.", flush=True)


def convert_results_to_vcf(out_file: str):
    """Takes the results from SPLIT and converts them to a VCF file with the same format as the original SVs, but with the new SV IDs and coordinates. The non-ref sample IDs should reflect the mode that the sample was assigned to. Also builds a lookup file that maps the original SV IDs to the new split SV IDs."""

    print("Writing SV expanded VCF...", flush=True)
    collapsed = pd.read_csv("results/sv_stats_collapsed.csv")

    # initialize a new vcf and copy headers from the original vcf, keeping ID, CHROM, POS, END, and sample columns
    template_vcf = pysam.VariantFile("1kgp/1kg.subset.vcf.gz")
    expanded_vcf = pysam.VariantFile(out_file, "w", header=template_vcf.header)

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
        og_sv_coords = (row["start"], row["stop"])
        modes = ast.literal_eval(row["modes"])
        # sort modes from most to least reciprocal overlap with the original SV
        modes = sorted(
            modes,
            key=lambda x: reciprocal_overlap(
                og_sv_coords, (x["start"], x["end"])
            ),
            reverse=True,
        )
        for i, mode in enumerate(modes):
            # write a new row in the vcf with the new SV ID, coordinates, and non-ref sample IDs
            new_record = expanded_vcf.new_record()
            new_record.id = f"{row['id']}_{i}"
            new_record.chrom = f"chr{row['chr']}"
            new_record.pos = mode["start"]
            new_record.stop = mode["end"]
            new_record.info["SVTYPE"] = "DEL"
            new_record.alleles = ("N", "<DEL>")
            for sample in mode["sample_ids"]:
                new_record.samples[sample]["GT"] = (1, 1)
            expanded_vcf.write(new_record)

            lookup_df.loc[len(lookup_df)] = [
                row["id"],
                new_record.id,
                new_record.chrom,
                new_record.pos,
                new_record.stop,
                ",".join(mode["sample_ids"]),
            ]
    lookup_df.to_csv(out_file.replace(".vcf", "_lookup.csv"), index=False)
    expanded_vcf.close()


def main():
    # build_sr_lr_overlap_set once for the original SR SVs, then for the LR SVs
    lr_vcf = "long_reads/all.sniffles.hg38.1kGP.ont.7Aug2023.vcf.gz"
    expanded_sr_vcf = "results/1kg_expanded.vcf"
    if not os.path.exists(expanded_sr_vcf):
        try:
            convert_results_to_vcf(expanded_sr_vcf)
        except Exception:
            os.remove(expanded_sr_vcf)
            raise Exception(f"Failed to write {expanded_sr_vcf}. Exiting now.")
    for sr_vcf, output_file_root in zip(
        ["1kgp/1kg.subset.vcf.gz", expanded_sr_vcf],
        ["original_sr", "expanded_sr"],
    ):
        output_file = f"{output_file_root}_lr_overlaps.csv"
        if os.path.exists(os.path.join(FILE_DIR, output_file)):
            continue
        build_sr_lr_overlap_set(
            sr_vcf=sr_vcf,
            lr_vcf=lr_vcf,
            bedtools_path="/Users/vili4418/sv/bedtools/bin/bedtools",
            output_file=output_file,
        )


if __name__ == "__main__":
    # main()
    vcf_to_csv_filtered(
        "1kgp/1kg.subset.vcf.gz", "sv_genotyping/1kg.subset.split_svs_only.csv"
    )
