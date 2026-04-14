import multiprocessing
import os
import subprocess

import pandas as pd

from src.utils.helper import get_sv_chr, get_sv_lookup


def get_bam_files_region(region: str):
    """Standalone function to get BAM files for all samples in a given region (assuming the region does not correspond with a known SV)."""
    df = pd.read_csv("data/long_reads/sample_sv_lookup.csv")
    long_read_samples = pd.read_csv("data/long_reads/long_read_samples.csv")
    df = df.merge(long_read_samples, on="sample_id")
    file_dir = f"calibration/bam_files/{region}"
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    for _, row in df.iterrows():
        output_file = f"{file_dir}/{row['sample_id']}.bam"
        subprocess.run(
            ["bash", "../../src/data/bash/get_cigar.sh"]
            + [row["cram_file"], region, output_file],
            capture_output=True,
            text=True,
        )


def get_bam_files(sv_id: str):
    """Get BAM files for all samples that have the given SV."""
    # get samples with the given sv_id
    df = pd.read_csv("data/long_reads/sample_sv_lookup.csv")
    df = df[df["sv_id"] == sv_id]
    long_read_samples = pd.read_csv("data/long_reads/long_read_samples.csv")
    df = df.merge(long_read_samples, on="sample_id")

    # get bam files
    file_dir = f"data/long_reads/bam_files/{sv_id}"
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    chr, start, stop = get_sv_chr(sv_id, "data/1kg")
    region = f"chr{chr}:{start}-{stop}"
    for _, row in df.iterrows():
        output_file = f"{file_dir}/{row['sample_id']}.bam"
        subprocess.run(
            ["bash", "../../src/data/bash/get_cigar.sh"]
            + [row["cram_file"], region, output_file],
            capture_output=True,
            text=True,
        )


def get_all_bam_files():
    """Get BAM files for all samples/SV regions for SVs with 2+ modes."""
    df = pd.read_csv("1kg/svs_n_modes.csv")
    df = df[df["num_modes"] > 1]
    sv_ids = df["sv_id"].unique()
    with multiprocessing.Manager():
        # when using http, limit to 4 processes at once
        p = multiprocessing.Pool(4)
        args = []
        for sv_id in sv_ids:
            args.append((sv_id,))
        p.starmap(get_bam_files, args)
        p.close()
        p.join()

    # remove all indexed cram files
    files = os.listdir()
    for file in files:
        if file.endswith(".cram.crai"):
            os.remove(file)


def samplot_viz():
    """Use samplot to visualize each bam file for all SVs with 2+ modes. Make sure to activate the conda env for samplot to work."""
    filename = "data/long_reads/sv_bam_files.txt"
    fiji_root = "vili4418@fiji.colorado.edu:/scratch/Users/vili4418/data/long_reads/bam_files/"
    lookup = get_sv_lookup("data/1kg")
    with open(filename, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            sv_id = line.strip("\n")
            print(f"Processing {sv_id}... ({i + 1}/{len(lines)})", end="\r")
            dir = os.path.join(fiji_root, sv_id)

            # download bam files from fiji
            subprocess.run(
                f"rsync -avz --progress {dir} data/long_reads/bam_files/",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not os.path.exists(f"data/long_reads/bam_files/{sv_id}"):
                print(f"skipping... {sv_id} not found")
                continue

            # use samplot to visualize each bam file
            row = lookup[lookup["id"] == sv_id].iloc[0]
            subprocess.run(
                ["bash", "../../src/utils/bash/samplot_viz.sh"]
                + [  # noqa
                    sv_id,
                    row["chr"],
                    str(row["start"]),
                    str(row["stop"]),
                    "data/long_reads/bam_files",
                    "data/long_reads/samplot_viz",
                ],
                capture_output=True,
                text=True,
            )

            # remove the downloaded bam files to save space
            subprocess.run(
                f"rm -r data/long_reads/bam_files/{sv_id}",
                shell=True,
            )


if __name__ == "__main__":
    # get_all_bam_files()
    samplot_viz()
