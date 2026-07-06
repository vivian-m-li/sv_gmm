import multiprocessing
import os
import shutil
import subprocess

import pandas as pd

from src.utils.config_loader import load_config
from src.utils.helper import get_sv_chr, get_sv_lookup


def get_bam_files_sv(
    *,
    sample_sv_lookup_path: str,
    sample_files_path: str,
    out_dir: str,
    sv_id: str | None = None,
    region: str | None = None,
    sample_ids: list | None = None,
    filter_ref_samples: bool = False,
    cfg: dict | None = None,
):
    """Standalone function to get BAM files for all samples in a given SV or region of the genome."""
    assert sv_id is not None or region is not None

    if cfg is None:
        cfg = load_config()

    sample_sv_lookup = pd.read_csv(sample_sv_lookup_path)
    samples = pd.read_csv(sample_files_path)
    df = sample_sv_lookup.merge(samples, on="sample_id")
    if sample_ids is not None:
        df = df[df["sample_id"].isin(sample_ids)]

    all_sample_ids = df["sample_id"].unique()

    if sv_id is not None:
        file_dir = os.path.join(out_dir, sv_id)
        lookup = pd.read_csv(
            os.path.join(
                cfg["paths"]["input_dir"], cfg["input_files"]["sv_lookup_file"]
            )
        )
        row = lookup[lookup["id"] == sv_id]
        region = f"chr{row['chr'].values[0]}:{row['start'].values[0]}-{row['stop'].values[0]}"

        if filter_ref_samples:
            ref_samples = [
                sample_id
                for sample_id in all_sample_ids
                if row[sample_id].values[0] == "(0, 0)"
            ]
            df = df[~df["sample_id"].isin(ref_samples)]
    else:
        file_dir = os.path.join(out_dir, region)

    os.makedirs(file_dir, exist_ok=True)

    for _, row in df.iterrows():
        output_file = os.path.join(file_dir, f"{row['sample_id']}.bam")
        if os.path.exists(output_file):
            continue

        print(f"Downloading bam file for {row['sample_id']}")
        subprocess.run(
            ["bash", "src/data/bash/read_cram_file.sh"]
            + [
                row["file"],
                region,
                output_file,
                cfg["samtools"]["bin"],
                os.path.join(
                    cfg["paths"]["input_dir"],
                    cfg["input_files"]["reference_genome_file"],
                ),
            ],
            capture_output=True,
            text=True,
        )


def get_bam_files(sv_id: str, cfg: dict):
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
            ["bash", "src/data/bash/read_cram_file.sh"]
            + [
                row["file"],
                region,
                output_file,
                cfg["samtools"]["bin"],
                os.path.join(
                    cfg["paths"]["input_dir"],
                    cfg["input_files"]["reference_genome_file"],
                ),
            ],
            capture_output=True,
            text=True,
        )


def get_all_bam_files():
    """Get BAM files for all samples/SV regions for SVs with 2+ modes."""
    cfg = load_config()

    df = pd.read_csv("1kg/svs_n_modes.csv")
    df = df[df["num_modes"] > 1]
    sv_ids = df["sv_id"].unique()
    with multiprocessing.Manager():
        # when using http, limit to 4 processes at once
        p = multiprocessing.Pool(4)
        args = []
        for sv_id in sv_ids:
            args.append((sv_id, cfg))
        p.starmap(get_bam_files, args)
        p.close()
        p.join()

    # remove all indexed cram files
    files = os.listdir()
    for file in files:
        if file.endswith(".cram.crai"):
            os.remove(file)


def samplot_viz(
    *,
    sv_id: str,
    input_dir: str,
    out_dir: str,
    lookup: pd.DataFrame | None = None,
    sample_ids: list | None = None,
) -> None:
    if lookup is None:
        cfg = load_config()
        lookup = get_sv_lookup(cfg["paths"]["input_dir"])
    row = lookup[lookup["id"] == sv_id].iloc[0]

    os.makedirs(out_dir, exist_ok=True)

    if sample_ids is not None:
        temp_dir = os.path.join(input_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        for sample_id in sample_ids:
            for ext in ["bam", "bam.bai"]:
                shutil.copyfile(
                    os.path.join(input_dir, f"{sample_id}.{ext}"),
                    os.path.join(temp_dir, f"{sample_id}.{ext}"),
                )
        bam_files_dir = temp_dir
    else:
        bam_files_dir = input_dir

    subprocess.run(
        ["bash", "src/utils/bash/samplot_viz.sh"]
        + [  # noqa: W503
            sv_id,
            str(row["chr"]),
            str(row["start"]),
            str(row["stop"]),
            bam_files_dir,
            out_dir,
        ],
        capture_output=True,
        text=True,
    )
    print("Samplot file saved to", os.path.join(out_dir, f"{sv_id}.png"))

    if sample_ids is not None:
        shutil.rmtree(temp_dir)


def samplot_viz_all_svs():
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
            samplot_viz(
                sv_id=sv_id,
                bam_files_dir=os.path.join("data/long_reads/bam_files", sv_id),
                out_dir="data/long_reads/samplot_viz",
                lookup=lookup,
            )

            # remove the downloaded bam files to save space
            subprocess.run(
                f"rm -r data/long_reads/bam_files/{sv_id}",
                shell=True,
            )


if __name__ == "__main__":
    get_bam_files_sv(
        sv_id="HGSV_245267",
        sample_sv_lookup_path="data/long_reads/sample_sv_lookup.csv",
        sample_files_path="data/long_reads/long_read_samples.csv",
        out_dir="output/bam_files",
        filter_ref_samples=True,
    )
    samplot_viz(
        sv_id="HGSV_245267",
        input_dir="output/high_cov_bam_files/HGSV_245267",
        out_dir="output/samplot_viz",
    )
