import os
import subprocess
import multiprocessing
import pandas as pd
from helper import get_sv_chr


def get_bam_files(sv_id: str):
    """Get BAM files for all samples that have the given SV."""
    # get samples with the given sv_id
    df = pd.read_csv("long_reads/sample_sv_lookup.csv")
    df = df[df["sv_id"] == sv_id]
    long_read_samples = pd.read_csv("long_reads/long_read_samples.csv")
    df = df.merge(long_read_samples, on="sample_id")

    # get bam files
    file_dir = f"long_reads/bam_files/{sv_id}"
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    chr, start, stop = get_sv_chr(sv_id)
    region = f"chr{chr}:{start}-{stop}"
    for _, row in df.iterrows():
        output_file = f"{file_dir}/{row['sample_id']}.bam"
        subprocess.run(
            ["bash", "get_cigar.sh"] + [row["cram_file"], region, output_file],
            capture_output=True,
            text=True,
        )


def get_all_bam_files():
    """Get BAM files for all samples/SV regions for SVs with 2+ modes."""
    df = pd.read_csv("1kgp/svs_n_modes.csv")
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


if __name__ == "__main__":
    get_all_bam_files()
