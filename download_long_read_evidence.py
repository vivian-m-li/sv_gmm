import os
import subprocess
import pandas as pd
import multiprocessing
from parse_long_reads import get_long_read_svs
from timeout import break_after

SCRATCH_DIR = "/scratch/Users/vili4418"


def get_svs_by_sample():
    df = pd.read_csv("long_reads/long_read_samples.csv")
    deletions_df = pd.read_csv("1kgp/deletions_df.csv")

    lookup_df = pd.DataFrame(columns=["sample_id", "sv_id"])
    sv_alleles = set(["(0, 1)", "(1, 0)", "(1, 1)"])
    for _, row in deletions_df.iterrows():
        for _, sample_row in df.iterrows():
            if row[sample_row["sample_id"]] in sv_alleles:
                lookup_df.loc[len(lookup_df)] = [
                    sample_row["sample_id"],
                    row["id"],
                ]
    lookup_df.to_csv("long_reads/sample_sv_lookup.csv", index=False)


def process_sample_evidence(sample_id, cram_file, sv_ids):
    for sv_id in sv_ids:
        get_long_read_svs(
            sv_id, [sample_id], cram_file, tolerance=300, scratch=True
        )


def remove_cram_file(file):
    os.remove(file)
    os.remove(f"{file}.crai")


def download_sample_evidence(sample_row, sv_ids):
    output_file = os.path.join(
        SCRATCH_DIR, sample_row["cram_file"].split("/")[-1]
    )
    if not os.path.exists(sample_row["cram_file"]):
        subprocess.run(
            ["wget", "-O", output_file, sample_row["cram_file"]],
            capture_output=True,
            text=True,
        )

    if not os.path.exists(sample_row["indexed_cram_file"]):
        subprocess.run(
            [
                "wget",
                "-O",
                f"{output_file}.crai",
                sample_row["indexed_cram_file"],
            ],
            capture_output=True,
            text=True,
        )

    process_sample_evidence(sample_row["sample_id"], output_file, sv_ids)
    remove_cram_file(output_file)


@break_after(hours=95, minutes=55)
def download_long_read_evidence(*, with_multiprocessing=True):
    long_read_samples = pd.read_csv("long_reads/long_read_samples.csv")
    sample_sv_lookup = pd.read_csv("long_reads/sample_sv_lookup.csv")
    if with_multiprocessing:
        with multiprocessing.Manager():
            cpu_count = multiprocessing.cpu_count()
            p = multiprocessing.Pool(cpu_count)
            args = []
            for _, row in long_read_samples.iterrows():
                sv_ids = sample_sv_lookup[
                    sample_sv_lookup["sample_id"] == row["sample_id"]
                ]["sv_id"].values
                if len(sv_ids) == 0:
                    print("No SVs found for sample", row["sample_id"])
                    continue
                args.append((row.to_dict()), sv_ids)
            p.starmap(download_sample_evidence, args)
            p.close()
            p.join()
    else:
        for _, row in long_read_samples.iterrows():
            sv_ids = sample_sv_lookup[
                sample_sv_lookup["sample_id"] == row["sample_id"]
            ]["sv_id"].values
            if len(sv_ids) == 0:
                print("No SVs found for sample", row["sample_id"])
                continue
            download_sample_evidence(row.to_dict(), sv_ids)


if __name__ == "__main__":
    download_long_read_evidence(with_multiprocessing=True)
