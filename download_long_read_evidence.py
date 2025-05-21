import time
import os
import shutil
import subprocess
import csv
import pandas as pd
import multiprocessing as mp
from parse_long_reads import (
    get_sv_region,
    get_bam_file,
    read_cigars_from_file,
    remove_bam_file,
)
from timeout import break_after
from typing import List

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


def move_evidence_files(sv_ids, *, to_scratch: bool):
    for sv_id in sv_ids:
        file_name = f"long_reads/evidence/{sv_id}.csv"
        scratch_file = os.path.join(SCRATCH_DIR, file_name)
        if to_scratch:
            if not os.path.exists(file_name):
                continue
            shutil.move(file_name, scratch_file)
        else:
            if not os.path.exists(scratch_file):
                continue
            shutil.move(scratch_file, file_name)


def is_sample_processed(sv_id: str, sample_id: str) -> bool:
    file_name = os.path.join(SCRATCH_DIR, f"long_reads/evidence/{sv_id}.csv")
    try:
        with open(file_name, "r") as file:
            return sample_id in file.read()
    except Exception:
        return False


def process_sample_evidence(
    sample_id: str, cram_file: str, sv_ids: List[str], queue: mp.Queue
):
    for sv_id in sv_ids:
        region, sv_len = get_sv_region(sv_id, 300)
        output_file = get_bam_file(
            sv_id,
            sample_id,
            region=region,
            cram_file=cram_file,
            scratch=True,
        )

        evidence = read_cigars_from_file(output_file, sv_len)
        row = [sample_id]
        for deletion in evidence:
            row.extend([deletion["start"], deletion["stop"]])

        # put the csv row in the queue to be written
        file_name = os.path.join(
            SCRATCH_DIR, f"long_reads/evidence/{sv_id}.csv"
        )
        queue.put((file_name, row))

        remove_bam_file(output_file, scratch=True)


def remove_cram_file(file):
    os.remove(file)
    os.remove(f"{file}.crai")


def download_sample_evidence(
    sample_row: pd.Series, sv_ids: List[str], queue: mp.Queue
):
    sample_id = sample_row["sample_id"]
    print(f"Processing sample {sample_id}")

    # download the cram file and the indexed cram file
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

    # process the sample for each sv
    process_sample_evidence(sample_id, output_file, sv_ids, queue)

    # remove the cram file at the end
    remove_cram_file(output_file)

    print("Finished processing sample", sample_id)


def worker(sample_row: pd.Series, sv_ids: List[str], queue: mp.Queue):
    download_sample_evidence(sample_row, sv_ids, queue)


def listener(queue):
    """Listens for messages on the queue and writes to the file"""
    try:
        while True:
            m = queue.get()
            if m == "kill":
                break

            file_name, row = m
            if os.path.exists(file_name): # append to existing file
                f = open(file_name, "a")
            else: # file does not exist
                f = open(file_name, "w")        
            csv_writer = csv.writer(f)
            csv_writer.writerow(row)
            f.close()
    except Exception as e:
        print(f"Listener error: {e}")
        return


@break_after(hours=95, minutes=0)  # leave time to move files
def download_long_read_evidence_inner(long_read_samples, sample_sv_lookup):
    with mp.Manager() as manager:
        cpu_count = mp.cpu_count()
        pool = mp.Pool(cpu_count)
        queue = manager.Queue()

        # put listener to work
        pool.apply_async(listener, (queue,))

        # fire off jobs
        jobs = []
        start = time.time()
        for _, row in long_read_samples.iterrows():
            sv_ids = sample_sv_lookup[
                sample_sv_lookup["sample_id"] == row["sample_id"]
            ]["sv_id"].values
            svs_to_process = []

            for sv_id in sv_ids:
                if not is_sample_processed(sv_id, row["sample_id"]):
                    svs_to_process.append(sv_id)

            if len(svs_to_process) == 0:
                print(
                    f"Sample {row['sample_id']} already processed for all SVs"
                )
                continue

            job = pool.apply_async(worker, (row, svs_to_process, queue))
            jobs.append(job)

        print("Time to prepare jobs:", (time.time() - start) / 60, "minutes")

        for job in jobs:
            job.get()

        queue.put("kill")
        pool.close()
        pool.join()


def download_long_read_evidence():
    long_read_samples = pd.read_csv("long_reads/long_read_samples.csv")
    sample_sv_lookup = pd.read_csv("long_reads/sample_sv_lookup.csv")

    all_sv_ids = set(sample_sv_lookup["sv_id"].unique())

    # move sv files to scratch dir for faster I/O
    move_evidence_files(all_sv_ids, to_scratch=True)

    # run the multiprocessing code
    download_long_read_evidence_inner(long_read_samples, sample_sv_lookup)

    # move sv files back to home dir
    move_evidence_files(all_sv_ids, to_scratch=False)


if __name__ == "__main__":
    start = time.time()
    download_long_read_evidence()
    end = time.time()
    print("Time taken:", (end - start) / 60, "minutes")
