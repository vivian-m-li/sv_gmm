import time
import random
import os
import shutil
import subprocess
import pandas as pd
import multiprocessing as mp
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
            sv_id, [sample_id], cram_file=cram_file, tolerance=300, scratch=True
        )


def remove_cram_file(file):
    os.remove(file)
    os.remove(f"{file}.crai")


def download_sample_evidence(sample_row, sv_ids, queue):
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


def worker(args, queue):
    sample_row, sv_ids = args
    download_sample_evidence(sample_row, sv_ids, queue)


def listener(queue, file):
    """Listens for messages on the queue and writes to the file"""
    with open(file, "a") as f:
        while True:
            m = queue.get()
            if m == "kill":
                break
            f.write(m + "\n")
            f.flush()


@break_after(hours=15, minutes=55)
def download_long_read_evidence():
    long_read_samples = pd.read_csv("long_reads/long_read_samples.csv").head(160)
    sample_sv_lookup = pd.read_csv("long_reads/sample_sv_lookup.csv")
    with mp.Manager() as manager:
        cpu_count = mp.cpu_count()
        pool = mp.Pool(cpu_count)
        queue = manager.Queue()

        # put listener to work
        watcher = pool.apply_async(listener, (queue, ))

        # fire off jobs
        jobs = []
        all_sv_ids = set()
        for _, row in long_read_samples.iterrows():
            sv_ids = sample_sv_lookup[
                sample_sv_lookup["sample_id"] == row["sample_id"]
            ]["sv_id"].values
            all_sv_ids.update(sv_ids)
            if len(sv_ids) == 0:
                print("No SVs found for sample", row["sample_id"])
                continue
            job = pool.apply_async(worker, ((row.to_dict(), sv_ids), queue))
            jobs.append(job)
            # args.append((row.to_dict(), sv_ids))
        
        # move all sv files from home dir to scratch
        for sv_id in all_sv_ids:
            file_name = f"long_reads/evidence/{sv_id}.csv"
            if not os.path.exists(file_name):
                continue
            shutil.move(file_name, os.path.join(SCRATCH_DIR, file_name))

        for job in jobs:
            job.get()
            
        # pool.starmap(download_sample_evidence, args)
        queue.put("kill")
        pool.close()
        pool.join()

        # move sv files back to home dir
        for sv_id in all_sv_ids:
            file_name = f"long_reads/evidence/{sv_id}.csv"
            shutil.move(os.path.join(SCRATCH_DIR, file_name), file_name)


if __name__ == "__main__":
    start = time.time()
    download_long_read_evidence()
    end = time.time()
    print("Time taken:", (end - start) / 60, "minutes")
