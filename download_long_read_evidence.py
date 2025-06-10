import sys
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
    write_samples_to_redo,
)
from timeout import break_after
from typing import List, Optional

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


def write_samples_to_redo():
    # read all samples that need to be redone
    # run this in the sv directory in fiji
    sample_ids = set()
    for file in os.listdir("eofiles"):
        if not file.endswith(".out"):
            continue
        with open(os.path.join("eofiles", file), "r") as f:
            for line in f.readlines():
                pattern = f"Redo sample [\S]+-([\S]+)"
                match = re.search(pattern, line)
                if match:
                    sample_id = match.group(1)
                    sample_ids.add(sample_id)
    with open("/Users/vili4418/sv/sv_gmm/long_reads/redo_samples.txt", "w") as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")



def get_samples_to_redo()
    file = "long_reads/redo_samples.txt"
    samples = {}
    with open(file, "r") as f:
        for line in f.readlines():
            sample_id, sv_ids = line.strip().split(":")
            sv_ids = sv_ids.split(",")
            sv_ids = [sv_id.strip() for sv_id in sv_ids]
            sv_ids = [sv_id for sv_id in sv_ids if sv_id != ""]
            if len(sv_ids) == 0:
                continue
            samples[sample_id] = sv_ids
    return samples
    

def get_completed_samples():
    file = "long_reads/completed_samples.txt"
    samples = []
    with open(file, "r") as f:
        for line in f.readlines():
            sample_id = line.strip()
            if sample_id == "":
                continue
            samples.append(sample_id)
    return samples


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


def write_to_evidence_file(file_name, row):
    if os.path.exists(file_name):  # append to existing file
        f = open(file_name, "a")
    else:  # file does not exist
        f = open(file_name, "w")
    csv_writer = csv.writer(f)
    csv_writer.writerow(row)
    f.close()


def process_sample_evidence(
    sample_id: str, cram_file: str, sv_ids: List[str], queue: Optional[mp.Queue]
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

        try:
            evidence = read_cigars_from_file(output_file, sv_len)
        except Exception:
            print(f"Redo sample {sv_id}-{sample_id}")
            continue

        row = [sample_id]
        for deletion in evidence:
            row.extend([deletion["start"], deletion["stop"]])

        # put the csv row in the queue to be written
        file_name = os.path.join(
            SCRATCH_DIR, f"long_reads/evidence/{sv_id}.csv"
        )

        if queue is None:
            write_to_evidence_file(file_name, row)
        else:
            queue.put((file_name, row))

        remove_bam_file(output_file, scratch=True)


def remove_cram_file(file):
    os.remove(file)
    os.remove(f"{file}.crai")


def download_sample_evidence(
    sample_row: pd.Series, sv_ids: List[str], queue: Optional[mp.Queue] = None
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
    sys.stdout.flush()


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
            if os.path.exists(file_name):  # append to existing file
                f = open(file_name, "a")
            else:  # file does not exist
                f = open(file_name, "w")
            csv_writer = csv.writer(f)
            csv_writer.writerow(row)
            f.close()
    except Exception as e:
        print(f"Listener error: {e}")
        return


@break_after(hours=82, minutes=0)  # leave time to move files
def download_long_read_evidence_inner(
    long_read_samples,
    sample_sv_lookup,
    redo_samples,
):
    if redo_samples:
        samples_to_redo = get_samples_to_redo()
        long_read_samples = long_read_samples[
            long_read_samples["sample_id"].isin(samples_to_redo.keys())
        ]

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

            if redo_samples:
                svs_to_process = set(samples_to_redo[row["sample_id"]])
            else:
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


@break_after(hours=98, minutes=0) 
def download_long_read_evidence_synchronous(
    long_read_samples,
    sample_sv_lookup,
    redo_samples,
):
    if redo_samples:
        samples_to_redo = get_samples_to_redo()
        long_read_samples = long_read_samples[
            long_read_samples["sample_id"].isin(samples_to_redo.keys())
        ]

    for _, row in long_read_samples.iterrows():
        sv_ids = sample_sv_lookup[
            sample_sv_lookup["sample_id"] == row["sample_id"]
        ]["sv_id"].values
        svs_to_process = []

        if redo_samples:
            svs_to_process = set(samples_to_redo[row["sample_id"]])
        else:
            for sv_id in sv_ids:
                if not is_sample_processed(sv_id, row["sample_id"]):
                    svs_to_process.append(sv_id)

        if len(svs_to_process) == 0:
            print(
                f"Sample {row['sample_id']} already processed for all SVs"
            )
            continue

        download_sample_evidence(row, svs_to_process)


def download_long_read_evidence(
    *,
    move_files: bool = True,
    redo_samples: bool = False,
):
    sample_sv_lookup = pd.read_csv("long_reads/sample_sv_lookup.csv")
    long_read_samples = pd.read_csv("long_reads/long_read_samples.csv")

    completed_samples = get_completed_samples()
    long_read_samples = long_read_samples[
        ~long_read_samples["sample_id"].isin(completed_samples)
    ]

    all_sv_ids = set(sample_sv_lookup["sv_id"].unique())

    # move sv files to scratch dir for faster I/O
    if move_files:
        move_evidence_files(all_sv_ids, to_scratch=True)

    # run the multiprocessing code
    download_long_read_evidence_synchronous(
        long_read_samples, sample_sv_lookup, redo_samples
    )

    # move sv files back to home dir
    if move_files:
        move_evidence_files(all_sv_ids, to_scratch=False)

    # flush the buffered output
    sys.stdout.flush()
    write_samples_to_redo()


if __name__ == "__main__":
    start = time.time()

    download_long_read_evidence(move_files=True, redo_samples=False)

    end = time.time()
    print("Time taken:", (end - start) / 60, "minutes")
