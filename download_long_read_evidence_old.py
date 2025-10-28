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
)
from query_sv import load_squiggle_data
from timeout import break_after
from typing import List, Optional

SCRATCH_DIR = "/scratch/Users/vili4418"


def find_failing_sv():
    """Finds the svs that are failing to be processed by attempting to load them with load_squiggle_data."""
    error_file = "long_reads/svs_to_redo.txt"
    with open(error_file, "w") as f:
        files = os.listdir("long_reads/evidence")
        for file in files:
            try:
                load_squiggle_data(
                    f"long_reads/evidence/{file}", rewrite_file=True
                )
            except ValueError as e:
                f.write(f"{file}: {e}\n")


def get_svs_by_sample():
    """Creates a lookup table of samples and the svs they don't have the reference allele for. Standalone function."""
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


def get_samples_to_redo():
    """Gets the samples that need to be redone."""
    file = "long_reads/redo_samples.txt"
    if not os.path.exists(file):
        return set()

    sample_ids = set()
    with open(file, "r") as f:
        for line in f.readlines():
            if line.strip() == "":
                continue
            sample_id = line.strip()
            sample_ids.add(sample_id)
    return sample_ids


def get_completed_samples():
    """Gets the samples that have finished processing."""
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
    """Moves files between home directory and scratch directory."""
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
    """Checks if the sample has been processed for an sv by looking for the sample id in the sv evidence file."""
    file_name = os.path.join(SCRATCH_DIR, f"long_reads/evidence/{sv_id}.csv")
    try:
        with open(file_name, "r") as file:
            return sample_id in file.read()
    except Exception:  # sv evidence file does not exist yet
        return False


def write_to_evidence_file(file_name, row):
    """Writes the new row (sample id and start/stop of the deletions) to the evidence file for the sv."""
    if os.path.exists(file_name):  # append to existing file
        f = open(file_name, "a")
    else:  # file does not exist
        f = open(file_name, "w")
    csv_writer = csv.writer(f)
    csv_writer.writerow(row)
    f.close()


def process_sample_evidence_inner(
    sv_id: str,
    sample_id: str,
    cram_file: str,
    queue: Optional[mp.Queue],
):
    """For each sv id for a given sample, download the bam file, extract cigar strings, and write to evidence file."""
    # add a tolerance of 500 bp on either side of the sv start/stop
    region, _, start, stop, sv_len = get_sv_region(sv_id, 500)
    output_file = get_bam_file(
        sv_id,
        sample_id,
        region=region,
        cram_file=cram_file,
        scratch=True,
    )

    try:
        evidence = read_cigars_from_file(output_file, start, stop, sv_len)
    except Exception:
        print(f"Redo sample {sv_id}-{sample_id}")
        return

    row = [sample_id]
    for deletion in evidence:
        row.extend([deletion["start"], deletion["stop"]])

    file_name = os.path.join(SCRATCH_DIR, f"long_reads/evidence/{sv_id}.csv")

    # not used in the synchronous version
    if queue is None:
        write_to_evidence_file(file_name, row)
    else:
        # put the csv row in the queue to be written
        queue.put((file_name, row))

    remove_bam_file(output_file, scratch=True)


def process_sample_evidence(
    sample_id: str,
    cram_file: str,
    sv_ids: List[str],
    queue: Optional[mp.Queue],
    with_mp: bool = False,
):
    """Parallelizes the processing function for each sv in sv_ids after the cram file has been downloaded for a sample."""
    if with_mp:
        with mp.Manager():
            cpu_count = mp.cpu_count()
            pool = mp.Pool(cpu_count)
            args = []
            for sv_id in sv_ids:
                args.append((sv_id, sample_id, cram_file, None))
            pool.starmap(process_sample_evidence_inner, args)
            pool.close()
            pool.join()
    else:
        for sv_id in sv_ids:
            process_sample_evidence_inner(sv_id, sample_id, cram_file, queue)


def remove_cram_file(file):
    os.remove(file)
    os.remove(f"{file}.crai")


def download_sample_evidence(
    sample_row: pd.Series, sv_ids: List[str], queue: Optional[mp.Queue] = None
):
    """Download the cram file for a sample, process it for each sv, and then remove the cram file."""
    start = time.time()

    sample_id = sample_row["sample_id"]
    print(f"Processing sample {sample_id}")

    # download the cram file and the indexed cram file
    output_file = os.path.join(
        SCRATCH_DIR, sample_row["cram_file"].split("/")[-1]
    )
    if not os.path.exists(sample_row["cram_file"]):
        download_start = time.time()
        subprocess.run(
            ["wget", "-O", output_file, sample_row["cram_file"]],
            capture_output=True,
            text=True,
        )
        download_end = time.time()
        print(
            "Downloaded cram file in ",
            int((download_end - download_start) / 60),
            "minutes",
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

    # parallelized function to process the sample evidence for each sv
    process_sample_evidence(sample_id, output_file, sv_ids, queue, with_mp=True)

    # remove the cram file at the end
    remove_cram_file(output_file)

    end = time.time()
    print(
        "Finished processing sample",
        sample_id,
        "in ",
        int((end - start) / 60),
        "minutes",
    )
    sys.stdout.flush()


@break_after(hours=335, minutes=30)  # takes about 14 days to run all SVs
def download_sv_subset():
    """Download long read evidence for a subset of svs that are failing."""
    sv_ids = set()
    with open("long_reads/svs_to_redo.txt", "r") as f:
        for line in f.readlines():
            if line.strip() == "":
                continue
            row = line.split(" ")
            sv_id = row[0].strip(".csv")
            sv_ids.add(sv_id)

    sample_sv_lookup = pd.read_csv("long_reads/sample_sv_lookup.csv")
    sample_ids = sample_sv_lookup[sample_sv_lookup["sv_id"].isin(sv_ids)][
        "sample_id"
    ].unique()

    long_read_samples = pd.read_csv("long_reads/long_read_samples.csv")
    long_read_samples = long_read_samples[
        long_read_samples["sample_id"].isin(sample_ids)
    ]

    for _, row in long_read_samples.iterrows():
        sample_sv_ids = sample_sv_lookup[
            (sample_sv_lookup["sample_id"] == row["sample_id"])
            & (sample_sv_lookup["sv_id"].isin(sv_ids))
        ]["sv_id"].values
        download_sample_evidence(row, sample_sv_ids)

    move_evidence_files(sv_ids, to_scratch=False)


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


def download_long_read_evidence_inner(
    long_read_samples,
    sample_sv_lookup,
    redo_samples,
):
    """
    Parallelized version of downloading long read evidence using multiprocessing.
    Requires a queue to write to the same evidence files from different processes.
    Don't use this function -- use the synchronous version instead.
    Downloading multiple cram files at the same time uses too many resources from the cluster.
    """
    if redo_samples:
        samples_to_redo = get_samples_to_redo()
        long_read_samples = long_read_samples[
            long_read_samples["sample_id"].isin(samples_to_redo)
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

            for sv_id in sv_ids:
                if redo_samples or not is_sample_processed(
                    sv_id, row["sample_id"]
                ):
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


@break_after(hours=70, minutes=0)
def download_long_read_evidence_synchronous(
    long_read_samples,
    sample_sv_lookup,
    redo_samples,
):
    """Download long read evidence (cram -> bam -> cigar string) for all samples and SVs in the lookup table. Uses multiprocessing for one sample at a time."""

    # if redo samples, only process these samples
    if redo_samples:
        samples_to_redo = get_samples_to_redo()
        long_read_samples = long_read_samples[
            long_read_samples["sample_id"].isin(samples_to_redo)
        ]

    # for each sample and its SVs, download sample evidence
    for _, row in long_read_samples.iterrows():
        sv_ids = sample_sv_lookup[
            sample_sv_lookup["sample_id"] == row["sample_id"]
        ]["sv_id"].values
        svs_to_process = []

        for sv_id in sv_ids:
            if redo_samples or not is_sample_processed(sv_id, row["sample_id"]):
                svs_to_process.append(sv_id)

        if len(svs_to_process) == 0:
            print(f"Sample {row['sample_id']} already processed for all SVs")
            continue

        download_sample_evidence(row, svs_to_process)


def download_long_read_evidence(
    *,
    move_files: bool = True,
    redo_samples: bool = False,
):
    """Download long read evidence (cram -> bam -> cigar string) for all samples and SVs in the lookup table. Moves evidence files between home directory and scratch to check for previous progress and speed up I/O on fiji."""

    # only process SVs that have at least one long read sample, and only process samples that have short-read evidence
    # it would take too long to process all svs for all samples
    sample_sv_lookup = pd.read_csv("long_reads/sample_sv_lookup.csv")
    long_read_samples = pd.read_csv("long_reads/long_read_samples.csv")

    # skip samples that have already been completed
    # completed_samples = get_completed_samples()
    # long_read_samples = long_read_samples[
    #     ~long_read_samples["sample_id"].isin(completed_samples)
    # ]

    # skip these samples because they keep failing
    # samples_to_redo = get_samples_to_redo()
    # long_read_samples = long_read_samples[
    #     ~long_read_samples["sample_id"].isin(samples_to_redo)
    # ]

    # sv ids to process
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


if __name__ == "__main__":
    start = time.time()

    download_long_read_evidence(move_files=True, redo_samples=False)

    end = time.time()
    print("Time taken:", (end - start) / 60, "minutes")
