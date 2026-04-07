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
from query_sv import giggle_format
from helper import stix_output_to_df
from timeout import break_after
from typing import List

SCRATCH_DIR = "/scratch/Shares/layer/1kg_lr_crams"
STIX_DATA_DIR = "/Users/vili4418/sv/sv_gmm/data_dump/lr_stix_output/"


def write_completed_sample(sample_id: str):
    """Writes the completed sample id to the completed samples file."""
    file = "long_reads/completed_samples.txt"
    with open(file, "a") as f:
        f.write(f"{sample_id}\n")


def get_svs_by_sample():
    """Creates a lookup table of samples and the SVs that are 0/1, 1/0, or 1/1. Standalone function."""
    df = pd.read_csv("long_reads/long_read_samples.csv")
    deletions_df = pd.read_csv("1kgp/deletions.csv")

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


def get_sample_sv_reads():
    """
    Creates a lookup table of samples and the SVs that have long read evidence from stix output.
    WARNING: This function takes a long time to run. This lookup is done when processing each sample's evidence.
    """
    lookup = pd.read_csv("long_reads/sample_sv_lookup.csv")
    dir = "data_dump/lr_stix_output/"
    files = os.listdir(dir)  # one file for each sv
    df = pd.DataFrame(
        columns=[
            "sample_id",
            "sv_id",
            "chr",
            "l_start",
            "l_end",
            "r_start",
            "r_end",
        ]
    )

    for i, file in enumerate(files):
        print(f"File {i}/{len(files)}", end="\r")
        sys.stdout.flush()

        stix_output = stix_output_to_df(os.path.join(dir, file), True)
        sv_id = file.strip(".txt")
        for _, row in stix_output.iterrows():
            sample_id = row["sample_id"]

            # check that lookup has this sample_id, sv_id pair
            if (
                lookup[
                    (lookup["sample_id"] == sample_id)
                    & (lookup["sv_id"] == sv_id)
                ].shape[0]
                == 0
            ):
                continue

            # check that this row does not exist already in the df
            if (
                df[
                    (df["sample_id"] == sample_id)
                    & (df["sv_id"] == sv_id)
                    & (df["l_start"] == row["l_start"])
                    & (df["l_end"] == row["l_end"])
                ].shape[0]
                > 0
            ):
                continue

            df.loc[len(df)] = [
                sample_id,
                sv_id,
                row["l_chr"],
                row["l_start"],
                row["l_end"],
                row["r_start"],
                row["r_end"],
            ]

    df.to_csv("long_reads/sample_sv_reads.csv", index=False)


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
):
    """
    For each sv id for a given sample, download the bam file for the sv region, extract cigar strings, and write to evidence file. Processes all of the stix reads for one sample/sv id pair at a time to avoid writing to the same file in parallel.
    cram_file can be either the downloaded cram file path or an http link to the cram file.
    """
    # use the stix read coordinates to subset the bam file
    _, sv_chr, sv_start, sv_stop, _ = get_sv_region(sv_id, 0)

    # get the reads for this sample and sv id
    stix_output_file = os.path.join(
        STIX_DATA_DIR,
        f"{giggle_format(sv_chr, sv_start)}_{giggle_format(sv_chr, sv_stop)}.txt",
    )
    stix_output = stix_output_to_df(stix_output_file, True)
    reads = stix_output[stix_output["sample_id"] == sample_id]

    deletions = [sample_id]
    seen_reads = set()
    for _, row in reads.iterrows():
        read_start, read_stop = row["l_start"], row["l_end"]
        # read start is used to identify unique reads
        if read_start in seen_reads:
            continue
        bam_region = f"chr{sv_chr}:{read_start}-{read_stop}"
        output_file = get_bam_file(
            sv_id,
            sample_id,
            region=bam_region,
            cram_file=cram_file,
            scratch=True,
        )

        try:
            # this should only output one or 0 deletions that matches the stix read coords
            evidence = read_cigars_from_file(
                output_file, (sv_start, sv_stop), (read_start, read_stop)
            )
        except Exception as e:
            print(
                f"Redo sample {sv_id}-{sample_id}, read coords {(read_start, read_stop)}. Error={e}"
            )
            remove_bam_file(output_file, scratch=True)
            continue

        if len(evidence) > 0:
            deletions.extend([evidence[0]["start"], evidence[0]["stop"]])

        seen_reads.add(read_start)
        remove_bam_file(output_file, scratch=True)

    file_name = os.path.join(SCRATCH_DIR, f"long_reads/evidence/{sv_id}.csv")
    if len(deletions) > 1:  # found evidence for at least one read
        write_to_evidence_file(file_name, deletions)


def process_sample_evidence(
    sample_id: str,
    cram_file: str,
    sv_ids: List[str],
    with_mp: bool = False,
):
    """Parallelizes the processing function for each sv in sv_ids after the cram file has been downloaded for a sample."""
    if with_mp:
        with mp.Manager():
            cpu_count = mp.cpu_count()
            pool = mp.Pool(cpu_count)
            args = []
            for sv_id in sv_ids:
                args.append((sv_id, sample_id, cram_file))
            pool.starmap(process_sample_evidence_inner, args)
            pool.close()
            pool.join()
    else:
        for sv_id in sv_ids:
            process_sample_evidence_inner(sv_id, sample_id, cram_file)


def remove_cram_file(file):
    os.remove(file)
    os.remove(f"{file}.crai")


def download_sample_evidence(
    sample_row: pd.Series,
    sv_ids: List[str],
):
    """Download the cram file for a sample, process it for each sv, and then remove the cram file."""
    start = time.time()

    sample_id = sample_row["sample_id"]
    print(f"Processing sample {sample_id}", flush=True)

    # download the cram file and the indexed cram file
    output_file = os.path.join(
        SCRATCH_DIR, sample_row["cram_file"].split("/")[-1]
    )
    if not os.path.exists(output_file):
        download_start = time.time()
        subprocess.run(
            ["wget", "-O", output_file, sample_row["cram_file"]],
            capture_output=True,
            text=True,
        )
        download_end = time.time()
        print(
            f"Downloaded cram file for {sample_id} in ",
            int((download_end - download_start) / 60),
            "minutes",
            flush=True,
        )
    indexed_file = f"{output_file}.crai"
    if not os.path.exists(indexed_file):
        subprocess.run(
            [
                "wget",
                "-O",
                indexed_file,
                sample_row["indexed_cram_file"],
            ],
            capture_output=True,
            text=True,
        )

    # parallelized function to process the sample evidence for each sv
    process_sample_evidence(sample_id, output_file, sv_ids, with_mp=True)

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
    write_completed_sample(sample_id)


def download_sample_evidence_http(
    sample_row: pd.Series,
    sv_ids: List[str],
):
    """Uses samtools to intersect the bam file from the https link. Takes much longer than downloading the cram file for each sample since fewer processes can run in parallel."""
    start = time.time()

    # parallelize the download and processing of each sv for the sample, but only 4 at a time
    with mp.Manager():
        pool = mp.Pool(4)
        args = []
        for sv_id in sv_ids:
            args.append(
                (sv_id, sample_row["sample_id"], sample_row["cram_file"])
            )
        pool.starmap(process_sample_evidence_inner, args)
        pool.close()
        pool.join()

    end = time.time()
    print(
        "Finished processing sample",
        sample_row["sample_id"],
        "in ",
        int((end - start) / 60),
        "minutes",
    )
    sys.stdout.flush()


@break_after(hours=335, minutes=30)  # takes about 14 days to run all SVs
def download_sv_subset():
    """Download long read evidence for all SVs."""
    sample_sv_lookup = pd.read_csv("long_reads/sample_sv_lookup.csv")
    sample_ids = sample_sv_lookup["sample_id"].unique()
    sv_ids = sample_sv_lookup["sv_id"].unique()

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


@break_after(hours=46, minutes=0)
def download_long_read_evidence(
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


def download_long_read_evidence_wrapper(
    *,
    move_files: bool = True,
    redo_samples: bool = False,
):
    """Download long read evidence (cram -> bam -> cigar string) for all samples and SVs in the lookup table. Moves evidence files between home directory and scratch to check for previous progress and speed up I/O on fiji."""

    # only process SVs that have at least one long read sample, and only process samples that have short-read evidence
    # it would take too long to process all svs for all samples
    sample_sv_lookup = pd.read_csv("long_reads/sample_sv_lookup.csv")
    long_read_samples = pd.read_csv("long_reads/long_read_samples.csv")
    completed_samples = get_completed_samples()

    long_read_samples = long_read_samples[
        ~long_read_samples["sample_id"].isin(completed_samples)
    ]

    # sv ids to process
    all_sv_ids = set(sample_sv_lookup["sv_id"].unique())

    # move sv files to scratch dir for faster I/O
    if move_files:
        move_evidence_files(all_sv_ids, to_scratch=True)

    # run the multiprocessing code
    download_long_read_evidence(
        long_read_samples, sample_sv_lookup, redo_samples
    )

    # move sv files back to home dir
    if move_files:
        move_evidence_files(all_sv_ids, to_scratch=False)

    # flush the buffered output
    sys.stdout.flush()


def download_long_read_cram(sample_row: pd.Series):
    """Download the cram file for a sample and store it in scratch."""
    sample_id = sample_row["sample_id"]
    print(f"Processing sample {sample_id}", flush=True)

    # download the cram file and the indexed cram file
    output_file = os.path.join(
        SCRATCH_DIR, sample_row["cram_file"].split("/")[-1]
    )
    if not os.path.exists(output_file):
        download_start = time.time()
        subprocess.run(
            ["wget", "-O", output_file, sample_row["cram_file"]],
            capture_output=True,
            text=True,
        )
        download_end = time.time()
        print(
            f"Downloaded cram file for {sample_id} in ",
            int((download_end - download_start) / 60),
            "minutes",
            flush=True,
        )
    indexed_file = f"{output_file}.crai"
    if not os.path.exists(indexed_file):
        subprocess.run(
            [
                "wget",
                "-O",
                indexed_file,
                sample_row["indexed_cram_file"],
            ],
            capture_output=True,
            text=True,
        )


def download_long_read_crams_all():
    """
    Downloads all long read cram files.
    Takes just over 48 hours when using four cores.
    """
    long_read_samples = pd.read_csv("long_reads/long_read_samples.csv")
    files = os.listdir(SCRATCH_DIR)
    completed_samples = set([file.split(".")[0] for file in files])
    long_read_samples = long_read_samples[
        ~long_read_samples["sample_id"].isin(completed_samples)
    ]

    with mp.Manager():
        pool = mp.Pool(4)
        args = []
        for _, sample_row in long_read_samples.iterrows():
            args.append((sample_row,))
        pool.starmap(download_long_read_cram, args)
        pool.close()
        pool.join()


if __name__ == "__main__":
    # for testing purposes
    # process_sample_evidence_inner("HGSV_204941", "NA19657", "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1KG_ONT_VIENNA/hg38/NA19657.hg38.cram")

    start = time.time()

    download_long_read_crams_all()
    # download_long_read_evidence_wrapper(move_files=True, redo_samples=False)

    end = time.time()
    print("Time taken:", (end - start) / 60, "minutes")
