import os
import pandas as pd
import csv
import random
import ast
from viz import run_viz_gmm
from collections import defaultdict
from dataclasses import fields
from gmm_types import *
from query_sv import giggle_format, query_stix


def find_missing_sample_ids():
    sample_ids = set()
    for file in os.listdir("processed_stix_output"):
        with open(f"processed_stix_output/{file}") as f:
            for line in f:
                sample_id = line.strip().split(",")[0]
                if sample_id[0].isalpha():
                    sample_ids.add(sample_id)

    # print(len(sample_ids))  # 2535
    deletions_df = pd.read_csv("1000genomes/deletions_df.csv", low_memory=False)
    missing = sample_ids - set(deletions_df.columns[11:-1])

    # print(len(missing))  # 31
    # print(missing)
    # {'HG00702', 'NA20898', 'HG00124', 'NA20336', 'NA19675', 'HG03715', 'HG02024', 'NA19311', 'NA19685', 'HG02363', 'NA20871', 'NA20322', 'HG00501', 'nan', 'NA19240', 'HG01983', 'NA19985', 'HG02381', 'HG02388', 'NA20341', 'HG02377', 'NA20526', 'HG00635', 'NA19313', 'HG02387', 'NA19660', 'HG00733', 'NA20893', 'HG03948', 'HG02372', 'HG02046', 'NA20344'}

    return missing


def find_missing_processed_svs():
    processed_sv_ids = set([file.strip(".csv") for file in os.listdir("processed_svs")])
    deletions_df = pd.read_csv("1000genomes/deletions_df.csv", low_memory=False)
    missing = set(deletions_df["id"]) - processed_sv_ids
    print(len(missing))
    print(missing)
    return missing


def concat_processed_sv_files():
    with open("1000genomes/sv_stats.csv", mode="w", newline="") as out:
        fieldnames = [field.name for field in fields(SVInfoGMM)]
        csv_writer = csv.DictWriter(out, fieldnames=fieldnames)
        csv_writer.writeheader()
        for file in os.listdir("processed_svs"):
            with open(f"processed_svs/{file}") as f:
                for line in f:
                    out.write(line)


def query_random_svs(num_sample_range):
    df = pd.read_csv("1000genomes/deletions_df.csv", low_memory=False)
    df = df[df.num_samples >= num_sample_range[0]]
    df = df[df.num_samples <= num_sample_range[1]]
    for _ in range(50):
        row = df.sample()
        num_samples = row.num_samples.values[0]
        chr = str(row.chr.values[0])
        start = row.start.values[0]
        stop = row.stop.values[0]
        af = round(ast.literal_eval(row.af.values[0])[0], 3)
        print(
            f"Chr {chr}: {start}-{stop} ({num_samples} samples, allele frequency={af})"
        )

        l = giggle_format(chr, start)
        r = giggle_format(chr, stop)
        query_stix(l, r)


def generate_weights(num_svs: int):
    if num_svs == 1:
        return [1.0]

    elif num_svs == 2:
        p1 = random.uniform(0.05, 0.95)
        return [p1, 1 - p1]

    elif num_svs == 3:
        while True:
            p1 = random.uniform(0.05, 0.95)
            p2 = random.uniform(0.05, 0.95)
            if p1 + p2 <= 0.95:
                return [p1, p2, 1 - p1 - p2]


def assign_modes(weights, samples):
    num_samples = len(samples)
    assigned = [round(w * num_samples) for w in weights]
    if sum(assigned) != num_samples:
        assigned[0] += num_samples - sum(assigned)
    modes = []
    for i, num_samples in enumerate(assigned):
        modes.extend([i] * num_samples)
    return modes


"""
Generates synthetic SV data for testing purposes
chr: chromosome number (does not support X/Y), as a str
svs: List of (start, stop) for each SV
"""


def generate_synthetic_sv_data(chr: int, svs: List[Tuple[int, int]]):
    num_svs = len(svs)

    # Decide how many samples we want in our population
    num_samples = random.randint(30, 1000)
    samples = [f"sample_{i}" for i in range(num_samples)]

    # Decide how we want to divide the samples between the SVs
    weights = generate_weights(num_svs)
    modes = assign_modes(weights, samples)

    print(
        f"{num_samples} total samples, {[modes.count(i) for i in range(num_svs)]} samples per mode"
    )

    # For each sample, generate random evidence
    evidence = defaultdict(list)
    for sample, mode in zip(samples, modes):
        num_evidence = random.randint(3, 20)
        mode_start, mode_end = svs[mode]
        for _ in range(num_evidence):
            read_length = int(random.gauss(450, 50))
            # split = random.randint(1, read_length - 1) # if the read is too close to the start/end of the SV, it gets filtered out
            split = random.randint(
                int(read_length / 2) - 100, int(read_length / 2) + 100
            )
            evidence_start = mode_start - split
            evidence_stop = mode_end + split
            evidence[sample].extend([evidence_start, evidence_stop])

    # Pass synthetic data through SV analysis pipeline
    L = min([start for start, _ in svs])
    R = max([stop for _, stop in svs])
    evidence = {key: np.array(value) for key, value in evidence.items()}
    gmm, evidence_by_mode = run_viz_gmm(
        evidence,
        file_name=None,
        chr=str(chr),
        L=L,
        R=R,
        plot=False,
        plot_bokeh=False,
        synthetic_data=True,
    )

    return gmm, evidence_by_mode


if __name__ == "__main__":
    concat_processed_sv_files()
