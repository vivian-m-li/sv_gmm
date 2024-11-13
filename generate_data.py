import numpy as np
import pandas as pd
import random
import ast
from viz import run_viz_gmm
from query_sv import giggle_format, query_stix
from collections import defaultdict
from typing import List, Tuple

"""
Data generation functions
"""


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
        num_evidence = random.randint(2, 20)
        mode_start, mode_end = svs[mode]
        for _ in range(num_evidence):
            read_length = int(random.gauss(450, 50))
            # split = random.randint(1, read_length - 1) # if the read is too close to the start/end of the SV, it gets filtered out
            split = random.randint(
                int(read_length / 2) - 100, int(read_length / 2) + 100
            )
            evidence_start = mode_start - split
            evidence_stop = mode_end + (read_length - split)
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
        plot=True,
        plot_bokeh=False,
        synthetic_data=True,
    )