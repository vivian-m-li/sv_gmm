import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from process_data import run_viz_gmm
from collections import defaultdict
from typing import List, Tuple, Optional

"""
Data generation functions
"""


def generate_weights(num_svs: int):
    """Generates random weights (0.5 <= p <= 0.95) for each SV mode."""
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
    """Assigns the samples to modes depending on their weights."""
    num_samples = len(samples)
    assigned = [round(w * num_samples) for w in weights]
    if sum(assigned) != num_samples:
        assigned[0] += num_samples - sum(assigned)
    modes = []
    for i, num_samples in enumerate(assigned):
        modes.extend([i] * num_samples)
    return modes


def get_random_insert_size(df):
    """Gets a random insert size from the 1kgp insert size distribution."""
    return df.sample().insert_size.values[0]


def generate_synthetic_sv_data(
    chr: int,  # chromosome number (does not support X/Y), as a str
    svs: List[Tuple[int, int]],  # List of (start, stop) for each SV
    *,
    n_samples: Optional[int] = None,
    p: Optional[List[float]] = None,
    gmm_model: str = "2d",
    run_gmm: bool = True,
    plot: bool = False,
    plot_reads: bool = False,
    write_data: bool = False,
):
    """Generates synthetic SV data for testing purposes and runs the data through the SV analysis pipeline."""
    num_svs = len(svs)

    # Decide how many samples we want in our population
    num_samples = random.randint(30, 1000) if n_samples is None else n_samples
    samples = [f"sample_{i}" for i in range(num_samples)]

    # Decide how we want to divide the samples between the SVs
    weights = generate_weights(num_svs) if p is None else p
    modes = assign_modes(weights, samples)

    # print(
    #     f"{num_samples} total samples, {[modes.count(i) for i in range(num_svs)]} samples per mode"
    # )

    insert_size_df = pd.read_csv(
        "1kgp/insert_sizes.csv", dtype={"mean_insert_size": int}
    )

    # For each sample, generate random evidence
    evidence = defaultdict(list)
    insert_size_lookup = {}
    for sample, mode in zip(samples, modes):
        num_evidence = random.randint(2, 30)
        mode_start, mode_end = svs[mode]
        insert_size = insert_size_df.sample()["mean_insert_size"].values[0]
        insert_size_lookup[sample] = insert_size

        for _ in range(num_evidence):
            read_length = min(550, max(350, int(random.gauss(insert_size, 25))))
            # split = random.randint(1, read_length - 1) # the split can be anywhere
            split = random.randint(
                int(read_length / 2) - 100, int(read_length / 2) + 100
            )  # to prevent the read from being filtered out (too far from the y=x line)
            evidence_start = mode_start - split
            evidence_stop = mode_end + (read_length - split)
            evidence[sample].extend([evidence_start, evidence_stop])

    # Pass synthetic data through SV analysis pipeline
    L = np.mean([start for start, _ in svs])
    R = np.mean([stop for _, stop in svs])
    evidence = {key: np.array(value) for key, value in evidence.items()}

    if plot_reads:
        plt.figure()
        for reads in evidence.values():
            plt.scatter(
                reads[::2],
                reads[1::2],
                color="blue",
                alpha=0.6,
            )
        plt.xlabel("L")
        plt.ylabel("R")
        plt.show()

    if write_data:
        # TODO: write data in the vcf file format
        reads_df = pd.DataFrame(
            columns=["sample_id", "L", "R", "mean_insert_size"]
        )
        for sample_id, values in evidence.items():
            mean_insert_size = insert_size_lookup[sample_id]
            for read_L, read_R in zip(values[::2], values[1::2]):
                reads_df.loc[len(reads_df)] = [
                    sample_id,
                    read_L,
                    read_R,
                    mean_insert_size,
                ]
        reads_df.to_csv("synthetic_data/generated_data.csv", index=False)

    if run_gmm:
        gmm, evidence_by_mode = run_viz_gmm(
            evidence,
            file_name=None,
            chr=str(chr),
            L=L,
            R=R,
            plot=plot,
            plot_bokeh=False,
            synthetic_data=True,
            gmm_model=gmm_model,
            insert_size_lookup=insert_size_lookup,
        )
        return gmm, evidence_by_mode

    return None, []
