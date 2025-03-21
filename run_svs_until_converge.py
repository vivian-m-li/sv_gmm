import os
import shutil
import pandas as pd
import re
import multiprocessing
from write_sv_output import (
    get_raw_data,
    init_sv_stat_row,
    write_sv_stats,
    write_posterior_distributions,
    concat_multi_processed_sv_files,
)
from run_dirichlet import run_dirichlet
from helper import get_deletions_df, get_sample_ids
from typing import Set, Dict
from timeout import break_after


FILE_DIR = "processed_svs_converge"
SCRATCH_FILE_DIR = os.path.join("/scratch/Users/vili4418", FILE_DIR)
OUTPUT_FILE_NAME = "sv_stats_converge.csv"


def run_dirichlet_wrapper(
    row: Dict,
    population_size: int,
    sample_set: Set[int],
):
    sv_id = row["id"]
    squiggle_data, num_samples = get_raw_data(row, sample_set)

    if len(squiggle_data) == 0:
        gmms, alphas, posterior_distributions = [(None, [])], [], []
    else:
        gmms, alphas, posterior_distributions = run_dirichlet(
            squiggle_data,
            **{
                "file_name": None,
                "chr": row["chr"],
                "L": row["start"],
                "R": row["stop"],
                "plot": False,
                "plot_bokeh": False,
            },
        )

    for i, (gmm, evidence_by_mode) in enumerate(gmms):
        sv_stat = init_sv_stat_row(
            row,
            num_samples=num_samples,
            num_reference=num_samples - len(squiggle_data),
        )
        write_sv_stats(
            sv_stat, gmm, evidence_by_mode, population_size, SCRATCH_FILE_DIR, i
        )

    if gmms[0][0] is not None:
        write_posterior_distributions(
            sv_id, alphas, posterior_distributions, SCRATCH_FILE_DIR
        )

    print(sv_id)

@break_after(hours=23, minutes=50)
def run_svs_until_convergence():
    deletions_df = get_deletions_df()
    sample_ids = set(get_sample_ids())
    population_size = len(sample_ids)

    with multiprocessing.Manager():
        cpu_count = multiprocessing.cpu_count()
        p = multiprocessing.Pool(cpu_count)
        args = []
        for _, row in deletions_df.iterrows():
            args.append((row.to_dict(), population_size, sample_ids))
        p.starmap(run_dirichlet_wrapper, args)
        p.close()
        p.join()


def run_svs():
    run_svs_until_convergence()

    # move files from scratch to home dir (even after timeout)
    for file in os.listdir(SCRATCH_FILE_DIR):
        shutil.move(os.path.join(SCRATCH_FILE_DIR, file), os.path.join(FILE_DIR, file))

    concat_multi_processed_sv_files(FILE_DIR, OUTPUT_FILE_NAME)


if __name__ == "__main__":
    run_svs()
