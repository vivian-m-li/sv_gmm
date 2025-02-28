import pandas as pd
import multiprocessing
from process_data import *
from gmm_types import *
from write_sv_output import *
from run_dirichlet import run_dirichlet
from helper import get_deletions_df, get_sample_ids
from typing import Set


FILE_DIR = "processed_svs_converge"
OUTPUT_FILE_NAME = "sv_stats_converge.csv"


def run_dirichlet_wrapper(row: Dict, population_size: int, sample_set: Set[int]):
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
            row, num_samples=num_samples, num_reference=num_samples - len(squiggle_data)
        )
        write_sv_stats(sv_stat, gmm, evidence_by_mode, population_size, FILE_DIR, i)

    if gmms[0][0] is not None:
        write_posterior_distributions(sv_id, alphas, posterior_distributions, FILE_DIR)

    print(sv_id)


def run_svs_until_convergence():
    deletions_df = get_deletions_df()

    sample_ids = set(get_sample_ids())
    population_size = len(sample_ids)

    rows = []
    for _, row in deletions_df.iterrows():
        rows.append(row)

    with multiprocessing.Manager() as manager:
        cpu_count = multiprocessing.cpu_count()
        p = multiprocessing.Pool(cpu_count)
        args = []
        for row in rows:
            args.append((row.to_dict(), population_size, sample_ids))
        p.starmap(run_dirichlet_wrapper, args)
        p.close()
        p.join()

    concat_multi_processed_sv_files(FILE_DIR, OUTPUT_FILE_NAME)


if __name__ == "__main__":
    run_svs_until_convergence()
