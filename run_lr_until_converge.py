import time
import os
import sys
import shutil
import multiprocessing
import pandas as pd
from write_sv_output import (
    init_sv_stat_row,
    write_sv_stats,
    write_posterior_distributions,
    concat_multi_processed_sv_files,
)
from download_long_read_evidence import get_samples_to_redo
from query_sv import load_squiggle_data
from run_dirichlet import run_dirichlet
from helper import get_deletions_df
from typing import Set, Dict
from timeout import break_after


FILE_DIR = "long_reads/processed_svs_converge"
SCRATCH_FILE_DIR = os.path.join("/scratch/Users/vili4418", FILE_DIR)
OUTPUT_FILE_NAME = "sv_stats_converge.csv"


def run_lr_dirichlet_wrapper(
    row: Dict,
    population_size: int,
    sample_set: Set[str],
    samples_to_skip: Set[str] = set(),
):
    sv_id = row["id"]
    insert_size_lookup = {
        sample_id: 0 for sample_id in sample_set
    }  # don't need to remove insert size for long reads

    # filter for sample IDs that have the SV allele
    filtered_sample_ids = []
    sv_alleles = set(["(0, 1)", "(1, 0)", "(1, 1)"])
    for sample_id in sample_set:
        if row[sample_id] in sv_alleles:
            filtered_sample_ids.append(sample_id)

    # load deletions for the SV
    squiggle_data = load_squiggle_data(f"long_reads/evidence/{sv_id}.csv")

    # remove samples to skip
    for sample_id in samples_to_skip:
        if sample_id in squiggle_data:
            del squiggle_data[sample_id]

    num_samples = len(squiggle_data)

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
                "insert_size_lookup": insert_size_lookup,
                "min_pairs": 1,  # reduce number of points required per sample
            },
        )

    for i, (gmm, evidence_by_mode) in enumerate(gmms):
        sv_stat = init_sv_stat_row(
            row,
            num_samples=num_samples,
            num_reference=num_samples
            - len(
                squiggle_data
            ),  # this will be 0 since we are pre-filtering them out
        )
        write_sv_stats(
            sv_stat, gmm, evidence_by_mode, population_size, SCRATCH_FILE_DIR, i
        )

    if gmms[0][0] is not None:
        write_posterior_distributions(
            sv_id, alphas, posterior_distributions, SCRATCH_FILE_DIR
        )

    print(f"Completed running {sv_id}")
    sys.stdout.flush()


@break_after(hours=22, minutes=0)
def run_svs_until_convergence(with_multiprocessing, use_subset):
    if use_subset:
        deletions_df = pd.read_csv("1kgp/deletions_df_subset.csv")
    else:
        deletions_df = get_deletions_df()

    long_read_samples_df = pd.read_csv("long_reads/long_read_samples.csv")
    # skip these samples because they keep failing
    samples_to_skip = get_samples_to_redo()
    long_read_samples_df = long_read_samples_df[
        ~long_read_samples_df["sample_id"].isin(samples_to_skip)
    ]
    sample_ids = set(long_read_samples_df["sample_id"].tolist())
    population_size = len(sample_ids)

    if with_multiprocessing:
        with multiprocessing.Manager():
            cpu_count = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(cpu_count)
            args = []
            for _, row in deletions_df.iterrows():
                args.append(
                    (
                        row.to_dict(),
                        population_size,
                        sample_ids,
                        samples_to_skip,
                    )
                )
            pool.starmap(run_lr_dirichlet_wrapper, args)
            pool.close()
            pool.join()
    else:
        for _, row in deletions_df.iterrows():
            run_lr_dirichlet_wrapper(
                row.to_dict(), population_size, sample_ids, samples_to_skip
            )


def run_svs(*, with_multiprocessing: bool = True, use_subset: bool = False):
    start = time.time()
    run_svs_until_convergence(with_multiprocessing, use_subset)

    # move files from scratch to home dir (even after timeout)
    for file in os.listdir(SCRATCH_FILE_DIR):
        shutil.move(
            os.path.join(SCRATCH_FILE_DIR, file),
            os.path.join(FILE_DIR, file),
        )

    concat_multi_processed_sv_files(
        FILE_DIR, OUTPUT_FILE_NAME, stem="long_reads"
    )

    end = time.time()
    print(f"Completed in {end - start}")


if __name__ == "__main__":
    run_svs(with_multiprocessing=True, use_subset=False)
