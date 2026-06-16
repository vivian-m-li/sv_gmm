import multiprocessing
import os
import shutil
import sys
import time

import pandas as pd

from paper.data_processing.download_long_read_evidence import (
    get_samples_to_redo,
)
from src.model.dirichlet import run_dirichlet
from src.utils.helper import get_deletions_df, stix_output_to_df
from src.utils.timeout import break_after
from src.utils.types import InsertSizeDistribution
from src.utils.write_sv_output import (
    init_sv_stat_row,
    write_sv_stats,
    write_posterior_distributions,
    concat_multi_processed_sv_files,
)

FILE_DIR = "long_reads/processed_svs_converge"
SCRATCH_FILE_DIR = os.path.join("/scratch/Users/vili4418", FILE_DIR)
OUTPUT_FILE_NAME = "sv_stats_converge.csv"


def run_lr_dirichlet_wrapper(
    row: dict,
    population_size: int,
    sample_set: set[str],
    samples_to_skip: set[str] = set(),
):
    """Runs the dirichlet/GMM process for a single SV using long read data."""
    sv_id = row["id"]

    # don't need to remove insert size for long reads
    insert_size_lookup = {
        sample_id: InsertSizeDistribution(mean=0, sd=0)
        for sample_id in sample_set
    }

    # filter for sample IDs that have the SV allele (not homozygous reference)
    filtered_sample_ids = []
    sv_alleles = set(["(0, 1)", "(1, 0)", "(1, 1)"])
    for sample_id in sample_set:
        if row[sample_id] in sv_alleles:
            filtered_sample_ids.append(sample_id)

    # load deletions for the SV from the home dir, not scratch
    # refactor note: unsure where the downloaded long read evidence files are stored, might need to change this path
    reads = stix_output_to_df(f"long_reads/evidence/{sv_id}.csv")

    # remove samples to skip
    reads = reads[~reads["sample_id"].isin(samples_to_skip)]
    num_samples = reads["sample_id"].nunique()
    if reads.empty:
        # no samples were found with evidence for this SV
        gmm_results, alphas, posterior_distributions = [(None, [])], [], []
    else:
        # run the dirichlet process
        gmm_results, alphas, posterior_distributions = run_dirichlet(
            reads,
            **{
                "chr": row["chr"],
                "L": row["start"],
                "R": row["stop"],
                "insert_size_lookup": insert_size_lookup,
                "min_pairs": 1,  # reduce number of points required per sample - long read data is sparse and "more accurate"
                "plot": False,
            },
        )

    # write the output files to scratch first
    for i, (gmm_result, evidence_by_mode) in enumerate(gmm_results):
        sv_stat = init_sv_stat_row(
            row,
            num_samples=num_samples,
            num_reference=0,  # these are pre-filtered out when we download data
        )
        write_sv_stats(
            sv_stat,
            gmm_result,
            evidence_by_mode,
            population_size,
            SCRATCH_FILE_DIR,
            i,
        )

    if gmm_results[0][0] is not None:
        write_posterior_distributions(
            sv_id, alphas, posterior_distributions, SCRATCH_FILE_DIR
        )

    print(f"Completed running {sv_id}")
    sys.stdout.flush()


@break_after(hours=22, minutes=0)
def run_svs_until_convergence(with_multiprocessing: bool, use_subset: bool):
    """Parallelized wrapper to cluster SVs using long read data."""
    if use_subset:
        deletions_df = pd.read_csv("1kg/deletions_df_subset.csv")
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
    """Cluster SVs using long read data."""
    start = time.time()
    run_svs_until_convergence(with_multiprocessing, use_subset)

    # move files from scratch to home dir (even after timeout)
    for file in os.listdir(SCRATCH_FILE_DIR):
        shutil.move(
            os.path.join(SCRATCH_FILE_DIR, file),
            os.path.join(FILE_DIR, file),
        )

    concat_multi_processed_sv_files("long_reads", FILE_DIR, OUTPUT_FILE_NAME)

    end = time.time()
    print(f"Completed in {end - start}")


if __name__ == "__main__":
    run_svs(with_multiprocessing=True, use_subset=False)
