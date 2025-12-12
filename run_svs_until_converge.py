import time
import os
import shutil
import multiprocessing
from write_sv_output import (
    get_raw_data,
    init_sv_stat_row,
    write_sv_stats,
    write_posterior_distributions,
    concat_multi_processed_sv_files,
)
from run_dirichlet import run_dirichlet
from helper import get_deletions_df, get_sample_ids, write_post_processed_files
from typing import Dict
from timeout import break_after


FILE_DIR = "processed_svs_converge"
SCRATCH_FILE_DIR = os.path.join("/scratch/Users/vili4418", FILE_DIR)
OUTPUT_FILE_NAME = "sv_stats_converge.csv"


def run_dirichlet_inner(
    row: Dict,
    population_size: int,
    stem: str,
):
    sv_id = row["id"]
    reads, num_samples = get_raw_data(row, stem)

    if reads.empty:
        # if no samples have any data supporting the SV then not included
        gmms, alphas, posterior_distributions = [(None, [])], [], []
    else:
        gmms, alphas, posterior_distributions = run_dirichlet(
            reads,
            **{
                "chr": row["chr"],
                "L": row["start"],
                "R": row["stop"],
                "plot": False,
                "stem": stem,
            },
        )

    for i, (gmm, evidence_by_mode) in enumerate(gmms):
        sv_stat = init_sv_stat_row(
            row,
            num_samples=num_samples,
            num_reference=num_samples - reads["sample_id"].nunique(),
        )
        write_sv_stats(
            sv_stat, gmm, evidence_by_mode, population_size, SCRATCH_FILE_DIR, i
        )

    if gmms[0][0] is not None:
        write_posterior_distributions(
            sv_id, alphas, posterior_distributions, SCRATCH_FILE_DIR
        )


def run_dirichlet_wrapper(
    row: Dict,
    population_size: int,
    stem: str,
):
    try:
        run_dirichlet_inner(row, population_size, stem)
    except Exception as e:
        print(f"Error processing SV {row['id']}: {e}")


@break_after(hours=64, minutes=00)
def run_svs_until_convergence(stem: str):
    deletions_df = get_deletions_df(stem).head(50)
    sample_ids = set(get_sample_ids(stem))
    population_size = len(sample_ids)

    with multiprocessing.Manager():
        cpu_count = multiprocessing.cpu_count()
        p = multiprocessing.Pool(cpu_count)
        args = []
        for _, row in deletions_df.iterrows():
            args.append((row.to_dict(), population_size, stem))
        p.starmap(run_dirichlet_wrapper, args)
        p.close()
        p.join()


def run_svs(*, input_dir: str = "1kgp", output_dir: str = "results"):
    start = time.time()
    run_svs_until_convergence(input_dir)

    # move files from scratch to home dir (even after timeout)
    for file in os.listdir(SCRATCH_FILE_DIR):
        shutil.move(
            os.path.join(SCRATCH_FILE_DIR, file), os.path.join(FILE_DIR, file)
        )

    print("Concatenating multi-processed SV files...")
    concat_multi_processed_sv_files(FILE_DIR, OUTPUT_FILE_NAME, output_dir)
    print("Writing post-processed files...")
    write_post_processed_files(input_dir, output_dir)
    end = time.time()
    print(f"Completed in {end - start}")


if __name__ == "__main__":
    run_svs(input_dir="1kgp", output_dir="results")
