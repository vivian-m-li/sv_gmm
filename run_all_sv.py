import sys
import os
import pandas as pd
import multiprocessing
from process_data import *
from gmm_types import *
from write_sv_output import *
from helper import get_deletions_df
from typing import Set

FILE_DIR = "processed_svs"
OUTPUT_FILE_NAME = "sv_stats.csv"


def run_all_sv_wrapper(
    row: Dict, population_size: int, sample_set: Set[int], iteration: int = 0
):
    squiggle_data, num_samples = get_raw_data(row["chr"], row["start"], row["stop"])
    sv_stat = init_sv_stat_row(
        row, num_samples=num_samples, num_reference=num_samples - len(squiggle_data)
    )

    if len(squiggle_data) == 0:
        gmm, evidence_by_mode = None, []
    else:
        gmm, evidence_by_mode = run_viz_gmm(
            squiggle_data,
            file_name=None,
            chr=row["chr"],
            L=row["start"],
            R=row["stop"],
            plot=False,
            plot_bokeh=False,
        )

    write_sv_stats(sv_stat, gmm, evidence_by_mode, population_size, FILE_DIR, iteration)


def run_all_sv(
    *,
    rerun_all_svs: bool = False,
    run_ambiguous_svs: bool = False,
    num_iterations: int = 1,
    query_chr: Optional[str] = None,
    subset: Optional[List[Tuple[str, int, int]]] = None,
):
    deletions_df = get_deletions_df()

    population_size = deletions_df.shape[1] - 12
    sample_ids = set(deletions_df.columns[11:-1])  # 2504 samples

    if subset is not None:
        for chr, start, stop in subset:
            row = deletions_df[
                (deletions_df["chr"] == chr)
                & (deletions_df["start"] == start)
                & (deletions_df["stop"] == stop)
            ].iloc[0]
            write_sv_stats(row.to_dict(), population_size, sample_ids)
    else:
        if rerun_all_svs:
            processed_sv_ids = set()
        else:
            processed_sv_ids = set(
                [file.strip(".csv") for file in os.listdir(FILE_DIR)]
            )
        rows = []
        if query_chr is not None:
            for _, row in deletions_df[deletions_df["chr"] == query_chr].iterrows():
                rows.append(row)
        elif run_ambiguous_svs:
            with open("1kgp/svs_to_rerun.txt") as f:
                for line in f:
                    sv_id = line.strip()
                    row = deletions_df[deletions_df["id"] == sv_id].iloc[0]
                    rows.append(row)
        else:
            for _, row in deletions_df.iterrows():
                rows.append(row)

        with multiprocessing.Manager() as manager:
            cpu_count = multiprocessing.cpu_count()
            p = multiprocessing.Pool(cpu_count)
            args = []
            for row in rows:
                if row.id in processed_sv_ids:
                    continue
                for i in range(1, num_iterations + 1):
                    args.append((row.to_dict(), population_size, sample_ids, i))
            p.starmap(run_all_sv_wrapper, args)
            p.close()
            p.join()

    concat_multi_processed_sv_files(FILE_DIR, OUTPUT_FILE_NAME)


if __name__ == "__main__":
    rerun_all_svs = False if len(sys.argv) < 2 else bool(sys.argv[1])
    run_all_sv(rerun_all_svs=rerun_all_svs)
