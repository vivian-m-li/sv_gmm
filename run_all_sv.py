import sys
import os
import pandas as pd
import multiprocessing
from process_data import *
from gmm_types import *
from write_sv_output import FILE_DIR, write_sv_stats


def run_all_sv(
    *,
    rerun_all_svs: bool = False,
    run_ambiguous_svs: bool = False,
    num_iterations: int = 1,
    query_chr: Optional[str] = None,
    subset: Optional[List[Tuple[str, int, int]]] = None,
):
    deletions_df = pd.read_csv("1000genomes/deletions_df.csv", low_memory=False)

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
            with open("1000genomes/svs_to_rerun.txt") as f:
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
            p.starmap(write_sv_stats, args)
            p.close()
            p.join()


if __name__ == "__main__":
    rerun_all_svs = False if len(sys.argv) < 2 else bool(sys.argv[1])
    run_all_sv(rerun_all_svs=rerun_all_svs)
