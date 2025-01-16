import sys
import os
import csv
import pandas as pd
import multiprocessing
from dataclasses import fields, asdict
from query_sv import query_stix, giggle_format
from process_data import *
from gmm_types import *
from typing import Set

FILE_DIR = "processed_svs"


def write_sv_file(sv: SVInfoGMM):
    with open(f"{FILE_DIR}/{sv.id}.csv", mode="w") as file:
        fieldnames = [field.name for field in fields(SVInfoGMM)]
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writerow(asdict(sv))


def get_reference_samples(
    row: pd.Series, sample_set: Set[int], squiggle_data: Dict[str, np.ndarray[float]]
) -> List[str]:
    samples = [sample_id for sample_id in sample_set if sample_id in squiggle_data]
    ref_samples = [col for col in samples if row[col] == "(0, 0)"]
    return ref_samples


def write_sv_stats(
    row: Dict,
    population_size: int,
    sample_set: Set[int],
) -> None:
    sv_stat = SVInfoGMM(
        id=row["id"],
        chr=row["chr"],
        start=row["start"],
        stop=row["stop"],
        svlen=row["svlen"],
        ref=row["ref"],
        alt=row["alt"],
        qual=row["qual"],
        af=row["af"],
        num_samples=0,
        num_pruned=0,
        num_reference=0,
        svlen_post=0,
        num_modes=0,
        num_iterations=0,
        overlap_between_modes=False,
        modes=[],
    )

    start = giggle_format(str(row["chr"]), row["start"])
    end = giggle_format(str(row["chr"]), row["stop"])

    squiggle_data = query_stix(start, end, False, filter_reference=False)

    sv_stat.num_samples = len(squiggle_data)

    reference_samples = get_reference_samples(
        row,
        sample_set,
        squiggle_data,
    )
    sv_stat.num_reference = len(reference_samples)
    for ref in reference_samples:
        squiggle_data.pop(ref, None)

    if len(squiggle_data) == 0:
        write_sv_file(sv_stat)
        return

    gmm, evidence_by_mode = run_viz_gmm(
        squiggle_data,
        file_name=None,
        chr=row["chr"],
        L=row["start"],
        R=row["stop"],
        plot=False,
        plot_bokeh=False,
    )

    if gmm is None:
        write_sv_file(sv_stat)
        return

    sv_stat.num_pruned = sum(gmm.num_pruned) + len(gmm.outliers)
    sv_stat.num_modes = gmm.num_modes
    sv_stat.num_iterations = gmm.num_iterations

    all_svlen = get_svlen(evidence_by_mode)
    sv_stat.svlen_post = int(
        np.mean(
            [sv.length - 450 for lst in all_svlen for sv in lst]
        )  # 450 is the length of the read - TODO(later): get the actual read length for each sample
    )

    mode_coords = []
    for i, mode in enumerate(evidence_by_mode):
        sample_ids = [e.sample.id for e in mode]
        num_samples = len(sample_ids)
        num_homozygous = len([e.sample for e in mode if e.sample.allele == "(1, 1)"])

        lengths = []
        starts = []
        ends = []
        min_start = float("inf")
        max_end = float("-inf")
        for evidence in mode:
            mean_l = np.mean([paired_end[0] for paired_end in evidence.paired_ends])
            mean_r = np.mean([paired_end[1] for paired_end in evidence.paired_ends])
            lengths.append(mean_r - mean_l - 450)  # 450 is the length of the read
            starts.append(mean_l)
            ends.append(mean_r)
            min_start = min(min_start, mean_l)
            max_end = max(max_end, mean_r)
        mode_coords.append((min_start, max_end))

        mode_stat = ModeStat(
            length=np.mean(lengths),
            length_sd=np.std(lengths),
            start=int(np.mean(starts)),
            start_sd=np.std(starts),
            end=int(np.mean(ends)),
            end_sd=np.std(ends),
            num_samples=num_samples,
            num_heterozygous=num_samples - num_homozygous,
            num_homozygous=num_homozygous,
            sample_ids=sample_ids,
            num_pruned=gmm.num_pruned[i],
            af=num_samples / population_size,
        )
        sv_stat.modes.append(mode_stat)

    if gmm.num_modes > 1:
        for i in range(len(evidence_by_mode) - 1):
            if mode_coords[i][1] > mode_coords[i + 1][0]:
                sv_stat.overlap_between_modes = True

    write_sv_file(sv_stat)


def dataclass_to_columns(dataclass_type):
    return [field.name for field in SVInfoGMM(dataclass_type)]


def create_sv_stats_file():
    df_fields = [field.name for field in fields(SVInfoGMM)]
    df = pd.DataFrame(columns=df_fields)
    return df


def run_all_sv(
    *,
    rerun_all_svs: bool = False,
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
                args.append((row.to_dict(), population_size, sample_ids))
            p.starmap(write_sv_stats, args)
            p.close()
            p.join()


if __name__ == "__main__":
    rerun_all_svs = False if len(sys.argv) < 2 else bool(sys.argv[1])
    run_all_sv(rerun_all_svs=rerun_all_svs)
