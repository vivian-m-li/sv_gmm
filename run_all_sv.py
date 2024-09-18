import csv
import pandas as pd
import multiprocessing
from dataclasses import fields, asdict
from query_sv import query_stix, giggle_format
from viz import *
from gmm_types import *
from typing import Set

FILE_DIR = "1000genomes"
FILE_NAME = "sv_stats.csv"


def get_reference_samples(
    row: pd.Series, sample_set: Set[int], squiggle_data: Dict[str, np.ndarray[float]]
) -> List[str]:
    samples = [sample_id for sample_id in sample_set if sample_id in squiggle_data]
    ref_samples = [col for col in samples if row[col] == "(0, 0)"]
    return ref_samples


def get_sv_stats(
    row: pd.Series,
    population_size: int,
    sample_set: Set[int],
    svs: List[SVStatGMM],
) -> None:
    sv_stat = SVStatGMM(
        id=row.id,
        chr=row.chr,
        start=row.start,
        stop=row.stop,
        svlen=row.svlen,
        ref=row.ref,
        alt=row.alt,
        qual=row.qual,
        # filter=row.filter,
        af=row.af,
        # info=row.info,
        num_samples=0,
        num_pruned=0,
        num_reference=0,
        svlen_post=0,
        num_modes=0,
        num_iterations=0,
        overlap_between_modes=False,
        modes=[],
    )

    start = giggle_format(str(row.chr), row.start)
    end = giggle_format(str(row.chr), row.stop)

    squiggle_data = query_stix(start, end, False, filter_reference=False)
    missing_keys = set(squiggle_data.keys()) - sample_set
    for key in missing_keys:
        squiggle_data.pop(key, None)

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
        svs.append(sv_stat)
        return

    gmm, evidence_by_mode = run_viz_gmm(
        squiggle_data,
        file_name=None,
        chr=row.chr,
        L=row.start,
        R=row.stop,
        plot=False,
        plot_bokeh=False,
    )

    if gmm is None:
        svs.append(sv_stat)
        return

    sv_stat.num_pruned = sum(gmm.num_pruned) + len(gmm.outliers)
    sv_stat.num_modes = gmm.num_modes
    sv_stat.num_iterations = gmm.num_iterations

    all_svlen = get_svlen(evidence_by_mode)
    sv_stat.svlen_post = int(
        np.mean(
            [sv.length - 450 for lst in all_svlen for sv in lst]
        )  # 450 is the length of the read
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
            max_l = max([paired_end[0] for paired_end in evidence.paired_ends])
            min_r = min([paired_end[1] for paired_end in evidence.paired_ends])
            lengths.append(min_r - max_l - 450)  # 450 is the length of the read
            starts.append(max_l)
            ends.append(min_r)
            min_start = min(min_start, max_l)
            max_end = max(max_end, min_r)
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

    svs.append(sv_stat)


def dataclass_to_columns(dataclass_type):
    return [field.name for field in SVStatGMM(dataclass_type)]


def create_sv_stats_file():
    df_fields = [field.name for field in fields(SVStatGMM)]
    df = pd.DataFrame(columns=df_fields)
    return df


def listener(queue, sv_stats_file):
    with open(sv_stats_file, mode="a", newline="") as file:
        fieldnames = [field.name for field in fields(SVStatGMM)]
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        while True:
            result = queue.get()
            if result == "DONE":
                break
            csv_writer.writerow(result)
            file.flush()


def run_all_sv(
    *,
    query_chr: Optional[str] = None,
    subset: Optional[List[Tuple[str, int, int]]] = None,
):
    deletions_df = pd.read_csv(f"{FILE_DIR}/deletions_df.csv", low_memory=False)
    sv_stats_file = f"{FILE_DIR}/{FILE_NAME}"
    with open(sv_stats_file, mode="a", newline="") as file:
        fieldnames = [field.name for field in fields(SVStatGMM)]
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        file.seek(0, 2)  # Check if the file is empty
        if file.tell() == 0:
            csv_writer.writeheader()

    population_size = deletions_df.shape[1] - 12
    sample_ids = set(deletions_df.columns[11:-1])  # 2504 samples

    if subset is not None:
        svs = []
        for chr, start, stop in subset:
            row = deletions_df[
                (deletions_df["chr"] == chr)
                & (deletions_df["start"] == start)
                & (deletions_df["stop"] == stop)
            ].iloc[0]
            get_sv_stats(row, population_size, sample_ids, svs)
        with open(sv_stats_file, mode="a", newline="") as file:
            for sv in svs:
                csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
                csv_writer.writerow(asdict(sv))
    else:
        sv_stats_df = pd.read_csv(sv_stats_file)
        rows = []
        if query_chr is not None:
            for _, row in deletions_df[deletions_df["chr"] == query_chr].iterrows():
                if row.id in sv_stats_df["id"].values:
                    continue
                rows.append(row)
        else:
            for _, row in deletions_df.iterrows():
                if row.id in sv_stats_df["id"].values:
                    continue
                rows.append(row)

        with multiprocessing.Manager() as manager:
            cpu_count = multiprocessing.cpu_count()
            p = multiprocessing.Pool(cpu_count)

            for i in range(0, len(rows), cpu_count):
                svs = manager.list()
                args = [
                    (row, population_size, sample_ids, svs)
                    for row in rows[i : i + cpu_count]
                ]
                p.starmap(get_sv_stats, args)
                with open(sv_stats_file, mode="a", newline="") as file:
                    for sv in svs:
                        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
                        csv_writer.writerow(asdict(sv))

            p.close()
            p.join()


if __name__ == "__main__":
    # run_all_sv(subset=[("11", 54894935, 54899781)])
    # run_all_sv(subset=[("18", 45379612, 45379807)])
    run_all_sv(query_chr="1")
