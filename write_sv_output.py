import csv
import pandas as pd
from dataclasses import fields, asdict
from query_sv import query_stix, giggle_format
from process_data import *
from gmm_types import *
from typing import Set


def concat_processed_sv_files(file_dir: str, output_file_name: str):
    with open(f"1000genomes/{output_file_name}", mode="w", newline="") as out:
        fieldnames = [field.name for field in fields(SVInfoGMM)]
        csv_writer = csv.DictWriter(out, fieldnames=fieldnames)
        csv_writer.writeheader()
        for file in os.listdir(file_dir):
            with open(f"{file_dir}/{file}") as f:
                for line in f:
                    out.write(line)


def concat_multi_processed_sv_files(file_dir: str, output_file_name: str):
    with open(f"1000genomes/{output_file_name}", mode="w", newline="") as out:
        fieldnames = [field.name for field in fields(SVInfoGMM)]
        csv_writer = csv.DictWriter(out, fieldnames=fieldnames)
        csv_writer.writeheader()
        for file in os.listdir(file_dir):
            if "iteration" not in file:
                continue
            with open(f"{file_dir}/{file}") as f:
                for line in f:
                    out.write(line)


def write_sv_file(sv: SVInfoGMM, file_dir: str, iteration: int):
    with open(f"{file_dir}/{sv.id}_iteration={iteration}.csv", mode="w") as file:
        fieldnames = [field.name for field in fields(SVInfoGMM)]
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writerow(asdict(sv))


def get_reference_samples(
    row: pd.Series, sample_set: Set[int], squiggle_data: Dict[str, np.ndarray[float]]
) -> List[str]:
    samples = [sample_id for sample_id in sample_set if sample_id in squiggle_data]
    ref_samples = [col for col in samples if row[col] == "(0, 0)"]
    return ref_samples


def init_sv_stat_row(row: Dict, sample_set: Set[int]) -> Tuple[SVInfoGMM, Dict]:
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

    return sv_stat, squiggle_data


def write_sv_stats(
    sv_stat: SVInfoGMM,
    gmm: Optional[GMM],
    evidence_by_mode: List[List[Evidence]],
    population_size: int,
    file_dir: str,
    iteration: int = 0,
) -> None:
    if gmm is None:
        write_sv_file(sv_stat, file_dir, iteration)
        return

    sv_stat.num_pruned = sum(gmm.num_pruned) + len(gmm.outliers)
    sv_stat.num_modes = gmm.num_modes
    sv_stat.num_iterations = gmm.num_iterations

    all_svlen = get_svlen(evidence_by_mode)
    sv_stat.svlen_post = int(np.mean([sv.length for lst in all_svlen for sv in lst]))

    mode_coords = []
    for i, mode in enumerate(evidence_by_mode):
        sample_ids = [e.sample.id for e in mode]
        num_samples = len(sample_ids)
        num_homozygous = len([e.sample for e in mode if e.sample.allele == "(1, 1)"])
        num_heterozygous = num_samples - num_homozygous
        af = ((num_homozygous * 2) + num_heterozygous) / population_size * 2

        lengths = []
        starts = []
        ends = []
        min_start = float("inf")
        max_end = float("-inf")
        for evidence in mode:
            mean_l = np.mean([paired_end[0] for paired_end in evidence.paired_ends])
            mean_r = np.mean([paired_end[1] for paired_end in evidence.paired_ends])
            lengths.append(mean_r - mean_l - evidence.mean_insert_size)
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
            num_heterozygous=num_heterozygous,
            num_homozygous=num_homozygous,
            sample_ids=sample_ids,
            num_pruned=gmm.num_pruned[i],
            af=af,
        )
        sv_stat.modes.append(mode_stat)

    if gmm.num_modes > 1:
        for i in range(len(evidence_by_mode) - 1):
            if mode_coords[i][1] > mode_coords[i + 1][0]:
                sv_stat.overlap_between_modes = True

    write_sv_file(sv_stat, file_dir, iteration)


def dataclass_to_columns(dataclass_type):
    return [field.name for field in SVInfoGMM(dataclass_type)]


def create_sv_stats_file():
    df_fields = [field.name for field in fields(SVInfoGMM)]
    df = pd.DataFrame(columns=df_fields)
    return df


def write_posterior_distributions(sv_stat, alphas, posteriors, file_dir):
    file = f"{file_dir}/{sv_stat.id}_posteriors.csv"
    fieldnames = [
        "trial",
        "alpha",
        "posterior_probabilities",
        "posterior_variances",
    ]
    csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
    for i, (alpha, (probs, vars)) in enumerate(zip(alphas, posteriors)):
        csv_writer.writerow(
            {
                "trial": i + 1,
                "alpha": alpha,
                "posterior_probabilities": probs,
                "posterior_variances": vars,
            }
        )
