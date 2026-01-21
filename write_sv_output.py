"""Functions for writing synthetic or real data outputs."""

import ast
import os
import subprocess
import csv
import pandas as pd
import numpy as np
from scipy.spatial.distance import braycurtis
from dataclasses import fields, asdict
from query_sv import query_stix, giggle_format
from helper import (
    get_deletions_df,
    get_sv_stats_converge_df,
    get_sv_stats_collapsed_df,
    get_sv_lookup,
    get_sv_chr,
    get_sample_ids,
    get_svlen,
    calc_af,
    calculate_posteriors_from_trials,
    calculate_ci,
    reciprocal_overlap,
    df_to_bed,
)
from gmm_types import SVInfoGMM, GMM, Evidence, ModeStat, SUPERPOPULATIONS
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter


def concat_processed_sv_files(
    file_dir: str, output_file_name: str, *, stem: str = "1kgp"
):
    """Concatenates individual files in file_dir into one file."""
    with open(f"{stem}/{output_file_name}", mode="w", newline="") as out:
        fieldnames = [field.name for field in fields(SVInfoGMM)]
        csv_writer = csv.DictWriter(out, fieldnames=fieldnames)
        csv_writer.writeheader()
        for file in os.listdir(file_dir):
            with open(f"{file_dir}/{file}") as f:
                for line in f:
                    out.write(line)


def concat_multi_processed_sv_files(
    file_dir: str, output_file_name: str, stem: str = "1kgp"
):
    """Concatenates individual files output from the dirichlet process (where the GMM is run for x iterations) in file_dir into one file. Used at the end of both short read and long read clustering processes."""
    with open(f"{stem}/{output_file_name}", mode="w", newline="") as out:
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
    """Writes the output after one iteration of the GMM in the dirichlet process."""
    with open(
        f"{file_dir}/{sv.id}_iteration={iteration}.csv", mode="w"
    ) as file:
        fieldnames = [field.name for field in fields(SVInfoGMM)]
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writerow(asdict(sv))


def get_reference_samples(row: pd.Series, reads: pd.DataFrame) -> List[str]:
    """Returns the samples with evidence that are actually homozygous for the reference allele (0, 0)."""
    samples_with_reads = set(reads["sample_id"].tolist())
    ref_samples = [col for col in samples_with_reads if row[col] == "(0, 0)"]
    return ref_samples


def init_sv_stat_row(
    row: Dict,
    *,
    num_samples: Optional[int] = 0,
    num_reference: Optional[int] = 0,
) -> SVInfoGMM:
    """Initializes a row for the SV output file."""
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
        num_samples=num_samples,
        num_samples_run=0,
        num_pruned=0,
        num_reference=num_reference,
        svlen_post=0,
        num_modes=0,
        num_iterations=0,
        overlap_between_modes=False,
        modes=[],
    )

    return sv_stat


def get_raw_data(
    row, input_dir: str = "1kgp"
) -> Tuple[Dict[str, np.ndarray[float]], int]:
    """
    Gets the samples and evidence for an SV. Filters out samples that are homozygous for the reference allele.
    The STIX index and paths are hardcoded for query_stix.
    """
    start = giggle_format(str(row["chr"]), row["start"])
    end = giggle_format(str(row["chr"]), row["stop"])
    reads = query_stix(
        l=start,
        r=end,
        input_dir=input_dir,
        output_dir="",  # project home directory
        run_gmm=False,
        filter_reference=False,
        stix_bin="/Users/vili4418/sv/stix/bin/stix",
        stix_index="/scratch/Shares/layer/stix/indices/1kg_high_coverage_vivian/shard",
        stix_database="/scratch/Shares/layer/stix/indices/1kg_high_coverage_vivian/shard",
        num_stix_shards=8,
    )
    num_samples = reads["sample_id"].nunique()

    reference_samples = get_reference_samples(row, reads)
    reads = reads[~reads["sample_id"].isin(reference_samples)]

    return reads, num_samples


def write_sv_stats(
    sv_stat: SVInfoGMM,
    gmm: Optional[GMM],
    evidence_by_mode: List[List[Evidence]],
    population_size: int,
    file_dir: str,
    iteration: int = 0,
) -> None:
    """
    Writes the output after one iteration of the GMM in the dirichlet process.

    Output: SV info (id/chr/start/stop/svlen/ref/alt/qual/af), # samples total/pruned/reference, final svlen, # modes as determined by the GMM, total # iterations run in the EM algorithm, if the mode coordinates overlap, and the start/stop/length/samples/af for each mode.
    """
    if gmm is None:
        write_sv_file(sv_stat, file_dir, iteration)
        return

    sv_stat.num_pruned = sum(gmm.num_pruned) + len(gmm.outliers)
    sv_stat.num_modes = gmm.num_modes
    sv_stat.num_iterations = gmm.num_iterations

    all_svlen = get_svlen(evidence_by_mode)
    sv_stat.svlen_post = int(
        np.mean([sv.length for lst in all_svlen for sv in lst])
    )

    mode_coords = []
    num_samples_run = 0
    for i, mode in enumerate(evidence_by_mode):
        sample_ids = [e.sample.id for e in mode]
        num_samples = len(sample_ids)
        num_samples_run += num_samples
        num_homozygous = len(
            [e.sample for e in mode if e.sample.allele == "(1, 1)"]
        )
        num_heterozygous = num_samples - num_homozygous
        af = calc_af(num_homozygous, num_heterozygous, population_size)

        lengths = []
        starts = []
        ends = []
        min_start = float("inf")
        max_end = float("-inf")
        for evidence in mode:
            mean_l = np.mean(
                [paired_end[0] for paired_end in evidence.paired_ends]
            )
            mean_r = np.mean(
                [paired_end[1] for paired_end in evidence.paired_ends]
            )
            mean_length = np.mean(
                [
                    paired_end[1] - paired_end[0] - evidence.mean_insert_size
                    for paired_end in evidence.paired_ends
                ]
            )
            lengths.append(mean_length)
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
    sv_stat.num_samples_run = num_samples_run

    if gmm.num_modes > 1:
        for i in range(len(evidence_by_mode) - 1):
            if mode_coords[i][1] > mode_coords[i + 1][0]:
                sv_stat.overlap_between_modes = True

    write_sv_file(sv_stat, file_dir, iteration)


def dataclass_to_columns(dataclass_type):
    """Returns the field names for the SVInfoGMM dataclass."""
    return [field.name for field in SVInfoGMM(dataclass_type)]


def create_sv_stats_file():
    """Creates the SV output file with the appropriate columns. Unused"""
    df_fields = [field.name for field in fields(SVInfoGMM)]
    df = pd.DataFrame(columns=df_fields)
    return df


def write_posterior_distributions(sv_id, alphas, posteriors, file_dir):
    """Writes the posterior distributions for each mode calculated during the dirichlet process."""
    with open(
        f"{file_dir}/{sv_id}_posteriors.csv", mode="w", newline=""
    ) as file:
        fieldnames = [
            "trial",
            "alpha",
            "posterior_probabilities",
            "posterior_variances",
        ]
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()
        for i, (alpha, (probs, vars)) in enumerate(zip(alphas, posteriors)):
            csv_writer.writerow(
                {
                    "trial": i + 1,
                    "alpha": alpha,
                    "posterior_probabilities": probs,
                    "posterior_variances": vars,
                }
            )


"""
Analyze SVs that have been run
"""


def write_sv_stats_collapsed(output_dir: str):
    """Collapse sv_stats_converge.csv to sv_stats_collapsed.csv by picking the most common result for each SV."""
    df = get_sv_stats_converge_df(output_dir)
    svs_n_modes = pd.read_csv(f"{output_dir}/svs_n_modes.csv")
    svs_n_modes.rename(
        columns={"num_modes": "consensus_num_modes"}, inplace=True
    )
    df = df.merge(svs_n_modes, left_on="id", right_on="sv_id")
    df.loc[df["num_modes"] == 0, "num_modes"] = (
        1  # set num_modes = 1 where it is 0
    )
    sv_ids = df["id"].unique()
    with open(
        f"{output_dir}/sv_stats_collapsed.csv", mode="w", newline=""
    ) as out:
        fieldnames = [field.name for field in fields(SVInfoGMM)]
        fieldnames.append("num_gmm_runs")
        csv_writer = csv.DictWriter(out, fieldnames=fieldnames)
        csv_writer.writeheader()
        for sv_id in sv_ids:
            rows = df[df["id"] == sv_id]
            if len(rows) < 1:
                continue

            row["num_gmm_runs"] = len(rows)
            consensus_num_modes = rows["consensus_num_modes"].values[0]
            rows = rows[rows["num_modes"] == consensus_num_modes]
            samples = Counter()
            samples_by_row = {}
            for i, row in rows.iterrows():
                modes = sorted(
                    ast.literal_eval(row["modes"]), key=lambda x: x["start"]
                )
                sample_ids = [
                    ",".join(sorted(mode["sample_ids"])) for mode in modes
                ]
                samples_joined = ";".join(sample_ids)
                samples.update([samples_joined])
                samples_by_row[samples_joined] = i

            most_common = samples.most_common(1)[0][0]
            row = rows.loc[samples_by_row[most_common]].copy()
            row = row.drop(
                ["consensus_num_modes", "num_modes_2", "sv_id", "confidence"]
            )
            csv_writer.writerow(row.to_dict())


def write_ancestry_dissimilarity(output_dir: str):
    """Calculate the dissimilarity in ancestry between modes for each SV."""
    df = get_sv_stats_collapsed_df(output_dir)
    confidence = pd.read_csv(f"{output_dir}/svs_n_modes.csv")
    confidence.rename(
        columns={"num_modes": "consensus_num_modes"}, inplace=True
    )
    df = df.merge(
        confidence,
        left_on="id",
        right_on="sv_id",
    )
    df = df[(df["confidence"] != "low") & (df["consensus_num_modes"] > 1)]
    ancestry_df = pd.read_csv("1kgp/ancestry.tsv", delimiter="\t")

    results_df = pd.DataFrame(
        columns=["chr", "start", "stop", "num_samples", "dissimilarity"]
    )
    for i, row in df.iterrows():
        ancestry = []
        modes = ast.literal_eval(row["modes"])
        for mode in modes:
            sample_ids = mode["sample_ids"]
            ancestry_counter = {anc: 0 for anc in SUPERPOPULATIONS}
            for sample_id in sample_ids:
                ancestry_row = ancestry_df[
                    ancestry_df["Sample name"] == sample_id
                ]
                superpopulation = (
                    ancestry_row["Superpopulation code"].values[0].split(",")[0]
                )
                ancestry_counter[superpopulation] += 1
            ancestry_counter = {
                k: v / len(sample_ids) for k, v in ancestry_counter.items()
            }
            ancestry.append([v for v in ancestry_counter.values()])

        dissimilarities = []
        for i in range(row["num_modes"] - 1):
            for j in range(i + 1, row["num_modes"]):
                dissimilarity = braycurtis(ancestry[i], ancestry[j])
                dissimilarities.append(dissimilarity)

        results_df.loc[len(results_df)] = [
            row["id"],
            row["chr"],
            row["start"],
            row["stop"],
            row["num_samples"],
            np.mean(dissimilarities),
        ]

    # explicitly cast columns to correct types - everything was converting to float because of dissimilarity
    results_df["chr"] = results_df["chr"].astype(int)
    results_df["start"] = results_df["start"].astype(int)
    results_df["stop"] = results_df["stop"].astype(int)
    results_df["num_samples"] = results_df["num_samples"].astype(int)

    results_df = results_df.sort_values(by="dissimilarity", ascending=False)
    results_df.to_csv(f"{output_dir}/ancestry_dissimilarity.csv", index=False)


def get_n_modes(input_dir: str, output_dir: str):
    """Get the number of modes and confidence for each SV."""
    sv_df = pd.DataFrame(
        columns=["sv_id", "num_modes", "confidence", "num_modes_2"]
    )

    deletions_df = get_deletions_df(input_dir)
    deletions_df = deletions_df[deletions_df["num_samples"] > 0]
    svs = deletions_df["id"].unique()
    df = get_sv_stats_converge_df(output_dir)
    for sv_id in svs:
        rows = df[df["id"] == sv_id]

        # if the GMM didn't run at all on a sample due to lack of evidence, skip the row
        if len(rows) == 0 or (
            len(rows) == 1 and rows["num_iterations"].values[0] == 0
        ):
            continue

        # if the GMM defaulted to 1 mode because there were 1-10 samples,
        # label it as 1 mode inconclusively
        if rows["num_samples_run"].values[0] <= 10:
            sv_df.loc[len(sv_df)] = [sv_id, 1, "inconclusive", np.nan]
            continue

        outcomes = rows["num_modes"].values
        counter = Counter(outcomes)
        most_common = counter.most_common(2)
        num_modes = max(1, most_common[0][0])
        num_modes_2 = int(most_common[1][0]) if len(counter) > 1 else np.nan

        p, var = calculate_posteriors_from_trials(outcomes)
        ci = calculate_ci(p, var, len(outcomes))
        new_row = [sv_id, num_modes]
        if ci[0] >= 0.6:
            new_row.append("high")
        elif ci[0] >= 0.3:
            new_row.append("medium")
        else:
            new_row.append("low")
        new_row.append(num_modes_2)

        sv_df.loc[len(sv_df)] = new_row
    sv_df.to_csv(f"{output_dir}/svs_n_modes.csv", index=False)


def get_sv_outliers(sv_rows, threshold: float):
    """Get outlier samples for a given SV."""
    n = len(sv_rows)
    outlier_counts = defaultdict(lambda: 0)
    for i, row in sv_rows.iterrows():
        modes = ast.literal_eval(row["modes"])
        for mode in modes:
            if mode["num_samples"] == 1:
                outlier_sample = mode["sample_ids"][0]
                outlier_counts[outlier_sample] += 1

    confident_outliers = []
    for outlier, count in outlier_counts.items():
        # this threshold is important for determining when we can confidently say something is an outlier
        if count / n > threshold:
            confident_outliers.append(outlier)

    return confident_outliers


def get_outliers(output_dir: str, threshold: float = 0.9):
    """Get outlier samples for each SV and write to a file."""
    n_modes_df = pd.read_csv(f"{output_dir}/svs_n_modes.csv")
    n_modes_df = n_modes_df[n_modes_df["num_modes"] > 1]
    df = get_sv_stats_converge_df(output_dir)
    with open(f"{output_dir}/outliers.csv", "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["sv_id", "sample_ids"])
        for _, row in n_modes_df.iterrows():
            sv_rows = df[df["id"] == row["sv_id"]]
            outliers = get_sv_outliers(sv_rows, threshold)
            if len(outliers) > 0:
                csv_writer.writerow([row["sv_id"], ",".join(outliers)])


def get_consensus_svs(output_dir: str):
    """Get consensus SVs by averaging the start/stop/length of each mode across all runs of the SV."""
    df = get_sv_stats_converge_df(output_dir)
    svs_n_modes = pd.read_csv(f"{output_dir}/svs_n_modes.csv")
    sv_ids = svs_n_modes["sv_id"].unique()
    consensus_df = pd.DataFrame(
        columns=[
            "id",
            "sv_id",
            "chr",
            "start",
            "start_sd",
            "end",
            "end_sd",
            "length",
            "length_sd",
            "num_samples",
            "num_samples_std",
        ]
    )
    for sv_id in sv_ids:
        sv_rows = df[df["id"] == sv_id]
        chr = sv_rows["chr"].values[0]
        modes_count = Counter(sv_rows["num_modes"])
        num_modes = max(modes_count, key=modes_count.get)
        sv_rows = sv_rows[sv_rows["num_modes"] == num_modes]
        all_mode_stats = [[] for _ in range(num_modes)]
        for _, row in sv_rows.iterrows():
            modes = ast.literal_eval(row["modes"])
            modes = sorted(modes, key=lambda x: x["start"])
            for i, mode in enumerate(modes):
                all_mode_stats[i].append(
                    {
                        "length": mode["length"],
                        "start": mode["start"],
                        "end": mode["end"],
                        "num_samples": mode["num_samples"],
                    }
                )

        # write mean and sd start/stop/length for each mode
        for i, mode in enumerate(all_mode_stats):
            row = [f"{sv_id}_{i + 1}", sv_id, chr]
            for key in ["start", "end", "length", "num_samples"]:
                values = [mode_stat[key] for mode_stat in mode]
                row.append(int(np.mean(values)))
                row.append(np.std(values))
            consensus_df.loc[len(consensus_df)] = row
    consensus_df.to_csv(f"{output_dir}/consensus_svs.csv", index=False)


def get_new_gene_intersections():
    """Get new gene intersections created by splitting multi-modal SVs into multiple single-modal SVs."""
    og_intersections = set()  # (sv_id, gene_id)
    with open("1kgp/original_gene_intersections.bed", "r") as f:
        for line in f:
            row = line.strip().split("\t")
            sv_id, gene_id = row[3], row[7]
            og_intersections.add((sv_id, gene_id))

    df_to_bed(
        in_file="1kgp/consensus_svs.csv", out_file="1kgp/consensus_svs.bed"
    )
    subprocess.run(
        ["bash", "bed_intersect.sh"]
        + [  # noqa503
            "1kgp/consensus_svs.bed",
            "1kgp/genes.bed",
            "1kgp/consensus_gene_intersections.bed",
        ],
        capture_output=True,
        text=True,
    )

    split_intersections = set()
    sv_split_lookup = {}
    with open("1kgp/consensus_gene_intersections.bed", "r") as f:
        for line in f:
            row = line.strip().split("\t")
            sv_id, sv_split_id, gene_id = row[3], row[4], row[8]
            split_intersections.add((sv_id, gene_id))
            sv_split_lookup[(sv_id, gene_id)] = sv_split_id

    new_intersections = split_intersections - og_intersections
    with open("1kgp/new_gene_intersections.bed", "w") as f:
        for sv_id, gene_id in new_intersections:
            sv_split_id = sv_split_lookup[(sv_id, gene_id)]
            f.write(f"{sv_split_id}\t{gene_id}\n")


def outlier_gene_intersections():
    """Get SVs that are both outliers and have new gene intersections."""
    svs_with_new_genes = set()
    with open("1kgp/new_gene_intersections.bed", "r") as f:
        for line in f:
            sv_id = line.strip().split()[0]
            sv_id = "_".join(sv_id.split("_")[0:2])
            svs_with_new_genes.add(sv_id)

    svs_with_outliers = set()
    with open("1kgp/outliers.txt", "r") as f:
        for line in f:
            sv_id = line.strip().split()[0]
            svs_with_outliers.add(sv_id)

    print(svs_with_outliers.intersection(svs_with_new_genes))


def high_confidence_gene_intersections():
    """Get SVs that are both high confidence and have new gene intersections."""
    svs_with_new_genes = set()
    with open("1kgp/new_gene_intersections.bed", "r") as f:
        for line in f:
            sv_id = line.strip().split()[0]
            sv_id = "_".join(sv_id.split("_")[0:2])
            svs_with_new_genes.add(sv_id)

    df = pd.read_csv("1kgp/svs_n_modes.csv")
    high_confidence_svs = set(
        df[(df["confidence"] == "high") & (df["num_modes"] >= 1)]["sv_id"]
    )
    print("n high confidence SVs:", len(high_confidence_svs))
    print(
        "n high confidence SVs with new gene intersections:",
        len(high_confidence_svs.intersection(svs_with_new_genes)),
    )


def recalculate_afs():
    """Recalculate allele frequencies based on genotypes in deletions_df and compare to allele frequencies in sv_stats_collapsed_df."""
    df = get_deletions_df().head(1000)
    results_df = get_sv_stats_collapsed_df()
    sample_ids = get_sample_ids()
    for _, row in df.iterrows():
        gt_samples = set()
        for sample_id in sample_ids:
            gt = ast.literal_eval(row[sample_id])
            gt = tuple([0 if g is None else g for g in gt])  # convert None to 0
            gt_sum = sum(gt)
            if gt_sum > 0:
                gt_samples.add(sample_id)

        results_row = results_df[results_df["id"] == row["id"]]
        if results_row.empty and len(gt_samples) == 0:
            continue

        model_samples = set()
        if not results_row.empty:
            modes = ast.literal_eval(results_row["modes"])
            if len(modes) == 0:
                continue
            for mode in modes:
                for sample_id in mode["sample_ids"]:
                    model_samples.add(sample_id)

        print(
            f"SV {row['id']}: n samples with alleles={len(gt_samples)}, n samples through model={len(model_samples)}"
        )


def compare_short_long_reads():
    """Compare the number of modes and confidence between short read and long read analyses."""
    sr_df = pd.read_csv("1kgp/svs_n_modes.csv")
    sr_df = sr_df[sr_df["confidence"] != "inconclusive"]
    sr_df = sr_df.rename(
        columns={
            "num_modes": "sr_num_modes",
            "confidence": "sr_confidence",
            "num_modes_2": "sr_num_modes_2",
        }
    )

    lr_df = pd.read_csv("long_reads/svs_n_modes.csv")
    lr_df = lr_df[lr_df["confidence"] != "inconclusive"]
    lr_df = lr_df.rename(
        columns={
            "num_modes": "lr_num_modes",
            "confidence": "lr_confidence",
            "num_modes_2": "lr_num_modes_2",
        }
    )

    merged = lr_df.merge(sr_df, on="sv_id", how="inner")
    merged["consensus"] = merged.apply(
        lambda row: (
            row["lr_num_modes"]
            if row["lr_num_modes"] == row["sr_num_modes"]
            else np.nan
        ),
        axis=1,
    )
    # consensus_df = merged[~merged["consensus"].isna()] # get all rows where the consensus is not NaN
    merged.to_csv("1kgp/sr_lr_merged.csv", index=False)


def bed_to_df(bed_file: str, remove_dupes: bool = False) -> pd.DataFrame:
    """Convert a BED file to a pandas DataFrame."""
    df = pd.read_csv(
        bed_file,
        delimiter="\t",
        names=[
            "sv1_chr",
            "sv1_l",
            "sv1_r",
            "sv1_id",
            "sv1_id_dup",
            "sv2_chr",
            "sv2_l",
            "sv2_r",
            "sv2_id",
            "sv2_svid",
        ],
    )
    if remove_dupes:
        df = df[df["sv1_id"] != df["sv2_id"]]

    df["r"] = df.apply(
        lambda row: reciprocal_overlap(
            (row["sv1_l"], row["sv1_r"]), (row["sv2_l"], row["sv2_r"])
        ),
        axis=1,
    )
    return df


def get_overlapping_clustered_svs():
    """Checks for overlaps between clustered SVs and the original SV breakpoints."""
    overlap_threshold = 0.7
    if not os.path.exists("1kgp/sv_lookup_intersect.csv"):
        lookup = get_sv_lookup()
        lookup["sv_id"] = lookup["id"].astype(str)
        lookup.rename(columns={"stop": "end"}, inplace=True)
        bed_file = "1kgp/sv_lookup.bed"
        df_to_bed(in_df=lookup, out_file=bed_file)

        output_bed_file = "1kgp/sv_lookup_intersect.bed"
        subprocess.run(
            ["bash", "bed_intersect.sh"]
            + [
                bed_file,
                "1kgp/consensus_svs.bed",
                output_bed_file,
            ],
            capture_output=True,
            text=True,
        )

        df = bed_to_df(output_bed_file)
        df.to_csv("1kgp/sv_lookup_intersect.csv", index=False)
    else:
        df = pd.read_csv("1kgp/sv_lookup_intersect.csv")

    overlaps = df[
        (df["r"] >= overlap_threshold) & (df["sv1_id"] != df["sv2_id"])
    ]

    # this gets all overlapping SVs - however, we want to check whether the original SVs were overlapping as well (pre-clustering)
    # print(
    #     overlaps[["sv1_id", "sv1_l", "sv1_r", "sv2_id", "sv2_l", "sv2_r", "r"]]
    # )

    og_overlaps = pd.read_csv("1kgp/og_svs_intersect.csv")
    og_overlaps = og_overlaps[og_overlaps["r"] >= overlap_threshold]
    # only need sv ids to check for overlaps
    og_overlaps = og_overlaps[["sv1_id", "sv2_id"]]

    # merge new overlaps with og_overlaps
    new_overlaps = overlaps.merge(
        og_overlaps,
        left_on=["sv1_id", "sv2_id"],
        right_on=["sv1_id", "sv2_id"],
        how="left",
        indicator=True,
    )
    # filter for entries in overlaps that are not in og_overlaps
    new_overlaps = new_overlaps[new_overlaps["_merge"] == "left_only"]

    # there are 22 overlapping SVs (>= 0.7 reciprocal overlap)
    print(new_overlaps.shape[0], new_overlaps)


def write_samplot_files():
    """Write samplot images for all SVs with 2+ modes."""
    sv_ids = os.listdir("long_reads/bam_files")
    for sv_id in sv_ids:
        chr, start, stop = get_sv_chr(sv_id)
        subprocess.run(
            ["bash", "samplot_viz.sh"] + [sv_id, chr, str(start), str(stop)],
            capture_output=True,
            text=True,
        )


def write_post_processed_files(input_dir: str, output_dir: str):
    """Write all post-processed files after running the dirichlet process with short or long reads."""
    get_n_modes(input_dir, output_dir)
    print("wrote svs_n_modes.csv")
    if input_dir == "long_reads":
        compare_short_long_reads()
        print("wrote sr_lr_merged.csv")
    get_consensus_svs(output_dir)
    print("wrote consensus_svs.csv")
    write_sv_stats_collapsed(output_dir)
    print("wrote sv_stats_collapsed.csv")
    get_outliers(output_dir)
    print("wrote outliers.csv")
    write_ancestry_dissimilarity(output_dir)
    print("wrote ancestry_dissimilarity.csv")
    # get_new_gene_intersections() # need bedtools to run this
