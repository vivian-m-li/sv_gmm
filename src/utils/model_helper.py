from collections import Counter
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import pysam

from src.utils.constants import CHR_LENGTHS
from src.utils.helper import get_sample_ids
from src.utils.types import StixQueryRegion, InsertSizeDistribution


# ------------------------------
# Data/input processing helpers
# -------------------------------
def load_vcf(dir: str, vcf_filename: str):
    """Loads a VCF file and converts it to a CSV file for easier processing later."""
    vcf_in = pysam.VariantFile(os.path.join(dir, vcf_filename))
    header = [
        "id",
        "chr",
        "start",
        "stop",
        "svlen",
        "ref",
        "alt",
        "qual",
        "filter",
        "af",
        "info",
    ] + list(vcf_in.header.samples)
    data = []
    n_removed = 0
    for record in vcf_in.fetch():
        info = dict(record.info)
        chr = record.chrom.strip("chr")

        sv_type = info.get("SVTYPE")
        if sv_type is None:
            pattern = r"^chr[^-]+-\d+-([A-Z]+)->"
            match = re.match(pattern, record.id)
            sv_type = match.group(1)

        if sv_type != "DEL" or chr in ["X", "Y"]:
            continue
        row = [
            record.id,
            chr,
            record.start,
            record.stop,
            record.rlen,
            record.ref,
            ",".join([str(alt) for alt in record.alts]),
            record.qual,
            record.filter.keys(),
            info["AF"],  # placeholder until it's recalculated below
            info,
        ]
        n_homozygous = 0
        n_heterozygous = 0
        for sample in record.samples:
            gt = record.samples[sample]["GT"]
            gt = tuple([0 if g is None else g for g in gt])  # convert None to 0
            gt_sum = sum(gt)
            if gt_sum == 1:
                n_heterozygous += 1
            elif gt_sum == 2:
                n_homozygous += 1
            row.append(gt)

        af = calc_af(n_homozygous, n_heterozygous, len(record.samples))
        row[9] = af

        # only keep the rows where at least one sample has an allele for the SV (i.e. a 1 in their GT)
        # removed 1760 rows without genotypes
        if n_homozygous > 0 or n_heterozygous > 0:
            data.append(row)

        else:
            n_removed += 1

    # print(f"Removed {n_removed} rows without genotypes")
    df = pd.DataFrame(data, columns=header)
    df["num_samples"] = 0

    out_filename = (
        vcf_filename.strip("vcf")
        if vcf_filename.endswith(".vcf")
        else vcf_filename.strip(".vcf.gz")
    ) + ".csv"
    df.to_csv(os.path.join(dir, out_filename), index=False)
    return df, out_filename


def get_reference_samples(
    # Gets the samples with the homozygous reference genotype (0, 0)
    reads: pd.DataFrame,
    chr: str,
    start: int,
    stop: int,
    input_dir: str,
) -> list[str]:
    df = pd.read_csv(os.path.join(input_dir, "svs_by_chr", f"chr{chr}.csv"))
    row = df[(df["start"] == start) & (df["stop"] == stop)]
    if row.empty:  # query region does not correspond with an SV in the callset
        return []
    samples = list(reads["sample_id"].unique())
    ref_samples = [
        sample_id for sample_id in samples if row.iloc[0][sample_id] == "(0, 0)"
    ]
    return ref_samples


def write_sample_ids_file(dir: str, filename: str, df: pd.DataFrame):
    """Writes a list of sample IDs."""
    sample_ids = set(df.columns[11:-1])
    with open(os.path.join(dir, filename), "w") as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")
    return sample_ids


def write_sv_lookup(dir: str, df: pd.DataFrame):
    """Writes an SV lookup file for easier access to SV info without loading entire vcf/csv."""
    sv_lookup_df = df[["id", "chr", "start", "stop", "svlen", "af"]]
    sv_lookup_df.to_csv(f"{dir}/sv_lookup.csv", index=False)


def write_svs_by_chr(dir: str, df: pd.DataFrame, chr: str | None = None):
    """Splits the SVs into separate files by chromosome for easier access during querying."""
    if not os.path.isdir(f"{dir}/svs_by_chr"):
        os.mkdir(f"{dir}/svs_by_chr")
    chrs = range(1, 23) if chr is None else [chr]
    for i in chrs:
        chr_df = df[df["chr"] == int(i)]
        chr_df.to_csv(f"{dir}/svs_by_chr/chr{i}.csv", index=False)


def write_default_insert_sizes(
    dir: str, sample_ids: str, default_size: int = 450
):
    """Writes a default insert size file with the specified default size for each sample."""
    filename = os.path.join(dir, "insert_sizes.csv")
    with open(filename, "w") as f:
        f.write("sample_id,mean_insert_size\n")
        for sample_id in sample_ids:
            f.write(f"{sample_id},{default_size}\n")
    return filename


def get_insert_size_lookup(
    dir: str,
    filename: str | None,
    default_insert_size: int | None,
    sample_ids: set[str],
) -> dict[str, InsertSizeDistribution]:
    """Returns a dictionary mapping sample IDs to their mean insert sizes from sequencing high-coverage short-reads."""
    insert_size_file = os.path.join(dir, filename)
    if filename is None:
        if default_insert_size is None:
            warnings.warn(
                "Setting the default insert size to 450bp for all samples. To specify a different default insert size, set the default_insert_size parameter in the config file."
            )
            default_insert_size = 450
        insert_size_file = write_default_insert_sizes(
            dir, sample_ids, default_insert_size
        )

    insert_size_df = pd.read_csv(
        insert_size_file,
        dtype={
            "sample_id": str,
            "mean_insert_size": float,
            "insert_size_sd": float,
        },
    )
    return {
        row["sample_id"]: InsertSizeDistribution(
            mean=int(row["mean_insert_size"]), sd=row["insert_size_sd"]
        )
        for _, row in insert_size_df.iterrows()
    }


def process_input_files(
    dir: str,
    sv_lookup_file: str,
    sample_id_file: str | None,
    insert_size_file: str | None,
    default_insert_size: int | None,
) -> tuple[pd.DataFrame, dict]:
    """Processes input files for more efficient lookup during querying."""

    df = None
    if sv_lookup_file.endswith(".vcf") or sv_lookup_file.endswith(".vcf.gz"):
        df, _ = load_vcf(dir, sv_lookup_file)
        write_svs_by_chr(dir, df)
    elif not os.path.isdir(os.path.join(dir, "svs_by_chr")):
        df = pd.read_csv(os.path.join(dir, sv_lookup_file))
        write_svs_by_chr(dir, df)

    if sample_id_file is None:
        if df is None:
            df = pd.read_csv(os.path.join(dir, sv_lookup_file))
        sample_ids = write_sample_ids_file(dir, "sample_ids.txt", df)
    else:
        sample_ids = get_sample_ids(os.path.join(dir, sample_id_file))

    if not os.path.isfile(os.path.join(dir, "sv_lookup.csv")):
        if df is None:
            df = pd.read_csv(os.path.join(dir, sv_lookup_file))
        write_sv_lookup(dir, df)

    insert_size_lookup = get_insert_size_lookup(
        dir, insert_size_file, default_insert_size, sample_ids
    )

    return df, insert_size_lookup


def giggle_format(chromosome: str, position: int):
    chr_formatted = (
        chromosome.lower().strip("chr")
        if type(chromosome) is str
        else str(chromosome)
    )
    return f"{chr_formatted}:{position}"


def stix_format(s: str):
    chr, pos = s.split(":")
    return chr, int(pos)


def reverse_giggle_format(l: str, r: str):  # noqa741
    chr = l.split(":")[0]
    start = int(l.split(":")[1])
    stop = int(r.split(":")[1])
    return chr, start, stop


def lookup_sv_position(sv_id: str, dir: str = "1kg"):
    lookup = pd.read_csv(os.path.join(dir, "sv_lookup.csv"))
    row = lookup[lookup["id"] == sv_id]
    if row.empty:
        raise ValueError(f"SV ID {sv_id} not found in lookup table.")

    chr = str(row["chr"].values[0])
    start = int(row["start"].values[0])
    stop = int(row["stop"].values[0])
    return chr, start, stop


def get_query_region(
    l: str, r: str, overlap: float = 0.5
) -> StixQueryRegion:  # noqa741
    """Returns the STIX query region information from the original SV position."""
    l_chr, l_pos = stix_format(l)
    _, r_pos = stix_format(r)
    svlen = r_pos - l_pos
    if overlap < 0:
        overlap = 1
    elif overlap > 1:
        overlap = min(1, overlap / 100)
    cutoff = int(svlen * (1 - overlap))
    l_start = max(0, l_pos - cutoff)
    l_stop = l_pos + cutoff
    r_start = r_pos - cutoff
    r_stop = min(r_pos + cutoff, CHR_LENGTHS[l_chr])
    return StixQueryRegion(
        chr=l_chr,
        left_start=l_start,
        left_stop=l_stop,
        right_start=r_start,
        right_stop=r_stop,
        file_name=f"{l}_{r}",
    )


def parse_input(input: str) -> str:
    parts = input.split(":")
    if len(parts) != 2:
        print("Input string must be in the format 'chromosome:position'")
        sys.exit(1)
    try:
        chromosome = parts[0]
        position = int(parts[1])
    except ValueError:
        print("Input string must be in the format 'chromosome:position'")
        sys.exit(1)
    return giggle_format(chromosome, position)


# ------------------
# Model helpers
# ------------------
def calc_af(n_homozygous, n_heterozygous, population_size):
    """Calculate allele frequency from number of homozygous and heterozygous individuals."""
    return ((n_homozygous * 2) + n_heterozygous) / (population_size * 2)


def calculate_posteriors(alpha):
    """Calculate the posterior probabilities and variances for each mode given alpha parameters."""
    p = alpha / np.sum(alpha)
    sum_alpha_post = np.sum(alpha)
    var = (alpha * (sum_alpha_post - alpha)) / (
        sum_alpha_post**2 * (sum_alpha_post + 1)
    )
    return p, var


def calculate_ci(p, var, n):
    """Calculate the 95% confidence interval for the difference in means between the two most probable modes."""
    # Calculate the difference in means between the two most probable modes
    posterior_mu_sorted_indices = np.argsort(p)
    posterior_mu_sorted = p[posterior_mu_sorted_indices]
    diff_in_means = posterior_mu_sorted[-1] - posterior_mu_sorted[-2]

    # Calculate the confidence interval for our difference in means
    diff_var = (var[posterior_mu_sorted_indices[-1]]) / n + (
        var[[posterior_mu_sorted_indices[-2]]]
    ) / n
    confidence = 1.96 * np.sqrt(diff_var)
    ci = [diff_in_means - confidence, diff_in_means + confidence]

    return ci


def calculate_posteriors_from_trials(outcomes):
    """Calculate the posterior probabilities and variances for each mode given a list of outcomes."""
    counts = Counter(outcomes)
    alpha = np.array([1, 1, 1]) + np.array([counts[1], counts[2], counts[3]])
    return calculate_posteriors(alpha)


def reciprocal_overlap(sv1: tuple[int, int], sv2: tuple[int, int]) -> float:
    """
    Calculates the reciprocal overlap between two structural variants.
    r = min(% overlap sv 1, % overlap sv 2)
    """
    start1, end1 = sv1
    start2, end2 = sv2
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_start >= overlap_end:
        return 0.0
    overlap_length = overlap_end - overlap_start
    sv1_length = end1 - start1
    sv2_length = end2 - start2
    return min(overlap_length / sv1_length, overlap_length / sv2_length)


def f1_score(confusion_mat: dict) -> float:
    """Calculates the F1 score given a confusion matrix."""
    TP = confusion_mat["TP"]
    FP = confusion_mat["FP"]
    FN = confusion_mat["FN"]
    if TP == 0:
        return 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
