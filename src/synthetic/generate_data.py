from collections import defaultdict
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from src.model.dirichlet import run_dirichlet
from src.utils.helper import stix_output_to_df
from src.utils.model_helper import get_insert_size_lookup
from src.utils.types import InsertSizeDistribution


# -----------------------------------------
# Generate data for GATK SVCluster pipeline
# -----------------------------------------
def data_to_vcf(
    evidence,
    insert_size_lookup: dict[str, InsertSizeDistribution],
    vcf_filename: str,
):
    """
    Writes the synthetic data to a VCF file to be used for the GATK SVCluster pipeline.
    Each record represents one deletion event defined by paired-end read evidence.
    Deletion events are calculated from the average L and R positions of each read pair for each sample.
    """
    reads_df = pd.DataFrame(columns=["sample_id", "L", "R", "mean_insert_size"])
    for sample_id, values in evidence.items():
        mean_insert_size = insert_size_lookup[sample_id].mean
        for read_L, read_R in zip(values[::2], values[1::2]):
            reads_df.loc[len(reads_df)] = [
                sample_id,
                min(read_L, read_R),  # enforce L < R
                max(read_L, read_R),
                mean_insert_size,
            ]

    vcf_records = []
    chrom = "1"
    # cluster rows by sample_id
    # take the average L and (R - mean_insert_size) for each read pair as the deletion coordinates
    reads_df = (
        reads_df.groupby("sample_id")
        .agg({"L": "mean", "R": "mean", "mean_insert_size": "first"})
        .reset_index()
    )

    for i, row in reads_df.iterrows():
        variant_id = f"{row['sample_id']}_{i + 1}"
        sv_type = "DEL"
        ref = "N"
        alt = "<DEL>"  # symbolic alt allele for deletions

        # INFO field per VCF 4.2 structural variant conventions
        info = (
            f"END={int(row['R'] - row['mean_insert_size'])};"
            f"SVTYPE={sv_type};"
            f"SAMPLE={row['sample_id']};"
            f"ALGORITHMS=PESR"
        )

        vcf_records.append(
            [chrom, int(row["L"]), variant_id, ref, alt, ".", "PASS", info]
        )

    vcf_df = pd.DataFrame(
        vcf_records,
        columns=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"],
    )
    vcf_df = vcf_df.sort_values(by=["CHROM", "POS"]).reset_index(drop=True)

    # write VCF
    with open(vcf_filename, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write('##ALT=<ID=DEL,Description="Deletion">\n')
        f.write(
            '##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">\n'
        )
        f.write(
            '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">\n'
        )
        f.write(
            '##INFO=<ID=SAMPLE,Number=1,Type=String,Description="Sample ID">\n'
        )
        f.write(
            '##INFO=<ID=ALGORITHMS,Number=.,Type=String,Description="Algorithms that discovered this variant">\n'
        )
        f.write("##contig=<ID=1,length=200000>\n")
        f.write("#" + "\t".join(vcf_df.columns) + "\n")
        vcf_df.to_csv(f, sep="\t", index=False, header=False)

    # ploidy file
    ploidy_filename = vcf_filename.replace(".vcf", ".tsv")
    with open(ploidy_filename, "w") as f:
        f.write("sample_id\tploidy\n")
        for sample_id in reads_df["sample_id"].unique():
            f.write(f"{sample_id}\t2\n")

    return vcf_df


def write_fasta(out_fasta, contig_specs, line_width=60):
    """
    contig_specs: list of (name, length)
    Writes FASTA with sequence lines of 'N' repeated.
    Returns list of tuples (name, length, fasta_header_offset, first_base_offset, line_bases, line_bytes)
    where offsets are byte offsets used for .fai
    """
    records = []
    with open(out_fasta, "wb") as fh:
        for name, length in contig_specs:
            header = f">{name}\n".encode()
            header_offset = fh.tell()
            fh.write(header)
            # Write sequence in chunks
            full_line = (b"N" * line_width) + b"\n"
            n_full = length // line_width
            tail = length % line_width
            first_base_offset = fh.tell()
            for _ in range(n_full):
                fh.write(full_line)
            if tail:
                fh.write(b"N" * tail + b"\n")
            # compute line_bases = number of bases per line (except maybe last)
            line_bases = line_width
            line_bytes = line_width + 1  # newline included
            # If length < line_width, line_bytes is tail + 1
            if length < line_width:
                line_bases = length
                line_bytes = length + 1
            records.append(
                (
                    name,
                    length,
                    header_offset,
                    first_base_offset,
                    line_bases,
                    line_bytes,
                )
            )
    return records


def write_fai(fai_path, records):
    """
    .fai format: name\tlength\toffset\tline_bases\tline_bytes\n
    """
    with open(fai_path, "w") as fh:
        for (
            name,
            length,
            header_offset,
            first_base_offset,
            line_bases,
            line_bytes,
        ) in records:
            fh.write(
                f"{name}\t{length}\t{first_base_offset}\t{line_bases}\t{line_bytes}\n"
            )


def write_dict(dict_path, records, fasta_name):
    """
    Minimal Picard/GATK dictionary:
      @HD	VN:1.0
      @SQ	SN:<name>	LN:<len>
    Optionally could include M5:md5, but GATK doesn't require MD5 in every case
    """
    with open(dict_path, "w") as fh:
        fh.write("@HD\tVN:1.0\n")
        for (name, length, *_rest) in records:
            fh.write(f"@SQ\tSN:{name}\tLN:{length}\n")


def write_contig_list(list_path, records):
    with open(list_path, "w") as fh:
        for (name, *_rest) in records:
            fh.write(f"{name}\n")


def parse_contigs_arg(contigs):
    """Accepts a list of contig specifications of the form chr_name:chr_length"""
    specs = []
    for contig in contigs:
        assert ":" in contig, f"Contig specifier {contig} missing ':'"
        name, chr_length = contig.split(":")
        specs.append((name, int(chr_length)))
    return specs


def generate_reference_files(out_file: str, contigs, line_width: int = 60):
    """
    Generates reference.fasta, reference.fasta.fai, reference.dict, contig_list.txt
    for use with the GATK SVCluster pipeline.
    """
    out_prefix = Path(out_file)
    fasta = out_prefix.with_suffix(".fasta")
    fai = fasta.with_suffix(".fasta.fai")
    dictf = out_prefix.with_suffix(".dict")
    contig_list = out_prefix.with_suffix(".contig_list")

    contig_specs = parse_contigs_arg(contigs)
    records = write_fasta(fasta, contig_specs, line_width=line_width)
    write_fai(fai, records)
    write_dict(dictf, records, fasta.name)
    write_contig_list(contig_list, records)


# -----------------------------------------
# Generate synthetic data for testing SPLIT
# -----------------------------------------
def generate_weights(num_svs: int):
    """Generates random weights (0.5 <= p <= 0.95) for each SV mode."""
    if num_svs == 1:
        return [1.0]

    elif num_svs == 2:
        p1 = random.uniform(0.05, 0.95)
        return [p1, 1 - p1]

    elif num_svs == 3:
        while True:
            p1 = random.uniform(0.05, 0.95)
            p2 = random.uniform(0.05, 0.95)
            if p1 + p2 <= 0.95:
                return [p1, p2, 1 - p1 - p2]


def assign_modes(weights, samples):
    """Assigns the samples to modes depending on their weights."""
    num_samples = len(samples)
    assigned = [round(w * num_samples) for w in weights]
    if sum(assigned) != num_samples:
        assigned[0] += num_samples - sum(assigned)
    modes = []
    for i, num_samples in enumerate(assigned):
        modes.extend([i] * num_samples)
    return modes


def get_random_insert_size(df):
    """Gets a random insert size from the 1kg insert size distribution."""
    return df.sample().insert_size.values[0]


def generate_synthetic_sv_vcf(
    chr: int,  # chromosome number (does not support X/Y), as a str
    svs: list[tuple[int, int]],
    *,
    vcf_filename: str,  # writes the data to the file
):  # list of (start, stop) for each SV):
    """Writes 'called' SVs into VCF format."""

    vcf_records = []
    chrom = str(chr)
    for i, (start, stop) in enumerate(svs):
        variant_id = f"SV_{i + 1}"
        sv_type = "DEL"
        ref = "N"
        alt = "<DEL>"
        info = f"END={stop};" f"SVTYPE={sv_type};" f"ALGORITHMS=PESR"
        vcf_records.append(
            [chrom, start, variant_id, ref, alt, ".", "PASS", info]
        )

    vcf_df = pd.DataFrame(
        vcf_records,
        columns=["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"],
    )
    vcf_df = vcf_df.sort_values(by=["CHROM", "POS"]).reset_index(drop=True)

    # write VCF
    with open(vcf_filename, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write('##ALT=<ID=DEL,Description="Deletion">\n')
        f.write(
            '##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">\n'
        )
        f.write(
            '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">\n'
        )
        f.write(
            '##INFO=<ID=ALGORITHMS,Number=.,Type=String,Description="Algorithms that discovered this variant">\n'
        )
        f.write("#" + "\t".join(vcf_df.columns) + "\n")
        vcf_df.to_csv(f, sep="\t", index=False, header=False)

    return vcf_df


def generate_mapped_pairs_for_sv(
    mode_start: int,
    mode_end: int,
    insert_mean: int,
    n_pairs: int = 10,
    fragment_length_sd: int = 108,
    read_jitter: int = 5,
):
    """
    Return a list of tuples (l_start, l_end, r_start, r_end) representing
    mapped coordinates on the reference for discordant pairs supporting
    a deletion between mode_start and mode_end.

    - insert_mean: per-sample mean insert size
    - insert size sampled sampled ~ N(insert_mean, fragment_length_sd)
    - left mate placed so its end sits a small jitter upstream of mode_start.
    """
    pairs = []
    for _ in range(n_pairs):
        # fragment length + deletion length = insert size
        # addn noise to fragment length (but not deletion length since that's fixed for the SV)

        # generate the fragment length from a normal distribution
        fragment_length = int(
            max(10, min(900, random.gauss(insert_mean, fragment_length_sd)))
        )

        # split the fragment into left and right fragments
        # the split point is sampled from a normal distribution so that most splits are near the middle of the fragment
        split_point = random.gauss(0.5, 0.1)
        left_fragment_length = int(fragment_length * split_point)
        right_fragment_length = fragment_length - left_fragment_length

        # each pair's inner bounds are the same as the variant breakpoints with +- 5bp
        left_start = mode_start - left_fragment_length
        left_end = mode_start + random.randint(-read_jitter, read_jitter)

        right_end = mode_end + right_fragment_length
        right_start = mode_end + random.randint(-read_jitter, read_jitter)

        pairs.append((left_start, left_end, right_start, right_end))

    return pairs


def generate_and_split_sample_reads(
    chr: int,  # chromosome number (does not support X/Y), as a str
    svs: list[tuple[int, int]],  # list of (start, stop) for each SV
    *,
    input_dir: str,
    insert_size_file: str,
    model_params: dict,
    n_samples: int | None = None,
    p: list[float] | None = None,
    gmm_model: str = "2d",
    run_split: bool = True,
    plot: bool = False,
    plot_reads: bool = False,
    vcf_filename: str | None = None,  # writes the data to the file
):
    """Generates synthetic short-read data for testing purposes and runs the data through the SV analysis pipeline."""
    num_svs = len(svs)

    # decide how many samples we want in our population
    num_samples = random.randint(11, 2504) if n_samples is None else n_samples
    samples = [f"sample_{i}" for i in range(num_samples)]

    # decide how we want to divide the samples between the SVs
    weights = generate_weights(num_svs) if p is None else p
    modes = assign_modes(weights, samples)

    insert_sizes = get_insert_size_lookup(input_dir, insert_size_file)
    insert_size_distribution = [sample.mean for sample in insert_sizes.values()]
    insert_size_sds = [sample.sd for sample in insert_sizes.values()]

    # for each sample, generate random evidence
    reads = stix_output_to_df("", write_empty_file=True)
    evidence = defaultdict(list)
    insert_size_lookup = {}
    for sample, mode in zip(samples, modes):
        num_evidence = random.randint(2, 30)
        mode_start, mode_end = svs[mode]

        # sample a per-sample mean insert size from your table
        insert_size = random.choice(insert_size_distribution)
        insert_size_lookup[sample] = InsertSizeDistribution(
            mean=insert_size, sd=random.choice(insert_size_sds)
        )

        pairs = generate_mapped_pairs_for_sv(
            mode_start=mode_start,
            mode_end=mode_end,
            insert_mean=insert_size,
            n_pairs=num_evidence,
        )
        for pair in pairs:
            reads.loc[len(reads)] = [
                0,
                sample,
                1,
                pair[0],
                pair[1],
                1,
                pair[2],
                pair[3],
                "split",
            ]
            evidence[sample].extend([pair[1], pair[2]])  # l_end, r_start

    # pass synthetic data through SV analysis pipeline
    L = np.median([start for start, _ in svs])
    R = np.median([stop for _, stop in svs])
    evidence = {key: np.array(value) for key, value in evidence.items()}

    if plot_reads:
        plt.figure()
        plt.scatter(
            reads["l_start"].tolist(),
            reads["r_end"].tolist(),
            color="blue",
            alpha=0.6,
        )
        plt.xlabel("L")
        plt.ylabel("R")
        plt.show()

    if vcf_filename:
        data_to_vcf(evidence, insert_size_lookup, vcf_filename)

    if run_split:
        gmm_results, _, _ = run_dirichlet(
            reads,
            chr=str(chr),
            L=L,
            R=R,
            plot=plot,
            synthetic_data=True,
            gmm_model=gmm_model,
            insert_size_lookup=insert_size_lookup,
            init=model_params["init"],
            repulsion=model_params["repulsion"],
            r_threshold=model_params["r_threshold"],
            repulsion_stepsize=model_params["repulsion_stepsize"],
            model_comparison_func=model_params["model_comparison_func"],
        )
        gmm_result, evidence_by_mode = min(
            gmm_results, key=lambda x: x[0].score if x[0] else np.inf
        )

        return gmm_result, evidence_by_mode

    return None, []


def generate_sv_coordinates(
    case: str, svlen: int, *, r: float | None = None, start: int | None = None
):
    """Generates synthetic data with increasing r (reciprocal overlap) and runs the GMM on it to test accuracy."""
    # SVLEN = 802  # median SV length for high coverage data
    SV1_L = 1000000 if start is None else start
    SV1_R = SV1_L + svlen
    SV1 = (SV1_L, SV1_R)
    rs = np.arange(0.05, 1.01, 0.05) if r is None else [r]

    data = []
    match case:
        case "A":
            data.append([case, 0, [[SV1_L, SV1_L + svlen]]])
        case "B":
            # two nested SVs
            for r in rs:
                midpoint = SV1_L + (0.5 * svlen)
                sv2_len = int(svlen / r)
                data.append(
                    [
                        case,
                        r,
                        [
                            SV1,
                            (
                                midpoint - 0.5 * sv2_len,
                                midpoint + 0.5 * sv2_len,
                            ),
                        ],
                    ]
                )
        case "C":
            # two overlapping SVs
            for r in rs:
                overlap = int(r * svlen)
                sv2_start = SV1_R - overlap
                data.append([case, r, [SV1, (sv2_start, sv2_start + svlen)]])
        case "D":
            # three overlapping SVs
            for r1 in rs:  # overlap between sv1 and sv2
                overlap12 = int(r1 * svlen)
                sv2_start = SV1_R - overlap12
                sv2_end = sv2_start + svlen
                for r2 in rs:  # overlap between sv2 and sv3
                    overlap23 = int(r2 * svlen)
                    sv3_start = sv2_end - overlap23
                    sv3_end = sv3_start + svlen
                    data.append(
                        [
                            case,
                            (round(r1, 2), round(r2, 2)),
                            [SV1, (sv2_start, sv2_end), (sv3_start, sv3_end)],
                        ]
                    )
        case _:
            raise Exception("Invalid case")
    data_cleaned = []
    for case, r, svs in data:
        svs_int = [(int(sv[0]), int(sv[1])) for sv in svs]
        data_cleaned.append([case, r, svs_int])
    return data_cleaned
