import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from process_data import run_viz_gmm
from collections import defaultdict
from typing import List, Tuple, Optional, Dict


"""
Generate data for GATK SVCluster pipeline
"""


def data_to_vcf(
    evidence, insert_size_lookup: Dict[str, int], vcf_filename: str
):
    """
    Writes the synthetic data to a VCF file to be used for the GATK SVCluster pipeline.
    Each record represents one deletion event defined by paired-end read evidence.
    """
    reads_df = pd.DataFrame(columns=["sample_id", "L", "R", "mean_insert_size"])
    for sample_id, values in evidence.items():
        mean_insert_size = insert_size_lookup[sample_id]
        for read_L, read_R in zip(values[::2], values[1::2]):
            reads_df.loc[len(reads_df)] = [
                sample_id,
                min(read_L, read_R),  # enforce L < R
                max(read_L, read_R),
                mean_insert_size,
            ]

    vcf_records = []
    chrom = "1"
    for i, row in reads_df.iterrows():
        variant_id = f"{row['sample_id']}_{i + 1}"
        sv_type = "DEL"
        ref = "N"
        alt = "<DEL>"  # symbolic alt allele for deletions

        # INFO field per VCF 4.2 structural variant conventions
        info = (
            f"END={int(row['R'])};"
            f"SVTYPE={sv_type};"
            f"SAMPLE={row['sample_id']};"
            f"MEAN_INSERT={row['mean_insert_size']};"
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
            '##INFO=<ID=MEAN_INSERT,Number=1,Type=Float,Description="Mean insert size for sample">\n'
        )
        f.write(
            '##INFO=<ID=ALGORITHMS,Number=.,Type=String,Description="Algorithms that discovered this variant">\n'
        )
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


"""
Data generation functions
"""


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
    """Gets a random insert size from the 1kgp insert size distribution."""
    return df.sample().insert_size.values[0]


def generate_synthetic_sv_data(
    chr: int,  # chromosome number (does not support X/Y), as a str
    svs: List[Tuple[int, int]],  # List of (start, stop) for each SV
    *,
    n_samples: Optional[int] = None,
    p: Optional[List[float]] = None,
    gmm_model: str = "2d",
    run_gmm: bool = True,
    plot: bool = False,
    plot_reads: bool = False,
    vcf_filename: str = False,  # writes the data to the file
):
    """Generates synthetic SV data for testing purposes and runs the data through the SV analysis pipeline."""
    num_svs = len(svs)

    # Decide how many samples we want in our population
    num_samples = random.randint(30, 1000) if n_samples is None else n_samples
    samples = [f"sample_{i}" for i in range(num_samples)]

    # Decide how we want to divide the samples between the SVs
    weights = generate_weights(num_svs) if p is None else p
    modes = assign_modes(weights, samples)

    # print(
    #     f"{num_samples} total samples, {[modes.count(i) for i in range(num_svs)]} samples per mode"
    # )

    insert_size_df = pd.read_csv(
        "1kgp/insert_sizes.csv", dtype={"mean_insert_size": int}
    )

    # For each sample, generate random evidence
    evidence = defaultdict(list)
    insert_size_lookup = {}
    for sample, mode in zip(samples, modes):
        num_evidence = random.randint(2, 30)
        mode_start, mode_end = svs[mode]
        insert_size = insert_size_df.sample()["mean_insert_size"].values[0]
        insert_size_lookup[sample] = insert_size

        for _ in range(num_evidence):
            read_length = min(550, max(350, int(random.gauss(insert_size, 25))))
            # split = random.randint(1, read_length - 1) # the split can be anywhere
            split = random.randint(
                int(read_length / 2) - 100, int(read_length / 2) + 100
            )  # to prevent the read from being filtered out (too far from the y=x line)
            evidence_start = mode_start - split
            evidence_stop = mode_end + (read_length - split)
            evidence[sample].extend([evidence_start, evidence_stop])

    # Pass synthetic data through SV analysis pipeline
    L = np.mean([start for start, _ in svs])
    R = np.mean([stop for _, stop in svs])
    evidence = {key: np.array(value) for key, value in evidence.items()}

    if plot_reads:
        plt.figure()
        for reads in evidence.values():
            plt.scatter(
                reads[::2],
                reads[1::2],
                color="blue",
                alpha=0.6,
            )
        plt.xlabel("L")
        plt.ylabel("R")
        plt.show()

    if vcf_filename:
        data_to_vcf(evidence, insert_size_lookup, vcf_filename)

    if run_gmm:
        gmm, evidence_by_mode = run_viz_gmm(
            evidence,
            file_name=None,
            chr=str(chr),
            L=L,
            R=R,
            plot=plot,
            plot_bokeh=False,
            synthetic_data=True,
            gmm_model=gmm_model,
            insert_size_lookup=insert_size_lookup,
        )
        return gmm, evidence_by_mode

    return None, []


if __name__ == "__main__":
    # generate_reference_files("synthetic_data/reference", ["1:200000"])
    generate_synthetic_sv_data(
        1,
        [(100000, 100167), (100067, 100067 + 33)],
        n_samples=66,
        p=[0.5, 0.5],
        run_gmm=False,
        vcf_filename="synthetic_data/data/test.vcf",
    )
