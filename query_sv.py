import os
import sys
import re
import subprocess
import argparse
import csv
import pysam
import pandas as pd
import numpy as np
from process_data import run_viz_gmm, get_insert_size_lookup
from run_dirichlet import run_dirichlet
from helper import stix_output_to_df, get_sample_ids, get_deletions_df, calc_af
from gmm_types import StixQueryRegion, CHR_LENGTHS
from typing import List, Dict, Optional

# Increase the field size limit to avoid triggering the error
csv.field_size_limit(sys.maxsize)

FILE_DIR = "stix_output"
PLOT_DIR = "plots"

"""File processing functions to convert user-provided SV files into a format taken by SPLIT"""


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
        if info["SVTYPE"] != "DEL" or chr in ["X", "Y"]:
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
    df.to_csv(f"{dir}/deletions.csv", index=False)
    return df


def extract_data_from_deletions_df(
    input_dir: str = "1kgp",
    chr: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
):
    """Extracts sample ids and splits deletions into separate files by chromosome. Makes SV lookup more efficient if the SV chromosome is known."""
    deletions_df = get_deletions_df() if df is None else df

    # write sample ids for easy lookup later
    sample_ids = set(deletions_df.columns[11:-1])
    with open(f"{input_dir}/sample_ids.txt", "w") as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")

    # write an sv lookup for easier access to sv info without loading entire deletions_df
    sv_lookup_df = deletions_df[["id", "chr", "start", "stop", "svlen", "af"]]
    sv_lookup_df.to_csv(f"{input_dir}/sv_lookup.csv", index=False)

    # split the deletions into separate files by chromosome
    os.mkdir(f"{input_dir}/svs_by_chr")
    chrs = range(1, 23) if chr is None else [chr]
    for i in chrs:
        chr_df = deletions_df[deletions_df["chr"] == i]
        chr_df.to_csv(f"{input_dir}/svs_by_chr/chr{i}.csv", index=False)


def write_sample_ids_file(dir: str, df: pd.DataFrame):
    """Writes a list of sample IDs."""
    sample_ids = set(df.columns[11:-1])
    with open(f"{dir}/sample_ids.txt", "w") as f:
        for sample_id in sample_ids:
            f.write(f"{sample_id}\n")
    return sample_ids


def write_sv_lookup(dir: str, df: pd.DataFrame):
    """Writes an SV lookup file for easier access to SV info without loading entire vcf/csv."""
    sv_lookup_df = df[["id", "chr", "start", "stop", "svlen", "af"]]
    sv_lookup_df.to_csv(f"{dir}/sv_lookup.csv", index=False)


def write_svs_by_chr(dir: str, df: pd.DataFrame, chr: Optional[str] = None):
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
    with open(os.path.join(dir, "insert_sizes.csv"), "w") as f:
        f.write("sample_id,mean_insert_size\n")
        for sample_id in sample_ids:
            f.write(f"{sample_id},{default_size}\n")


def process_input_files(
    dir: str, sv_lookup_file: str, insert_size_file: Optional[str]
):
    """Processes input files for more efficient lookup during querying."""

    if sv_lookup_file.endswith(".vcf") or sv_lookup_file.endswith(".vcf.gz"):
        df = load_vcf(dir, sv_lookup_file)
        write_svs_by_chr(dir, df)
    elif not os.path.isdir(os.path.join(dir, "svs_by_chr")):
        df = pd.read_csv(os.path.join(dir, sv_lookup_file))
        write_svs_by_chr(dir, df)

    if not os.path.isfile(os.path.join(dir, "sample_ids.txt")):
        sample_ids = write_sample_ids_file(dir, df)
    else:
        sample_ids = get_sample_ids(dir)

    if not os.path.isfile(os.path.join(dir, "sv_lookup.csv")):
        write_sv_lookup(dir, df)

    if not os.path.isfile(os.path.join(dir, insert_size_file)):
        write_default_insert_sizes(dir, sample_ids)
        insert_size_lookup = get_insert_size_lookup(f"{dir}/insert_sizes.csv")
    else:
        insert_size_lookup = get_insert_size_lookup(
            os.path.join(dir, insert_size_file)
        )

    return insert_size_lookup


def giggle_format(chromosome: str, position: int):
    return f"{chromosome.lower()}:{position}"


def stix_format(s: str):
    chr, pos = s.split(":")
    return chr, int(pos)


def reverse_giggle_format(l: str, r: str):  # noqa741
    chr = l.split(":")[0]
    start = int(l.split(":")[1])
    stop = int(r.split(":")[1])
    return chr, start, stop


def lookup_sv_position(sv_id: str, dir: str = "1kgp"):
    lookup = pd.read_csv(f"{dir}/sv_lookup.csv")
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
        file_name=f"{l_chr}:{l_start}-{l_stop}_{l_chr}:{r_start}-{r_stop}",
    )


def load_processed_data(filename: str, rewrite_file: bool = False):
    """DEPRECATED: Loads previously-processed STIX output from a CSV file into a dictionary mapping sample IDs to arrays of SV evidence coordinates. Use stix_output_to_df instead."""
    squiggle_data = {}
    if not os.path.isfile(filename):
        return squiggle_data

    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 2:
                continue
            pattern = r"[\S]*([A-Z]{2}\d{5})"
            match = re.match(pattern, row[0])
            if match is None:
                print("Invalid sample ID:", row[0])
                # instead of raising an error, continue
                # raise ValueError("Invalid sample ID")
                continue
            sample_id = match.group(1)
            evidence = []
            for i in range(1, len(row), 2):
                try:
                    start, end = int(float(row[i])), int(float(row[i + 1]))
                    evidence.extend([start, end])
                except ValueError:
                    # skip this pair if either is nan
                    continue

            if len(evidence) > 0:
                squiggle_data[sample_id] = evidence

    # write squiggle data
    if rewrite_file:
        with open(filename, "w") as file:
            for sample_id, evidence in squiggle_data.items():
                evidence_str = ",".join(map(str, evidence))
                file.write(f"{sample_id},{evidence_str}\n")
    return squiggle_data


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


def get_reference_samples(
    # Gets the samples with the homozygous reference genotype (0, 0)
    reads: pd.DataFrame,
    chr: str,
    start: int,
    stop: int,
    input_dir: str,
) -> List[str]:
    df = pd.read_csv(f"{input_dir}/svs_by_chr/chr{chr}.csv")
    row = df[(df["start"] == start) & (df["stop"] == stop)]
    if row.empty:  # query region does not correspond with an SV in the callset
        return []
    samples = reads["sample_id"].tolist()
    ref_samples = [
        sample_id for sample_id in samples if row.iloc[0][sample_id] == "(0, 0)"
    ]
    return ref_samples


def query_stix_bash(
    query_region: StixQueryRegion,
    output_dir: str,
    stix_bin: str,
    index_path: str,
    database_path: str,
    num_shards: int,
):
    """
    Runs a bash query file to query STIX for all the read (paired-end and split
    data within the defined coordinate space.
    num_shards: number of shards the stix index and database are split into
    - this must be defined properly to pull all of the relevant data
    """
    partial_outputs_dir = os.path.join(output_dir, "partial_outputs")
    if num_shards > 1:
        if not os.path.exists(partial_outputs_dir):
            os.mkdir(partial_outputs_dir)
        stix_output_file = os.path.join(
            output_dir, "partial_outputs", query_region.file_name
        )
    else:
        stix_output_file = os.path.join(output_dir, query_region.file_name)

    stix_path = "/".join(index_path.split("/")[:-1])
    subprocess.run(
        ["bash", "query_stix.sh"]
        + [  # noqa503
            f"{query_region.chr}:{query_region.left_start}-{query_region.left_stop}",
            f"{query_region.chr}:{query_region.right_start}-{query_region.right_stop}",
            stix_path,
            index_path,
            database_path,
            str(num_shards),
            stix_output_file,
            stix_bin,
        ],
        capture_output=True,
        text=True,
    )

    # the query output from each shard was written to a different file, and we want to write them to the same file
    if num_shards > 1:
        output_file = os.path.join(output_dir, f"{query_region.file_name}.txt")
        with open(output_file, "w") as out_file:
            for i in range(num_shards):
                temp_file = os.path.join(
                    partial_outputs_dir, f"{query_region.file_name}_{i}.txt"
                )
                with open(temp_file, "r") as partial_file:
                    # write the partial file to the main file
                    out_file.write(partial_file.read())

                # remove partial file
                os.remove(temp_file)
        os.rmdir(partial_outputs_dir)
    else:
        output_file = f"{stix_output_file}.txt"

    return output_file


def write_processed_output(
    output_file: str,
    processed_output_file: str,
    l_col: str = "l_end",
    r_col: str = "r_start",
) -> Dict[str, np.ndarray[float]]:
    """
    DEPRECATED: reads are now handled directly from the raw read file in the stix_output directory
    Parses the raw stix output (from the patched -g version) into pairs of coordinates for each sample. Each pair represents the start/stop of the deletion.
    Uses l_end and r_start (the tightest bounds of the left/right reads) to calculate the deletion.
    For short reads, split reads are weighted above discordant read pairs due to their higher accuracy in defining breakpoints.
    All long reads returned from stix are split reads, so the l_start and r_end coordinates capture the entire read length (not just the deletion).
    """
    df = stix_output_to_df(output_file)

    grouped = df.groupby("sample_id")
    squiggle_data = {}
    processed_stix_output = []
    for _, group in grouped:
        ls = group[l_col].tolist()
        rs = group[r_col].tolist()
        sample_id = group["sample_id"].iloc[0]
        sv_evidence = [item for pair in zip(ls, rs) for item in pair]

        squiggle_data[sample_id] = np.array(sv_evidence)
        sv_evidence = [sample_id] + sv_evidence
        processed_stix_output.append(sv_evidence)

    with open(processed_output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(processed_stix_output)

    return squiggle_data


def query_stix(
    *,
    l: str = "",
    r: str = "",
    sv_id: str = "",
    input_dir: str = "assets",
    output_dir: Optional[str] = None,
    sv_lookup_file: str = "deletions.csv",
    insert_size_file: str = "insert_sizes.csv",
    stix_bin: Optional[str] = None,
    stix_index: Optional[str] = None,
    stix_database: Optional[str] = None,
    num_stix_shards: int = 1,
    run_gmm: bool = True,
    filter_reference: bool = True,
    single_trial: bool = True,
    plot: bool = True,
    long_reads: bool = False,
):
    # check that the user inputted an sv id or sv coordinates
    if sv_id == "" and (l == "" or r == ""):
        raise ValueError("Missing SV coordinates or ID")

    # check for required input files and process for more efficient lookup
    if not os.path.isfile(os.path.join(input_dir, sv_lookup_file)):
        raise FileNotFoundError(f"SV lookup file {sv_lookup_file} not found.")

    # write input files that will be used later on during querying
    insert_size_lookup = process_input_files(
        input_dir, sv_lookup_file, insert_size_file
    )

    # if an SV id is provided, then get the chromosomes for that
    if sv_id != "":
        chr, start, stop = lookup_sv_position(sv_id, input_dir)
        l = giggle_format(chr, start)  # noqa741
        r = giggle_format(chr, stop)
    else:
        chr, start, stop = reverse_giggle_format(l, r)

    # Note: x/y chromosomes are ignored in the analysis and are not queried by the script
    if chr.lower() in ["x", "y"]:
        raise ValueError("X/Y chromosomes are not supported.")

    # set filepaths
    if output_dir is None:
        output_dir = input_dir
    output_file_dir = os.path.join(output_dir, FILE_DIR)
    plot_dir = os.path.join(output_dir, PLOT_DIR)

    for directory in [
        output_dir,
        output_file_dir,
        plot_dir,
    ]:
        if not os.path.exists(directory) and directory != "":
            os.mkdir(directory)

    # deprecated: previously, stix queries weren't run with left/right regions, so queries were identified by l_r only
    # file_name = f"{l}_{r}"  # base file name
    query_region = get_query_region(l, r)
    file_name = query_region.file_name

    # check if this SV has already been queried for in STIX
    # check multiple locations for this file
    output_file = None
    for directory in [
        input_dir,
        output_dir,
        os.path.join(input_dir, FILE_DIR),
        output_file_dir,
    ]:
        if os.path.isfile(f"{directory}/{file_name}.txt"):
            output_file = f"{directory}/{file_name}.txt"
            break
    if output_file is None:
        print(
            "This variant has not been previously quiered or processed. Using STIX to do this now.\n"
        )
        # stix path is required if the SV evidence has not been queried for yet
        if stix_bin is None or stix_index is None or stix_database is None:
            raise FileNotFoundError(
                "Missing STIX executable, index, or database path."
            )

        # run the bash script to query stix
        output_file = query_stix_bash(
            query_region,
            output_file_dir,
            stix_bin,
            stix_index,
            stix_database,
            num_stix_shards,
        )
    else:
        print("Using previously-queried data from", output_file, "\n")

    # load the data as a dataframe
    reads = stix_output_to_df(output_file)

    # remove samples queried by STIX but missing in the vcf/csv
    # in 1KG, this happens because the extended high coverage dataset includes samples that did not appear in the original study
    sample_ids = get_sample_ids(input_dir)
    reads = reads[reads["sample_id"].isin(sample_ids)]

    # remove samples with a genotype of (0, 0)
    if filter_reference:
        ref_samples = get_reference_samples(reads, chr, start, stop, input_dir)
        reads = reads[~reads["sample_id"].isin(ref_samples)]

    if run_gmm:
        if reads.empty:
            print("No evidence for structural variants found in this region.")
            return

        if single_trial:
            run_viz_gmm(
                reads,
                chr=chr,
                L=start,
                R=stop,
                plot=plot,
                stem=input_dir,
                plot_file=f"{plot_dir}/{file_name}",
                insert_size_lookup=insert_size_lookup,
            )
        else:
            run_dirichlet(
                reads,
                insert_size_file=f"{input_dir}/{insert_size_file}",
                **{
                    "chr": chr,
                    "L": start,
                    "R": stop,
                    "stem": input_dir,
                    "plot": plot,
                    "plot_file": f"{plot_dir}/{file_name}",
                },
            )

    return reads


def main():
    """
    Main function to parse command line arguments and call query_stix.
    Arguments:
    -l: Left position of the structural variant, format=chromosome:position
    -r: Right position of the structural variant, format=chromosome:position
    -id: Structural variant ID from 1000 Genomes Project
    -p: Plot the length and L coordinate of each sample (default: False)
    -d: Rerun the SV until >= 80% confident in the outcome (default: True)
    -lr: Cluster using long-read data instead of the default short-reads (default: False)
    --input_dir: Input directory for incoming data (default: assets)
    --output_dir: Output directory for processed data (default: None)
    --sv_lookup: VCF or CSV file with structural variants (default: deletions.csv)
    --insert_size_file: Insert size for each sample (default: insert_sizes.csv)
    --stix_bin: Path to STIX executable
    --stix_index: Path to STIX index for querying sample reads
    --stix_database: Path to STIX database for querying sample reads
    --num_stix_shards: Number of shards the STIX index and database are split into (default: 1)
    """

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Queries structural variants in a specific region"
    )
    parser.add_argument(
        "-l",
        type=str,
        help="Left position of the structural variant, format=chromosome:position",
    )
    parser.add_argument(
        "-r",
        type=str,
        help="Right position of the structural variant, format=chromosome:position",
    )
    parser.add_argument(
        "-id",
        type=str,
        help="Structural variant ID from 1000 Genomes Project",
    )
    parser.add_argument(
        "-p",
        type=bool,
        help="Plot the length and L coordinate of each sample",
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "-d",
        type=bool,
        help="Rerun the SV until >= 80% confident in the outcome",
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "-lr",
        type=bool,
        help="Cluster using long-read data instead of the default short-reads",
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "-ref", type=str, help="Reference genome", default="grch38"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory for incoming data",
        default="assets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--sv_lookup",
        type=str,
        help="VCF or CSV file with structural variants",
        default="deletions.csv",
    )
    parser.add_argument(
        "--insert_size_file",
        type=str,
        help="Insert size for each sample. Defaults to 450 bp if not provided.",
        default="insert_sizes.csv",
    )
    parser.add_argument(
        "--stix_bin",
        type=str,
        help="Path to STIX executable.",
    )
    parser.add_argument(
        "--stix_index",
        type=str,
        help="Path to STIX index for querying sample reads.",
    )
    parser.add_argument(
        "--stix_database",
        type=str,
        help="Path to STIX database for querying sample reads.",
    )
    parser.add_argument(
        "--num_stix_shards",
        type=int,
        help="Number of shards the STIX index and database are split into.",
        default=1,
    )

    args = parser.parse_args()
    l = parse_input(args.l) if args.l is not None else ""  # noqa741
    r = parse_input(args.r) if args.r is not None else ""
    sv_id = args.id or ""

    print("Using the following arguments:", args, "\n")

    query_stix(
        l=l,
        r=r,
        sv_id=sv_id,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sv_lookup_file=args.sv_lookup,
        insert_size_file=args.insert_size_file,
        stix_bin=args.stix_bin,
        stix_index=args.stix_index,
        stix_database=args.stix_database,
        num_stix_shards=args.num_stix_shards,
        single_trial=not args.d,
        plot=args.p,
        long_reads=args.lr,
    )


if __name__ == "__main__":
    main()
