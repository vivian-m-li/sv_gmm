import argparse
import csv
import os
import sys

from src.data.query_stix import query_stix
from model.gmm import gmm_trial
from src.model.dirichlet import run_dirichlet
from src.utils.config_loader import load_config
from src.utils.helper import (
    stix_output_to_df,
    get_sample_ids,
)
from src.utils.model_helper import (
    get_reference_samples,
    process_input_files,
    lookup_sv_position,
    giggle_format,
    reverse_giggle_format,
    get_query_region,
    parse_input,
)

# Increase the field size limit to avoid triggering the error
csv.field_size_limit(sys.maxsize)


def split_sv(
    *,
    # sv identifier
    l: str = "",
    r: str = "",
    sv_id: str = "",
    # file i/o
    input_dir: str = "assets",
    output_dir: str | None = None,
    sv_lookup_file: str = "deletions.csv",
    insert_size_file: str | None = None,
    default_insert_size: int | None = None,
    sample_id_file: str | None = None,
    stix_file_dir: str = "stix_output",
    # query/model parameters
    read_overlap: float = 1.0,
    d_threshold: int | None = None,
    r_threshold: float | None = None,
    max_penalty: int | None = None,
    # stix setup
    stix_bin: str | None = None,
    stix_index: str | None = None,
    stix_database: str | None = None,
    num_stix_shards: int = 1,
    # flags
    gmm: bool = True,
    filter_reference: bool = True,
    single_trial: bool = True,
    plot: bool = True,
    print_messages: bool = True,
):
    # check that the user inputted an sv id or sv coordinates
    if sv_id == "" and (l == "" or r == ""):
        raise ValueError("Missing SV coordinates or ID")

    # check for required input files and process for more efficient lookup
    if not os.path.isfile(os.path.join(input_dir, sv_lookup_file)):
        raise FileNotFoundError(f"SV lookup file {sv_lookup_file} not found.")

    # write input files that will be used later on during querying
    insert_size_lookup = process_input_files(
        input_dir,
        sv_lookup_file,
        sample_id_file,
        insert_size_file,
        default_insert_size,
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
    output_file_dir = os.path.join(output_dir, stix_file_dir)
    plot_dir = os.path.join(output_dir, "plots")

    for directory in [
        output_dir,
        output_file_dir,
        plot_dir,
    ]:
        if not os.path.exists(directory) and directory != "":
            os.mkdir(directory)

    query_region = get_query_region(l, r, read_overlap)
    file_name = query_region.file_name

    # check if this SV has already been queried for in STIX
    # check multiple locations for this file
    output_file = None
    for directory in [
        input_dir,
        output_dir,
        os.path.join(input_dir, stix_file_dir),
        output_file_dir,
    ]:
        file_path = os.path.join(directory, f"{file_name}.txt")
        if os.path.isfile(file_path):
            output_file = file_path
            break
    if output_file is None:
        print(
            "This variant has not been previously queried or processed. Using STIX to do this now.\n"
        )
        # stix path is required if the SV evidence has not been queried for yet
        if stix_bin is None or stix_index is None or stix_database is None:
            raise FileNotFoundError(
                "Missing STIX executable, index, or database path."
            )

        # run the bash script to query stix
        output_file = query_stix(
            query_region,
            output_file_dir,
            stix_bin,
            stix_index,
            stix_database,
            num_stix_shards,
        )
    else:
        if print_messages:
            print("Using previously-queried data from", output_file, "\n")

    # load the data as a dataframe
    reads = stix_output_to_df(output_file)

    # remove samples queried by STIX but missing in the vcf/csv
    # in 1KG, this happens because the extended high coverage dataset includes samples that did not appear in the original study
    sample_ids = get_sample_ids(os.path.join(input_dir, sample_id_file))
    reads = reads[reads["sample_id"].isin(sample_ids)]

    # remove samples with a genotype of (0, 0)
    if filter_reference:
        ref_samples = get_reference_samples(reads, chr, start, stop, input_dir)
        reads = reads[~reads["sample_id"].isin(ref_samples)]

    if gmm:
        if reads.empty:
            print("No evidence for structural variants found in this region.")
            return

        if single_trial:
            gmm_trial(
                reads,
                chr=chr,
                L=start,
                R=stop,
                d_threshold=d_threshold,
                r_threshold=r_threshold,
                max_penalty=max_penalty,
                plot=plot,
                stem=input_dir,
                plot_file=os.path.join(plot_dir, file_name),
                insert_size_lookup=insert_size_lookup,
            )
        else:
            run_dirichlet(
                reads,
                insert_size_file=os.path.join(input_dir, insert_size_file),
                **{
                    "chr": chr,
                    "L": start,
                    "R": stop,
                    "d_threshold": d_threshold,
                    "r_threshold": r_threshold,
                    "max_penalty": max_penalty,
                    "stem": input_dir,
                    "plot": plot,
                    "plot_file": os.path.join(plot_dir, file_name),
                    "insert_size_lookup": insert_size_lookup,
                },
            )

    return reads


def main():
    """
    Main function to parse command line arguments and call query_stix.

    All file paths and model parameters can be supplied either via a config
    file (--config) or as explicit CLI flags. Explicit flags take precedence
    over config-file values.

    Arguments:
    --config:           Path to a TOML config file (default: config.toml if present)
    -l:                 Left position of the SV, format=chromosome:position
    -r:                 Right position of the SV, format=chromosome:position
    -id:                Structural variant ID
    -p:                 Plot the length and L coordinate of each sample
    -d:                 Rerun until >= 80% confident
    --input_dir:        Input directory
    --output_dir:       Output directory
    --sv_lookup:        VCF or CSV file with structural variants
    --sample_id_file:   Text file with sample IDs
    --insert_size_file: Insert size for each sample
    --read_overlap:     Overlap allowed when querying STIX
    --stix_bin:         Path to STIX executable
    --stix_index:       Path to STIX index
    --stix_database:    Path to STIX database
    --num_stix_shards:  Number of shards for STIX index/database
    --d_threshold:      Distance threshold for cluster merging (model param)
    --r_threshold:      Reciprocal-overlap threshold (model param)
    --max_penalty:      Maximum penalty for spurious cluster suppression (model param)
    """
    parser = argparse.ArgumentParser(
        description="Queries structural variants in a specific region"
    )

    # config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a TOML config file. CLI flags override config values.",
    )

    # SV identifier - either ID or l and r coordinates can be provided
    parser.add_argument(
        "-l", type=str, help="Left position, format=chromosome:position"
    )
    parser.add_argument(
        "-r", type=str, help="Right position, format=chromosome:position"
    )
    parser.add_argument("-id", type=str, help="Structural variant ID")

    # flags
    parser.add_argument(
        "-p",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="Plot the length and L coordinate of each sample",
    )
    parser.add_argument(
        "-d",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="Rerun the SV until >= 80% confident in the outcome",
    )
    # TODO: make sure reference genome is accessible in asset files
    parser.add_argument(
        "-ref", type=str, help="Reference genome", default="grch38"
    )

    # I/O paths
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Input directory for incoming data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--sv_lookup",
        type=str,
        default=None,
        help="VCF or CSV file with structural variants",
    )
    parser.add_argument(
        "--sample_id_file",
        type=str,
        default=None,
        help="Text file with sample IDs",
    )
    parser.add_argument(
        "--insert_size_file",
        type=str,
        default=None,
        help="Insert size for each sample",
    )
    parser.add_argument(
        "--read_overlap",
        type=float,
        default=None,
        help="Overlap fraction allowed when querying STIX",
    )

    # STIX paths
    parser.add_argument(
        "--stix_bin",
        type=str,
        default=None,
        help="Path to STIX executable",
    )
    parser.add_argument(
        "--stix_index",
        type=str,
        default=None,
        help="Path to STIX index",
    )
    parser.add_argument(
        "--stix_database",
        type=str,
        default=None,
        help="Path to STIX database",
    )
    parser.add_argument(
        "--num_stix_shards",
        type=int,
        default=None,
        help="Number of shards the STIX index/database are split into.",
    )

    # Model parameters
    parser.add_argument(
        "--d_threshold",
        type=int,
        default=None,
        help="Distance threshold for penalizing cluster assignments (bp)",
    )
    parser.add_argument(
        "--r_threshold",
        type=float,
        default=None,
        help="Reciprocal-overlap threshold for penalizing cluster assignments",
    )
    parser.add_argument(
        "--max_penalty",
        type=int,
        default=None,
        help="Maximum penalty for spurious cluster suppression",
    )
    args = parser.parse_args()

    # load config, then let CLI flags override individual values
    config_path = args.config or (
        "config.toml" if os.path.isfile("config.toml") else None
    )
    cfg: dict = {}
    if config_path is not None:
        cfg = load_config(config_path)

    def _get(cli_val, section: str, key: str, fallback=None):
        """Return cli_val if explicitly provided, else config value, else fallback."""
        if cli_val is not None:
            return cli_val
        return cfg.get(section, {}).get(key, fallback)

    input_dir = _get(args.input_dir, "paths", "input_dir", "assets")
    output_dir = _get(args.output_dir, "paths", "output_dir", None)
    sv_lookup_file = _get(
        args.sv_lookup, "input_files", "sv_lookup_file", "deletions.csv"
    )
    sample_id_file = _get(
        args.sample_id_file, "input_files", "sample_id_file", "sample_ids.txt"
    )
    insert_size_file = _get(
        args.insert_size_file,
        "input_files",
        "insert_size_file",
        "insert_sizes.csv",
    )
    read_overlap = _get(args.read_overlap, "query", "read_overlap", 1.0)
    stix_bin = _get(args.stix_bin, "stix", "bin", None) or None
    stix_index = _get(args.stix_index, "stix", "index", None) or None
    stix_database = _get(args.stix_database, "stix", "database", None) or None
    num_stix_shards = _get(args.num_stix_shards, "stix", "num_shards", 1)
    d_threshold = _get(args.d_threshold, "model", "d_threshold", 50)
    r_threshold = _get(args.r_threshold, "model", "r_threshold", 0.9)
    max_penalty = _get(args.max_penalty, "model", "max_penalty", 100)

    l = parse_input(args.l) if args.l is not None else ""  # noqa741
    r = parse_input(args.r) if args.r is not None else ""
    sv_id = args.id or ""

    query_stix(
        l=l,
        r=r,
        sv_id=sv_id,
        input_dir=input_dir,
        output_dir=output_dir,
        sv_lookup_file=sv_lookup_file,
        sample_id_file=sample_id_file,
        insert_size_file=insert_size_file,
        read_overlap=read_overlap,
        d_threshold=d_threshold,
        r_threshold=r_threshold,
        max_penalty=max_penalty,
        stix_bin=stix_bin,
        stix_index=stix_index,
        stix_database=stix_database,
        num_stix_shards=num_stix_shards,
        single_trial=not args.d,
        plot=args.p,
    )


if __name__ == "__main__":
    main()
