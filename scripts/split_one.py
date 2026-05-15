import argparse
import os

from src.model.split import split_sv
from src.utils.config_loader import load_config
from src.utils.model_helper import parse_input


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
        "--stix_file_dir",
        type=str,
        default=None,
        help="Output directory for STIX queries",
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
    parser.add_argument(
        "--init",
        type=str,
        default=None,
        help="Initialization method for GMM clustering",
    )
    parser.add_argument(
        "--repulsion",
        type=bool,
        default=None,
        help="Whether to apply cluster repulsion during GMM clustering",
    )
    parser.add_argument(
        "--model_comparison_func",
        type=str,
        default=None,
        help="Model comparison function to use for selecting the best model",
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
    stix_file_dir = _get(
        args.stix_file_dir, "paths", "stix_output_dir", "stix_output"
    )
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
    init = _get(args.init, "model", "init", "dp_kmeans++")
    repulsion = _get(args.repulsion, "model", "repulsion", False)
    model_comparison_func = _get(
        args.model_comparison_func, "model", "model_comparison_func", "aic"
    )

    l = parse_input(args.l) if args.l is not None else ""  # noqa741
    r = parse_input(args.r) if args.r is not None else ""
    sv_id = args.id or ""

    split_sv(
        l=l,
        r=r,
        sv_id=sv_id,
        input_dir=input_dir,
        output_dir=output_dir,
        stix_file_dir=stix_file_dir,
        sv_lookup_file=sv_lookup_file,
        sample_id_file=sample_id_file,
        insert_size_file=insert_size_file,
        read_overlap=read_overlap,
        d_threshold=d_threshold,
        r_threshold=r_threshold,
        max_penalty=max_penalty,
        init=init,
        repulsion=repulsion,
        model_comparison_func=model_comparison_func,
        stix_bin=stix_bin,
        stix_index=stix_index,
        stix_database=stix_database,
        num_stix_shards=num_stix_shards,
        single_trial=not args.d,
        plot=args.p,
    )


if __name__ == "__main__":
    main()
