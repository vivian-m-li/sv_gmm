import csv
import os
import sys

from src.data.query_stix import query_stix
from model.gmm_trial import gmm_trial
from src.model.dirichlet import run_dirichlet
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
    r_threshold: float | None = None,
    repulsion_stepsize: float | None = None,
    init: str | None = None,
    repulsion: bool = False,
    model_comparison_func: str | None = None,
    # stix setup
    stix_bin: str | None = None,
    stix_index: str | None = None,
    stix_database: str | None = None,
    num_stix_shards: int = 1,
    # flags
    run_split: bool = True,
    filter_reference: bool = True,
    single_trial: bool = True,
    parallel: bool = False,
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
    _, insert_size_lookup = process_input_files(
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
    plot_dir = os.path.join(output_dir, "plots")

    for directory in [output_dir, stix_file_dir]:
        if directory:
            os.makedirs(directory, exist_ok=True)
    if plot:
        os.makedirs(plot_dir, exist_ok=True)

    query_region = get_query_region(l, r, read_overlap)
    file_name = query_region.file_name

    # check if this SV has already been queried for in STIX
    stix_file = os.path.join(stix_file_dir, f"{file_name}.txt")
    if not os.path.isfile(stix_file):
        print(
            "This variant has not been previously queried or processed. Using STIX to do this now.\n"
        )
        # stix path is required if the SV evidence has not been queried for yet
        if stix_bin is None or stix_index is None or stix_database is None:
            raise FileNotFoundError(
                "Missing STIX executable, index, or database path."
            )

        # run the bash script to query stix
        stix_file = query_stix(
            query_region,
            stix_file_dir,
            stix_bin,
            stix_index,
            stix_database,
            num_stix_shards,
            parallel,
        )
    else:
        if print_messages:
            print("Using previously-queried data from", stix_file, "\n")

    # load the data as a dataframe
    reads = stix_output_to_df(stix_file)

    # remove samples queried by STIX but missing in the vcf/csv
    # in 1KG, this happens because the extended high coverage dataset includes samples that did not appear in the original study
    sample_ids = get_sample_ids(os.path.join(input_dir, sample_id_file))
    reads = reads[reads["sample_id"].isin(sample_ids)]

    # remove samples with a genotype of (0, 0)
    if filter_reference:
        ref_samples = get_reference_samples(reads, chr, start, stop, input_dir)
        reads = reads[~reads["sample_id"].isin(ref_samples)]

    if run_split:
        if reads.empty:
            print("No evidence for structural variants found in this region.")
            return

        if single_trial:
            gmm_trial(
                reads,
                chr=chr,
                L=start,
                R=stop,
                r_threshold=r_threshold,
                repulsion_stepsize=repulsion_stepsize,
                init=init,
                repulsion=repulsion,
                model_comparison_func=model_comparison_func,
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
                    "r_threshold": r_threshold,
                    "repulsion_stepsize": repulsion_stepsize,
                    "init": init,
                    "repulsion": repulsion,
                    "model_comparison_func": model_comparison_func,
                    "stem": input_dir,
                    "plot": plot,
                    "plot_file": os.path.join(plot_dir, file_name),
                    "insert_size_lookup": insert_size_lookup,
                },
            )

    return reads
