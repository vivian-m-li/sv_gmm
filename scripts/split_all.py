import argparse
import multiprocessing
import os
import shutil
import time

import pandas as pd
from pprint import pprint

from src.model.dirichlet import run_dirichlet
from src.utils.config_loader import load_config
from src.utils.helper import get_sample_ids
from src.utils.model_helper import process_input_files
from src.utils.timeout import break_after
from src.utils.write_sv_output import (
    get_raw_data,
    init_sv_stat_row,
    write_sv_stats,
    write_posterior_distributions,
    concat_multi_processed_sv_files,
    write_post_processed_files,
)

SLURM_CPUS = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))


def run_split_trial(
    row: dict,
    cfg: dict,
    population_size: int,
    insert_size_lookup: dict[str, int],
):
    sv_id = row["id"]
    input_dir = cfg["paths"]["input_dir"]
    reads, num_samples = get_raw_data(
        row, cfg, filter_reference_samples=True, print_messages=False
    )

    # load model parameters from the config file
    raw_model = cfg.get("model", {})
    model_params = {
        k: v
        for k, v in {
            "r_threshold": raw_model.get("r_threshold"),
            "repulsion_stepsize": raw_model.get("repulsion_stepsize"),
            "init": raw_model.get("init"),
            "repulsion": raw_model.get("repulsion"),
            "model_comparison_func": raw_model.get("model_comparison_func"),
        }.items()
        if v is not None
    }
    model_params["insert_size_lookup"] = insert_size_lookup

    if reads.empty:
        # if no samples have any data supporting the SV then not included
        gmm_results, alphas, posterior_distributions = [(None, [])], [], []
    else:
        gmm_results, alphas, posterior_distributions = run_dirichlet(
            reads,
            **{
                "chr": row["chr"],
                "L": row["start"],
                "R": row["stop"],
                "plot": False,
                "stem": input_dir,
                **model_params,
            },
        )

    intermediate_output_dir = cfg["paths"]["intermediate_output_dir"]
    for i, (gmm_result, evidence_by_mode) in enumerate(gmm_results):
        sv_stat = init_sv_stat_row(
            row,
            num_samples=num_samples,
            num_reference=num_samples - reads["sample_id"].nunique(),
        )
        write_sv_stats(
            sv_stat,
            gmm_result,
            evidence_by_mode,
            population_size,
            intermediate_output_dir,
            i,
        )

    if gmm_results[0][0] is not None:
        write_posterior_distributions(
            sv_id, alphas, posterior_distributions, intermediate_output_dir
        )


def run_split_wrapper(
    row: dict,
    cfg: dict,
    population_size: int,
    insert_size_lookup: dict[str, int],
):
    try:
        run_split_trial(row, cfg, population_size, insert_size_lookup)
    except Exception as e:
        print(f"Error processing SV {row['id']}: {e}")
        raise Exception("Exiting split_all...")


@break_after(hours=62, minutes=00)
def split_all(
    cfg: dict, sample_ids: set[str], insert_size_lookup: dict[str, int]
):
    input_dir = cfg["paths"]["input_dir"]
    sv_lookup_file = cfg["input_files"]["sv_lookup_file"]
    svs = pd.read_csv(os.path.join(input_dir, sv_lookup_file), low_memory=False)
    population_size = len(sample_ids)

    with multiprocessing.Manager():
        p = multiprocessing.Pool(SLURM_CPUS)
        args = [
            (row.to_dict(), cfg, population_size, insert_size_lookup)
            for _, row in svs.iterrows()
        ]
        p.starmap(run_split_wrapper, args)
        p.close()
        p.join()


def main(config_path: str = "config.toml"):
    cfg = load_config(config_path)
    print("Config arguments:")
    pprint(cfg)

    input_dir = cfg["paths"]["input_dir"]
    output_dir = cfg["paths"]["output_dir"]
    intermediate_output_dir = cfg["paths"]["intermediate_output_dir"]
    local_intermediate_output_dir = cfg["paths"][
        "local_intermediate_output_dir"
    ]

    sample_id_file = cfg["input_files"]["sample_id_file"]
    sample_ids = get_sample_ids(os.path.join(input_dir, sample_id_file))

    # write input files that will be used later on during querying
    _, insert_size_lookup = process_input_files(
        input_dir,
        cfg["input_files"]["sv_lookup_file"],
        sample_id_file,
        cfg["input_files"].get("insert_size_file"),
        cfg["model"].get("default_insert_size"),
    )
    sv_lookup_file = cfg["input_files"]["sv_lookup_file"]
    if sv_lookup_file.endswith(".vcf"):
        cfg["input_files"]["sv_lookup_file"] = (
            sv_lookup_file.strip(".vcf") + ".csv"
        )
    elif sv_lookup_file.endswith(".vcf.gz"):
        cfg["input_files"]["sv_lookup_file"] = (
            sv_lookup_file.strip(".vcf.gz") + ".csv"
        )

    os.makedirs(intermediate_output_dir, exist_ok=True)
    os.makedirs(local_intermediate_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()

    # this is the main function that splits each SV
    split_all(cfg, sample_ids, insert_size_lookup)

    # move files from scratch to home dir (even after timeout)
    if intermediate_output_dir != local_intermediate_output_dir:
        for file in os.listdir(intermediate_output_dir):
            shutil.move(
                os.path.join(intermediate_output_dir, file),
                os.path.join(local_intermediate_output_dir, file),
            )

    print("Concatenating multi-processed SV files...")
    concat_multi_processed_sv_files(
        output_dir, local_intermediate_output_dir, "all_split_trials.csv"
    )
    print("Writing post-processed files...")
    sv_lookup = pd.read_csv(
        os.path.join(input_dir, cfg["input_files"]["sv_lookup_file"]),
        low_memory=False,
    )
    write_post_processed_files(
        output_dir,
        sample_ids,
        sv_lookup,
        ancestry_file=os.path.join(
            input_dir, cfg["input_files"].get("ancestry_file")
        ),
    )

    end = time.time()
    elapsed_time = end - start
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, _ = divmod(remainder, 60)
    print(f"Completed in {int(hours)}h {int(minutes)}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SPLIT across an entire SV callset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to the TOML configuration file (default: config.toml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
