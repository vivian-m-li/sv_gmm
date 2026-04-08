import argparse
import multiprocessing
import os
import shutil
import time

from src.model.dirichlet import run_dirichlet
from src.utils.config_loader import load_config
from src.utils.helper import get_deletions_df, get_sample_ids
from src.utils.timeout import break_after
from src.utils.write_sv_output import (
    get_raw_data,
    init_sv_stat_row,
    write_sv_stats,
    write_posterior_distributions,
    concat_multi_processed_sv_files,
    write_post_processed_files,
)


def run_dirichlet_inner(
    row: dict,
    population_size: int,
    stem: str,
    intermediate_output_dir: str,
    model_params: dict,
):
    sv_id = row["id"]
    reads, num_samples = get_raw_data(row, stem)

    if reads.empty:
        # if no samples have any data supporting the SV then not included
        gmms, alphas, posterior_distributions = [(None, [])], [], []
    else:
        gmms, alphas, posterior_distributions = run_dirichlet(
            reads,
            **{
                "chr": row["chr"],
                "L": row["start"],
                "R": row["stop"],
                "plot": False,
                "stem": stem,
                **model_params,  # d_threshold, r_threshold, max_penalty
            },
        )

    for i, (gmm, evidence_by_mode) in enumerate(gmms):
        sv_stat = init_sv_stat_row(
            row,
            num_samples=num_samples,
            num_reference=num_samples - reads["sample_id"].nunique(),
        )
        write_sv_stats(
            sv_stat,
            gmm,
            evidence_by_mode,
            population_size,
            intermediate_output_dir,
            i,
        )

    if gmms[0][0] is not None:
        write_posterior_distributions(
            sv_id, alphas, posterior_distributions, intermediate_output_dir
        )


def run_dirichlet_wrapper(
    row: dict,
    population_size: int,
    stem: str,
    intermediate_output_dir: str,
    model_params: dict,
):
    try:
        run_dirichlet_inner(
            row, population_size, stem, intermediate_output_dir, model_params
        )
    except Exception as e:
        print(f"Error processing SV {row['id']}: {e}")


@break_after(hours=30, minutes=00)
def run_svs_until_convergence(
    stem: str,
    intermediate_output_dir: str,
    sample_ids: set[str],
    model_params: dict,
):
    deletions_df = get_deletions_df(stem)
    population_size = len(sample_ids)

    with multiprocessing.Manager():
        cpu_count = multiprocessing.cpu_count()
        p = multiprocessing.Pool(cpu_count)
        args = [
            (
                row.to_dict(),
                population_size,
                stem,
                intermediate_output_dir,
                model_params,
            )
            for _, row in deletions_df.iterrows()
        ]
        p.starmap(run_dirichlet_wrapper, args)
        p.close()
        p.join()


def run_svs(config_path: str = "config.toml"):
    cfg = load_config(config_path)

    input_dir = cfg["paths"]["input_dir"]
    output_dir = cfg["paths"]["output_dir"]
    intermediate_output_dir = cfg["paths"]["intermediate_output_dir"]
    local_intermediate_output_dir = cfg["paths"][
        "local_intermediate_output_dir"
    ]

    sample_id_file = cfg["input_files"]["sample_id_file"]
    sample_ids = get_sample_ids(sample_id_file)

    # load model parameters from the config file
    raw_model = cfg.get("model", {})
    model_params = {
        k: v
        for k, v in {
            "d_threshold": raw_model.get("d_threshold"),
            "r_threshold": raw_model.get("r_threshold"),
            "max_penalty": raw_model.get("max_penalty"),
        }.items()
        if v is not None
    }

    os.makedirs(intermediate_output_dir, exist_ok=True)
    os.makedirs(local_intermediate_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()
    run_svs_until_convergence(input_dir, intermediate_output_dir, model_params)

    # move files from scratch to home dir (even after timeout)
    if intermediate_output_dir != local_intermediate_output_dir:
        for file in os.listdir(intermediate_output_dir):
            shutil.move(
                os.path.join(intermediate_output_dir, file),
                os.path.join(local_intermediate_output_dir, file),
            )

    print("Concatenating multi-processed SV files...")
    concat_multi_processed_sv_files(
        local_intermediate_output_dir, "all_split_trials.csv", output_dir
    )
    print("Writing post-processed files...")
    write_post_processed_files(input_dir, output_dir, sample_ids)

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
    run_svs(config_path=args.config)
