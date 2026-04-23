import multiprocessing
import os
import shutil

from ax.adapter.registry import Generators
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.transition_criterion import MinTrials
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.service.ax_client import AxClient, ObjectiveProperties
import numpy as np
import pandas as pd

from src.data.query_stix import query_stix
from src.model.dirichlet import run_dirichlet
from src.utils.model_helper import (
    giggle_format,
    get_query_region,
    calc_pr_auc,
    get_insert_size_lookup,
)
from src.utils.write_sv_output import (
    get_raw_data,
    init_sv_stat_row,
    write_sv_stats,
    write_posterior_distributions,
    concat_multi_processed_sv_files,
    write_post_processed_files,
)

SLURM_CPUS = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))


def get_confusion_matrix(
    sv_subset: pd.DataFrame,
    svs_n_modes: pd.DataFrame,
    q: float,
) -> dict[str, float]:
    """
    Calculate confusion matrix based on predicted vs actual number of SVs.
    The actual number of SVs are determined based on the long read data
    and are defined in the sv_subset.csv file.
    """
    # remove rows that didn't run due to lack of data
    svs_n_modes = svs_n_modes.rename(
        columns={"sv_id": "id", "num_modes": "n_svs_predicted"},
    )
    svs_n_modes = svs_n_modes[svs_n_modes["confidence"] != "inconclusive"]
    if "q" in sv_subset.columns:
        sv_subset_q = sv_subset[sv_subset["q"] == q]
    else:
        sv_subset_q = sv_subset
    merged = sv_subset_q.merge(svs_n_modes, on="id", how="right")

    TP = merged[
        (merged["n_svs_actual"] == 2) & (merged["n_svs_predicted"] >= 2)
    ].shape[0]
    FP = merged[
        (merged["n_svs_actual"] == 1) & (merged["n_svs_predicted"] >= 2)
    ].shape[
        0
    ]  # this covers cases where we predicted 3 modes when actual is 1 or 2
    FN = merged[
        (merged["n_svs_actual"] == 2) & (merged["n_svs_predicted"] == 1)
    ].shape[0]
    TN = merged[
        (merged["n_svs_actual"] == 1) & (merged["n_svs_predicted"] == 1)
    ].shape[0]

    values = {"TP": TP, "FP": FP, "FN": FN, "TN": TN}
    return values


def run_dirichlet_inner(
    row: dict,
    cfg: dict,
    output_dir: str,
    sample_ids: set[str],
    insert_size_lookup: dict[str, int],
    q: float,
    d: int,
    r: float,
    pen: int,
):
    stix_output_dir = f"{cfg['paths']['stix_output_dir']}_{q}"
    cfg_copy = cfg.copy()
    cfg_copy["paths"]["stix_output_dir"] = stix_output_dir

    # get reads from stix file and filter reference samples
    reads, num_samples = get_raw_data(
        row,
        cfg_copy,
        read_overlap=q,
        filter_reference_samples=True,
        samples_to_keep=list(
            sample_ids
        ),  # keep only samples with long reads (used in genotyping)
        print_messages=False,
    )

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
                "d_threshold": d,
                "r_threshold": r,
                "max_penalty": pen,
                "synthetic_data": True,
                "plot": False,
                "stem": cfg["paths"]["input_dir"],
                "insert_size_lookup": insert_size_lookup,
            },
        )

    population_size = len(sample_ids)
    for i, (gmm_result, evidence_by_mode) in enumerate(gmm_results):
        sv_stat = init_sv_stat_row(
            row,
            num_samples=num_samples,
            num_reference=num_samples - reads["sample_id"].nunique(),
        )
        # TODO: in this function, a mode is getting added as a separate row. could just be a one-time issue
        write_sv_stats(
            sv_stat,
            gmm_result,
            evidence_by_mode,
            population_size,
            output_dir,
            i,
        )

    if gmm_results[0][0] is not None:
        write_posterior_distributions(
            row["id"], alphas, posterior_distributions, output_dir
        )


def run_calibration_test(
    sv_df: pd.DataFrame,
    cfg: dict,
    *,
    d: int,
    r: float,
    q: float,
    pen: int,
    sample_ids: set[str],
    insert_size_lookup: dict[str, int],
):
    """
    Run a single calibration test for given distance and reciprocal overlap thresholds.
    TODO: do analysis later on which combinations led to "partial correctness"
    """
    print(f"Running calibration for d={d}, r={r}, q={q}, p={pen}")
    processed_file_dir = os.path.join(
        cfg["paths"]["intermediate_output_dir"],
        "d{}_r{:.2f}_p{}".format(d, r, pen),
    )
    if not os.path.exists(processed_file_dir):
        os.makedirs(processed_file_dir)

    with multiprocessing.Manager():
        p = multiprocessing.Pool(SLURM_CPUS)
        args = []
        for _, row in sv_df.iterrows():
            args.append(
                (
                    row,
                    cfg,
                    processed_file_dir,
                    sample_ids,
                    insert_size_lookup,
                    q,
                    d,
                    r,
                    pen,
                )
            )

        p.starmap(run_dirichlet_inner, args)
        p.close()
        p.join()

    output_dir = cfg["paths"]["output_dir"]
    results_dir = os.path.join(
        output_dir, "d{}_r{:.2f}_q{:.2f}_p{}".format(d, r, q, pen)
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    concat_multi_processed_sv_files(
        results_dir, processed_file_dir, "all_split_trials.csv"
    )
    write_post_processed_files(results_dir, sample_ids, sv_df[["id"]])
    shutil.rmtree(processed_file_dir)

    svs_n_modes = pd.read_csv(os.path.join(results_dir, "svs_n_modes.csv"))

    columns = ["id", "n_svs_actual"]
    if "q" in sv_df.columns:
        columns.append("q")
    results = get_confusion_matrix(sv_df[columns], svs_n_modes, q)

    final_output_file = os.path.join(output_dir, "results.csv")
    if not os.path.exists(final_output_file):
        with open(final_output_file, "w") as f:
            f.write("d,r,q,p,TP,FP,FN,TN\n")
    with open(final_output_file, "a") as f:
        f.write(
            "{},{},{},{},{},{},{},{}\n".format(
                d,
                r,
                q,
                pen,
                results["TP"],
                results["FP"],
                results["FN"],
                results["TN"],
            )
        )

    return results


def snap_to_grid(value: float, min_val: float, step: float) -> float:
    """Snap a continuous value to the nearest grid point defined by min + k*step."""
    k = round((value - min_val) / step)
    return round(min_val + k * step, 10)


def run_calibration_bayesian_opt(
    sv_df: pd.DataFrame,
    sample_ids: set[str],
    insert_size_lookup: dict[str, int],
    cfg: dict,
    *,
    n_trials: int = 30,
    batch_size: int = 1,
    d_min: int,
    d_max: int,
    d_step: int,
    r_min: float,
    r_max: float,
    r_step: float,
    q_min: float,
    q_max: float,
    q_step: float,
    p_min: int,
    p_max: int,
    p_step: int,
):
    """
    Run calibration using Bayesian Optimization to find the parameter
    combination that maximizes PR-AUC.

    If calibration/results/results.csv already exists, those reults are fed into
    the BO model as prior observations before any new runs are launched.
    """

    output_dir = cfg["paths"]["output_dir"]
    results_file = os.path.join(output_dir, "results.csv")
    has_prev_results = os.path.exists(results_file)

    # define a BO generation strategy that starts with the center node, then transitions to Sobol for 5 trials, then transitions to BoTorch for the rest of the trials
    generator_spec = GeneratorSpec(generator_enum=Generators.BOTORCH_MODULAR)
    botorch_node = GenerationNode(
        name="BoTorch",
        generator_specs=[generator_spec],
    )
    sobol_node = GenerationNode(
        name="Sobol",
        generator_specs=[
            GeneratorSpec(generator_enum=Generators.SOBOL),
        ],
        transition_criteria=[
            # transition to BoTorch node once there are 15 trials on the experiment.
            MinTrials(
                threshold=15,
                transition_to=botorch_node.name,
                use_all_trials_in_exp=True,
            )
        ],
    )
    center_node = CenterGenerationNode(next_node_name=sobol_node.name)

    if has_prev_results:
        gs = GenerationStrategy(
            name="BoTorch",
            nodes=[botorch_node],
        )
    else:
        gs = GenerationStrategy(
            name="Center+Sobol+BoTorch",
            nodes=[center_node, sobol_node, botorch_node],
        )

    # define the search space
    ax_client = AxClient(generation_strategy=gs, verbose_logging=False)
    ax_client.create_experiment(
        name="calibration_experiment",
        parameters=[
            {
                "name": "d",
                "type": "range",
                "bounds": [d_min, d_max],
                "value_type": "int",
            },
            {
                "name": "r",
                "type": "range",
                "bounds": [r_min, r_max],
                "value_type": "float",
            },
            {
                "name": "q",
                "type": "range",
                "bounds": [q_min, q_max],
                "value_type": "float",
            },
            {
                "name": "p",
                "type": "range",
                "bounds": [p_min, p_max],
                "value_type": "int",
            },
        ],
        objectives={"pr_auc": ObjectiveProperties(minimize=False)},
    )

    # load pre-existing results
    # if BO has been searching in one space for a while, then we should subsample
    # the existing results so the model doesn't get stuck in one space
    if has_prev_results:
        df = pd.read_csv(results_file)
        for i, row in df.iterrows():
            params = {
                "d": int(row["d"]),
                "r": round(row["r"], 2),
                "q": round(snap_to_grid(row["q"], q_min, q_step), 2),
                "p": int(row["p"]),
            }
            score = calc_pr_auc(row["TP"], row["FP"], row["FN"])
            _, trial_index = ax_client.attach_trial(params)
            ax_client.complete_trial(
                trial_index=trial_index,
                raw_data={"pr_auc": (score, None)},
            )

    # BO loop
    for trial in range(n_trials):
        print(f"Running Bayesian Optimization Trial {trial + 1}/{n_trials}")

        # ask for the next candidate(s)
        if batch_size > 1:
            parameterizations, trial_index = ax_client.get_next_trials(
                max_trials=batch_size
            )
        else:
            parameterization, trial_index = ax_client.get_next_trial()
            parameterizations = {trial_index: parameterization}

        # evaluate each candidate
        for t_index, params in parameterizations.items():
            d = int(params["d"])
            r = round(params["r"], 2)
            q = round(snap_to_grid(params["q"], q_min, q_step), 2)
            p = int(params["p"])

            # this function is parallelized for each SV but runs one calibration test for the given parameters
            results = run_calibration_test(
                sv_df,
                cfg,
                d=d,
                r=r,
                q=q,
                pen=p,
                sample_ids=sample_ids,
                insert_size_lookup=insert_size_lookup,
            )

        for t_index, params in parameterizations.items():
            score = calc_pr_auc(results["TP"], results["FP"], results["FN"])
            print(f"Trial {t_index} F1 score = {score:.4f}")
            ax_client.complete_trial(
                trial_index=t_index,
                raw_data={"pr_auc": (score, None)},
            )

    # after BO loop, get the best parameters and run one final calibration test with those parameters to save the outputs
    best_params, metrics = ax_client.get_best_parameters()
    best_d = int(best_params["d"])
    best_r = round(best_params["r"], 2)
    best_q = round(snap_to_grid(best_params["q"], q_min, q_step), 2)
    best_p = int(best_params["p"])

    print(
        f"Best parameters found by Bayesian Optimization: d={best_d}, r={best_r}, q={best_q}, p={best_p}"
    )
    print(f"Best F1 score: {metrics[0]['pr_auc']:.4f}")

    best_result = pd.DataFrame(
        [
            {
                "d": best_d,
                "r": best_r,
                "q": best_q,
                "p": best_p,
                "pr_auc": metrics[0]["pr_auc"],
            }
        ]
    )
    best_result.to_csv(os.path.join(output_dir, "best_params.csv"), index=False)

    return best_params


def run_calibration_grid_search(
    sv_df: pd.DataFrame,
    sample_ids: set[str],
    insert_size_lookup: dict[str, int],
    cfg: dict,
    *,
    d_min: int,
    d_max: int,
    d_step: int,
    r_min: float,
    r_max: float,
    r_step: float,
    q_min: float,
    q_max: float,
    q_step: float,
    p_min: int,
    p_max: int,
    p_step: int,
):
    """Run calibration tests over a grid of distance and reciprocal overlap thresholds."""
    # run calibration tests over grid of d, r, and q values
    for q in np.arange(q_min, q_max + 0.01, q_step):
        for d in range(d_min, d_max + 1, d_step):
            for r in np.arange(r_min, r_max + 0.01, r_step):
                for pen in range(p_min, p_max + 1, p_step):
                    run_calibration_test(
                        sv_df,
                        cfg,
                        d=d,
                        r=round(r, 2),
                        q=round(q, 2),
                        pen=pen,
                        sample_ids=sample_ids,
                        insert_size_lookup=insert_size_lookup,
                    )


def download_stix_data_inner(
    row: pd.Series,
    q: float,
    output_dir: str,
    stix_bin: str,
    index_path: str,
    database_path: str,
    num_shards: int,
):
    # check if the data for this region already exists
    query_region = get_query_region(
        giggle_format(row.chr, row.start),
        giggle_format(row.chr, row.stop),
        q,
    )
    filename = f"{query_region.file_name}.txt"
    file = os.path.join(output_dir, filename)
    if os.path.isfile(file):
        return

    stix_file = query_stix(
        query_region,
        output_dir,
        stix_bin,
        index_path,
        database_path,
        num_shards,
        True,
    )
    print(f"Downloaded STIX data {stix_file}", flush=True)


def download_stix_data(
    sv_subset: pd.DataFrame,
    q: float,
    stix_output_dir: str,
    stix_bin: str,
    index_path: str,
    database_path: str,
    num_shards: int,
):
    """Download stix data for all regions in the subset before running calibration tests."""
    output_dir = f"{stix_output_dir}_{q}"

    # set up the correct file paths in case we need to query more SVs
    partial_outputs_dir = os.path.join(output_dir, "partial_outputs")
    os.makedirs(partial_outputs_dir, exist_ok=True)

    with multiprocessing.Manager():
        p = multiprocessing.Pool(SLURM_CPUS)
        args = []
        for _, row in sv_subset.iterrows():
            args.append(
                (
                    row,
                    q,
                    output_dir,
                    stix_bin,
                    index_path,
                    database_path,
                    num_shards,
                )
            )

        p.starmap(download_stix_data_inner, args)
        p.close()
        p.join()

    os.rmdir(partial_outputs_dir)


def calibrate(cfg: dict):
    input_dir = cfg["paths"]["input_dir"]

    sv_subset = pd.read_csv(
        os.path.join(input_dir, cfg["calibrate"]["truth_set"])
    )
    sv_subset["id"] = sv_subset.apply(
        lambda row: f"{giggle_format(str(row['chr']), row['start'])}_{giggle_format(str(row['chr']), row['stop'])}",
        axis=1,
    )
    sv_df = pd.read_csv(
        os.path.join(input_dir, cfg["input_files"]["sv_lookup_file"])
    )
    sv_df.rename(columns={"id": "og_sv_id"}, inplace=True)
    merged_df = sv_subset.merge(sv_df, on=["chr", "start", "stop"], how="left")
    sample_ids = set()
    with open(
        os.path.join(input_dir, cfg["input_files"]["sample_id_file"]), "r"
    ) as f:
        for line in f:
            sample_ids.add(line.strip())

    insert_size_lookup = get_insert_size_lookup(
        input_dir,
        cfg["input_files"]["insert_size_file"],
        cfg["model"]["default_insert_size"],
        sample_ids,
    )

    for q in np.arange(
        cfg["calibrate"]["q_min"],
        cfg["calibrate"]["q_max"] + 0.01,
        cfg["calibrate"]["q_step"],
    ):
        # check that all stix data has been queried and saved for regions in sv_subset before running calibration tests
        download_stix_data(
            sv_subset,
            round(q, 2),
            cfg["paths"]["stix_output_dir"],
            cfg["stix"]["bin"],
            cfg["stix"]["index"],
            cfg["stix"]["database"],
            cfg["stix"]["num_shards"],
        )

    # make sure output directories exist
    for dir in ["output_dir", "stix_output_dir", "intermediate_output_dir"]:
        os.makedirs(cfg["paths"][dir], exist_ok=True)

    search_func = cfg["calibrate"]["search_func"]
    if search_func == "bo":
        calibration_function = run_calibration_bayesian_opt
    elif search_func == "grid":
        calibration_function = run_calibration_grid_search
    else:
        raise ValueError(
            f"Invalid search function: {search_func}. Supported options are 'bo' and 'grid'."
        )

    calibrate_args = cfg["calibrate"].copy()
    del calibrate_args["truth_set"]
    del calibrate_args["search_func"]

    calibration_function(
        merged_df,
        sample_ids,
        insert_size_lookup,
        cfg,
        **calibrate_args,
    )
