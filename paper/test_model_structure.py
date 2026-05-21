import copy
import os
import re
import time

import pandas as pd

from scripts.split_all import split_all
from src.utils.config_loader import load_config
from src.utils.helper import get_sample_ids
from src.utils.model_helper import process_input_files, get_insert_size_lookup
from src.utils.write_sv_output import (
    concat_multi_processed_sv_files,
    write_post_processed_files,
)


def write_test_result(name: str, cfg: dict, runtime: float):
    results_file = os.path.join(cfg["paths"]["output_dir"], "results.csv")
    if not os.path.exists(results_file):
        df = pd.DataFrame(
            columns=[
                "test_name",
                "TP",
                "FN",
                "TN",
                "FP",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "runtime",
            ]
        )
    else:
        df = pd.read_csv(results_file)

    truth_set = pd.read_csv(
        os.path.join(cfg["paths"]["input_dir"], cfg["calibrate"]["truth_set"])
    )
    # compare results to truth set
    n_modes = pd.read_csv(
        os.path.join(cfg["paths"]["output_dir"], name, "svs_n_modes.csv")
    )
    n_modes = n_modes.rename({"num_modes": "n_svs_predicted"}, axis=1)
    for i, row in n_modes.iterrows():
        sv_id = row["sv_id"]
        match = re.match(r"[\w+]_(\d+):(\d+)_[\d+]:(\d+)", sv_id)
        chr, start, stop = match.groups()
        n_modes.at[i, "chr"] = int(chr)
        n_modes.at[i, "start"] = int(start)
        n_modes.at[i, "stop"] = int(stop)
    merged = pd.merge(
        truth_set, n_modes, on=["chr", "start", "stop"], how="left"
    )

    TP = merged[
        (merged["n_svs_actual"] == 2) & (merged["n_svs_predicted"] >= 2)
    ].shape[0]
    FN = merged[
        (merged["n_svs_actual"] == 2) & (merged["n_svs_predicted"] == 1)
    ].shape[0]
    TN = merged[
        (merged["n_svs_actual"] == 1) & (merged["n_svs_predicted"] == 1)
    ].shape[0]
    FP = merged[
        (merged["n_svs_actual"] == 1) & (merged["n_svs_predicted"] > 1)
    ].shape[0]
    accuracy = (TP + TN) / merged.shape[0]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    df.loc[len(df)] = [
        name,
        TP,
        FN,
        TN,
        FP,
        accuracy,
        precision,
        recall,
        f1,
        runtime,
    ]
    df.to_csv(results_file, index=False)


def single_test(name: str, cfg: dict, updated_cfg_vals: dict):
    print("Running test:", name)
    start = time.time()

    cfg_copy = copy.deepcopy(cfg)
    input_dir = cfg_copy["paths"]["input_dir"]

    output_dir = os.path.join(cfg_copy["paths"]["output_dir"], name)
    cfg_copy["paths"]["output_dir"] = output_dir

    intermediate_output_dir = os.path.join(
        cfg_copy["paths"]["intermediate_output_dir"], name
    )
    cfg_copy["paths"]["intermediate_output_dir"] = intermediate_output_dir

    os.makedirs(intermediate_output_dir, exist_ok=True)

    for k, v in updated_cfg_vals.items():
        cfg_copy["model"][k] = v

    sample_ids = get_sample_ids(
        os.path.join(
            input_dir,
            cfg_copy["input_files"]["sample_id_file"],
        )
    )
    insert_size_lookup = get_insert_size_lookup(
        input_dir,
        cfg_copy["input_files"]["insert_size_file"],
        cfg_copy["model"]["default_insert_size"],
        sample_ids,
    )

    split_all(cfg_copy, sample_ids, insert_size_lookup)

    concat_multi_processed_sv_files(
        output_dir, intermediate_output_dir, "all_split_trials.csv"
    )
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

    # convert runtime to hours
    end = time.time()
    runtime = (end - start) / 3600

    write_test_result(name, cfg, runtime=runtime)

    print("Finished running test:", name, "Runtime (hours):", runtime)


def test_model_structure(cfg: dict):
    process_input_files(
        cfg["paths"]["input_dir"],
        cfg["input_files"]["sv_lookup_file"],
        cfg["input_files"]["sample_id_file"],
        cfg["input_files"]["insert_size_file"],
        None,
    )

    # Baseline: run with the default config
    single_test("baseline", cfg, {})

    # Test 1: kmeans++ vs random initialization with dirichlet process
    single_test("init_kmeans++", cfg, {"init": "kmeans++"})
    single_test("init_random", cfg, {"init": "random"})

    # Test 2: cluster repulsion vs baseline
    single_test("repulsion_true", cfg, {"repulsion": "True"})

    # Test 3: model comparison function (AIC vs BIC vs ICL)
    # aic test is the baseline
    single_test(
        "model_comparison_func_bic", cfg, {"model_comparison_func": "bic"}
    )
    single_test(
        "model_comparison_func_icl", cfg, {"model_comparison_func": "icl"}
    )

    # Test the remaining permutations

    # kmeans++ (bic/icl)
    single_test(
        "init_kmeans++_model_comparison_func_bic",
        cfg,
        {"init": "kmeans++", "model_comparison_func": "bic"},
    )
    single_test(
        "init_kmeans++_model_comparison_func_icl",
        cfg,
        {"init": "kmeans++", "model_comparison_func": "icl"},
    )

    # random (repulsion true/false, aic/bic/icl)
    single_test(
        "init_random_repulsion_true_tau_scaled_2",
        cfg,
        {"init": "random", "repulsion": "True"},
    )
    single_test(
        "init_random_repulsion_true_model_comparison_func_bic",
        cfg,
        {"init": "random", "repulsion": "True", "model_comparison_func": "bic"},
    )
    single_test(
        "init_random_repulsion_true_model_comparison_func_icl",
        cfg,
        {"init": "random", "repulsion": "True", "model_comparison_func": "icl"},
    )
    single_test(
        "init_random_model_comparison_func_bic",
        cfg,
        {"init": "random", "model_comparison_func": "bic"},
    )
    single_test(
        "init_random_model_comparison_func_icl",
        cfg,
        {"init": "random", "model_comparison_func": "icl"},
    )

    # dp_kmeans++ (repulsion + bic/icl)
    single_test(
        "repulsion_true_model_comparison_func_bic",
        cfg,
        {"repulsion": "True", "model_comparison_func": "bic"},
    )
    single_test(
        "repulsion_true_model_comparison_func_icl",
        cfg,
        {"repulsion": "True", "model_comparison_func": "icl"},
    )


if __name__ == "__main__":
    cfg = load_config()

    # hard-code paths in case the config file is off
    cfg["paths"]["input_dir"] = "data/synthetic_calibration"
    cfg["paths"]["output_dir"] = "output/synthetic_calibration/results"
    cfg["paths"]["stix_output_dir"] = "output/synthetic_calibration/stix_output"
    cfg["paths"][
        "intermediate_output_dir"
    ] = "output/synthetic_calibration/calibration_outputs"
    cfg["input_files"]["sv_lookup_file"] = "deletions.csv"
    cfg["calibrate"]["truth_set"] = "sv_subset.csv"

    test_model_structure(cfg)
