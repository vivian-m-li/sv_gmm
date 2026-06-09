import argparse
import ast
import csv
import math
import multiprocessing
import os
from pprint import pprint

import numpy as np
import pandas as pd

from src.synthetic.generate_data import (
    generate_sv_coordinates,
    generate_and_split_sample_reads,
)
from src.utils.config_loader import load_config
from src.utils.model_helper import reciprocal_overlap
from src.utils.timeout import break_after
from src.utils.types import Evidence

SLURM_CPUS = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))


def write_reciprocal_overlap(dir: str):
    """Calculates and writes the reciprocal overlap for synthetic data files that have already been written."""
    files = os.listdir(dir)
    for file in files:
        df = pd.read_csv(os.path.join(dir, file))
        df = df[df["expected_num_modes"] == 2]
        df["reciprocal_overlap"] = 0.0
        for _, row in df.iterrows():
            Ls = np.array(ast.literal_eval(row["expected_Ls"]))
            lengths = np.array(ast.literal_eval(row["expected_lengths"]))
            Rs = Ls + lengths
            df.at[_, "reciprocal_overlap"] = reciprocal_overlap(
                [Ls[0], Rs[0]], [Ls[1], Rs[1]]
            )
        df.to_csv(os.path.join(dir, file), index=False)


def run_split(case, r, svs, weights, n_samples, cfg, model_params, results):
    """Generates synthetic data and runs the GMM on it. Appends the results to the multiprocessing-managed list to be written to a CSV later."""
    gmm_result, evidence_by_mode = generate_and_split_sample_reads(
        1,
        svs,
        insert_size_file=os.path.join(
            cfg["paths"]["input_dir"], cfg["input_files"]["insert_size_file"]
        ),
        model_params=model_params,
        n_samples=n_samples,
        p=weights,
    )
    results.append(
        [
            case,
            r,
            "split",
            svs,
            n_samples,
            weights,
            gmm_result,
            evidence_by_mode,
        ]
    )


def get_len_L(evidence_by_mode: list[list[Evidence]]):
    """Gets the average length and L coordinate for each mode from the evidence."""
    lengths = []
    Ls = []
    for mode in evidence_by_mode:
        lens = []
        starts = []
        for evidence in mode:
            mean_l = np.mean(
                [paired_end[0] for paired_end in evidence.paired_ends]
            )
            mean_length = np.mean(
                [
                    paired_end[1] - paired_end[0] - evidence.mean_insert_size
                    for paired_end in evidence.paired_ends
                ]
            )
            starts.append(mean_l)
            if not np.isnan(mean_length):
                lens.append(mean_length)
        lengths.append(int(np.mean(lens)))
        Ls.append(int(np.mean(starts)))
    return lengths, Ls


def write_csv(
    all_results,
    output_dir: str,
    *,
    write_new_file: bool = False,
    fixed_n_samples: int | None = None,
    fixed_svlen: int | None = None,
):
    """Writes the results of the synthetic data tests to a CSV file."""
    file = os.path.join(
        output_dir,
        f"results{'' if fixed_n_samples is None else 'n=' + str(fixed_n_samples)}.csv",
    )
    if not os.path.exists(file):
        write_new_file = True
    with open(
        file,
        mode="w" if write_new_file else "a",
        newline="",
    ) as out:
        fieldnames = [
            "case",
            "r",
            "r2",
            "gmm_model",
            "expected_num_modes",
            "expected_lengths",
            "expected_Ls",
            "num_modes",
            "lengths",
            "Ls",
            "num_samples",
            "svlen",
            "weights",
        ]
        csv_writer = csv.DictWriter(out, fieldnames=fieldnames)
        if write_new_file:
            csv_writer.writeheader()

        for (
            case,
            rs,
            gmm_model,
            svs,
            n_samples,
            weights,
            gmm_result,
            evidence_by_mode,
        ) in all_results:
            if gmm_result is None:
                continue

            lengths, Ls = get_len_L(evidence_by_mode)
            if type(rs) is tuple:
                r, r2 = rs
            else:
                r = rs
                r2 = None
            csv_writer.writerow(
                {
                    "case": case,
                    "r": r,
                    "r2": r2,
                    "gmm_model": gmm_model,
                    "expected_num_modes": len(svs),
                    "expected_lengths": [sv[1] - sv[0] for sv in svs],
                    "expected_Ls": [sv[0] for sv in svs],
                    "num_modes": gmm_result.num_modes,
                    "lengths": lengths,
                    "Ls": Ls,
                    "num_samples": n_samples,
                    "svlen": fixed_svlen,
                    "weights": weights,
                }
            )

    df = pd.read_csv(file)
    df = df.sort_values(by=["case", "gmm_model", "r", "r2"])
    df.to_csv(file, index=False)


def get_coordinates(sv1, sv2, d):
    """Get coordinates of a third SV that is d distance away the sv1 point towards the direction of sv2. Used in the distance/accuracy tests."""
    x1, y1 = sv1
    x2, y2 = sv2
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    x3 = int(x1 + d * (x2 - x1) / dist)
    y3 = int(y1 + d * (y2 - y1) / dist)
    return (x3, y3)


@break_after(hours=15, minutes=55)
def split_synthetic_svs(
    cfg: dict,
    *,
    n_samples: int,
    svlen: int,
    model_params: dict,
    test_case: str | None = None,
    vary_weights: bool = False,
):
    """Parallelized synthetic data tests with varying r and (optional) cluster weights. Each set of parameters is repeated 50 times. Run with the bash script run_synthetic_data_tests.sh to test different sample sizes and sv lengths."""
    # generate synthetic data
    if test_case is None:
        data = []
        for case in ["A", "B", "C", "D"]:
            data.extend(generate_sv_coordinates(case, svlen))
    else:
        data = generate_sv_coordinates(test_case, svlen)

    with multiprocessing.Manager() as manager:
        p = multiprocessing.Pool(SLURM_CPUS)
        results = manager.list()
        args = []
        for case, r, svs in data:
            if vary_weights:
                weights = []
                if len(svs) == 1:
                    weights.append([1.0])
                elif len(svs) == 2:
                    for w1 in np.arange(0.1, 1.0, 0.1):
                        w2 = 1.0 - w1
                        weights.append([w1, w2])
                else:
                    for w1 in np.arange(0.1, 0.9, 0.1):
                        for w2 in np.arange(0.1, 1.0 - w1, 0.1):
                            w3 = 1.0 - w1 - w2
                            weights.append([w1, w2, w3])
                weights = [round(w, 2) for ws in weights for w in ws]
            else:
                weights = [[1.0 / len(svs) for _ in range(len(svs))]]
            for weight in weights:
                # run each case 10 times and average at the end
                for _ in range(10):
                    args.append(
                        (
                            case,
                            r,
                            svs,
                            weight,
                            n_samples,
                            cfg,
                            model_params,
                            results,
                        )
                    )

        p.starmap(run_split, args)
        p.close()
        p.join()

        output_dir = cfg["paths"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        # results: [(case, r, gmm_model, svs, n_samples, weights, gmm, evidence_by_mode), ...]
        write_csv(
            results,
            output_dir=output_dir,
            write_new_file=False,
            fixed_n_samples=n_samples,
            fixed_svlen=svlen,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run synthetic data tests using SPLIT."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to the TOML configuration file (default: config.toml)",
    )
    parser.add_argument(
        "-n",
        type=int,
        help="Number of samples to generate for each synthetic data test case",
    )
    parser.add_argument(
        "--svlen",
        type=int,
        help="Length of the synthetic SVs to generate for each test case",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Test case to run (A, B, C, D). If not specified, all cases will be run.",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)
    print(
        f"Running synthetic data tests with n={args.n}, svlen={args.svlen}, case={args.case}"
    )
    print("Config args:")
    pprint(cfg)

    split_synthetic_svs(
        cfg,
        n_samples=args.n,
        svlen=args.svlen,
        test_case=args.case,
        model_params=cfg["model"],
    )
