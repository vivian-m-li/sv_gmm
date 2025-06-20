import os
import ast
import sys
import csv
import multiprocessing
import numpy as np
import pandas as pd
import math
from generate_data import generate_synthetic_sv_data
from gmm_types import GMM_MODELS, Evidence
from typing import Optional, List, Tuple
from timeout import break_after


def reciprocal_overlap(sv1, sv2):
    start1, end1 = sv1
    start2, end2 = sv2
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_start >= overlap_end:
        return 0.0
    overlap_length = overlap_end - overlap_start
    sv1_length = end1 - start1
    sv2_length = end2 - start2
    return min(overlap_length / sv1_length, overlap_length / sv2_length)


def write_reciprocal_overlap():
    files = os.listdir("synthetic_data")
    for file in files:
        df = pd.read_csv(f"synthetic_data/{file}")
        df = df[df["expected_num_modes"] == 2]
        df["reciprocal_overlap"] = 0.0
        for _, row in df.iterrows():
            Ls = np.array(ast.literal_eval(row["expected_Ls"]))
            lengths = np.array(ast.literal_eval(row["expected_lengths"]))
            Rs = Ls + lengths
            df.at[_, "reciprocal_overlap"] = reciprocal_overlap(
                [Ls[0], Rs[0]], [Ls[1], Rs[1]]
            )
        df.to_csv(f"synthetic_data/{file}")


def run_gmm(case, d, svs, weights, n_samples, results):
    for gmm_model in GMM_MODELS:
        gmm, evidence_by_mode = generate_synthetic_sv_data(
            1,
            svs,
            n_samples=n_samples,
            p=weights,
            gmm_model=gmm_model,
        )
        results.append(
            [case, d, gmm_model, svs, n_samples, gmm, evidence_by_mode]
        )


def get_len_L(evidence_by_mode: List[List[Evidence]]):
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
            lens.append(mean_length)
        lengths.append(int(np.mean(lens)))
        Ls.append(int(np.mean(starts)))
    return lengths, Ls


def write_csv(
    all_results,
    *,
    write_new_file: bool = False,
    fixed_n_samples: Optional[int] = None,
):
    file = f"synthetic_data/results{'' if fixed_n_samples is None else 'n=' + str(fixed_n_samples)}.csv"
    with open(
        file,
        mode="w" if write_new_file else "a",
        newline="",
    ) as out:
        if write_new_file:
            fieldnames = [
                "case",
                "d",
                "gmm_model",
                "expected_num_modes",
                "expected_lengths",
                "expected_Ls",
                "num_modes",
                "lengths",
                "Ls",
                "num_samples",
            ]
            csv_writer = csv.DictWriter(out, fieldnames=fieldnames)
            csv_writer.writeheader()
        else:
            csv_writer = csv.DictWriter(out)

        for (
            case,
            d,
            gmm_model,
            svs,
            n_samples,
            gmm,
            evidence_by_mode,
        ) in all_results:
            if gmm is None:
                continue

            lengths, Ls = get_len_L(evidence_by_mode)
            csv_writer.writerow(
                {
                    "case": case,
                    "d": d,
                    "gmm_model": gmm_model,
                    "expected_num_modes": 3 if case in ["D", "E"] else 2,
                    "expected_lengths": [sv[1] - sv[0] for sv in svs],
                    "expected_Ls": [sv[0] for sv in svs],
                    "num_modes": gmm.num_modes,
                    "lengths": lengths,
                    "Ls": Ls,
                    "num_samples": n_samples,
                }
            )


def is_valid_svs(svs):
    valid_sv = True
    for i, sv1 in enumerate(svs):
        sv1_size = sv1[1] - sv1[0]
        for sv2 in svs[(i + 1) :]:
            sv2_size = sv2[1] - sv2[0]
            valid_sv = valid_sv and (
                np.abs(sv2_size - sv1_size) >= 100
                or np.abs(sv2[0] - sv1[0]) > 50  # noqa: 503
                or np.abs(sv2[1] - sv1[1]) > 50  # noqa: 503
            )
    return valid_sv


def get_coordinates(sv1, sv2, d):
    x1, y1 = sv1
    x2, y2 = sv2
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    x3 = int(x1 + d * (x2 - x1) / dist)
    y3 = int(y1 + d * (y2 - y1) / dist)
    return (x3, y3)


"""Generates synthetic data to test the model with. See Figure 2 for the 5 scenarios."""


def generate_data(case: str) -> List[Tuple[int, int]]:
    SVLEN = 2553  # median SV length
    SV1_L = 100000
    SV1_R = SV1_L + SVLEN
    SV1 = (SV1_L, SV1_R)

    # for case D and E
    SV2_L = 101000
    SV2_LEN = 2003
    SV2 = (SV2_L, SV2_L + SV2_LEN)

    SV3_L = 100782
    SV3_LEN = 3383
    SV3 = (SV3_L, SV3_L + SV3_LEN)

    MIDPOINT = (100500, 100500 + 2278)

    data = []
    match case:
        case "A":
            for d in range(0, 505, 5):
                # sv1 is larger, sv2 is smaller within sv1
                data.append([case, d, [SV1, (int(SV1_L + d), int(SV1_R - d))]])
        case "B":
            for d in range(0, 505, 5):
                # sv 1 and 2 are the same length. sv1 starts first and sv2 starts at the start of sv1 + d
                sv2_start = SV1_L + d
                data.append([case, d, [SV1, (sv2_start, sv2_start + SVLEN)]])
        case "C":
            for d in range(0, 505, 5):
                # sv 1 and 2 are the same length. sv1 starts first and sv2 starts at the end of sv1 + d
                sv2_start = SV1_R + d
                data.append([case, d, [SV1, (sv2_start, sv2_start + SVLEN)]])
        case "D":
            for d in range(0, 1110, 10):
                sv3 = get_coordinates(SV1, SV3, d)
                data.append([case, d, [SV1, SV2, sv3]])
        case "E":
            for d in range(0, 1110, 10):
                sv3 = get_coordinates(MIDPOINT, SV3, d)
                data.append([case, d, [SV1, SV2, sv3]])
        case _:
            raise Exception("Invalid case")

    return data


@break_after(hours=17, minutes=55)
def d_accuracy_test(
    n_samples: int, test_case: Optional[str] = None, vary_weights: bool = False
):
    # generate synthetic data
    if test_case is None:
        data = []
        for case in ["A", "B", "C", "D", "E"]:
            data.extend(generate_data(case))
    else:
        data = generate_data(test_case)

    with multiprocessing.Manager() as manager:
        cpu_count = multiprocessing.cpu_count()
        p = multiprocessing.Pool(cpu_count)
        results = manager.list()
        args = []
        for case, d, svs in data:
            # run each case 100 times and average at the end
            # TODO: do a range of weights if vary_weights is True. save data in a different output file
            weights = [1.0 / len(svs) for _ in range(len(svs))]
            for _ in range(100):
                args.append((case, d, svs, weights, n_samples, results))

        p.starmap(run_gmm, args)
        p.close()
        p.join()

        # results: [(case, d, gmm_model, svs, n_samples, gmm, evidence_by_mode), ...]
        write_csv(
            results, write_new_file=test_case is None, fixed_n_samples=n_samples
        )


if __name__ == "__main__":
    # N_SAMPLES = [72, 118]  # the median and mean number of samples
    n_samples = int(sys.argv[1])
    d_accuracy_test(n_samples=n_samples)
