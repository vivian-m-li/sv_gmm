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
    """
    Calculates the reciprocal overlap between two structural variants.
    r = min(% overlap sv 1, % overlap sv 2)
    """
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


def run_gmm(case, r, svs, weights, n_samples, results):
    for gmm_model in GMM_MODELS:
        gmm, evidence_by_mode = generate_synthetic_sv_data(
            1,
            svs,
            n_samples=n_samples,
            p=weights,
            gmm_model=gmm_model,
        )
        results.append(
            [case, r, gmm_model, svs, n_samples, weights, gmm, evidence_by_mode]
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
                "r",
                "gmm_model",
                "expected_num_modes",
                "expected_lengths",
                "expected_Ls",
                "num_modes",
                "lengths",
                "Ls",
                "num_samples",
                "weights",
            ]
            csv_writer = csv.DictWriter(out, fieldnames=fieldnames)
            csv_writer.writeheader()
        else:
            csv_writer = csv.DictWriter(out)

        for (
            case,
            rs,
            gmm_model,
            svs,
            n_samples,
            weights,
            gmm,
            evidence_by_mode,
        ) in all_results:
            if gmm is None:
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
                    "num_modes": gmm.num_modes,
                    "lengths": lengths,
                    "Ls": Ls,
                    "num_samples": n_samples,
                    "weights": weights,
                }
            )


def get_coordinates(sv1, sv2, d):
    x1, y1 = sv1
    x2, y2 = sv2
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    x3 = int(x1 + d * (x2 - x1) / dist)
    y3 = int(y1 + d * (y2 - y1) / dist)
    return (x3, y3)


# DEPRECATED
def generate_data(case: str) -> List[Tuple[int, int]]:
    """Generates synthetic data to test the model with. See Figure 2 for the 5 scenarios."""
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


# DEPRECATED: Use `r_accuracy_test` instead
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


def generate_data_r(case: str, svlen: int):
    """Generates data varying the reciprocal overlap."""
    # SVLEN = 802  # median SV length for high coverage data
    SV1_L = 100000
    SV1_R = SV1_L + svlen
    SV1 = (SV1_L, SV1_R)

    data = []
    match case:
        case "A":
            data.append([case, None, [[SV1_L, SV1_L + svlen]]])
        case "B":
            # two nested SVs
            for r in np.arange(0.05, 1.05, 0.05):
                midpoint = SV1_L + (0.5 * svlen)
                sv2_len = int(r * svlen)
                data.append(
                    [
                        case,
                        r,
                        [
                            SV1,
                            (
                                midpoint - int(0.5 * sv2_len),
                                midpoint + int(0.5 * sv2_len),
                            ),
                        ],
                    ]
                )
        case "C":
            # two overlapping SVs
            for r in np.arange(0, 1.05, 0.05):
                overlap = int(r * svlen)
                sv2_start = SV1_R - overlap
                data.append([case, r, [SV1, (sv2_start, sv2_start + svlen)]])
        case "D":
            # three overlapping SVs
            for r1 in np.arange(0, 1.05, 0.05):  # overlap between sv1 and sv2
                overlap12 = int(r1 * svlen)
                sv2_start = SV1_R - overlap12
                sv2_end = sv2_start + svlen
                for r2 in np.arange(
                    0, 1.05, 0.05
                ):  # overlap between sv2 and sv3
                    overlap23 = int(r2 * svlen)
                    sv3_start = sv2_end - overlap23
                    sv3_end = sv3_start + svlen
                    data.append(
                        [
                            case,
                            (round(r1, 2), round(r2, 2)),
                            [SV1, (sv2_start, sv2_end), (sv3_start, sv3_end)],
                        ]
                    )
        case _:
            raise Exception("Invalid case")
    return data


@break_after(hours=23, minutes=55)
def r_accuracy_test(
    n_samples: int,
    svlen: int,
    test_case: Optional[str] = None,
    vary_weights: bool = False,
):
    # generate synthetic data
    if test_case is None:
        data = []
        for case in ["A", "B", "C", "D"]:
            data.extend(generate_data_r(case, svlen))
    else:
        data = generate_data_r(test_case, svlen)

    with multiprocessing.Manager() as manager:
        cpu_count = multiprocessing.cpu_count()
        p = multiprocessing.Pool(cpu_count)
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
                    args.append((case, r, svs, weights, n_samples, results))

        p.starmap(run_gmm, args)
        p.close()
        p.join()

        # results: [(case, r, gmm_model, svs, n_samples, weights, gmm, evidence_by_mode), ...]
        write_csv(
            results, write_new_file=test_case is None, fixed_n_samples=n_samples
        )


if __name__ == "__main__":
    n_samples = int(sys.argv[1])
    svlen = int(sys.argv[2]) if len(sys.argv) > 2 else 802
    r_accuracy_test(n_samples=n_samples, svlen=svlen)
