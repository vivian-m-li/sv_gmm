import csv
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn import metrics
from generate_data import generate_synthetic_sv_data
from gmm_types import GMM_MODELS
from typing import Optional, List, Tuple


def confusion_mat():
    for model in GMM_MODELS:
        df = pd.read_csv(f"synthetic_data/results_{model}.csv", low_memory=False)
        actual = df["expected_num_modes"]
        predicted = df["num_modes"]
        confusion_matrix = metrics.confusion_matrix(actual, predicted)
        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix,
            display_labels=["1 mode", "2 modes", "3 modes"],
        )
        cm_display.plot()
        plt.title(model)
        plt.show()


def run_gmm(case, d, svs, weights, n_samples, results):
    for gmm_model in GMM_MODELS:
        gmm, evidence_by_mode = generate_synthetic_sv_data(
            1, svs, n_samples=n_samples, p=weights, gmm_model=gmm_model
        )
        results.append([case, d, gmm_model, svs, n_samples, gmm, evidence_by_mode])


def get_len_L(evidence_by_mode):
    lengths = []
    Ls = []
    for mode in evidence_by_mode:
        lens = []
        starts = []
        for evidence in mode:
            mean_l = np.mean([paired_end[0] for paired_end in evidence.paired_ends])
            mean_r = np.mean([paired_end[1] for paired_end in evidence.paired_ends])
            lens.append(mean_r - mean_l - 450)  # 450 is the length of the read
            starts.append(mean_l)
        lengths.append(int(np.mean(lens)))
        Ls.append(int(np.mean(starts)))
    return lengths, Ls


def write_csv(
    all_results, *, write_new_file: bool = False, fixed_n_samples: Optional[int] = None
):
    file = f"synthetic_data/results{'' if fixed_n_samples is None else fixed_n_samples}.csv"
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

        for case, d, gmm_model, svs, n_samples, gmm, evidence_by_mode in all_results:
            # there is an error where all generated points are getting filtered out in process_data
            # skip those rows in that case
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
        for sv2 in svs[i + 1 :]:
            sv2_size = sv2[1] - sv2[0]
            valid_sv = valid_sv and (
                np.abs(sv2_size - sv1_size) >= 100
                or np.abs(sv2[0] - sv1[0]) > 50
                or np.abs(sv2[1] - sv1[1]) > 50
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
    SVLEN = 2553  #  median SV length
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
            for d in range(0, 510, 2):
                data.append([case, d, [SV1, (int(100000 + d / 2), int(SV1_R - d / 2))]])
        case "B":
            for d in range(0, 510, 2):
                sv2_start = SV1_L - d
                data.append([case, d, [(sv2_start, sv2_start + SVLEN), SV1]])
        case "C":
            # TODO: all of the values end up getting filtered out since the SVs are distinct
            sv2_end = SV1_R + 500 + SVLEN
            for d in range(0, 510, 2):
                data.append([case, d, [SV1, (SV1_R + d, sv2_end)]])
        case "D":
            for d in range(0, 1150, 10):
                sv3 = get_coordinates(SV1, SV3, d)
                data.append([case, d, [SV1, SV2, sv3]])
        case "E":
            for d in range(0, 1150, 10):
                sv3 = get_coordinates(MIDPOINT, SV3, d)
                data.append([case, d, [SV1, SV2, sv3]])
        case _:
            raise Exception("Invalid case")

    return data


def d_accuracy_test(n_samples: int, test_case: Optional[str] = None):
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
            weights = [1.0 / len(svs)] * len(svs)
            for _ in range(
                1
            ):  # TODO: increase the number of iterations to get a smoother curve
                args.append((case, d, svs, weights, n_samples, results))

        p.starmap(run_gmm, args)
        p.close()
        p.join()

        # results: [(case, d, gmm_model, svs, n_samples, gmm, evidence_by_mode), ...]
        write_csv(results, write_new_file=test_case is None, fixed_n_samples=n_samples)


# DEPRECATED
def run_gmm_synthetic_data():
    n_samples = [
        30,
        50,
        70,
        100,
        200,
        500,
        1000,
    ]

    with multiprocessing.Manager() as manager:
        cpu_count = multiprocessing.cpu_count()
        p = multiprocessing.Pool(cpu_count)
        results = manager.dict()
        for gmm_model in GMM_MODELS:
            results[gmm_model] = manager.list()

        args = []

        # case 1A: one SV
        sv_start = 100000
        for n in n_samples:
            for sv_size in [200, 500, 1000]:
                args.append(("1A", [(sv_start, sv_start + sv_size)], [1.0], n, results))

        # constants for each of the 2-mode cases
        sv1 = (100000, 100500)
        ps = [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]

        # case 2A: 1 smaller SV inside a bigger SV
        # vary: big SV size, big SV L
        for sv2_size in list(range(600, 1050, 50)):
            for diff in list(range(0, 250, 50)):
                sv2_start = sv1[0] - diff
                svs = [sv1, (sv2_start, sv2_start + sv2_size)]
                if sv2_start + sv2_size < sv1[1] or not is_valid_svs(svs):
                    continue
                for weights in ps:
                    for n in n_samples:
                        args.append(("2A", svs, weights, n, results))

        # case 2B: overlapping SVs
        # vary: SV 2 size, SV 2 L
        for sv2_size in list(range(300, 900, 100)):
            for diff in list(range(50, 500, 50)):
                sv2_start = sv1[0] + diff
                svs = [sv1, (sv2_start, sv2_start + sv2_size)]
                if sv2_start + sv2_size < sv1[1] or not is_valid_svs(svs):
                    continue
                for weights in ps:
                    for n in n_samples:
                        args.append(("2B", svs, weights, n, results))

        # case 2C: non overlapping SVs
        # vary: SV 2 size, SV 2 L (SV 2 start - SV 1 end)
        for sv2_size in list(range(300, 900, 100)):
            for diff in list(range(0, 300, 50)):
                sv2_start = sv1[1] + diff
                svs = [sv1, (sv2_start, sv2_start + sv2_size)]
                for weights in ps:
                    for n in n_samples:
                        args.append(("2C", svs, weights, n, results))

        ps = [
            (0.15, 0.15, 0.7),
            (0.15, 0.7, 0.15),
            (0.7, 0.15, 0.15),
            (0.33, 0.34, 0.33),
        ]
        # case 3A: combining case 2A, 2B, and 2C
        for sv2_size in list(range(600, 1050, 50)):
            for sv2_diff in list(range(0, 250, 50)):
                sv2_start = sv1[0] - sv2_diff
                sv2 = (sv2_start, sv2_start + sv2_size)
                if sv2_start + sv2_size < sv1[1] or not is_valid_svs([sv1, sv2]):
                    continue
                for sv3_size in list(range(300, 650, 50)):
                    for sv3_diff in list(range(-150, 150, 50)):
                        sv3_start = sv2[1] + sv3_diff
                        svs = [sv1, sv2, (sv3_start, sv3_start + sv3_size)]
                        for weights in ps:
                            for n in n_samples:
                                args.append(("3A", svs, weights, n, results))

        p.starmap(run_gmm, args)
        p.close()
        p.join()

        write_csv(results)


if __name__ == "__main__":
    N_SAMPLES = [47, 181, 500]  # the median and mean number of samples
    d_accuracy_test(n_samples=47)
