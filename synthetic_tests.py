import csv
import multiprocessing
import numpy as np
from generate_data import generate_synthetic_sv_data


def run_gmm(case, svs, weights, n_samples, results):
    gmm, evidence_by_mode = generate_synthetic_sv_data(
        1, svs, n_samples=n_samples, p=weights
    )
    results.append((case, svs, n_samples, gmm, evidence_by_mode))


def get_len_L(evidence_by_mode):
    lengths = []
    Ls = []
    for mode in evidence_by_mode:
        lens = []
        starts = []
        for evidence in mode:
            max_l = max([paired_end[0] for paired_end in evidence.paired_ends])
            min_r = min([paired_end[1] for paired_end in evidence.paired_ends])
            lens.append(min_r - max_l - 450)  # 450 is the length of the read
            starts.append(max_l)
        lengths.append(np.mean(lens))
        Ls.append(np.mean(starts))
    return lengths, Ls


def write_csv(results):
    with open("synthetic_data/results_2d.csv", mode="w", newline="") as out:
        fieldnames = [
            "case",
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
        for case, svs, n_samples, gmm, evidence_by_mode in results:
            lengths, Ls = get_len_L(evidence_by_mode)
            csv_writer.writerow(
                {
                    "case": case,
                    "expected_num_modes": int(case[0]),
                    "expected_lengths": [sv[1] - sv[0] for sv in svs],
                    "expected_Ls": [sv[0] for sv in svs],
                    "num_modes": gmm.num_modes,
                    "lengths": lengths,
                    "Ls": Ls,
                    "num_samples": n_samples,
                }
            )


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
        results = manager.list()
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
                if sv2_start + sv2_size < sv1[1]:
                    continue
                svs = [sv1, (sv2_start, sv2_start + sv2_size)]
                for weights in ps:
                    for n in n_samples:
                        args.append(("2A", svs, weights, n, results))

        # case 2B: overlapping SVs
        # vary: SV 2 size, SV 2 L
        for sv2_size in list(range(300, 900, 100)):
            for diff in list(range(50, 500, 50)):
                sv2_start = sv1[0] + diff
                if sv2_start + sv2_size < sv1[1]:
                    continue
                svs = [sv1, (sv2_start, sv2_start + sv2_size)]
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
    run_gmm_synthetic_data()
