import os
import re

import numpy as np
import pandas as pd


def trial_stats():
    truth_set = pd.read_csv("data/synthetic_calibration/sv_subset.csv")
    truth_set["svlen"] = truth_set["stop"] - truth_set["start"]

    all_results = pd.read_csv(
        "output/synthetic_calibration/results/results.csv"
    )
    for _, row in all_results.iterrows():
        trial = row["test_name"]
        trial_dir = os.path.join("output/synthetic_calibration/results", trial)
        if not os.path.exists(trial_dir):
            continue

        n_modes = pd.read_csv(os.path.join(trial_dir, "svs_n_modes.csv"))
        for i, result_row in n_modes.iterrows():
            match = re.match(
                r"[\w+]_(\d+):(\d+)_[\d+]:(\d+)", result_row["sv_id"]
            )
            chr, start, stop = match.groups()
            n_modes.at[i, "chr"] = int(chr)
            n_modes.at[i, "start"] = int(start)
            n_modes.at[i, "stop"] = int(stop)
        n_modes = n_modes.rename(columns={"num_modes": "n_svs_predicted"})

        merged = pd.merge(
            truth_set, n_modes, on=["chr", "start", "stop"], how="left"
        )

        tp = merged[
            (merged["n_svs_actual"] == 2) & (merged["n_svs_predicted"] >= 2)
        ]
        fn = merged[
            (merged["n_svs_actual"] == 2) & (merged["n_svs_predicted"] == 1)
        ]
        tn = merged[
            (merged["n_svs_actual"] == 1) & (merged["n_svs_predicted"] == 1)
        ]
        fp = merged[
            (merged["n_svs_actual"] == 1) & (merged["n_svs_predicted"] > 1)
        ]

        for label, df in zip(["TP", "FN", "TN", "FP"], [tp, fn, tn, fp]):
            for col in ["r", "n", "svlen"]:
                print(
                    f"{trial} {label} {col} median: {np.median(df[col].values)}"
                )


if __name__ == "__main__":
    trial_stats()
