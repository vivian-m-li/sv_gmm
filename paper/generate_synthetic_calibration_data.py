import os
import random

import numpy as np
import pandas as pd

from src.synthetic.generate_data import (
    generate_sv_coordinates,
    generate_weights,
    assign_modes,
    generate_mapped_pairs_for_sv,
)
from src.utils.config_loader import load_config
from src.utils.helper import get_sample_ids, stix_output_to_df
from src.utils.model_helper import (
    calc_af,
    get_insert_size_lookup,
    reciprocal_overlap,
)
from src.utils.types import InsertSizeDistribution

from src.model.gmm_trial import gmm_trial


def amend_sv_subset():
    """
    Amends the sv_subset.csv file to include:
        - "r" column after calculating the reciprocal overlap for each SV
        - "n" column for the number of nonref samples for the SV
    """
    dir = "data/synthetic_calibration"

    truth_set = pd.read_csv(os.path.join(dir, "sv_subset.csv"))
    truth_set["r"] = 0.0
    truth_set["n"] = 0

    sample_ids = get_sample_ids(os.path.join(dir, "sample_ids.txt"))
    insert_size_lookup = get_insert_size_lookup(
        dir, "insert_sizes.csv", 450, sample_ids
    )

    for row_idx, row in truth_set.iterrows():
        chr = str(int(row["chr"]))
        start = int(row["start"])
        stop = int(row["stop"])

        stix_file = os.path.join(
            "output/synthetic_calibration/stix_output",
            f"{chr}:{start}_{chr}:{stop}.txt",
        )
        reads = stix_output_to_df(stix_file)
        truth_set.at[row_idx, "n"] = len(reads["sample_id"].unique())

        n_svs_actual = int(row["n_svs_actual"])
        if n_svs_actual == 1:
            continue

        gmm_result, evidence_by_mode = gmm_trial(
            reads,
            chr=chr,
            L=start,
            R=stop,
            init="kmeans++",
            repulsion=False,
            model_comparison_func="aic",
            plot=False,
            stem="data/synthetic_calibration",
            insert_size_lookup=insert_size_lookup,
            force_n_modes=n_svs_actual,
        )

        svs = []
        for mode in evidence_by_mode:
            lengths = []
            starts = []
            ends = []
            for evidence in mode:
                med_l = np.median(
                    [paired_end[0] for paired_end in evidence.paired_ends]
                )
                med_r = np.median(
                    [paired_end[1] for paired_end in evidence.paired_ends]
                )
                med_length = np.median(
                    [
                        paired_end[1]
                        - paired_end[0]
                        - evidence.mean_insert_size
                        for paired_end in evidence.paired_ends
                    ]
                )
                lengths.append(med_length)
                starts.append(med_l)
                ends.append(med_r)

            mode_start = int(np.median(starts))
            mode_end = int(np.median(ends))
            svs.append((mode_start, mode_end))

        truth_set.at[row_idx, "r"] = round(
            reciprocal_overlap(svs[0], svs[1]), 2
        )
        print(f"Processed {row_idx}/{truth_set.shape[0]} SVs", end="\r")

    truth_set.to_csv(os.path.join(dir, "sv_subset.csv"), index=False)


def generate_and_write_deletions(
    dir: str, n_svs: int
) -> tuple[pd.DataFrame, dict]:
    """
    Generates synthetic deletions with varying lengths and overlaps.
    """
    rng = np.random.default_rng()

    start = 1000000
    deletions = pd.DataFrame(columns=["id", "chr", "start", "stop", "svlen"])
    truth_set = pd.DataFrame(
        columns=["chr", "start", "stop", "n_svs_actual", "r"]
    )
    all_generated_svs = {}
    for _ in range(n_svs):
        case = random.choice(["A", "B", "C"])
        while True:
            svlen = int(rng.lognormal(mean=6, sigma=1.0))
            if svlen > 50:
                break
        r = random.uniform(0.4, 0.95)

        _, _, generated_svs = generate_sv_coordinates(
            case, svlen, r=r, start=start
        )[0]

        sv_start = generated_svs[0][0]
        sv_end = generated_svs[0][1]
        sv_id = f"{case}_1:{sv_start}_1:{sv_end}"

        all_generated_svs[sv_id] = generated_svs

        deletions.loc[len(deletions)] = [
            sv_id,
            "1",
            sv_start,
            sv_end,
            sv_end - sv_start,
        ]

        truth_set.loc[len(truth_set)] = [
            "1",
            sv_start,
            sv_end,
            len(generated_svs),
            r,
        ]

        start += svlen * 2  # space out the deletions

    deletions.to_csv(os.path.join(dir, "deletions.csv"), index=False)
    truth_set.to_csv(os.path.join(dir, "sv_subset.csv"), index=False)
    return deletions, all_generated_svs


def generate_and_write_reads(
    input_dir: str,
    output_dir: str,
    sv_lookup: pd.DataFrame,
    generated_svs: dict,
    sample_ids: set,
    insert_size_lookup: dict[str, InsertSizeDistribution],
) -> None:
    """
    Generates synthetic read pairs for each SV and writes them to files in the same format as stix output.
    """
    os.makedirs(output_dir, exist_ok=True)

    sv_lookup["af"] = 0.0
    sample_columns = {sample_id: "(0, 0)" for sample_id in sample_ids}
    sv_lookup = pd.concat(
        [sv_lookup, pd.DataFrame(sample_columns, index=sv_lookup.index)], axis=1
    )

    for i, sv_row in sv_lookup.iterrows():
        sv_id = sv_row["id"]
        svs = generated_svs[sv_id]

        n_samples = random.randint(11, 1000)
        samples = random.sample(sample_ids, n_samples)

        weights = generate_weights(len(svs))
        modes = assign_modes(weights, samples)

        af = calc_af(n_samples, 0, len(sample_ids))
        sv_lookup.at[i, "af"] = af

        reads = stix_output_to_df("", write_empty_file=True)
        for sample, mode in zip(samples, modes):
            sv_lookup.at[i, sample] = "(1, 1)"
            n_pairs = random.randint(2, 30)

            mode_start, mode_end = svs[mode]
            insert_size = insert_size_lookup[sample].mean
            pairs = generate_mapped_pairs_for_sv(
                mode_start, mode_end, insert_size, n_pairs
            )
            for l_start, l_end, r_start, r_end in pairs:
                reads.loc[len(reads)] = [
                    0,
                    sample,
                    1,
                    int(l_start),
                    int(l_end),
                    1,
                    int(r_start),
                    int(r_end),
                    "paired",
                ]

        filename = "_".join(sv_id.split("_")[1:])
        reads.to_csv(
            os.path.join(output_dir, f"{filename}.txt"),
            sep="\t",
            header=False,
            index=False,
        )

    sv_lookup.to_csv(os.path.join(input_dir, "deletions.csv"), index=False)


def generate(cfg: dict, n_svs: int) -> None:
    """
    Generates synthetic SVs for calibration.
    Uses the same sample ids and insert sizes from the 1kg data, but vary the
    SV coordinates/lengths and number of supporting reads.
    """
    input_dir = cfg["paths"]["input_dir"]
    output_dir = cfg["paths"]["stix_output_dir"]

    sample_ids = list(
        get_sample_ids(
            os.path.join(input_dir, cfg["input_files"]["sample_id_file"])
        )
    )
    insert_size_lookup = get_insert_size_lookup(
        input_dir, cfg["input_files"]["insert_size_file"]
    )

    sv_lookup, generated_svs = generate_and_write_deletions(input_dir, n_svs)
    generate_and_write_reads(
        input_dir,
        output_dir,
        sv_lookup,
        generated_svs,
        sample_ids,
        insert_size_lookup,
    )


if __name__ == "__main__":
    cfg = load_config()
    generate(cfg, 1000)
