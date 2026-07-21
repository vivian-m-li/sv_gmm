import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.helper import get_sample_ids, stix_output_to_df
from src.utils.model_helper import giggle_format
from src.utils.write_sv_output import get_raw_data

COLORS = matplotlib.colormaps["tab10"].colors

AXES = {
    "l_diff": "Start diff (l_start - sv_start)",
    "r_diff": "End diff (r_end - sv_end)",
    "fragment_size": "Fragment Size (r_end - l_start)",
    "insert_size": "Insert Size (r_end - l_start - sv_len)",
}


def get_nonref_reads(
    sv_id: str,
    lookup: pd.DataFrame | None = None,
    *,
    stix_output_dir: str = "output/stix_output",
):
    if lookup is None:
        lookup = pd.read_csv("data/1kg/1kg.subset.csv", low_memory=False)
    row = lookup[lookup["id"] == sv_id].iloc[0]

    print(
        f"{sv_id}: {row['chr']}:{row['start']}-{row['stop']}, svlen={row['svlen']}"
    )

    # get all nonref samples
    sample_ids = get_sample_ids("data/1kg/sample_ids.txt")
    nonref_samples = [
        sample_id for sample_id in sample_ids if row[sample_id] != "(0, 0)"
    ]

    # filter reads to only nonref samples
    chr, start, stop = row["chr"], row["start"], row["stop"]
    reads = stix_output_to_df(
        os.path.join(
            stix_output_dir,
            f"{giggle_format(chr, start)}_{giggle_format(chr, stop)}.txt",
        )
    )
    reads = reads[reads["sample_id"].isin(nonref_samples)]
    reads["l_diff"] = reads["l_start"] - start
    reads["r_diff"] = reads["r_end"] - stop
    reads["fragment_len"] = reads["r_end"] - reads["l_start"]
    reads["insert_size"] = reads["fragment_len"] - (stop - start)

    return reads, (chr, start, stop)


def plot_read_distribution(sv_id: str, reads: np.ndarray, x: str, y: str):
    plt.figure()
    plt.scatter(
        reads[:, 0],
        reads[:, 1],
        alpha=0.6,
    )

    plt.title(f"Read distribution for SV {sv_id}")
    plt.xlabel(AXES[x])
    plt.ylabel(AXES[y])
    plt.savefig(f"output/plots/outer_bounds/{sv_id}_{x}_{y}.png")
    plt.show()


def plot_1d_read_distribution(sv_id: str, reads: np.ndarray, x: str):
    plt.figure()
    plt.hist(
        reads,
        bins=30,
        alpha=0.6,
    )

    plt.title(f"Distribution for SV {sv_id}")
    plt.xlabel(AXES[x])
    plt.ylabel("Count")
    plt.show()
    plt.savefig(f"output/plots/outer_bounds/{sv_id}_{x}.png")


def plot_sample_summary_read_distribution(
    sample_id: str, sv_id: str, sample_reads: pd.DataFrame, x: str, y: str
):
    plt.figure()
    plt.scatter(
        sample_reads[x],
        sample_reads[y],
        alpha=0.6,
    )
    plt.scatter(
        sample_reads[x].median(),
        sample_reads[y].median(),
        color="red",
        label="Median",
    )

    plt.title(f"Read distribution for sample {sample_id} and SV {sv_id}")
    plt.xlabel(AXES[x])
    plt.ylabel(AXES[y])
    plt.show()


def get_read_distribution(
    sv_id: str,
    *,
    x: str,
    y: str,
    plot: bool = False,
    plot_1d: bool = False,
    sample_summary: bool = False,
    lookup: pd.DataFrame | None = None,
):
    """
    Gets the distribution of reads for each sample for a given SV.
    Hard-coded file paths.
    """
    reads, _ = get_nonref_reads(sv_id, lookup=lookup)

    read_ends = []
    if sample_summary:
        # within-sample distribution
        for i, sample_id in enumerate(reads["sample_id"].unique()):
            sample_reads = reads[reads["sample_id"] == sample_id]
            read_ends.append(
                [
                    sample_reads[x].median(),
                    sample_reads[y].median(),
                ]
            )
            if i == 0 and plot:
                plot_sample_summary_read_distribution(
                    sample_id, sv_id, sample_reads, x, y
                )
            elif not plot:
                print(f"Sample {sample_id}:")
                print(
                    f"{AXES[x]}: mean={np.mean(sample_reads[x]):.2f}, std={np.std(sample_reads[x]):.2f}"
                )
                print(
                    f"{AXES[y]}: mean={np.mean(sample_reads[y]):.2f}, std={np.std(sample_reads[y]):.2f}"
                )
    else:
        read_ends = reads[[x, y]].values

    # between-sample read medians distribution
    read_ends = np.array(read_ends)
    print(
        f"{AXES[x]}: mean={np.mean(read_ends[:, 0]):.2f}, std={np.std(read_ends[:, 0]):.2f}"
    )
    print(
        f"{AXES[y]}: mean={np.mean(read_ends[:, 1]):.2f}, std={np.std(read_ends[:, 1]):.2f}"
    )

    if plot:
        plot_read_distribution(sv_id, read_ends, x, y)

    if plot_1d:
        plot_1d_read_distribution(sv_id, read_ends[:, 0], x)
        plot_1d_read_distribution(sv_id, read_ends[:, 1], y)


def plot_nearby_svs(cfg: dict, svs: str):
    input_dir = cfg["paths"]["input_dir"]
    sv_lookup_file = cfg["paths"]["sv_lookup_file"]
    deletions = pd.read_csv(
        os.path.join(input_dir, sv_lookup_file), low_memory=False
    )

    plt.figure()
    for i, sv_id in svs:
        row = deletions[deletions["id"] == sv_id].iloc[0]
        reads, _ = get_raw_data(row, cfg, filter_reference_samples=True)
        l_r_points = []
        for sample_id in reads["sample_id"].unique():
            sample_reads = reads[reads["sample_id"] == sample_id]
            l_r_points.append(
                [
                    sample_reads["l_start"].median() - row["start"],
                    sample_reads["r_end"].median() - row["stop"],
                ]
            )
        l_r_points = np.array(l_r_points)

        plt.scatter(
            l_r_points[:, 0], l_r_points[:, 1], alpha=0.6, color=COLORS[i]
        )
        plt.axvline(x=0, color="gray", linestyle="--", label="SV start")
        plt.axhline(y=0, color="gray", linestyle="--", label="SV end")

        plt.title(f"Read distribution for SV {sv_id}")
        plt.xlabel("Start diff (l_start - sv_start)")
        plt.ylabel("End diff (r_end - sv_end)")
        plt.show()


def analyze_read_types(
    sv_id: str,
    *,
    plot: bool = False,
    sample_summary: bool = False,
    lookup: pd.DataFrame | None = None,
):
    """
    Analyzes the distribution of read types (split vs paired) for a given SV.
    """
    reads, pos = get_nonref_reads(sv_id, lookup=lookup)
    total_samples = reads["sample_id"].nunique()

    if plot:
        fig, axs = plt.subplots(1, 2)
        for read_type, color in zip(["split", "paired"], ["blue", "red"]):
            reads_subset = reads[reads["type"] == read_type]
            if reads_subset.empty:
                continue

            for i, col in enumerate(["l_start", "r_end"]):
                values = []
                if sample_summary:
                    for sample_id in reads["sample_id"].unique():
                        sample_reads = reads_subset[
                            reads_subset["sample_id"] == sample_id
                        ]
                        values.append(sample_reads[col].median())
                else:
                    values = reads_subset[col]

                axs[i].hist(
                    values,
                    bins=30,
                    alpha=0.6,
                    label=read_type,
                    color=color,
                )

        axs[0].axvline(x=pos[1], color="gray", linestyle="--")
        axs[1].axvline(x=pos[2], color="gray", linestyle="--")
        axs[1].set_yticks([])

        plt.text(0.5, 0.1, "Read position")
        axs[0].set_ylabel("Count")
        plt.suptitle(f"Split vs PE Reads for SV {sv_id}")
        plt.legend()
        plt.savefig(
            f"output/plots/read_distribution_outer_bounds/read_types/{sv_id}_{'median' if sample_summary else 'all'}.png"
        )
        plt.close(fig)
    else:
        reads["l_start_diff"] = reads["l_start"] - pos[1]
        reads["l_end_diff"] = reads["l_end"] - pos[1]
        reads["r_start_diff"] = reads["r_start"] - pos[2]
        reads["r_end_diff"] = reads["r_end"] - pos[2]
        for read_type in ["split", "paired"]:
            reads_subset = reads[reads["type"] == read_type]
            print(
                f"{reads_subset.shape[0]} {read_type} reads, {reads_subset['sample_id'].nunique()}/{total_samples} samples"
            )
            for col in [
                "l_start_diff",
                "l_end_diff",
                "r_start_diff",
                "r_end_diff",
            ]:
                print(
                    "{}: mean={:.2f}, std={:.2f}".format(
                        col,
                        np.mean(reads_subset[col]),
                        np.std(reads_subset[col]),
                    )
                )
        print("\n")


def analyze_query_region(
    cfg: dict,
    sv_id: str,
    *,
    download_reads: bool = False,
    sample_summary: bool = False,
    lookup: pd.DataFrame | None = None,
):
    if lookup is None:
        lookup = pd.read_csv("data/1kg/1kg.subset.csv", low_memory=False)
    row = lookup[lookup["id"] == sv_id].iloc[0]
    start, stop = row["start"], row["stop"]

    stix_output_dir = "output/query_region_analysis/"
    fig, axs = plt.subplots(5, 2, figsize=(6, 8))
    q_vals = [0.6, 0.7, 0.8, 0.9, 1.0]
    min_l = np.inf
    max_r = -np.inf
    for i, q in enumerate(q_vals):
        print(f"Query region: {q}")

        output_dir = os.path.join(stix_output_dir, f"stix_output_{q}")
        if download_reads:
            os.makedirs(output_dir, exist_ok=True)

            temp_cfg = cfg.copy()
            temp_cfg["paths"]["stix_output_dir"] = output_dir
            get_raw_data(
                row, temp_cfg, read_overlap=q, filter_reference_samples=True
            )

        reads, _ = get_nonref_reads(
            sv_id, lookup=lookup, stix_output_dir=output_dir
        )
        print(
            f"{reads.shape[0]} total reads, {reads['sample_id'].nunique()} unique samples for query region {q}"
        )

        for read_type, color in zip(["split", "paired"], ["blue", "red"]):
            reads_subset = reads[reads["type"] == read_type]
            if reads_subset.empty:
                continue

            for j, col in enumerate(["l_start", "r_end"]):
                values = []
                if sample_summary:
                    for sample_id in reads["sample_id"].unique():
                        sample_reads = reads_subset[
                            reads_subset["sample_id"] == sample_id
                        ]
                        values.append(sample_reads[col].median())
                else:
                    values = reads_subset[col]

                if j == 0:
                    min_l = min(min_l, min(values))
                elif j == 1:
                    max_r = max(max_r, max(values))

                axs[i][j].hist(
                    values,
                    bins=30,
                    alpha=0.6,
                    label=read_type,
                    color=color,
                )

            axs[i][0].axvline(x=start, color="gray", linestyle="--")
            axs[i][1].axvline(x=stop, color="gray", linestyle="--")
            axs[i][1].set_yticks([])

            axs[i][0].set_ylabel(f"q={q}")

    # set all x-limits to the same range
    buffer = (stop - start) * 0.02
    for i in range(len(q_vals)):
        axs[i][0].set_xlim(min_l, start + buffer)
        axs[i][1].set_xlim(stop - buffer, max_r)

    plt.suptitle(f"Split vs PE Reads for Varying Query Regions for SV {sv_id}")
    fig.text(0.45, 0.01, "Read position", fontsize=12)
    fig.text(0.01, 0.5, "Count", rotation=90, fontsize=12)
    axs[0][1].legend()
    plt.subplots_adjust(
        left=0.15, bottom=0.09, right=0.95, top=0.925, wspace=0.05, hspace=0.48
    )
    plt.savefig(
        f"output/plots/read_distribution_outer_bounds/query_region/{sv_id}_{'median' if sample_summary else 'all'}.png"
    )
    plt.close(fig)


if __name__ == "__main__":
    cfg = load_config()

    # example SVs
    # 1 mode: HGSV_15262, HGSV_143868, HGSV_39753, HGSV_204881, HGSV_226693, HGSV_5515, HGSV_218106, HGSV_89
    # 2 modes: HGSV_54541, HGSV_149774, HGSV_245658, HGSV_68297, HGSV_220750, HGSV_161412
    lookup = pd.read_csv("data/1kg/1kg.subset.plotting.csv", low_memory=False)
    for sv_id in [
        "HGSV_15262",
        "HGSV_143868",
        "HGSV_39753",
        "HGSV_204881",
        "HGSV_226693",
        "HGSV_5515",
        "HGSV_218106",
        "HGSV_89",
        "HGSV_54541",
        "HGSV_149774",
        "HGSV_245658",
        "HGSV_220750",
        "HGSV_161412",
    ]:
        analyze_read_types(
            sv_id,
            plot=True,
            sample_summary=False,
            lookup=lookup,
        )
        analyze_read_types(
            sv_id,
            plot=True,
            sample_summary=True,
            lookup=lookup,
        )
        analyze_query_region(
            cfg,
            sv_id,
            sample_summary=False,
            lookup=lookup,
        )
        analyze_query_region(
            cfg,
            sv_id,
            sample_summary=True,
            lookup=lookup,
        )
