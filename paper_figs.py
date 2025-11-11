import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from helper import get_sv_stats_collapsed_df, get_sv_lookup


def load_synthetic_data_results(sample_size: int) -> pd.DataFrame:
    results_file = "synthetic_data/results.csv"
    add_file = f"synthetic_data/resultsn={sample_size}.csv"
    df = pd.read_csv(results_file)
    add_df = pd.read_csv(add_file)
    add_df = add_df[add_df["gmm_model"] == "2d"]
    combined_df = pd.concat([df, add_df], ignore_index=True)
    return combined_df


def plot_reciprocal_overlap_svlen(
    case: str, sample_size: int, *, y_axis: str = "accuracy"
):
    df = load_synthetic_data_results(sample_size)
    df = df[(df["case"] == case)]

    models = ["2d", "gatk_MAX_CLIQUE", "gatk_SINGLE_LINKAGE"]
    colors = ["#ffed00", "#ffce0b", "#f8ac30", "#f0853c", "#eb5c3f"]
    fig, axs = plt.subplots(1, len(models), figsize=(5 * len(models), 5))

    for model in models:
        subset = df[df["gmm_model"] == model]
        ax = axs[models.index(model)]
        for i, svlen in enumerate(sorted(subset["svlen"].unique())):
            svlen_df = subset[subset["svlen"] == svlen]
            right = defaultdict(lambda: 0)
            total = defaultdict(lambda: 0)
            n_modes_counts = defaultdict(list)
            for _, row in svlen_df.iterrows():
                overlap = row["r"]
                total[overlap] += 1
                n_modes_counts[overlap].append(row["num_modes"])
                if row["expected_num_modes"] == row["num_modes"]:
                    right[overlap] += 1

            overlaps = sorted(total.keys())
            acc = [right[overlap] / total[overlap] for overlap in overlaps]
            n_modes = [np.mean(n_modes_counts[overlap]) for overlap in overlaps]

            y_vals = acc if y_axis == "accuracy" else n_modes
            ax.plot(
                overlaps,
                y_vals,
                marker="o",
                color=colors[i],
                label=f"svlen={svlen}",
                linewidth=2,
            )
            ax.set_xlabel("Reciprocal Overlap (r)", fontsize=14)
            ylabel = "Accuracy" if y_axis == "accuracy" else "Predicted # SVs"
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_title(model, fontsize=16)

            if y_axis == "accuracy":
                ax.set_ylim(-0.05, 1.1)

    plt.legend(fontsize=12, title="SV Length", title_fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_sv_breakdown():
    """Figure 2 - horizontal bar charts of the breakdown of SVs by number of modes"""
    full_df = get_sv_stats_collapsed_df()
    full_df["num_samples_run"] = full_df["num_samples"] - (
        full_df["num_pruned"] + full_df["num_reference"]
    )
    full_df.rename(columns={"id": "sv_id"}, inplace=True)
    full_df = full_df[["sv_id", "num_samples_run"]]
    df = pd.read_csv("1kgp/svs_n_modes.csv")
    df = df.merge(full_df, on="sv_id")

    sv_lookup = get_sv_lookup()
    svs_not_enough_evidence = df[
        (df["num_samples_run"] > 0) & (df["num_samples_run"] <= 10)
    ]
    svs_clustered = df[df["num_samples_run"] > 10]
    sv_ids_run = (
        svs_clustered["sv_id"].values.tolist()
        + svs_not_enough_evidence["sv_id"].values.tolist()
    )
    svs_no_evidence = sv_lookup[~sv_lookup["id"].isin(sv_ids_run)]
    svs_no_evidence.rename(columns={"id": "sv_id"}, inplace=True)

    n_svs = []
    afs = []
    for df_subset in [svs_no_evidence, svs_not_enough_evidence, svs_clustered]:
        n_svs.append(df_subset.shape[0])
        afs.append(
            np.mean(
                sv_lookup[sv_lookup["id"].isin(df_subset["sv_id"].values)]["af"]
            )
        )

    # Plot left figure: horizontal bar chart of the number of SVs
    fig, axs = plt.subplots(
        1, 2, figsize=(12, 3), gridspec_kw={"width_ratios": [1, 2]}
    )
    ax1 = axs[0]
    categories = ["Clustered", "Inconclusive", "No Evidence"]
    ax1.barh(categories, n_svs[::-1], color="#274c77")
    for i, (n, af) in enumerate(zip(n_svs[::-1], afs[::-1])):
        ax1.text(
            n - 500,
            i,
            f"AF = {af:.3f}",
            va="center",
            ha="right",
            fontsize=10,
            color="white",
        )
    ax1.set_xlabel("Number of SVs")
    ax1.set_title("SV Breakdown by Evidence")

    # Filter out SVs with too little evidence
    df = df[(df["confidence"] != "inconclusive") & (df["num_samples_run"] > 10)]
    modes = []
    confidence = []
    n_samples = []
    for n in [1, 2, 3]:
        modes_df = df[df["num_modes"] == n]
        modes.append(modes_df.shape[0])
        confidence.append(Counter(modes_df["confidence"]))
        n_samples.append(np.mean(modes_df["num_samples_run"]))

    # Plot right figure: horizontal bar chart of the number of SVs by number of modes
    ax2 = axs[1]
    mode_labels = ["1 Mode", "2 Modes", "3 Modes"]
    bottom = np.zeros(len(modes))
    colors = {"high": "#7BB662", "medium": "#FFD301", "low": "#E03C32"}
    for conf in ["high", "medium", "low"]:
        values = [confidence[i].get(conf, 0) for i in range(len(modes))]
        ax2.barh(
            mode_labels,
            values,
            left=bottom,
            label=conf.capitalize(),
            color=colors[conf],
        )
        bottom += values
    for i, n in enumerate(n_samples):
        pos_x = modes[i] if i == 0 else modes[i] + 100
        pos_y = i + 0.55 if i == 0 else i
        ha = "right" if i == 0 else "left"
        ax2.text(pos_x, pos_y, f"n = {n:.0f}", va="center", ha=ha, fontsize=10)
    ax2.set_xlabel("Number of SVs")
    ax2.set_title("SV Breakdown by Number of Modes")
    ax2.legend(title="Confidence")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # methods_figure_viz(
    #     "synthetic_data/data/B_r0.5_svlen802_n66_fcd8b157-8b23-4c8d-9574-2e8994ceb2d7.vcf"
    # )
    # plot_reciprocal_overlap_svlen(case="B", sample_size=66, y_axis="num_modes")
    plot_sv_breakdown()
