import random
import pysam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.lines import Line2D
from collections import Counter, defaultdict
from process_data import run_viz_gmm
from viz import plot_2d_coords, plot_single_sv
from helper import get_sv_stats_collapsed_df, get_sv_lookup
from gmm_types import COLORS
from matplotlib.gridspec import GridSpec


def plot_sv_coordinate_space(ax, n_svs, xcoords, svcoords, y, height, **kwargs):
    """
    Draws a capsule-like SV box using an ellipse for the ends and a rectangle in the middle.
    x, y = bottom-left of the rectangle
    width, height = dimensions of the total capsule
    """
    facecolor = kwargs.get("facecolor", "lightgrey")
    edgecolor = kwargs.get("edgecolor", "black")
    lw = kwargs.get("lw", 1)

    if kwargs.get("r") is not None:
        ellipse_r = kwargs.get("r")
    else:
        ellipse_r = 0.075 * (xcoords[1] - xcoords[0]) / n_svs
    rect_start = xcoords[0] + ellipse_r
    rect_end = xcoords[1] - ellipse_r

    # plot this line so the boxes have something to attach to
    ax.plot(
        [xcoords[0], xcoords[1]],
        [y + 0.1, y + 0.1],
        alpha=0,
        linewidth=2,
    )

    left_ellipse = Ellipse(
        (rect_start, y + height / 2),
        width=ellipse_r * 2,
        height=height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        lw=lw,
    )
    right_ellipse = Ellipse(
        (rect_end, y + height / 2),
        width=ellipse_r * 2,
        height=height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        lw=lw,
    )
    rect = Rectangle(
        (rect_start, y),
        rect_end - rect_start,
        height,
        facecolor=facecolor,
        edgecolor="none",
    )
    top_border = Line2D(
        [rect_start, rect_end],
        [y + height, y + height],
        color=edgecolor,
        linewidth=lw,
    )
    bottom_border = Line2D(
        [rect_start, rect_end],
        [y, y],
        color=edgecolor,
        linewidth=lw,
    )
    ax.add_patch(left_ellipse)
    ax.add_patch(right_ellipse)
    ax.add_patch(rect)
    ax.add_line(top_border)
    ax.add_line(bottom_border)

    sv_rect = Rectangle(
        (svcoords[0], y),
        svcoords[1] - svcoords[0],
        height,
        facecolor="#274c77",
        edgecolor="none",
    )
    ax.add_patch(sv_rect)

    return [left_ellipse, right_ellipse, rect, top_border, bottom_border]


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


def synthetic_data_fig():
    """Synthetic data results comparison: my method vs. GATK clustering methods"""
    cases = ["B", "C", "D"]
    sample_size = 313
    svlen = 802
    df = load_synthetic_data_results(sample_size)
    df = df[df["svlen"] == svlen]

    models = ["2d", "gatk_MAX_CLIQUE", "gatk_SINGLE_LINKAGE"]
    colors = ["#bfdbf7", "#1f7a8c", "#022b3a"]
    markers = ["o", "s", "D"]

    svs = [
        [(100000, 100802), (100200, 100601)],
        [(100000, 100802), (100401, 101203)],
        [(100000, 100802), (100401, 101203), (100802, 101604)],
    ]

    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(3, len(cases), figure=fig, height_ratios=[1, 4, 1])

    axs = []
    # set up the grid of axes
    for col in range(len(cases)):
        if col < len(cases) - 1:
            new_axes = [fig.add_subplot(gs[row, col]) for row in range(3)]
        else:
            gs_last_col = GridSpec(
                2,
                1,
                figure=fig,
                height_ratios=[5.12, 0.88],
                left=0.7,
                right=0.98,
                hspace=0.23,
            )
            new_axes = [None]
            new_axes.extend(
                [fig.add_subplot(gs_last_col[row]) for row in range(2)]
            )
        axs.append(new_axes)

    for col, case in enumerate(cases):
        tpr_ax = axs[col][1]
        fpr_ax = axs[col][2]
        subset = df[df["case"] == case]

        all_overlaps, all_accs, all_fps = [], [], []
        for i, model in enumerate(models):
            model_df = subset[subset["gmm_model"] == model]
            right, total, false_positives = (
                defaultdict(int),
                defaultdict(int),
                defaultdict(int),
            )
            for _, row in model_df.iterrows():
                overlap = row["r"]
                total[overlap] += 1
                if row["expected_num_modes"] == row["num_modes"]:
                    right[overlap] += 1
                if row["num_modes"] > row["expected_num_modes"]:
                    false_positives[overlap] += (
                        row["num_modes"] - row["expected_num_modes"]
                    )

            overlaps = np.array(sorted(total.keys()))
            acc = np.array([right[o] / total[o] for o in overlaps])
            fps = np.array([false_positives[o] for o in overlaps])
            all_overlaps.append(overlaps)
            all_accs.append(acc)
            all_fps.append(fps)

            # plot the lines without markers
            tpr_ax.plot(
                overlaps,
                acc,
                color=colors[i],
                alpha=0.8,
                label=model,
            )
            fpr_ax.plot(
                overlaps,
                fps,
                color=colors[i],
                alpha=0.8,
                label=model,
            )

        for i, model in enumerate(models):
            # plot only select markers to avoid clutter
            overlaps_subset = all_overlaps[i][i:][:: len(models)]
            tpr_ax.scatter(
                overlaps_subset,
                all_accs[i][i:][:: len(models)],
                marker=markers[i],
                color=colors[i],
                alpha=0.8,
                zorder=3,
            )

            fpr_ax.scatter(
                overlaps_subset,
                all_fps[i][i:][:: len(models)],
                marker=markers[i],
                color=colors[i],
                alpha=0.8,
                zorder=3,
            )

        xlabel = "r1" if case == "D" else "r"
        fpr_ax.set_xlabel(f"Reciprocal Overlap ({xlabel})", fontsize=13)
        fpr_ax.set_ylim(-0.05, 0.05)
        fpr_ax.set_yticks([0.0])
        fpr_ax.set_yticklabels(["0.0"])

        if case == "D":
            # set the tpr y axis to 0, 1.5
            # but tick marks only at 0.0, 0.2, ... 1.0
            tpr_ax.set_ylim(-0.05, 1.5)
            tpr_ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            tpr_ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        else:
            tpr_ax.set_ylim(-0.05, 1.05)

        if case == "B":
            tpr_ax.set_ylabel("True Positive Rate (TPR)", fontsize=13)
            fpr_ax.set_ylabel("FPR", fontsize=13)

        # draw the conceptual SVs (i.e. the patches representing the coordinate space)
        # manually add the drawn SVs to the last column (3 SV case)
        if col < 2:
            case_svs = svs[col]
            x_min = min([sv[0] for sv in case_svs]) - 100
            x_max = max([sv[1] for sv in case_svs]) + 100
            y_offset = 0
            sub_ax = axs[col][0]
            sub_ax.axis("off")
            for i, (L, R) in enumerate(case_svs):
                plot_sv_coordinate_space(
                    sub_ax,
                    len(case_svs),
                    (x_min, x_max),
                    (L, R),
                    y_offset,
                    0.2,
                    edgecolor="black",
                    facecolor="lightgrey",
                )
                y_offset += 0.25

    axs[1][-1].legend(
        loc="upper center", title="Model", ncols=3, bbox_to_anchor=(0.5, -0.9)
    )

    plt.tight_layout()
    plt.subplots_adjust(
        top=0.97,
        bottom=0.2,
        wspace=0.15,
        hspace=0.37,
    )
    plt.savefig("plots/synthetic_data.pdf")
    plt.savefig("plots/synthetic_data.png")
    plt.show()


def synthetic_data_additional_svs():
    svs = [
        [
            (100000, 100802),
            (100722, 101524),
            (101444, 102246),
        ],  # r1 = 0.1, r2 = 0.1
        [
            (100000, 100802),
            (100722, 101524),
            (100803, 101605),
        ],  # r1 = 0.1, r2 = 0.9
        [
            (100000, 100802),
            (100081, 100883),
            (100482, 101284),
        ],  # r1 = 0.9, r2 = 0.5
    ]

    fig, axs = plt.subplots(len(svs), 1, figsize=(8, 2 * len(svs)))
    for i, sv_coords in enumerate(svs):
        y_offset = 0.2
        x_min = min([sv[0] for sv in sv_coords]) - 100
        x_max = max([sv[1] for sv in sv_coords]) + 100
        for j, (L, R) in enumerate(sv_coords):
            plot_sv_coordinate_space(
                axs[i],
                len(sv_coords),
                (x_min, x_max),
                (L, R),
                y_offset,
                0.2,
                edgecolor="black",
                facecolor="lightgrey",
            )
            y_offset += 0.25
        axs[i].axis("off")

    plt.tight_layout()
    plt.savefig("plots/synthetic_data_additional_svs.pdf")
    plt.show()


def methods_raw_reads(svs, reads):
    """Figure 2a - Visualizes the reads in coordinate space."""
    sv_avg = (np.mean([sv[0] for sv in svs]), np.mean([sv[1] for sv in svs]))

    fig, ax = plt.subplots(figsize=(4, 3))

    # plot reads
    y_offset = 0.2
    all_reads = list(reads.values())
    random.shuffle(all_reads)
    reads_plotted = []
    n_reads_shown = 30
    for i, sample_reads in enumerate(all_reads):
        read = random.sample(sample_reads, 1)[0]
        if i >= n_reads_shown:
            reads_plotted.append(read)
            continue
        reads_plotted.append(read)
        color = COLORS[1] if i == n_reads_shown - 1 else "#274c77"
        ax.scatter(
            [read[0], read[1]],
            [y_offset, y_offset],
            color=color,
            s=10,
        )
        ax.plot(
            [read[0], read[1]],
            [y_offset, y_offset],
            color=color,
            alpha=0.9,
        )
        y_offset += 0.05

    reads_min = min([read[0] for read in reads_plotted])
    reads_max = max([read[1] for read in reads_plotted])
    plot_sv_coordinate_space(
        ax,
        1,
        xcoords=(reads_min - 100, reads_max + 100),
        svcoords=sv_avg,
        y=y_offset + 0.05,
        height=0.1,
        r=30,
        edgecolor="black",
        facecolor="lightgrey",
    )

    # draw a line with arrows on either end on the bottom
    ax.annotate(
        "",
        xy=(reads_min - 50, 0.1),
        xytext=(reads_max + 50, 0.1),
        arrowprops=dict(arrowstyle="<->", color="black"),
    )

    # draw vertical dotted lines from the first read to the drawn axis
    last_read = reads_plotted[n_reads_shown - 1]
    ax.plot(
        [last_read[0], last_read[0]],
        [y_offset - 0.05, 0.1],
        linestyle="dotted",
        color=COLORS[1],
        linewidth=2,
    )
    ax.plot(
        [last_read[1], last_read[1]],
        [y_offset - 0.05, 0.1],
        linestyle="dotted",
        color=COLORS[1],
        linewidth=2,
    )

    # label the L/R of the read as L_i and R_i
    ax.text(
        last_read[0],
        0,
        "Lᵢ",
        ha="center",
        va="center",
        color=COLORS[1],
        fontsize=16,
    )
    ax.text(
        last_read[1],
        0,
        "Rᵢ",
        ha="center",
        va="center",
        color=COLORS[1],
        fontsize=16,
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig("plots/methodsa.png")
    plt.savefig("plots/methodsa.pdf")
    plt.show()

    return reads_plotted


def methods_lr(reads_plotted):
    """Figure 2b - Visualizes the reads in length vs. read L position space."""
    fig, ax = plt.subplots(figsize=(4, 3))

    L, R = 0, 0
    read_i = reads_plotted[-1]
    for read in reads_plotted:
        read_L = read[0]
        read_R = read[1]
        if read_L == read_i[0] and read_R == read_i[1]:
            ax.scatter(read_L, read_R, color=COLORS[1], s=30)
            L, R = read_L, read_R
        else:
            ax.scatter(read_L, read_R, color="#274c77", s=10)

    # label the L/R of the read as L_i and R_i
    ax.text(
        L,
        -50,
        "Lᵢ",
        ha="center",
        va="center",
        color=COLORS[1],
        fontsize=10,
    )
    ax.text(
        -200,
        R,
        "Rᵢ",
        # "Rᵢ - Lᵢ",
        ha="center",
        va="center",
        color=COLORS[1],
        fontsize=10,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Read L", fontsize=16)
    ax.set_ylabel("Read R", fontsize=16)
    plt.tight_layout()
    plt.savefig("plots/methodsb.png")
    plt.savefig("plots/methodsb.pdf")
    plt.show()


def methods_clustered(svs, reads, insert_size_lookup):
    """Figure 2c - Visualizes the clustered reads in length vs. read L position space."""

    squiggle_data = {}
    for sample_id, sample_reads in reads.items():
        squiggle_data[sample_id] = []
        for read in sample_reads:
            squiggle_data[sample_id].extend(read)

    sv_avg = (np.mean([sv[0] for sv in svs]), np.mean([sv[1] for sv in svs]))
    gmm, evidence = run_viz_gmm(
        squiggle_data,
        chr="1",
        L=sv_avg[0],
        R=sv_avg[1],
        plot=False,
        synthetic_data=True,
        min_pairs=2,
        insert_size_lookup=insert_size_lookup,
    )

    fig, ax = plt.subplots(figsize=(4, 3))
    plot_2d_coords(
        ax,
        evidence,
        L=sv_avg[0],
        R=sv_avg[1],
        axis1="L",
        axis2="R",
        size_by="",
        show_mode_stats=False,
        show_1d_distributions=False,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Read L", fontsize=16)
    ax.set_ylabel("Read R", fontsize=16)
    plt.tight_layout()
    plt.savefig("plots/methodsc.png")
    plt.savefig("plots/methodsc.pdf")
    plt.show()


def methods_figure_viz(svs, vcf_file: str):
    """
    Figure 2 - Generates all of the sub-figures for the methods visualization figure.
    vcf_file is a vcf generated by the generate_synthetic_sv_data function in generate_data.py.
    """
    # parse vcf
    reads = defaultdict(list)
    mean_insert_sizes = {}
    vcf = pysam.VariantFile(vcf_file)
    for record in vcf.fetch():
        info = record.info
        sample_id = info["SAMPLE"]
        start = record.pos
        end = info["END2"]
        mean_insert_size = info["MEAN_INSERT"]
        reads[sample_id].append((start, end))
        mean_insert_sizes[sample_id] = mean_insert_size

    reads_plotted = methods_raw_reads(svs, reads)
    methods_lr(reads_plotted)
    methods_clustered(svs, reads, mean_insert_sizes)


def plot_sv_short_long_reads():
    # run gmm to get evidence by mode
    # pass into plot_single_sv
    pass


def plot_sv_breakdown():
    """Figure 4 - horizontal bar charts of the breakdown of SVs by number of modes"""
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

    colors = ["#7BB662", "#FFD301", "#E03C32"]

    # Plot left figure: horizontal bar chart of the number of SVs
    fig, axs = plt.subplots(
        1, 2, figsize=(12, 3), gridspec_kw={"width_ratios": [1, 2]}
    )
    ax1 = axs[0]
    categories = [
        "Clustered\n(Enough Reads)",
        "Inconclusive\n(Too Few Reads)",
        "No Evidence",
    ]
    mode_labels = ["1", "2", "3"]
    ax1.barh(categories, n_svs[::-1], color="#274c77")

    # Make the clustered category a stacked bar chart based on the number of modes
    clustered_modes = [modes[0], modes[1], modes[2]]  # 1 SV, 2 SVs, 3 SVs
    bottom = 0
    for i, (value, label) in enumerate(zip(clustered_modes, mode_labels)):
        ax1.barh(
            categories[0],
            value,
            left=bottom,
            label=label,
            color=colors[i],
        )
        bottom += value
    ax1.legend(
        title="SVs per Breakpoint Cluster",
        loc="lower right",
        ncol=3,
    )
    for i, (n, af) in enumerate(zip(n_svs[::-1], afs[::-1])):
        ypos = i + 0.28 if i == 0 else i
        ax1.text(
            n + 500,
            ypos,
            f"AF = {af:.3f}",
            va="center",
            ha="left",
            fontsize=10,
        )
    ax1.set_xlim(0, max(n_svs) + 15000)

    # Plot right figure: horizontal bar chart of the number of SVs by number of modes
    ax2 = axs[1]
    bottom = np.zeros(len(modes))
    for i, conf in enumerate(["high", "medium", "low"]):
        values = [confidence[i].get(conf, 0) for i in range(len(modes))]
        ax2.barh(
            mode_labels,
            values,
            left=bottom,
            label=conf.capitalize(),
            color=colors[i],
        )
        bottom += values
    ax2.set_xscale("log", base=10)
    ax2.set_xticks([10, 100, 1000, 10000])
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    for i, n in enumerate(n_samples):
        pos_x = modes[i] if i == 0 else modes[i] + 100
        pos_y = i + 0.55 if i == 0 else i
        ha = "right" if i == 0 else "left"
        ax2.text(pos_x, pos_y, f"n = {n:.0f}", va="center", ha=ha, fontsize=10)
    ax2.set_ylabel("SVs per Breakpoint Cluster", fontsize=12)
    ax2.legend(title="Confidence", loc="upper right")

    fig.text(0.3, 0.02, "Number of SVs per Category", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(left=0.11, bottom=0.19)
    plt.savefig("plots/sv_breakdown.pdf")
    plt.show()


if __name__ == "__main__":
    synthetic_data_fig()
    synthetic_data_additional_svs()
    # methods_figure_viz(
    #     [(100000, 100802), (100060, 100741)],
    #     "synthetic_data/data/B_r0.8500000000000001_svlen802_n66_fa6914bc-f836-4f50-8a14-5cc0b56a50c9.vcf",
    # )
    # plot_sv_breakdown()
