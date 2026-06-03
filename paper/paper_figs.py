import ast
from collections import Counter, defaultdict
import os
import random
import re
import subprocess

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd
import pysam

from src.model.split import split_sv
from model.gmm_trial import (
    process_data,
    get_evidence_by_mode,
    gmm_trial,
)
from src.model.gmm import gmm
from src.utils.constants import (
    COLORS,
    SUPERPOPULATIONS,
    SUBPOPULATIONS,
    ANCESTRY_COLORS,
)
from src.utils.helper import (
    get_most_common_split_df,
    get_all_split_trials_df,
    get_sv_lookup,
    get_sv_chr,
    stix_output_to_df,
)
from src.utils.model_helper import (
    calc_af,
    calculate_posteriors_from_trials,
    calculate_ci,
)
from src.utils.viz import plot_2d_coords, plot_single_sv


def plot_sv_coordinate_space(ax, n_svs, xcoords, svcoords, y, height, **kwargs):
    """
    Draws a capsule-like SV box using an ellipse for the ends and a rectangle in the middle.
    x, y = bottom-left of the rectangle
    width, height = dimensions of the total capsule
    """
    facecolor = kwargs.get("facecolor", "lightgrey")
    edgecolor = kwargs.get("edgecolor", "black")
    rectcolor = kwargs.get("rectcolor", "#274c77")
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
        facecolor=rectcolor,
        edgecolor="none",
    )
    ax.add_patch(sv_rect)

    return [left_ellipse, right_ellipse, rect, top_border, bottom_border]


def load_synthetic_data_results(
    sample_size: int, *, path: str = "", add_gatk_results: bool = True
) -> pd.DataFrame:
    split_file = os.path.join(
        "output/synthetic_tests", path, f"resultsn={sample_size}.csv"
    )
    split_df = pd.read_csv(split_file)
    split_df = split_df[split_df["gmm_model"] == "split"]
    if not add_gatk_results:
        return split_df

    gatk_file = "output/synthetic_tests/results.csv"
    gatk_df = pd.read_csv(gatk_file)
    combined_df = pd.concat([gatk_df, split_df], ignore_index=True)
    return combined_df


def plot_reciprocal_overlap_svlen(
    case: str, sample_size: int, *, y_axis: str = "accuracy", path: str = ""
):
    df = load_synthetic_data_results(sample_size, path=path)
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


def parameter_sweep(case: str = "B", path: str = ""):
    sample_sizes = []
    for file in os.listdir(os.path.join("output/synthetic_tests", path)):
        if "resultsn=" in file:
            sample_size = int(file.strip("resultsn=").strip(".csv"))
            sample_sizes.append(sample_size)
    sample_sizes = sorted(sample_sizes)
    file = load_synthetic_data_results(
        sample_sizes[-1], path=path, add_gatk_results=False
    )
    svlens = sorted(file["svlen"].unique())

    tprs = np.zeros((len(sample_sizes), len(svlens)))
    fprs = np.zeros((len(sample_sizes), len(svlens)))
    ros = np.zeros((len(sample_sizes), len(svlens)))
    for i, sample_size in enumerate(sample_sizes):
        df = load_synthetic_data_results(
            sample_size, path=path, add_gatk_results=False
        )
        df = df[(df["gmm_model"] == "split") & (df["case"] == case)]
        for j, svlen in enumerate(svlens):
            subset_df = df[df["svlen"] == svlen]
            if subset_df.empty:
                print("missing data for", sample_size, svlen)
                continue
            right, total, false_positives = (
                defaultdict(int),
                defaultdict(int),
                defaultdict(int),
            )
            for _, row in subset_df.iterrows():
                overlap = row["r"]
                total[overlap] += 1
                if row["expected_num_modes"] == row["num_modes"]:
                    right[overlap] += 1
                if row["num_modes"] > row["expected_num_modes"]:
                    false_positives[overlap] += 1

            # find the overlap at which
            overlaps = np.array(sorted(total.keys()))
            acc = np.array([right[o] / total[o] for o in overlaps])
            fps = np.array([false_positives[o] / total[o] for o in overlaps])
            tprs[i, j] = np.mean(acc)
            fprs[i, j] = np.mean(fps)
            ro_idx = np.where(acc > 0.5)[0]
            if len(ro_idx) > 0:
                ros[i, j] = overlaps[ro_idx[-1]]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i, (values, label) in enumerate(
        zip([tprs, fprs, ros], ["Average TPR", "Average FPR", "r at TPR > 0.5"])
    ):
        ax = axs[i]
        im = ax.imshow(values.T, cmap="Blues", vmin=0, vmax=1)
        # put numbers in each cell
        for j in range(len(sample_sizes)):
            for k in range(len(svlens)):
                text = f"{values[j, k]:.2f}"
                ax.text(
                    j,
                    k,
                    text,
                    ha="center",
                    va="center",
                    color="black" if values[j, k] < 0.5 else "white",
                    fontsize=10,
                )
        ax.set_xticks(np.arange(len(sample_sizes)))
        ax.set_yticks(np.arange(len(svlens)))
        ax.set_xticklabels(sample_sizes)
        ax.set_yticklabels(svlens)
        ax.set_xlabel("Sample Size", fontsize=14)
        ax.set_ylabel("SV Length", fontsize=14)
        ax.set_title(label, fontsize=16)
        fig.colorbar(im, ax=axs[i])

    plt.tight_layout(w_pad=0)
    plt.savefig(
        f"output/plots/parameter_sweep_heatmaps_case{case}{'_'if path else ''}{path}.pdf"
    )
    plt.show()


def synthetic_data_fig(sample_size: int, svlen: int, path: str = ""):
    """Synthetic data results comparison: my method vs. GATK clustering methods"""
    cases = ["B", "C", "D"]
    df = load_synthetic_data_results(sample_size, path=path)
    df = df[df["svlen"] == svlen]

    models = ["split", "gatk_MAX_CLIQUE", "gatk_SINGLE_LINKAGE"]
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
                    false_positives[overlap] += 1

            overlaps = np.array(sorted(total.keys()))
            acc = np.array([right[o] / total[o] for o in overlaps])
            fps = np.array([false_positives[o] / total[o] for o in overlaps])
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
    plt.savefig(
        f"output/plots/synthetic_data_n={sample_size}_len={svlen}{'_'if path else ''}{path}.pdf"
    )
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
    plt.savefig("output/plots/synthetic_data_additional_svs.pdf")
    plt.show()


def methods_raw_reads(ax, svs, reads):
    """Figure 2a - Visualizes the reads in coordinate space."""
    sv_avg = (np.mean([sv[0] for sv in svs]), np.mean([sv[1] for sv in svs]))

    # plot reads
    y_offset = 0.2
    n_reads_shown = 30
    for i, read in enumerate(reads[-n_reads_shown:]):
        color = COLORS[2] if i == n_reads_shown - 1 else "#274c77"
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

    reads_min = min([read[0] for read in reads[-n_reads_shown:]])
    reads_max = max([read[1] for read in reads[-n_reads_shown:]])
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
    last_read = reads[-1]
    ax.plot(
        [last_read[0], last_read[0]],
        [y_offset - 0.05, 0.1],
        linestyle="dotted",
        color=COLORS[2],
        linewidth=2,
    )
    ax.plot(
        [last_read[1], last_read[1]],
        [y_offset - 0.05, 0.1],
        linestyle="dotted",
        color=COLORS[2],
        linewidth=2,
    )

    # label the L/R of the read as L_i and R_i
    ax.text(
        last_read[0],
        -0.05,
        "Lᵢ",
        ha="center",
        va="center",
        color=COLORS[2],
        fontsize=14,
    )
    ax.text(
        last_read[1],
        -0.05,
        "Rᵢ",
        ha="center",
        va="center",
        color=COLORS[2],
        fontsize=14,
    )
    ax.axis("off")


def methods_lr(ax, reads_plotted):
    """Figure 2b - Visualizes the reads in length vs. read L position space."""
    read_i = reads_plotted[-1]
    for read in reads_plotted:
        read_L = read[0]
        read_R = read[1]
        if read_L == read_i[0] and read_R == read_i[1]:
            ax.scatter(read_L, read_R, color=COLORS[2], s=30)
            ax.text(
                read_L + 20,
                read_R - 20,
                "(Lᵢ, Rᵢ)",
                ha="center",
                va="center",
                color=COLORS[2],
                fontsize=14,
            )
        else:
            ax.scatter(read_L, read_R, color="#274c77", s=10)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("L-coordinate", fontsize=16)
    ax.set_ylabel("R-coordinate", fontsize=16)


def write_fake_stix_data(reads: dict[str, list[int]]):
    reads_df = stix_output_to_df("", write_empty_file=True)
    for sample_id, read_list in reads.items():
        for l, r in zip(read_list[::2], read_list[1::2]):  # noqaE741
            reads_df.loc[len(reads_df)] = [
                0,
                sample_id,
                1,
                l,
                l,
                1,
                r,
                r,
                "paired",
            ]
    return reads_df


def methods_clustered(ax, svs, reads, insert_size_lookup, num_modes):
    """Figure 2c - Visualizes the clustered reads in length vs. read L position space."""

    sv_avg = (np.mean([sv[0] for sv in svs]), np.mean([sv[1] for sv in svs]))
    reads_df = write_fake_stix_data(reads)

    L, R = sv_avg[0], sv_avg[1]
    points, sv_evidence = process_data(
        reads_df,
        file_name="",
        L=L,
        R=R,
        insert_size_lookup=insert_size_lookup,
        min_pairs=1,
    )
    gmm_result = gmm(points, L=L, R=R, force_n_modes=num_modes)
    evidence = get_evidence_by_mode(
        gmm_result,
        sv_evidence,
        L,
        R,
    )

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
        insert_size_lookup=insert_size_lookup,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("L-coordinate", fontsize=16)
    ax.set_ylabel("R-coordinate", fontsize=16)


def methods_svs(ax, svs):
    """Figure 2d - Visualizes the resulting SVs in coordinate space."""
    x_min = min([sv[0] for sv in svs]) - 100
    x_max = max([sv[1] for sv in svs]) + 100
    rect_height = 1.5
    ax.add_patch(
        Rectangle(
            (x_min, 0),
            x_max - x_min,
            rect_height,
            facecolor="white",
            edgecolor="none",
        )
    )

    y_offset = rect_height
    for i, (L, R) in enumerate(svs):
        plot_sv_coordinate_space(
            ax,
            len(svs),
            (x_min, x_max),
            (L, R),
            y_offset,
            0.2,
            edgecolor="black",
            facecolor="lightgrey",
            rectcolor=COLORS[i],
        )
        y_offset += 0.25
    ax.axis("off")


def methods_figure_viz(svs, vcf_file: str):
    """
    Figure 2 - Generates all of the sub-figures for the methods visualization figure.
    vcf_file is a vcf generated by the generate_synthetic_sv_data function in generate_data.py.
    """
    # parse vcf
    reads = defaultdict(list)  # all reads
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

    # pick only a subset of reads to plot to show more variation in reads
    # shuffle the samples first
    sample_ids = list(reads.keys())
    random.shuffle(sample_ids)
    reads_to_plot = {
        sample_id: random.sample(reads[sample_id], 1)[0]
        for sample_id in sample_ids
    }
    reads_list = list(reads_to_plot.values())

    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    methods_raw_reads(axes[0], svs, reads_list)
    methods_lr(axes[1], reads_list)
    axes[2].axis("off")  # placeholder for clustered reads
    methods_svs(axes[3], svs)

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.savefig("output/plots/methods.png")
    plt.savefig("output/plots/methods.pdf")
    plt.show()

    # Plot the clustered reads separately so I can stack them in post-processing
    for num_modes in range(1, 4):
        fig, ax = plt.subplots(figsize=(4, 3))
        methods_clustered(ax, svs, reads_to_plot, mean_insert_sizes, num_modes)
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        plt.savefig(f"plots/methods_clustered_{num_modes}modes.png")
        plt.savefig(f"plots/methods_clustered_{num_modes}modes.pdf")
        plt.show()


def plot_sv_short_long_reads(sv_id, sample_ids, skip_evidence_plot=False):
    chr, start, stop = get_sv_chr(sv_id)

    if not skip_evidence_plot:
        reads = split_sv(sv_id=sv_id, input_dir="1kg", run_split=False)
        gmm_result, evidence_by_mode = gmm_trial(
            reads,
            chr=chr,
            L=start,
            R=stop,
            plot=False,
        )
        plot_single_sv(
            evidence_by_mode,
            L=start,
            R=stop,
            axis1="L",
            axis2="Length",
            size_by="",
            add_right_padding=True,
        )

    if len(sample_ids) == 0:
        return

    if not os.path.exists(f"long_reads/bam_files/{sv_id}"):
        # copy files from fiji
        subprocess.run(
            [
                "scp",
                "-r",
                f"vili4418@fiji.colorado.edu:/Users/vili4418/sv/sv_gmm/long_reads/bam_files/{sv_id}",
                "long_reads/bam_files",
            ],
            capture_output=True,
            text=True,
        )

    os.mkdir(f"long_reads/bam_files_subset/{sv_id}")
    for sample_id in sample_ids:
        subprocess.run(
            [
                "cp",
                f"long_reads/bam_files/{sv_id}/{sample_id}.bam",
                f"long_reads/bam_files_subset/{sv_id}/{sample_id}.bam",
            ],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "cp",
                f"long_reads/bam_files/{sv_id}/{sample_id}.bam.bai",
                f"long_reads/bam_files_subset/{sv_id}/{sample_id}.bam.bai",
            ],
            capture_output=True,
            text=True,
        )

    # the samplot image is generated separately and needs to be added in post-processing
    subprocess.run(
        ["bash", "bash/samplot_paper_viz.sh"]
        + [sv_id, str(chr), str(start), str(stop)],
        capture_output=True,
        text=True,
    )

    subprocess.run(
        ["rm", "-r", f"long_reads/bam_files_subset/{sv_id}"],
        capture_output=True,
        text=True,
    )


def find_example_svs():
    df = pd.read_csv("output/results/svs_n_modes.csv")
    df.rename(columns={"sv_id": "id"}, inplace=True)
    ancestry = pd.read_csv("output/results/ancestry_dissimilarity.csv")
    df = df.merge(ancestry, on="id")
    df = df.sort_values(by="dissimilarity", ascending=False)
    df = df[
        (df["confidence"] == "high")
        & (df["num_modes"] == 3)
        & (df["dissimilarity"] >= 0.5)
    ]
    collapsed = pd.read_csv("output/results/most_common_split.csv")
    for _, row in df.iterrows():
        sv_id = row["id"]
        # check if there are bam files for samples in each mode
        modes = ast.literal_eval(
            collapsed[collapsed["id"] == sv_id]["modes"].values[0]
        )
        can_plot = [False for _ in range(len(modes))]
        for i, mode in enumerate(modes):
            for sample_id in mode["sample_ids"]:
                bam_path = f"long_reads/bam_files/{sv_id}/{sample_id}.bam"
                if os.path.exists(bam_path):
                    can_plot[i] = True
                    break
        if all(can_plot):
            print("plotting", sv_id)
            plot_sv_short_long_reads(sv_id, [])
        else:
            print("skipping", sv_id)


def get_all_svs_df(path: str = "") -> pd.DataFrame:
    filepath = os.path.join("output/results", path)
    full_df = get_most_common_split_df(filepath)
    full_df["num_samples_run"] = full_df.apply(
        lambda row: int(
            np.sum(
                [
                    len(mode["sample_ids"])
                    for mode in ast.literal_eval(row["modes"])
                ],
            )
        ),
        axis=1,
    )
    full_df.rename(columns={"id": "sv_id"}, inplace=True)
    full_df = full_df[
        [
            "chr",
            "sv_id",
            "af",
            "svlen",
            "num_samples_run",
            "modes",
        ]
    ]
    df = pd.read_csv(os.path.join(filepath, "svs_n_modes.csv"))
    df = df.merge(full_df, on="sv_id")
    return df


def plot_sv_breakdown(path: str = ""):
    """Figure 4 - horizontal bar charts of the breakdown of SVs by number of modes"""
    df = get_all_svs_df(path)
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
    svs_not_split = pd.concat(
        [svs_no_evidence, svs_not_enough_evidence], ignore_index=True
    )

    n_svs = []
    afs = []
    for df_subset in [svs_not_split, svs_clustered]:
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
    # ax2.set_xscale("log", base=10)
    # ax2.set_xticks([10, 100, 1000, 10000])
    # ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
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
    plt.savefig("output/plots/sv_breakdown.pdf")
    plt.show()


def plot_af_delta_histogram():
    """Figure 5 - Histogram of allele frequency changes after splitting SVs."""
    sv_df = get_most_common_split_df()
    sv_df = sv_df[sv_df["num_samples"] > 10]
    original_afs = sv_df[sv_df["num_modes"] == 1]["af"].values
    sv_df = sv_df[sv_df["num_modes"].isin([2, 3])]
    original_afs_split = sv_df["af"].values

    # calculate delta ratios
    delta_ratios = []
    for _, row in sv_df.iterrows():
        n_homozygous = 0
        n_heterozygous = 0
        modes = ast.literal_eval(row["modes"])
        mode_afs = []

        for mode in modes:
            n_homozygous += mode["num_homozygous"]
            n_heterozygous += mode["num_heterozygous"]
            af = calc_af(mode["num_homozygous"], mode["num_heterozygous"], 2504)
            mode_afs.append(af)

        original_af = calc_af(n_homozygous, n_heterozygous, 2504)
        for af in mode_afs:
            if original_af > 0:
                delta = af / original_af
                delta_ratios.append(delta)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(delta_ratios, bins=20, color="black", alpha=0.8, edgecolor="white")
    ax.set_xlim(-0.005, 1.015)

    ax_box = ax.inset_axes([0.4, 0.65, 0.4, 0.3])
    ax_box.boxplot(
        [original_afs, original_afs_split],
        positions=[1, 2],
        widths=0.4,
        patch_artist=True,
        boxprops=dict(facecolor="lightgrey", color="black"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", color="black", markersize=5),
    )
    ax_box.set_xticks([1, 2])
    ax_box.set_xticklabels(
        ["Unsplit SVs", "Split SVs"],
        fontsize=12,
    )
    ax_box.set_ylabel("Original Allele\nFrequency", fontsize=12)
    ax_box.set_ylim(0, 1.0)
    ax_box.tick_params(axis="y", labelsize=12)
    ax_box.yaxis.set_major_locator(FixedLocator(np.arange(0, 1.01, 0.2)))

    ax.set_xlabel("Allele Frequency Ratio, New/Original", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    # ax.xaxis.set_major_locator(FixedLocator(np.arange(0, 1.1, 0.2)))
    # ax.yaxis.set_major_locator(FixedLocator(np.arange(0, 181, 20)))
    # ax.xaxis.set_minor_locator(FixedLocator(np.arange(0, 1.1, 0.1)))
    # ax.yaxis.set_minor_locator(FixedLocator(np.arange(0, 181, 10)))
    # ax.tick_params(axis="x", which="minor", length=4, labelbottom=False)
    # ax.tick_params(axis="y", which="minor", length=4, labelleft=False)
    plt.tight_layout()
    plt.savefig("output/plots/af_delta_histogram.pdf")
    plt.show()


def compare_sv_ancestry_by_mode(by: str = "superpopulation"):
    ancestry_df = pd.read_csv("data/1kg/ancestry.tsv", delimiter="\t")

    # create a lookup of sample_id:population
    # if multiple populations are listed for a sample, take the first one (only happens for 1 sample)
    sample_lookup = {
        row["Sample name"]: row["Population code"].split(",")[0]
        for _, row in ancestry_df.iterrows()
    }
    population_lookup = {}
    for i, row in ancestry_df.iterrows():
        population_lookup[row["Population code"]] = row["Superpopulation code"]

    sv_df = get_most_common_split_df()
    sv_df = sv_df[sv_df["num_modes"] > 1]

    populations = (
        SUPERPOPULATIONS if by == "superpopulation" else SUBPOPULATIONS
    )[::-1]

    # all instances of two populations co-occurring so we can normalize later
    all_comparisons = [
        np.zeros(len(populations)) for _ in range(len(populations))
    ]
    # actual co-occurrences in same mode after splitting
    comparisons = [np.zeros(len(populations)) for _ in range(len(populations))]
    # instances of two populations occurring in different modes
    split_comparisons = [
        np.zeros(len(populations)) for _ in range(len(populations))
    ]

    # check pairwise population comparisons for each SV
    for _, row in sv_df.iterrows():
        modes = ast.literal_eval(row["modes"])
        mode_by_samples = {}
        for mode_index, mode in enumerate(modes):
            for sample in mode["sample_ids"]:
                mode_by_samples[sample] = mode_index
        for s1, m1 in mode_by_samples.items():
            p1 = populations.index(sample_lookup[s1])
            for s2, m2 in mode_by_samples.items():
                if s1 == s2:
                    continue
                p2 = populations.index(sample_lookup[s2])
                all_comparisons[p1][p2] += 1
                if m1 == m2:  # in the same mode
                    comparisons[p1][p2] += 1
                elif m1 != m2:  # in different modes
                    split_comparisons[p1][p2] += 1

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for i, (values, cmap) in enumerate(
        zip([comparisons, split_comparisons], ["Blues", "Reds"])
    ):
        values_normalized = np.array(values, dtype=float)
        # normalize comparisons by all_comparisons
        for pop_i in range(len(populations)):
            for pop_j in range(len(populations)):
                if all_comparisons[pop_i][pop_j] == 0:
                    values_normalized[pop_i][pop_j] = 0
                else:
                    values_normalized[pop_i][pop_j] /= all_comparisons[pop_i][
                        pop_j
                    ]

        ax = axs[i]
        im = ax.imshow(values_normalized, cmap=cmap, interpolation="nearest")
        ax.set_yticks(range(len(populations)))
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels(populations, fontsize=12)
        ax.text(7, -3.25, "Superpopulation", fontsize=14)
        ax.set_ylabel("Population" if by == "population" else "", fontsize=14)
        ax.set_ylim(len(populations) - 0.5, -2.5)
        if by == "population":
            superpop_seen = set()
            for i, label in enumerate(populations):
                superpop = population_lookup[label]
                if superpop not in superpop_seen:
                    ax.text(
                        i + 0.03,
                        -1,
                        superpop,
                        color="whitesmoke",
                        fontsize=14,
                        zorder=2,
                    )
                    superpop_seen.add(superpop)

                    if i != 0:
                        ax.axhline(
                            y=i - 0.5,
                            color="black",
                            linestyle="--",
                            linewidth=0.8,
                            zorder=10,
                        )
                        ax.axvline(
                            x=i - 0.5,
                            color="black",
                            linestyle="--",
                            linewidth=0.8,
                            zorder=10,
                        )

                rect = Rectangle(
                    (i - 0.5, -2.5),
                    1,
                    2,
                    linewidth=0,
                    edgecolor="none",
                    facecolor=ANCESTRY_COLORS[superpop],
                    alpha=0.9,
                    zorder=1,
                )
                ax.add_patch(rect)

        for spine in ax.spines.values():
            spine.set_visible(False)

        rect_border = Rectangle(
            (-0.5, -0.5),
            len(populations),
            len(populations),
            linewidth=1,
            edgecolor="black",
            facecolor="none",
            zorder=3,
        )
        ax.add_patch(rect_border)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.1)
        cbar.set_label("Clustering Likelihood", fontsize=12, labelpad=10)
        cbar.ax.tick_params(labelsize=14)
        ax.tick_params(bottom=False, top=False, labelbottom=False)
        ax.text(
            0.5,
            -0.1,
            "Spacing",
            fontsize=14,
            color="white",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    plt.tight_layout()
    plt.savefig("output/plots/ancestry_comparison.pdf", bbox_inches="tight")
    plt.show()


def print_sv_stats(path: str = ""):
    """Compares various statistics between split SVs and all SVs and plots boxplots."""
    df = get_all_svs_df(path)

    # load breakpoint precision info
    bp_df = pd.read_csv("1kg/cipos.csv")
    bp_df.rename(columns={"id": "sv_id"}, inplace=True)
    bp_df["bp_precision"] = bp_df["ciend"].apply(
        lambda x: eval(x)[1] - eval(x)[0]
    ) + bp_df["cipos"].apply(lambda x: eval(x)[1] - eval(x)[0])
    df = df.merge(bp_df[["sv_id", "bp_precision"]], on="sv_id", how="left")

    # get only SVs that were actually clustered
    df = df[df["num_samples_run"] > 10]
    split = df[df["num_modes"] > 1]

    stats = {}
    for field in ["af", "svlen", "num_samples_run", "bp_precision"]:
        # filter by non-nan values
        filtered_df = df[~df[field].isna()]
        filtered_split = split[~split[field].isna()]
        stats[field] = {
            "all": filtered_df[field],
            "split": filtered_split[field],
        }

    # print what % of SVs were split per chromosome
    split_counts = split["chr"].value_counts().sort_index()
    all_counts = df["chr"].value_counts().sort_index()
    chr_labels = all_counts.index.tolist()
    split_percents = [
        split_counts.get(chr, 0) / all_counts[chr] for chr in chr_labels
    ]
    print("% SVs split per chr", pd.DataFrame(split_percents).describe(), "\n")

    # print reciprocal overlap stats for split SVs
    reciprocal_overlaps = []
    for _, row in split.iterrows():
        modes = ast.literal_eval(row["modes"])
        sv_coords = [(mode["start"], mode["end"]) for mode in modes]
        for i in range(len(sv_coords)):
            for j in range(i + 1, len(sv_coords)):
                L1, R1 = sv_coords[i]
                L2, R2 = sv_coords[j]
                overlap = max(0, min(R1, R2) - max(L1, L2))
                union = max(R1, R2) - min(L1, L2)
                ro = overlap / union if union > 0 else 0
                reciprocal_overlaps.append(ro)
    stats["reciprocal_overlap"] = {
        "all": np.nan,  # Not applicable for "all" since it's only for split SVs
        "split": pd.Series(reciprocal_overlaps),
    }

    # plot side by side boxplots
    fields = [
        "af",
        "svlen",
        "num_samples_run",
        "bp_precision",
        "reciprocal_overlap",
    ]
    labels = [
        "Allele Frequency",
        "SV Length",
        "# Samples Run",
        "BP Precision",
        "Reciprocal Overlap",
    ]

    data_all = [stats[field]["all"] for field in fields]
    data_split = [stats[field]["split"] for field in fields]

    fig, axs = plt.subplots(1, len(fields), figsize=(20, 5), sharey=False)
    for i, ax in enumerate(axs):
        ax.boxplot(
            [data_all[i], data_split[i]],
            positions=[1, 2],
            widths=0.6,
            patch_artist=True,
        )
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["All SVs", "Split SVs"], fontsize=10)
        ax.set_title(labels[i], fontsize=12)

    fig.suptitle("Comparison of Pre-Split and Post-Split SV Stats", fontsize=16)
    plt.tight_layout()
    plt.savefig("output/plots/sv_stats_comparison_boxplots.pdf")
    plt.show()


def plot_num_trials(path: str = ""):
    """Plots the number of trials per confidence level."""
    # all low and medium confidence SVs ran all 100 trials
    # most high confidence SVs ran 8 trials
    # 4% of high confidence SVs ran >8 trials, with a mean of 25 and median of 18
    # 62 high confidence SVs ran >50 trials (0.3% of total SVs run)
    df = get_all_svs_df(path)
    df = df[df["confidence"] != "inconclusive"]

    confidence_levels = ["low", "medium", "high"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, conf in enumerate(confidence_levels):
        ax = axs[i]
        conf_df = df[df["confidence"] == conf]
        ax.hist(
            conf_df["num_gmm_runs"],
            bins=20,
            color="black",
            alpha=0.8,
            edgecolor="white",
        )
        ax.set_title(f"Confidence Level: {conf.capitalize()}", fontsize=14)
        ax.set_xlabel("Number of Trials", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    plt.show()

    converge = get_all_split_trials_df("results")
    # get all rows where num_gmm_runs > 50
    subset = df[(df["confidence"] == "high") & (df["num_gmm_runs"] > 50)].copy()
    for _, row in subset.iterrows():
        sv_id = row["sv_id"]
        all_rows = converge[converge["id"] == sv_id]
        first_50 = all_rows.head(50)
        random_50 = all_rows.sample(50, random_state=42)
        for label, rows in zip(
            ["first_50", "random_50"], [first_50, random_50]
        ):

            outcomes = rows["num_modes"].values
            counter = Counter(outcomes)
            most_common = counter.most_common(2)
            num_modes = max(1, most_common[0][0])
            p, var = calculate_posteriors_from_trials(outcomes)
            ci = calculate_ci(p, var, len(outcomes))
            confidence = "low"
            if ci[0] >= 0.6:
                confidence = "high"
            elif ci[0] >= 0.3:
                confidence = "medium"

            subset.loc[subset["sv_id"] == sv_id, f"num_modes_{label}"] = (
                num_modes
            )
            subset.loc[subset["sv_id"] == sv_id, f"confidence_{label}"] = (
                confidence
            )

    print("Reducing the max number of trials from 100 to 50 changes:")
    for label in ["first_50", "random_50"]:
        n_changed = np.sum(subset["num_modes"] != subset[f"num_modes_{label}"])
        print(
            f"  - Number of SVs with different num_modes ({label}): {n_changed} out of {subset.shape[0]}"
        )
        n_conf_changed = np.sum(
            subset["confidence"] != subset[f"confidence_{label}"]
        )
        print(
            f"  - Number of SVs with different confidence ({label}): {n_conf_changed} out of {subset.shape[0]}"
        )


def download_l_len_plots():
    df = pd.read_csv(
        "calibration/results/d300_r0.50_q0.80_p200/svs_n_modes.csv"
    )
    df = df[df["confidence"] == "high"]
    for n_modes in list(range(1, 4)):
        subset = df[df["num_modes"] == n_modes]
        out_dir = (
            f"calibration/results/d300_r0.50_q0.80_p200/plots/{n_modes}modes"
        )
        if not os.path.exists:
            os.mkdir(out_dir)
        for _, row in subset.sample(n=10).iterrows():
            l, r = row["sv_id"].split("_")
            split_sv(
                l=l,
                r=r,
                input_dir="calibration",
                output_dir=out_dir,
                sv_lookup_file="filtered_dels.csv",
                single_trial=False,
                plot=True,
            )


def bayesian_optimization_iterations(file: str):
    """
    Plot the F1 score and parameter values at each iteration of the
    Bayesian optimization process to show how the optimization converges
    over time.
    TODO: update parameter names
    """
    # parse eofile for each iteration
    param_names = [
        "distance_at_0_penalty",
        "overlap_at_0_penalty",
        "query_overlap",
        "max_penalty_value",
    ]
    param_values = [[] for _ in range(4)]  # d, r, q, p
    f1_scores = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            # match the output line for parameter values
            match = re.match(
                r"Running calibration for d=(\d+), r=([\d.]+), q=([\d.]+), p=(\d+)",
                line,
            )
            if match:
                d = int(match.group(1))
                param_values[0].append(d)

                r = float(match.group(2))
                param_values[1].append(r)

                q = float(match.group(3))
                param_values[2].append(q)

                p = int(match.group(4))
                param_values[3].append(p)

            # match the output line for F1 score: Trial 23 F1 score = 0.4281
            match_f1 = re.match(
                r"Trial \d+ F1 score = ([\d.]+)",
                line,
            )
            if match_f1:
                f1_score = float(match_f1.group(1))
                f1_scores.append(f1_score)
    f1_scores = np.array(f1_scores)
    iterations = list(range(1, len(f1_scores) + 1))
    best_so_far = np.maximum.accumulate(f1_scores)
    best_idx = np.argmax(f1_scores)

    # plot F1 score and parameter values over iterations
    fig = plt.figure(figsize=(13, 7))
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.6, 1],
        width_ratios=[1, 0.03],
        hspace=0.12,
        wspace=0.04,
        left=0.13,
        right=0.94,
        top=0.95,
        bottom=0.07,
    )

    # normalize parameter values
    params_raw = np.array(param_values).T  # shape (N_ITER, 4)
    params_norm = (params_raw - params_raw.min(axis=0)) / (
        params_raw.max(axis=0) - params_raw.min(axis=0)
    )

    ax_f1 = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[:, 1])

    # top plot of F1 scores
    # f1 score at each iteration
    ax_f1.scatter(iterations, f1_scores, color="mediumslateblue", zorder=2)

    # best so far line
    ax_f1.plot(iterations, best_so_far, color="steelblue")
    ax_f1.fill_between(iterations, best_so_far, alpha=0.10, color="steelblue")

    # mark the global best
    ax_f1.scatter([best_idx + 1], [f1_scores[best_idx]], color="gold", zorder=3)

    ax_f1.set_xlim(0.5, len(iterations) + 0.5)
    ax_f1.set_ylim(
        max(0, f1_scores.min() - 0.06), min(1, f1_scores.max() + 0.06)
    )
    ax_f1.set_ylabel("F1 Score")

    # bottom plot of parameter values as a heatmap
    ax_heat.imshow(
        params_norm.T,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        extent=[0.5, len(iterations) + 0.5, -0.5, len(param_names) - 0.5],
        origin="lower",
    )
    ax_heat.axvline(best_idx + 1, color="gold", zorder=3)
    ax_heat.set_xlim(0.5, len(iterations) + 0.5)
    ax_heat.set_ylim(-0.5, len(param_names) - 0.5)
    ax_heat.set_yticks(range(len(param_names)))
    ax_heat.set_yticklabels(param_names)
    ax_heat.set_xlabel("Iteration")
    ax_heat.xaxis.set_major_locator(ticker.MultipleLocator(5))
    for y in np.arange(0.5, len(param_names) - 0.5, 1):
        ax_heat.axhline(y)

    # heatmap colorbar
    norm = Normalize(vmin=0, vmax=1)
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap="viridis"),
        cax=ax_cbar,
        orientation="vertical",
    )
    cb.set_label("Normalized\nparameter value")

    plt.savefig("output/plots/bo.pdf")
    plt.show()


if __name__ == "__main__":
    figures = [5, 6, 7]

    # Figure 1
    if 1 in figures:
        methods_figure_viz(
            [(100000, 100802), (100060, 100741)],
            "synthetic_data/data/B_r0.8500000000000001_svlen802_n66_fa6914bc-f836-4f50-8a14-5cc0b56a50c9.vcf",
        )

    # Figure 2
    if 2 in figures:
        for case in ["B", "C", "D"]:
            parameter_sweep(case, "")
        # for n_samples in [11, 25, 50, 100, 200, 400]:
        #     for svlen in [50, 100, 200, 400, 800, 1600, 10000]:
        #         synthetic_data_fig(n_samples, svlen, "")
        # synthetic_data_additional_svs()

    # Figure 3
    if 3 in figures:
        plot_sv_breakdown()

    # Figure 4
    if 4 in figures:
        plot_sv_short_long_reads("HGSV_776", ["HG00096"])
        plot_sv_short_long_reads("HGSV_54541", ["HG03548", "HG00149"])
        plot_sv_short_long_reads(
            "HGSV_1289",
            ["HG00537", "NA19383", "NA19350"],
            skip_evidence_plot=True,
        )

    # Figure 5
    if 5 in figures:
        plot_af_delta_histogram()
        compare_sv_ancestry_by_mode(by="population")

    # Print SV stats
    if 6 in figures:
        print_sv_stats()

    # SI figure - confidence vs. number of trials
    if 7 in figures:
        plot_num_trials()

    if 8 in figures:
        bayesian_optimization_iterations("calibration/results/bo.out")
