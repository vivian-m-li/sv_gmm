import os
import ast
import random
import colorsys
import gzip
import math
import re
import numpy as np
import pandas as pd
from scipy.stats import norm, sem
from scipy.special import logit
from Bio import SeqIO, Seq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from brokenaxes import brokenaxes
from matplotlib.ticker import FixedLocator, StrMethodFormatter
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from collections import Counter, defaultdict
from typing import List
from em import run_em
from em_1d import run_em as run_em1d, get_scatter_data
from helper import (
    get_sample_sequencing_centers,
    get_sv_stats_df,
    get_sv_stats_collapsed_df,
    get_svlen,
    calc_af,
)
from gmm_types import (
    EstimatedGMM,
    Evidence,
    SVStat,
    Sample,
    ANCESTRY_COLORS,
    COLORS,
    GMM_AXES,
    GMM_MODELS,
    MODEL_NAMES,
    SUPERPOPULATIONS,
    SUBPOPULATIONS,
    SYNTHETIC_DATA_CENTROIDS,
)

REFERENCE_FILE = "hs37d5.fa.gz"
SAMPLES_DIR = "samples"


def random_color():
    return (random.random(), random.random(), random.random())


def add_noise(value, scale=0.07):
    return value + np.random.normal(scale=scale)


def get_evidence_by_mode(
    gmm: EstimatedGMM,
    sv_evidence: List[Evidence],
    L: int,
    R: int,
    *,
    gmm_model: str = "2d",
) -> List[List[Evidence]]:
    sv_evidence = np.array(sv_evidence)
    data = []
    for mode in gmm.x_by_mode:
        data_by_mode = []
        for x in mode:
            if gmm_model == "1d_len":
                data_by_mode.append((x + R))  # length
            elif gmm_model == "1d_L":
                data_by_mode.append((x + L))  # L-coordinate
            else:
                data_by_mode.append(
                    (x[0] + R, x[1] + L)
                )  # (length, L-coordinate)
        data.append(data_by_mode)
    evidence_by_mode = [[] for _ in range(len(data))]
    for evidence in sv_evidence:
        for i, mode in enumerate(data):
            try:
                if gmm_model == "1d_len":
                    mode_data = evidence.start_y  # length
                elif gmm_model == "1d_L":
                    mode_data = evidence.mean_l  # L-coordinate
                else:
                    mode_data = (
                        evidence.start_y,
                        evidence.mean_l,
                    )  # (length, L-coordinate)
                if (
                    mode_data in mode
                ):  # assumes that each mode has unique (length, L-coordinate) pairs
                    evidence_by_mode[i].append(evidence)
                    continue
            except ValueError:
                print(evidence)
                print(mode)
                raise ValueError
    lengths_by_mode = [
        np.mean([evidence.start_y for evidence in mode])
        for mode in evidence_by_mode
    ]
    try:
        evidence_by_mode = [
            x for _, x in sorted(zip(lengths_by_mode, evidence_by_mode))
        ]
    except Exception:
        print(L, R, lengths_by_mode, evidence_by_mode)
    return evidence_by_mode


def get_mean_std(label: str, values: List[float]):
    return (
        f"{label}={math.floor(np.mean(values))} +/- {round(np.std(values), 2)}"
    )


def print_sv_stats(sv_stats: List[List[SVStat]]):
    stats = []
    for i, lst in enumerate(sv_stats):
        lengths = [sv.length for sv in lst]
        starts = [sv.start for sv in lst]
        ends = [sv.end for sv in lst]
        stats.append(
            f"Mode {i + 1}\n{get_mean_std('Length', lengths)}\nMin, Max=[{math.floor(min(lengths))}, {math.floor(max(lengths))}]\n{get_mean_std('Start', starts)}\n{get_mean_std('End', ends)}\n"
        )
    return stats


def add_color_noise(hex_color: str):
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    hls = colorsys.rgb_to_hls(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    hue, lightness, saturation = hls
    new_saturation = saturation + random.uniform(-0.3, 0.3)
    new_saturation = min(max(new_saturation, 0), 1)
    new_rgb = colorsys.hls_to_rgb(hue, lightness, new_saturation)
    new_rgb = tuple(int(c * 255) for c in new_rgb)
    return "#{:02x}{:02x}{:02x}".format(*new_rgb)


def plot_evidence_by_mode(
    fig,
    gs,
    evidence_by_mode: List[List[Evidence]],
):
    num_modes = len(evidence_by_mode)
    mode_indices = list(range(num_modes))
    mode_indices_reversed = mode_indices[::-1]

    # Loop through data
    max_max_l = -np.inf
    min_min_r = np.inf
    all_mode_paired_ends = []
    population_data = []
    for i, mode in enumerate(reversed(evidence_by_mode)):
        all_paired_ends = []
        population_counter = Counter()

        # plots all evidence for each sample
        for evidence in mode:
            max_l = max([paired_end[0] for paired_end in evidence.paired_ends])
            min_r = min([paired_end[1] for paired_end in evidence.paired_ends])
            all_paired_ends.extend([max_l, min_r])
            max_max_l = max(max_l, max_max_l)
            min_min_r = min(min_r, min_min_r)
            all_mode_paired_ends.extend(evidence.paired_ends)
            population_counter[evidence.sample.superpopulation] += 1

        population_data.append(population_counter)

    all_mode_paired_end_flat = [
        p for paired_end in all_mode_paired_ends for p in paired_end
    ]

    # Plot the ancestry bar chart
    left_ax = fig.add_subplot(gs[0])
    bar_offset = 1 / len(ANCESTRY_COLORS) / 2
    for i, counter in enumerate(population_data):
        total = sum(counter.values())
        for j, (label, color) in enumerate(ANCESTRY_COLORS.items()):
            value = counter.get(label, 0) / total
            left_ax.barh(
                i + j * bar_offset,
                value,
                color=color,
                align="center",
                height=bar_offset,
            )
            if value > 0:
                left_ax.text(
                    value + 0.01,
                    i + j * bar_offset,
                    f"{value:.2f}",
                    va="center",
                    fontsize=10,
                    color="black",
                )

    left_ax.set_xlabel("Proportion", labelpad=18, fontsize=12)
    left_ax.yaxis.set_ticks([])
    left_ax.yaxis.set_ticklabels([])
    for spine_name, spine in left_ax.spines.items():
        if spine_name != "bottom":
            spine.set_visible(False)

    # Set up axes
    left_half = max_max_l - min(all_mode_paired_end_flat)
    right_half = max(all_mode_paired_end_flat) - min_min_r
    x_distance = max(left_half, right_half)
    bax = brokenaxes(
        xlims=(
            (max_max_l - x_distance - 20, max_max_l + 50),
            (min_min_r - 50, min_min_r + x_distance + 20),
        ),
        hspace=0.05,
        wspace=0.2,
        subplot_spec=gs[0, 1],
    )
    [
        x.remove() for x in bax.diag_handles
    ]  # remove diagonal lines since I can't get the positioning correct

    # SV plot
    min_y = np.inf
    all_sample_means = []
    for i, mode in enumerate(reversed(evidence_by_mode)):
        mode_color = COLORS[mode_indices_reversed[i]]

        # plot the paired ends
        max_y = -np.inf
        sample_means = []
        for evidence in mode:
            color = add_color_noise(mode_color)
            y = add_noise(i + 1)
            # determine the y shift for the histograms later
            max_y = max(y, max_y)
            min_y = min(y, min_y)

            # plot only the max L coordinate and min R coordinate (closest values to the actual SV, which is not sequenced)
            mean_l = np.mean(
                [paired_end[0] for paired_end in evidence.paired_ends]
            )
            mean_r = np.mean(
                [paired_end[1] for paired_end in evidence.paired_ends]
            )
            bax.plot(
                [mean_l, mean_r],
                [y, y],
                marker="o",
                markersize=5,
                linestyle="-",
                linewidth=1,
                color=color,
            )
            sample_means.extend([mean_l, mean_r])
        all_sample_means.append(sample_means)

        # plot the histograms
        for ax in bax.axs:
            n, bins, hist_patches = ax.hist(
                sample_means,
                bins=30,
                color=mode_color,
                alpha=0.8,
                range=(min(sample_means), max(sample_means)),
                orientation="vertical",
                align="mid",
            )
            max_hist_height = max(n)
            scale_factor = 0.5 / max_hist_height
            for patch in hist_patches:
                patch.set_height(patch.get_height() * scale_factor)
                patch.set_y(patch.get_y() + max_y + 0.05)
            ax.set_ylim(min_y - 0.1, max_y + 0.65)

    # plot the vertical lines
    for i in range(num_modes):
        sample_means = np.array(all_sample_means[i])
        bax.axvline(
            np.mean(sample_means[0::2]),
            color=COLORS[mode_indices_reversed[i]],
            linestyle="--",
        )
        bax.axvline(
            np.mean(sample_means[1::2]),
            color=COLORS[mode_indices_reversed[i]],
            linestyle="--",
        )

    # Set ticks and labels for SV plot
    for ax in bax.axs:
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelrotation=15)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.yaxis.set_major_locator(
            FixedLocator([mode + 1 for mode in mode_indices])
        )
    bax.locator_params(axis="x", nbins=4)
    bax.set_xlabel("Paired Ends", labelpad=35, fontsize=12)

    # Add the legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ANCESTRY_COLORS[value])
        for value in ANCESTRY_COLORS.keys()
    ]
    plt.legend(
        handles[::-1],
        list(ANCESTRY_COLORS.keys())[::-1],
        loc="center left",
        bbox_to_anchor=(-0.18, 0.9),
        title="Superpopulation",
    )

    # Plot the pie chart
    pie_ax = fig.add_axes([0.36, 0.68, 0.25, 0.25])
    counts = [len(mode) for mode in evidence_by_mode]
    total_population = sum(counts)
    mode_percentages = [count / total_population for count in counts]
    pie_ax.pie(
        mode_percentages,
        colors=COLORS[: len(mode_indices)],
        startangle=90,
    )
    pie_ax.set_title("Samples per Cluster", fontsize=10, pad=-10)
    pie_ax.set_aspect("equal")


def plot_sequence(
    evidence_by_mode: List[List[Evidence]], ref_sequence: Seq.Seq
):
    sv_stats = get_svlen(evidence_by_mode)
    mode_indices = list(range(len(evidence_by_mode)))
    mode_indices_reversed = mode_indices[::-1]
    for i, mode in enumerate(reversed(evidence_by_mode)):
        lefts = []
        rights = []
        for evidence in mode:
            lefts.append(
                max([paired_end[0] for paired_end in evidence.paired_ends])
            )
            rights.append(
                min([paired_end[1] for paired_end in evidence.paired_ends])
            )
        all_paired_ends = lefts + rights
        max_left = max(lefts)
        min_right = min(rights)

        n, bins, hist_patches = plt.hist(
            all_paired_ends,
            bins=20,
            color=COLORS[mode_indices_reversed[i]],
            alpha=0.8,
            range=(min(all_paired_ends), max(all_paired_ends)),
            orientation="vertical",
            align="mid",
        )
        max_hist_height = max(n)
        scale_factor = 0.8 / max_hist_height
        for patch in hist_patches:
            patch.set_height(patch.get_height() * scale_factor)
            patch.set_y(patch.get_y() + i + 1)
        plt.ylim(i + 0.9, i + 2)

        ref_seq = ref_sequence.seq[int(max_left) : int(min_right)]
        if len(ref_seq) > 20:
            truncated_ref_seq = f"{ref_seq[:10]}......{ref_seq[-10:]}"
        else:
            truncated_ref_seq = ref_seq

        plt.text(
            x=(max_left + min_right) / 2,
            y=i + 1.01,
            s=truncated_ref_seq,
            ha="center",
            fontsize=10,
        )

    for i in range(len(evidence_by_mode)):
        plt.axvline(
            np.mean([sv.start for sv in sv_stats[i]]),
            color=COLORS[mode_indices_reversed[i]],
            linestyle="--",
        )
        plt.axvline(
            np.mean([sv.end for sv in sv_stats[i]]),
            color=COLORS[mode_indices_reversed[i]],
            linestyle="--",
        )

    plt.xlabel("Paired Ends")
    plt.ylabel("Modes")
    plt.yticks(
        ticks=[mode + 1 for mode in mode_indices],
        labels=[y_label + 1 for y_label in mode_indices_reversed],
    )
    plt.show()


def plot_sv_lengths(evidence_by_mode: List[List[Evidence]]):
    plt.figure(figsize=(15, 8))
    for i, mode in enumerate(evidence_by_mode):
        all_lengths = []
        for evidence in mode:
            # TODO: recalculate the lengths with individual insert sizes
            lengths = [
                max(paired_end) - min(paired_end) - 450
                for paired_end in evidence.paired_ends
            ]
            all_lengths.append(np.mean(lengths))

        gmm_iters, _ = run_em1d(all_lengths, 1)
        gmm = gmm_iters[-1]
        ux, hx = get_scatter_data(all_lengths)
        plt.plot(
            ux,
            4
            * (len(all_lengths) * gmm.p[0])
            * norm.pdf(ux, gmm.mu[0], np.sqrt(gmm.vr[0])),
            linestyle="-",
            color=COLORS[i],
        )
        plt.hist(all_lengths, bins=10, color=COLORS[i], alpha=0.5)
    plt.xlabel("SV Length", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.show()


def plot_sv_coords(evidence_by_mode: List[List[Evidence]]):
    plt.figure(figsize=(15, 8))
    for i, mode in enumerate(evidence_by_mode):
        coords = [evidence.mean_l for evidence in mode]
        gmm_iters, _ = run_em1d(coords, 1)
        gmm = gmm_iters[-1]
        ux, hx = get_scatter_data(coords)
        plt.plot(
            ux,
            4
            * (len(coords) * gmm.p[0])
            * norm.pdf(ux, gmm.mu[0], np.sqrt(gmm.vr[0])),
            linestyle="-",
            color=COLORS[i],
        )
        plt.hist(coords, bins=10, color=COLORS[i], alpha=0.5)
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    plt.xlabel("L Coordinate", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.show()


def plot_2d_coords(
    ax_main,
    evidence_by_mode: List[List[Evidence]],
    *,
    axis1: str,
    axis2: str,
    add_error_bars: bool = False,
    color_by: str = "mode",
    size_by="num_evidence",
):
    scatter_cm = cm.get_cmap("tab20").colors + cm.get_cmap("tab20b").colors
    seq_center_df = get_sample_sequencing_centers()
    insert_sizes_df = pd.read_csv("1kgp/insert_sizes.csv")
    color_lookup = {}
    for i, mode in enumerate(evidence_by_mode):
        x = []
        num_evidence = []
        mean_insert_sizes = []
        sem_ax1 = []
        sem_ax2 = []
        scatter_colors = []
        scatter_labels = []
        scatter_sizes = []
        for evidence in mode:
            ax1_vals = [
                GMM_AXES[axis1](x, evidence.mean_insert_size)
                for x in evidence.paired_ends
            ]
            ax2_vals = [
                GMM_AXES[axis2](x, evidence.mean_insert_size)
                for x in evidence.paired_ends
            ]
            x.append([np.mean(ax1_vals), np.mean(ax2_vals)])
            num_evidence.append(len(evidence.paired_ends))
            sem_ax1.append(sem(ax1_vals))
            sem_ax2.append(sem(ax2_vals))

            try:
                seq_centers = ", ".join(
                    seq_center_df[
                        seq_center_df["SAMPLE_NAME"] == evidence.sample.id
                    ]["CENTER_NAME"].values[0]
                )
            except IndexError:
                seq_centers = "Unknown"  # to handle synthetic data

            scatter_labels.append(seq_centers)
            if color_by == "mode":
                scatter_colors.append(COLORS[i])
            elif color_by == "sequencing_center":
                if seq_centers not in color_lookup:
                    color = scatter_cm[len(color_lookup) % len(scatter_cm)]
                    color_lookup[seq_centers] = color
                scatter_colors.append(color_lookup[seq_centers])

            try:
                mean_insert_size = insert_sizes_df[
                    insert_sizes_df["sample_id"] == evidence.sample.id
                ]["mean_insert_size"].values[0]
            except IndexError:
                mean_insert_size = 450  # to handle synthetic data

            if mean_insert_size == 0:
                mean_insert_size = 450  # default value
            mean_insert_sizes.append(mean_insert_size)

            if size_by == "num_evidence":
                scatter_sizes.append(num_evidence[-1] * 40)
            elif size_by == "insert_size":
                scatter_sizes.append(mean_insert_size)

        x = np.array(x)
        num_evidence = np.array(num_evidence)

        gmm_iters, _ = run_em(x, 1)
        gmm = gmm_iters[-1]

        # plot 2D data
        ax_main.scatter(
            x[:, 0],
            x[:, 1],
            color=scatter_colors,
            label=scatter_labels,
            sizes=scatter_sizes,
            alpha=0.6,
        )

        if add_error_bars:
            for j in range(len(x)):
                ax_main.errorbar(
                    x[j, 0],
                    x[j, 1],
                    xerr=sem_ax1[j],
                    yerr=sem_ax2[j],
                    color=COLORS[i],
                    alpha=0.6,
                )

        # manually adjust x/y for each figure
        ax_main.text(
            gmm.mu[0][0],
            gmm.mu[0][1],
            f"n={len(mode)}\n{axis1}: {np.mean(x[:, 0]):.0f}\n{axis2}: {np.mean(x[:, 1]):.0f}",
            # f"n={len(mode)}\n{axis1}: {np.mean(x[:, 0]):.0f}\n{axis2}: {np.mean(x[:, 1]):.0f}\nAvg. num reads/sample: {np.mean(num_evidence):.1f}\nMean insert size: {int(np.mean(mean_insert_sizes))}",
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            zorder=10,
        )

        # plot the 2D gaussian distributions for each cluster
        eigenvalues, eigenvectors = np.linalg.eigh(gmm.cov[0])
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = patches.Ellipse(
            xy=gmm.mu[0],
            width=width,
            height=height,
            angle=angle,
            edgecolor=COLORS[i],
            fc="None",
            lw=2,
        )
        ax_main.add_patch(ellipse)

        # plot the 1D gaussian distributions along the axes
        ax_xhist = ax_main.inset_axes([0, 1, 1, 0.2], sharex=ax_main)
        mean_x, std_x = np.mean(x[:, 0]), np.std(x[:, 0])
        x_vals = np.linspace(mean_x - 3 * std_x, mean_x + 3 * std_x, 100)
        zorder = 10 - i
        ax_xhist.plot(
            x_vals,
            norm.pdf(x_vals, mean_x, std_x),
            color=COLORS[i],
            linewidth=2,
            alpha=0.8,
            zorder=zorder,
        )
        ax_xhist.fill_between(
            x_vals,
            norm.pdf(x_vals, mean_x, std_x),
            color=COLORS[i],
            alpha=0.8,
            zorder=zorder,
        )
        ax_xhist.axis("off")
        # set the zorder of the axes so the gaussians are plotted in the correct order
        ax_xhist.set_zorder(zorder - 1)

        ax_yhist = ax_main.inset_axes([1, 0, 0.2, 1], sharey=ax_main)
        ax_yhist.set_zorder(zorder - 1)
        mean_y, std_y = np.mean(x[:, 1]), np.std(x[:, 1])
        y_vals = np.linspace(mean_y - 3 * std_y, mean_y + 3 * std_y, 100)
        ax_yhist.plot(
            norm.pdf(y_vals, mean_y, std_y),
            y_vals,
            color=COLORS[i],
            linewidth=2,
            alpha=0.8,
            zorder=zorder,
        )
        ax_yhist.fill_betweenx(
            y_vals,
            0,
            norm.pdf(y_vals, mean_y, std_y),
            color=COLORS[i],
            alpha=0.8,
            zorder=zorder,
        )
        ax_yhist.axis("off")

    if axis1 != "length":
        ax_main.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    if axis2 != "length":
        ax_main.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))

    ax_labels = [axis1, axis2]
    for i, ax_label in enumerate(ax_labels):
        if ax_label == "L" or ax_label == "R":
            ax_labels[i] = f"{ax_label}-Position"
    ax_main.set_xlabel(ax_labels[0], fontsize=12)
    ax_main.set_ylabel(ax_labels[1], fontsize=12)

    if axis1 == "L" or axis1 == "R":
        ax_main.tick_params(axis="x", labelrotation=15)


def plot_single_sv(
    evidence_by_mode: List[List[Evidence]],
    *,
    sv_id: str = "",
    axis1: str,
    axis2: str,
    add_error_bars: bool = False,
    color_by: str = "mode",
    size_by="num_evidence",
):
    fig = plt.figure(figsize=(12, 3))
    gs = GridSpec(1, 3, width_ratios=[1, 5, 4], figure=fig)
    plot_evidence_by_mode(fig, gs, evidence_by_mode)
    ax2 = fig.add_subplot(gs[0, 2])
    plot_2d_coords(
        ax2,
        evidence_by_mode,
        axis1=axis1,
        axis2=axis2,
        add_error_bars=add_error_bars,
        size_by=size_by,
        color_by=color_by,
    )
    plt.tight_layout()
    plt.subplots_adjust(
        left=0.02, right=0.93, bottom=0.24, top=0.83, wspace=0.25, hspace=0
    )
    plot_title = f"plots/{sv_id}{'' if sv_id == '' else '_'}evidence_by_mode"
    plt.savefig(f"{plot_title}.pdf")
    plt.show()


def extract_hetero_homozygous(row):
    modes_data = eval(row["modes"])
    heterozygous_counts = [mode["num_heterozygous"] for mode in modes_data]
    homozygous_counts = [mode["num_homozygous"] for mode in modes_data]
    return heterozygous_counts, homozygous_counts


def get_num_samples_gmm(df):
    df["num_samples_gmm"] = df.apply(
        lambda row: [mode["num_samples"] for mode in eval(row["modes"])], axis=1
    )
    # TODO: subtract num outliers in the gmm
    return df


def get_svs_intersecting_genes(df: pd.DataFrame):
    sv_gene_overlap_df = pd.read_csv(
        "1kgp/sv_intersect.csv", header=None, delimiter="\t"
    )
    intersecting_svs = sv_gene_overlap_df.iloc[:, 3]
    return df[df["id"].isin(intersecting_svs)]


def plot_processed_sv_stats(filter_intersecting_genes: bool = False):
    """
    Plots the rectangle shapes showing the distribution of all SVs and how they're split
    """
    df = get_sv_stats_collapsed_df()
    df = df[df["num_samples"] > 0]

    if filter_intersecting_genes:
        # only show SVs that intersect with a gene
        get_svs_intersecting_genes(df)

    df = get_num_samples_gmm(df)
    df["num_heterozygous"], df["num_homozygous"] = zip(
        *df.apply(extract_hetero_homozygous, axis=1)
    )
    df["afs"] = df.apply(
        lambda row: [mode["af"] for mode in eval(row["modes"])], axis=1
    )

    mode_data = [df[df["num_modes"] == i + 1] for i in range(3)]
    total_ns = [x.shape[0] for x in mode_data]
    splits = [x * (i + 1) for i, x in enumerate(total_ns)]
    num_svs_pre_split = sum(total_ns)
    num_svs_post_split = sum(splits)
    scaling_factor = num_svs_post_split / num_svs_pre_split

    fig, ax = plt.subplots(figsize=(16, 10))
    gs = fig.add_gridspec(1, 4)

    # Plot bars showing SV distribution
    ax_bar = fig.add_subplot(gs[0, :3])

    y0 = 0
    # Plot the first bar
    for i, mode in enumerate(total_ns):
        proportion = mode / num_svs_pre_split
        height = proportion / scaling_factor
        ax_bar.add_patch(
            patches.Rectangle(
                (0, y0),
                0.3,
                height,
                facecolor=COLORS[i],
                edgecolor="none",
            )
        )
        ax_bar.text(
            0.15,
            y0 + height / 2,
            f"{i + 1} Mode{'' if i == 0 else 's'}",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )
        y0 += height

    y1 = 0
    bar_heights = []
    # Plot the 2nd bar
    for i, split in enumerate(splits):
        proportion = split / num_svs_post_split
        num_homo = mode_data[i]["num_homozygous"].apply(lambda x: sum(x))
        num_hetero = mode_data[i]["num_heterozygous"].apply(lambda x: sum(x))
        allele_ratios = num_hetero / (num_homo + num_hetero)
        mean_allele_ratio = allele_ratios.mean()
        proportion_hetero = mean_allele_ratio * proportion
        ax_bar.add_patch(
            patches.Rectangle(
                (0.6, y1),
                0.3,
                proportion,
                facecolor=COLORS[i],
                edgecolor="none",
            )
        )
        ax_bar.plot(
            [0.6, 0.9],
            [y1 + proportion_hetero, y1 + proportion_hetero],
            color="black",
            linestyle="--",
            linewidth=1,
        )
        bar_heights.append((y1, y1 + proportion))
        y1 += proportion

    y0_1 = 0
    y0_2 = 0
    # Plot the polygons connecting the bars
    for i, (mode, split) in enumerate(zip(total_ns, splits)):
        proportion1 = mode / num_svs_pre_split / scaling_factor
        proportion2 = split / num_svs_post_split

        polygon_points = [
            (0.3, y0_1),
            (0.6, y0_2),
            (0.6, y0_2 + proportion2),
            (0.3, y0_1 + proportion1),
        ]

        polygon = patches.Polygon(
            polygon_points,
            closed=True,
            facecolor=COLORS[i],
            alpha=0.4,
            edgecolor="none",
        )
        ax_bar.add_patch(polygon)

        y0_1 += proportion1
        y0_2 += proportion2

    ax_bar.set_xlim(0, 0.9)
    ax_bar.set_ylim(0, 1)

    # Plot histogram of allele frequencies
    height_ratios = [(y_max - y_min) / y0 for y_min, y_max in bar_heights[::-1]]
    gs_sub = GridSpecFromSubplotSpec(
        3,
        1,
        subplot_spec=gs[0, 3],
        height_ratios=height_ratios,
        hspace=0,
        wspace=0,
    )
    hist_axs = []
    for i, mode in enumerate(mode_data):
        afs = np.array([x for y in mode["afs"].tolist() for x in y])
        logit_afs = logit(afs)
        hist_ax = fig.add_subplot(gs_sub[len(height_ratios) - i - 1])
        hist_axs.append(hist_ax)
        hist_ax.hist(
            logit_afs,
            bins=10,
            orientation="horizontal",
            color=COLORS[i],
            alpha=0.6,
        )
        hist_ax.set_ylim(logit(1e-10), logit(0.5))

    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    for ax in hist_axs:
        allele_frequencies = [0.01, 0.1, 0.5, 0.9, 0.99]
        logit_ticks = logit(allele_frequencies)
        ax.set_yticks(logit_ticks)
        ax.set_yticklabels([f"{af:.2f}" for af in allele_frequencies])

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0)
    ax_bar.set_position([0.05, 0.05, 0.7, 0.9])
    for hist_ax in hist_axs:
        pos = hist_ax.get_position()
        hist_ax.set_position([0.752, pos.y0, 0.2, pos.height])

    plt.axis("off")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.show()


def plot_sample_size_per_mode(filter_intersecting_genes: bool = False):
    """
    For each # of modes, plots a boxplot of the sample size for each SV
    """
    df = get_sv_stats_collapsed_df()
    if filter_intersecting_genes:
        df = get_svs_intersecting_genes(df)

    df = get_num_samples_gmm(df)
    nonzero = df[df["num_samples"] > 0]
    mode_data = [nonzero] + [df[df["num_modes"] == i + 1] for i in range(3)]

    sample_sizes = [
        x["num_samples_gmm"].apply(lambda y: sum(y)) for x in mode_data
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(sample_sizes, positions=[1, 2, 3, 4], widths=0.6)
    ax.set_xticks([1, 2, 3, 4])
    labels = [
        f"{label} (n={len(sample_sizes[i])})\n{len(sample_sizes[i]) / len(sample_sizes[0]) * 100:.2f}%"
        for i, label in enumerate(["All SVs", "1 Mode", "2 Modes", "3 Modes"])
    ]
    ax.set_xticklabels(labels)
    ax.set_xlabel("")
    ax.set_ylabel("Sample Size")
    plt.show()


def plot_removed_evidence(sv_evidence: List[Evidence], L: int, R: int):
    """
    For an SV, plots the evidence that has been removed due to not enough evidence or deviation from the y=x+b line
    """
    plt.figure(figsize=(15, 8))
    colors = {0: "grey", 1: "red", 2: "orange", 3: "blue"}
    evidence_sorted = {0: [], 1: [], 2: [], 3: []}
    for evidence in sv_evidence:
        y = add_noise(evidence.removed)
        evidence_sorted[evidence.removed].extend(evidence.paired_ends)
        for paired_end in evidence.paired_ends:
            plt.plot(
                [paired_end[0], paired_end[1]],
                [y, y],
                marker="o",
                markersize=5,
                linestyle="-",
                linewidth=1,
                color=colors[evidence.removed],
                alpha=0.6,
            )
    for key, paired_ends in evidence_sorted.items():
        if len(paired_ends) == 0:
            continue
        plt.axvline(
            np.mean([paired_end[0] for paired_end in paired_ends]),
            color=colors[key],
            linestyle="--",
        )
        plt.axvline(
            np.mean([paired_end[1] for paired_end in paired_ends]),
            color=colors[key],
            linestyle="--",
        )

    plt.axvline(
        L,
        color="black",
        linestyle="--",
    )
    plt.axvline(
        R,
        color="black",
        linestyle="--",
    )

    legend = {
        0: "Not Removed",
        1: "Not Enough Evidence (1 pair of reads)",
        2: "Not Enough Evidence (2 pairs of reads)",
        3: "Deviation from y=x+b line",
    }
    plt.legend(
        [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors.values()],
        list(legend.values()),
        loc="upper right",
    )
    plt.show()


def query_ref_genome(chr: str) -> Seq.Seq:
    with gzip.open(REFERENCE_FILE, "rt") as handle:
        record_dict = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
    return record_dict[chr]


def query_sample(sample: str, chr: str, L: int, R: int):
    record_dict = SeqIO.to_dict(
        SeqIO.parse(f"{SAMPLES_DIR}/{sample}_consensus.fa", "fasta")
    )
    sequence = record_dict[chr].seq[L:R]
    return sequence


def populate_sample_info(
    sv_evidence: List[Evidence],
    chr: str,
    L: int,
    R: int,
) -> None:
    """
    Populates each sample with its sex, population, and superpopulation information
    """
    ancestry_df = pd.read_csv("1kgp/ancestry.tsv", delimiter="\t")
    deletions_df = pd.read_csv(f"1kgp/deletions_by_chr/chr{chr}.csv")
    deletions_row = deletions_df[
        (deletions_df["start"] == L) & (deletions_df["stop"] == R)
    ].iloc[0]
    for evidence in sv_evidence:
        sample_id = evidence.sample.id
        ancestry_row = ancestry_df[ancestry_df["Sample name"] == sample_id]
        superpopulation = (
            ancestry_row["Superpopulation code"].values[0].split(",")[0]
        )
        allele = deletions_row[sample_id]
        evidence.sample = Sample(
            id=sample_id,
            sex=ancestry_row["Sex"].values[0],
            population=ancestry_row["Population code"].values[0],
            superpopulation=superpopulation,
            allele=allele,
        )


def analyze_ancestry() -> None:
    """
    Plots a bar chart of the total ancestry and superancestry counts from the 1000 Genomes samples
    """
    df = pd.read_csv("1kgp/ancestry.tsv", delimiter="\t")
    population_data = Counter()
    superpopulation_data = Counter()
    population_lookup = {}
    for _, row in df.iterrows():
        population = row["Population code"].split(",")[0]
        superpopulation = row["Superpopulation code"].split(",")[0]
        population_data[population] += 1
        superpopulation_data[superpopulation] += 1
        population_lookup[population] = superpopulation

    population_labels = sorted(
        population_data.keys(), key=lambda x: population_lookup[x]
    )
    population_counts = [population_data[label] for label in population_labels]

    superpopulation_labels = sorted(superpopulation_data.keys())
    superpopulation_counts = [
        superpopulation_data[label] for label in superpopulation_labels
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for label, count in zip(population_labels, population_counts):
        ax1.bar(label, count, color=ANCESTRY_COLORS[population_lookup[label]])
    ax1.set_title("Population Counts", fontsize=14)
    ax1.set_xlabel("Population", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.tick_params(axis="x", rotation=90)

    for label, count in zip(superpopulation_labels, superpopulation_counts):
        ax2.bar(label, count, color=ANCESTRY_COLORS[label])
    ax2.set_title("Superpopulation Counts", fontsize=14)
    ax2.set_xlabel("Superpopulation", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)

    plt.tight_layout()
    plt.show()


def compare_sv_ancestry_by_mode(by: str = "superpopulation"):
    ancestry_df = pd.read_csv("1kgp/ancestry.tsv", delimiter="\t")
    sv_df = get_sv_stats_collapsed_df()
    sv_df = sv_df[sv_df["num_modes"] > 1]

    populations = (
        SUPERPOPULATIONS if by == "superpopulation" else SUBPOPULATIONS
    )[::-1]

    comparisons = [np.zeros(len(populations)) for _ in range(len(populations))]
    all_comparisons = [
        np.zeros(len(populations)) for _ in range(len(populations))
    ]
    for _, row in sv_df.iterrows():
        modes = ast.literal_eval(row["modes"])
        sp_by_mode = []
        for mode in modes:
            sample_ids = mode["sample_ids"]
            pops = set()
            for sample_id in sample_ids:
                ancestry_row = ancestry_df[
                    ancestry_df["Sample name"] == sample_id
                ]
                pop_key = (
                    "Superpopulation code"
                    if by == "superpopulation"
                    else "Population code"
                )
                population = ancestry_row[pop_key].values[0].split(",")[0]
                pops.add(population)
            sp_by_mode.append(pops)

        for i, p1 in enumerate(populations):
            all_sp = [sp for mode in sp_by_mode for sp in mode]
            if p1 not in all_sp:
                continue
            for j_idx, p2 in enumerate(populations[i + 1 :]):
                j = i + j_idx + 1
                if p2 not in all_sp:
                    continue
                all_comparisons[i][j] += 1
                all_comparisons[j][i] += 1
                for mode in sp_by_mode:
                    if p1 in mode and p2 in mode:
                        comparisons[i][j] += 1
                        comparisons[j][i] += 1
                        break

    for i in range(len(populations)):
        for j in range(len(populations)):
            if i == j:
                comparisons[i][j] = 1
            elif all_comparisons[i][j] == 0:
                comparisons[i][j] = 0
            else:
                comparisons[i][j] /= all_comparisons[i][j]

    population_lookup = {}
    for i, row in ancestry_df.iterrows():
        population_lookup[row["Population code"]] = row["Superpopulation code"]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(comparisons, cmap="Blues", interpolation="nearest")
    ax.set_yticks(range(len(populations)))
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels(populations, fontsize=12)
    ax.text(7, -3.25, "Superpopulation", fontsize=14)
    # ax.set_xlabel("Superpopulation", fontsize=14, labelpad=20)
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

            rect = patches.Rectangle(
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

    rect_border = patches.Rectangle(
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
    plt.savefig("plots/ancestry_comparison.pdf", bbox_inches="tight")
    plt.show()


def plot_afs():
    """
    Plots the allele frequencies for the SVs before and after being split by SVeperator
    """
    sv_df = get_sv_stats_collapsed_df()
    sv_df = sv_df[sv_df["num_samples"] > 0]
    # calc afs for all 3

    afs = {}
    for num_modes in range(1, 4):
        x = []  # original AFs
        y = []  # new AFs
        df = sv_df[sv_df["num_modes"] == num_modes]
        for _, row in df.iterrows():
            n_homozygous = 0
            n_heterozygous = 0
            modes = ast.literal_eval(row["modes"])
            for i, mode in enumerate(modes):
                n_homozygous += mode["num_homozygous"]
                n_heterozygous += mode["num_heterozygous"]

                af = calc_af(
                    mode["num_homozygous"],
                    mode["num_heterozygous"],
                    2504,  # hard code population size for efficiency
                )
                y.append(af)

            # row["num_samples"] includes all samples, including reference. these are the samples that were run through the GMM
            original_af = calc_af(n_homozygous, n_heterozygous, 2504)
            for _ in modes:
                x.append(original_af)

            afs[num_modes] = [x, y]

    for num_modes in range(1, 4):
        x, y = afs[num_modes]
        plt.figure(figsize=(6, 6))
        plot_lim = max(x)

        if num_modes == 2:
            plot_lim += 0.01
            x3, y3 = afs[3]
            x3_max = max(x3) + 0.002

            plt.gca().add_patch(
                patches.Rectangle(
                    (0, 0),
                    x3_max,
                    x3_max,
                    linewidth=1,
                    edgecolor="grey",
                    facecolor="grey",
                    alpha=0.2,
                )
            )
        elif num_modes == 3:
            plot_lim += 0.002
            x2, y2 = afs[2]

            plt.gca().add_patch(
                patches.Rectangle(
                    (0, 0),
                    plot_lim,
                    plot_lim,
                    linewidth=1,
                    edgecolor="grey",
                    facecolor="grey",
                    alpha=0.1,
                )
            )
            plt.scatter(x2, y2, color=COLORS[1], alpha=0.1)

        plt.scatter(x, y, color=COLORS[num_modes - 1], alpha=0.6)

        if num_modes == 2:
            x3, y3 = afs[3]
            plt.scatter(x3, y3, color=COLORS[2], alpha=0.15)

        plt.plot([0, plot_lim], [0, plot_lim], linestyle="--", color="darkgrey")
        plt.xlim(0, plot_lim)
        plt.ylim(0, plot_lim)

        # plt.set_title(
        #     f"Num Modes={num_modes + 1}, n={np.mean(n_samples):.1f}"
        # )
        plt.xlabel("Original Allele Frequency", fontsize=12)
        plt.ylabel("New Allele Frequency", fontsize=12)
        plt.show()


def plot_afs_hexbin():
    """
    Plots hexbin maps of allele frequencies for 2- and 3-mode SVs.
    """
    sv_df = get_sv_stats_collapsed_df()
    sv_df = sv_df[
        (sv_df["num_samples"] > 0) & (sv_df["num_modes"].isin([2, 3]))
    ]

    afs = {2: ([], []), 3: ([], [])}

    for _, row in sv_df.iterrows():
        num_modes = row["num_modes"]
        modes = ast.literal_eval(row["modes"])
        x = afs[num_modes][0]
        y = afs[num_modes][1]

        n_homozygous = 0
        n_heterozygous = 0

        for mode in modes:
            n_homozygous += mode["num_homozygous"]
            n_heterozygous += mode["num_heterozygous"]

            af = calc_af(
                mode["num_homozygous"],
                mode["num_heterozygous"],
                2504,
            )
            y.append(af)

        original_af = calc_af(n_homozygous, n_heterozygous, 2504)
        for _ in modes:
            x.append(original_af)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for num_modes, ax in zip([2, 3], axes):
        x, y = afs[num_modes]

        orig_cmap = cm.get_cmap("Blues")
        new_colors = orig_cmap(np.linspace(0.2, 1, 256))
        darker_blues = ListedColormap(new_colors)
        hb = ax.hexbin(x, y, gridsize=40, cmap=darker_blues, mincnt=1)

        # Add identity line
        plot_lim = max(max(x), max(y)) + 0.01
        ax.plot([0, plot_lim], [0, plot_lim], linestyle="--", color="darkgrey")

        ax.set_xlim(0, plot_lim)
        ax.set_ylim(0, plot_lim)
        ax.set_xlabel("Original Allele Frequency", fontsize=12)
        ax.set_ylabel("New Allele Frequency", fontsize=12)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Counts")

    plt.tight_layout()
    plt.savefig("plots/afs_hexbin.png")
    plt.show()


def plot_af_delta_histogram():
    sv_df = get_sv_stats_collapsed_df()
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
        [
            original_afs[original_afs <= 0.5],
            original_afs_split[original_afs_split <= 0.5],
        ],
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
    ax_box.set_ylim(0, 0.5)
    ax_box.tick_params(axis="y", labelsize=12)
    ax_box.yaxis.set_major_locator(FixedLocator(np.arange(0, 0.51, 0.1)))

    ax.set_xlabel("Allele Frequency Ratio, New/Original", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    ax.xaxis.set_major_locator(FixedLocator(np.arange(0, 1.1, 0.2)))
    ax.yaxis.set_major_locator(FixedLocator(np.arange(0, 181, 20)))
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(0, 1.1, 0.1)))
    ax.yaxis.set_minor_locator(FixedLocator(np.arange(0, 181, 10)))
    ax.tick_params(axis="x", which="minor", length=4, labelbottom=False)
    ax.tick_params(axis="y", which="minor", length=4, labelleft=False)
    plt.tight_layout()
    plt.savefig("plots/af_delta_histogram.pdf")
    plt.show()


def plot_pre_post_split_diffs():
    sv_df = get_sv_stats_collapsed_df()
    sv_df = sv_df[sv_df["num_samples"] > 10]
    original_afs = sv_df[sv_df["num_modes"] == 1]["af"].values
    original_n = sv_df[sv_df["num_modes"] == 1]["num_samples"].values
    split_afs = sv_df[sv_df["num_modes"].isin([2, 3])]["af"].values
    split_n = sv_df[sv_df["num_modes"].isin([2, 3])]["num_samples"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.violinplot(
        [original_n, split_n],
        showmeans=True,
        showmedians=True,
        widths=0.6,
    )
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["Unsplit SVs", "Split SVs"], fontsize=12)
    ax1.set_ylabel("Original Number of Samples (N)", fontsize=12)

    ax2.violinplot(
        [original_afs, split_afs],
        showmedians=True,
        widths=0.6,
    )
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Unsplit SVs", "Split SVs"], fontsize=12)
    ax2.set_ylabel("Original Allele Frequency", fontsize=12)
    plt.savefig("plots/pre_post_split_diffs.pdf", bbox_inches="tight")
    plt.show()


def plot_original_afs():
    sv_df = get_sv_stats_collapsed_df()
    sv_df = sv_df[sv_df["num_samples"] > 10]
    original_afs = sv_df[sv_df["num_modes"] == 1]["af"].values
    split_afs = sv_df[sv_df["num_modes"].isin([2, 3])]["af"].values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for ax, label, values, color in zip(
        [ax1, ax2],
        ["Unsplit SVs", "Split SVs"],
        [original_afs, split_afs],
        ["gray", "orange"],
    ):
        ax1.hist(
            values,
            bins=20,
            color=color,
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(FixedLocator(np.arange(0, 1.1, 0.2)))
        ax.xaxis.set_minor_locator(FixedLocator(np.arange(0, 1.1, 0.1)))
        ax.tick_params(axis="x", which="minor", length=4, labelbottom=False)
        ax.set_title(label, fontsize=14)
    ax1.set_ylabel("Count", fontsize=12)
    fig.text(0.5, 0.01, "Original Allele Frequency", fontsize=12, ha="center")
    plt.savefig("plots/original_afs.pdf", bbox_inches="tight")
    plt.show()


def plot_reciprocal_overlap(ax, case: str):
    file = "synthetic_data/resultsn=96.csv"
    df = pd.read_csv(file)
    df = df[
        (df["expected_num_modes"] == 2)
        & (df["case"] == case)
        & (df["reciprocal_overlap"] > 0)
    ]

    colors = ["#bfdbf7", "#1f7a8c", "#022b3a"]
    markers = ["o", "s", "D"]
    for i, model in enumerate(GMM_MODELS):
        model_df = df[df["gmm_model"] == model]
        right = Counter()
        total = Counter()
        for _, row in model_df.iterrows():
            overlap = row["reciprocal_overlap"]
            total[overlap] += 1
            if row["expected_num_modes"] == row["num_modes"]:
                right[overlap] += 1

        overlaps = sorted(total.keys())
        acc = [right[overlap] / total[overlap] for overlap in overlaps]

        ax.plot(
            overlaps,
            acc,
            marker=markers[i],
            color=colors[i],
            label=model,
            linewidth=2,
        )
        ax.set_xlabel("Reciprocal Overlap", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_ylim(0, 1)
    ax.legend(title="Model", fontsize=12, title_fontsize=14)


def plot_reciprocal_overlap_all():
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    cases = ["A", "B"]
    for i, case in enumerate(cases):
        plot_reciprocal_overlap(axs[i], case)
        axs[i].set_title(f"Case {case}", fontsize=16)
    plt.tight_layout()
    plt.savefig("plots/reciprocal_overlap_accuracy.pdf", bbox_inches="tight")
    plt.show()


def draw_conceptual_clusters(
    ax1, ax2, case, n_per_cluster: int = 50, *, fontsize: int = 12
):
    lr_centroids = np.array(SYNTHETIC_DATA_CENTROIDS[case])
    cluster_spread = 50

    centroids = []
    clusters = []
    for lr_center in lr_centroids:
        center = np.array([lr_center[0], lr_center[1] - lr_center[0]])
        centroids.append(center)
        points = np.random.normal(
            loc=center, scale=cluster_spread, size=(n_per_cluster, 2)
        ).astype(int)
        clusters.append(points)
    centroids = np.array(centroids)

    for i, points in enumerate(clusters):
        ax2.scatter(points[:, 0], points[:, 1], color=COLORS[i], s=10)

    # draw lines between centroids
    center1 = (
        centroids[0] if case != "E" else (100500, 2278)
    )  # midpoint between sv1 and sv2
    center2 = centroids[1] if len(centroids) == 2 else centroids[2]
    ax2.plot(
        [center1[0], center2[0]],
        [center1[1], center2[1]],
        linewidth=1.5,
        color="black",
    )

    padding = [0, 0]
    match case:
        case "A":
            padding = [50, 50]
        case "B":
            padding = [0, 25]
        case "C":
            padding = [0, 25]
        case "D":
            padding = [60, -60]
        case "E":
            padding = [60, -60]

    ax2.text(
        (center1[0] + center2[0]) / 2 + padding[0],
        (center1[1] + center2[1]) / 2 + padding[1],
        "d",
        fontsize=fontsize,
        ha="center",
        va="center",
    )

    if case == "E":
        ax2.plot(
            [centroids[0][0], centroids[1][0]],
            [centroids[0][1], centroids[1][1]],
            linestyle="--",
            color="black",
        )
        ax2.scatter([100500], [2278], color="black", s=30)

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines["left"].set_linewidth(1.5)
    ax2.spines["bottom"].set_linewidth(1.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # draw the patches representing the coordinate space
    x_min = lr_centroids.min() - 100
    x_max = lr_centroids.max() + 100
    y_offset = 0
    for i, (L, R) in enumerate(lr_centroids):
        ax1.plot(
            [x_min, L - 10],
            [y_offset + 0.1, y_offset + 0.1],
            color="black",
            linewidth=2,
        )
        ax1.plot(
            [R + 10, x_max],
            [y_offset + 0.1, y_offset + 0.1],
            color="black",
            linewidth=2,
        )
        ax1.add_patch(
            patches.Rectangle(
                (L, y_offset),
                R - L,
                0.2,
                color=COLORS[i],
            )
        )
        y_offset += 0.25

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(0, 1)
    ax1.axis("off")


def draw_conceptual_clusters_all(
    fig, gs, *, fontsize: int = 12, n_per_cluster: int = 100
):
    for i, case in enumerate(["A", "B", "C", "D", "E"]):
        ax1 = fig.add_subplot(gs[3, i])
        ax2 = fig.add_subplot(gs[5, i])
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        dx = 0.1
        ax1.set_position([pos1.x0 + dx, pos1.y0, pos1.width, pos1.height])
        ax2.set_position([pos2.x0 + dx, pos2.y0, pos2.width, pos2.height])
        draw_conceptual_clusters(
            ax1, ax2, case, n_per_cluster, fontsize=fontsize
        )
    fig.text(0.5, 0.01, "L-position", fontsize=fontsize, ha="center")
    fig.text(
        0.03,
        0.13,
        "Length",
        fontsize=fontsize,
        va="center",
        rotation="vertical",
    )


def plot_d_accuracy_by_n(n_samples: int):
    """
    Plots the accuracy of each of the GMM models for each example (see Figure 2) and distance
    """
    file = f"synthetic_data/results{n_samples}.csv"
    if not os.path.exists(file):
        print(f"File for {n_samples} samples does not exist")
        return

    df = pd.read_csv(file)
    fig, axs = plt.subplots(2, 3, figsize=(18, 12), sharey=True)
    fig.delaxes(axs[1, 2])
    df = df[df["num_samples"] == n_samples]
    for idx, case in enumerate(
        ["A", "B", "C", "D", "E"]
    ):  # skipping case C since it filters out all data
        ax = axs[idx // 3, idx % 3]
        case_df = df[df["case"] == case]
        for i, model in enumerate(GMM_MODELS):
            accuracy_by_dist = defaultdict(list)
            model_df = case_df[case_df["gmm_model"] == model]
            for _, row in model_df.iterrows():
                dist = row["d"]
                correct = (
                    1 if row["num_modes"] == row["expected_num_modes"] else 0
                )
                accuracy_by_dist[dist].append(correct)
            distances = sorted(accuracy_by_dist.keys())
            accuracies = [
                sum(accuracy_by_dist[d]) / len(accuracy_by_dist[d])
                for d in distances
            ]
            ax.plot(distances, accuracies, label=model, color=COLORS[i])
        ax.set_title(f"Case {case}")
        ax.set_xlabel("Distance")
        if idx == 0:
            ax.set_ylabel("Accuracy")
        ax.legend()

    plt.suptitle(f"N={n_samples}")
    plt.tight_layout()
    plt.show()


def plot_d_accuracy_by_case(
    ax_large,
    case: str,
    show_n_plots: bool = True,
    *,
    show_legend: bool = False,
    show_yticks: bool = True,
    show_xticks: bool = True,
    fontsize: int = 12,
):
    files = [
        f
        for f in os.listdir("synthetic_data")
        if re.match(r"resultsn=\d+\.csv", f)
    ]
    n_samples = sorted(
        [int(re.search(r"resultsn=(\d+)\.csv", f).group(1)) for f in files]
    )
    midpoint = int(len(n_samples) / 2)
    vals_to_plot = [
        n_samples[0],
        n_samples[int(midpoint / 2)],
        n_samples[int(midpoint)],
        n_samples[int(midpoint / 2 + midpoint)],
        n_samples[-1],
    ]

    d_acc_vals = defaultdict(list)  # only the d at accuracy = 80%
    all_d_acc_vals = defaultdict(
        list
    )  # for the small plots below the large plot, for vals_to_plot
    for n in n_samples:
        file = f"synthetic_data/resultsn={n}.csv"
        df = pd.read_csv(file)
        df = df[df["case"] == case]

        for model in GMM_MODELS:
            model_df = df[df["gmm_model"] == model]
            accuracies = []

            # distance values match the ones in synthetic_tests.py
            if case in ["A", "B", "C"]:
                distances = list(range(0, 505, 5))
            else:
                distances = list(range(0, 1110, 10))
            for dist in distances:
                dist_df = model_df[model_df["d"] == dist]
                correct = dist_df[
                    dist_df["num_modes"] == dist_df["expected_num_modes"]
                ]
                accuracies.append(
                    0.0 if len(dist_df) == 0 else len(correct) / len(dist_df)
                )

            (indices,) = np.where(np.array(accuracies) >= 0.8)
            if len(indices) > 0:
                d_acc_vals[model].append(distances[indices[0]])
            else:
                d_acc_vals[model].append(np.nan)

            if n in vals_to_plot:
                # prune from distances/accuracies lists where accuracies = 0
                distances = [d for d, a in zip(distances, accuracies) if a != 0]
                accuracies = [a for a in accuracies if a != 0]
                all_d_acc_vals[model].append([distances, accuracies])

    if show_n_plots:
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, len(vals_to_plot), height_ratios=[3, 1])
        ax_large = fig.add_subplot(gs[0, :])
        axs_small = [
            fig.add_subplot(gs[1, i]) for i in range(len(vals_to_plot))
        ]

    colors = ["#bfdbf7", "#1f7a8c", "#022b3a"]
    markers = ["o", "s", "D"]
    for i, model in enumerate(GMM_MODELS):
        color = colors[i]
        vals = d_acc_vals[model]
        ax_large.plot(
            n_samples,
            vals,
            label=MODEL_NAMES[i],
            color=color,
            marker=markers[i],
        )
        ax_large.scatter(n_samples, vals, color=color, marker=markers[i])

        if show_n_plots:
            for j, ax in enumerate(axs_small):
                # set all x axis to 0 to 500, set all y axis to 0 to 1, only show y axis label on the leftmost plot
                ax.set_xlim(-10, 510)
                ax.set_ylim(-0.05, 1.1)
                if j > 0:
                    ax.set_yticklabels([])
                distances, accuracies = all_d_acc_vals[model][j]
                ax.plot(distances, accuracies, label=model, color=color)
                ax.set_title(f"N={vals_to_plot[j]}", fontsize=10)
                if j == 0:
                    fig.text(
                        0.025,
                        0.15,
                        "Accuracy",
                        va="center",
                        rotation="vertical",
                        fontsize=12,
                    )
            fig.text(0.5, 0.0, "Distance (d)", ha="center", fontsize=12)

    ylim = [-10, 125]
    if case in ["B", "C"]:
        ylim = [-12, 150]
    elif case in ["D", "E"]:
        ylim = [-88, 1100]
        ax_large.yaxis.set_minor_locator(FixedLocator(np.arange(0, 1100, 250)))
    ax_large.set_ylim(ylim[0], ylim[1])
    ax_large.set_xlim(-5, 510)
    ax_large.tick_params(axis="both", labelsize=fontsize)
    ax_large.axhline(0, color="black", linestyle="--", linewidth=1)
    for spine in ["left", "bottom", "top", "right"]:
        ax_large.spines[spine].set_linewidth(1.5)
    if show_legend:
        ax_large.legend(loc="upper right", fontsize=fontsize)
    ax_large.xaxis.set_minor_locator(FixedLocator(np.arange(0, 510, 100)))
    ax_large.tick_params(axis="both", width=1.5, which="both")
    if not show_xticks:
        ax_large.set_xticklabels([])
    if not show_yticks:
        ax_large.set_yticklabels([])


def plot_d_accuracy_by_case_all(fig, gs, *, fontsize: int = 12):
    ax1_big = fig.add_subplot(gs[0:2, 0:3])
    ax_small_1 = fig.add_subplot(gs[0, 4])
    ax_small_2 = fig.add_subplot(gs[0, 5])
    ax_small_3 = fig.add_subplot(gs[1, 4])
    ax_small_4 = fig.add_subplot(gs[1, 5])
    for case, ax in zip(
        ["A", "B", "C", "D", "E"],
        [ax1_big, ax_small_1, ax_small_2, ax_small_3, ax_small_4],
    ):
        show_legend = case == "A"
        show_xticks = case not in ["B", "C"]
        show_yticks = case not in ["C", "E"]
        plot_d_accuracy_by_case(
            ax,
            case,
            False,
            show_legend=show_legend,
            show_yticks=show_yticks,
            show_xticks=show_xticks,
            fontsize=fontsize,
        )
    fig.text(
        0.33, 0.36, "Number of samples (N)", fontsize=fontsize, ha="center"
    )
    fig.text(
        0.82, 0.36, "Number of samples (N)", fontsize=fontsize, ha="center"
    )
    fig.text(
        0.008,
        0.67,
        "Distance at 80% Accuracy (d$^*$)",
        fontsize=fontsize,
        va="center",
        rotation="vertical",
    )
    fig.text(
        0.582,
        0.67,
        "Distance at 80% Accuracy (d$^*$)",
        fontsize=fontsize,
        va="center",
        rotation="vertical",
    )


def plot_synthetic_data_figure():
    fig = plt.figure(figsize=(12, 7))
    gs1 = GridSpec(
        6,
        6,
        figure=fig,
        height_ratios=[1, 1, 0.15, 0.4, 0.1, 0.6],
        width_ratios=[1, 1, 1, 0.35, 1, 1],
    )
    gs1.update(wspace=0.1, hspace=0.1)
    gs2 = GridSpec(6, 5, figure=fig, height_ratios=[1, 1, 0.15, 0.4, 0.1, 0.6])
    gs2.update(wspace=0.25)
    fontsize = 14
    plot_d_accuracy_by_case_all(fig, gs1, fontsize=fontsize)
    draw_conceptual_clusters_all(fig, gs2, fontsize=fontsize)

    # figure labels
    for x, y, label in [
        (0.025, 0.98, "a"),
        (0.6, 0.98, "b"),
        (0.025, 0.3, "c"),
    ]:
        fig.text(
            x,
            y,
            label,
            fontsize=fontsize,
            fontweight="bold",
            ha="center",
            va="center",
        )

    # test case labels
    for x, y, label in [
        (0.08, 0.95, "1"),
        (0.67, 0.95, "2"),
        (0.84, 0.95, "3"),
        (0.67, 0.67, "4"),
        (0.84, 0.67, "5"),
        (0.06, 0.22, "1"),
        (0.25, 0.22, "2"),
        (0.44, 0.22, "3"),
        (0.63, 0.22, "4"),
        (0.82, 0.22, "5"),
    ]:
        fig.text(
            x,
            y,
            label,
            fontsize=fontsize,
            ha="center",
            va="center",
        )
    plt.tight_layout()
    plt.subplots_adjust(
        left=0.07, right=0.985, top=0.975, bottom=0.05, wspace=0.0, hspace=0.0
    )
    fig.savefig("plots/synthetic_data.pdf")
    plt.show()


def plot_seq_center_distribution():
    df = get_sv_stats_df()
    df = df[df["num_modes"] == 2]
    seq_center_df = get_sample_sequencing_centers()
    seq_centers_by_mode = defaultdict(lambda: Counter())
    all_seq_centers = set()
    for _, row in df.iterrows():
        modes = ast.literal_eval(row.modes)
        modes = sorted(modes, key=lambda x: x["start"])
        for i, mode in enumerate(ast.literal_eval(row.modes)):
            for sample_id in mode["sample_ids"]:
                seq_centers = seq_center_df[
                    seq_center_df["SAMPLE_NAME"] == sample_id
                ]["CENTER_NAME"].values[0]
                for seq_center in seq_centers:
                    all_seq_centers.add(seq_center)
                    seq_centers_by_mode[i][seq_center] += 1

    # For each mode, plot a pie chart of the distribution of sequencing centers
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    seq_center_cm = cm.get_cmap("Set2").colors
    for i in seq_centers_by_mode.keys():
        seq_center_counts = seq_centers_by_mode[i]
        values = [
            seq_center_counts[seq_center] for seq_center in all_seq_centers
        ]
        axs[i].pie(
            values,
            labels=all_seq_centers,
            colors=seq_center_cm[: len(all_seq_centers)],
            autopct="%1.1f%%",
        )
        axs[i].set_title(f"Mode {i + 1}")
    plt.show()

    plt.figure()
    totals = {
        seq_center: sum(
            [
                seq_centers_by_mode[i][seq_center]
                for i in seq_centers_by_mode.keys()
            ]
        )
        for seq_center in all_seq_centers
    }
    pcts = {}
    for mode, counter in seq_centers_by_mode.items():
        pcts[mode] = [
            counter[seq_center] / totals[seq_center]
            for seq_center in all_seq_centers
        ]

    bottom = np.zeros(len(all_seq_centers))
    for mode, pct in pcts.items():
        plt.bar(
            all_seq_centers, pct, 0.5, label=f"Mode {mode + 1}", bottom=bottom
        )
        bottom += pct
    plt.legend(loc="upper right")
    plt.show()


def plot_insert_size_distribution(insert_sizes):
    df = get_sv_stats_df()
    df = df[df["num_modes"] == 2]
    insert_sizes_df = pd.read_csv("1kgp/insert_sizes.csv")
    insert_sizes = defaultdict(list)
    for _, row in df.iterrows():
        modes = ast.literal_eval(row.modes)
        modes = sorted(modes, key=lambda x: x["start"])
        for i, mode in enumerate(ast.literal_eval(row.modes)):
            for sample_id in mode["sample_ids"]:
                mean_insert_size = int(
                    insert_sizes_df[insert_sizes_df["sample_id"] == sample_id][
                        "mean_insert_size"
                    ].values[0]
                )
                if mean_insert_size == 0:
                    mean_insert_size = 450  # default value
                insert_sizes[i].append(mean_insert_size)

    # For each mode, plot a box plot of the distribution of insert sizes
    plt.figure()
    num_modes = len(insert_sizes)
    values = [insert_sizes[i] for i in range(num_modes)]
    plt.boxplot(values, labels=[f"Mode {i + 1}" for i in range(num_modes)])
    plt.ylabel("Mean Insert Size")
    plt.title("Mean Insert Sizes by Mode")
    plt.show()


def plot_insert_size_by_seq_center():
    insert_size_df = pd.read_csv(
        "1kgp/insert_sizes.csv",
        dtype={"sample_id": str, "mean_insert_size": int},
    )
    seq_center_df = get_sample_sequencing_centers()

    df = pd.merge(
        seq_center_df,
        insert_size_df,
        left_on="SAMPLE_NAME",
        right_on="sample_id",
    )
    df.drop(columns=["SAMPLE_NAME"], inplace=True)
    df.rename(columns={"CENTER_NAME": "center_name"}, inplace=True)

    values = defaultdict(list)
    for _, row in df.iterrows():
        for center in row["center_name"]:
            values[center].append(row["mean_insert_size"])

    plt.figure()
    plt.boxplot(values.values(), labels=values.keys())
    plt.title("Mean Insert Size by Sequencing Center")
    plt.xlabel("Sequencing Center")
    plt.ylabel("Mean Insert Size")
    plt.show()


def get_outlier_coverage():
    outlier_coverage = []
    nonoutlier_coverage = []
    df = pd.read_csv("1kgp/coverage.csv")
    outlier_df = df[df["num_samples"] == 1]
    for _, row in outlier_df.iterrows():
        modes = df[df["sv_id"] == row["sv_id"]]
        for _, mode in modes.iterrows():
            coverage = ast.literal_eval(mode["coverage"])
            if mode["num_samples"] == 1:
                outlier_coverage.append(coverage[0])
            else:
                nonoutlier_coverage.extend(coverage)

    plt.figure()
    plt.boxplot(
        [outlier_coverage, nonoutlier_coverage],
        labels=[
            f"Outliers (1-sample modes)\nn={len(outlier_coverage)}, μ={np.mean(outlier_coverage):.2f}, median={np.median(outlier_coverage)}",
            f"Non-outliers\nn={len(nonoutlier_coverage)}, μ={np.mean(nonoutlier_coverage):.2f}, median={np.median(nonoutlier_coverage)}",
        ],
    )
    plt.ylabel("Coverage")
    plt.title("Coverage (# paired end reads) for each sample in each mode")
    plt.show()


def bootstrap_runs_histogram():
    sv_stats_df = pd.read_csv("1kgp/sv_stats_merged.csv", low_memory=False)

    resolved_df = sv_stats_df.groupby("id").filter(lambda x: len(x) <= 7)
    ambiguous_df = sv_stats_df.groupby("id").filter(lambda x: len(x) > 7)

    plt.figure()
    x_labels = ["1 Mode", "2 Modes", "3 Modes"]
    bottom = np.zeros(3)

    for df, label in zip(
        [resolved_df, ambiguous_df], ["Resolved SVs", "Ambiguous SVs"]
    ):
        unique_svs = df["id"].unique()
        print("num unique svs", len(unique_svs))
        data = np.zeros(3)
        for sv_id in unique_svs:
            sv_df = df[df["id"] == sv_id]
            num_modes_count = Counter(sv_df["num_modes"])
            num_modes = max(num_modes_count, key=num_modes_count.get)
            if np.isnan(num_modes):
                print(sv_id)
                continue
            num_modes = int(num_modes)
            idx = 0 if num_modes == 0 else num_modes - 1
            data[idx] += 1

        plt.bar(x_labels, data, label=label, bottom=bottom)
        bottom += data

    plt.ylabel("Number of SVs")
    plt.legend()
    plt.show()


def long_read_comparison():
    df = pd.read_csv("long_reads/split_svs_lr.csv")
    df.dropna(inplace=True)
    sv_ids = df["sv_id"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sv_df = pd.DataFrame(
        columns=[
            "sv_id",
            "modes",
            "short_read_length_diff",
            "long_read_length_diff",
            "deviation",
        ]
    )
    n_svs = np.zeros(3, dtype=int)
    for sv_id in sv_ids:
        rows = df[df["sv_id"] == sv_id]
        # svs may have been filtered out if there was no long read data
        if len(rows) == 1:
            continue

        n_svs[len(rows) - 1] += 1
        for i, row1 in rows.iterrows():
            for j, row2 in rows.iterrows():
                if i >= j:
                    continue
                # sort the rows by start position
                if row1["start"] > row2["start"]:
                    row1, row2 = row2, row1
                short_read_length = row2["length"] - row1["length"]
                long_read_length = row2["lr_length"] - row1["lr_length"]
                sv_df.loc[len(sv_df)] = [sv_id, f"{i + 1}, {j + 1}"] + [
                    short_read_length,
                    long_read_length,
                    long_read_length - short_read_length,
                ]

    axes[0].scatter(
        sv_df["short_read_length_diff"].values,
        sv_df["long_read_length_diff"].values,
        alpha=0.4,
    )
    # y=x line is where the difference in lengths between short read and long read SVs is equal
    axes[0].plot(
        [
            min(sv_df["short_read_length_diff"].values),
            max(sv_df["short_read_length_diff"].values),
        ],
        [
            min(sv_df["short_read_length_diff"].values),
            max(sv_df["short_read_length_diff"].values),
        ],
        linestyle="--",
        color="darkgrey",
    )
    axes[0].set_ylim(
        min(sv_df["short_read_length_diff"].values),
        max(sv_df["short_read_length_diff"].values),
    )
    axes[0].set_xlabel("Δ Length Between Modes (Short Reads)")
    axes[0].set_ylabel("Δ Length Between Modes (Long Reads)")

    axes[1].boxplot(
        sv_df["deviation"].values, tick_labels=["Deviation from y=x"]
    )
    stats = sv_df["deviation"].describe()
    axes[1].set_title(
        f"Mean Deviation: {stats['mean']:.2f} ± {stats['std']:.2f}"
    )
    plt.suptitle(f"{n_svs[1] + n_svs[2]} SV Comparisons with Long Read Data")
    plt.show()

    sv_df["abs_deviation"] = sv_df["deviation"].abs()
    print(sv_df.sort_values("abs_deviation").head(10))


def plot_cipos():
    df = pd.read_csv("1kgp/cipos.csv")
    starts = []
    ends = []
    for _, row in df.iterrows():
        cipos = ast.literal_eval(row["cipos"])
        ciend = ast.literal_eval(row["ciend"])
        starts.extend([cipos[0], ciend[0]])
        ends.extend([cipos[1], ciend[1]])

    plt.figure()
    plt.hist(
        np.abs(starts) + 1,
        bins=np.logspace(0, 3, 50),
        alpha=0.5,
        label="CI Start",
    )
    plt.hist(
        np.abs(ends) + 1,
        bins=np.logspace(0, 3, 50),
        alpha=0.5,
        label="CI End",
    )
    plt.xscale("log")
    plt.gca().xaxis.set_minor_locator(FixedLocator(np.arange(-2000, 2001, 100)))
    plt.tick_params(axis="x", which="minor", length=4, labelbottom=False)
    plt.ylabel("Count")
    plt.xlabel("Confidence Interval")
    plt.show()


# plot_synthetic_data_figure()
# plot_af_delta_histogram()
# compare_sv_ancestry_by_mode(by="population")
plot_reciprocal_overlap_all()
