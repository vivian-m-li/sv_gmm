import os
import random
import colorsys
import gzip
import math
import numpy as np
import pandas as pd
from scipy.stats import norm, sem
from scipy.special import logit
from Bio import SeqIO, Seq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from brokenaxes import brokenaxes
from matplotlib.ticker import FixedLocator, StrMethodFormatter
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from collections import Counter, defaultdict
from typing import List
from em import run_gmm, run_em
from em_1d import run_em as run_em1d, get_scatter_data
from gmm_types import *

REFERENCE_FILE = "hs37d5.fa.gz"
SAMPLES_DIR = "samples"


def random_color():
    return (random.random(), random.random(), random.random())


def add_noise(value, scale=0.07):
    return value + np.random.normal(scale=scale)


def get_evidence_by_mode(
    gmm: GMM, sv_evidence: List[Evidence], L: int, R: int, *, gmm_model: str = "2d"
) -> List[List[Evidence]]:
    sv_evidence = np.array(sv_evidence)
    x_by_mode = []
    for mode in gmm.x_by_mode:
        data = []
        for x in mode:
            if gmm_model == "1d_len":
                data.append((x + R))  # length
            elif gmm_model == "1d_L":
                data.append((x + L))  # L-coordinate
            else:
                data.append((x[0] + R, x[1] + L))  # (length, L-coordinate)
        x_by_mode.append(data)
    evidence_by_mode = [[] for _ in range(len(x_by_mode))]
    for evidence in sv_evidence:
        for i, mode in enumerate(x_by_mode):
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
    return evidence_by_mode


def get_mean_std(label: str, values: List[float]):
    return f"{label}={math.floor(np.mean(values))} +/- {round(np.std(values), 2)}"


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


def get_svlen(evidence_by_mode: List[List[Evidence]]) -> List[List[SVStat]]:
    all_stats = []
    for mode in evidence_by_mode:
        stats = []
        for evidence in mode:
            lengths = [
                max(paired_end) - min(paired_end) for paired_end in evidence.paired_ends
            ]
            stats.append(
                SVStat(
                    length=np.mean(lengths),
                    start=max([paired_end[0] for paired_end in evidence.paired_ends]),
                    end=min([paired_end[1] for paired_end in evidence.paired_ends]),
                )
            )
        all_stats.append(stats)
    return all_stats


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


def plot_evidence_by_mode(evidence_by_mode: List[List[Evidence]]):
    sv_stats = get_svlen(evidence_by_mode)
    num_modes = len(evidence_by_mode)
    mode_indices = list(range(num_modes))
    mode_indices_reversed = mode_indices[::-1]

    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(1, 2, width_ratios=[1, 5], figure=fig, wspace=0.33)

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

    left_ax.set_xlabel("Proportion", labelpad=20, fontsize=12)
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
        fig=fig,
        subplot_spec=gs[1],
    )

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
            mean_l = np.mean([paired_end[0] for paired_end in evidence.paired_ends])
            mean_r = np.mean([paired_end[1] for paired_end in evidence.paired_ends])
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
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.yaxis.set_major_locator(FixedLocator([mode + 1 for mode in mode_indices]))
    bax.locator_params(axis="x", nbins=4)

    # Add the legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ANCESTRY_COLORS[value])
        for value in ANCESTRY_COLORS.keys()
    ]
    plt.legend(
        handles[::-1],
        list(ANCESTRY_COLORS.keys())[::-1],
        loc="center left",
        bbox_to_anchor=(-0.17, 0.9),
        title="Superpopulation",
    )

    bax.set_xlabel("Paired Ends", labelpad=30, fontsize=12)

    # Plot the pie chart
    pie_ax = fig.add_axes([0.3, 0.7, 0.25, 0.25])
    counts = [len(mode) for mode in evidence_by_mode]
    total_population = sum(counts)
    mode_percentages = [count / total_population for count in counts]
    pie_ax.pie(
        mode_percentages,
        autopct="%1.1f%%",
        colors=COLORS[: len(mode_indices)],
        startangle=90,
    )
    pie_ax.set_aspect("equal")
    plt.show()


def plot_sequence(evidence_by_mode: List[List[Evidence]], ref_sequence: Seq.Seq):
    sv_stats = get_svlen(evidence_by_mode)
    mode_indices = list(range(len(evidence_by_mode)))
    mode_indices_reversed = mode_indices[::-1]
    for i, mode in enumerate(reversed(evidence_by_mode)):
        lefts = []
        rights = []
        for evidence in mode:
            lefts.append(max([paired_end[0] for paired_end in evidence.paired_ends]))
            rights.append(min([paired_end[1] for paired_end in evidence.paired_ends]))
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
    fig = plt.figure(figsize=(15, 8))
    for i, mode in enumerate(evidence_by_mode):
        coords = [evidence.mean_l for evidence in mode]
        gmm_iters, _ = run_em1d(coords, 1)
        gmm = gmm_iters[-1]
        ux, hx = get_scatter_data(coords)
        plt.plot(
            ux,
            4 * (len(coords) * gmm.p[0]) * norm.pdf(ux, gmm.mu[0], np.sqrt(gmm.vr[0])),
            linestyle="-",
            color=COLORS[i],
        )
        plt.hist(coords, bins=10, color=COLORS[i], alpha=0.5)
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    plt.xlabel("L Coordinate", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.show()


def plot_2d_coords(
    evidence_by_mode: List[List[Evidence]],
    *,
    axis1: str,
    axis2: str,
    add_error_bars: bool = False,
):
    fig, ax_main = plt.subplots(figsize=(15, 8))
    for i, mode in enumerate(evidence_by_mode):
        x = []
        num_evidence = []
        sem_ax1 = []
        sem_ax2 = []
        for evidence in mode:
            ax1_vals = [GMM_AXES[axis1](x) for x in evidence.paired_ends]
            ax2_vals = [GMM_AXES[axis2](x) for x in evidence.paired_ends]
            x.append([np.mean(ax1_vals), np.mean(ax2_vals)])
            num_evidence.append(len(evidence.paired_ends))
            sem_ax1.append(sem(ax1_vals))
            sem_ax2.append(sem(ax2_vals))
        x = np.array(x)
        num_evidence = np.array(num_evidence)

        gmm_iters, _ = run_em(x, 1)
        gmm = gmm_iters[-1]

        # plot 2D data
        ax_main.scatter(
            x[:, 0], x[:, 1], color=COLORS[i], sizes=num_evidence * 40, alpha=0.6
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

        mse_x = np.mean(sem_ax1)
        mse_y = np.mean(sem_ax2)
        ax_main.text(
            gmm.mu[0][0],
            gmm.mu[0][1],
            f"n={len(mode)}\nAvg. num reads/sample: {np.mean(num_evidence):.1f}\nMSE {axis1}: {mse_x:.2f}\nMSE {axis2}: {mse_y:.2}",
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
        )

        # plot the 2D gaussian distributions
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

        # plot the 1D gaussian distributions
        ax_xhist = ax_main.inset_axes([0, 1, 1, 0.2], sharex=ax_main)
        ax_xhist.hist(x[:, 0], bins=20, color=COLORS[i], alpha=0.6, density=True)
        mean_x, std_x = np.mean(x[:, 0]), np.std(x[:, 0])
        x_vals = np.linspace(mean_x - 3 * std_x, mean_x + 3 * std_x, 100)
        ax_xhist.plot(x_vals, norm.pdf(x_vals, mean_x, std_x), color=COLORS[i])
        ax_xhist.axis("off")

        ax_yhist = ax_main.inset_axes([1, 0, 0.2, 1], sharey=ax_main)
        ax_yhist.hist(
            x[:, 1],
            bins=20,
            color=COLORS[i],
            alpha=0.6,
            density=True,
            orientation="horizontal",
        )
        mean_y, std_y = np.mean(x[:, 1]), np.std(x[:, 1])
        y_vals = np.linspace(mean_y - 3 * std_y, mean_y + 3 * std_y, 100)
        ax_yhist.plot(norm.pdf(y_vals, mean_y, std_y), y_vals, color=COLORS[i])
        ax_yhist.axis("off")

    if axis1 != "length":
        ax_main.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    if axis2 != "length":
        ax_main.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))

    ax_main.set_xlabel(axis1, fontsize=12)
    ax_main.set_ylabel(axis2, fontsize=12)
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
        "1000genomes/sv_intersect.csv", header=None, delimiter="\t"
    )
    intersecting_svs = sv_gene_overlap_df.iloc[:, 3]
    return df[df["id"].isin(intersecting_svs)]


def plot_processed_sv_stats(filter_intersecting_genes: bool = False):
    df = pd.read_csv("1000genomes/sv_stats.csv")
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
            f"{i+1} Mode{'' if i == 0 else 's'}",
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
    df = pd.read_csv("1000genomes/sv_stats.csv")
    if filter_intersecting_genes:
        df = get_svs_intersecting_genes(df)

    df = get_num_samples_gmm(df)
    nonzero = df[df["num_samples"] > 0]
    mode_data = [nonzero] + [df[df["num_modes"] == i + 1] for i in range(3)]

    sample_sizes = [x["num_samples_gmm"].apply(lambda y: sum(y)) for x in mode_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(sample_sizes, positions=[1, 2, 3, 4], widths=0.6)
    ax.set_xticks([1, 2, 3, 4])
    labels = [
        f"{label} (n={len(sample_sizes[i])})\n{len(sample_sizes[i])/len(sample_sizes[0])*100:.2f}%"
        for i, label in enumerate(["All SVs", "1 Mode", "2 Modes", "3 Modes"])
    ]
    ax.set_xticklabels(labels)
    ax.set_xlabel("")
    ax.set_ylabel("Sample Size")
    plt.show()


def plot_removed_evidence(sv_evidence: List[Evidence], L: int, R: int):
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
    ancestry_df = pd.read_csv("1000genomes/ancestry.tsv", delimiter="\t")
    deletions_df = pd.read_csv("1000genomes/deletions_df.csv", low_memory=False)
    for evidence in sv_evidence:
        sample_id = evidence.sample.id
        ancestry_row = ancestry_df[ancestry_df["Sample name"] == sample_id]
        superpopulation = ancestry_row["Superpopulation code"].values[0].split(",")[0]
        deletions_row = deletions_df[
            (deletions_df["chr"] == chr)
            & (deletions_df["start"] == L)
            & (deletions_df["stop"] == R)
        ]
        allele = deletions_row.iloc[0][sample_id]
        evidence.sample = Sample(
            id=sample_id,
            sex=ancestry_row["Sex"].values[0],
            population=ancestry_row["Population code"].values[0],
            superpopulation=superpopulation,
            allele=allele,
        )


def analyze_ancestry() -> None:
    df = pd.read_csv("1000genomes/ancestry.tsv", delimiter="\t")
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


def plot_d_accuracy(n_samples: int):
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
                correct = 1 if row["num_modes"] == row["expected_num_modes"] else 0
                accuracy_by_dist[dist].append(correct)
            distances = sorted(accuracy_by_dist.keys())
            accuracies = [
                sum(accuracy_by_dist[d]) / len(accuracy_by_dist[d]) for d in distances
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


def processed_svs_intersecting_genes(num_modes: int):
    # create a bed file with the each mode from each variant as a separate row and use bedtools intersect
    pass
