import random
import colorsys
import gzip
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import logit
from Bio import SeqIO, Seq
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from brokenaxes import brokenaxes
from matplotlib.ticker import FixedLocator, StrMethodFormatter
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from bokeh.plotting import figure, show, output_notebook, output_file, save
from bokeh.models import HoverTool, Range1d, ColumnDataSource, NumeralTickFormatter
from collections import Counter
from typing import Optional, List, Tuple, Dict
from em import run_gmm, run_em, get_scatter_data
from gmm_types import *

REFERENCE_FILE = "hs37d5.fa.gz"
SAMPLES_DIR = "samples"


def sv_viz(data: List[np.ndarray[float]], *, file_name: str):
    p = figure(width=800, height=600)
    colors = cm.Set1.colors
    for i, sample_data in enumerate(data):
        if sample_data.size > 0:
            points = sample_data.reshape(-1, 2)
            x = points[:, 0]
            y = points[:, 1]
            rgb = colors[i % len(colors)]
            color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            p.circle(x, i * 2, color=color, size=5)
            p.circle(y, i * 2, color=color, size=5)
            for read_length, r in zip(x, y):
                p.line([read_length, r], [i * 2, i * 2], line_width=2, color=color)

    output_file(f"{file_name}_horizontal_sequences.html")
    save(p)


# First Visualization with ALL points
def bokeh_scatterplot(
    data: List[np.ndarray[float]],
    *,
    file_name: str,
    lower_bound: int,
    upper_bound: int,
    L: Optional[int] = None,
    R: Optional[int] = None,
    read_length: Optional[int] = None,
    sig: int = 50,  # allowed error in short reads
):
    """Creates a scatterplot with an SV region polygon using Bokeh, taking a list of arrays as input."""

    p = figure(
        width=800,
        height=600,
        title="Line Segments and Points with SV Region",
        x_axis_label="X Coordinate",
        y_axis_label="Y Coordinate",
        x_range=(lower_bound, upper_bound),
        y_range=(lower_bound, upper_bound),
    )

    # Plot SV_region polygon if parameters are provided
    if L is not None and read_length is not None and R is not None:
        p.patch(
            [L - read_length - sig, L + sig, L + sig, L - read_length - sig],
            [R - 2 * sig, R + read_length, R + read_length + 2 * sig, R],
            color="grey",
            alpha=0.5,
        )

    for sample_data in data:
        if sample_data.size > 0:
            # Reshape pairs of points in each row
            points = sample_data.reshape(-1, 2)

            # Extract x and y coordinates (left-start and right-end of the paired end)
            x = points[:, 0]
            y = points[:, 1]

            source = ColumnDataSource(data=dict(x=x, y=y))

            p.line("x", "y", source=source, line_width=2)
            p.scatter("x", "y", source=source, size=10, fill_color="blue")

    p.grid.grid_line_alpha = 0.3

    output_file(f"{file_name}_SVregion.html")
    save(p)


# Second Viz deviations from the line
def plot_deviation_bokeh(
    data: List[np.ndarray[float]],
    *,
    file_name: str,
    L: int,
    R: int,
    read_length: int,
    lower_x_lim: int = 130,
    upper_x_lim: int = 600,
    lower_y_lim: int = -400,
    upper_y_lim: int = 200,
):
    p = figure(
        width=600,
        height=400,
        x_range=(lower_x_lim, upper_x_lim),
        y_range=(lower_y_lim, upper_y_lim),
        title="Deviation Plot",
    )

    ux = np.array([])
    for yi in data:
        ux = np.union1d(ux, yi[0::2])  # Collect unique x values

        # Detrend y-coordinates
        zi = np.copy(yi)
        zi[1::2] = yi[1::2] - (yi[0::2] + (R - (L - read_length)))

        # Shift x-values to the left by min(x) units
        zi[0::2] = yi[0::2] - np.min(ux)

        # Plot the transformed points
        p.scatter(zi[0::2], zi[1::2], size=8)  # Use circle glyphs
        p.line(zi[0::2], zi[1::2], line_width=2)  # Connect points with lines

    # Plot the specific line
    xp = np.linspace(np.min(ux), np.max(ux), 100)
    p.line(
        xp - np.min(ux),
        (xp + (R - L + read_length)) - (xp + (R - (L - read_length))),
        line_color="black",
        line_dash="dashed",
        line_width=2,
    )

    output_file(f"{file_name}_deviations.html")
    save(p)


def get_unique_x_values(data: List[np.ndarray]) -> np.ndarray[int]:
    all_x_values = []
    for array in data:
        x_values = array[0::2]
        all_x_values.extend(x_values)
    return np.unique(all_x_values)


# Third Viz with 3 + more point
def filter_and_plot_sequences_bokeh(
    y: Dict[str, np.ndarray[float]],
    *,
    file_name: Optional[str],
    L: int,
    R: int,
    read_length: int,
    sig: int = 50,
    plot_bokeh: bool,
) -> Tuple[np.ndarray[np.ndarray[float]], List[Evidence]]:
    ux = get_unique_x_values(list(y.values()))

    p = figure(
        title=f"L = {L}, R = {R}",
        width=800,
        height=600,
        x_axis_label="distance from L",  # TODO: this only applies if we know L is the start of the SV - otherwise, why do we care?
        y_axis_label="deviation from y=x+b",
    )

    # Background rectangle
    p.quad(top=sig, bottom=-sig, left=0, right=max(ux) - min(ux), color="#F0F0F0")

    # y=x line
    p.line(ux - min(ux), ux - ux, line_dash="dashed", line_width=1, color="black")

    mb = np.zeros(
        (len(y), 4)
    )  # [# of pts that indicate sv | # of pieces of evidence returned by stix for the sample | average intercept for all pairs of points | sv-flag]
    sv_evidence = [None] * len(y)
    colors = ["#0000FF", "#FF0000", "#00FF00", "#00FFFF", "#FF00FF", "#000000"]

    for i, (sample_id, yi) in enumerate(y.items()):
        z = yi.copy()
        for j in range(0, len(yi), 2):
            if (
                yi[j] < (L - 2 * read_length)
                or yi[j] > (L + 1.5 * read_length)
                or yi[j + 1] < (R - 2 * read_length)
                or yi[j + 1] > (R + 5 * read_length)
            ):
                z[j] = z[j + 1] = np.nan

        z = z[~np.isnan(z)]
        z_filtered = z.copy()

        if len(z) > 0:
            b = int(
                np.mean(z[1::2]) - np.mean(z[0::2])
            )  # TODO: we don't know that these evidence points all belong to the same SV, but we're calculating one intercept for all of them
            z[1::2] -= z[0::2] + b
            z[0::2] -= min(ux)
            if len(z) >= 6:  # if there are more than 3 pairs of points
                # NOTE: should we filter out for under 5 points?
                xp, yp = z[0::2], z[1::2]
                sdl = np.sum(np.abs(yp) <= sig)
                mb[i, :] = [sdl, len(xp), b, 0]
                sv_evidence[i] = Evidence(
                    sample=Sample(id=sample_id),
                    intercept=b,
                    paired_ends=[
                        [z_filtered[i], z_filtered[i + 1]]
                        for i in range(0, len(z_filtered), 2)
                    ],
                )

                # if more than 3 pieces of evidence (paired read_length-r ends), then there is an SV here for this sample
                if sdl >= 3:
                    p.line(xp, yp, line_width=2, color=colors[i % len(colors)])
                    p.scatter(xp, yp, size=6, color=colors[i % len(colors)], alpha=0.6)
                    mb[i, 3] = 1
                else:  # this sample does not have an SV
                    p.line(xp, yp, line_width=2, color="#999999")
                    p.scatter(xp, yp, size=6, color="#999999", alpha=0.6)
            else:  # not enough evidence for an SV
                mb[i, :] = [-np.inf, len(z) // 2, -np.inf, 0]
                p.line(z[0::2], z[1::2], line_width=2, color="#999999")
                p.scatter(z[0::2], z[1::2], size=6, color="#999999", alpha=0.6)

    p.grid.grid_line_alpha = 0.3

    if plot_bokeh and file_name is not None:
        output_file(f"{file_name}_sequences.html")
        save(p)

    return mb, sv_evidence


# Viz 4 with the intercepts
def plot_fitted_lines_bokeh(
    mb: np.ndarray[np.ndarray[float]],
    sv_evidence_unfiltered: List[Evidence],
    *,
    file_name: Optional[str],
    L: int,
    read_length: int,
    R: int,
    sig: int,
    plot_bokeh: bool,
) -> np.ndarray[np.ndarray[float]]:
    p = figure(
        title="Fitted Lines Plot",
        width=800,
        height=600,
        x_axis_label="Reverse position",
        y_axis_label="Forward position",
        tools="pan, wheel_zoom, box_zoom, reset",
    )
    hover_tool = HoverTool(
        tooltips=[("x", "$x{0,0}"), ("y", "$y{0,0}")]
    )  # Corrected tooltips for hover display

    p.add_tools(hover_tool)
    # Add a grey polygon for background
    p.patch(
        [L - read_length - sig, L + sig, L + sig, L - read_length - sig],
        [R - 2 * sig, R + read_length, R + read_length + 2 * sig, R],
        color="grey",
        line_color="grey",
    )

    start_points = []
    sv_evidence = []
    # Loop through each row in mb to add lines and points at the start of each line
    for i, row in enumerate(mb):
        if row[3] == 1:  # sv-flag is True
            start_x = L - read_length
            start_y = start_x + row[2]  # intercept
            p.line([start_x, L], [start_y, L + row[2]], line_width=2, color="red")
            p.scatter([start_x], [start_y], size=2, color="red", alpha=0.6)
            start_points.append((start_x, start_y))

            evidence = sv_evidence_unfiltered[i]
            evidence.start_y = start_y
            sv_evidence.append(evidence)

    start_points_array = np.array(start_points)

    p.line(
        [L - read_length, L],
        [R, R + read_length],
        line_width=5,
        color="black",
        line_dash="dashed",
    )  # y=x dashed line
    p.x_range.start = L - read_length
    p.x_range.end = L
    p.y_range.start = R
    p.y_range.end = R + read_length
    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"
    p.xaxis.formatter = NumeralTickFormatter(format="0")
    p.yaxis.formatter = NumeralTickFormatter(format="0")

    if plot_bokeh and file_name is not None:
        output_file(f"{file_name}_fitted_lines.html")
        save(p)

    return start_points_array, sv_evidence


def random_color():
    return (random.random(), random.random(), random.random())


def add_noise(value, scale=0.07):
    return value + np.random.normal(scale=scale)


def get_evidence_by_mode(
    gmm: GMM, sv_evidence: List[Evidence], R: int
) -> List[List[Evidence]]:
    sv_evidence = np.array(sv_evidence)
    x_by_mode = [sorted(x + R) for x in gmm.x_by_mode]
    evidence_by_mode = [[] for _ in range(len(x_by_mode))]
    for evidence in sv_evidence:
        for i, mode in enumerate(x_by_mode):
            if evidence.start_y in mode:  # assumes that each mode has unique values
                evidence_by_mode[i].append(evidence)
                continue
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

        for evidence in mode:
            max_l = max([paired_end[0] for paired_end in evidence.paired_ends])
            min_r = min([paired_end[1] for paired_end in evidence.paired_ends])
            all_paired_ends.extend([max_l, min_r])
            max_max_l = max(max_l, max_max_l)
            min_min_r = min(min_r, min_min_r)
            all_mode_paired_ends.extend(evidence.paired_ends)
            population_counter[evidence.sample.superpopulation] += 1

        population_data.append(population_counter)

    all_mode_paired_ends = [
        p for paired_end in all_mode_paired_ends for p in paired_end
    ]

    # Plot the bar chart
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
    left_half = max_max_l - min(all_mode_paired_ends)
    right_half = max(all_mode_paired_ends) - min_min_r
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
    for i, mode in enumerate(reversed(evidence_by_mode)):
        mode_color = COLORS[mode_indices_reversed[i]]
        max_y = -np.inf
        all_paired_ends = []
        for evidence in mode:
            color = add_color_noise(mode_color)
            y = add_noise(i + 1)
            max_y = max(y, max_y)
            min_y = min(y, min_y)
            for paired_end in evidence.paired_ends:
                x_values = [paired_end[0], paired_end[1]]
                y_values = [y, y]
                bax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    markersize=5,
                    linestyle="-",
                    linewidth=1,
                    color=color,
                )
            max_l = max([paired_end[0] for paired_end in evidence.paired_ends])
            min_r = min([paired_end[1] for paired_end in evidence.paired_ends])
            all_paired_ends.extend([max_l, min_r])

        for ax in bax.axs:
            n, bins, hist_patches = ax.hist(
                all_paired_ends,
                bins=20,
                color=mode_color,
                alpha=0.8,
                range=(min(all_paired_ends), max(all_paired_ends)),
                orientation="vertical",
                align="mid",
            )
            max_hist_height = max(n)
            scale_factor = 0.5 / max_hist_height
            for patch in hist_patches:
                patch.set_height(patch.get_height() * scale_factor)
                patch.set_y(patch.get_y() + max_y + 0.05)
            ax.set_ylim(min_y - 0.1, max_y + 0.65)

    for i in range(num_modes):
        bax.axvline(
            np.mean([sv.start for sv in sv_stats[i]]),
            color=COLORS[mode_indices[i]],
            linestyle="--",
        )
        bax.axvline(
            np.mean([sv.end for sv in sv_stats[i]]),
            color=COLORS[mode_indices[i]],
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

    mean_length = int(np.mean([sv.length for lst in sv_stats for sv in lst]))
    # print(f"Average SV Length={mean_length}\n\n{'\n\n'.join(print_sv_stats(sv_stats))}")
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
                max(paired_end) - min(paired_end) for paired_end in evidence.paired_ends
            ]
            all_lengths.append(np.mean(lengths))

        gmm_iters, _ = run_em(all_lengths, 1)
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


def plot_processed_sv_stats():
    df = pd.read_csv("1000genomes/sv_stats.csv")

    # TODO: remove rows with no samples after GMM outlier removal
    df = df[df["num_samples"] > 0]

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


def plot_sample_size_per_mode():
    df = pd.read_csv("1000genomes/sv_stats.csv")
    df = get_num_samples_gmm(df)
    nonzero = df[df["num_samples"] > 0]
    mode_data = [nonzero] + [df[df["num_modes"] == i + 1] for i in range(3)]

    sample_sizes = [x["num_samples_gmm"].apply(lambda y: sum(y)) for x in mode_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(sample_sizes, positions=[1, 2, 3, 4], widths=0.6)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["All SVs", "1 Mode", "2 Modes", "3 Modes"])
    ax.set_xlabel("")
    ax.set_ylabel("Sample Size")

    plt.show()


def get_intercepts(
    squiggle_data: Dict[str, np.ndarray[float]],
    *,
    file_name: Optional[str],
    L: int,
    R: int,
    plot_bokeh: bool,
) -> Tuple[np.ndarray[np.ndarray[float]], List[Evidence]]:
    mb, sv_evidence_unfiltered = filter_and_plot_sequences_bokeh(
        squiggle_data,
        file_name=file_name,
        L=L,
        R=R,
        read_length=450,
        sig=50,
        plot_bokeh=plot_bokeh,
    )
    intercepts, sv_evidence = plot_fitted_lines_bokeh(
        mb,
        sv_evidence_unfiltered,
        file_name=file_name,
        L=L,
        R=R,
        read_length=450,
        sig=50,
        plot_bokeh=plot_bokeh,
    )
    points = np.array([np.array(i)[1] for i in intercepts if len(i) > 1]) - R
    return points, sv_evidence


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


def run_viz_gmm(
    squiggle_data: Dict[str, np.ndarray[float]],
    *,
    file_name: str,
    chr: str,
    L: int,
    R: int,
    plot: bool = True,
    plot_bokeh: bool = False,
) -> None:
    # plots that don't update data format
    if plot_bokeh:
        data = list(squiggle_data.values())
        sv_viz(data, file_name=file_name)
        bokeh_scatterplot(
            data,
            file_name=file_name,
            lower_bound=L - 1900,
            upper_bound=R + 1900,
            L=L,
            read_length=450,
            R=R,
        )

    # transforms data
    points, sv_evidence = get_intercepts(
        squiggle_data, file_name=file_name, L=L, R=R, plot_bokeh=plot_bokeh
    )

    if len(points) == 0:
        # print("No structural variants found in this region.")
        return None, None

    gmm = run_gmm(points, plot=plot, pr=False)

    populate_sample_info(
        sv_evidence, chr, L, R
    )  # mutates sv_evidence with ancestry data and homo/heterozygous for each sample
    evidence_by_mode = get_evidence_by_mode(gmm, sv_evidence, R)

    if plot:
        plot_evidence_by_mode(evidence_by_mode)
        plot_sv_lengths(evidence_by_mode)

    return gmm, evidence_by_mode
