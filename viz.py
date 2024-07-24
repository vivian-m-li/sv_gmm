from bokeh.plotting import figure, show, output_notebook, output_file, save
from bokeh.models import HoverTool, Range1d, ColumnDataSource, NumeralTickFormatter
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import matplotlib.cm as cm
from em import run_gmm
from gmm_types import *


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
    y: List[np.ndarray[float]],
    *,
    file_name: Optional[str],
    L: int,
    R: int,
    read_length: int,
    sig: int = 50,
) -> Tuple[np.ndarray[np.ndarray[float]], List[Evidence]]:
    ux = get_unique_x_values(y)

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

    for i, yi in enumerate(y):
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
                xp, yp = z[0::2], z[1::2]
                sdl = np.sum(np.abs(yp) <= sig)
                mb[i, :] = [sdl, len(xp), b, 0]
                sv_evidence[i] = Evidence(
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

    if file_name is not None:
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

    if file_name is not None:
        output_file(f"{file_name}_fitted_lines.html")
        save(p)

    return start_points_array, sv_evidence


def random_color():
    return (random.random(), random.random(), random.random())


def add_noise(value, scale=0.1):
    return value + np.random.normal(scale=scale)


def plot_evidence_by_mode(gmm: GMM, sv_evidence: List[Evidence], R: int):
    sv_evidence = np.array(sv_evidence)
    x_by_mode = [sorted(x + R) for x in gmm.x_by_mode]
    evidence_by_mode = [[] for _ in range(len(x_by_mode))]
    for evidence in sv_evidence:
        for i, mode in enumerate(x_by_mode):
            if evidence.start_y in mode:  # assumes that each mode has unique values
                evidence_by_mode[i].append(evidence)
                continue

    for i, mode in enumerate(evidence_by_mode):
        for evidence in mode:
            color = random_color()
            y = add_noise(i)
            for paired_end in evidence.paired_ends:
                x_values = [paired_end[0], paired_end[1]]
                y_values = [y, y]
                plt.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linestyle="-",
                    color=color,
                )

    plt.xlabel("Paired Ends")
    plt.ylabel("Modes")
    plt.show()


def get_intercepts(
    squiggle_data: List[np.ndarray[float]], *, file_name: Optional[str], L: int, R: int
) -> Tuple[np.ndarray[np.ndarray[float]], List[Evidence]]:
    mb, sv_evidence_unfiltered = filter_and_plot_sequences_bokeh(
        squiggle_data,
        file_name=file_name,
        L=L,
        R=R,
        read_length=450,
        sig=50,
    )
    intercepts, sv_evidence = plot_fitted_lines_bokeh(
        mb,
        sv_evidence_unfiltered,
        file_name=file_name,
        L=L,
        read_length=450,
        R=R,
        sig=50,
    )
    points = np.array([np.array(i)[1] for i in intercepts if len(i) > 1]) - R
    return points, sv_evidence


def run_viz_gmm(
    squiggle_data: List[np.ndarray[float]], *, file_name: str, L: int, R: int
):
    # plots that don't update data format
    sv_viz(squiggle_data, file_name=file_name)
    bokeh_scatterplot(
        squiggle_data,
        file_name=file_name,
        lower_bound=L - 1900,
        upper_bound=R + 1900,
        L=L,
        read_length=450,
        R=R,
    )

    # functions that transform data
    points, sv_evidence = get_intercepts(squiggle_data, file_name=file_name, L=L, R=R)
    gmm = run_gmm(points, plot=False, pr=True)
    plot_evidence_by_mode(gmm, sv_evidence, R)
    return gmm


def run_gmm_l_r(
    squiggle_data: List[np.ndarray[float]], *, file_name: str, L: int, R: int
):
    all_xp = []
    for row in squiggle_data:
        all_xp.extend(row[0::2] - L)
    gmm_l = run_gmm(all_xp)

    for i in range(gmm_l.num_modes):
        xp = gmm_l.x_by_mode[i] + L
        yp = []
        for row in squiggle_data:
            for j in range(0, len(row), 2):
                x = row[j]
                y = row[j + 1]
                if x in xp:
                    yp.append(y)
        yp = np.array(yp) - L

        gmm_r = run_gmm(yp)
        for mu_r in gmm_r.mu:
            print((gmm_l.mu[i], mu_r))
