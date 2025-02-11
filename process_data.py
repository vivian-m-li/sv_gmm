import warnings
import numpy as np
import matplotlib.cm as cm
from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool, ColumnDataSource, NumeralTickFormatter
from typing import Optional, List, Tuple, Dict
from em import run_gmm
from em_1d import run_gmm as run_gmm_1d
from viz import *
from gmm_types import *
from profiler import profile, dump_stats


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
    insert_size_lookup: Dict[str, int],
    sig: int = 50,
    plot_bokeh: bool,
) -> Tuple[np.ndarray[np.ndarray[float]], List[Evidence]]:
    ux = get_unique_x_values(list(y.values()))

    p = figure(
        title=f"L = {L}, R = {R}",
        width=800,
        height=600,
        x_axis_label="distance from L",  # TODO: this only applies if we know L is the start of the SV
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

        # Remove the filtering process so the model can capture "outliers" that are actually SVs
        # for j in range(0, len(yi), 2):  # look at pairs of points (L-R coordinates)
        #     # filter out points that are too far from the original SV's L and R coordinates
        #     if (
        #         yi[j] < (L - 2 * read_length)
        #         or yi[j] > (L + 1.5 * read_length)
        #         or yi[j + 1] < (R - 2 * read_length)
        #         or yi[j + 1] > (R + 5 * read_length)
        #     ):
        #         z[j] = z[j + 1] = (
        #             0  # setting the values to nan was giving me an error on the synthetic data
        #         )

        z = z[z != 0]
        z_filtered = z.copy()
        paired_ends = [
            [z_filtered[i], z_filtered[i + 1]] for i in range(0, len(z_filtered), 2)
        ]

        if len(z) > 0:
            # TODO: does this assume that the read length is the same for each sample?
            b = int(
                np.mean(z[1::2]) - np.mean(z[0::2])
            )  # R - L (including read length)
            z[1::2] -= z[0::2] + b  # subtract MLE y=x+b line
            z[0::2] -= min(ux)  # shift left by min(x) units
            mean_l = int(np.mean([paired_end[0] for paired_end in paired_ends]))
            if len(z) >= 4:  # if there are more than 2 pairs of points
                # Note: 2 vs >= 3 pairs of points didn't make a difference in the mean L/R coordinates of the reads
                xp, yp = z[0::2], z[1::2]
                sdl = np.sum(
                    np.abs(yp) <= 2 * sig
                )  # checking if the points are within 2 SD of read noise
                mb[i, :] = [sdl, len(xp), b, 0]
                sv_evidence[i] = Evidence(
                    sample=Sample(id=sample_id),
                    intercept=b,
                    mean_l=mean_l,
                    removed=3 if sdl < 3 else 0,
                    paired_ends=paired_ends,
                    mean_insert_size=insert_size_lookup[sample_id],
                )

                # if more than 3 pieces of evidence (paired read_length-r ends), then there is an SV here for this sample
                # include if >=3 (x,y) points within 2 sig distance of y=x+b line
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
                sv_evidence[i] = Evidence(
                    sample=Sample(id=sample_id),
                    intercept=0,
                    mean_l=mean_l,
                    removed=len(z) / 2,
                    paired_ends=paired_ends,
                    mean_insert_size=insert_size_lookup[sample_id],
                )

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
    mean_insert_size = int(
        np.mean([e.mean_insert_size for e in sv_evidence_unfiltered])
    )
    p.patch(
        [L - mean_insert_size - sig, L + sig, L + sig, L - mean_insert_size - sig],
        [R - 2 * sig, R + mean_insert_size, R + mean_insert_size + 2 * sig, R],
        color="grey",
        line_color="grey",
    )

    start_points = []
    sv_evidence = []
    # Loop through each row in mb to add lines and points at the start of each line
    for i, row in enumerate(mb):
        evidence = sv_evidence_unfiltered[i]
        if row[3] == 1:  # sv-flag is True

            start_x = (
                L - evidence.mean_insert_size
            )  # subtract the mean insert size for this sample
            start_y = start_x + row[2]  # intercept
            p.line([start_x, L], [start_y, L + row[2]], line_width=2, color="red")
            p.scatter([start_x], [start_y], size=2, color="red", alpha=0.6)
            start_points.append((start_x, start_y))

            evidence.start_y = start_y
            sv_evidence.append(evidence)

    start_points_array = np.array(start_points)

    p.line(
        [L - mean_insert_size, L],
        [R, R + mean_insert_size],
        line_width=5,
        color="black",
        line_dash="dashed",
    )  # y=x dashed line
    p.x_range.start = L - mean_insert_size
    p.x_range.end = L
    p.y_range.start = R
    p.y_range.end = R + mean_insert_size
    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"
    p.xaxis.formatter = NumeralTickFormatter(format="0")
    p.yaxis.formatter = NumeralTickFormatter(format="0")

    if plot_bokeh and file_name is not None:
        output_file(f"{file_name}_fitted_lines.html")
        save(p)

    return start_points_array, sv_evidence


def get_intercepts(
    squiggle_data: Dict[str, np.ndarray[float]],
    *,
    file_name: Optional[str],
    L: int,
    R: int,
    insert_size_lookup: Dict[str, int],
    plot_bokeh: bool,
) -> Tuple[np.ndarray[Tuple[float, int]], List[Evidence]]:
    mb, sv_evidence_unfiltered = filter_and_plot_sequences_bokeh(
        squiggle_data,
        file_name=file_name,
        L=L,
        R=R,
        insert_size_lookup=insert_size_lookup,
        sig=50,
        plot_bokeh=plot_bokeh,
    )
    intercepts, sv_evidence = plot_fitted_lines_bokeh(
        mb,
        sv_evidence_unfiltered,
        file_name=file_name,
        L=L,
        R=R,
        sig=50,
        plot_bokeh=plot_bokeh,
    )

    # plot_removed_evidence(sv_evidence_unfiltered, L, R)

    # save the largest x value associated with each intercept
    points = []  # [[intercept, max_l], ...]
    for bs, evidence in zip(intercepts, sv_evidence):
        if len(bs) > 0:
            # scale the intercept and maxL values
            points.append((bs[1] - R, evidence.mean_l - L))  # np.array(bs)[1]=start_y
    points = np.array(points)

    return points, sv_evidence


# @profile
def run_viz_gmm(
    squiggle_data: Dict[str, np.ndarray[float]],
    *,
    file_name: str,
    chr: str,
    L: int,
    R: int,
    plot: bool = True,
    plot_bokeh: bool = False,
    synthetic_data: bool = False,
    gmm_model: str = "2d",  # 1d_len, 1d_L, 2d
    insert_size_lookup: Optional[Dict[str, int]] = None,
) -> None:
    if insert_size_lookup is None:
        insert_size_df = pd.read_csv(
            "1000genomes/insert_sizes.csv",
            dtype={"sample_id": str, "mean_insert_size": int},
        )
        insert_size_lookup = {
            sample_id: mean_insert_size
            for sample_id, mean_insert_size in insert_size_df.values
        }

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
            read_length=450,  # TODO(later): update the function to take a lookup
            R=R,
        )

    # transforms data
    points, sv_evidence = get_intercepts(
        squiggle_data,
        file_name=file_name,
        L=L,
        R=R,
        insert_size_lookup=insert_size_lookup,
        plot_bokeh=plot_bokeh,
    )

    if len(points) == 0:
        warnings.warn("No structural variants found in this region.")
        return None, []

    if gmm_model == "2d":
        gmm_func = run_gmm
    else:
        gmm_func = run_gmm_1d
        if gmm_model == "1d_len":
            points = points[:, 0]
        elif gmm_model == "1d_L":
            points = points[:, 1]

    gmm = gmm_func(points, plot=plot, pr=False)

    if not synthetic_data:
        populate_sample_info(
            sv_evidence, chr, L, R
        )  # mutates sv_evidence with ancestry data and homo/heterozygous for each sample
    evidence_by_mode = get_evidence_by_mode(gmm, sv_evidence, L, R, gmm_model=gmm_model)
    if plot:
        # plot_evidence_by_mode(evidence_by_mode)
        plot_2d_coords(
            evidence_by_mode,
            axis1="L",
            axis2="Length",
            add_error_bars=False,
            color_by="sequencing_center",
            size_by="insert_size",
        )
        # plot_2d_coords(
        #     evidence_by_mode,
        #     axis1="L",
        #     axis2="R",
        #     add_error_bars=False,
        #     color_by="sequencing_center",
        #     size_by="insert_size",
        # )

    # dump_stats("run_viz_gmm")
    return gmm, evidence_by_mode
