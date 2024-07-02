from bokeh.plotting import figure, show, output_notebook, output_file, save
from bokeh.models import ColumnDataSource
from bokeh.models import Range1d
from bokeh.models import HoverTool
import numpy as np
from typing import Optional, List
import matplotlib.cm as cm


def sv_viz(data: List[np.ndarray], *, file_name: str):
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
            for l, r in zip(x, y):
                p.line([l, r], [i * 2, i * 2], line_width=2, color=color)

    output_file(f"{file_name}_horizontal_sequences.html")
    save(p)


# First Visualization with ALL points
def bokeh_scatterplot(
    data: List[np.ndarray],
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
    data: List[np.ndarray],
    *,
    file_name: str,
    L: int,
    l: int,
    R: int,
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
        zi[1::2] = yi[1::2] - (yi[0::2] + (R - (L - l)))

        # Shift x-values to the left by min(x) units
        zi[0::2] = yi[0::2] - np.min(ux)

        # Plot the transformed points
        p.scatter(zi[0::2], zi[1::2], size=8)  # Use circle glyphs
        p.line(zi[0::2], zi[1::2], line_width=2)  # Connect points with lines

    # Plot the specific line
    xp = np.linspace(np.min(ux), np.max(ux), 100)
    p.line(
        xp - np.min(ux),
        (xp + (R - L + l)) - (xp + (R - (L - l))),
        line_color="black",
        line_dash="dashed",
        line_width=2,
    )

    output_file(f"{file_name}_deviations.html")
    save(p)


def get_unique_x_values(data):
    all_x_values = []
    for array in data:
        x_values = array[0::2]
        all_x_values.extend(x_values)
    return np.unique(all_x_values)


# Third Viz with 3 + more point
def filter_and_plot_sequences_bokeh(
    y: List[np.ndarray], *, file_name: str, L: int, l: int, R: int, sig: int = 50
):
    ux = get_unique_x_values(y)

    p = figure(
        title=f"L = {L}, R = {R}",
        width=800,
        height=600,
        x_axis_label="distance from L",
        y_axis_label="deviation from y=x+b",
    )

    # Background rectangle
    p.quad(top=sig, bottom=-sig, left=0, right=max(ux) - min(ux), color="#F0F0F0")

    # y=x line
    p.line(ux - min(ux), ux - ux, line_dash="dashed", line_width=1, color="black")

    mb = np.zeros((len(y), 4))  # [ #pts | length | intercept | sv-flag]
    colors = ["#0000FF", "#FF0000", "#00FF00", "#00FFFF", "#FF00FF", "#000000"]

    for i, yi in enumerate(y):
        z = yi.copy()
        for j in range(0, len(yi), 2):
            if (
                yi[j] < (L - 2 * l)
                or yi[j] > (L + 1.5 * l)
                or yi[j + 1] < (R - 2 * l)
                or yi[j + 1] > (R + 5 * l)
            ):
                z[j] = z[j + 1] = np.nan

        z = z[~np.isnan(z)]

        if len(z) > 0:
            b = int(np.mean(z[1::2]) - np.mean(z[0::2]))
            z[1::2] -= z[0::2] + b
            z[0::2] -= min(ux)
            if len(z) >= 6:
                xp, yp = z[0::2], z[1::2]
                sdl = np.sum(np.abs(yp) <= sig)
                mb[i, :] = [sdl, len(xp), b, 0]

                if sdl >= 3:
                    p.line(xp, yp, line_width=2, color=colors[i % len(colors)])
                    p.scatter(xp, yp, size=6, color=colors[i % len(colors)], alpha=0.6)
                    mb[i, 3] = 1
                else:
                    p.line(xp, yp, line_width=2, color="#999999")
                    p.scatter(xp, yp, size=6, color="#999999", alpha=0.6)
            else:
                mb[i, :] = [-np.inf, len(z) // 2, -np.inf, 0]
                p.line(z[0::2], z[1::2], line_width=2, color="#999999")
                p.scatter(z[0::2], z[1::2], size=6, color="#999999", alpha=0.6)

    p.grid.grid_line_alpha = 0.3

    output_file(f"{file_name}_sequences.html")
    save(p)

    return mb


# Viz 4 with the intercepts
def plot_fitted_lines_bokeh(
    mb: List[np.ndarray], *, file_name: str, L: int, l: int, R: int, sig: int
):
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
        [L - l - sig, L + sig, L + sig, L - l - sig],
        [R - 2 * sig, R + l, R + l + 2 * sig, R],
        color="grey",
        line_color="grey",
    )

    start_points = []
    # Loop through each row in mb to add lines and points at the start of each line
    for row in mb:
        if row[3] == 1:  # Assuming the 4th column (index 3) indicates 'on target'
            start_x = L - l
            start_y = (L - l) + row[2]
            p.line([start_x, L], [start_y, L + row[2]], line_width=2, color="red")
            p.scatter([start_x], [start_y], size=2, color="red", alpha=0.6)
            start_points.append((start_x, start_y))

    start_points_array = np.array(start_points)

    p.line(
        [L - l, L], [R, R + l], line_width=5, color="black", line_dash="dashed"
    )  # y=x dashed line
    p.xaxis.major_label_text_font_size = "16pt"
    p.yaxis.major_label_text_font_size = "16pt"
    p.x_range.start = L - l
    p.x_range.end = L
    p.y_range.start = R
    p.y_range.end = R + l

    output_file(f"{file_name}_fitted_lines.html")
    save(p)

    return start_points_array
