import os
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool, ColumnDataSource, NumeralTickFormatter

from src.model.gmm import gmm
from src.utils.model_helper import reciprocal_overlap
from src.utils.types import (
    Evidence,
    Sample,
    EstimatedGMM,
    InsertSizeDistribution,
)
from src.utils.viz import plot_2d_coords_fig


def sv_viz(data: list[np.ndarray[float]], *, file_name: str):
    """Plots the paired-end read data as horizontal lines using Bokeh (written by Kit)."""
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
                p.line(
                    [read_length, r], [i * 2, i * 2], line_width=2, color=color
                )

    output_file(f"{file_name}_horizontal_sequences.html")
    save(p)


def bokeh_scatterplot(
    data: list[np.ndarray[float]],
    *,
    file_name: str,
    lower_bound: int,
    upper_bound: int,
    L: int | None = None,
    R: int | None = None,
    read_length: int | None = None,
    sig: int = 50,  # allowed error in short reads
):
    """Creates a scatterplot with an SV region polygon using Bokeh, taking a list of arrays as input (written by Kit). Points have not been filtered yet."""
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
    data: list[np.ndarray[float]],
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
    """Plots the deviation of paired-end reads from the expected y=x+b line using Bokeh (written by Kit)."""
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


def get_unique_x_values(data: list[np.ndarray]) -> np.ndarray[int]:
    """Extracts and returns unique x-values from a list of arrays."""
    all_x_values = []
    for array in data:
        x_values = array[0::2]
        all_x_values.extend(x_values)
    return np.unique(all_x_values)


# Third Viz with 3 + more point
def filter_and_plot_sequences_bokeh(
    y: dict[str, np.ndarray[float]],
    *,
    file_name: str | None,
    L: int,
    R: int,
    insert_size_lookup: dict[str, InsertSizeDistribution],
    sig: int = 50,
    min_pairs: int = 2,  # minimum number of paired end reads for a sample needed to keep the sample
    plot_bokeh: bool,
) -> tuple[np.ndarray[np.ndarray[float]], list[Evidence]]:
    """
    DEPRECATED
    Filters out samples with too few paired-end reads and plots the remaining sequences using Bokeh (written by Kit).
    """
    ux = get_unique_x_values(list(y.values()))

    if plot_bokeh:
        p = figure(
            title=f"L = {L}, R = {R}",
            width=800,
            height=600,
            x_axis_label="distance from L",
            y_axis_label="deviation from y=x+b",
        )

        # Background rectangle
        p.quad(
            top=sig,
            bottom=-sig,
            left=0,
            right=max(ux) - min(ux),
            color="#F0F0F0",
        )

        # y=x line
        p.line(
            ux - min(ux),
            ux - ux,
            line_dash="dashed",
            line_width=1,
            color="black",
        )

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

        z = z[(z != 0) & (~np.isnan(z))]
        z_filtered = z.copy()
        paired_ends = [
            [z_filtered[i], z_filtered[i + 1]]
            for i in range(0, len(z_filtered), 2)
        ]

        if len(z) > 0:
            b = int(
                np.mean(z[1::2]) - np.mean(z[0::2])
            )  # R - L (including read length)
            z[1::2] -= z[0::2] + b  # subtract MLE y=x+b line
            z[0::2] -= min(ux)  # shift left by min(x) units
            mean_l = int(np.mean([paired_end[0] for paired_end in paired_ends]))
            mean_r = int(np.mean([paired_end[1] for paired_end in paired_ends]))
            if (
                len(z) >= min_pairs * 2
            ):  # if there are at least 5 pairs of paired-end reads
                # Note: 2 vs >= 3 pairs of points didn't make a difference in the mean L/R coordinates of the reads
                xp, yp = z[0::2], z[1::2]
                # checking if the points are within 2 SD of read noise
                # this check isn't being used anywhere anymore
                sdl = np.sum(np.abs(yp) <= 2 * sig)
                # set mb[i, 3] to 1 because we're not going to filter out samples for having points that deviate too far
                mb[i, :] = [sdl, len(xp), b, 1]
                sv_evidence[i] = Evidence(
                    sample=Sample(id=sample_id),
                    svlen=b,
                    start=mean_l,
                    end=mean_r,
                    paired_ends=paired_ends,
                    mean_insert_size=insert_size_lookup[sample_id].mean,
                    insert_size_sd=insert_size_lookup[sample_id].sd,
                )

                # if more than 2 pieces of evidence (paired read_length-r ends), then there is an SV here for this sample
                # include if >=2 (x,y) points within 2 sig distance of y=x+b line
                # if sdl >= 2:
                #     mb[i, 3] = 1

                if plot_bokeh:  # remove plotting to improve efficiency
                    if sdl >= 2:
                        p.line(
                            xp, yp, line_width=2, color=colors[i % len(colors)]
                        )
                        p.scatter(
                            xp,
                            yp,
                            size=6,
                            color=colors[i % len(colors)],
                            alpha=0.6,
                        )
                        mb[i, 3] = 1
                    else:  # this sample does not have an SV
                        p.line(xp, yp, line_width=2, color="#999999")
                        p.scatter(xp, yp, size=6, color="#999999", alpha=0.6)
            else:  # not enough evidence for an SV
                mb[i, :] = [-np.inf, len(z) // 2, -np.inf, 0]
                sv_evidence[i] = Evidence(
                    sample=Sample(id=sample_id),
                    svlen=0,
                    start=mean_l,
                    end=mean_r,
                    paired_ends=paired_ends,
                    mean_insert_size=insert_size_lookup[sample_id].mean,
                    insert_size_sd=insert_size_lookup[sample_id].sd,
                )

                if plot_bokeh:
                    p.line(z[0::2], z[1::2], line_width=2, color="#999999")
                    p.scatter(
                        z[0::2], z[1::2], size=6, color="#999999", alpha=0.6
                    )

    if plot_bokeh and file_name is not None:
        p.grid.grid_line_alpha = 0.3
        output_file(f"{file_name}_sequences.html")
        save(p)

    return mb, sv_evidence


# Viz 4 with the intercepts
def plot_fitted_lines_bokeh(
    mb: np.ndarray[np.ndarray[float]],
    sv_evidence_unfiltered: list[Evidence],
    *,
    file_name: str | None,
    L: int,
    R: int,
    sig: int,
    plot_bokeh: bool,
) -> np.ndarray[np.ndarray[float]]:
    """
    DEPRECATED
    Loop through the samples and return only points that are not filtered out. Plot the fitted lines using Bokeh (written by Kit).
    """
    mean_insert_size = int(
        np.mean([e.mean_insert_size for e in sv_evidence_unfiltered])
    )

    if plot_bokeh:
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
            [
                L - mean_insert_size - sig,
                L + sig,
                L + sig,
                L - mean_insert_size - sig,
            ],
            [
                R - 2 * sig,
                R + mean_insert_size,
                R + mean_insert_size + 2 * sig,
                R,
            ],
            color="grey",
            line_color="grey",
        )

    start_points = []
    sv_evidence = []
    # Loop through each row in mb to add lines and points at the start of each line
    for i, row in enumerate(mb):
        evidence = sv_evidence_unfiltered[i]
        # sv-flag is True (doesn't deviate too far from the y=x line)
        if row[3] == 1:

            start_x = (
                L - evidence.mean_insert_size
            )  # subtract the mean insert size for this sample
            start_y = start_x + row[2]  # intercept
            start_points.append((start_x, start_y))

            if plot_bokeh:
                p.line(
                    [start_x, L],
                    [start_y, L + row[2]],
                    line_width=2,
                    color="red",
                )
                p.scatter([start_x], [start_y], size=2, color="red", alpha=0.6)

            sv_evidence.append(evidence)

    start_points_array = np.array(start_points)

    if plot_bokeh and file_name is not None:
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
        output_file(f"{file_name}_fitted_lines.html")
        save(p)

    return start_points_array, sv_evidence


def populate_sample_info(
    sv_evidence: list[Evidence],
    chr: str,
    L: int,
    R: int,
    *,
    stem: str,
) -> None:
    """
    Populates each sample with its sex, population, and superpopulation information
    """
    if os.path.exists(f"{stem}/ancestry.tsv"):
        ancestry_df = pd.read_csv(f"{stem}/ancestry.tsv", delimiter="\t")
    else:
        ancestry_df = pd.DataFrame()

    deletions_df = pd.read_csv(f"{stem}/svs_by_chr/chr{chr}.csv")
    deletions_row = deletions_df[
        (deletions_df["start"] == L) & (deletions_df["stop"] == R)
    ]
    if deletions_row.empty:
        # query region does not correspond with an SV in the callset
        return
    deletions_row = deletions_row.iloc[0]
    for evidence in sv_evidence:
        sample_id = evidence.sample.id
        try:
            ancestry_row = ancestry_df[ancestry_df["Sample name"] == sample_id]
        except KeyError:
            ancestry_row = pd.DataFrame()

        sex = "Unknown"
        population = "Unknown"
        superpopulation = "Unknown"
        if not ancestry_row.empty:
            sex = ancestry_row["Sex"].values[0]
            population = ancestry_row["Population code"].values[0]
            superpopulation = (
                ancestry_row["Superpopulation code"].values[0].split(",")[0]
            )
        allele = deletions_row[sample_id]
        evidence.sample = Sample(
            id=sample_id,
            sex=sex,
            population=population,
            superpopulation=superpopulation,
            allele=allele,
        )


def get_evidence_by_mode(
    gmm_result: EstimatedGMM,
    sv_evidence: list[Evidence],
) -> list[list[Evidence]]:
    sv_evidence = np.array(sv_evidence)
    evidence_by_mode = [[] for _ in range(gmm_result.num_modes)]

    all_sample_ids = set()
    assigned_sample_ids = set()
    for sample_idx, evidence in enumerate(sv_evidence):
        # the indices should be the same because we don't shuffle the evidence
        evidence.mode_probabilities = gmm_result.responsibility[
            sample_idx
        ].tolist()
        all_sample_ids.add(evidence.sample.id)
        for i, mode in enumerate(gmm_result.x_index_by_mode):
            # previously, we tried assigning samples to modes based on a unique
            # (length, L-coordinate) pair
            # however, this assumption is false
            # now, we rely on the index of the sample and the index of the point
            # passed into the gmm func. this is hacky and is possible will
            # break in the future
            if sample_idx in mode:
                evidence_by_mode[i].append(evidence)
                if evidence.sample.id in assigned_sample_ids:
                    raise ValueError(
                        f"Warning: sample {evidence.sample.id} is assigned to multiple modes."
                    )
                assigned_sample_ids.add(evidence.sample.id)
                continue

    # assert that each piece of evidence is assigned to at most one mode
    assert len(all_sample_ids) == len(assigned_sample_ids)
    return evidence_by_mode


def get_intercepts(
    squiggle_data: dict[str, np.ndarray[float]],
    *,
    file_name: str | None,
    L: int,
    R: int,
    insert_size_lookup: dict[str, InsertSizeDistribution],
    plot_bokeh: bool = False,
    min_pairs: int = 2,
) -> tuple[np.ndarray[tuple[float, int]], list[Evidence]]:
    """
    DEPRECATED: use process_data instead
    Wrapper function that filters and plots sequences.
    """
    mb, sv_evidence_unfiltered = filter_and_plot_sequences_bokeh(
        squiggle_data,
        file_name=file_name,
        L=L,
        R=R,
        insert_size_lookup=insert_size_lookup,
        sig=50,
        plot_bokeh=plot_bokeh,
        min_pairs=min_pairs,
    )

    if len(sv_evidence_unfiltered) == 0:
        return np.array([]), []

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
            points.append(
                (bs[1] - R, evidence.mean_l - L)
            )  # np.array(bs)[1]=start_y
    points = np.array(points)

    return points, sv_evidence


def process_squiggle_data(
    squiggle_data: dict[str, np.ndarray[float]],
    *,
    L: int,
    R: int,
    insert_size_lookup: dict[str, InsertSizeDistribution],
    min_pairs: int = 2,  # minimum number of paired end reads for a sample needed to keep the sample
    plot_bokeh: bool = False,  # deprecated
    file_name: str | None,  # deprecated - used for plotting
):
    """
    DEPRECATED: processes data in the old format (sample_id, paired-end reads as a flat array)
    Processes and filters samples and paired-end reads to be used in clustering.
    Paired-end data is filtered out if points are 0 or nan.
    Samples are filtered out if too few paired-end reads are present (< min_pairs). Long reads require fewer reads to support an SV.
    Returns a list of points to cluster and a list of Evidence objects for the samples that passed filtering.
    """
    # list of evidence to keep after filtering
    sv_evidence = []

    # list of points to cluster in the shape [[intercept, mean_l], ...]
    points = []

    for sample_id, r in squiggle_data.items():
        reads = np.array(r.copy())  # of the format [l1, r1, l2, r2, ...]
        reads = reads[(reads != 0) & (~np.isnan(reads))]
        paired_ends = [
            [reads[i], reads[i + 1]] for i in range(0, len(reads), 2)
        ]  # reformat the reads to be [[l1, r1], [l2, r2], ...]
        if len(reads) >= min_pairs * 2:  # filter out samples with too few reads
            # before, we would filter out paired-end reads that were too far from the SV coordinates
            # however, we should be able to rely on STIX to return only relevant reads
            # taking the mean of the coordinates will average out noise in the reads
            mean_l = int(np.mean([paired_end[0] for paired_end in paired_ends]))
            mean_r = int(np.mean([paired_end[1] for paired_end in paired_ends]))
            # here, the intercept is just the length of the segment between the paired-end reads
            # subtract the insert size to get the length of the supposed deletion.
            b = (
                mean_r - mean_l - insert_size_lookup[sample_id].mean
            )  # R - L (including read length)
            sv_evidence.append(
                Evidence(
                    sample=Sample(id=sample_id),
                    svlen=b,
                    start=mean_l,
                    end=mean_r,
                    paired_ends=paired_ends,
                    mean_insert_size=insert_size_lookup[sample_id].mean,
                    insert_size_sd=insert_size_lookup[sample_id].sd,
                )
            )
            # scale this by the SV coordinates so that the points are closer together
            points.append((b - (R - L), mean_l - L))  # (length, L-coordinate)

    return np.array(points), sv_evidence


def process_data(
    reads: pd.DataFrame,
    *,
    L: int,
    R: int,
    insert_size_lookup: dict[str, InsertSizeDistribution],
    min_pairs: int = 2,  # minimum number of paired end reads for a sample needed to keep the sample
):
    """
    Processes and filters samples and paired-end reads to be used in clustering.
    Paired-end data is filtered out if points are 0 or nan.
    Samples are filtered out if too few paired-end reads are present (< min_pairs). Long reads require fewer reads to support an SV.
    Returns a list of points to cluster and a list of Evidence objects for the samples that passed filtering.
    """
    # filter out reads that don't share enough reciprocal overlap with the SV region
    # this is not being done right now because we should be able to rely on STIX to return only relevant reads
    reads["r"] = reads.apply(
        lambda row: reciprocal_overlap((row["l_end"], row["r_start"]), (L, R)),
        axis=1,
    )
    # this filtering doesn't take into account the sequenced distance between L and R
    # reads = reads[reads["r"] >= 0.5]

    # list of evidence to keep after filtering
    sv_evidence = []

    # list of points to cluster in the shape [[intercept, mean_l], ...]
    points = []

    # reads: DataFrame with columns: sample_id, left, right, type
    # group reads by sample id
    for sample_id, group in reads.groupby("sample_id"):
        if group.shape[0] < min_pairs:
            continue  # skip samples with too few reads

        # take the innermost bounds of the reads
        ls = group["l_end"].tolist()
        rs = group["r_start"].tolist()
        paired_ends = [[l, r] for l, r in zip(ls, rs)]  # noqaE741

        # duplicate rows where the type is "split" to weight them more heavily
        ls_split = group[group["type"] == "split"]["l_end"].tolist()
        ls += ls_split
        rs_split = group[group["type"] == "split"]["r_start"].tolist()
        rs += rs_split

        # we should be able to rely on STIX to return only relevant reads
        # taking the mean of the coordinates will average out noise in the reads
        med_l = int(np.median(ls))
        med_r = int(np.median(rs))

        # don't subtract insert size anymore; this is already taken into account in the read coordinates
        # TODO: are we subtracting insert size somewhere else? we don't actually need to cluster with the insert size taken into account, since the axes values are arbitrary. need to make sure plotting takes this into account though.
        svlen = med_r - med_l

        sv_evidence.append(
            Evidence(
                sample=Sample(id=sample_id),
                start=med_l,
                end=med_r,
                svlen=svlen,
                paired_ends=paired_ends,
                mean_insert_size=insert_size_lookup[sample_id].mean,
                insert_size_sd=insert_size_lookup[sample_id].sd,
            )
        )
        # scale this by the SV coordinates so that the points are closer together

        points.append((svlen - (R - L), med_l - L))  # (length, L-coordinate)

    return np.array(points), sv_evidence


def gmm_trial(
    reads: pd.DataFrame,
    *,
    chr: str,
    L: int,  # sv start
    R: int,  # sv stop
    insert_size_lookup: dict[str, InsertSizeDistribution],
    init: str = "kmeans++",
    repulsion: bool = False,
    r_threshold: float = 0.8,
    repulsion_stepsize: float = 10.0,
    model_comparison_func: str = "aic",
    min_pairs: int = 2,
    synthetic_data: bool = False,
    gmm_model: str = "2d",  # 1d_len, 1d_L, 2d
    stem: str = "1kg",
    plot: bool = True,
    plot_file: str | None = None,
    force_n_modes: int | None = None,
):
    """Runs the GMM pipeline and visualizes the results."""
    # transforms data to cluster
    points, sv_evidence = process_data(
        reads,
        L=L,
        R=R,
        insert_size_lookup=insert_size_lookup,
        min_pairs=min_pairs,
    )

    if len(points) == 0:
        # warnings.warn("No structural variants found in this region.")
        return None, []

    gmm_result = gmm(
        points,
        L=L,
        R=R,
        r_threshold=r_threshold,
        repulsion_stepsize=repulsion_stepsize,
        init=init,
        repulsion=repulsion,
        model_comparison_func=model_comparison_func,
        plot=plot,
        force_n_modes=force_n_modes,
    )

    if not synthetic_data:
        populate_sample_info(
            sv_evidence,
            chr,
            L,
            R,
            stem=stem,
        )  # mutates sv_evidence with ancestry data and homo/heterozygous for each sample

    evidence_by_mode = get_evidence_by_mode(gmm_result, sv_evidence)
    if plot:
        plot_2d_coords_fig(
            evidence_by_mode,
            plot_file,
            L=L,
            R=R,
            axis1="L",
            axis2="Length",
            add_error_bars=False,
            size_by="",
            show_mode_stats=True,
            show_1d_distributions=True,
            insert_size_lookup=insert_size_lookup,
            init="kmeans++",
            repulsion=repulsion,
        )
        # plot_single_sv(
        #     evidence_by_mode,
        #     sv_id=sv_id,
        #     L=L,
        #     R=R,
        #     axis1="L",
        #     axis2="Length",
        #     add_error_bars=False,
        #     size_by="insert_size",
        #     color_by="mode",
        # )

    return gmm_result, evidence_by_mode
