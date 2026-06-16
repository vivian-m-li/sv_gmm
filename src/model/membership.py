import numpy as np

from src.utils.helper import stix_output_to_df
from src.utils.model_helper import giggle_format
from src.utils.types import InsertSizeDistribution


def check_group_membership(
    sv: tuple[str, int, int],
    samples: list[str],
    insert_size_lookup: dict[str, InsertSizeDistribution],
    threshold: float = 2.0,
) -> dict[str, bool]:
    """
    For each cluster, check that the sample's SV is within the allowable
    deviation from the SV represented by the cluster centroid

    Parameters:
    - sv: (chr, start, end) tuple representing the sample's SV
    - samples: list of sample IDs assigned to the cluster
    - insert_size_lookup: a dictionary mapping sample IDs to their insert size distribution (mean and standard deviation)
    - threshold: the maximum allowable deviation (measured by the z-score)

    Returns:
    - A dictionary mapping sample IDs to a boolean indicating whether the
    sample's SV can be considered a member of the cluster
    """

    reads = stix_output_to_df(
        f"{giggle_format(sv[0], sv[1])}_{giggle_format(sv[0], sv[2])}.txt"
    )

    is_member = {}
    for sample in samples:
        # calculate the z-score for the sample's SV compared to the cluster centroid
        sample_reads = reads[reads["sample_id"] == sample]
        if sample_reads.empty:
            is_member[sample] = False
            continue

        sample_sv = (
            sv[0],
            np.median(sample_reads["l_end"].values()),
            np.median(sample_reads["r_start"].values()),
        )
        z_score_start = (
            abs(sample_sv[1] - sv[1]) / insert_size_lookup[sample].sd
        )
        z_score_end = abs(sample_sv[2] - sv[2]) / insert_size_lookup[sample].sd
        is_member[sample] = (
            z_score_start <= threshold and z_score_end <= threshold
        )

    return is_member
