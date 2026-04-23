import numpy as np
from dataclasses import dataclass


@dataclass
class StixQueryRegion:
    chr: str
    left_start: int
    left_stop: int
    right_start: int
    right_stop: int
    file_name: str


@dataclass
class GaussianDistribution1D:
    mu: list[float]  # means for a 2D GMM
    vr: list[float]  # variances
    p: list[float]  # component weights


@dataclass
class GaussianDistribution2D:
    mu: list[list[float]]  # means for a 2D GMM
    cov: list[list[float]]  # covariance matrices
    p: list[float]  # component weights


@dataclass
class SampleData(GaussianDistribution1D):
    x: np.ndarray
    x_by_mu: list[list[float]]


@dataclass
class EstimatedGMM:
    num_modes: int
    logL: float | None
    aic: float | None
    outliers: list[float]
    window_size: tuple[int, int]
    x_by_mode: list[np.ndarray[int]]
    num_pruned: list[int]
    num_iterations: int


@dataclass
class EstimatedGMM1D(EstimatedGMM, GaussianDistribution1D):
    pass


@dataclass
class EstimatedGMM2D(EstimatedGMM, GaussianDistribution2D):
    pass


@dataclass
class GMM:
    logL: float  # log-likelihood


@dataclass
class GMM1D(GMM, GaussianDistribution1D):
    pass


@dataclass
class GMM2D(GMM, GaussianDistribution2D):
    pass


@dataclass
class Sample:
    id: str
    sex: str = ""
    population: str = ""
    superpopulation: str = ""
    allele: str = ""


@dataclass
class Evidence:
    sample: Sample
    svlen: float
    start: int  # median L-coordinate
    end: int
    paired_ends: list[list[float]]
    mean_insert_size: int


@dataclass
class SVStat:
    length: int
    start: int
    end: int


@dataclass
class ModeStat:
    length: int
    length_sd: float
    start: int
    start_sd: float
    end: int
    end_sd: float
    num_samples: int
    num_heterozygous: int
    num_homozygous: int
    sample_ids: list[str]
    num_pruned: int
    af: float


@dataclass
# Optional fields are for compatibility with calibration test output
class SVInfoGMM:
    id: str | None
    chr: str
    start: int
    stop: int
    svlen: int | None
    ref: str | None
    alt: str | None
    qual: str | None
    # filter: list[str]
    af: str | None
    # info: dict
    num_samples: int
    num_pruned: int  # number of samples pruned by the GMM. Samples can also be dropped if they're reference samples, don't have enough evidence, or aren't in the vcf index but aren't counted here
    num_samples_run: int
    num_reference: int
    svlen_post: int
    num_modes: int
    num_iterations: int
    overlap_between_modes: bool
    modes: list[ModeStat]
