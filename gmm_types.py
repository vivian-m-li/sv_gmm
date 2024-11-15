import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

CHR_LENGTHS = {
    "1": 249250621,
    "2": 243199373,
    "3": 198022430,
    "4": 191154276,
    "5": 180915260,
    "6": 171115067,
    "7": 159138663,
    "8": 146364022,
    "9": 141213431,
    "10": 135534747,
    "11": 135006516,
    "12": 133851895,
    "13": 115169878,
    "14": 107349540,
    "15": 102531392,
    "16": 90354753,
    "17": 81195210,
    "18": 78077248,
    "19": 59128983,
    "20": 63025520,
    "21": 48129895,
    "22": 51304566,
    # ignore X and Y chromosomes in SV analysis
    # "X": 155270560,
    # "Y": 59373566,
}
COLORS = ["#459395", "#EB7C69", "#FDA638"]
ANCESTRY_COLORS = {
    "AFR": "#45597e",
    "AMR": "#1d6295",
    "EUR": "#a4def4",
    "EAS": "#ffbf00",
    "SAS": "#ffe69f",
}

GMM_MODELS = ["1d_len", "1d_L", "2d"]


@dataclass
class GaussianDistribution1D:
    mu: List[float]  # means for a 2D GMM
    vr: List[float]  # variances
    p: List[float]  # component weights


@dataclass
class GaussianDistribution2D:
    mu: List[List[float]]  # means for a 2D GMM
    cov: List[List[float]]  # covariance matrices
    p: List[float]  # component weights


@dataclass
class SampleData(GaussianDistribution1D):
    x: np.ndarray
    x_by_mu: List[List[float]]


@dataclass
class EstimatedGMM:
    num_modes: int
    logL: Optional[float]
    aic: Optional[float]
    outliers: List[float]
    window_size: Tuple[int, int]
    x_by_mode: List[np.ndarray[int]]
    num_pruned: List[int]
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
    intercept: float
    mean_l: int
    paired_ends: List[List[float]]
    removed: int  # 0 if not removed, 1 if removed due to 1 paired end only, 2 if removed due to 2 paired ends, 3 if removed due to deviation from y=x+b line
    start_y: float = 0


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
    sample_ids: List[str]
    num_pruned: int
    af: float


@dataclass
class SVInfoGMM:
    id: str
    chr: str
    start: int
    stop: int
    svlen: int
    ref: str
    alt: str
    qual: str
    # filter: List[str]
    af: str
    # info: dict
    num_samples: int
    num_pruned: int  # number of samples pruned by the GMM. Samples can also be dropped if they're reference samples, don't have enough evidence, or aren't in the vcf index but aren't counted here
    num_reference: int
    svlen_post: int
    num_modes: int
    num_iterations: int
    overlap_between_modes: bool
    modes: List[ModeStat]
