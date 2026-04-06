import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

CHR_LENGTHS = {  # for grch38
    "1": 248956422,
    "2": 242193529,
    "3": 198295559,
    "4": 190214555,
    "5": 181538259,
    "6": 170805979,
    "7": 159345973,
    "8": 145138636,
    "9": 138394717,
    "10": 133797422,
    "11": 135086622,
    "12": 133275309,
    "13": 114364328,
    "14": 107043718,
    "15": 101991189,
    "16": 90338345,
    "17": 83257441,
    "18": 80373285,
    "19": 58617616,
    "20": 64444167,
    "21": 46709983,
    "22": 50818468,
    # ignore X and Y chromosomes in SV analysis
    # "X": 156040895,
    # "Y": 57227415,
}
COLORS = ["#459395", "#EB7C69", "#FDA638"]
SUPERPOPULATIONS = ["AFR", "AMR", "EUR", "EAS", "SAS"]
SUBPOPULATIONS = [  # sorted by superpopulation
    "LWK",
    "YRI",
    "ESN",
    "ASW",
    "ACB",
    "MSL",
    "GWD",
    "PUR",
    "MXL",
    "CLM",
    "PEL",
    "CDX",
    "KHV",
    "CHS",
    "CHB",
    "JPT",
    "CEU",
    "TSI",
    "IBS",
    "FIN",
    "GBR",
    "ITU",
    "GIH",
    "STU",
    "BEB",
    "PJL",
]
ANCESTRY_COLORS = {
    "AFR": "#45597e",
    "AMR": "#1d6295",
    "EUR": "#a4def4",
    "EAS": "#ffbf00",
    "SAS": "#ffe69f",
}

GMM_MODELS = ["1d_len", "1d_L", "2d"]
MODEL_NAMES = ["Length-only", "L-only", "Length-L"]

GMM_AXES = {
    "L": lambda x: x[0],
    "R": lambda x: x[1],
    "Length": lambda x: x[1] - x[0],
}

SYNTHETIC_DATA_CENTROIDS = {
    "A": [[100000, 102553], [100500, 102053]],
    "B": [[100000, 102553], [100500, 103053]],
    "C": [[100000, 102553], [103053, 105606]],
    "D": [[100000, 102553], [101000, 103003], [100436, 103452]],
    "E": [[100000, 102553], [101000, 103003], [100699, 103757]],
}


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
    mean_l: int  # mean L-coordinate
    paired_ends: List[List[float]]
    mean_insert_size: int
    start_y: float = 0  # this is not used anywhere


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
# Optional fields are for compatibility with calibration test output
class SVInfoGMM:
    id: Optional[str]
    chr: str
    start: int
    stop: int
    svlen: Optional[int]
    ref: Optional[str]
    alt: Optional[str]
    qual: Optional[str]
    # filter: List[str]
    af: Optional[str]
    # info: dict
    num_samples: int
    num_pruned: int  # number of samples pruned by the GMM. Samples can also be dropped if they're reference samples, don't have enough evidence, or aren't in the vcf index but aren't counted here
    num_samples_run: int
    num_reference: int
    svlen_post: int
    num_modes: int
    num_iterations: int
    overlap_between_modes: bool
    modes: List[ModeStat]
