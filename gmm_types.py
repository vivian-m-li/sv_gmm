import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

COLORS = ["#459395", "#EB7C69", "#FDA638"]
ANCESTRY_COLORS = {
    "EUR": "#f2cc8f",
    "EAS": "#81b29a",
    "AFR": "#3d405b",
    "AMR": "#e07a5f",
    "SAS": "#f4f1de",
}


@dataclass
class GaussianDistribution:
    mu: List[float]  # means
    vr: List[float]  # variances
    p: List[float]  # component weights


@dataclass
class SampleData(GaussianDistribution):
    x: np.ndarray
    x_by_mu: List[List[float]]


@dataclass
class EstimatedGMM(GaussianDistribution):
    num_modes: int
    logL: Optional[float]
    aic: Optional[float]
    outliers: List[float]
    percent_data_removed: float
    window_size: Tuple[int, int]
    x_by_mode: List[np.ndarray[int]]


@dataclass
class GMM(GaussianDistribution):
    logL: float  # log-likelihood


@dataclass
class Sample:
    id: str
    sex: str = ""
    population: str = ""
    superpopulation: str = ""


@dataclass
class Evidence:
    sample: Sample
    intercept: float
    paired_ends: List[List[float]]
    start_y: float = 0


@dataclass
class SVStat:
    length: int
    start: int
    end: int
