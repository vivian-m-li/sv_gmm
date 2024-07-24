import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


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
class Evidence:
    intercept: float
    paired_ends: List[List[float]]
    start_y: float = 0
