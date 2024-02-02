import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class GaussianDistribution:
    mu: List[float]  # means
    vr: List[float]  # variances
    p: List[float]  # component weights


@dataclass
class SampleData(GaussianDistribution):
    x: np.ndarray[int]


@dataclass
class EstimatedGMM(GaussianDistribution):
    num_modes: int


@dataclass
class GMM(GaussianDistribution):
    logL: float  # log-likelihood
