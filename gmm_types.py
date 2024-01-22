from dataclasses import dataclass
from typing import List


@dataclass
class GaussianDistribution:
    mu: List[float]  # estimated means
    vr: List[float]  # estimated variances
    p: List[float]  # estimated component weights
    logL: float  # log-likelihood
