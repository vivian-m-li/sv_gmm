import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from gmm_types import *
from typing import Tuple, List, Dict, Union, Optional

RESPONSIBILITY_THRESHOLD = 1e-10


"""
GMM/EM helper functions
"""


def calc_log_likelihood(
    x: np.ndarray,
    mu: List[np.ndarray],
    cov: List[np.ndarray],
    p: np.ndarray,
) -> float:
    """Calculates the log-likelihood of the data fitting the GMM."""
    num_modes = len(mu)
    logL = 0.0
    for i in range(len(x)):
        pdf = []
        for k in range(num_modes):
            pdf_i = None
            while pdf_i is None:
                try:
                    pdf_i = p[k] * multivariate_normal.pdf(x[i], mu[k], cov[k])
                except np.linalg.LinAlgError:
                    # if the covariance matrix is not positive definite, add a small value to the diagonal and try again
                    cov[k] += np.eye(cov[k].shape[0]) * 1e-6
            pdf.append(pdf_i)
        likelihood_i = np.sum(pdf)
        logL += np.log(likelihood_i)
    return logL


def calc_aic(
    logL: float, num_modes: int, mu: np.ndarray, cov: List[np.ndarray]
) -> float:
    """Calculates the penalized log-likelihood using AIC."""
    num_params = (
        (num_modes * 2) + (num_modes * 3) + (num_modes - 1)
    )  # number of parameters for mu + cov + p - 1
    aic = (2 * num_params) - (2 * logL)  # scale the AIC

    # compare the lengths and L coordinates of the different modes - penalize if they're within 2 SD of our sampling SD (50)
    # TODO: scale this penalty based on the number of samples, since the logL is larger for bigger sample sizes
    if num_modes > 1:
        for i in range(num_modes):
            for j in range(i + 1, num_modes):
                dist = np.linalg.norm(mu[i] - mu[j])
                # add a large penalty for mus that are too close to each other
                penalty = 0 if dist >= 141 else 100
                if dist > 70 and dist < 141:  # between 1 and 2 SDs away
                    # smaller penalty for having a larger gap between mus
                    penalty -= round(dist * 0.71)
                aic += penalty

    return aic


def init_em(
    x: np.ndarray,
    num_modes: int,
) -> Tuple[int, np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Initializes the expectation-maximization algorithm using k-means clustering on the data.
    Returns the sample size, means, variances, weights, and log likelihood of the initial GMM.
    """
    # initial conditions
    kmeans = KMeans(n_clusters=num_modes)
    kmeans.fit(x)

    mu = kmeans.cluster_centers_  # initial means
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    count_lookup = dict(zip(unique, counts))
    cov = [
        (
            np.cov(x[kmeans.labels_ == i].T)
            if count_lookup[i] > 1
            else np.eye(x.shape[1])
        )
        for i in range(num_modes)
    ]  # initial covariances, sets a default covariance if only one point in a cluster
    p = [count_lookup[i] / len(x) for i in range(num_modes)]  # initial p_k proportions
    logL = []  # logL values
    n = len(x)  # sample size

    # initial log-likelihood
    logL.append(calc_log_likelihood(x, mu, cov, p))

    return n, mu, cov, p, logL


def calc_responsibility(
    x: np.ndarray,
    n: int,
    mu: List[np.ndarray],
    cov: List[np.ndarray],
    p: np.ndarray,
):
    """Calculates the responsibility matrix for each point/mode."""
    gz = np.zeros((n, len(mu)))  # number of points by number of modes
    for i in range(len(x)):
        # For each point, calculate the probability that a point is from a gaussian using the mean, standard deviation, and weight of each gaussian
        gz[i, :] = [
            p[k] * multivariate_normal.pdf(x[i], mu[k], cov[k]) for k in range(len(mu))
        ]
        gz[i, :] /= np.sum(gz[i, :])
    return gz


def em(
    x: np.ndarray,
    num_modes: int,
    n: int,
    mu: List[np.ndarray],
    cov: List[np.ndarray],
    p: np.ndarray,
) -> GMM2D:
    """Performs one iteration of the expectation-maximization algorithm."""
    # Expectation step: calculate the posterior probabilities
    gz = calc_responsibility(x, n, mu, cov, p)

    # Ensure that each point contributes to the responsibility matrix above some threshold
    gz[(gz < RESPONSIBILITY_THRESHOLD) | np.isnan(gz)] = RESPONSIBILITY_THRESHOLD

    # Maximization step: estimate gaussian parameters
    # Given the probability that each point belongs to particular gaussian, calculate the mean, variance, and weight of the gaussian
    nk = np.sum(gz, axis=0)
    mu = [(1.0 / nk[j]) * np.sum(gz[:, j, None] * x, axis=0) for j in range(num_modes)]
    cov = [
        (1.0 / nk[j]) * np.dot((gz[:, j, None] * (x - mu[j])).T, (x - mu[j]))
        for j in range(num_modes)
    ]
    p = nk / n

    # update likelihood
    logL = calc_log_likelihood(x, mu, cov, p)
    return GMM2D(mu, cov, p, logL)


def run_em(
    x: np.ndarray,  # data
    num_modes: int,
    plot: bool = False,
) -> Tuple[List[GMM2D], int]:
    """
    Given a dataset and an estimated number of modes for the GMM, estimates the parameters for each distribution.
    The algorithm is initialized using the k-means clustering approach, then the EM algorithm is run for up to 30 iterations, or a convergence of the log-likelihood; whichever comes first.
    Returns the GMM estimated by each iteration of the EM algorithm.
    """
    all_params: List[GMM2D] = []

    n, mu, cov, p, logL = init_em(x, num_modes)  # initialize parameters
    all_params.append(GMM2D(mu, cov, p, logL[0]))

    max_iterations = 30
    i = 0
    while i < max_iterations:
        # update likelihood
        gmm = em(x, num_modes, n, mu, cov, p)
        logL.append(gmm.logL)
        all_params.append(gmm)
        mu, cov, p = gmm.mu, gmm.cov, gmm.p

        # Convergence check
        if abs(logL[-1] - logL[-2]) < 0.05:
            break

        i += 1

    return all_params, i


def assign_values_to_modes(
    x: np.ndarray,
    num_modes: int,
    mu: np.ndarray,
    cov: List[np.ndarray],
    p: np.ndarray,
) -> Tuple[List[np.ndarray], List[int]]:
    gz = calc_responsibility(x, len(x), mu, cov, p)
    assignments = np.argmax(gz, axis=1)
    x_by_mode = [[] for _ in range(num_modes)]
    for i, mode in enumerate(assignments):
        if mode is not None:
            x_by_mode[mode].append(x[i])
    x_by_mode = [np.array(data_points) for data_points in x_by_mode]
    return x_by_mode


def run_gmm(
    x: np.ndarray[Tuple[float, int]], *, plot: bool = False, pr: bool = False
) -> Optional[EstimatedGMM2D]:
    """
    Runs the GMM estimation process to determine the number of structural variants in a DNA reading frame.
    x is a 2D list of data points where each data point consists of:
      - the y-intercept of the L-R position calculated after a shift in the genome due to a deletion of greater than 1 base pair, and
      - the max L value read for each sample
    If x contains 10 or fewer data points, then 1 structural variant is estimated. If x has more than 10 data points, then the EM algorithm is then run for a 1, 2, or 3 mode GMM, and the resulting AIC scores are calculated and compared across the estimated GMMs. The GMM with the lowest AIC score is returned as the optimal fit to the data. The GMM's mu value represents the length and L coordinate of each structural variant estimated.
    """
    x = np.array(x, dtype=float)
    n = len(x)
    if len(x) == 0:
        # warnings.warn("Input data is empty")
        return None
    if len(x) == 1:
        # warnings.warn("Input data contains one SV")
        singleton = x[0]
        return EstimatedGMM2D(
            mu=[singleton],
            cov=[
                [0, 0],
                [0, 0],
            ],  # can't determine the covariance for only one pair of points
            p=[1],
            num_modes=1,
            logL=0,
            aic=0,
            outliers=[],
            window_size=(singleton, singleton),
            x_by_mode=[x],
            num_pruned=[0],
            num_iterations=0,
        )

    outliers = []
    aic_vals = []
    if len(x) <= 10:  # small number of SVs detected
        opt_params, num_iterations = run_em(x, 1, plot)
        num_iterations_final = num_iterations
        num_sv = 1
    else:
        all_params = []
        iterations = []
        for num_modes in range(1, 4):
            params, num_iterations = run_em(x, num_modes, plot)
            aic = calc_aic(params[-1].logL, num_modes, params[-1].mu, params[-1].cov)
            all_params.append(params)
            iterations.append(num_iterations)
            aic_vals.append(aic)
            # print(num_modes, params[-1].logL, aic)
        min_aic_idx = aic_vals.index(min(aic_vals))
        opt_params = all_params[min_aic_idx]
        num_iterations_final = iterations[min_aic_idx]
        num_sv = len(opt_params[0].mu)

    final_params = opt_params[-1]

    x_by_mode = assign_values_to_modes(
        x, num_sv, final_params.mu, final_params.cov, final_params.p
    )

    return EstimatedGMM2D(
        mu=final_params.mu,
        cov=final_params.cov,
        p=final_params.p,
        num_modes=num_sv,
        logL=final_params.logL,
        aic=min(aic_vals) if len(aic_vals) > 0 else None,
        outliers=outliers,
        window_size=(min(x[:, 0]), max(x[:, 0])),
        x_by_mode=x_by_mode,
        num_pruned=[0 for _ in range(num_sv)],
        num_iterations=num_iterations_final,
    )
