import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from gmm_types import *
from typing import Tuple, List, Dict, Union, Optional

RESPONSIBILITY_THRESHOLD = 1e-10


"""
Plotting functions
"""


def get_scatter_data(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fits the data points to a histogram and returns the formatted data to be plotted as a scatter plot."""
    ux = np.arange(min(x), max(x) + 0.25, 0.25)
    hx, edges = np.histogram(x, bins=ux)
    ux = (edges[:-1] + edges[1:]) / 2

    # filter out x values with 0 points
    nonzero_filter = hx > 0
    ux = ux[nonzero_filter]
    hx = hx[nonzero_filter]

    return ux, hx


def plot_distributions(
    x: np.ndarray[int],
    n: int,
    mu: np.ndarray[float],
    cov: List[np.ndarray[float]],
    p: np.ndarray[float],
    *,
    title: str = None,
    outliers: List[float] = [],
) -> None:
    """Plots the GMM."""
    # visualize the data and the generative model
    ux, hx = get_scatter_data(x)
    plt.figure(figsize=(15, 8))
    plt.scatter(ux, hx, marker=".", color="blue")
    for i in range(len(mu)):
        plt.plot(
            ux,
            2
            * (n * p[i])
            * norm.pdf(ux, mu[i], np.sqrt(cov[i])),  # TODO: use covariance matrix
            linestyle="-",
            color=COLORS[i],
        )
    if outliers is not None:
        plt.scatter(outliers, len(outliers) * [1], marker=".", color="blue")
    if title is not None:
        plt.title(title)
    plt.xlabel("intercept")
    plt.ylabel("density")
    plt.show()


"""
GMM/EM helper functions
"""


def calc_log_likelihood(
    x: np.ndarray[int],
    mu: np.ndarray[float],
    cov: List[np.ndarray[float]],
    p: np.ndarray[float],
) -> float:
    """Calculates the log-likelihood of the data fitting the GMM."""
    num_modes = len(mu)
    logL = 0.0
    for i in range(len(x)):
        pdf = [
            p[k] * multivariate_normal.pdf(x[i], mu[k], cov[k])
            for k in range(num_modes)
        ]
        likelihood_i = np.sum(pdf)
        logL += np.log(likelihood_i)
    return logL


def calc_aic(
    logL: float, num_modes: int, mu: np.ndarray[float], cov: List[np.ndarray[float]]
) -> float:
    """Calculates the penalized log-likelihood using AIC."""
    num_params = (
        num_modes * 3
    ) - 1  # mu, cov, and p as the params to predict, - 1 because the last p value is determined by the other(s)
    aic = (5 * num_params) - (2 * logL)  # scale the AIC

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
    kmeans = KMeans(n_init=1, n_clusters=num_modes)
    kmeans.fit(x)

    mu = np.sort(np.ravel(kmeans.cluster_centers_))  # initial means
    cov = [np.cov(x.T)] * num_modes  # intial covariance matrices
    p = np.ones(num_modes) / num_modes  # initial p_k proportions
    logL = []  # logL values
    n = len(x)  # sample size

    # initial log-likelihood
    logL.append(calc_log_likelihood(x, mu, cov, p))

    return n, mu, cov, p, logL


def calc_responsibility(
    x: np.ndarray[int],
    n: int,
    mu: np.ndarray[float],
    cov: List[np.ndarray[float]],
    p: np.ndarray[float],
):
    """Calculates the responsibility matrix for each point/mode."""
    gz = np.zeros((n, len(mu)))
    for i in range(len(x)):
        # For each point, calculate the probability that a point is from a gaussian using the mean, standard deviation, and weight of each gaussian
        gz[i, :] = [
            p[k] * multivariate_normal.pdf(x[i], mu[k], cov[k]) for k in range(len(mu))
        ]
        gz[i, :] /= np.sum(gz[i, :])
    return gz


def em(
    x: np.ndarray[int],
    num_modes: int,
    n: int,
    mu: np.ndarray[float],
    cov: List[np.ndarray[float]],
    p: np.ndarray[float],
) -> GMM:
    """Performs one iteration of the expectation-maximization algorithm."""
    # Expectation step: calculate the posterior probabilities
    gz = calc_responsibility(x, n, mu, cov, p)

    # Ensure that each point contributes to the responsibility matrix above some threshold
    gz[(gz < RESPONSIBILITY_THRESHOLD) | np.isnan(gz)] = RESPONSIBILITY_THRESHOLD

    # Maximization step: estimate gaussian parameters
    # Given the probability that each point belongs to particular gaussian, calculate the mean, variance, and weight of the gaussian
    nk = np.sum(gz, axis=0)
    mu = [(1.0 / nk[j]) * np.sum(gz[:, j, None] * x) for j in range(num_modes)]
    cov = [
        (1.0 / nk[j]) * np.dot((gz[:, j, None] * (x - mu[j])).T, axis=0)
        for j in range(num_modes)
    ]
    p = nk / n

    # update likelihood
    logL = calc_log_likelihood(x, mu, cov, p)
    return GMM(mu, cov, p, logL)


def run_em(
    x: np.ndarray[int],  # data
    num_modes: int,
    plot: bool = False,
) -> Tuple[List[GMM], int]:
    """
    Given a dataset and an estimated number of modes for the GMM, estimates the parameters for each distribution.
    The algorithm is initialized using the k-means clustering approach, then the EM algorithm is run for up to 30 iterations, or a convergence of the log-likelihood; whichever comes first.
    Returns the GMM estimated by each iteration of the EM algorithm.
    """
    all_params: List[GMM] = []

    n, mu, cov, p, logL = init_em(x, num_modes)  # initialize parameters
    all_params.append(GMM(mu, cov, p, logL[0]))

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
    x: np.ndarray[int],
    num_modes: int,
    mu: np.ndarray[float],
    cov: List[np.ndarray[float]],
    p: np.ndarray[float],
) -> Tuple[List[np.ndarray[int]], List[int]]:
    gz = calc_responsibility(x, len(x), mu, cov, p)
    assignments = np.argmax(gz, axis=1)
    x_by_mode = [[] for _ in range(num_modes)]
    for i, mode in enumerate(assignments):
        if mode is not None:
            x_by_mode[mode].append(x[i])
    x_by_mode = [np.array(data_points) for data_points in x_by_mode]
    return x_by_mode


def run_gmm(
    x: np.ndarray, *, plot: bool = False, pr: bool = False
) -> Optional[EstimatedGMM]:
    """
    Runs the GMM estimation process to determine the number of structural variants in a DNA reading frame.
    x is a list of data points corresponding to the y-intercept of the L-R position calculated after a shift in the genome due to a deletion of greater than 1 base pair.
    If x contains 10 or fewer data points, then 1 structural variant is estimated. If x has more than 10 data points, then outliers are first identified, and the reading frame is resized to exclude these outliers. The EM algorithm is then run for a 1, 2, or 3 mode GMM, and the resulting AIC scores are calculated and compared across the estimated GMMs. The GMM with the lowest AIC score is returned as the optimal fit to the data.
    """
    x = np.array(x, dtype=float)
    n = len(x)
    if len(x) == 0:
        # warnings.warn("Input data is empty")
        return None
    if len(x) == 1:
        # warnings.warn("Input data contains one SV")
        singleton = x[0]
        return EstimatedGMM(
            mu=[singleton],
            cov=[0],  # TODO: determine this value
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
        min_aic_idx = aic_vals.index(min(aic_vals))
        opt_params = all_params[min_aic_idx]
        num_iterations_final = iterations[min_aic_idx]
        num_sv = len(opt_params[0].mu)

    final_params = opt_params[-1]

    x_by_mode = assign_values_to_modes(
        x, num_sv, final_params.mu, final_params.cov, final_params.p
    )

    return EstimatedGMM(
        mu=final_params.mu,
        cov=final_params.cov,
        p=final_params.p,
        num_modes=num_sv,
        logL=final_params.logL,
        aic=min(aic_vals) if len(aic_vals) > 0 else None,
        outliers=outliers,
        window_size=(min(x), max(x)),
        x_by_mode=x_by_mode,
        num_pruned=0,
        num_iterations=num_iterations_final,
    )
