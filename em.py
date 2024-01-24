import sys
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from gmm_types import *
from typing import Tuple, List


def round_lst(lst):
    return [round(x, 2) for x in lst]


def print_lst(lst) -> str:
    return ", ".join([str(x) for x in round_lst(lst)])


def print_stats(logL, mu, vr, p) -> str:
    return f"logL={round(logL, 2)}, means={round_lst(mu)}, variances={round_lst(vr)}, weights={round_lst(p)}"


def plot_distributions(
    x: np.ndarray[float],
    n: int,
    mu: np.ndarray[float],
    vr: np.ndarray[float],
    p: np.ndarray[float],
    *,
    title: str = None,
) -> None:
    # visualize the data and the generative model
    ux = np.arange(min(x), max(x) + 0.25, 0.25)
    hx, edges = np.histogram(x, bins=ux)
    ux = (edges[:-1] + edges[1:]) / 2

    # filter out x values with 0 points
    nonzero_filter = hx > 0
    ux = ux[nonzero_filter]
    hx = hx[nonzero_filter]

    plt.figure()
    plt.scatter(ux, hx, marker=".", color="blue")
    for i in range(len(mu)):
        plt.plot(
            ux,
            (n * p[i] / 4) * norm.pdf(ux, mu[i], np.sqrt(vr[i])),
            linestyle="-",
            color=cm.Set1.colors[i],
        )
    if title is not None:
        plt.title(title)
    plt.xlabel("x value")
    plt.ylabel("density")
    plt.show()


def plot_likelihood(logL: np.ndarray[float]) -> None:
    plt.figure()
    plt.plot(np.arange(1, len(logL) + 1), logL, "bo-")
    plt.xlabel("iterations")
    plt.ylabel("log-likelihood")
    plt.show()


def calc_log_likelihood(
    x: np.ndarray[float],
    mu: np.ndarray[float],
    vr: np.ndarray[float],
    p: np.ndarray[float],
):
    Nxi = np.column_stack(
        [norm.pdf(x, mu[i], np.sqrt(vr[i])) for i in range(len(mu))]
    )  # TODO: check this
    # Nxi2 = np.where(Nxi == 0, sys.float_info.min, Nxi)
    logL = np.sum(np.log(p * Nxi))
    if np.isnan(logL):
        # TODO log likelihood sometimes nan
        import pdb

        pdb.set_trace()
    return logL


def calc_aic(logL: float, num_modes: int) -> float:
    num_params = (
        num_modes * 3
    ) - 1  # mu, vr, and p as the params to predict, - 1 because the last p value is determined by the other(s)
    aic = (2 * num_params) - (2 * logL)
    # TODO: still not calculating correctly
    print(f"logL: {round(logL, 2)}, num_modes: {num_modes}, aic: {round(aic, 2)}")
    return aic


"""
Generates component weights for 1, 2, or 3 mode Gaussian distributions.
Weights are constrained between 0.1 and 0.9 if more than one mode.
"""


def generate_weights(num_modes: int) -> np.ndarray[float]:
    if num_modes == 1:
        return np.array([1.0])

    while True:
        random_numbers = [random.uniform(0.1, 0.9) for _ in range(num_modes - 1)]
        last_num = 1 - sum(random_numbers)
        if last_num >= 0.1:
            random_numbers.append(last_num)
            return np.array(random_numbers)


def generate_means(num_modes: int) -> np.ndarray[int]:
    while True:
        mu = np.array([random.randint(0, 100) for _ in range(num_modes)])
        if len(mu) == len(set(mu)):
            return mu


"""
Generates sample data according to a 1, 2, or 3 mode Gaussian distribution.
"""


def generate_data(
    n: int,  # sample size
    *,
    mode_means: np.ndarray[float] = None,
    mode_variances: np.ndarray[float] = None,
    weights: np.ndarray[float] = None,
    num_modes: int = None,
    plot: bool = False,
) -> np.ndarray[float]:
    assert mode_means is not None or num_modes is not None

    num_modes = len(mode_means) if mode_means is not None else num_modes
    if mode_means is not None:
        mu = mode_means
    else:
        mu = generate_means(num_modes)

    if mode_variances is not None:
        vr = mode_variances
    else:
        vr = np.array([random.randint(1, 5) for _ in range(num_modes)])

    p = np.array([1 / num_modes] * num_modes) if weights is None else weights
    if weights is not None:
        p = weights
    else:
        p = generate_weights(num_modes)

    nk = (p * n).astype(int)

    x = []
    for i in range(num_modes):
        x.extend(mu[i] + np.random.randn(nk[i]) * np.sqrt(vr[i]))

    title = f"Means = {print_lst(mu)}, Vars = {print_lst(vr)}, Weights = {print_lst(p)}"
    if plot:
        plot_distributions(
            x,
            n,
            mu,
            vr,
            p,
            title=title,
        )
    else:
        print(title)
    return np.array(x)


def init_em(
    x: np.ndarray,
    num_modes: int,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # initial conditions
    # if num_modes == 1:
    #     mu = [np.median(x)]
    # elif num_modes == 2:
    #     mu = [np.percentile(x, 25), np.percentile(x, 75)]
    # else:
    #     mu = [min(x), np.median(x), max(x)]  # initial means
    # vr = [1] * num_modes  # initial variances

    kmeans_data = np.ravel(x).astype(float).reshape(-1, 1)
    kmeans = KMeans(n_init=3, n_clusters=num_modes)
    kmeans.fit(kmeans_data)

    mu = np.sort(np.ravel(kmeans.cluster_centers_))  # initial means
    vr = [np.var(x)] * num_modes  # initial variances

    # logL: -828039.12, num_modes: 3, aic: 1656094.23

    p = np.ones(num_modes) / num_modes  # initial p_k proportions
    logL = []  # logL values
    n = len(x)  # sample size

    # initial log-likelihood
    logL.append(calc_log_likelihood(x, mu, vr, p))

    # plot_distributions(
    #     x, n, mu, vr, p, title=f"Initial Stats: {print_stats(logL[0], mu, vr, p)}"
    # )
    return n, mu, vr, p, logL


def run_em(
    x: np.ndarray,  # data
    num_modes: int,
) -> List[GaussianDistribution]:
    all_params: List[GaussianDistribution] = []

    n, mu, vr, p, logL = init_em(x, num_modes)  # initialize parameters
    all_params.append(GaussianDistribution(mu, vr, p, logL[0]))

    gz = np.zeros((n, len(mu)))
    num_iterations = 20
    for jj in range(1, num_iterations):
        # Expectation step: calculate the posterior probabilities
        for i in range(len(x)):
            # for each point, calculate the probability that a point is from a gaussian using the mean, standard deviation, and weight of each gaussian
            gz[i, :] = p * norm.pdf(x[i], mu, np.sqrt(vr))
            gz[i, :] /= np.sum(gz[i, :])

        # Maximization step: estimate gaussian parameters
        # given the probability that each point belongs to particular gaussian, calculate the mean, variance, and weight of the gaussian
        nk = np.sum(gz, axis=0)
        mu = [(1.0 / nk[j]) * np.sum(gz[:, j] * x) for j in range(num_modes)]
        vr = [
            (1.0 / nk[j]) * np.sum(gz[:, j] * (x - mu[j]) ** 2)
            for j in range(num_modes)
        ]
        p = nk / n

        # update likelihood
        logL.append(calc_log_likelihood(x, mu, vr, p))
        all_params.append(GaussianDistribution(mu, vr, p, logL[-1]))
        if np.isnan(logL[-1]):
            raise Exception("logL is nan")

        # plot_distributions(
        #     x,
        #     n,
        #     mu,
        #     vr,
        #     p,
        #     title=f"Iteration {jj + 1}: {print_stats(logL[-1], mu, vr, p)}",
        # )

        # Convergence check
        if abs(logL[-1] - logL[-2]) < 0.05:
            print(f"Converged at {jj} iterations")
            break
        if jj == num_iterations - 1:
            warnings.warn(
                "Maximum number of iterations reached without logL convergence"
            )

    # Plot the likelihood function over time
    plot_likelihood(logL)

    # Visualize the final model
    plot_distributions(
        x, n, mu, vr, p, title=f"Final Stats: {print_stats(logL[-1], mu, vr, p)}"
    )

    return all_params


def run_gmm(x: np.ndarray) -> None:
    if len(x) == 0:
        warnings.warn("Input data is empty")
        return
    if len(x) == 1:
        warnings.warn("Input data contains one SV")
        return

    opt_params = None
    num_sv = 0
    if len(x) <= 10:  # TODO: "very small integer"
        opt_params = run_em(x, 1)
        num_sv = 1
    else:
        aic_vals = []
        all_params = []
        for num_modes in range(1, 4):
            params = run_em(x, num_modes)
            aic = calc_aic(params[-1].logL, num_modes)
            all_params.append(params)
            aic_vals.append(aic)

            min_aic_idx = aic_vals.index(min(aic_vals))
            opt_params = all_params[min_aic_idx]
            num_sv = len(opt_params[0].mu)

    final_params = opt_params[-1]
    print(
        f"\nNumber of SVs: {num_sv}\n{print_stats(final_params.logL, final_params.mu, final_params.vr, final_params.p)}"
    )
