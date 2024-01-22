import sys
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm
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
    Nxi = np.column_stack([norm.pdf(x, mu[i], np.sqrt(vr[i])) for i in range(len(mu))])
    Nxi2 = np.where(Nxi == 0, sys.float_info.min, Nxi)
    logL = np.sum(np.log(p * Nxi2))
    if np.isnan(logL):
        # TODO log likelihood sometimes nan
        import pdb

        pdb.set_trace()
    return logL


def calc_aic(logL: float, num_modes: int) -> float:
    num_params = (
        num_modes * 3
    ) - 1  # mu, vr, and p as the params to predict, - 1 because the last p value is determined by the other(s)
    aic = -2 * logL + 2 * num_params
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
    if num_modes == 1:
        mu = [np.median(x)]
    elif num_modes == 2:
        mu = [np.percentile(x, 25), np.percentile(x, 75)]
    else:
        mu = [min(x), np.median(x), max(x)]  # initial means
    vr = [1] * num_modes  # initial variances
    p = np.ones(num_modes) / num_modes  # initial p_k proportions
    logL = np.zeros(10)  # logL values
    n = len(x)  # sample size

    # initial log-likelihood
    logL[0] = calc_log_likelihood(x, mu, vr, p)

    print(f"Initial Stats: {print_stats(logL[0], mu, vr, p)}")
    return n, mu, vr, p, logL


def run_em(
    x: np.ndarray,  # data
    num_modes: int,
) -> List[GaussianDistribution]:
    all_params: List[GaussianDistribution] = []

    n, mu, vr, p, logL = init_em(x, num_modes)  # initialize parameters
    all_params.append(GaussianDistribution(mu, vr, p, logL[0]))
    # plot_distributions(x, n, mu, vr, p)

    gz = np.zeros((n, len(mu)))
    for jj in range(1, len(logL)):  # len(logL) is number of iterations
        # E-step: calculate the posterior probabilities
        for i in range(len(x)):
            # for each point, calculate the probability that a point is from a gaussian using the mean, standard deviation, and weight of each gaussian
            gz[i, :] = p * norm.pdf(x[i], mu, np.sqrt(vr))
            gz[i, :] /= np.sum(gz[i, :])

        # M-step: estimate gaussian parameters
        # given the probability that each point belongs to particular gaussian, calculate the mean, variance, and weight of the gaussian
        nk = np.sum(gz, axis=0)
        mu = [(1.0 / nk[j]) * np.sum(gz[:, j] * x) for j in range(num_modes)]
        vr = [
            (1.0 / nk[j]) * np.sum(gz[:, j] * (x - mu[j]) ** 2)
            for j in range(num_modes)
        ]
        p = nk / n

        # update likelihood
        logL[jj] = calc_log_likelihood(x, mu, vr, p)
        all_params.append(GaussianDistribution(mu, vr, p, logL[jj]))
        if np.isnan(logL[jj]):
            raise Exception("logL is nan")

        # print(f"Iteration {jj}: {print_stats(logL[jj], mu, vr, p)}")
        # plot_distributions(x, n, mu, vr, p)

        # Convergence check
        if (logL[jj] - logL[jj - 1]) < 0.05:
            break
        if jj == len(logL) - 1:
            warnings.warn(
                "Maximum number of iterations reached without logL convergence"
            )

    # Plot the likelihood function over time
    plot_likelihood(logL)

    # Visualize the final model
    plot_distributions(x, n, mu, vr, p)

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
        max_logL = -sys.maxsize - 1
        for num_modes in range(1, 4):
            params = run_em(x, num_modes)
            aic = calc_aic(params[-1].logL, num_modes)
            print(num_modes, aic)
            if aic > max_logL:  # TODO: calculate penalized likelihood
                opt_params = params
                num_sv = num_modes

    final_params = opt_params[-1]
    print(
        f"Number of SVs: {num_sv}\n{print_stats(final_params.logL, final_params.mu, final_params.vr, final_params.p)}"
    )
