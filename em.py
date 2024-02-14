import sys
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from IPython.display import HTML
from gmm_types import *
from typing import Tuple, List, Dict

OUTLIER_THRESHOLD = 0.01
RESPONSIBILITY_THRESHOLD = 1e-10


def round_lst(lst):
    return [round(x, 2) for x in lst]


def print_lst(lst) -> str:
    return ", ".join([str(x) for x in round_lst(lst)])


def print_stats(logL, mu, vr, p) -> str:
    return f"logL={round(logL, 2)}, means={round_lst(mu)}, variances={round_lst(vr)}, weights={round_lst(p)}"


"""
Plotting functions
"""


def get_scatter_data(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ux = np.arange(min(x), max(x) + 0.25, 0.25)
    hx, edges = np.histogram(x, bins=ux)
    ux = (edges[:-1] + edges[1:]) / 2

    # filter out x values with 0 points
    nonzero_filter = hx > 0
    ux = ux[nonzero_filter]
    hx = hx[nonzero_filter]

    return ux, hx


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
    ux, hx = get_scatter_data(x)
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


def calc_y_vals(ux, n, gmm, i):
    return (n * gmm.p[i] / 4) * norm.pdf(ux, gmm.mu[i], np.sqrt(gmm.vr[i]))


class UpdateDist:
    def __init__(self, ax, x, gmms):
        self.ax = ax
        self.x = x
        self.gmms = gmms
        self.num_modes = len(gmms[0].mu)
        self.n = len(x)

        self.ux, self.hx = get_scatter_data(x)

        self.scatter = ax.scatter([], [], marker=".", color="blue")
        self.lines = ()
        for i in range(self.num_modes):
            (line,) = ax.plot(
                [],
                [],
                linestyle="-",
                color=cm.Set1.colors[i],
            )
            self.lines = self.lines + (line,)

    def start(self):
        self.scatter = self.ax.scatter(self.ux, self.hx, marker=".", color="blue")
        lines = ()
        for i in range(self.num_modes):
            (line,) = self.ax.plot(
                self.ux,
                calc_y_vals(self.ux, self.n, self.gmms[0], i),
                linestyle="-",
                color=cm.Set1.colors[i],
            )
            lines = lines + (line,)
        self.lines = lines
        return (self.scatter,) + self.lines

    def __call__(self, frame):
        for i, line in enumerate(self.lines):
            line.set_ydata(calc_y_vals(self.ux, self.n, self.gmms[frame], i))
        self.scatter.set_offsets(np.column_stack((self.ux, self.hx)))
        return (self.scatter,) + self.lines


def animate_distribution(x: np.ndarray, gmms: List[GMM]) -> None:
    fig, ax = plt.subplots()
    ud = UpdateDist(ax, x, gmms)
    anim = FuncAnimation(
        fig=fig, func=ud, init_func=ud.start, frames=len(gmms), interval=300, blit=True
    )

    plt.xlabel("x value")
    plt.ylabel("density")
    plt.title("animation")

    # HTML(anim.to_html5_video())
    plt.show()


def plot_likelihood(logL: np.ndarray[float]) -> None:
    plt.figure()
    plt.plot(np.arange(1, len(logL) + 1), logL, "bo-")
    plt.xlabel("iterations")
    plt.ylabel("log-likelihood")
    plt.show()


def plot_param_vs_aic(
    n: int, gmm_lookup: Dict[float, EstimatedGMM], xlabel: str
) -> None:
    cmap = cm.Set2.colors

    x_vals = []
    y_vals = []
    colors = []
    for sig, gmm in gmm_lookup.items():
        x_vals.append(sig)
        y_vals.append(gmm.aic)
        colors.append(cmap[gmm.num_modes])

    plt.figure()
    plt.scatter(x_vals, y_vals, c=colors)
    plt.title(f"n={n}")
    plt.legend(
        handles=[
            plt.Line2D([], [], color=color, label=label)
            for color, label in zip(
                [cmap[1], cmap[2], cmap[3]], ["1 mode", "2 modes", "3 modes"]
            )
        ],
        loc="lower right",
    )
    plt.xlabel(xlabel)
    plt.ylabel("aic")
    plt.show()


"""
GMM/EM helper functions
"""


def calc_log_likelihood(
    x: np.ndarray[float],
    mu: np.ndarray[float],
    vr: np.ndarray[float],
    p: np.ndarray[float],
):
    num_modes = len(mu)
    logL = 0.0
    for i in range(len(x)):
        pdf = [p[k] * norm.pdf(x[i], mu[k], np.sqrt(vr[k])) for k in range(num_modes)]
        likelihood_i = np.sum(pdf)
        logL += np.log(likelihood_i)
    return logL


def calc_aic(logL: float, num_modes: int) -> float:
    num_params = (
        num_modes * 3
    ) - 1  # mu, vr, and p as the params to predict, - 1 because the last p value is determined by the other(s)
    aic = (2 * num_params) - (2 * logL)
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
    pr: bool = True,
) -> Tuple[np.ndarray[float], np.ndarray, np.ndarray, np.ndarray]:
    assert mode_means is not None or num_modes is not None

    num_modes = len(mode_means) if mode_means is not None else num_modes
    if mode_means is not None:
        mu = mode_means
    else:
        mu = generate_means(num_modes)
    mu = sorted(mu)

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
    elif pr:
        print(title)
    return SampleData(mu, vr, p, np.array(x))


def init_em(
    x: np.ndarray,
    num_modes: int,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # initial conditions
    kmeans_data = np.ravel(x).astype(float).reshape(-1, 1)
    kmeans = KMeans(n_init=3, n_clusters=num_modes)
    kmeans.fit(kmeans_data)

    mu = np.sort(np.ravel(kmeans.cluster_centers_))  # initial means
    vr = [np.var(x)] * num_modes  # initial variances

    p = np.ones(num_modes) / num_modes  # initial p_k proportions
    logL = []  # logL values
    n = len(x)  # sample size

    # initial log-likelihood
    logL.append(calc_log_likelihood(x, mu, vr, p))

    return n, mu, vr, p, logL


def identify_outliers(x, mu, vr):
    outliers = []
    for i, x_i in enumerate(x):
        contributions = norm.pdf(x[i], mu, np.sqrt(vr))
        poss_outlier = np.any(contributions < OUTLIER_THRESHOLD / len(x))
        if poss_outlier:
            outliers.append(x_i)
    return outliers


def run_em(
    x: np.ndarray,  # data
    num_modes: int,
    plot: bool,
) -> List[GMM]:
    all_params: List[GMM] = []

    n, mu, vr, p, logL = init_em(x, num_modes)  # initialize parameters
    all_params.append(GMM(mu, vr, p, logL[0]))

    gz = np.zeros((n, len(mu)))
    num_iterations = 15
    for jj in range(1, num_iterations):
        # Expectation step: calculate the posterior probabilities
        for i in range(len(x)):
            # For each point, calculate the probability that a point is from a gaussian using the mean, standard deviation, and weight of each gaussian
            gz[i, :] = p * norm.pdf(x[i], mu, np.sqrt(vr))
            gz[i, :] /= np.sum(gz[i, :])

        # Ensure that each point contributes to the responsibility matrix above some threshold
        gz[(gz < RESPONSIBILITY_THRESHOLD) | np.isnan(gz)] = RESPONSIBILITY_THRESHOLD

        # Maximization step: estimate gaussian parameters
        # Given the probability that each point belongs to particular gaussian, calculate the mean, variance, and weight of the gaussian
        nk = np.sum(gz, axis=0)
        mu = [(1.0 / nk[j]) * np.sum(gz[:, j] * x) for j in range(num_modes)]
        vr = [
            (1.0 / nk[j]) * np.sum(gz[:, j] * (x - mu[j]) ** 2)
            for j in range(num_modes)
        ]
        p = nk / n

        # update likelihood
        logL.append(calc_log_likelihood(x, mu, vr, p))
        all_params.append(GMM(mu, vr, p, logL[-1]))

        # Convergence check
        if abs(logL[-1] - logL[-2]) < 0.05:
            break

        # if jj == num_iterations - 1:
        #     warnings.warn(
        #         "Maximum number of iterations reached without logL convergence"
        #     )

    outliers = identify_outliers(x, mu, vr)

    if plot:
        # Visualize the final model
        plot_distributions(
            x, n, mu, vr, p, title=f"Final Stats: {print_stats(logL[-1], mu, vr, p)}"
        )

    return all_params


def run_gmm(x: np.ndarray, *, plot: bool = True, pr: bool = True) -> EstimatedGMM:
    if len(x) == 0:
        warnings.warn("Input data is empty")
        return 0
    if len(x) == 1:
        warnings.warn("Input data contains one SV")
        return 1

    opt_params = None
    num_sv = 0
    aic_vals = []
    if len(x) <= 10:  # small number of SVs detected
        opt_params = run_em(x, 1, plot)
        num_sv = 1
    else:
        all_params = []
        for num_modes in range(1, 4):
            params = run_em(x, num_modes, plot)
            aic = calc_aic(params[-1].logL, num_modes)
            all_params.append(params)
            aic_vals.append(aic)
        min_aic_idx = aic_vals.index(min(aic_vals))
        opt_params = all_params[min_aic_idx]
        num_sv = len(opt_params[0].mu)

    final_params = opt_params[-1]

    if pr:
        print(
            f"\nNumber of SVs: {num_sv}\n{print_stats(final_params.logL, final_params.mu, final_params.vr, final_params.p)}"
        )

    if plot:
        pass
        # Plot the likelihood function over time
        # plot_likelihood([x.logL for x in opt_params])
        # animate_distribution(x, opt_params)

    return EstimatedGMM(
        mu=final_params.mu,
        vr=final_params.vr,
        p=final_params.p,
        num_modes=num_sv,
        aic=min(aic_vals) if len(aic_vals) > 0 else None,
    )


if __name__ == "__main__":
    x = generate_data(10000, num_modes=3).x
    run_gmm(x)
