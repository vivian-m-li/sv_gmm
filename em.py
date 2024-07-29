import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from collections import Counter
from pprint import pprint
from gmm_types import *
from typing import Tuple, List, Dict, Union

OUTLIER_THRESHOLD = 0.001
RESPONSIBILITY_THRESHOLD = 1e-10
MODE_WEIGHT_THRESHOLD = 0.05


def round_lst(lst: List[float]) -> List[float]:
    """Rounds the items of the list to 2 decimal places."""
    return [round(x, 2) for x in lst]


def print_lst(lst: List[Union[str, float]]) -> str:
    """Formats and prints a list of items."""
    return ", ".join([str(x) for x in round_lst(lst)])


def print_stats(logL: float, mu: List[float], vr: List[float], p: List[float]) -> str:
    """Formats and prints the parameters for a GMM."""
    return f"logL={round(logL, 2)}, means={round_lst(mu)}, variances={round_lst(vr)}, weights={round_lst(p)}"


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
    vr: np.ndarray[float],
    p: np.ndarray[float],
    *,
    title: str = None,
    outliers: List[float] = [],
) -> None:
    """Plots the GMM."""
    # visualize the data and the generative model
    ux, hx = get_scatter_data(x)
    plt.figure()
    plt.scatter(ux, hx, marker=".", color="blue")
    for i in range(len(mu)):
        plt.plot(
            ux,
            2 * (n * p[i]) * norm.pdf(ux, mu[i], np.sqrt(vr[i])),
            linestyle="-",
            color=cm.Set1.colors[i],
        )
    if outliers is not None:
        plt.scatter(outliers, len(outliers) * [1], marker=".", color="blue")
    if title is not None:
        plt.title(title)
    plt.xlabel("intercept")
    plt.ylabel("density")
    plt.show()


def calc_y_vals(
    ux: np.ndarray,
    n: int,
    gmm: GMM,
    i: int,
) -> List[float]:
    """Calculates the density for each x value in the GMM."""
    return 2 * (n * gmm.p[i]) * norm.pdf(ux, gmm.mu[i], np.sqrt(gmm.vr[i]))


class UpdateDist:
    """Determines the plots to be drawn at each frame of the animation."""

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
        """Initializes the animation."""
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
        """Updates each frame of the animation."""
        for i, line in enumerate(self.lines):
            line.set_ydata(calc_y_vals(self.ux, self.n, self.gmms[frame], i))
        self.scatter.set_offsets(np.column_stack((self.ux, self.hx)))
        return (self.scatter,) + self.lines


def animate_distribution(x: np.ndarray[int], gmms: List[GMM]) -> None:
    """
    Animates the change in the GMM over each iteration of the EM algorithm.
    Note: This function does not produce an animation in Jupyter Notebook.
    """
    fig, ax = plt.subplots()
    ud = UpdateDist(ax, x, gmms)
    anim = FuncAnimation(
        fig=fig, func=ud, init_func=ud.start, frames=len(gmms), interval=300, blit=True
    )

    plt.xlabel("intercept")
    plt.ylabel("density")
    plt.show()


def plot_likelihood(logL: np.ndarray[float]) -> None:
    """Plots the log likelihood of the GMM over each iteration of the EM algorithm."""
    plt.figure()
    plt.plot(np.arange(1, len(logL) + 1), logL, "bo-")
    plt.xlabel("iterations")
    plt.ylabel("log-likelihood")
    plt.show()


def plot_param_vs_aic(
    n: int, gmm_lookup: Dict[float, EstimatedGMM], xlabel: str, logL: bool = False
) -> None:
    """Plots the aic or log likelihood value when comparing different GMMs."""
    cmap = cm.Set2.colors
    x_vals = []
    y_vals = []
    colors = []
    for sig, gmm in gmm_lookup.items():
        x_vals.append(sig)
        y_vals.append(gmm.logL if logL else gmm.aic)
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
    plt.ylabel(f"{'logL' if logL else 'aic'}")
    plt.show()


def plot_mae(
    n: int, mae: List[Tuple[int, float]], *, labels: List[str], legend_title: str
) -> None:
    """Plots the mean average error as the sample size increases."""
    plt.figure()
    for i, avg_mae in enumerate(mae):
        plt.loglog(
            n,
            avg_mae,
            color=cm.Set1.colors[i],
            label=labels[i],
        )

    plt.title("Mean Average Error")
    plt.legend(title=legend_title)
    plt.xlabel("n")
    plt.ylabel("MAE")
    plt.show()


def plot_auc(
    n: int, auc: List[Tuple[int, float]], *, labels: List[str], legend_title: str
) -> None:
    """Plots the area under the curve of 2 Gaussian distributions as the sample size increases."""
    plt.figure()
    for i, avg_auc in enumerate(auc):
        plt.plot(
            n,
            avg_auc,
            color=cm.Set1.colors[i],
            label=labels[i],
        )

    plt.title("AUC as Sample Size Increases")
    plt.legend(title=legend_title)
    plt.xscale("log")
    plt.xlabel("log n")
    plt.ylabel("AUC")
    plt.show()


def plot_gmm_accuracy(num_modes_estimated: List[int], num_modes_expected: int) -> None:
    """Plots the accuracy of detecting the correct number of modes for multiple trials of the GMM-estimation function."""
    counts = Counter(num_modes_estimated)
    accuracy = (counts[num_modes_expected] / len(num_modes_estimated)) * 100

    plt.figure()
    plt.hist(num_modes_estimated, bins=[0.5, 1.5, 2.5, 3.5], rwidth=0.8, align="mid")
    plt.title(
        f"{round(accuracy, 2)}% Accuracy For Estimating {num_modes_expected} Mode(s)"
    )
    plt.xlabel("Number of Modes")
    plt.ylabel("Count")
    plt.show()


def plot_fitted_lines(y_intercepts: List[float]):
    """Plots the L-R coordinate of each structural variant given their y-intercept."""
    xrange = [1000000, 1000200]
    plt.figure(figsize=(10, 8))
    for b in y_intercepts:
        y_values = [x + b for x in xrange]
        plt.plot(xrange, y_values, "r-")
    plt.plot(xrange, xrange, "k--", linewidth=5)
    plt.ylabel("Forward position", fontsize=16)
    plt.xlabel("Reverse position", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(xrange)
    plt.show()


def plot_clusters(x: np.ndarray[int], kmeans_labels: List[int]):
    """Plots and colors each data point according to the k-means cluster they're inferred to belong in."""
    labels = set(kmeans_labels)
    color_lookup = {label: cm.Set1.colors[i] for i, label in enumerate(labels)}
    colors = [color_lookup[label] for label in kmeans_labels]
    plt.figure(figsize=(8, 6))
    plt.scatter(list(range(len(x))), x, c=colors)
    plt.xlabel("index")
    plt.ylabel("x value")
    plt.show()


"""
GMM/EM helper functions
"""


def calc_log_likelihood(
    x: np.ndarray[int],
    mu: np.ndarray[float],
    vr: np.ndarray[float],
    p: np.ndarray[float],
) -> float:
    """Calculates the log-likelihood of the data fitting the GMM."""
    num_modes = len(mu)
    logL = 0.0
    for i in range(len(x)):
        pdf = [p[k] * norm.pdf(x[i], mu[k], np.sqrt(vr[k])) for k in range(num_modes)]
        likelihood_i = np.sum(pdf)
        logL += np.log(likelihood_i)
    return logL


def calc_aic(
    logL: float, num_modes: int, mu: np.ndarray[float], vr: np.ndarray[float]
) -> float:
    """Calculates the penalized log-likelihood using AIC."""
    num_params = (
        num_modes * 3
    ) - 1  # mu, vr, and p as the params to predict, - 1 because the last p value is determined by the other(s)
    aic = (2 * num_params) - (2 * logL)

    # Penalize model for any modes that have std > 25
    high_vr_penalty = sum([10 for variance in vr if variance > 625])
    aic += high_vr_penalty

    # Penalize model if 2+ modes have too much overlap
    overlap_threshold = 1 / 3
    overlap_penalty = 0
    for i in range(num_modes):
        for j in range(i + 1, num_modes):
            mean_diff = abs(mu[i] - mu[j])
            avg_var = (vr[i] + vr[j]) / 2
            if mean_diff < overlap_threshold * 2 * np.sqrt(avg_var):
                overlap_penalty += 10
    aic += overlap_penalty

    return aic


def calc_auc(data: SampleData) -> float:
    """Calculates the area under the curve representing the overlap between the data generated by 2 Gaussian distributions."""
    assert len(data.mu) == 2  # can only calculate auc if there are 2 modes
    larger_mu_idx = data.mu.index(max(data.mu))
    smaller_mu_idx = 0 if larger_mu_idx == 1 else 1
    labels = np.concatenate(
        [
            np.zeros(len(data.x_by_mu[smaller_mu_idx])),
            np.ones(len(data.x_by_mu[larger_mu_idx])),
        ]
    )
    auc_score = roc_auc_score(labels, data.x)
    return auc_score


def match_mus(list1: List[float], list2: List[float]) -> List[Tuple[float, float]]:
    """
    Matches the items in list1 and list2 based on the closeness of each value.
    The length of list2 must be greater than or equal to the length of list1.
    """
    # match items in list1 and list2, where len(list2) >= len(list1)
    list1 = sorted(list1)
    list2 = sorted(list2)
    list1.extend(
        [list1[-1]] * (len(list2) - len(list1))
    )  # fill in missing values in list1

    pairs: List[Tuple[float, float]] = []
    for mu1, mu2 in zip(list1, list2):
        pairs.append((mu1, mu2))
    return pairs


def calc_mae(true_mus: List[float], sample_mus: List[float]) -> float:
    """Calculates the mean average error of the actual GMM compared to the estimated GMM."""
    if len(true_mus) > len(sample_mus):
        matched_mus = match_mus(sample_mus, true_mus)
    else:
        matched_mus = match_mus(true_mus, sample_mus)

    mae = []
    for mu1, mu2 in matched_mus:
        mae.append(abs(mu1 - mu2))
    return np.mean(mae)


def generate_weights(num_modes: int) -> np.ndarray[float]:
    """
    Generates component weights for 1, 2, or 3 modes in a GMM.
    Weights are constrained between 0.1 and 0.9 if more than one mode.
    """
    if num_modes == 1:
        return np.array([1.0])

    while True:
        upper_weight = 1 - (num_modes * 0.05)
        random_numbers = [
            random.uniform(0.05, upper_weight) for _ in range(num_modes - 1)
        ]
        last_num = 1 - sum(random_numbers)
        if last_num >= 0.05:
            random_numbers.append(last_num)
            return np.array(random_numbers)


def generate_means(num_modes: int, x_range: List[int]) -> np.ndarray[int]:
    """
    Generates means between 0 and 100, inclusive, for multiple modes in a GMM.
    Means are guaranteed to be different values.
    """
    while True:
        mu = np.array(
            [random.randint(x_range[0], x_range[1]) for _ in range(num_modes)]
        )
        if len(mu) == len(set(mu)):
            return mu


def generate_data(
    n: int,  # sample size
    *,
    mode_means: Union[np.ndarray[float], List[float]] = None,
    mode_variances: Union[np.ndarray[float], List[float]] = None,
    weights: Union[np.ndarray[float], List[float]] = None,
    num_modes: int = None,
    x_range: List[int] = [0, 100],
    vr_range: List[int] = [1, 5],
    plot: bool = False,
    pr: bool = False,
) -> SampleData:
    """
    Generates sample data.
    """
    assert mode_means is not None or num_modes is not None

    num_modes = len(mode_means) if mode_means is not None else num_modes
    if mode_means is not None:
        mu = np.array(mode_means, dtype=float)
    else:
        mu = generate_means(num_modes, x_range)
    mu = sorted(mu)

    if mode_variances is not None:
        vr = np.array(mode_variances, dtype=float)
    else:
        vr = np.array(
            [random.randint(vr_range[0], vr_range[1]) for _ in range(num_modes)]
        )

    p = (
        np.array([1 / num_modes] * num_modes)
        if weights is None
        else np.array(weights, dtype=float)
    )

    nk = (p * n).astype(int)

    x = []
    x_by_mu = []
    for i in range(num_modes):
        vals = mu[i] + np.random.randn(nk[i]) * np.sqrt(vr[i])
        x.extend(vals)
        x_by_mu.append(vals)

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
    return SampleData(mu, vr, p, np.array(x), x_by_mu)


def init_em(
    x: np.ndarray[int],
    num_modes: int,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initializes the expectation-maximization algorithm using k-means clustering on the data.
    Returns the sample size, means, variances, weights, and log likelihood of the initial GMM.
    """
    # initial conditions
    kmeans_data = np.ravel(x).astype(float).reshape(-1, 1)
    kmeans = KMeans(n_init=1, n_clusters=num_modes)
    kmeans.fit(kmeans_data)

    mu = np.sort(np.ravel(kmeans.cluster_centers_))  # initial means
    vr = [np.var(x)] * num_modes  # initial variances

    p = np.ones(num_modes) / num_modes  # initial p_k proportions
    logL = []  # logL values
    n = len(x)  # sample size

    # initial log-likelihood
    logL.append(calc_log_likelihood(x, mu, vr, p))

    return n, mu, vr, p, logL


def calc_responsibility(
    x: np.ndarray[int],
    n: int,
    mu: np.ndarray[float],
    vr: np.ndarray[float],
    p: np.ndarray[float],
):
    """Calculates the responsibility matrix for each point/mode."""
    gz = np.zeros((n, len(mu)))
    for i in range(len(x)):
        # For each point, calculate the probability that a point is from a gaussian using the mean, standard deviation, and weight of each gaussian
        gz[i, :] = p * norm.pdf(x[i], mu, np.sqrt(vr))
        gz[i, :] /= np.sum(gz[i, :])
    return gz


def assign_values_to_modes(
    x: np.ndarray[int],
    num_modes: int,
    mu: np.ndarray[float],
    vr: np.ndarray[float],
    p: np.ndarray[float],
) -> List[np.ndarray[int]]:
    gz = calc_responsibility(x, len(x), mu, vr, p)
    assignments = np.argmax(gz, axis=1)
    x_by_mode = [[] for _ in range(num_modes)]
    for i, mode in enumerate(assignments):
        x_by_mode[mode].append(x[i])
    x_by_mode = [np.array(data_points) for data_points in x_by_mode]
    return x_by_mode


def em(
    x: np.ndarray[int],
    num_modes: int,
    n: int,
    mu: np.ndarray[float],
    vr: np.ndarray[float],
    p: np.ndarray[float],
) -> GMM:
    """Performs one iteration of the expectation-maximization algorithm."""
    # Expectation step: calculate the posterior probabilities
    gz = calc_responsibility(x, n, mu, vr, p)

    # Ensure that each point contributes to the responsibility matrix above some threshold
    gz[(gz < RESPONSIBILITY_THRESHOLD) | np.isnan(gz)] = RESPONSIBILITY_THRESHOLD

    # Maximization step: estimate gaussian parameters
    # Given the probability that each point belongs to particular gaussian, calculate the mean, variance, and weight of the gaussian
    nk = np.sum(gz, axis=0)
    mu = [(1.0 / nk[j]) * np.sum(gz[:, j] * x) for j in range(num_modes)]
    vr = [(1.0 / nk[j]) * np.sum(gz[:, j] * (x - mu[j]) ** 2) for j in range(num_modes)]
    p = nk / n

    # update likelihood
    logL = calc_log_likelihood(x, mu, vr, p)
    return GMM(mu, vr, p, logL)


def run_em(
    x: np.ndarray[int],  # data
    num_modes: int,
    plot: bool = False,
) -> List[GMM]:
    """
    Given a dataset and an estimated number of modes for the GMM, estimates the parameters for each distribution.
    The algorithm is initialized using the k-means clustering approach, then the EM algorithm is run for up to 30 iterations, or a convergence of the log-likelihood; whichever comes first.
    Returns the GMM estimated by each iteration of the EM algorithm.
    """
    all_params: List[GMM] = []

    n, mu, vr, p, logL = init_em(x, num_modes)  # initialize parameters
    all_params.append(GMM(mu, vr, p, logL[0]))

    max_iterations = 30
    i = 0
    while i < max_iterations:
        # update likelihood
        gmm = em(x, num_modes, n, mu, vr, p)
        logL.append(gmm.logL)
        all_params.append(gmm)
        mu, vr, p = gmm.mu, gmm.vr, gmm.p

        # Convergence check
        if abs(logL[-1] - logL[-2]) < 0.05:
            break

        if i == max_iterations - 1:
            warnings.warn(
                "Maximum number of iterations reached without logL convergence"
            )
        i += 1

    # if plot:
    #     # Visualize the final model
    #     plot_distributions(
    #         x, n, mu, vr, p, title=f"Final Stats: {print_stats(logL[-1], mu, vr, p)}"
    #     )

    return all_params


def identify_outliers(
    x: np.ndarray[int], mu: np.ndarray[float], vr: np.ndarray
) -> List[Tuple[int, float]]:
    """
    Given the data and distribution, identifies the outlier values based on their contribution to each mode in the GMM.
    Returns the indices and values of the outliers, if any.
    """
    outliers: List[Tuple[int, float]] = []
    for i, x_i in enumerate(x):
        contributions = norm.pdf(x_i, mu, np.sqrt(vr))
        poss_outlier = np.all(contributions < OUTLIER_THRESHOLD / len(x))
        if poss_outlier:
            outliers.append((i, x_i))
    return outliers


def remove_outliers(outliers: List[Tuple[int, float]], x: np.ndarray) -> np.ndarray:
    """Removes outlier values from the dataset."""
    for i, _ in outliers[::-1]:
        x = np.delete(x, i)
    return x


def resize_data_window(data: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Identifies and removes the outliers from the dataset."""
    x = data[:]
    outlier_values = []
    for num_modes in [1, 2]:
        params = run_em(x, num_modes)
        mu, vr = params[-1].mu, params[-1].vr
        outliers = identify_outliers(x, mu, vr)
        outlier_values.extend([x_i for _, x_i in outliers])
        if len(outliers) > 0:
            x = remove_outliers(outliers, x)
    return x, outlier_values


def run_gmm(
    x: Union[List[int], np.ndarray], *, plot: bool = False, pr: bool = False
) -> EstimatedGMM:
    """
    Runs the GMM estimation process to determine the number of structural variants in a DNA reading frame.
    x is a list of data points corresponding to the y-intercept of the L-R position calculated after a shift in the genome due to a deletion of greater than 1 base pair.
    If x contains 10 or fewer data points, then 1 structural variant is estimated. If x has more than 10 data points, then outliers are first identified, and the reading frame is resized to exclude these outliers. The EM algorithm is then run for a 1, 2, or 3 mode GMM, and the resulting AIC scores are calculated and compared across the estimated GMMs. The GMM with the lowest AIC score is returned as the optimal fit to the data.
    """
    x = np.array(x, dtype=float)
    n = len(x)
    if len(x) == 0:
        warnings.warn("Input data is empty")
        return EstimatedGMM(
            mu=[],
            vr=[],
            p=[],
            num_modes=0,
            logL=0,
            aic=0,
            outliers=[],
            percent_data_removed=0,
            window_size=(0, 0),
            x_by_mode=[],
        )
    if len(x) == 1:
        warnings.warn("Input data contains one SV")
        singleton = x[0]
        return EstimatedGMM(
            mu=[singleton],
            vr=[0],
            p=[1],
            num_modes=1,
            logL=0,
            aic=0,
            outliers=[],
            percent_data_removed=0,
            window_size=(singleton, singleton),
            x_by_mode=[],
        )

    outliers = []
    aic_vals = []
    if len(x) <= 10:  # small number of SVs detected
        opt_params = run_em(x, 1, plot)
        num_sv = 1
    else:
        x, outliers = resize_data_window(x)
        all_params = []
        for num_modes in range(1, 4):
            params = run_em(x, num_modes, plot)
            aic = calc_aic(params[-1].logL, num_modes, params[-1].mu, params[-1].vr)
            all_params.append(params)
            aic_vals.append(aic)
        print(aic_vals)
        min_aic_idx = aic_vals.index(min(aic_vals))
        opt_params = all_params[min_aic_idx]
        num_sv = len(opt_params[0].mu)

    final_params = opt_params[-1]

    x_by_mode = assign_values_to_modes(
        x, num_sv, final_params.mu, final_params.vr, final_params.p
    )

    if pr:
        print(
            f"\nNumber of SVs: {num_sv}\n{print_stats(final_params.logL, final_params.mu, final_params.vr, final_params.p)}. {len(outliers)} outliers removed."
        )

    if plot:
        # Plot the likelihood function over time
        # plot_likelihood([x.logL for x in opt_params])
        # animate_distribution(x, opt_params)
        plot_distributions(x, len(x), final_params.mu, final_params.vr, final_params.p)

    return EstimatedGMM(
        mu=final_params.mu,
        vr=final_params.vr,
        p=final_params.p,
        num_modes=num_sv,
        logL=final_params.logL,
        aic=min(aic_vals) if len(aic_vals) > 0 else None,
        outliers=outliers,
        percent_data_removed=len(outliers) / n,
        window_size=(min(x), max(x)),
        x_by_mode=x_by_mode,
    )


if __name__ == "__main__":
    data = generate_data(n=100, num_modes=2)
    gmm = run_gmm(data.x, plot=False)
