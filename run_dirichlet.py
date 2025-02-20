import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
from scipy.integrate import nquad
from scipy.stats import dirichlet
from collections import Counter
from process_data import run_viz_gmm
from gmm_types import *
from typing import List, Tuple

MAX_N = 100

"""Visualizations of the Dirichlet process"""


def animate_dirichlet(posterior_distributions):
    """Animates the evolution of the Dirichlet probabilities over trials."""
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.set_xlim(1, len(posterior_distributions))
    ax.set_xlabel("Trials")

    ax.set_ylabel("Probability")
    lines = [
        ax.plot([], [], label=f"{i+1} mode{'s' if i > 0 else ''}", color=COLORS[i])[0]
        for i in range(3)
    ]
    error_bands = [
        ax.fill_between([], [], [], color=COLORS[i], alpha=0.2) for i in range(3)
    ]
    ax.legend()

    def init():
        for line in lines:
            line.set_data([], [])
        return lines + error_bands

    def update(frame):
        x = np.arange(frame + 1) + 1
        for i, line in enumerate(lines):
            y = [h[0][i] for h in posterior_distributions[: frame + 1]]
            line.set_data(x, y)
        for i, band in enumerate(error_bands):
            y = [h[0][i] for h in posterior_distributions[: frame + 1]]
            yerr = [np.sqrt(h[1][i]) for h in posterior_distributions[: frame + 1]]
            band.remove()
            error_bands[i] = ax.fill_between(
                x,
                np.array(y) - np.array(yerr),
                np.array(y) + np.array(yerr),
                color=COLORS[i],
                alpha=0.2,
            )
        return lines + error_bands

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(posterior_distributions),
        init_func=init,
        blit=False,
        repeat=True,
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="upper left")
    plt.show()


def animate_dirichlet_heatmap(alphas):
    """Animates the evolution of the Dirichlet distribution heatmap over trials."""
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        ax.clear()
        alpha = alphas[frame]
        samples = dirichlet(alpha).rvs(size=10000)
        x = samples[:, 0] + 0.5 * samples[:, 1]
        y = np.sqrt(3) / 2 * samples[:, 1]
        # sns.kdeplot(x=x, y=y, fill=True, cmap="Blues", levels=30, ax=ax)
        ax.hexbin(x, y, gridsize=100, cmap="Blues")  # to improve performance

        ax.set_xlim(0, 1)
        ax.set_ylim(0, np.sqrt(3) / 2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Draw the triangle
        triangle = plt.Polygon(
            [[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]], edgecolor="k", fill=None
        )
        ax.add_patch(triangle)

        # Label the edges of the triangle
        ax.text(1, -0.05, "1 mode", ha="center", va="center", fontsize=12)
        ax.text(
            0.5, np.sqrt(3) / 2 + 0.05, "2 modes", ha="center", va="center", fontsize=12
        )
        ax.text(0, -0.05, "3 modes", ha="center", va="center", fontsize=12)

        ax.text(
            0,
            np.sqrt(3) / 2 + 0.05,
            f"Trial {frame + 1}",
            ha="center",
            va="center",
            fontsize=12,
        )

    ani = animation.FuncAnimation(fig, update, frames=len(alphas))
    plt.show()


def animate_dirichlet_history(df):
    alpha = np.array([1, 1, 1])
    outcomes = df["num_modes"].values
    alphas = [alpha]
    for i in range(len(outcomes)):
        # alpha = update_dirichlet(alpha, outcomes[: i + 1]) #TODO: fix this
        alphas.append(alpha.copy())
    animate_dirichlet_heatmap(alphas)


def run_trial(squiggle_data, **kwargs) -> Tuple[GMM, List[List[Evidence]]]:
    kwargs["plot"] = False
    kwargs["plot_bokeh"] = False
    gmm, evidence_by_mode = run_viz_gmm(squiggle_data, **kwargs)
    return gmm, evidence_by_mode


def run_dirichlet(squiggle_data, **kwargs) -> Tuple[List[GMM], List[np.ndarray]]:
    display_output = kwargs.get("plot", False)

    alpha = np.array([1, 1, 1])  # initialize alpha values
    counts = np.array([0, 0, 0])  # count of num_modes
    alphas = [alpha]  # alphas over time
    posterior_distributions = []  # distributions over time
    gmms = []  # keep track of output gmms
    n = 0
    while n < MAX_N:
        n += 1  # number of trials

        # Run the model to get the next outcome
        gmm, evidence_by_mode = run_trial(squiggle_data, **kwargs)
        gmms.append((gmm, evidence_by_mode))
        num_modes = 1 if gmm is None else gmm.num_modes
        counts[num_modes - 1] += 1

        # Update the alpha values (used as a conjugate prior for Bayes)
        alpha_posterior = alpha + counts
        alphas.append(alpha_posterior)

        # Get the posterior distributions for each of the 3 modes
        posterior_probs = alpha_posterior / np.sum(alpha_posterior)
        sum_alpha_post = np.sum(alpha_posterior)
        posterior_var = (alpha_posterior * (sum_alpha_post - alpha_posterior)) / (
            sum_alpha_post**2 * (sum_alpha_post + 1)
        )
        posterior_distributions.append((posterior_probs, posterior_var))

        # Calculate the difference in means between the two most probable modes
        # sort posterior_mus
        posterior_mu_sorted_indices = np.argsort(posterior_probs)
        posterior_mu_sorted = posterior_probs[posterior_mu_sorted_indices]
        diff_in_means = posterior_mu_sorted[-1] - posterior_mu_sorted[-2]

        # Calculate the confidence interval for our difference in means
        diff_var = (posterior_var[posterior_mu_sorted_indices[-1]]) / n + (
            posterior_var[[posterior_mu_sorted_indices[-2]]]
        ) / n
        confidence = 1.96 * np.sqrt(diff_var)
        ci = [diff_in_means - confidence, diff_in_means + confidence]

        if display_output:
            print(f"Trial {n}: outcome={num_modes}, probabilities={posterior_probs}")

        # Check our stopping condition
        if ci[0] >= 0.6:
            if display_output:
                print(
                    f"Stopping after {n} iterations, {np.argmax(posterior_probs) + 1} modes"
                )
            break

    if display_output:
        if ci[0] >= 0.3 and ci[1] < 0.6:
            print(f"No clear convergence after {n} iterations")
        # animate_dirichlet_heatmap(alphas)
        animate_dirichlet(posterior_distributions)

    return gmms, posterior_distributions


if __name__ == "__main__":
    df = pd.read_csv("1000genomes/sv_stats_converge.csv", low_memory=False)
    df = df[df["id"] == "DEL_pindel_98"]
    animate_dirichlet_history(df)
