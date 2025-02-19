import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
from scipy.stats import dirichlet
from collections import Counter
from process_data import run_viz_gmm
from gmm_types import *
from typing import List, Tuple

MAX_N = 1000


def run_trial(squiggle_data, **kwargs) -> Tuple[GMM, List[List[Evidence]]]:
    kwargs["plot"] = False
    kwargs["plot_bokeh"] = False
    gmm, evidence_by_mode = run_viz_gmm(squiggle_data, **kwargs)
    return gmm, evidence_by_mode


def update_dirichlet(alphas: np.ndarray, outcomes: List[int]) -> np.ndarray:
    counts = Counter(outcomes)
    for num_modes, count in counts.items():
        alphas[num_modes - 1] += count
    return alphas


def animate_dirichlet(ps):
    """Animates the evolution of the Dirichlet probabilities over trials."""
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.set_xlim(1, len(ps))
    ax.set_xlabel("Trials")
    ax.set_ylabel("Probability")

    lines = [
        ax.plot([], [], label=f"{i+1} mode{'s' if i > 0 else ''}", color=COLORS[i])[0]
        for i in range(3)
    ]
    ax.legend()

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        x = np.arange(frame + 1) + 1
        for i, line in enumerate(lines):
            y = [h[i] for h in ps[: frame + 1]]
            line.set_data(x, y)
        return lines

    ani = animation.FuncAnimation(
        fig, update, frames=len(ps), init_func=init, blit=False, repeat=True
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
        alpha = update_dirichlet(alpha, outcomes[: i + 1])
        alphas.append(alpha.copy())
    animate_dirichlet_heatmap(alphas)


def run_dirichlet(squiggle_data, **kwargs) -> Tuple[List[GMM], List[np.ndarray]]:
    display_output = kwargs.get("plot", False)

    alpha = np.array([1, 1, 1])  # initialize alpha values
    outcomes = []  # num_modes
    ps = [alpha / np.sum(alpha)]  # probabilities over time
    alphas = [alpha]  # alphas over time
    gmms = []  # keep track of output gmms
    n = 0
    while n < MAX_N:
        gmm, evidence_by_mode = run_trial(squiggle_data, **kwargs)
        gmms.append((gmm, evidence_by_mode))
        num_modes = 1 if gmm is None else gmm.num_modes
        outcomes.append(num_modes)

        alpha = update_dirichlet(alpha, outcomes)
        probabilities = alpha / np.sum(alpha)
        alphas.append(alpha.copy())
        ps.append(probabilities)

        if display_output:
            print(f"Trial {n + 1}: outcome={num_modes}, probabilities={probabilities}")

        if np.max(probabilities) >= 0.8:
            if display_output:
                print(
                    f"Stopping after {n + 1} iterations, {np.argmax(probabilities) + 1} modes"
                )
            break

        n += 1

    if display_output:
        animate_dirichlet_heatmap(alphas)
        animate_dirichlet(ps)

    return gmms, ps


if __name__ == "__main__":
    df = pd.read_csv("1000genomes/sv_stats_converge.csv", low_memory=False)
    df = df[df["id"] == "DEL_pindel_98"]
    animate_dirichlet_history(df)
