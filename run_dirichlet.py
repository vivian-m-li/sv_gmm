import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
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
    return (gmm, evidence_by_mode)


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


def animate_dirichlet_heatmap(history):
    """Animates the evolution of the Dirichlet distribution heatmap over trials."""
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        ax.clear()
        alpha = history[frame]
        samples = dirichlet(alpha).rvs(size=10000)
        x = samples[:, 0] + 0.5 * samples[:, 1]
        y = np.sqrt(3) / 2 * samples[:, 1]
        sns.kdeplot(x=x, y=y, fill=True, cmap="Blues", levels=30, ax=ax)
        # ax.hexbin(x, y, gridsize=50, cmap="Blues")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, np.sqrt(3) / 2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Dirichlet Distribution Heatmap - Trial {frame+1}")

    ani = animation.FuncAnimation(fig, update, frames=len(history), repeat=False)
    plt.show()


def run_dirichlet(squiggle_data, **kwargs) -> Tuple[List[GMM], List[np.ndarray]]:
    display_output = kwargs.get("plot", False)

    alpha = np.array([1, 1, 1])  # initialize alpha values
    outcomes = []  # num_modes
    ps = []  # probabilities over time
    alphas = []  # alphas over time
    gmms = []  # keep track of output gmms
    n = 0
    while n < MAX_N:
        gmm = run_trial(
            squiggle_data, **kwargs
        )  # outputs tuple of gmm and evidence_by_mode
        gmms.append(gmm)
        outcomes.append(gmm.num_modes)

        alpha = update_dirichlet(alpha, outcomes)
        probabilities = alpha / np.sum(alpha)
        alphas.append(alpha.copy())
        ps.append(probabilities)

        if display_output:
            print(
                f"Trial {n + 1}: outcome={gmm.num_modes}, probabilities={probabilities}"
            )

        if np.max(probabilities) > 0.95:
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
