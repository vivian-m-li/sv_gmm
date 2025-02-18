import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
from process_data import run_viz_gmm
from gmm_types import *

MAX_N = 1000


def run_trial(squiggle_data, **kwargs):
    kwargs["plot"] = False
    kwargs["plot_bokeh"] = False
    gmm, evidence_by_mode = run_viz_gmm(squiggle_data, **kwargs)
    return gmm


def update_dirichlet(alphas, outcomes):
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

    lines = [ax.plot([], [], label=f"Mode {i+1}", color=COLORS[i])[0] for i in range(3)]
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


def dirichlet(squiggle_data, **kwargs):
    display_output = kwargs.get("plot", False)

    alphas = np.array([1, 1, 1])  # initialize alpha values
    outcomes = []  # num_modes
    ps = []  # probabilities over time
    gmms = []
    n = 0
    while n < MAX_N:
        gmm = run_trial(squiggle_data, **kwargs)
        gmms.append(gmm)
        outcomes.append(gmm.num_modes)

        alphas = update_dirichlet(alphas, outcomes)
        probabilities = alphas / np.sum(alphas)
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
        animate_dirichlet(ps)
    return gmms, ps
