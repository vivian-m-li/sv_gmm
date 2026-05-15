import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import dirichlet

from src.model.gmm_trial import gmm_trial
from src.utils.constants import COLORS
from src.utils.model_helper import calculate_posteriors
from src.utils.types import GMM, Evidence
from src.utils.viz import plot_2d_coords

MAX_N = 100  # maximum number of iterations for the Dirichlet process


# ----------------------------------------
# Visualizations of the Dirichlet process
# ----------------------------------------
def animate_dirichlet(posterior_distributions):
    """Animates the evolution of the Dirichlet probabilities over trials."""
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.set_xlim(1, len(posterior_distributions))
    ax.set_xlabel("Trials")

    ax.set_ylabel("Probability")
    lines = [
        ax.plot(
            [], [], label=f"{i + 1} mode{'s' if i > 0 else ''}", color=COLORS[i]
        )[0]
        for i in range(3)
    ]
    error_bands = [
        ax.fill_between([], [], [], color=COLORS[i], alpha=0.2)
        for i in range(3)
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
            yerr = [
                np.sqrt(h[1][i]) for h in posterior_distributions[: frame + 1]
            ]
            band.remove()
            error_bands[i] = ax.fill_between(
                x,
                np.array(y) - np.array(yerr),
                np.array(y) + np.array(yerr),
                color=COLORS[i],
                alpha=0.2,
            )
        return lines + error_bands

    _ = animation.FuncAnimation(
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
            0.5,
            np.sqrt(3) / 2 + 0.05,
            "2 modes",
            ha="center",
            va="center",
            fontsize=12,
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

    _ = animation.FuncAnimation(fig, update, frames=len(alphas))
    plt.show()


def animate_dirichlet_history(df):
    """Animates the evolution of the Dirichlet distribution heatmap over trials."""
    alpha = np.array([1, 1, 1])
    outcomes = df["num_modes"].values
    alphas = [alpha]
    for i in range(len(outcomes)):
        # alpha = update_dirichlet(alpha, outcomes[: i + 1]) #TODO: fix this
        alphas.append(alpha.copy())
    animate_dirichlet_heatmap(alphas)


# ----------------------------------------
# Dirichlet process implementation
# ----------------------------------------
def run_trial(reads, **kwargs) -> tuple[GMM, list[list[Evidence]]]:
    """Runs a single trial of Gaussian Mixture Model."""
    kwargs["plot"] = False
    gmm_result, evidence_by_mode = gmm_trial(reads, **kwargs)
    return gmm_result, evidence_by_mode


def run_dirichlet(
    reads, insert_size_file=None, **kwargs
) -> tuple[list[GMM], list[np.ndarray]]:
    """
    Runs the Dirichlet process until convergence or max iterations.
    1. Initialize alpha values and counts. Each outcome (1, 2, or 3 modes) is equally likely.
    2. For each trial:
        a. Run the GMM model to get the number of modes.
        b. Update counts and alpha values.
        c. Calculate posterior distributions and confidence intervals.
        d. Check stopping condition based on confidence interval.
    3. Return the outcomes, alpha values, and posterior distributions over all trials.

    If init="kmeans++", then we skip the Dirichlet process and just run a single trial.
    """
    chr, L, R = kwargs["chr"], kwargs["L"], kwargs["R"]

    init = kwargs.get("init", "dp_kmeans++")
    display_output = kwargs.get("plot", False)

    alpha = np.array([1, 1, 1])  # initialize alpha values
    counts = np.array([0, 0, 0])  # count of num_modes
    alphas = [alpha]  # alphas over time
    posterior_distributions = []  # distributions over time
    gmm_results = []  # keep track of output gmms
    n = 0

    max_n_trials = 1 if init == "kmeans++" else MAX_N
    while n < max_n_trials:
        n += 1  # number of trials

        # Run the model to get the next outcome
        gmm_result, evidence_by_mode = run_trial(reads, **kwargs)
        if gmm_result is None:  # all samples are filtered out
            gmm_results.append((None, []))
            print(f"{chr}:{L}-{R} - stopping due to gmm_result = None")
            break

        print("result:", gmm_result.num_modes, gmm_result.score)
        gmm_results.append((gmm_result, evidence_by_mode))
        num_modes = gmm_result.num_modes
        counts[num_modes - 1] += 1

        # Update the alpha values (used as a conjugate prior for Bayes)
        alpha_posterior = alpha + counts
        alphas.append(alpha_posterior)

        # Get the posterior distributions for each of the 3 modes
        p, var = calculate_posteriors(alpha_posterior)
        posterior_distributions.append((p, var))

        # TODO: remove early stopping conditions - run to max # trials
        # # Calculate the confidence interval
        # ci = calculate_ci(p, var, n)

        # if display_output:
        #     print(f"Trial {n}: outcome={num_modes}, probabilities={p}")

        # # Check our stopping condition
        # if ci[0] >= 0.6:
        #     if display_output:
        #         print(
        #             f"{chr}:{L}-{R} - stopping after {n} iterations, {np.argmax(p) + 1} modes, ci={ci}"
        #         )
        #     break

    _, best_gmm_evidence = min(
        gmm_results, key=lambda x: x[0].score if x[0] else np.inf
    )

    if display_output:
        # get the file name of the plot
        plot_file = kwargs["plot_file"]
        print(f"\nSaving plot to {plot_file}\n")
        # show the last L-len plot
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_2d_coords(
            ax,
            best_gmm_evidence,
            L=L,
            R=R,
            axis1="L",
            axis2="Length",
            add_error_bars=False,
            size_by="",
            show_mode_stats=True,
            show_1d_distributions=True,
            insert_size_file=insert_size_file,
            repulsion=kwargs.get("repulsion", False),
            init="dp_kmeans++",
        )
        plt.tight_layout()
        plt.savefig(f"{plot_file}.png", dpi=200)

        # animate_dirichlet_heatmap(alphas)
        # animate_dirichlet(posterior_distributions)

    return gmm_results, alphas, posterior_distributions
