import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Tuple


def init_em(
    x: np.ndarray,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # initial conditions
    mu = [min(x), np.median(x), max(x)]  # initial means
    vr = [1, 1, 1]  # initial variances
    p = np.ones(3) / 3  # initial p_k proportions
    logL = np.zeros(10)  # logL values
    n = len(x)  # sample size

    # initial log-likelihood
    Nxi = np.column_stack([norm.pdf(x, mu[i], np.sqrt(vr[i])) for i in range(len(mu))])
    logL[0] = np.sum(np.log(p * Nxi))

    print(f"logL = {logL[0]}")
    print(f"means = {mu}")
    print(f"variances = {vr}")
    print(f"p = {p}")

    return n, mu, vr, p, logL


def run_em(
    x: np.ndarray,  # data
) -> None:
    n, mu, vr, p, logL = init_em(x)  # initialize parameters
    ux = np.arange(np.min(x), np.max(x) + 0.25, 0.25)
    hx, edges = np.histogram(x, bins=ux)
    ux = (edges[:-1] + edges[1:]) / 2
    plt.figure(1)
    plt.plot(ux, hx, "bo")
    for i in range(len(mu)):
        plt.plot(ux, (n * p[i] / 4) * norm.pdf(ux, mu[i], np.sqrt(vr[i])), "r-")
    plt.show()

    gz = np.zeros((n, len(mu)))
    for jj in range(1, len(logL)):
        # E-step: calculate the posterior probabilities
        for i in range(len(x)):
            gz[i, :] = p * norm.pdf(x[i], mu, np.sqrt(vr))
            gz[i, :] /= np.sum(gz[i, :])

        # M-step: estimate gaussian parameters
        nk = np.sum(gz, axis=0)
        mu = [(1.0 / nk[j]) * np.sum(gz[:, j] * x) for j in range(len(mu))]
        vr = [
            (1.0 / nk[j]) * np.sum(gz[:, j] * (x - mu[j]) ** 2) for j in range(len(vr))
        ]
        p = nk / n

        # update likelihood
        Nxi = np.column_stack(
            [norm.pdf(x, mu[i], np.sqrt(vr[i])) for i in range(len(mu))]
        )
        logL[jj] = np.sum(np.log(p * Nxi))

        print(f"[{jj}] logL = {logL[jj]:8.2f}")
        print(f"mu = {mu}")
        print(f"vr = {vr}")
        print(f"p  = {p}")

        plt.figure(1)
        plt.plot(ux, hx, "bo")
        for i in range(len(mu)):
            plt.plot(ux, (n * p[i] / 4) * norm.pdf(ux, mu[i], np.sqrt(vr[i])), "r-")
        plt.show()

        # Convergence check
        if (logL[jj] - logL[jj - 1]) < 0.05:
            break

    # Plot the likelihood function over time
    plt.figure(2)
    plt.plot(np.arange(1, len(logL) + 1), logL, "bo-")
    plt.xlabel("iterations")
    plt.ylabel("log-likelihood")
    plt.show()

    # Visualize the final model
    ux = np.arange(np.min(x), np.max(x) + 0.25, 0.25)
    hx, edges = np.histogram(x, bins=ux)
    ux = (edges[:-1] + edges[1:]) / 2  # Adjusting bins to get midpoints

    plt.figure(1)
    plt.plot(ux, hx, "bo")
    for i in range(len(mu)):
        plt.plot(ux, (n * p[i] / 4) * norm.pdf(ux, mu[i], np.sqrt(vr[i])), "r-")
    plt.xlabel("x value")
    plt.ylabel("density")
    plt.show()
