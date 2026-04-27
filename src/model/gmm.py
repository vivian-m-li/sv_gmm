import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

from src.utils.model_helper import reciprocal_overlap
from src.utils.types import GMM2D, EstimatedGMM2D

RESPONSIBILITY_THRESHOLD = 1e-10

# ------------------------------
# GMM/EM helper functions
# ------------------------------


def hinge_loss_distance(
    mu: list[np.ndarray], *, cutoff=100, max_penalty=200
) -> float:
    """Adds a penalty to the log-likelihood to prevent cluster centroids from getting too close to each other (decreasing d), accounting for sequencing noise."""
    # c - alpha * d(mu_i, mu_j)
    # alpha is the slope
    # 0 at d(mu_i, mu_j) = 141, which is 2 SD
    # c1 = -ln n -- max penalty
    num_modes = len(mu)
    if num_modes <= 1:
        return 0.0

    loss = 0
    for i in range(num_modes):
        for j in range(i + 1, num_modes):
            # add a large penalty for mus that are too close to each other
            dist = np.linalg.norm(mu[i] - mu[j])
            d_norm = np.maximum(0, cutoff - dist) / cutoff
            loss += max_penalty * (d_norm**2)
    return loss


def get_sv_from_mu(mu: np.ndarray, L: int, R: int) -> tuple[int, int]:
    """Converts the mu value (length, L-coordinate) to the SV coordinates (L, R)."""
    svlen = R - L
    sv_L = int(mu[1] + L)
    sv_R = int(sv_L + (svlen + mu[0]))
    return sv_L, sv_R


def hinge_loss_overlap(
    mu: list[np.ndarray], L: int, R: int, *, cutoff=0.8, max_penalty=200
) -> float:
    """Adds a penalty to the log-likelihood to prevent high reciprocal overlap and avoid over-clustering."""
    num_modes = len(mu)
    if num_modes <= 1:
        return 0.0

    loss = 0
    for i in range(num_modes):
        for j in range(i + 1, num_modes):
            sv1, sv2 = get_sv_from_mu(mu[i], L, R), get_sv_from_mu(mu[j], L, R)
            r = reciprocal_overlap(sv1, sv2)
            r_norm = np.maximum(0, r - cutoff) / (1 - cutoff)
            loss += max_penalty * (r_norm**2)
    return loss


def model_penalty(
    mu: list[np.ndarray],
    L: int,
    R: int,
    d_threshold: int,
    r_threshold: float,
    max_penalty: int,
) -> float:
    """
    Aggregate model penalties from distance and reciprocal overlap into a single score.
    """
    svlen = R - L
    # penalty decreases with sv size since clusters will be closer together due to noise in read data
    d_penalty = hinge_loss_distance(
        mu, cutoff=d_threshold, max_penalty=max_penalty
    )
    r_penalty = hinge_loss_overlap(
        mu, L, R, cutoff=r_threshold, max_penalty=max_penalty
    )

    # weight the two penalties based on sv length
    # shorter SVs -> more weight on overlap penalty since noise can cause clusters to be closer in distance
    svlen_bounds = [50, 500]
    weight_bounds = [0, 0.5]
    if svlen < svlen_bounds[0]:
        dist_weight = weight_bounds[0]
    elif svlen > svlen_bounds[1]:
        dist_weight = weight_bounds[1]
    else:
        dist_weight = 0.1 + (
            (svlen - svlen_bounds[0]) / (svlen_bounds[1] - svlen_bounds[0])
        ) * (weight_bounds[1] - weight_bounds[0])
    overlap_weight = 1 - dist_weight
    return dist_weight * d_penalty + overlap_weight * r_penalty


def calc_log_likelihood(
    x: np.ndarray,
    mu: list[np.ndarray],
    cov: list[np.ndarray],
    p: np.ndarray,
    L: int,
    R: int,
    d_threshold: int,
    r_threshold: float,
    max_penalty: int,
) -> float:
    """Calculates the log-likelihood of the data fitting the GMM."""
    num_modes = len(mu)
    logL = 0.0
    for i in range(len(x)):
        pdf = []
        for k in range(num_modes):
            pdf_i = None
            while pdf_i is None:
                try:
                    pdf_i = p[k] * multivariate_normal.pdf(x[i], mu[k], cov[k])
                except np.linalg.LinAlgError:
                    # if the covariance matrix is not positive definite, add a small value to the diagonal and try again
                    cov[k] += np.eye(cov[k].shape[0]) * 1e-6
            pdf.append(pdf_i)
        likelihood_i = np.sum(pdf)
        logL += np.log(likelihood_i + 1e-300)  # avoid log(0)

    # calculate final penalized logL
    penalty = model_penalty(mu, L, R, d_threshold, r_threshold, max_penalty)
    penalized_logL = logL - penalty
    return penalized_logL


def calc_aic(
    logL: float,
    num_modes: int,
) -> float:
    """Calculates the penalized log-likelihood using AIC."""
    num_params = (
        (num_modes * 2) + (num_modes * 3) + (num_modes - 1)
    )  # number of parameters for mu + cov + p - 1
    # aic = 2k - 2ln(L)
    aic = (2 * num_params) - (2 * logL)
    return aic


def init_em(
    x: np.ndarray,
    num_modes: int,
    L: int,
    R: int,
    d_threshold: int,
    r_threshold: float,
    max_penalty: int,
) -> tuple[int, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Initializes the expectation-maximization algorithm using k-means clustering on the data.
    Returns the sample size, means, variances, weights, and log likelihood of the initial GMM.
    """

    # check that there are at least the same number of unique values as num_modes
    # if not, then we can't split into more modes than there are values and we throw an error
    unique_vals = np.unique(x, axis=0)
    n_unique = len(unique_vals)
    if n_unique < num_modes:
        raise ValueError(f"Too few unique values ({n_unique}) to cluster into {num_modes}")

    # initial conditions
    kmeans = KMeans(n_clusters=num_modes)
    kmeans.fit(x)

    mu = kmeans.cluster_centers_  # initial means
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    count_lookup = dict(zip(unique, counts))
    cov = [
        (
            np.cov(x[kmeans.labels_ == i].T)
            if count_lookup[i] > 1
            else np.eye(x.shape[1])
        )
        for i in range(num_modes)
    ]  # initial covariances, sets a default covariance if only one point in a cluster
    p = [
        count_lookup[i] / len(x) for i in range(num_modes)
    ]  # initial p_k proportions
    logL = []  # logL values
    n = len(x)  # sample size

    # initial log-likelihood
    logL.append(
        calc_log_likelihood(
            x, mu, cov, p, L, R, d_threshold, r_threshold, max_penalty
        )
    )

    return n, mu, cov, p, logL


def calc_responsibility(
    x: np.ndarray,
    n: int,
    mu: list[np.ndarray],
    cov: list[np.ndarray],
    p: np.ndarray,
) -> np.ndarray:
    """
    Calculates the responsibility matrix for each point/mode.
    Returns a matrix of size (number of samples) x (number of modes) where each entry represents the probability that a point belongs to a particular mode.
    """
    gz = np.zeros((n, len(mu)))  # number of points by number of modes
    for i in range(len(x)):
        # For each point, calculate the probability that a point is from a gaussian using the mean, standard deviation, and weight of each gaussian
        gz[i, :] = [
            p[k] * multivariate_normal.pdf(x[i], mu[k], cov[k])
            for k in range(len(mu))
        ]
        gz[i, :] /= np.sum(gz[i, :])
    return gz


def em(
    x: np.ndarray,
    num_modes: int,
    n: int,
    mu: list[np.ndarray],
    cov: list[np.ndarray],
    p: np.ndarray,
    L: int,
    R: int,
    d_threshold: int,
    r_threshold: float,
    max_penalty: int,
) -> GMM2D:
    """Performs one iteration of the expectation-maximization algorithm."""
    # Expectation step: calculate the posterior probabilities
    gz = calc_responsibility(x, n, mu, cov, p)

    # Ensure that each point contributes to the responsibility matrix above some threshold
    gz[(gz < RESPONSIBILITY_THRESHOLD) | np.isnan(gz)] = (
        RESPONSIBILITY_THRESHOLD
    )

    # Maximization step: estimate gaussian parameters
    # Given the probability that each point belongs to particular gaussian, calculate the mean, variance, and weight of the gaussian
    nk = np.sum(gz, axis=0)
    mu = [
        (1.0 / nk[j]) * np.sum(gz[:, j, None] * x, axis=0)
        for j in range(num_modes)
    ]
    cov = [
        (1.0 / nk[j]) * np.dot((gz[:, j, None] * (x - mu[j])).T, (x - mu[j]))
        for j in range(num_modes)
    ]
    p = nk / n

    # update likelihood
    logL = calc_log_likelihood(
        x, mu, cov, p, L, R, d_threshold, r_threshold, max_penalty
    )
    return GMM2D(mu, cov, p, logL)


def run_em(
    x: np.ndarray,  # data
    num_modes: int,
    L: int,
    R: int,
    d_threshold: int = 100,
    r_threshold: float = 0.8,
    max_penalty: int = 200,
    plot: bool = False,
) -> tuple[list[GMM2D], int]:
    """
    Given a dataset and an estimated number of modes for the GMM, estimates the parameters for each distribution.
    The algorithm is initialized using the k-means clustering approach, then the EM algorithm is run for up to 30 iterations, or a convergence of the log-likelihood; whichever comes first.
    Returns the GMM estimated by each iteration of the EM algorithm.
    """
    all_params: list[GMM2D] = []

    n, mu, cov, p, logL = init_em(
        x, num_modes, L, R, d_threshold, r_threshold, max_penalty
    )  # initialize parameters
    all_params.append(GMM2D(mu, cov, p, logL[0]))

    # only need to estimate mean and covariance matrix if we have 1 mode
    if num_modes == 1:
        return all_params, 0

    max_iterations = 30
    i = 0
    while i < max_iterations:
        prev_gmm = all_params[-1]
        gmm_result = em(
            x,
            num_modes,
            n,
            prev_gmm.mu,
            prev_gmm.cov,
            prev_gmm.p,
            L,
            R,
            d_threshold,
            r_threshold,
            max_penalty,
        )
        logL.append(gmm_result.logL)
        all_params.append(gmm_result)

        # Convergence check
        if abs(logL[-1] - logL[-2]) < 0.05:
            break

        i += 1

    return all_params, i


def assign_values_to_modes(
    x: np.ndarray,
    num_modes: int,
    mu: np.ndarray,
    cov: list[np.ndarray],
    p: np.ndarray,
) -> tuple[np.ndarray, tuple[list[np.ndarray], list[int]]]:
    """Assigns each data point to a mode based on the highest responsibility value."""
    gz = calc_responsibility(x, len(x), mu, cov, p)
    assignments = np.argmax(gz, axis=1)
    x_by_mode = [[] for _ in range(num_modes)]
    for i, mode in enumerate(assignments):
        if mode is not None:
            x_by_mode[mode].append(x[i])
    x_by_mode = [np.array(data_points) for data_points in x_by_mode]
    return gz, x_by_mode


def gmm(
    x: np.ndarray[tuple[float, int]],
    *,
    L,
    R,
    d_threshold: int = 100,
    r_threshold: float = 0.8,
    max_penalty: int = 200,
    plot: bool = False,
    pr: bool = False,
    force_n_modes: int | None = None,
) -> EstimatedGMM2D | None:
    """
    Runs the GMM estimation process to determine the number of structural variants in a DNA reading frame.
    x is a 2D list of data points where each data point consists of:
      - the y-intercept of the L-R position calculated after a shift in the genome due to a deletion of greater than 1 base pair, and
      - the max L value read for each sample
    If x contains 10 or fewer data points, then 1 structural variant is returned by default. If x has more than 10 data points, then the EM algorithm is then run for a 1, 2, or 3 mode GMM, and the resulting AIC scores are calculated and compared across the estimated GMMs. The GMM with the lowest AIC score is returned as the optimal fit to the data. The GMM's mu value represents the length and L coordinate of each structural variant estimated.
    """
    x = np.array(x, dtype=float)
    if len(x) == 0:
        raise Exception("No data points provided.")
    if len(x) == 1:
        singleton = x[0]
        return EstimatedGMM2D(
            mu=[singleton],
            cov=[
                [0, 0],
                [0, 0],
            ],  # can't determine the covariance for only one pair of points
            p=[1],
            num_modes=1,
            logL=0,
            aic=0,
            outliers=[],
            window_size=(singleton, singleton),
            x_by_mode=[np.array(x)],
            responsibility=np.array([[1.0]]),
            num_pruned=[0],
            num_iterations=0,
        )

    outliers = []
    aic_vals = []
    if len(x) <= 10:  # small number of samples detected
        opt_params, num_iterations = run_em(
            x, 1, L, R, d_threshold, r_threshold, max_penalty, plot
        )
        num_iterations_final = num_iterations
        num_sv = 1
    else:
        all_params = []
        iterations = []
        mode_options = (
            [force_n_modes] if force_n_modes is not None else range(1, 4)
        )
        for num_modes in mode_options:
            try:
                params, num_iterations = run_em(
                    x, num_modes, L, R, d_threshold, r_threshold, max_penalty, plot
                )
                aic = calc_aic(params[-1].logL, num_modes)
            except ValueError as e:
                # print(e)
                params = [
                    GMM2D(
                        mu=[[0, 0] for _ in range(num_modes)],
                        cov=[[np.inf, np.inf] for _ in range(num_modes)],
                        p=[1 / num_modes for _ in range(num_modes)],
                        logL=-np.inf,
                    )
                ]
                aic = np.inf
                num_iterations = 0

            all_params.append(params)
            iterations.append(num_iterations)
            aic_vals.append(aic)
        min_aic_idx = aic_vals.index(min(aic_vals))
        opt_params = all_params[min_aic_idx]
        num_iterations_final = iterations[min_aic_idx]
        num_sv = len(opt_params[0].mu)

    final_params = opt_params[-1]

    responsibility, x_by_mode = assign_values_to_modes(
        x, num_sv, final_params.mu, final_params.cov, final_params.p
    )

    return EstimatedGMM2D(
        mu=final_params.mu,
        cov=final_params.cov,
        p=final_params.p,
        num_modes=num_sv,
        logL=final_params.logL,
        aic=min(aic_vals) if len(aic_vals) > 0 else None,
        outliers=outliers,
        window_size=(min(x[:, 0]), max(x[:, 0])),
        x_by_mode=x_by_mode,
        responsibility=responsibility,
        num_pruned=[0 for _ in range(num_sv)],
        num_iterations=num_iterations_final,
    )
