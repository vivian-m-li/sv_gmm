import numpy as np
from scipy.special import logsumexp
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
    if num_modes == 1:
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
    if num_modes == 1:
        return 0.0

    loss = 0
    for i in range(num_modes):
        for j in range(i + 1, num_modes):
            sv1, sv2 = get_sv_from_mu(mu[i], L, R), get_sv_from_mu(mu[j], L, R)
            r = reciprocal_overlap(sv1, sv2)
            r_norm = np.maximum(0, r - cutoff) / (1 - cutoff)
            loss += max_penalty * (r_norm**2)
    return loss


def sample_size_scale(sample_size: int, n_pivot: int = 100) -> float:
    """
    Scale the model penalty based on sample size to prevent over-penalizing small datasets where noise can cause clusters to be closer together, and under-penalizing large datasets where we have more confidence in the cluster estimates.

    n_pivot is the sample size at which no scaling is applied (scale = 1.0)
    """
    return np.log(sample_size) / np.log(n_pivot)


def model_penalty(
    mu: list[np.ndarray],
    sample_size: int,
    L: int,
    R: int,
    d_threshold: int,
    r_threshold: float,
    max_penalty: int,
) -> float:
    """
    Aggregate model penalties from distance and reciprocal overlap into a single score.
    """
    # penalties apply only to models with multiple clusters
    if len(mu) == 1:
        return 0.0

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
    svlen_bounds = [d_threshold, d_threshold * 2]
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

    penalty = dist_weight * d_penalty + overlap_weight * r_penalty
    return penalty


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
        log_pdfs = []
        for k in range(num_modes):
            log_pdf_i = None
            iters = 0
            while log_pdf_i is None:
                try:
                    log_pdf_i = np.log(p[k]) + multivariate_normal.logpdf(
                        x[i], mu[k], cov[k]
                    )
                except np.linalg.LinAlgError:
                    # if the covariance matrix is not positive definite, add a small value to the diagonal and try again
                    cov[k] += np.eye(cov[k].shape[0]) * 1e-6
                    iters += 1

                    # break after too many iterations
                    if iters > 10:
                        log_pdf_i = -np.inf

            log_pdfs.append(log_pdf_i)

        with np.errstate(under="ignore"):
            logL += logsumexp(log_pdfs)

    return logL

    # TODO: remove penalties
    # calculate final penalized logL
    penalty = model_penalty(
        mu, len(x), L, R, d_threshold, r_threshold, max_penalty
    )
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
    # aic = - 2ln(L) + 2k
    aic = -(2 * logL) + (2 * num_params)
    return aic


def calc_bic(
    logL: float,
    num_modes: int,
    num_samples: int,
) -> float:
    """Calculates the penalized log-likelihood using BIC."""
    num_params = (
        (num_modes * 2) + (num_modes * 3) + (num_modes - 1)
    )  # number of parameters for mu + cov + p - 1
    bic = -(2 * logL) + (num_params * np.log(num_samples))
    return bic


def calc_icl(
    logL: float,
    num_modes: int,
    num_samples: int,
    gz: np.ndarray,
) -> float:
    """Calculates the penalized log-likelihood using ICL."""
    bic = calc_bic(logL, num_modes, num_samples)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -2 * np.nansum(gz * np.log(gz))
    icl = bic + entropy
    return icl


def calc_model_score(
    logL: float,
    num_modes: int,
    num_samples: int,
    gz: np.ndarray,
    f: str = "aic",
) -> float:
    """Calculates the penalized log-likelihood using the specified model selection criterion."""
    if f == "aic":
        return calc_aic(logL, num_modes)
    elif f == "bic":
        return calc_bic(logL, num_modes, num_samples)
    elif f == "icl":
        return calc_icl(logL, num_modes, num_samples, gz)
    else:
        raise ValueError(f"Invalid model selection criterion: {f}")


def init_em_random(
    x: np.ndarray,
    num_modes: int,
    L: int,
    R: int,
    d_threshold: int,
    r_threshold: float,
    max_penalty: int,
):
    n = len(x)
    gaussian_params = []

    # copy scikit-learn's random-from-data initialization approach
    for _ in range(10):  # try up to 10 times to find valid initial means
        # randomly assign all points to a component
        assignments = np.random.randint(0, num_modes, size=n)

        # guarantee that every component gets at least one ponit
        for k in range(num_modes):
            if not np.any(assignments == k):
                # if any component has no points assigned, then assign a random point to that component
                assignments[np.random.randint(n)] = k

        responsibility = np.zeros((n, num_modes))
        responsibility[np.arange(n), assignments] = 1

        # raw counts per component
        counts = (
            np.sum(responsibility, axis=0)
            + 10 * np.finfo(responsibility.dtype).eps
        )

        mu_i = (responsibility.T @ x) / counts[:, np.newaxis]

        # # set the initial responsibility matrix
        # responsibility = np.zeros((n, num_modes))
        # indices = np.random.choice(n, num_modes, replace=False)
        # for col, index in enumerate(indices):
        #     responsibility[index, col] = 1

        # p_i = (
        #     np.sum(responsibility, axis=0)
        #     + 10 * np.finfo(responsibility.dtype).eps
        # )

        avg_X2 = (responsibility.T @ (x * x)) / counts[:, np.newaxis]
        avg_means2 = mu_i**2
        cov_i = avg_X2 - avg_means2 + RESPONSIBILITY_THRESHOLD

        # normalize weights to sum to 1
        p_i = counts / counts.sum()

        logL_i = calc_log_likelihood(
            x, mu_i, cov_i, p_i, L, R, d_threshold, r_threshold, max_penalty
        )
        gaussian_params.append((mu_i, cov_i, p_i, logL_i))

    return max(gaussian_params, key=lambda x: x[3])


def init_em(
    x: np.ndarray,
    num_modes: int,
    L: int,
    R: int,
    d_threshold: int,
    r_threshold: float,
    max_penalty: int,
    init: str,
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
        raise ValueError(
            f"Too few unique values ({n_unique}) to cluster into {num_modes}"
        )

    n = len(x)
    # force kmeans++ initialization if we only had 1 mode since we can easily calculate the mu and cov from the data without running EM
    if num_modes == 1 or "kmeans++" in init:
        # initial conditions
        kmeans = KMeans(n_clusters=num_modes, init="k-means++")

        # fit the k means model to the data and get the initial cluster centers, covariances, and proportions
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

        logL = calc_log_likelihood(
            x, mu, cov, p, L, R, d_threshold, r_threshold, max_penalty
        )
    else:  # default to random init
        mu, cov, p, logL = init_em_random(
            x, num_modes, L, R, d_threshold, r_threshold, max_penalty
        )

    # initial log-likelihood
    logLs = [logL]

    return n, mu, cov, p, logLs


def calc_responsibility(
    x: np.ndarray,
    n: int,
    mu: list[np.ndarray],
    cov: list[np.ndarray],
    p: np.ndarray,
    *,
    reassign_small_values: bool = False,
) -> np.ndarray:
    """
    Calculates the responsibility matrix for each point/mode.
    Returns a matrix of size (number of samples) x (number of modes) where each entry represents the probability that a proint belongs to a particular mode.
    If reassign_small_values is True, then we reassign points with a very small responsibility to the nearest cluster based on distance. This forces each point to contribute to the parameter estimates of at least one cluster.
    """
    gz = np.zeros((n, len(mu)))  # number of points by number of modes
    for i in range(len(x)):
        # For each point, calculate the probability that a point is from a gaussian using the mean, standard deviation, and weight of each gaussian
        densities = np.array(
            [
                p[k] * multivariate_normal.pdf(x[i], mu[k], cov[k])
                for k in range(len(mu))
            ]
        )

        if reassign_small_values and (
            np.all(densities <= RESPONSIBILITY_THRESHOLD)
            or np.any(np.isnan(densities))
        ):
            # find the nearest cluster based on distance and assign responsibility of 1 to that cluster
            distances = [np.linalg.norm(x[i] - mu[k]) for k in range(len(mu))]
            gz[i, np.argmin(distances)] = 1.0
        else:
            gz[i, :] = densities / (densities.sum() + 1e-10)
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
    repulsion: bool,  # whether to apply a repulsive force to clusters
    lambda_rep,  # strength of the repulsive force
    tau,  # interaction radius for the repulsive force
    repulsion_stepsize,  # step size for applying the repulsive force to cluster centers
) -> GMM2D:
    """Performs one iteration of the expectation-maximization algorithm."""
    # Expectation step: calculate the posterior probabilities using previous parameters (Gaussian distributions)
    gz = calc_responsibility(x, n, mu, cov, p, reassign_small_values=True)

    # Ensure that each point contributes to the responsibility matrix above some threshold
    # this avoids math errors in the maximization step
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

    # Cluster repulsion - adjust mu using a penalty function to prevent clusters from getting too close to each other and over-clustering
    if repulsion:
        repulsion_grads = [np.zeros_like(mu[k]) for k in range(num_modes)]
        for k in range(num_modes):
            for j in range(num_modes):
                if j == k:
                    continue
                diff = mu[k] - mu[j]
                dist = np.linalg.norm(diff) + 1e-8

                # decays from 1 (clusters on top of each other) to 0 (far apart)
                # tau controls the length scale: repulsion is negligible beyond -3*tau
                weight = np.exp(-dist / tau)

                # unit vector pointing k away from j, scaled by weight; lambda_rep is redundant
                repulsion_grads[k] += lambda_rep * weight * (diff / dist)

        # update mu with repulsion gradient
        for k in range(num_modes):
            mu[k] += repulsion_stepsize * repulsion_grads[k]

    # update likelihood
    logL = calc_log_likelihood(
        x, mu, cov, p, L, R, d_threshold, r_threshold, max_penalty
    )
    return GMM2D(mu, cov, p, logL)


def run_em(
    x: np.ndarray,  # data
    *,
    num_modes: int,
    L: int,
    R: int,
    d_threshold: int = 100,
    r_threshold: float = 0.8,
    max_penalty: int = 200,
    init: str = "kmeans++",
    repulsion: bool = False,
    lambda_rep: float = 1,
    tau: float = 200,
    repulsion_stepsize: float = 10.0,
) -> tuple[list[GMM2D], int]:
    """
    Given a dataset and an estimated number of modes for the GMM, estimates the parameters for each distribution.
    The algorithm is initialized using the k-means clustering approach, then the EM algorithm is run for up to 30 iterations, or a convergence of the log-likelihood; whichever comes first.
    Returns the GMM estimated by each iteration of the EM algorithm.
    """
    all_params: list[GMM2D] = []

    n, mu, cov, p, logL = init_em(
        x, num_modes, L, R, d_threshold, r_threshold, max_penalty, init
    )  # initialize parameters
    all_params.append(GMM2D(mu, cov, p, logL[0]))

    # only need to estimate mean and covariance matrix if we have 1 mode
    if num_modes == 1 or init == "kmeans++":
        return all_params, 1

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
            repulsion,
            lambda_rep,
            tau,
            repulsion_stepsize,
        )
        logL.append(gmm_result.logL)
        all_params.append(gmm_result)

        # Convergence check
        if abs(logL[-1] - logL[-2]) < 1e-3:
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
    x_index_by_mode = [[] for _ in range(num_modes)]
    for i, mode in enumerate(assignments):
        if mode is not None:
            x_by_mode[mode].append(x[i])
            x_index_by_mode[mode].append(i)
    x_by_mode = [np.array(data_points) for data_points in x_by_mode]
    return gz, x_by_mode, x_index_by_mode


def select_model(x, model_scores, all_params, iterations):
    """
    Selects the optimal GMM model based on the AIC scores. Checks that each
    model is valid by verifying that all clusters have at least one data point
    assigned to them.
    """
    # only consider models that are valid
    valid_model_scores = []
    for i, gmm_params in enumerate(all_params):
        last_iter = gmm_params[-1]
        gz = calc_responsibility(
            x, len(x), last_iter.mu, last_iter.cov, last_iter.p
        )
        assignments = np.argmax(gz, axis=1)
        if len(set(assignments)) == len(last_iter.mu):
            valid_model_scores.append(model_scores[i])
        else:
            valid_model_scores.append(np.inf)

    if np.all(np.isinf(valid_model_scores)):
        raise ValueError("No valid models found.")

    # get the valid model with the lowest AIC score
    min_score_idx = valid_model_scores.index(min(valid_model_scores))

    opt_params = all_params[min_score_idx]
    num_iterations_final = iterations[min_score_idx]
    final_params = opt_params[-1]

    return final_params, num_iterations_final


def gmm(
    x: np.ndarray[tuple[float, int]],
    *,
    L,
    R,
    d_threshold: int = 100,
    r_threshold: float = 0.8,
    max_penalty: int = 200,
    init: str = "kmeans++",
    repulsion: bool = False,
    lambda_rep: float = 1.0,
    tau: float = 200,
    repulsion_stepsize: float = 10.0,
    model_comparison_func: str = "aic",
    force_n_modes: int | None = None,
    plot: bool = False,
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
            logL=-9999999,
            score=9999999,
            outliers=[],
            window_size=(singleton, singleton),
            x_by_mode=[np.array(x)],
            x_index_by_mode=[[0]],
            responsibility=np.array([[1.0]]),
            num_pruned=[0],
            num_iterations=0,
        )

    outliers = []
    model_scores = []
    if len(x) <= 10:  # small number of samples detected
        opt_params, num_iterations = run_em(
            x, num_modes=1, L=L, R=R, init="kmeans++"
        )
        num_iterations_final = num_iterations
        num_sv = 1

        final_params = opt_params[-1]
    else:
        all_params = []
        iterations = []
        mode_options = (
            [force_n_modes] if force_n_modes is not None else range(1, 4)
        )
        for num_modes in mode_options:
            try:
                params, num_iterations = run_em(
                    x,
                    num_modes=num_modes,
                    L=L,
                    R=R,
                    d_threshold=d_threshold,
                    r_threshold=r_threshold,
                    max_penalty=max_penalty,
                    init=init,
                    repulsion=repulsion,
                    lambda_rep=lambda_rep,
                    tau=tau,
                    repulsion_stepsize=repulsion_stepsize,
                )

                responsibility = calc_responsibility(
                    x, len(x), params[-1].mu, params[-1].cov, params[-1].p
                )
                model_score = calc_model_score(
                    params[-1].logL,
                    num_modes,
                    len(x),
                    responsibility,
                    f=model_comparison_func,
                )

            except ValueError:
                params = [
                    GMM2D(
                        mu=[[0, 0] for _ in range(num_modes)],
                        cov=[[np.inf, np.inf] for _ in range(num_modes)],
                        p=[1 / num_modes for _ in range(num_modes)],
                        logL=-np.inf,
                    )
                ]
                model_score = np.inf
                num_iterations = 0

            all_params.append(params)
            iterations.append(num_iterations)
            model_scores.append(model_score)

        final_params, num_iterations_final = select_model(
            x, model_scores, all_params, iterations
        )
        num_sv = len(final_params.mu)

    responsibility, x_by_mode, x_index_by_mode = assign_values_to_modes(
        x, num_sv, final_params.mu, final_params.cov, final_params.p
    )

    best_score = calc_model_score(
        final_params.logL,
        num_sv,
        len(x),
        responsibility,
        f=model_comparison_func,
    )

    return EstimatedGMM2D(
        mu=final_params.mu,
        cov=final_params.cov,
        p=final_params.p,
        num_modes=num_sv,
        logL=final_params.logL,
        score=best_score,
        outliers=outliers,
        window_size=(min(x[:, 0]), max(x[:, 0])),
        x_by_mode=x_by_mode,
        x_index_by_mode=x_index_by_mode,
        responsibility=responsibility,
        num_pruned=[0 for _ in range(num_sv)],
        num_iterations=num_iterations_final,
    )
