import logging

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# |rho| = 0.7 corresponds to R^2 ~ 0.49, i.e. roughly 50 % shared variance.
# This is the conventional boundary for "strong" correlation in statistics
# and a natural threshold: above it observations are more redundant than not,
# below it they are more independent than not.  Combined with complete linkage
# (every pair in a cluster must exceed this) the effective within-cluster
# correlations are well above 0.7, producing meaningful Kish scaling factors.
DEFAULT_MIN_ABS_CORRELATION = 0.7


def ledoit_wolf_correlation(
    X: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute a shrunk correlation matrix using sklearn's Ledoit-Wolf
    covariance estimator, then standardize to correlation.

    Args:
        X: Data matrix of shape (n_samples, n_features).

    Returns:
        Correlation matrix of shape (n_features, n_features).
    """
    shrunk_cov = LedoitWolf().fit(X).covariance_
    std = np.sqrt(np.diag(shrunk_cov))
    zero_var = std == 0.0
    std = np.maximum(std, 1e-10)
    corr = shrunk_cov / np.outer(std, std)
    corr[zero_var, :] = 0.0
    corr[:, zero_var] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


def cluster_responses(
    correlation: npt.NDArray[np.float64],
    min_abs_correlation: float = DEFAULT_MIN_ABS_CORRELATION,
) -> npt.NDArray[np.int_]:
    """Cluster observations using a correlation-threshold cut.

    The distance is defined as d = 1 - |correlation|. The hierarchical tree
    is built with complete linkage, then cut at the distance threshold
    t = 1 - min_abs_correlation. This yields clusters where members are only
    merged when their within-cluster correlation structure remains above the
    requested minimum absolute correlation.

    Args:
        correlation: Correlation matrix of shape
            (n_observations, n_observations).
        min_abs_correlation: Minimum absolute correlation required for
            observations to be grouped together. Must lie in [0.0, 1.0].

    Returns:
        Array of cluster assignments for each observation.
    """
    if not 0.0 <= min_abs_correlation <= 1.0:
        raise ValueError("min_abs_correlation must be between 0.0 and 1.0")

    distance_matrix = 1.0 - np.abs(correlation)
    # Self-distance must be exactly zero.
    np.fill_diagonal(distance_matrix, 0.0)
    condensed_dist = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method="complete")
    distance_threshold = 1.0 - min_abs_correlation
    return fcluster(linkage_matrix, t=distance_threshold, criterion="distance")


def get_kish_scaling_factor(
    correlation: npt.NDArray[np.float64],
) -> float:
    """Compute the scaling factor using Kish's design effect formula.

    For a cluster with correlation matrix C:
        rho_bar = mean absolute off-diagonal correlation
        gamma = 1 + (N - 1) * rho_bar
        scaling_factor = sqrt(gamma)

    Args:
        correlation: Correlation sub-matrix for a single cluster,
            shape (n_cluster, n_cluster).

    Returns:
        sqrt(gamma), the observation-error inflation factor.
    """
    n = correlation.shape[0]
    if n <= 1:
        return 1.0
    abs_corr = np.abs(correlation)
    off_diag_sum = np.sum(abs_corr) - n  # subtract diagonal (all 1s)
    rho_bar = off_diag_sum / (n * (n - 1))
    gamma = 1.0 + (n - 1) * rho_bar
    return float(np.sqrt(gamma))


def main(
    responses: npt.NDArray[np.float64],
    obs_errors: npt.NDArray[np.float64],
    min_abs_correlation: float = DEFAULT_MIN_ABS_CORRELATION,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]:
    """
    Perform 'Auto Scaling' to mitigate issues with correlated observations
    in ensemble smoothers.

    The procedure involves several steps:

    1. Correlation Estimation:
        A shrunk covariance matrix is estimated using sklearn's Ledoit-Wolf
        estimator, then standardized to a correlation matrix. This is more
        robust than the sample correlation when the number of realizations
        is small relative to the number of observations.
    2. Threshold-Based Clustering:
        Hierarchical clustering is performed with distance
        d = 1 - |correlation| and complete linkage. The tree is cut at
        the threshold t = 1 - min_abs_correlation so that only sufficiently
        correlated observations are grouped together.
    3. Kish Scaling:
        For each cluster, Kish's design effect formula is used to compute
        the effective sample size from the mean absolute off-diagonal
        correlation within the cluster. The scaling factor
        sqrt(1 + (N-1)*rho_bar) inflates observation errors to account
        for redundancy.

    Parameters:
    -----------
    responses : npt.NDArray[np.float_]
        2D array of response data. Shape: (n_observations, n_realizations)
    obs_errors : npt.NDArray[np.float_]
        1D array of observation errors. Length: n_observations
    min_abs_correlation : float
        Minimum absolute correlation required for observations to be placed
        in the same cluster.

    Returns:
    --------
    Tuple[npt.NDArray[np.float_], npt.NDArray[np.int_]]
        - scale_factors: Array of scaling factors for observation errors
        - clusters: Array of cluster assignments for each observation
    """
    nr_obs = len(obs_errors)
    scale_factors = np.ones(nr_obs)

    if nr_obs <= 2:
        logger.info("Observations not correlated or only correlated each other")
        return scale_factors, np.ones(nr_obs, dtype=int)

    correlation = ledoit_wolf_correlation(responses.T)

    clusters = cluster_responses(correlation, min_abs_correlation)

    for cluster in np.unique(clusters):
        index = np.where(clusters == cluster)[0]
        if len(index) == 1:
            continue
        sub_corr = correlation[np.ix_(index, index)]
        scale_factor = get_kish_scaling_factor(sub_corr)
        scale_factors[index] = scale_factor
    logger.info(
        f"Calculated scaling factors for {nr_obs} observations "
        f"in {len(np.unique(clusters))} clusters"
    )
    return scale_factors, clusters
