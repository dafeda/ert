import logging

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def get_scaling_factor(nr_observations: int, effective_rank: float) -> float:
    """Calculate the observation-error scaling factor.

    Parameters
    ----------
    nr_observations
        Number of observations in the cluster.
    effective_rank
        Participation-ratio effective rank. A zero value is treated as one.

    Returns
    -------
    float
        Standard-deviation multiplier for the observation errors.
    """
    if effective_rank == 0:
        effective_rank = 1.0
        logger.warning(
            "Participation rank is 0. "
            "Setting to 1 to avoid division by zero "
            "when calculating scaling factor"
        )

    return float(np.sqrt(nr_observations / effective_rank))


def get_participation_rank(responses: npt.NDArray[np.float64]) -> float:
    """Return the participation rank of the Pearson response correlation.

    The smaller Gram matrix has the same nonzero spectrum as the response
    correlation matrix up to a common factor. Its squared trace divided by
    its squared Frobenius norm therefore gives the same participation rank
    without an eigendecomposition or a large observation-space matrix.

    Parameters
    ----------
    responses
        Response samples with shape (n_realizations, n_observations).

    Returns
    -------
    float
        Effective rank bounded by one and the centered ensemble rank limit.

    Raises
    ------
    ValueError
        If the matrix is empty, has fewer than two realizations, or contains
        nonfinite responses or an observation without ensemble spread.
    """
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim != 2 or responses.shape[0] < 2 or responses.shape[1] == 0:
        raise ValueError(
            "Responses must contain at least two realizations and one observation"
        )
    if not np.isfinite(responses).all():
        raise ValueError("Responses must be finite and have positive ensemble spread")
    centered = responses - responses.mean(axis=0)
    std = centered.std(axis=0)
    if np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("Responses must be finite and have positive ensemble spread")
    standardized = centered / std
    nr_realizations, nr_observations = responses.shape
    gram = (
        standardized @ standardized.T
        if nr_realizations < nr_observations
        else standardized.T @ standardized
    )
    rank = np.trace(gram) ** 2 / np.sum(gram**2)
    return float(np.clip(rank, 1.0, min(nr_observations, nr_realizations - 1)))


def cluster_responses(
    responses: npt.NDArray[np.float64],
    nr_clusters: int,
) -> npt.NDArray[np.int_]:
    """Cluster responses using direct absolute Spearman distance.

    Parameters
    ----------
    responses
        Response samples with shape (n_realizations, n_observations).
    nr_clusters
        Maximum number of clusters requested from average-linkage clustering.

    Returns
    -------
    ndarray
        One-based cluster labels. Tied distances can produce fewer clusters
        than requested.
    """
    correlation = spearmanr(responses).statistic
    if np.ndim(correlation) == 0:
        correlation = np.array([[1, correlation], [correlation, 1]])
    distance = 1.0 - np.abs(correlation)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    np.clip(distance, 0.0, None, out=distance)
    linkage_matrix = linkage(squareform(distance), method="average")
    return fcluster(linkage_matrix, nr_clusters, criterion="maxclust")


def main(
    responses: npt.NDArray[np.float64],
    obs_errors: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """Calculate participation-ratio Auto Scaling for one observation group.

    The ceiling of the global Pearson participation rank bounds the cluster
    count. Clustering uses direct absolute Spearman distance and average
    linkage. Each cluster's errors receive the square root of its size divided
    by its unrounded Pearson participation rank. Factors are uncapped and
    recomputed from the current responses on every call.

    Parameters
    ----------
    responses
        Current response ensemble with shape (n_observations, n_realizations).
        Active observations must have finite responses and positive spread.
    obs_errors
        Observation-error standard deviations. Only their count is used.
        The caller applies the returned multipliers to the errors.

    Returns
    -------
    scale_factors
        Standard-deviation multiplier for each observation.
    clusters
        One-based cluster label for each observation.
    effective_ranks
        Unrounded participation rank of each observation's cluster. Input
        groups with at most two observations retain unit factors and ranks.

    Raises
    ------
    ValueError
        If the response shape does not match the errors, there are fewer than
        two realizations, or an active correlation is undefined.
    """
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim != 2 or responses.shape[0] != len(obs_errors):
        raise ValueError("Response rows must match the number of observation errors")
    if responses.shape[1] < 2:
        raise ValueError("Responses must contain at least two realizations")
    scale_factors = np.ones(len(obs_errors))
    effective_ranks = np.ones(len(obs_errors))

    if len(obs_errors) <= 2:
        logger.info("Skipping auto scaling for two or fewer observations")
        return scale_factors, np.ones(len(obs_errors), dtype=int), effective_ranks

    global_rank = get_participation_rank(responses.T)
    requested_clusters = int(np.ceil(global_rank))
    clusters = cluster_responses(responses.T, nr_clusters=requested_clusters)

    for cluster in np.unique(clusters):
        index = np.where(clusters == cluster)[0]
        rank = 1.0 if len(index) == 1 else get_participation_rank(responses[index].T)
        effective_ranks[index] = rank
        scale_factors[index] = get_scaling_factor(len(index), rank)
    logger.info(
        "Participation-ratio auto scaling for %d observations with global rank %.6f "
        "requested %d clusters and produced %d clusters. "
        "Standard-deviation multipliers range from %.6f to %.6f "
        "with mean variance multiplier %.6f",
        len(scale_factors),
        global_rank,
        requested_clusters,
        len(np.unique(clusters)),
        scale_factors.min(),
        scale_factors.max(),
        np.mean(scale_factors**2),
    )
    return scale_factors, clusters, effective_ranks
