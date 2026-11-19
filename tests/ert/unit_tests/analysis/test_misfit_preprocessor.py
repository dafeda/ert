"""Check participation-ratio ranks, direct clustering, and error multipliers."""

import logging

import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from ert.analysis import misfit_preprocessor
from ert.analysis.misfit_preprocessor import (
    cluster_responses,
    get_participation_rank,
    get_scaling_factor,
    main,
)


def _responses_with_correlation(correlation):
    """Construct centered response rows with a prescribed positive-definite correlation.

    Parameters
    ----------
    correlation
        Target response-correlation matrix.

    Returns
    -------
    ndarray
        Observation rows with one more realization than observations.
    """
    basis = np.eye(len(correlation) + 1)[:, :-1]
    basis -= basis.mean(axis=0)
    orthonormal = np.linalg.qr(basis)[0].T
    return np.linalg.cholesky(correlation) @ orthonormal


def test_get_scaling_factor():
    assert get_scaling_factor(8, 2.0) == 2
    assert get_scaling_factor(4, 0.0) == 2
    assert get_scaling_factor(4, 1.0) == 2
    assert get_scaling_factor(5, 1.25) == 2


@pytest.mark.parametrize(
    ("count", "rho"), [(4, -0.2), (4, 0.0), (4, 0.3), (10, 0.1), (5, 0.8)]
)
def test_participation_rank_matches_equicorrelated_closed_form(count, rho):
    correlation = (1 - rho) * np.eye(count) + rho * np.ones((count, count))
    responses = _responses_with_correlation(correlation)

    rank = get_participation_rank(responses.T)

    np.testing.assert_allclose(rank, count / (1 + (count - 1) * rho**2))


@pytest.mark.parametrize("shape", [(200, 12), (4, 30), (10, 1)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_participation_rank_matches_pearson_spectrum(shape, dtype):
    responses = np.random.default_rng(42).normal(size=shape).astype(dtype)
    correlation = np.atleast_2d(np.corrcoef(responses, rowvar=False))
    eigenvalues = np.linalg.eigvalsh(correlation)
    expected = eigenvalues.sum() ** 2 / np.sum(eigenvalues**2)

    rank = get_participation_rank(responses)

    np.testing.assert_allclose(rank, expected, rtol=1e-12)
    assert 1.0 <= rank <= min(shape[1], shape[0] - 1)


def test_participation_rank_uses_small_gram_without_decomposition(monkeypatch):
    def unexpected_calculation(*args, **kwargs):
        raise AssertionError("Unexpected observation-space matrix or decomposition")

    responses = np.random.default_rng(8).normal(size=(4, 2000))
    monkeypatch.setattr(np, "corrcoef", unexpected_calculation)
    monkeypatch.setattr(np.linalg, "svd", unexpected_calculation)
    monkeypatch.setattr(np.linalg, "eigvalsh", unexpected_calculation)

    assert 1.0 <= get_participation_rank(responses) <= 3.0


@pytest.mark.parametrize(
    "responses",
    [
        np.empty((5, 0)),
        np.ones((1, 5)),
        np.ones(5),
        np.ones((5, 3)),
        np.column_stack([np.arange(5), np.ones(5)]),
        np.array([[0.0, 1.0], [1.0, np.nan], [2.0, 3.0]]),
        np.array([[0.0, 1.0], [1.0, np.inf], [2.0, 3.0]]),
    ],
)
def test_participation_rank_rejects_undefined_correlations(responses):
    with pytest.raises(ValueError, match="Responses must"):
        get_participation_rank(responses)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_small_input_groups_keep_unit_factors_without_computing_correlations(count):
    scale_factors, clusters, ranks = main(np.zeros((count, 5)), np.ones(count))

    np.testing.assert_array_equal(scale_factors, np.ones(count))
    np.testing.assert_array_equal(clusters, np.ones(count, dtype=int))
    np.testing.assert_array_equal(ranks, np.ones(count))
    assert ranks.dtype == np.float64


@pytest.mark.parametrize("count", [3, 5, 12])
def test_identical_and_sign_reversed_responses_have_rank_one(count):
    latent = np.random.default_rng(7).normal(size=100)
    responses = (-2.0) ** np.arange(count)[:, None] * latent + np.arange(count)[:, None]

    scale_factors, clusters, ranks = main(responses, np.ones(count))

    assert len(np.unique(clusters)) == 1
    np.testing.assert_allclose(ranks, 1.0)
    np.testing.assert_allclose(scale_factors, np.sqrt(count))


def test_orthogonal_responses_keep_unit_factors():
    responses = _responses_with_correlation(np.eye(6))

    scale_factors, clusters, ranks = main(responses, np.ones(6))

    np.testing.assert_allclose(scale_factors, 1.0)
    for label in np.unique(clusters):
        members = clusters == label
        np.testing.assert_allclose(ranks[members], members.sum())


def test_global_pearson_rank_bounds_direct_spearman_clusters_and_cluster_factors():
    rng = np.random.default_rng(42)
    responses = rng.normal(size=200) + 0.4 * rng.normal(size=(30, 200))
    correlation = np.corrcoef(responses)
    global_rank = len(responses) ** 2 / np.sum(correlation**2)
    requested = int(np.ceil(global_rank))
    distance = 1.0 - np.abs(spearmanr(responses, axis=1).statistic)
    np.fill_diagonal(distance, 0.0)
    expected_labels = fcluster(
        linkage(squareform(distance), method="average"), requested, criterion="maxclust"
    )

    scale_factors, clusters, ranks = main(responses, np.ones(30))

    np.testing.assert_array_equal(clusters, expected_labels)
    assert 1 <= len(np.unique(clusters)) <= requested
    assert np.any(np.abs(ranks - np.round(ranks)) > 1e-3)
    for label in np.unique(clusters):
        members = np.flatnonzero(clusters == label)
        subcorrelation = correlation[np.ix_(members, members)]
        gamma = np.sum(subcorrelation**2) / len(members)
        np.testing.assert_allclose(ranks[members], len(members) / gamma)
        np.testing.assert_allclose(scale_factors[members] ** 2, gamma)


def test_global_count_can_merge_independent_signals_when_group_sizes_are_unbalanced(
    monkeypatch,
):
    independent = _responses_with_correlation(np.eye(3))
    responses = np.vstack([np.tile(independent[0], (10, 1)), independent[1:]])
    requested = []
    calculate_clusters = cluster_responses

    def record_cluster_count(responses, nr_clusters):
        requested.append(nr_clusters)
        return calculate_clusters(responses, nr_clusters)

    monkeypatch.setattr(misfit_preprocessor, "cluster_responses", record_cluster_count)

    _, clusters, _ = main(responses, np.ones(12))

    np.testing.assert_allclose(get_participation_rank(responses.T), 144 / 102)
    assert requested == [2]
    assert len(np.unique(clusters)) == 2


@pytest.mark.parametrize("sizes", [(3, 2), (5, 5), (4, 6)])
@pytest.mark.parametrize("reverse_signs", [False, True])
def test_distinct_correlated_groups_are_separated(sizes, reverse_signs):
    latent = np.random.default_rng(12345).normal(size=(2, 1000))
    groups = []
    for size, signal in zip(sizes, latent, strict=True):
        scales = np.arange(1, size + 1)
        if reverse_signs:
            scales *= (-1) ** np.arange(size)
        groups.append(scales[:, None] * signal)
    responses = np.vstack(groups)

    scale_factors, clusters, ranks = main(responses, np.ones(sum(sizes)))

    assert len(np.unique(clusters)) == 2
    assert clusters[0] != clusters[sizes[0]]
    np.testing.assert_array_equal(clusters[: sizes[0]], clusters[0])
    np.testing.assert_array_equal(clusters[sizes[0] :], clusters[sizes[0]])
    np.testing.assert_allclose(ranks, 1.0)
    np.testing.assert_allclose(scale_factors[: sizes[0]], np.sqrt(sizes[0]))
    np.testing.assert_allclose(scale_factors[sizes[0] :], np.sqrt(sizes[1]))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scaling_is_invariant_to_units_sign_offsets_and_error_magnitudes(dtype):
    responses = np.random.default_rng(23).normal(size=(12, 100)).astype(dtype)
    errors = np.ones(12)
    original_responses = responses.copy()
    original_errors = errors.copy()
    scales = (-1.5) ** np.arange(12)
    transformed = responses.astype(float) * scales[:, None] + np.arange(12)[:, None]

    original = main(responses, errors)
    changed = main(transformed, np.geomspace(0.01, 100.0, 12))

    for actual, expected in zip(changed, original, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_array_equal(responses, original_responses)
    np.testing.assert_array_equal(errors, original_errors)


@pytest.mark.parametrize("corr_de_target", [0.85, 0.8, 0.6, 0.4, 0.3])
def test_direct_clustering_merges_most_spearman_correlated_pair_first(corr_de_target):
    rng = np.random.default_rng(42)
    a, c, d, noise = rng.normal(size=(4, 10000))
    b = a + c
    e = corr_de_target * d + np.sqrt(1 - corr_de_target**2) * noise
    responses = np.array([a, b, c, d, e]).T
    absolute_correlation = np.abs(spearmanr(responses).statistic)
    np.fill_diagonal(absolute_correlation, -np.inf)
    expected_pair = tuple(
        sorted(
            np.unravel_index(absolute_correlation.argmax(), absolute_correlation.shape)
        )
    )

    clusters = cluster_responses(responses, nr_clusters=4)

    merged_pairs = [
        (i, j)
        for i in range(len(clusters))
        for j in range(i + 1, len(clusters))
        if clusters[i] == clusters[j]
    ]
    assert merged_pairs == [expected_pair]


@pytest.mark.parametrize("count", [2, 8])
def test_direct_clustering_handles_tied_response_values(count):
    responses = np.random.default_rng(2).integers(0, 3, size=(15, count)).astype(float)

    clusters = cluster_responses(responses, nr_clusters=2)

    assert clusters.shape == (count,)
    assert set(clusters).issubset({1, 2})


@pytest.mark.slow
def test_noisy_seismic_and_pressure_groups_use_fractional_cluster_ranks():
    rng = np.random.default_rng(42)
    latent = rng.normal(size=(2, 100))
    responses = np.vstack(
        [
            latent[0] + rng.normal(0, 0.1, (500, 100)),
            latent[1] + rng.normal(0, 0.1, (20, 100)),
        ]
    )
    errors = np.concatenate([np.full(500, 2.0), np.full(20, 0.05)])

    scale_factors, clusters, ranks = main(responses, errors)

    assert len(np.unique(clusters)) == 2
    assert clusters[0] != clusters[500]
    np.testing.assert_array_equal(clusters[:500], clusters[0])
    np.testing.assert_array_equal(clusters[500:], clusters[500])
    for members in (slice(0, 500), slice(500, None)):
        correlation = np.corrcoef(responses[members])
        gamma = np.sum(correlation**2) / len(correlation)
        np.testing.assert_allclose(scale_factors[members] ** 2, gamma)
        np.testing.assert_allclose(ranks[members], len(correlation) / gamma)
        assert np.all(ranks[members] > 1.0)


@pytest.mark.parametrize(
    ("responses", "errors", "message"),
    [
        (np.ones(3), np.ones(3), "Response rows"),
        (np.ones((3, 8)), np.ones(2), "Response rows"),
        (np.ones((3, 1)), np.ones(3), "at least two realizations"),
        (np.ones((3, 8)), np.ones(3), "positive ensemble spread"),
        (np.full((3, 8), np.nan), np.ones(3), "Responses must be finite"),
    ],
)
def test_main_rejects_invalid_active_responses(responses, errors, message):
    with pytest.raises(ValueError, match=message):
        main(responses, errors)


def test_logs_global_rank_cluster_counts_and_factor_summaries(caplog):
    responses = np.random.default_rng(8).normal(size=(12, 40))

    with caplog.at_level(logging.INFO, logger=misfit_preprocessor.__name__):
        _, clusters, _ = main(responses, np.ones(12))

    requested = int(np.ceil(get_participation_rank(responses.T)))
    assert "Participation-ratio auto scaling" in caplog.text
    assert f"requested {requested} clusters" in caplog.text
    assert f"produced {len(np.unique(clusters))} clusters" in caplog.text
    assert "global rank" in caplog.text
    assert "mean variance multiplier" in caplog.text
