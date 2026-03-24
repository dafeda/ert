import numpy as np
import pytest

from ert.analysis.misfit_preprocessor import (
    cluster_responses,
    get_kish_scaling_factor,
    ledoit_wolf_correlation,
    main,
)


@pytest.mark.parametrize("nr_observations", [4, 7, 12])
def test_that_correlated_and_independent_observations_are_grouped_separately(
    nr_observations,
):
    """
    Test the preprocessor's ability to cluster correlated observations
    separately from multiple independent observations.

    We create a response matrix with `nr_observations` rows, where the
    first `nr_observations - 2` rows are strongly correlated, while the
    last two are independent of both the main group and each other.

    This will result in a request for 3 clusters, correctly separating the
    data into its three natural groups:

    1. The main group of correlated observations.
    2. The first independent observation.
    3. The second independent observation.
    """
    rng = np.random.default_rng(1234)
    nr_realizations = 1000
    nr_uncorrelated_obs = 2
    nr_correlated_obs = nr_observations - nr_uncorrelated_obs

    parameters_a = rng.standard_normal(nr_realizations)
    parameters_b = rng.standard_normal(nr_realizations)
    parameters_c = rng.standard_normal(nr_realizations)

    Y = np.zeros((nr_observations, nr_realizations))
    for i in range(nr_correlated_obs):
        Y[i] = (i + 1) * parameters_a
    # The last two observations are independent
    Y[-2] = 10 + parameters_b
    Y[-1] = 5 + parameters_c

    obs_errors = Y.std(axis=1)
    Y_original = Y.copy()
    obs_error_copy = obs_errors.copy()

    scale_factors, clusters = main(Y, obs_errors)

    # We expect three distinct clusters now.
    cluster_label_correlated = clusters[0]
    cluster_label_independent_1 = clusters[-2]
    cluster_label_independent_2 = clusters[-1]

    # Check that the three labels are all different
    assert cluster_label_correlated != cluster_label_independent_1
    assert cluster_label_correlated != cluster_label_independent_2
    assert cluster_label_independent_1 != cluster_label_independent_2

    # Check that the main group is clustered together
    for i in range(nr_correlated_obs):
        assert clusters[i] == cluster_label_correlated

    # Perfectly correlated group: rho_bar ≈ 1, so scale_factor ≈ sqrt(N).
    # Independent singletons: scale_factor = 1.
    expected_scale_factors = np.array(
        [np.sqrt(nr_correlated_obs)] * nr_correlated_obs + [1.0] * nr_uncorrelated_obs
    )
    np.testing.assert_allclose(scale_factors, expected_scale_factors, rtol=0.005)

    np.testing.assert_equal(Y, Y_original)
    np.testing.assert_equal(obs_errors, obs_error_copy)


def test_that_two_or_fewer_observations_return_unit_scaling_factors():
    """
    Test that edge cases with 2 observations return default scaling values.
    We create an example of a response matrix where all rows are perfectly
    correlated, which should lead to a single cluster with a scaling factor of
    sqrt(1 + (N-1)*rho_bar) = sqrt(2).
    However, since the number of observations is <= 2, the function should
    skip the clustering and return default values of 1.0 for all
    scaling factors
    """
    nr_realizations = 1000
    nr_observations = 2
    Y = np.ones((nr_observations, nr_realizations))

    rng = np.random.default_rng(1234)
    parameters_a = rng.normal(10, 1, nr_realizations)

    # create response matrix
    for i in range(nr_observations):
        Y[i] = (i + 1) * parameters_a

    # Add observation errors
    obs_errors = np.ones(nr_observations) * 0.1

    scale_factors, _ = main(Y, obs_errors)

    np.testing.assert_equal(
        scale_factors,
        np.array(nr_observations * [1.0]),
    )


@pytest.mark.parametrize(
    ("nr_obs_group_a", "nr_obs_group_b"),
    [
        (3, 2),
        (5, 5),
        (4, 6),
    ],
)
def test_main_correctly_separates_distinct_correlation_groups(
    nr_obs_group_a, nr_obs_group_b
):
    """
    Creates a response matrix with two distinct and independent groups of
    correlated observations.
    - Group A contains `nr_obs_group_a` responses that are all correlated
      with each other.
    - Group B contains `nr_obs_group_b` responses that are also correlated
      with each other, but are independent of Group A.

    This test asserts that the algorithm places
    the two groups into two separate clusters.
    """
    rng = np.random.default_rng(seed=12345)
    nr_realizations = 1000
    nr_observations = nr_obs_group_a + nr_obs_group_b

    # Create two independent random signals that will form the
    # basis for the two correlation groups.
    params_a = rng.standard_normal(nr_realizations)
    params_b = rng.standard_normal(nr_realizations)

    # Create the final response matrix Y
    Y = np.zeros((nr_observations, nr_realizations))

    # Create Group A: `nr_obs_group_a` perfectly correlated
    # responses based on `params_a`
    for i in range(nr_obs_group_a):
        Y[i] = (i + 1) * params_a

    # Create Group B: `nr_obs_group_b` perfectly correlated
    # responses based on `params_b`
    for i in range(nr_obs_group_b):
        Y[nr_obs_group_a + i] = (i + 1) * params_b

    # Calculate observation errors,
    # required as input for the main function
    obs_errors = Y.std(axis=1)

    _scale_factors, clusters = main(Y, obs_errors)

    # Assert that the two groups were placed in different clusters.
    # The absolute cluster labels (e.g., 1 vs 2) can change between runs,
    # so we check the grouping structure dynamically.
    cluster_label_group_a = clusters[0]
    cluster_label_group_b = clusters[nr_obs_group_a]

    assert cluster_label_group_a != cluster_label_group_b, (
        "The two distinct correlation groups should be in different clusters."
    )

    # Assert that all members of Group A are in the same cluster
    expected_clusters_a = np.full(nr_obs_group_a, cluster_label_group_a)
    np.testing.assert_array_equal(clusters[:nr_obs_group_a], expected_clusters_a)

    # Assert that all members of Group B are in the same cluster
    expected_clusters_b = np.full(nr_obs_group_b, cluster_label_group_b)
    np.testing.assert_array_equal(clusters[nr_obs_group_a:], expected_clusters_b)


@pytest.mark.parametrize(
    ("nr_obs_group_a", "nr_obs_group_b"),
    [
        (3, 2),
        (5, 5),
        (4, 6),
    ],
)
def test_autoscale_clusters_observations_by_correlation_pattern_ignoring_sign(
    nr_obs_group_a, nr_obs_group_b
):
    """
    Creates a response matrix with two distinct and independent groups:

    - Group A contains `nr_obs_group_a` responses that are all positively
      correlated with each other, following the pattern (a+i)*X_1.
    - Group B contains `nr_obs_group_b` responses with alternating signs
      following the pattern (b+j)*(-1)^j*X_2, creating a checkerboard
      correlation pattern within the group, but independent of Group A.

    """
    rng = np.random.default_rng(seed=12345)
    nr_realizations = 1000
    nr_observations = nr_obs_group_a + nr_obs_group_b

    # Create two independent random signals
    X_1 = rng.standard_normal(nr_realizations)
    X_2 = rng.standard_normal(nr_realizations)

    Y = np.zeros((nr_observations, nr_realizations))

    # Create Group A: (a+i)*X_1 pattern - all positively correlated
    a = 1  # base scaling factor for group A
    for i in range(nr_obs_group_a):
        Y[i] = (a + i) * X_1

    # Create Group B: (b+j)*(-1)^j*X_2 pattern - checkerboard correlation
    b = 1  # base scaling factor for group B
    for j in range(nr_obs_group_b):
        sign = (-1) ** j  # alternates: +1, -1, +1, -1, ...
        Y[nr_obs_group_a + j] = (b + j) * sign * X_2

    obs_errors = Y.std(axis=1)
    _scale_factors, clusters = main(Y, obs_errors)

    # Assert that all members of Group A are in the same cluster
    group_a_clusters = clusters[:nr_obs_group_a]
    assert len(np.unique(group_a_clusters)) == 1, (
        "All Group A responses should be in the same cluster"
    )

    # Assert that all members of Group B are in the same cluster
    group_b_clusters = clusters[nr_obs_group_a:]
    assert len(np.unique(group_b_clusters)) == 1, (
        "All Group B responses should be in the same cluster"
    )

    # Assert that responses from Group A are assigned
    # to a different cluster than responses from Group B
    assert np.unique(group_a_clusters) != np.unique(group_b_clusters)


def test_that_cluster_responses_respects_the_absolute_correlation_threshold():
    """Observations with absolute correlation below the threshold should be
    separated, while stronger pairs remain clustered.
    """
    correlation = np.array(
        [
            [1.0, 0.92, 0.10, 0.05],
            [0.92, 1.0, 0.12, 0.08],
            [0.10, 0.12, 1.0, -0.75],
            [0.05, 0.08, -0.75, 1.0],
        ]
    )

    clusters = cluster_responses(correlation, min_abs_correlation=0.7)

    assert clusters[0] == clusters[1]
    assert clusters[2] == clusters[3]
    assert clusters[0] != clusters[2]


def test_that_cluster_responses_with_complete_linkage_prevents_chain_merges():
    """A threshold cut with complete linkage should not merge a chain of
    pairwise-acceptable observations when the end points violate the threshold.
    """
    correlation = np.array(
        [
            [1.0, 0.90, 0.60],
            [0.90, 1.0, 0.90],
            [0.60, 0.90, 1.0],
        ]
    )

    clusters = cluster_responses(correlation, min_abs_correlation=0.7)

    assert len(np.unique(clusters)) == 2
    assert clusters[0] != clusters[2]


def test_that_cluster_responses_rejects_invalid_minimum_absolute_correlation():
    correlation = np.eye(3)

    with pytest.raises(
        ValueError, match=r"min_abs_correlation must be between 0\.0 and 1\.0"
    ):
        cluster_responses(correlation, min_abs_correlation=1.1)


def test_that_seismic_and_pressure_groups_are_separated_into_two_clusters():
    """
    Integration test for the autoscaler with two independent observation
    groups.

    Scenario:
    500 noisy seismic observations and 20 precise well pressure observations.
    All seismic observations are correlated with each other, all pressure
    observations are correlated with each other, but the two groups are
    independent.

    The autoscaler should identify two clusters, one for seismic and one
    for pressure, and assign scaling factors based on within-cluster
    correlation structure.
    """

    rng = np.random.default_rng(42)
    n_realizations = 100

    n_seismic = 500
    n_pressure = 20

    # Two independent underlying parameters
    param_shallow = rng.normal(0, 1, size=(n_realizations, 1))
    param_deep = rng.normal(0, 1, size=(n_realizations, 1))

    # Seismic: sensitive to shallow param
    seismic_sensitivity = rng.uniform(0.5, 1.5, size=(1, n_seismic))
    seismic_responses = param_shallow @ seismic_sensitivity
    seismic_responses += rng.normal(0, 0.1, size=(n_realizations, n_seismic))

    # Pressure: sensitive to deep param (independent of seismic)
    pressure_sensitivity = rng.uniform(0.5, 1.5, size=(1, n_pressure))
    pressure_responses = param_deep @ pressure_sensitivity
    pressure_responses += rng.normal(0, 0.1, size=(n_realizations, n_pressure))

    responses = np.hstack([seismic_responses, pressure_responses])

    # Observation errors: seismic is NOISY, pressure is PRECISE
    seismic_errors = np.full(n_seismic, 2.0)
    pressure_errors = np.full(n_pressure, 0.05)
    obs_errors = np.hstack([seismic_errors, pressure_errors])

    _scale_factors, clusters = main(responses.T, obs_errors)

    # The correlation-threshold cut correctly identifies two clusters
    assert len(np.unique(clusters)) == 2

    # Seismic observations share one cluster, pressure observations another
    seismic_cluster = clusters[0]
    pressure_cluster = clusters[n_seismic]
    assert seismic_cluster != pressure_cluster
    assert np.all(clusters[:n_seismic] == seismic_cluster)
    assert np.all(clusters[n_seismic:] == pressure_cluster)


def test_that_independent_observations_with_irregular_errors_get_unit_scaling():
    """
    Test that independent observations with irregular errors are not
    incorrectly clustered together.

    Scenario:
    100 independent observations where r_1 has a small error and
    r_2,...,r_100 have large errors. Since the observations are
    independent, they should not receive large scaling factors.
    """

    # Create 100 independent responses
    rng = np.random.default_rng(42)
    n_observations = 100
    n_realizations = 1000
    responses = rng.standard_normal((n_observations, n_realizations))

    # Create irregular observation errors: one small, the rest large
    obs_errors = np.array([0.1] + [10.0] * (n_observations - 1))

    # Run clustering algorithm
    scale_factors, _clusters = main(responses, obs_errors)

    # For independent observations, scaling factors should be close to 1
    # (no redundancy to account for)
    np.testing.assert_allclose(scale_factors, 1.0, atol=0.5)


def test_that_main_uses_default_minimum_absolute_correlation_threshold():
    rng = np.random.default_rng(2026)
    n_realizations = 3000

    signal_a = rng.standard_normal(n_realizations)
    signal_b = rng.standard_normal(n_realizations)
    bridge_noise = rng.standard_normal(n_realizations)

    responses = np.vstack(
        [
            signal_a,
            1.1 * signal_a + 0.05 * rng.standard_normal(n_realizations),
            0.72 * signal_a + 0.72 * signal_b + 0.05 * bridge_noise,
            signal_b,
            1.2 * signal_b + 0.05 * rng.standard_normal(n_realizations),
        ]
    )
    obs_errors = np.ones(responses.shape[0])

    _scale_factors, clusters = main(responses, obs_errors)

    assert clusters[0] == clusters[1]
    assert clusters[3] == clusters[4]
    assert clusters[0] != clusters[3]


@pytest.mark.parametrize(
    ("n_obs", "rho"),
    [
        (2, 0.0),
        (2, 0.5),
        (2, 1.0),
        (5, 0.3),
        (5, 0.7),
        (10, 0.4),
    ],
)
def test_that_kish_scaling_equals_one_plus_n_minus_one_times_rho_bar(n_obs, rho):
    """Verify that get_kish_scaling_factor returns sqrt(1 + (N-1)*rho_bar)
    for an equicorrelation matrix with off-diagonal entries equal to rho.

    This is the exact Kish design effect formula (Option B in the paper).
    For rho=0 the scaling factor is 1 (independent observations).
    For rho=1 it equals sqrt(N) (fully redundant).
    """
    corr = np.full((n_obs, n_obs), rho)
    np.fill_diagonal(corr, 1.0)

    expected_gamma = 1.0 + (n_obs - 1) * rho
    expected_sf = np.sqrt(expected_gamma)

    sf = get_kish_scaling_factor(corr)
    np.testing.assert_allclose(sf, expected_sf, rtol=1e-12)


def test_that_ledoit_wolf_returns_identity_for_uncorrelated_features():
    """When features are drawn independently and the sample size is much
    larger than the number of features, Ledoit-Wolf shrinkage should
    produce a correlation matrix close to the identity.
    """
    rng = np.random.default_rng(42)
    n_samples = 10000
    n_features = 5
    X = rng.standard_normal((n_samples, n_features))

    corr = ledoit_wolf_correlation(X)

    assert corr.shape == (n_features, n_features)
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)

    off_diag = corr[~np.eye(n_features, dtype=bool)]
    np.testing.assert_allclose(off_diag, 0.0, atol=0.05)


def test_that_ledoit_wolf_recovers_strong_correlation():
    """When two features are generated from the same underlying signal,
    the Ledoit-Wolf estimator should recover a correlation close to 1.
    """
    rng = np.random.default_rng(42)
    n_samples = 5000
    signal = rng.standard_normal(n_samples)
    X = np.column_stack([signal, signal + 0.01 * rng.standard_normal(n_samples)])

    corr = ledoit_wolf_correlation(X)

    assert corr.shape == (2, 2)
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)
    assert corr[0, 1] > 0.99
    assert corr[1, 0] > 0.99


def test_that_ledoit_wolf_returns_identity_for_constant_features():
    """When all features are constant (zero variance), the estimator
    should return the identity matrix without errors.
    """
    X = np.ones((100, 3))
    corr = ledoit_wolf_correlation(X)

    np.testing.assert_array_equal(corr, np.eye(3))
