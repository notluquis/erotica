"""Tests for cosmic.core._error_aware — error-aware pseudo-probability."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import QTable

from cosmic.core._error_aware import (
    _cholesky_factors,
    error_aware_pseudoprobability,
    gaia_covariance,
)


def _field_with_cluster(seed: int = 0):
    """Tight cluster at the origin (first `ncl`) embedded in a broad uniform field."""
    rng = np.random.default_rng(seed)
    ncl = 120
    cluster = rng.normal(0.0, 0.3, size=(ncl, 2))
    field = rng.uniform(-6.0, 6.0, size=(200, 2))
    return np.vstack([cluster, field]), ncl


# ---------------------------------------------------------------------------
# Core behaviour: errors must lower and hedge a star's recovery frequency
# ---------------------------------------------------------------------------

def test_high_error_star_gets_lower_and_more_uncertain_frequency():
    X, ncl = _field_with_cluster()
    n = len(X)
    errors = np.full((n, 2), 0.05)  # tight astrometry for everyone...
    errors[0] = 3.0                 # ...except one wildly uncertain cluster member

    f_mean, f_std = error_aware_pseudoprobability(
        X, errors=errors, n_mc=30,
        min_cluster_size_samples=range(10, 40, 5),
        min_samples=5, random_state=1,
    )

    peers = f_mean[1:ncl]  # tight-error cluster members
    # the injected star is recovered less often than a typical member...
    assert f_mean[0] < peers.mean() - peers.std()
    # ...and its membership is markedly more uncertain across draws
    assert f_std[0] > np.median(f_std[1:ncl])


def test_error_free_members_are_recovered_almost_always():
    X, ncl = _field_with_cluster(seed=3)
    n = len(X)
    f_mean, _ = error_aware_pseudoprobability(
        X, errors=np.full((n, 2), 0.02), n_mc=15,
        min_cluster_size_samples=range(10, 40, 5),
        min_samples=5, random_state=2,
    )
    assert f_mean[:ncl].mean() > 0.8          # cluster core reliably clustered
    assert f_mean[:ncl].mean() > f_mean[ncl:].mean()  # and above the field


# ---------------------------------------------------------------------------
# Consistency: cov path == diagonal-errors path when uncorrelated
# ---------------------------------------------------------------------------

def test_cov_and_diag_errors_agree_when_uncorrelated():
    X, _ = _field_with_cluster(seed=2)
    n = len(X)
    sigma = 0.1
    errors = np.full((n, 2), sigma)
    cov = np.zeros((n, 2, 2))
    cov[:, 0, 0] = cov[:, 1, 1] = sigma**2
    kw = dict(n_mc=12, min_cluster_size_samples=range(10, 30, 5),
              min_samples=5, random_state=7)

    f_err, s_err = error_aware_pseudoprobability(X, errors=errors, **kw)
    f_cov, s_cov = error_aware_pseudoprobability(X, cov=cov, **kw)
    # same RNG stream + identical (diagonal) Cholesky -> identical perturbations
    assert np.allclose(f_err, f_cov)
    assert np.allclose(s_err, s_cov)


# ---------------------------------------------------------------------------
# The headline: correlated-covariance sampling (LL^T = Sigma, correlated draws)
# ---------------------------------------------------------------------------

def test_cholesky_reconstructs_correlated_covariance():
    rho = 0.9
    sa, sb = 0.3, 0.5
    cov = np.array([[[sa**2, rho * sa * sb], [rho * sa * sb, sb**2]]])  # (1, 2, 2)
    L = _cholesky_factors(cov)
    assert np.allclose(L @ L.transpose(0, 2, 1), cov, atol=1e-6)  # LL^T = Sigma


def test_correlated_sampling_recovers_rho():
    """Draws through the exact _cholesky_factors + einsum path must be rho-correlated."""
    rho = 0.9
    sa, sb = 0.3, 0.5
    n = 40000
    one = np.array([[sa**2, rho * sa * sb], [rho * sa * sb, sb**2]])
    cov = np.repeat(one[None], n, axis=0)          # same covariance for every source
    L = _cholesky_factors(cov)
    rng = np.random.default_rng(0)
    z = rng.standard_normal((n, 2))
    noise = np.einsum("nij,nj->ni", L, z)          # identical formula to the sampler
    emp = np.corrcoef(noise[:, 0], noise[:, 1])[0, 1]
    assert abs(emp - rho) < 0.02                    # correlated draws actually flow through
    assert abs(noise[:, 0].std() - sa) < 0.02
    assert abs(noise[:, 1].std() - sb) < 0.02


# ---------------------------------------------------------------------------
# Covariance assembly from Gaia columns
# ---------------------------------------------------------------------------

def test_gaia_covariance_uses_errors_and_correlations():
    t = QTable({
        "pmra": [1.0, 2.0], "pmdec": [1.0, 2.0],
        "pmra_error": [0.2, 0.3], "pmdec_error": [0.4, 0.1],
        "pmra_pmdec_corr": [0.5, -0.5],
    })
    cov = gaia_covariance(t, ["pmra", "pmdec"])
    assert cov.shape == (2, 2, 2)
    assert np.allclose(cov[0, 0, 0], 0.2**2)
    assert np.allclose(cov[0, 1, 1], 0.4**2)
    assert np.allclose(cov[0, 0, 1], 0.5 * 0.2 * 0.4)   # rho * sig_a * sig_b
    assert np.allclose(cov[1, 0, 1], -0.5 * 0.3 * 0.1)
    assert np.allclose(cov[0, 0, 1], cov[0, 1, 0])       # symmetric


def test_gaia_covariance_missing_corr_is_zero_and_nan_error_unperturbed():
    t = QTable({
        "pmra": [1.0], "pmdec": [1.0],
        "pmra_error": [np.nan], "pmdec_error": [0.4],  # missing pmra error
    })
    cov = gaia_covariance(t, ["pmra", "pmdec"])
    assert cov[0, 0, 0] == 0.0            # NaN error -> zero variance (unperturbed)
    assert np.allclose(cov[0, 1, 1], 0.16)
    assert cov[0, 0, 1] == 0.0            # no corr column -> zero off-diagonal


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_requires_cov_or_errors():
    X, _ = _field_with_cluster(seed=5)
    with pytest.raises(ValueError, match="cov.*errors"):
        error_aware_pseudoprobability(X, n_mc=2, min_cluster_size_samples=range(10, 20, 5))
