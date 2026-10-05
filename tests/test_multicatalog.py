"""Tests for erotica.multicatalog. Oracles are closed forms that do not use the GLS algebra.

1. Independent, diagonal errors: the combination is the per-axis inverse-variance mean (Pena Ramirez
   et al. 2022 precedent), written here as 1/(1/a + 1/b) by hand.
2. Full inheritance (e_V = e_G + eta): the second catalogue carries no information, so the result is
   exactly the first catalogue.
3. The offset is subtracted: shifting mu_v by delta and passing offset=delta changes nothing.
4. Monte Carlo: with errors drawn from the declared joint covariance (lam = 0.5), the scatter of
   mu_hat about the truth matches C_hat (Hotelling-type chi^2 on 20000 draws, fixed seed).
5. The difference log-likelihood equals scipy's multivariate normal.
"""

import numpy as np
import pytest
from scipy import stats

from erotica.multicatalog import (
    combine_proper_motions,
    difference_loglike,
    inheritance_cross_covariance,
)


def _diag(a, b):
    C = np.zeros((len(a), 2, 2))
    C[:, 0, 0], C[:, 1, 1] = a, b
    return C


def test_independent_diagonal_is_inverse_variance_mean():
    rng = np.random.default_rng(1)
    n = 50
    vg, vv = rng.uniform(0.01, 1, (n, 2)), rng.uniform(0.01, 1, (n, 2))
    yg, yv = rng.normal(0, 1, (n, 2)), rng.normal(0, 1, (n, 2))
    mu, C, d, Cd = combine_proper_motions(yg, _diag(*vg.T), yv, _diag(*vv.T))
    var = 1 / (1 / vg + 1 / vv)
    np.testing.assert_allclose(mu, var * (yg / vg + yv / vv), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(C[:, 0, 0], var[:, 0], rtol=1e-12)
    np.testing.assert_allclose(C[:, 1, 1], var[:, 1], rtol=1e-12)
    np.testing.assert_allclose(C[:, 0, 1], 0.0, atol=1e-15)
    np.testing.assert_allclose(d, yv - yg, atol=1e-15)
    np.testing.assert_allclose(Cd[:, 0, 0], vg[:, 0] + vv[:, 0], rtol=1e-12)


def test_full_inheritance_adds_nothing():
    rng = np.random.default_rng(2)
    n = 40
    L = rng.normal(0, 0.3, (n, 2, 2))
    Cg = L @ np.transpose(L, (0, 2, 1)) + 0.01 * np.eye(2)
    K = rng.uniform(0.05, 0.5, (n, 2))
    Cv = Cg + _diag(*K.T)  # e_V = e_G + eta, Cov(eta) = diag(K)
    X = inheritance_cross_covariance(Cg, 1.0)
    yg, yv = rng.normal(0, 1, (n, 2)), rng.normal(0, 1, (n, 2))
    mu, C, _, Cd = combine_proper_motions(yg, Cg, yv, Cv, cross=X)
    np.testing.assert_allclose(mu, yg, atol=1e-9)
    np.testing.assert_allclose(C, Cg, atol=1e-9)
    np.testing.assert_allclose(Cd, _diag(*K.T), atol=1e-12)


def test_offset_is_subtracted():
    rng = np.random.default_rng(3)
    n = 10
    Cg, Cv = _diag(*rng.uniform(0.01, 0.2, (2, n))), _diag(*rng.uniform(0.01, 0.2, (2, n)))
    yg, yv = rng.normal(0, 1, (n, 2)), rng.normal(0, 1, (n, 2))
    delta = np.array([-0.13, 0.02])
    a = combine_proper_motions(yg, Cg, yv, Cv)
    b = combine_proper_motions(yg, Cg, yv + delta, Cv, offset=delta)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_allclose(x, y, atol=1e-12)


def test_monte_carlo_covariance_matches_C_hat():
    rng = np.random.default_rng(4)
    m = 20000
    Cg = np.array([[0.04, 0.01], [0.01, 0.02]])
    Cv = np.array([[0.09, -0.02], [-0.02, 0.06]])
    lam = 0.5
    # e_V = lam e_G + eta  =>  C_V = lam^2 C_G + Cov(eta); choose Cov(eta) so that C_V is as above
    Ceta = Cv - lam**2 * Cg
    eg = rng.multivariate_normal([0, 0], Cg, m)
    ev = lam * eg + rng.multivariate_normal([0, 0], Ceta, m)
    truth = np.array([2.5, -1.7])
    X = inheritance_cross_covariance(np.repeat(Cg[None], m, 0), lam)
    mu, C, _, _ = combine_proper_motions(
        truth + eg, np.repeat(Cg[None], m, 0), truth + ev, np.repeat(Cv[None], m, 0), cross=X
    )
    r = mu - truth
    Ci = np.linalg.inv(C[0])
    chi2 = np.einsum("ni,ij,nj->n", r, Ci, r)
    # sum of m chi^2_2 is chi^2_{2m}; two-sided 0.1% bounds (absolute, not self-widening)
    lo, hi = stats.chi2.ppf([0.0005, 0.9995], 2 * m)
    assert lo < chi2.sum() < hi
    # and it beats the independent-errors combination, which is over-confident here
    mu0, C0, _, _ = combine_proper_motions(
        truth + eg, np.repeat(Cg[None], m, 0), truth + ev, np.repeat(Cv[None], m, 0)
    )
    chi2_0 = np.einsum("ni,ij,nj->n", mu0 - truth, np.linalg.inv(C0[0]), mu0 - truth).sum()
    assert chi2_0 > hi


def test_difference_loglike_matches_scipy():
    rng = np.random.default_rng(5)
    n = 30
    L = rng.normal(0, 0.3, (n, 2, 2))
    Cd = L @ np.transpose(L, (0, 2, 1)) + 0.02 * np.eye(2)
    d = rng.normal(0, 0.3, (n, 2))
    ref = np.array(
        [stats.multivariate_normal(mean=[0, 0], cov=Cd[i]).logpdf(d[i]) for i in range(n)]
    )
    np.testing.assert_allclose(difference_loglike(d, Cd), ref, rtol=1e-10)


def test_shape_errors():
    with pytest.raises(ValueError):
        combine_proper_motions(
            np.zeros((3, 2)), np.zeros((2, 2, 2)), np.zeros((3, 2)), np.zeros((3, 2, 2))
        )
