"""The mode search sees a mode the chains never visited, and a mode whose weight the initialisation set.

Oracles with a known answer (hub ``agent-findings/multimodality-automatic-check-2026-10-14.md`` §0.3, T1):
a 7-D two-Gaussian mixture, mode 2 at 10 sd with weight w, so its Laplace Delta log mass is exactly
log(w / (1 - w)); chains drawn from mode 1 only.
"""

import numpy as np
import pytest

from erotica.analysis.modesearch import mode_search, verdict

D = 7
MU2 = np.r_[10.0, np.zeros(D - 1)]


def mixture(w):
    def lp(x):
        a = np.log1p(-w) - 0.5 * np.sum(x**2)
        return float(np.logaddexp(a, np.log(w) - 0.5 * np.sum((x - MU2) ** 2))) if w else a

    return lp


def run(w, draws):
    rng = np.random.default_rng(0)
    box = (np.full(D, -15.0), np.full(D, 15.0))
    return mode_search(mixture(w), rng.uniform(-15, 15, (16, D)), bounds=box, draws=draws)


@pytest.fixture
def mode1_draws():
    return np.random.default_rng(1).standard_normal((300, 4, D))


def test_unvisited_mode_fails(mode1_draws):
    res = run(0.1, mode1_draws)
    m2 = [m for m in res["modes"] if abs(m["x"][0] - 10) < 0.5]
    assert m2 and abs(m2[0]["dlog_mass"] - np.log(0.1 / 0.9)) < 0.05
    text, ok = verdict(res)
    assert not ok and "no chain visited it" in text


def test_unimodal_passes(mode1_draws):
    assert verdict(run(0.0, mode1_draws))[1]


def test_negligible_mode_passes(mode1_draws):
    # Delta log mass log(1e-3) = -6.9 < -5: reported, not a failure
    assert verdict(run(1e-3, mode1_draws))[1]


def test_initialisation_weight_fails(mode1_draws):
    # R26-04 shape: one of 4 chains sits in a mode holding 1e-3 of the mass
    draws = mode1_draws.copy()
    draws[:, 3] += MU2
    text, ok = verdict(run(1e-3, draws))
    assert not ok and "initialisation" in text


def wide_mixture(w, s, sep):
    mu = np.r_[sep, np.zeros(D - 1)]

    def lp(x):
        return float(np.logaddexp(np.log1p(-w) - 0.5 * np.sum(x**2),
                                  np.log(w) - 0.5 * np.sum((x - mu) ** 2) / s**2 - D * np.log(s)))

    return lp


def test_wide_unvisited_mode_fails_and_its_mass_is_right():
    # R68-01: a broad mode the chains never visited; the per-draw assignment must not veto the failure.
    # R68-04: unequal widths, so the Laplace log det enters the mass (Delta log mass = log(w/(1-w))).
    draws = np.random.default_rng(2).standard_normal((1000, 4, D))
    rng = np.random.default_rng(3)
    res = mode_search(wide_mixture(0.3, 3.0, 10.0), rng.uniform(-15, 15, (32, D)),
                      bounds=(np.full(D, -15.0), np.full(D, 15.0)), draws=draws)
    m2 = [m for m in res["modes"] if abs(m["x"][0] - 10) < 1]
    assert m2 and abs(m2[0]["dlog_mass"] - np.log(0.3 / 0.7)) < 0.05
    text, ok = verdict(res)
    assert not ok and "no chain visited it" in text


def test_chain_that_lived_in_a_minor_mode_is_seen():
    # R68-05: chain 3 spends 499 of 500 draws in a mode holding 1e-3 of the mass, last draw back home
    draws = np.random.default_rng(1).standard_normal((500, 4, D))
    draws[:-1, 3] += MU2
    text, ok = verdict(run(1e-3, draws))
    assert not ok and "initialisation" in text


def test_embed_pushes_the_covariance_through_its_jacobian():
    # R68-04: y = 2 x, so the covariance in y is 4 times the one in x
    lp = lambda x: -0.5 * float(np.sum(x**2))  # noqa: E731
    res = mode_search(lp, np.random.default_rng(0).uniform(-3, 3, (4, 2)), embed=lambda x: 2 * x,
                      ridge_fracs=())
    m = res["modes"][0]
    assert np.allclose(np.linalg.inv(m["prec_y"]), 4 * m["cov"], rtol=1e-4)


def test_ridge_follows_the_modes_not_the_local_hessian():
    # R68-04 / P1: rippled ridge (hub finding T2); the global mode sits at u = 3.5 along it
    def lp(x):
        u, v = (x[0] + x[1]) / np.sqrt(2), (x[0] - x[1]) / np.sqrt(2)
        env = -0.5 * u**2 + 3 * np.cos(2 * np.pi * u / 0.1)
        return float(-0.5 * (v / 0.01) ** 2 - 0.5 * np.sum(x[2:] ** 2) + np.logaddexp(env, 5 - 0.5 * ((u - 3.5) / 0.03) ** 2))

    found = 0
    for seed in range(3):
        rng = np.random.default_rng(seed + 800)
        draws = rng.standard_normal((50, 4, D))  # chains stuck on 4 ripples near u = 0, as in T2
        uu = np.array([-0.2, -0.1, 0.0, 0.1])[None, :] + 0.005 * rng.standard_normal((50, 4))
        vv = 0.01 * rng.standard_normal((50, 4))
        draws[:, :, 0], draws[:, :, 1] = (uu + vv) / np.sqrt(2), (uu - vv) / np.sqrt(2)
        res = mode_search(lp, rng.uniform(-5, 5, (8, D)), bounds=(np.full(D, -5.0), np.full(D, 5.0)), draws=draws)
        b = res["modes"][0]["x"]
        found += abs((b[0] + b[1]) / np.sqrt(2) - 3.5) < 0.05
    assert found == 3


@pytest.mark.parametrize("gap, decides", [(0.1, True), (10.0, False)])
def test_like_vs_post_sees_a_prior_that_picks_the_mode(gap, decides):
    # hub finding §3.6: one posterior mode, two likelihood humps `gap` apart; the prior N(0, 1) on x0 picks
    from erotica.analysis.modesearch import like_vs_post

    def loglike(x):
        return float(np.logaddexp(-0.5 * np.sum(x**2), -gap - 0.5 * ((x[0] - 6) ** 2 + np.sum(x[1:] ** 2))))

    def logpost(x):
        return loglike(x) - 0.5 * x[0] ** 2

    box = (np.full(3, -10.0), np.full(3, 10.0))
    st = np.random.default_rng(0).uniform(-10, 10, (16, 3))
    rl, rp = mode_search(loglike, st, bounds=box), mode_search(logpost, st, bounds=box)
    text, ok = like_vs_post(rl, rp, logpost)
    assert ok is not decides and (("PRIOR DECIDES" in text) is decides)
