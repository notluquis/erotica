"""Wilson (1975) lowered-isothermal profile: oracles independent of the implementation.

Oracles, in decreasing independence:

* ``scipy.integrate.quad`` over the Abel projection of the 3D density, evaluated on the same ODE
  solution (different quadrature, so it catches the projection nodes, not the Poisson solve);
* LIMEPY (Gieles & Zocchi 2015), **frozen** reference values taken on 2026-10-27 with
  ``astro-limepy`` 1.2 on SciPy 1.14 (it fails on current SciPy, see the module docstring of
  ``erotica.analysis.wilson``), plus a live cross-check that runs only if LIMEPY is installed;
* the King (1966) ``g = 1`` member against the King (1962) empirical formula, the known
  relationship between those two (they agree to ~1 % of the central density at W0 = 7).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize

from erotica.analysis import wilson as w

requires_bayes_extra = pytest.mark.skipif(
    importlib.util.find_spec("pymc") is None, reason="requires the 'bayes' extra"
)

# (g, W0) -> (r_t / r_0, S(s) at s = R / r_t in S_POINTS), LIMEPY 1.2, 5 significant digits.
S_POINTS = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6]
LIMEPY_REFERENCE = {
    (2, 3.0): (9.31710, [1.0, 0.99849, 0.98448, 0.87432, 0.32181, 0.012913, 0.00021555]),
    (2, 6.0): (62.50773, [1.0, 0.96048, 0.67743, 0.15743, 0.0078952, 0.00016244, 2.609e-06]),
    (2, 9.0): (
        2131.74560,
        [1.0, 0.027124, 0.0035745, 0.00042718, 2.3029e-05, 5.4607e-07, 9.08e-09],
    ),
    (1, 6.0): (17.99161, [1.0, 0.99674, 0.96559, 0.75307, 0.18971, 0.013412, 0.00065164]),
}


@pytest.mark.parametrize(("g", "W0"), sorted(LIMEPY_REFERENCE))
def test_profile_matches_frozen_limepy(g, W0):
    """Truncation radius to 1e-5 (the ODE is the same equation); Sigma to 2e-3 of the central value.

    The Sigma tolerance is LIMEPY's own projection error, not ours: against scipy.integrate.quad
    this code agrees to ~1e-5 where LIMEPY is off by up to 1.3e-3 (W0 = 3) -- see the quad test.
    """
    rt_ref, s_ref = LIMEPY_REFERENCE[(g, W0)]
    r_t, s, S, _ = w.projected_profile(W0, g)
    assert r_t == pytest.approx(rt_ref, rel=1e-5)
    np.testing.assert_allclose(np.interp(S_POINTS, s, S), s_ref, atol=2e-3)


def test_projection_matches_quad_at_a_concentrated_model():
    """W0 = 9 has r_t/r_0 = 2132 and a core at s ~ 5e-4: uniform line-of-sight nodes were 6 % off."""
    r_t, psi = w.solve_lowered_isothermal(9.0, 2)

    def rho(r):
        return float(w.lowered_isothermal_density(psi(np.array([r])), 2)[0])

    def sigma(R):
        top = np.sqrt(r_t**2 - R**2)
        return (
            2
            * quad(
                lambda u: rho(np.sqrt(R * R + u * u)), 0, top, epsabs=1e-13, epsrel=1e-11, limit=400
            )[0]
        )

    s0 = sigma(0.0)
    _, s, S, _ = w.projected_profile(9.0, 2)
    for sv in (0.001, 0.003, 0.03, 0.1):
        assert np.interp(sv, s, S) == pytest.approx(sigma(sv * r_t) / s0, rel=1e-3)


@pytest.mark.skipif(importlib.util.find_spec("limepy") is None, reason="LIMEPY not installed")
def test_profile_matches_live_limepy():  # pragma: no cover - needs astro-limepy on a compatible SciPy
    from limepy import limepy

    m = limepy(5.5, 2, M=1.0, rh=1.0, project=True)
    r_t, s, S, _ = w.projected_profile(5.5, 2)
    assert r_t == pytest.approx(m.rt / m.r0, rel=1e-5)
    np.testing.assert_allclose(
        np.interp(S_POINTS, s, S),
        np.interp(np.array(S_POINTS) * m.rt, m.R, m.Sigma) / m.Sigma[0],
        atol=2e-3,
    )


def test_king_member_reduces_to_the_king_1962_formula():
    """g = 1 is King (1966); King (1962) is the empirical fit to it, with r_c ~ r_0 at high W0."""
    r_t, s, S, _ = w.projected_profile(7.0, 1)
    x = s * r_t
    inside = s < 0.999

    def king62(rc):
        edge = 1 / np.sqrt(1 + (r_t / rc) ** 2)
        return (1 / np.sqrt(1 + (x / rc) ** 2) - edge) ** 2 / (1 - edge) ** 2

    fit = minimize(
        lambda p: np.max(np.abs(king62(p[0])[inside] - S[inside])), [1.0], method="Nelder-Mead"
    )
    # measured 2026-10-27: rc/r_0 = 0.975, max|dS| = 0.0026 (0.0098 at W0 = 5, 0.0176 at W0 = 3)
    assert 0.9 < fit.x[0] < 1.05
    assert fit.fun < 0.005
    # and Wilson is NOT King: its envelope is more extended (same truncation radius in r_0 units
    # is 7x larger at W0 = 7)
    assert w.solve_lowered_isothermal(7.0, 2)[0] > 5 * r_t


@pytest.mark.parametrize("W0", [1.1, 2.37, 5.1, 7.37, 9.9])
def test_table_interpolation_error_is_bounded(W0):
    """The table is on a 0.25 grid in W0: off-grid values are linearly interpolated at fixed R/r_0."""
    r_t, s, S, C = w.projected_profile(W0, 2)
    R = np.array([0.0, 0.2, 1.0, 3.0, 10.0, 30.0, 100.0])
    R = R[R < 0.95 * r_t]
    np.testing.assert_allclose(
        w.wilson_surface_density(R, k=1.0, r_0=1.0, W0=W0), np.interp(R / r_t, s, S), atol=3e-3
    )
    rf = min(30.0, r_t)
    exact = 2 * np.pi * r_t**2 * np.interp(rf / r_t, s, C)
    assert w.wilson_expected_count(1.0, 0.0, 1.0, W0, rf) == pytest.approx(exact, rel=3e-3)
    assert w.wilson_truncation_radius(W0, 1.0) == pytest.approx(r_t, rel=5e-3)


def test_expected_count_is_the_integral_of_the_surface_density():
    k, b, r_0, W0, rf = 40.0, 0.02, 2.0, 4.5, 25.0
    num = quad(
        lambda R: 2 * np.pi * R * w.wilson_surface_density(R, k=k, r_0=r_0, W0=W0, b=b),
        0,
        rf,
        limit=500,
    )[0]
    assert w.wilson_expected_count(k, b, r_0, W0, rf) == pytest.approx(num, rel=2e-3)


def test_outside_the_tidal_radius_only_the_background_remains():
    r_t = w.wilson_truncation_radius(3.0, 1.0)
    sig = w.wilson_surface_density(
        np.array([0.5 * r_t, 1.5 * r_t]), k=10.0, r_0=1.0, W0=3.0, b=0.25
    )
    assert sig[1] == 0.25 and sig[0] > 0.25


def test_interpolated_model_is_truncated_at_its_own_tidal_radius():
    """Blending two table rows leaves the larger neighbour's tail; the cut must be at r_t(W0)."""
    W0 = 5.1
    r_t = w.wilson_truncation_radius(W0, 1.0)
    just_out = w.wilson_surface_density(np.array([1.0001 * r_t]), k=10.0, r_0=1.0, W0=W0, b=0.0)
    just_in = w.wilson_surface_density(np.array([0.999 * r_t]), k=10.0, r_0=1.0, W0=W0, b=0.0)
    assert just_out[0] == 0.0
    assert just_in[0] > 0.0


def test_w0_outside_the_table_is_refused():
    with pytest.raises(ValueError, match="outside the tabulated range"):
        w.wilson_surface_density(np.array([1.0]), k=1.0, r_0=1.0, W0=12.0)


def _draw_from_exact_profile(seed, k, r_0, W0, b, field):
    """Independent generator: the exact per-call ODE solve, NOT the interpolation table."""
    r_t, s, S, C = w.projected_profile(W0, 2)
    rng = np.random.default_rng(seed)
    scale = r_0 * r_t  # truncation radius in arcmin
    s_f = min(1.0, field / scale)
    n_cl = rng.poisson(2 * np.pi * k * scale**2 * np.interp(s_f, s, C))
    u = rng.random(n_cl) * np.interp(s_f, s, C)
    r_cl = np.interp(u, C, s) * scale
    n_bg = rng.poisson(np.pi * b * field**2)
    r_bg = field * np.sqrt(rng.random(n_bg))
    return np.concatenate([r_cl, r_bg])


def _fit_ml(r, field):
    def nll(p):
        if not (w.W0_MIN <= p[2] <= w.W0_MAX):
            return 1e30
        ll = w.wilson_log_likelihood(
            r, k=np.exp(p[0]), r_0=np.exp(p[1]), W0=p[2], b=np.exp(p[3]), field_radius=field
        )
        return -ll if np.isfinite(ll) else 1e30

    best = None
    for W in (3.0, 5.0, 7.0):  # several starts: W0 and r_0 are correlated
        res = minimize(
            nll,
            [np.log(50), np.log(2), W, np.log(0.03)],
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 3000},
        )
        if best is None or res.fun < best.fun:
            best = res
    return np.exp(best.x[0]), np.exp(best.x[1]), best.x[2], np.exp(best.x[3])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recovers_parameters_drawn_from_the_exact_profile(seed):
    """Truth k=100, r_0=1.5, W0=5, b=0.05 in a 30-arcmin field, ~1600 stars.

    Measured over 6 seeds with the table-generated data: W0 within 0.15, r_0 within 5 %, k within
    8 %. The tolerances below are ~2x those and absolute, tied to the injected truth. The fixture
    sits where the parameters are recoverable: with k=6 (~110 stars) the same fit returned W0 from
    2.1 to 5.9, i.e. a recovery test that cannot fail.
    """
    field = 30.0
    r = _draw_from_exact_profile(seed, 100.0, 1.5, 5.0, 0.05, field)
    assert 1300 < r.size < 1900
    k, r_0, W0, b = _fit_ml(r[r > 0], field)
    assert abs(W0 - 5.0) < 0.4
    assert abs(r_0 / 1.5 - 1) < 0.12
    assert abs(k / 100.0 - 1) < 0.2
    assert abs(b / 0.05 - 1) < 0.35


@requires_bayes_extra
@pytest.mark.slow
def test_bayes_factor_identifies_the_generating_model_in_both_directions():
    """Generating-model identification through the public API, run both ways.

    Measured 2026-10-27 (SMC, 500 draws, 2 chains, evidence chain sd 0.02-0.35): on data from the
    exact Wilson profile with W0 = 3 (N = 3021) King loses by ln B = -24; at W0 = 5 (N = 5793)
    by -39 and at N = 1506 by -10. At W0 = 8 (N = 3796) King *wins* by 0.5: a high-concentration
    Wilson profile is inside what King's free R_t can imitate, so no claim of identification is
    made there. The fixture below sits where the models are separable.
    """
    from erotica.analysis.structure import compare_radial_profiles

    from .test_structure import FIELD as KING_FIELD
    from .test_structure import _sample_king

    field = 30.0
    r = _draw_from_exact_profile(7, 400.0, 1.5, 3.0, 0.02, field)
    wil = compare_radial_profiles(
        r[r > 0],
        field_radius=field,
        models=("king", "wilson"),
        draws=500,
        chains=2,
        cores=1,
        random_seed=1,
    )
    assert np.isfinite(wil["log_marginal_likelihood"]["wilson"])
    assert wil["best"] == "wilson", wil["log_marginal_likelihood"]
    assert wil["resolvable"]["king"]
    assert wil["log_bayes_factor_vs_best"]["king"] < -5.0

    king_data = _sample_king(41, k=6.0, b=0.002, R_c=4.0, R_t=25.0, field=KING_FIELD)
    kin = compare_radial_profiles(
        king_data,
        field_radius=KING_FIELD,
        models=("king", "wilson"),
        draws=500,
        chains=2,
        cores=1,
        random_seed=1,
    )
    assert kin["best"] == "king", kin["log_marginal_likelihood"]
