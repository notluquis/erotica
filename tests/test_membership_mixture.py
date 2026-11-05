"""erotica.membership: the cluster + field mixture with the selection in every component.

Oracles, each independent of the code under test:

* **V1, construction control** (hub ``agent-findings/membership-rule-literature-2026-11-03.md`` §6): without
  the field the mixture is the single-component truncated MLE. The oracle computes that MLE with different
  algebra -- P_A as a radial integral of the Rice distribution by Gauss-Legendre (isotropic errors), the
  parallax with ``scipy.stats.truncnorm`` -- and maximises it with Nelder-Mead from the sample moments.
* **closed forms**: with no selection, no field and homoscedastic errors the MLE is the sample mean and
  ``σ̂² = mean |x − x̄|² / 2 − e²``, and the Laplace sd of the mean is ``sqrt((σ̂² + e²) / N)``; a field
  with ``η → 0`` leaves the cluster log-likelihood; the field prior adds exactly its log-normal term;
  Bernoulli propagation of a count has mean ``Σp`` and variance ``Σp(1 − p)``.
* **recovery against injected truth** with absolute tolerances, in a fixture where the truncation bites
  (cluster centred 0.77 R from the region centre), and the three mutations of C5 §15 -- ``nopa``,
  ``nofield``, ``nosel`` -- each of which must break that same recovery.

Tolerances were measured on seeds 0-3 on 2026-11-05 (the numbers are in each test's docstring); the
fixed seeds below are among them.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize, special, stats

import erotica.membership as M
from erotica.membership import (
    CircularRegion,
    GaussianField,
    MembershipData,
    MixtureModel,
    Truncation,
    TwoLineRule,
    auc,
    calibrate_by_injection,
    propagate,
    sample_field,
    sample_members,
)

# ---------------------------------------------------------------------------------------------- fixture
TRUTH = dict(mu_c=(1.15, 0.0), sigma_c=0.40, plx_c=1.0, s_plx_c=0.02,
             mu_f=(-0.5, 0.3), sf_1=3.0, sf_2=2.5, rho_f=0.2, plx_f=0.7, s_plx_f=0.4, eta=0.35)
R, TAU = 1.5, 0.03
SEL = Truncation(CircularRegion((0.0, 0.0), R, TAU), plx_window=(0.4, 1.6), n_qmc=256, n_support=512)
# what a detection stage would hand over: near the cluster, not at the truth
INIT = dict(mu_c=(1.0, 0.1), sigma_c=0.3, plx_c=1.0, eta=0.5)
SEED = 0


def _donors(rng, n):
    e = rng.uniform(0.05, 0.25, n)
    cov = np.zeros((n, 2, 2))
    cov[:, 0, 0] = e**2
    cov[:, 1, 1] = (1.2 * e) ** 2
    cov[:, 0, 1] = 0.3 * 1.2 * e * e
    return MembershipData(np.zeros((n, 2)), cov, np.zeros(n), rng.uniform(0.02, 0.08, n), {"G": 12 + 30 * e})


def make(seed, n=400, select_members=True):
    rng = np.random.default_rng(seed)
    don = _donors(rng, n)
    y = rng.random(n) >= TRUTH["eta"]
    m = sample_members(TRUTH, don.take(np.where(y)[0]), rng, SEL if select_members else None)
    f = sample_field(TRUTH, don.take(np.where(~y)[0]), rng, SEL)
    return MembershipData.concat(m, f), np.r_[np.ones(m.n, bool), np.zeros(f.n, bool)]


def fit(data, selection=SEL, field=GaussianField(), seed=SEED):
    model = MixtureModel(data, selection, field)
    model.fit(n_starts=3, seed=seed, ridge=False, init=INIT)
    return model


@pytest.fixture(scope="module")
def fitted():
    data, y = make(SEED)
    return fit(data), y


def recovered(model):
    """The recovery criterion every mutation must break: absolute errors against the injected truth."""
    P = model.params
    return dict(mu_x=abs(P["mu_c"][0] - TRUTH["mu_c"][0]), sigma=abs(P["sigma_c"] - TRUTH["sigma_c"]),
                eta=abs(P["eta"] - TRUTH["eta"]))


# Measured 2026-11-05, same fixture, 3 prior starts + INIT, seeds 0-3 (|error| against TRUTH):
#   correct model   σ 0.003-0.038   μ_x 0.009-0.125 (seed 3 at the 0.85 R box edge)   η 0.013-0.030
#   nosel (model)   σ 0.076-0.090   μ_x 0.149-0.216                                    η 0.022-0.040
#   nofield         σ 0.45-0.58     (η fixed: no field)
# σ separates the arms with margin; μ_x barely does (the cut makes μ_x and σ trade along x), so its
# tolerance only guards against gross failures. nopa: see its test.
TOL = dict(mu_x=0.14, sigma=0.055, eta=0.05)


def recovers(model):
    r = recovered(model)
    return all(r[k] < TOL[k] for k in TOL), r


# ---------------------------------------------------------------------------------------------- V1
_x1, _w1 = np.polynomial.legendre.leggauss(96)


def _gl(a, b):
    return 0.5 * (b - a) * _x1 + 0.5 * (b + a), 0.5 * (b - a) * _w1


_RN, _RW = (np.r_[u, v] for u, v in zip(_gl(0, R - 10 * TAU), _gl(R - 10 * TAU, R + 15 * TAU), strict=True))
_ACC = 1 / (1 + np.exp(-(R - _RN) / TAU))


def _log_pa_rice(nu, s):
    """P_A = ∫ acc(r) Rice(r; ν, s) dr; Rice pdf = r/s² exp(−(r − ν)²/2s²) i0e(rν/s²)."""
    r, s = _RN[None, :], s[:, None]
    pdf = r / s**2 * np.exp(-((r - nu) ** 2) / (2 * s**2)) * special.i0e(r * nu / s**2)
    return np.log(pdf @ (_ACC * _RW))


def _oracle_truncated_mle(d, e, window):
    def nll_pm(q):
        mu, st = q[:2], np.sqrt(e**2 + np.exp(2 * q[2]))
        ll = -np.sum((d.pm - mu) ** 2, 1) / (2 * st**2) - np.log(2 * np.pi * st**2)
        return -np.sum(ll - _log_pa_rice(np.hypot(*mu), st))

    def nll_plx(q):
        sd = np.sqrt(d.plx_err**2 + np.exp(2 * q[1]))
        return -np.sum(stats.truncnorm.logpdf(d.plx, (window[0] - q[0]) / sd, (window[1] - q[0]) / sd, loc=q[0], scale=sd))

    opt = dict(xatol=1e-9, fatol=1e-11, maxiter=20000, maxfev=20000)
    a = optimize.minimize(nll_pm, np.r_[d.pm.mean(0), np.log(d.pm.std(0).mean())], method="Nelder-Mead", options=opt)
    b = optimize.minimize(nll_plx, np.r_[d.plx.mean(), np.log(0.05)], method="Nelder-Mead", options=opt)
    return np.r_[a.x, b.x]


def test_rice_oracle_reduces_to_the_closed_form_at_nu_zero():
    """Check of the oracle itself: a centred Gaussian under a hard edge has P_A = 1 − exp(−R²/2s²); the
    logistic edge of width τ = 0.03 moves it by less than 2e-4 at s = 0.4."""
    s = np.array([0.4])
    assert np.exp(_log_pa_rice(0.0, s))[0] == pytest.approx(1 - np.exp(-(R**2) / (2 * s[0] ** 2)), abs=2e-4)


def test_v1_without_field_reproduces_the_truncated_single_component_mle():
    """V1. The PM block's residual is the QMC error of P_A, not the optimiser: the model's log posterior is
    higher at its own optimum than at the oracle's. Measured 2026-11-05 (200 stars, cluster 0.77 R off
    centre, edge width 0.03): model − oracle 3.9e-3 with 4096 normals, 4e-4 with 16384 (n^-0.75-ish: the
    sharp edge makes the integrand nearly discontinuous). The parallax block (closed form on both sides)
    agrees to 1e-5. Removing P_A from the cluster moves the PM block by ~0.1 (nopa test)."""
    rng = np.random.default_rng(1)
    n = 200
    e = rng.uniform(0.05, 0.25, n)
    cov = np.zeros((n, 2, 2))
    cov[:, 0, 0] = cov[:, 1, 1] = e**2
    # the parallax window cuts the cluster too (lower edge 1 sd below ϖ_c): with the fixture's (0.4, 1.6)
    # the cluster's W ≈ 1 and deleting W from the code left this test green (mutation, 2026-11-05)
    sel = Truncation(CircularRegion((0.0, 0.0), R, TAU), plx_window=(0.96, 1.6), n_qmc=16384)
    don = MembershipData(np.zeros((n, 2)), cov, np.zeros(n), rng.uniform(0.02, 0.08, n))
    d = sample_members(TRUTH, don, rng, sel)
    m = MixtureModel(d, sel, None)
    m.fit(n_starts=3, seed=1, ridge=False)
    got, ref = m.fit_result.theta[:5], _oracle_truncated_mle(d, e, sel.plx_window)
    np.testing.assert_allclose(got[:3], ref[:3], atol=1.5e-3)
    np.testing.assert_allclose(got[3:], ref[3:], atol=1e-4)


def test_no_selection_no_field_gives_the_closed_form_mle_and_laplace_sd():
    """Homoscedastic isotropic errors, no cut: μ̂ = x̄, σ̂² = mean|x − x̄|²/2 − e², ϖ̂ = mean, s² = var − e²;
    the Laplace sd of μ_x is sqrt((σ̂² + e²)/N) (the mean decouples at the MLE). in_model propagation of μ_x
    must reproduce that sd (4000 draws: Monte Carlo error ~1.1 %)."""
    rng = np.random.default_rng(7)
    n, e, ep = 500, 0.1, 0.01
    cov = np.broadcast_to(np.eye(2) * e**2, (n, 2, 2)).copy()
    don = MembershipData(np.zeros((n, 2)), cov, np.zeros(n), np.full(n, ep))
    tr = dict(mu_c=(0.3, -0.2), sigma_c=0.25, plx_c=1.0, s_plx_c=0.05)
    d = sample_members(tr, don, rng, None)
    m = MixtureModel(d, None, None)
    m.fit(n_starts=3, seed=0, ridge=False)
    P = m.params
    xbar = d.pm.mean(0)
    np.testing.assert_allclose(P["mu_c"], xbar, atol=1e-6)
    assert P["sigma_c"] ** 2 == pytest.approx(np.mean(np.sum((d.pm - xbar) ** 2, 1)) / 2 - e**2, abs=1e-6)
    assert P["plx_c"] == pytest.approx(d.plx.mean(), abs=1e-7)
    assert P["s_plx_c"] ** 2 == pytest.approx(d.plx.var() - ep**2, abs=1e-7)
    sd_mean = np.sqrt((P["sigma_c"] ** 2 + e**2) / n)
    assert np.sqrt(m.fit_result.cov[0, 0]) == pytest.approx(sd_mean, rel=1e-4)
    res = propagate(lambda p: p["mu_c"][0], m, mode="in_model", n_draws=4000, seed=3)
    assert res.sd == pytest.approx(sd_mean, rel=0.05)


def test_field_with_vanishing_fraction_is_the_cluster_alone(fitted):
    """η → 0 (logit −700, η ~ 1e-304): the mixture log-likelihood equals the field-free one at the same
    cluster parameters. (At logit −40 it does not: field stars sit more than 40 nats below the cluster.)"""
    model, _ = fitted
    th = model.fit_result.theta.copy()
    th[12] = -700.0
    alone = MixtureModel(model.data, SEL, None)
    assert model.logpost(th) == pytest.approx(alone.logpost(th), abs=1e-9)


def test_field_prior_adds_exactly_its_lognormal_term(fitted):
    model, _ = fitted
    th = model.fit_result.theta
    med, sd = 120.0, 0.07
    nf = M._natural(th)["eta"] * model.data.n
    diff = model.logpost(th, (med, sd)) - model.logpost(th)
    assert diff == pytest.approx(-0.5 * ((np.log(nf) - np.log(med)) / sd) ** 2, rel=1e-10)


# ---------------------------------------------------------------------------------------------- recovery
def test_recovery_on_the_truncated_fixture(fitted):
    """Absolute errors against the injected truth. Measured on seeds 0-3 (2026-11-05): see TOL and the
    mutation tests, which must exceed it."""
    model, y = fitted
    ok, r = recovers(model)
    assert ok, r
    assert model.fit_result.n_starts == 3 and "VERDICT" in model.fit_result.verdict
    p = model.p
    assert abs(p.sum() - y.sum()) < 0.06 * y.sum()
    assert auc(p, y) > 0.95


def test_nosel_model_without_the_truncation_fails_recovery():
    """``nosel``: the same truncated data fitted with ``selection=None``. The cluster sits 0.77 R from the
    region centre, so ignoring the cut biases its mean and dispersion."""
    data, _ = make(SEED)
    ok, r = recovers(fit(data, selection=None))
    assert not ok, r


def test_nosel_generator_without_the_truncation_fails_recovery():
    """``nosel`` as C5 §15 wrote it: members generated without the cut while the model keeps it. It proves
    the fixture's truncation is load-bearing, not a no-op the model happens to tolerate."""
    data, _ = make(SEED, select_members=False)
    ok, r = recovers(fit(data))
    assert not ok, r


def test_nofield_model_without_the_field_fails_recovery():
    data, _ = make(SEED)
    ok, r = recovers(fit(data, field=None))
    assert not ok, r


@pytest.mark.parametrize("component", ["field", "cluster"])
def test_nopa_pa_missing_from_one_component_fails_recovery(monkeypatch, component):
    """``nopa``: P_A must divide **every** component (§A.3 step 2, §A.1.4). Dropping it from one of them
    (C5's audit found the §14 code with P_A in the field only) must break recovery.

    Measured, seeds 0-3: without the cluster's P_A, σ errs by 0.088-0.099 (all four fail on σ). Without the
    field's P_A, seed 0 fails on η (0.066), seed 1 grossly (η → 0), seed 3 on σ (0.062) and seed 2 only
    narrowly on η (0.053 against 0.05): the field arm is the weakest guard of the three."""
    orig = M._log_pa_pm

    def patched(mean, L00, L10, L11, region, nodes, comp):
        out = orig(mean, L00, L10, L11, region, nodes, comp)
        return out * 0.0 if comp == component else out

    monkeypatch.setattr(M, "_log_pa_pm", patched)
    data, _ = make(SEED)
    ok, r = recovers(fit(data))
    assert not ok, r


# ---------------------------------------------------------------------------------------------- rule
def test_two_line_rule_by_hand():
    """§3.5 written for single stars, one case per clause."""
    rule = TwoLineRule(p_hi=0.9, p_lo=0.1, veto=np.log(20))
    p = np.array([0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.5, 0.95])
    A = np.array([1.0, 1.0, 1.0, 2.0, -1.0, -1.0, 1.0, 1.0])
    B = np.array([0.5, -0.5, -4.0, np.nan, -0.5, 4.0, 1.0, 0.5])
    C = np.array([np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, np.nan, -3.5])
    got = rule.apply(p, {"A": A, "B": B, "C": C})
    assert got.tolist() == [
        "member",      # p high, two lines for, none against strongly
        "undecided",   # only one line for
        "undecided",   # two lines for, but B < −ln 20 vetoes
        "undecided",   # B missing: one line only
        "field",       # symmetric
        "undecided",   # B > ln 20 vetoes the field call
        "undecided",   # p in between
        "undecided",   # A and B for, but C = −3.5 < −ln 20 ≈ −3.00 vetoes
    ]
    # the same star with C = −2.5 (against, but not strongly) is a member
    assert rule.apply(p[7:], {"A": A[7:], "B": B[7:], "C": np.array([-2.5])}).tolist() == ["member"]


def test_gaia_astrometry_alone_never_establishes_a_member(fitted):
    """PM and parallax are one line (A); with nothing else the rule cannot reach two lines. A second line
    that agrees makes high-p stars members; it is the only way in."""
    model, y = fitted
    assert set(model.axis_llr) == {"A"}
    assert not np.any(model.classify() == "member")
    B = np.where(y, 1.0, -1.0)  # an oracle photometric line
    cls = model.classify(TwoLineRule(), extra_lines={"B": B})
    assert np.sum(cls == "member") > 0.5 * y.sum()
    assert np.mean(y[cls == "member"]) > 0.98
    with pytest.raises(ValueError):
        model.classify(extra_lines={"A": B})


# ---------------------------------------------------------------------------------------------- propagation
def test_bernoulli_propagation_of_a_count_has_the_binomial_moments(fitted):
    model, _ = fitted
    p = model.p
    res = propagate(lambda d, mem: mem.sum(), model, mode="bernoulli", n_draws=4000, seed=5)
    var = np.sum(p * (1 - p))
    assert res.mean == pytest.approx(p.sum(), abs=4 * np.sqrt(var / 4000))
    assert res.sd**2 == pytest.approx(var, rel=0.1)
    with pytest.raises(ValueError):
        propagate(lambda d, m: 0.0, model, mode="cut")


# ---------------------------------------------------------------------------------------------- injection
def test_calibrate_by_injection_scores_p_against_the_injected_truth(fitted):
    """Minimal run (2 replicates) of the N2 machinery on a synthetic 'real' field catalogue: it must return
    pooled and per-subgroup CORP, AUC, Σp − N and the rule's risks, with the injected members labelled."""
    model, _ = fitted
    rng = np.random.default_rng(11)
    fld = sample_field(TRUTH, _donors(rng, 100), rng, SEL)
    out = calibrate_by_injection(
        model, fld, 2, subgroups={"bright": lambda d: d.aux["G"] < 16, "faint": lambda d: d.aux["G"] >= 16},
        rule=TwoLineRule(), extra_lines=lambda d, y, r: {"B": np.where(y, 1.0, -1.0)}, n_starts=1, init=INIT, seed=2,
        ridge=False)
    y = out["arrays"]["y"]
    assert y.sum() == sum(out["n_members_injected"]) and (~y).sum() == 2 * fld.n
    assert out["auc"] > 0.95
    assert out["corp"].MCB < 0.03
    assert set(out["corp_by_subgroup"]) == {"bright", "faint"}
    assert abs(out["sum_p_minus_true"]["mean"]) < 0.1 * np.mean(out["n_members_injected"])
    assert out["rule_risk"]["false_member_rate"] < 0.05


# ---------------------------------------------------------------------------------------------- normalisation
def test_each_component_is_a_normalised_density_on_the_selected_volume():
    """Closed form that needs no fit: conditioned on selection, each component's density times the
    acceptance must integrate to 1 over (μ, ϖ). That is what "P_A and W in every component" means; the
    recovery tests cannot see a missing field W (deleting it left them green, 2026-11-05: the field's
    parallax shape absorbs a near-constant factor). The density factorises, so with
    S_μ = Σ_μ acc·exp l(μ, ϖ0) dA and S_ϖ = Σ_ϖ exp l(μ0, ϖ) dϖ, the product S_μ S_ϖ / exp l(μ0, ϖ0) is
    ∫_μ · ∫_ϖ and must be 1. Grid step 0.035 against a logistic edge of 0.1 and Gaussians ≥ 0.4 wide;
    measured 1.00012 (cluster) and 0.99997 (field). Deleting the field's W, the field's P_A or the
    cluster's P_A turns it red; the cluster's W ≈ 1 here and is guarded by V1 instead."""
    sel = Truncation(CircularRegion((0.0, 0.0), R, 0.1), plx_window=(0.4, 1.6), n_qmc=2048, n_support=2048)
    h = 0.035
    g = np.arange(-R - 1.0, R + 1.0 + h / 2, h)
    gx, gy = np.meshgrid(g, g)
    pm = np.c_[gx.ravel(), gy.ravel()]
    hp = 0.002
    pg = np.arange(0.4 + hp / 2, 1.6, hp)
    mu0, w0 = np.array([0.8, 0.1]), 0.9
    n1, n2 = len(pm), len(pg)
    cov = np.broadcast_to(np.diag([0.1**2, 0.12**2]), (n1 + n2 + 1, 2, 2)).copy()
    d = MembershipData(np.r_[pm, np.tile(mu0, (n2, 1)), mu0[None]], cov, np.r_[np.full(n1, w0), pg, w0],
                       np.full(n1 + n2 + 1, 0.05))
    m = MixtureModel(d, sel)
    th = np.r_[1.15, 0.0, np.log(0.4), 1.0, np.log(0.02), -0.5, 0.3, np.log(3.0), np.log(2.5), np.arctanh(0.2),
               0.7, np.log(0.4), 0.0]
    lc, lf = (np.asarray(v) for v in m._fns["comp"](M.jnp.asarray(th), m._d))
    acc = sel.region.prob(pm)
    for name, ll in (("cluster", lc), ("field", lf)):
        s_mu = np.sum(acc * np.exp(ll[:n1])) * h * h
        s_w = np.sum(np.exp(ll[n1:n1 + n2])) * hp
        assert s_mu * s_w / np.exp(ll[-1]) == pytest.approx(1.0, abs=2e-3), name
