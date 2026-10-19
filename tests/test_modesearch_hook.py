"""The mode search is wired into the PyMC fitting paths and its adapter is exact.

Oracles are independent of the adapter: PyMC's own compiled ``logp``, analytic transforms, finite
differences, and a bimodal ``pm.Model`` whose second mode no NUTS chain visits (hub
``agent-findings/multimodality-automatic-check-2026-10-14.md`` §3.8).  Lesson 29: the NON-default value
of every new knob is passed first, because with the default the existing suite exercises the new branch
without being able to tell it is wrong.
"""

import json
import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

pm = pytest.importorskip("pymc")

from erotica.analysis.inference import (  # noqa: E402
    SamplingConfig,
    _sample,
    distance_model,
    fit_parallax_model,
    mode_search_summary,
    proper_motion_2d_gaussian,
    velocity_model,
)
from erotica.analysis.modesearch import ModeSearchConfig, PyMCTarget, search_pymc  # noqa: E402
from erotica.analysis.structure import king_unbinned  # noqa: E402

FAST = dict(draws=150, tune=150, chains=2, random_seed=3, progressbar=False)


def _gauss_model():
    y = np.random.default_rng(0).normal(1.0, 2.0, 40)
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 10)
        s = pm.HalfNormal("s", 5)
        lo = pm.Uniform("lo", 0, 1)
        pm.Uniform("u", lower=lo, upper=lo + 2)  # bounds that depend on another RV
        pm.Normal("y", mu, s, observed=y)
    return m


def test_adapter_logp_grad_and_transform_match_independent_oracles():
    m = _gauss_model()
    tg = PyMCTarget(m, jacobian=True)
    z = tg.to_z([1.0, 2.0, 0.3, 1.0])
    # analytic transforms: identity, log, logit((x-lo)/(hi-lo)) with lo = 0, 1 and for u: lo = 0.3, hi = 2.3
    want = [1.0, np.log(2.0), np.log(0.3 / 0.7), np.log((1.0 - 0.3) / (2.3 - 1.0))]
    assert np.allclose(z, want, atol=1e-9)
    point = dict(zip([v.name for v in m.value_vars], z, strict=True))
    assert tg.logp(z) == pytest.approx(float(m.compile_logp(jacobian=True)(point)), abs=1e-9)
    assert PyMCTarget(m, jacobian=False).logp(z) == pytest.approx(float(m.compile_logp(jacobian=False)(point)), abs=1e-9)
    e = 1e-6
    num = [(tg.logp(z + e * np.eye(4)[i]) - tg.logp(z - e * np.eye(4)[i])) / (2 * e) for i in range(4)]
    assert np.allclose(tg.grad(z), num, rtol=1e-5, atol=1e-5)
    assert tg.loglike(z) < 0 and np.allclose(tg.loglike_grad(z)[2:], 0.0)  # likelihood does not see lo, u


def test_jacobian_flag_changes_the_density_by_the_log_jacobian():
    with pm.Model() as m:
        pm.HalfNormal("s", 5)
    z = np.array([0.7])  # s = exp(z): the log Jacobian is +z
    d = PyMCTarget(m, jacobian=True).logp(z) - PyMCTarget(m, jacobian=False).logp(z)
    assert d == pytest.approx(0.7, abs=1e-9)


def _bimodal(w=0.3, sep=12.0):
    with pm.Model() as m:
        x = pm.Flat("x")
        pm.Potential("two", pm.math.logsumexp(pm.math.stack([
            np.log1p(-w) - 0.5 * x**2, np.log(w) - 0.5 * (x - sep) ** 2])))
    return m


def _stuck_in_mode_one(m):
    with m:
        return pm.sample(draws=200, tune=200, chains=2, cores=1, progressbar=False, random_seed=1,
                         initvals={"x": 0.0}, init="adapt_diag")


def test_unvisited_mode_of_a_pymc_model_fails_with_the_laplace_mass_of_the_toy():
    # Flat prior -> no prior_predictive start: give the search proper starts through a Normal prior instead
    with pm.Model() as m:
        x = pm.Normal("x", 0, 8)
        pm.Potential("two", pm.math.logsumexp(pm.math.stack([
            np.log(0.7) - 0.5 * x**2, np.log(0.3) - 0.5 * (x - 12.0) ** 2])))
    with m:
        idata = pm.sample(draws=200, tune=200, chains=2, cores=1, progressbar=False, random_seed=1,
                          initvals={"x": 0.0}, init="adapt_diag")
    assert np.abs(idata.posterior["x"].values).max() < 5  # the chains never saw the second mode
    rep = search_pymc(m, idata, ModeSearchConfig(n_starts=12, seed=2))
    assert rep["error"] is None and rep["ok"] is False and rep["n_modes"] >= 2
    assert "no chain visited it" in rep["text"]
    json.dumps(rep)  # serialisable, so it can ride in idata.attrs


def test_flat_prior_is_reported_as_an_error_not_raised():
    m = _bimodal()
    rep = search_pymc(m, None, ModeSearchConfig(n_starts=4))
    assert rep["ok"] is None and rep["error"] and "mode search ERROR" in rep["text"]


def test_sample_attaches_the_report_and_a_fail_warns_but_keeps_the_fit():
    with pm.Model() as m:
        x = pm.Normal("x", 0, 8)
        pm.Potential("two", pm.math.logsumexp(pm.math.stack([
            np.log(0.7) - 0.5 * x**2, np.log(0.3) - 0.5 * (x - 12.0) ** 2])))
    cfg = SamplingConfig(draws=200, tune=200, chains=2, random_seed=1, progressbar=False,
                         extra_kwargs=dict(initvals={"x": 0.0}, init="adapt_diag", cores=1),
                         mode_search=ModeSearchConfig(n_starts=12, seed=2))
    with pytest.warns(UserWarning, match="mode search did not pass"):
        trace = _sample(pm, m, cfg)
    rep = mode_search_summary(trace)
    assert rep["ok"] is False and trace.posterior["x"].size == 400  # not blocked


def test_mode_search_none_turns_it_off_and_nondefault_jacobian_is_declared():
    off = SamplingConfig(**FAST, mode_search=None)
    assert mode_search_summary(_sample(pm, _gauss_model(), off)) is None
    cfg = SamplingConfig(**FAST, mode_search=ModeSearchConfig(n_starts=4, jacobian=False, likelihood=True))
    rep = mode_search_summary(_sample(pm, _gauss_model(), cfg))
    assert rep["jacobian"] is False and "jacobian=False" in rep["text"] and rep["prior_decides"] is not None


def test_default_is_on_and_every_fit_path_carries_a_report_without_error():
    rng = np.random.default_rng(5)
    cfg = SamplingConfig(**FAST)  # the DEFAULT mode_search
    par = fit_parallax_model(QTable({"parallax": rng.normal(2.0, 0.05, 80) * u.mas}), sampling=cfg)
    dist = distance_model(QTable({"r_med_geo": rng.normal(0.5, 0.03, 80) * u.kpc,
                                  "parallax": np.full(80, 2.0) * u.mas}), sampling=cfg)
    pmo = proper_motion_2d_gaussian(rng.normal(-1.4, 0.2, 80), rng.normal(-1.9, 0.2, 80),
                                    sampling=SamplingConfig(**{**FAST, "chains": 1}))  # chains=2 raises EOFError in the worker pool of proper_motion_2d_gaussian on 33b3a41 too (no search involved)
    vel = velocity_model(rng.normal(20.0, 2.0, 80), sampling=cfg)
    for name, meta in (("parallax", par.metadata), ("distance", dist.metadata), ("pm", pmo.metadata),
                       ("velocity", vel.metadata)):
        rep = meta["mode_search"]
        assert rep is not None and rep["error"] is None and rep["ok"] is not None, (name, rep)
    # structure: the dict-returning paths
    r = np.sqrt(rng.uniform(0, 1, 150)) * 20
    res = king_unbinned(r, field_radius=20.0, sampling=SamplingConfig(**FAST))
    assert res["mode_search"]["error"] is None and res["mode_search"]["ok"] is not None
    assert "mode_search" not in king_unbinned(r, field_radius=20.0,
                                               sampling=SamplingConfig(**FAST, mode_search=None))


def test_a_fit_never_fails_because_the_search_does(monkeypatch):
    import erotica.analysis.modesearch as ms

    def boom(*a, **k):
        raise RuntimeError("optimiser exploded")

    monkeypatch.setattr(ms, "mode_search", boom)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        trace = _sample(pm, _gauss_model(), SamplingConfig(**FAST))
    rep = mode_search_summary(trace)
    assert rep["ok"] is None and "optimiser exploded" in rep["error"]
    assert any("mode search did not pass" in str(x.message) for x in w)


def test_search_backend_is_honoured_and_agrees_with_the_default():
    """Non-default knob first (lesson 29): ``backend="JAX"`` must reach the compiled functions, and give
    the same density and gradient as the default backend."""
    pytest.importorskip("jax")
    m = _gauss_model()
    a, b = PyMCTarget(m), PyMCTarget(m, backend="JAX")
    assert type(b._lp.maker.linker).__name__ == "JAXLinker" and type(a._lp.maker.linker).__name__ != "JAXLinker"
    z = a.to_z([1.0, 2.0, 0.3, 1.0])
    assert b.logp(z) == pytest.approx(a.logp(z), rel=1e-9)
    assert np.allclose(b.grad(z), a.grad(z), rtol=1e-7, atol=1e-9)
    assert search_pymc(m, None, ModeSearchConfig(n_starts=4, backend="JAX"))["error"] is None
