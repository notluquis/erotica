"""Tests for the identifiability toolkit.

The load-bearing oracle is **the verdict must flip**: the same model, the same N, the same
generating parameters, differing only in footprint, must be reported as measured at a wide
footprint and prior-dominated at a narrow one. A diagnostic that says the same thing in both
regimes is not diagnosing anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from erotica.analysis.identifiability import (
    attach_log_likelihood,
    munoz_criteria,
    posterior_geometry,
)

pm = pytest.importorskip("pymc")

A_TRUE, GAMMA_TRUE, FIELD = 1.65, 2.0, 70.0


def _eff_radii(rng, n, *, a=A_TRUE, gamma=GAMMA_TRUE, field_radius=FIELD):
    grid = np.linspace(0.0, field_radius, 100_001)
    pdf = 2.0 * np.pi * grid * (1.0 + (grid / a) ** 2) ** (-gamma / 2.0)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    return np.interp(rng.uniform(0.0, 1.0, n), cdf, grid)


def test_attached_log_likelihood_sums_to_the_point_process_density():
    """Oracle: the per-star split must reproduce ``sum(log lambda) - Lambda`` exactly.

    That total is the quantity the Potential adds to the model, so an independent recomputation of
    it is a real check on the decomposition rather than a restatement of it. The split attributes
    ``Lambda/N`` to each star, which is exact by the conditioning property of a Poisson process.
    """
    from erotica.analysis.inference import SamplingConfig
    from erotica.analysis.structure import EFFPriors, eff_expected_count, eff_unbinned

    rng = np.random.default_rng(4)
    r = _eff_radii(rng, 200)
    fit = eff_unbinned(
        r, field_radius=FIELD, priors=EFFPriors(),
        sampling=SamplingConfig(draws=300, tune=400, chains=2, random_seed=1, progressbar=False),
    )
    idata = attach_log_likelihood(fit["eff_trace"], r, FIELD, model="eff")
    assert "log_likelihood" in [g.strip("/") for g in idata.groups]

    per_star = np.asarray(idata["log_likelihood"]["point_process"].values)
    post = idata.posterior
    k = np.asarray(post["k"].values)[..., None]
    b = np.asarray(post["b"].values)[..., None]
    a = np.asarray(post["a"].values)[..., None]
    g = np.asarray(post["gamma"].values)[..., None]
    sigma = k * (1.0 + (r / a) ** 2) ** (-g / 2.0) + b
    expected = eff_expected_count(k[..., 0], b[..., 0], a[..., 0], g[..., 0], FIELD)
    independent = np.log(2.0 * np.pi * r * sigma).sum(axis=-1) - expected

    np.testing.assert_allclose(per_star.sum(axis=-1), independent, rtol=1e-10)


def test_posterior_geometry_finds_a_planted_correlation():
    """Oracle: two parameters constructed to be perfectly dependent must be flagged."""
    import xarray as xr

    draws = np.random.default_rng(0).normal(size=(2, 500))
    idata = xr.DataTree()
    idata["posterior"] = xr.DataTree(xr.Dataset(
        {"x": (("chain", "draw"), draws[:1]),
         "y": (("chain", "draw"), draws[:1] * 3.0 + 1.0),
         "z": (("chain", "draw"), draws[1:])},
        coords={"chain": [0], "draw": np.arange(500)}))

    geom = posterior_geometry(idata, ["x", "y", "z"])
    locked = {(p["a"], p["b"]) for p in geom["not_separately_identified"]}
    assert ("x", "y") in locked, geom["pairs"]
    assert all("z" not in pair for pair in locked)
    assert geom["condition_number"] > 100


def test_munoz_criteria_score_against_the_published_thresholds():
    """Pins the published numbers: 3, 1000, 20 (Muñoz, Padmanabhan & Geha 2012)."""
    ngc6383 = munoz_criteria(
        field_radius=70.0, half_number_radius=30.7, n_stars=628, central_to_background=325.0
    )
    assert ngc6383["fov_over_half_radius"]["value"] == pytest.approx(70.0 / 30.7)
    assert not ngc6383["fov_over_half_radius"]["passes"]
    assert not ngc6383["total_stars"]["passes"]
    assert ngc6383["central_to_background"]["passes"]
    assert ngc6383["fov_over_half_radius"]["threshold"] == 3.0
    assert ngc6383["total_stars"]["threshold"] == 1000


@pytest.mark.slow
def test_the_verdict_flips_with_the_footprint():
    """THE test. Same model, same N, same truth -- only the footprint differs.

    At ``r_tot/a = 2`` the EFF slope is prior-dominated and must be reported as not measured; at
    ``r_tot/a = 42`` the likelihood dominates and it must be reported as measured. A diagnostic that
    returns the same verdict in both regimes is detecting nothing, which is why this is the test
    that matters rather than the unit checks above.
    """
    from erotica.analysis.identifiability import identifiability_report
    from erotica.analysis.inference import SamplingConfig
    from erotica.analysis.structure import EFFPriors, _eff_model, eff_unbinned

    cfg = SamplingConfig(draws=1500, tune=1200, chains=4, random_seed=31, progressbar=False)
    verdicts = {}
    for ratio in (2.0, 42.0):
        field = ratio * A_TRUE
        r = _eff_radii(np.random.default_rng(9100), 150, field_radius=field)
        fit = eff_unbinned(r, field_radius=field, priors=EFFPriors(), sampling=cfg,
                           progressbar=False)
        idata = fit["eff_trace"]
        with _eff_model(pm, r, field, EFFPriors(), None, None):
            pm.compute_log_prior(idata)
        report = identifiability_report(
            idata, ["gamma", "a"], radii=r, field_radius=field, model="eff"
        )
        verdicts[ratio] = report["verdict"]["gamma"]

    assert "NOT MEASURED" in verdicts[2.0], verdicts
    assert verdicts[42.0] == "measured", verdicts
