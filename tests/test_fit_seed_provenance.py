"""Seeds and provenance of a fit (hub audit ``ngc6383-erotica-reproducibility-audit-2026-10-21.md``,
rows 10, 15 and 17).

Oracles, none of them the code under test:

* **the draws themselves**: a seed that is recorded must regenerate the fit (same process, same compiled
  model), and a different seed must give different draws -- the negative control without which a seed
  that is silently ignored would pass;
* **what the search received**: the seed argument reaching ``PyMCTarget.prior_starts`` is spied, not
  read back from the report the same code wrote;
* **``to_netcdf``**: the provenance must survive the file format a fit is stored in.

Lesson 29 (``~/.claude/CLAUDE.md``): ``ModeSearchConfig.seed`` gained a ``None`` default that derives the
seed; the NON-default value (an explicit seed) is tested first, then the derived one.
"""

import numpy as np
import pytest

pm = pytest.importorskip("pymc")

from erotica.analysis import modesearch  # noqa: E402
from erotica.analysis.inference import (  # noqa: E402
    SamplingConfig,
    _sample,
    fit_provenance_summary,
    mode_search_summary,
)
from erotica.analysis.modesearch import ModeSearchConfig, derive_seed  # noqa: E402

FAST = dict(draws=60, tune=60, chains=2, progressbar=False)


def _model():
    y = np.random.default_rng(0).normal(1.0, 2.0, 30)
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 10)
        s = pm.HalfNormal("s", 5)
        pm.Normal("y", mu, s, observed=y)
    return m


@pytest.fixture
def spy_prior_starts(monkeypatch):
    seen = []
    orig = modesearch.PyMCTarget.prior_starts

    def spy(self, n, seed=0):
        seen.append(seed)
        return orig(self, n, seed)

    monkeypatch.setattr(modesearch.PyMCTarget, "prior_starts", spy)
    return seen


def test_an_explicit_mode_search_seed_is_kept_whatever_the_fit_seed(spy_prior_starts):
    """Non-default first (lesson 29): ``seed=2`` reaches the search for two different fit seeds."""
    for fit_seed in (3, 4):
        cfg = SamplingConfig(**FAST, random_seed=fit_seed, mode_search=ModeSearchConfig(n_starts=4, seed=2))
        rep = mode_search_summary(_sample(pm, _model(), cfg))
        assert rep["seed"] == 2
    assert spy_prior_starts == [2, 2]


def test_the_default_mode_search_seed_follows_the_fit_seed(spy_prior_starts):
    """``seed=None`` (the default) is derived from ``random_seed``: two fit seeds, two search seeds, and the
    one recorded is the one the search received. Until 2026-10-21 every search used seed 0."""
    reps = []
    for fit_seed in (3, 4):
        cfg = SamplingConfig(**FAST, random_seed=fit_seed, mode_search=ModeSearchConfig(n_starts=4))
        trace = _sample(pm, _model(), cfg)
        reps.append(mode_search_summary(trace))
        assert fit_provenance_summary(trace)["seeds"]["mode_search"] == reps[-1]["seed"]
    assert spy_prior_starts == [reps[0]["seed"], reps[1]["seed"]]
    assert spy_prior_starts[0] != spy_prior_starts[1]
    assert 0 not in spy_prior_starts
    assert reps[0]["seed"] == derive_seed(3, "mode_search")


def test_a_fit_without_seed_records_one_that_regenerates_it():
    """``random_seed=None``: the seed drawn is recorded, and refitting with it gives the same draws
    (bit for bit, same process). A different seed gives different draws: the negative control."""
    m = _model()
    a = _sample(pm, m, SamplingConfig(**FAST, random_seed=None, mode_search=None))
    seed = fit_provenance_summary(a)["seeds"]["random_seed"]
    assert isinstance(seed, int)
    b = _sample(pm, m, SamplingConfig(**FAST, random_seed=seed, mode_search=None))
    c = _sample(pm, m, SamplingConfig(**FAST, random_seed=seed + 1, mode_search=None))
    assert np.array_equal(a.posterior["mu"].values, b.posterior["mu"].values)
    assert not np.array_equal(a.posterior["mu"].values, c.posterior["mu"].values)


def test_provenance_names_engine_versions_and_survives_netcdf(tmp_path):
    import arviz as az

    trace = _sample(pm, _model(), SamplingConfig(**FAST, random_seed=5, mode_search=None))
    rec = fit_provenance_summary(trace)
    assert rec["seeds"] == {"random_seed": 5, "mode_search": None}
    assert rec["nuts_sampler"] == "nutpie" and rec["nuts_backend"]["mode"] == "JAX"
    for dist in ("nutpie", "pymc", "pytensor", "jax"):
        assert dist in rec["dependencies"]
    assert rec["erotica_version"]
    path = tmp_path / "t.nc"
    trace.to_netcdf(str(path))
    assert fit_provenance_summary(az.from_netcdf(str(path))) == rec
