"""Same seed, same fit -- across a fresh and a cached compile (decisions.md, 2026-10-21).

Measured before the fix: with nutpie, ``IsochroneFitter.fit`` gave 3 different draw hashes in 5
same-seed processes (hub audit ``ngc6383-erotica-reproducibility-audit-2026-10-21.md`` §2.3). Staged hashes
put the divergence before the sampler: ``find_start`` compiled its log L with PyTensor's default backend,
numba with ``fastmath``, and that function's value at a FIXED point differed in the last bits between a
fresh compile and the on-disk cache (``0x1.64aa9ad250cb2p+6`` vs ``0x1.64aa9ad250ca8p+6``); with fastmath
off, and with JAX, every process agreed. L-BFGS-B moved the mode in the 11th digit, nutpie took the
other start, and the draws were another set (log t mean 6.5417 vs 6.5371). Given the same start, nutpie on
JAX gave identical draws in 4 of 4 processes, two of them with a cold PyTensor cache.

The oracle is the draws of two processes that differ only in whether the compile cache was empty, plus a
third with another seed: without that negative control a fit that ignored its seed would pass.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymc")
pytest.importorskip("nutpie")
pytest.importorskip("jax")

REPO = Path(__file__).resolve().parents[1]

_SCRIPT = textwrap.dedent(
    """
    import hashlib, json, pathlib, sys
    import numpy as np
    from tests.test_isochrone import _TOY_TRUTH, _toy_fitter, _toy_stars
    from erotica.analysis.inference import fit_provenance_summary

    t = _TOY_TRUTH
    f = _toy_fitter(pathlib.Path(sys.argv[1]))
    f.setup(_toy_stars(200, t["loga"], t["met"], t["dm"], t["Av"], np.random.default_rng(11)),
            prob_threshold=0)
    idata = f.fit(draws=40, tune=40, chains=1, cores=1, random_seed=int(sys.argv[2]),
                  progressbar=False, mode_search=None)
    h = hashlib.sha256()
    for v in sorted(idata.posterior.data_vars):
        h.update(np.ascontiguousarray(idata.posterior[v].values).tobytes())
    rec = fit_provenance_summary(idata)
    print(json.dumps({"start": rec["start_sha256"], "draws": h.hexdigest(),
                      "seed": rec["seeds"]["random_seed"]}))
    """
)


def _fit_in_a_new_process(tmp: Path, compiledir: Path, seed: int) -> dict:
    tmp.mkdir()
    env = dict(os.environ)
    env["PYTENSOR_FLAGS"] = f"base_compiledir={compiledir}"
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), env.get("PYTHONPATH", "")])
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(tmp), str(seed)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=1800, check=False,
    )  # fmt: skip
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_same_seed_gives_the_same_draws_from_a_fresh_and_a_cached_compile(tmp_path):
    """Mutation (2026-10-21): ``mode="JAX"`` removed from the nutpie branch of ``fit`` -> the start and
    the draws of the fresh and the cached process differ (see ``decisions.md``)."""
    cache = tmp_path / "pytensor"  # empty: the first process compiles, the second loads the cache
    fresh = _fit_in_a_new_process(tmp_path / "a", cache, 11)
    cached = _fit_in_a_new_process(tmp_path / "b", cache, 11)
    other = _fit_in_a_new_process(tmp_path / "c", cache, 12)
    assert fresh["seed"] == cached["seed"] == 11
    assert fresh["start"] == cached["start"]
    assert fresh["draws"] == cached["draws"]
    assert other["draws"] != fresh["draws"]  # the seed reaches the sampler


def test_the_nutpie_start_search_is_compiled_with_jax(monkeypatch, tmp_path):
    """The cheap guard of the same fix, in the fast suite: ``fit`` asks ``find_start`` for JAX, and seeds
    it from ``random_seed``. Stops at the sampler call (no sampling).
    Mutation: drop ``mode="JAX"`` -> red; pass ``default_rng()`` unseeded -> red."""
    import pymc as pm

    from erotica.analysis import _isochrone
    from tests.test_isochrone import _TOY_TRUTH, _toy_fitter, _toy_stars

    t = _TOY_TRUTH
    f = _toy_fitter(tmp_path)
    f.setup(_toy_stars(60, t["loga"], t["met"], t["dm"], t["Av"], np.random.default_rng(1)), prob_threshold=0)
    seen = {}

    def fake_find_start(self, n_chains=4, rng=None, mode=None):
        seen["mode"] = mode
        seen["u"] = rng.uniform()
        return {"mode": {"met": t["met"], "loga": t["loga"], "dm": t["dm"], "Av": t["Av"],
                         "sigma_int": 0.02, "f_bg": 0.03}}  # fmt: skip

    class Stop(Exception):
        pass

    def fake_sample(*a, **k):
        seen["random_seed"] = k["random_seed"]
        raise Stop

    monkeypatch.setattr(_isochrone.IsochroneFitter, "find_start", fake_find_start)
    monkeypatch.setattr(pm, "sample", fake_sample)
    with pytest.raises(Stop):
        f.fit(draws=5, tune=5, chains=1, random_seed=7, progressbar=False, mode_search=None)
    assert seen["mode"] == "JAX"
    assert seen["random_seed"] == 7
    assert seen["u"] == np.random.default_rng(7).uniform()


def test_isochrone_fit_records_seed_start_law_and_derived_search_seed(monkeypatch, tmp_path):
    """Provenance of an isochrone fit (audit rows 10, 15, 17), with the sampler and the search stubbed:
    ``random_seed=None`` (draws one), a non-default ``extinction_law`` and the DEFAULT mode search, whose
    seed must be the one derived from the recorded fit seed -- the value the search received is spied."""
    import arviz as az
    import pymc as pm

    from erotica.analysis import _isochrone, inference
    from erotica.analysis.inference import fit_provenance_summary
    from erotica.analysis.modesearch import derive_seed
    from tests.test_isochrone import _TOY_TRUTH, _toy_fitter, _toy_stars

    t = _TOY_TRUTH
    f = _toy_fitter(tmp_path)  # extinction_law="ccm89": NOT the package default
    f.setup(_toy_stars(60, t["loga"], t["met"], t["dm"], t["Av"], np.random.default_rng(1)), prob_threshold=0)
    mode = {"met": t["met"], "loga": t["loga"], "dm": t["dm"], "Av": t["Av"], "sigma_int": 0.02, "f_bg": 0.03}
    monkeypatch.setattr(_isochrone.IsochroneFitter, "find_start", lambda self, *a, **k: {"mode": dict(mode)})
    seen = {}

    def fake_sample(*a, **k):
        seen["random_seed"] = k["random_seed"]
        return az.from_dict({"posterior": {"loga": np.zeros((1, 3))}})

    def fake_search(model, idata, cfg, **kw):
        seen["search_seed"] = cfg.seed

    monkeypatch.setattr(pm, "sample", fake_sample)
    monkeypatch.setattr(inference, "attach_mode_search", fake_search)
    idata = f.fit(draws=3, tune=3, chains=1, random_seed=None, progressbar=False)
    rec = fit_provenance_summary(idata)
    seed = rec["seeds"]["random_seed"]
    assert isinstance(seed, int) and seed == seen["random_seed"]
    assert rec["seeds"]["mode_search"] == seen["search_seed"] == derive_seed(seed, "mode_search")
    assert rec["extinction_law"] == "ccm89" and rec["fitter"] == "IsochroneFitter"
    assert rec["start"] == pytest.approx(mode) and len(rec["start_sha256"]) == 64
    assert rec["nuts_sampler"] == "nutpie" and "nutpie" in rec["dependencies"]
