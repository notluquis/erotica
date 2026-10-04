"""The isochrone run registry (``tools/validation/isochrone_runs``) says what its sources say.

Two oracles, both independent of ``export_run.py``:

* ``ngc_0.json`` -- C1's summary, written by ``isochrone_c1_chunked.py finalize`` through PyMC's own
  jaxified transforms. The exported quantiles come from the raw chains through ``export_run``'s
  numpy transforms, so agreement checks the transforms, the bounds and the chain order.
  ``ngc_0.json`` is rounded to 5 decimals: hence ``atol=1e-5`` (``assert_allclose`` defaults to 0).
* ``IsochroneFitter._median_isochrone`` -- the package's own path from a posterior to the curve.
  Needs the local MIST files, so it is skipped where they are absent (CI).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "tools/validation/isochrone_runs/runs"
C1_RUN = RUNS / "c1-ngc6383-unbinned-eep-v1-96ce4c0.json"
C1_SUMMARY = ROOT / "tools/validation/isochrone_unbinned/ngc_0.json"
REQUIRED = {
    "schema_version",
    "id",
    "date",
    "erotica_commit",
    "model",
    "cluster",
    "method",
    "config",
    "posterior",
    "diagnostics",
    "cmd",
    "isochrone",
    "notes",
}


@pytest.mark.parametrize("path", sorted(RUNS.glob("*.json")), ids=lambda p: p.stem)
def test_every_run_carries_the_schema(path):
    run = json.loads(path.read_text())
    assert REQUIRED <= set(run), REQUIRED - set(run)
    assert run["id"] == path.stem
    cmd = run["cmd"]
    n = len(cmd["mag"])
    assert n > 0 and all(len(cmd[k]) == n for k in ("color", "e_mag", "e_color", "source_id"))
    med = run["isochrone"]["median"]
    m = len(med["eep"])
    assert m > 10 and all(len(med[k]) == m for k in ("mass", "color", "mag"))
    assert np.all(np.diff(med["mass"]) >= 0), (
        "median curve must be ordered along EEP, mass non-decreasing"
    )


def test_c1_export_reproduces_ngc0_quantiles_and_diagnostics():
    run = json.loads(C1_RUN.read_text())
    ref = json.loads(C1_SUMMARY.read_text())
    assert run["cmd"]["mag"] and len(run["cmd"]["mag"]) == ref["N"] == 254
    for p in ("met", "loga", "dm", "Av", "sigma_int", "f_bg"):
        got = [run["posterior"][p][k] for k in ("q05", "q16", "q50", "q84", "q95")]
        np.testing.assert_allclose(got, ref[p]["q05_16_50_84_95"], rtol=0, atol=1e-5, err_msg=p)
        np.testing.assert_allclose(
            run["posterior"][p]["chain_means"], ref[p]["chain_means"], rtol=0, atol=1e-5, err_msg=p
        )
        assert abs(run["posterior"][p]["rhat"] - ref[p]["rhat"]) < 1e-4, p
        assert abs(run["posterior"][p]["ess_bulk"] - ref[p]["ess_bulk"]) < 1.0, p
    d = run["diagnostics"]
    assert d["divergences"] == ref["divergences"]
    np.testing.assert_allclose(d["bfmi"], ref["bfmi"], rtol=0, atol=1e-3)


def test_c1_median_curve_matches_the_package():
    pytest.importorskip("pytensor")
    import sys

    sys.path.insert(0, str(ROOT / "tools/validation"))
    try:
        from isochrone_nuts_convergence import MIST, PRIORS, SAMPLE
    except Exception as e:  # pragma: no cover - the module reads hardcoded local paths
        pytest.skip(f"validation module unavailable: {e}")
    if not Path(MIST).is_dir() or not Path(SAMPLE).exists():
        pytest.skip("MIST files / NGC 6383 sample not on this machine")
    from arviz_base import from_dict
    from astropy.table import QTable, Table

    from erotica.analysis._isochrone import IsochroneFitter

    run = json.loads(C1_RUN.read_text())
    pri = {k: v for k, v in PRIORS.items() if k not in ("M_met", "M_loga")}
    f = IsochroneFitter(isochs_path=MIST, **pri)
    f.setup(QTable(Table.read(SAMPLE)), prob_threshold=0.0)
    med = {p: run["posterior"][p]["q50"] for p in ("met", "loga", "dm", "Av")}
    # a one-draw posterior: its median is the point itself
    idata = from_dict({"posterior": {p: np.full((1, 1), v) for p, v in med.items()}})
    got_med, mass, G, col = f._median_isochrone(idata)
    assert got_med == pytest.approx(med)
    eep = np.asarray(run["isochrone"]["median"]["eep"])
    np.testing.assert_allclose(run["isochrone"]["median"]["mag"], G[eep], rtol=0, atol=1e-5)
    np.testing.assert_allclose(run["isochrone"]["median"]["color"], col[eep], rtol=0, atol=1e-5)
    np.testing.assert_allclose(run["isochrone"]["median"]["mass"], mass[eep], rtol=0, atol=1e-5)
