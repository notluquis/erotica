"""Closed-form oracles for the phase-by-phase profile tools (``tools/validation/isochrone_phases``).

A Gaussian profile ln L = -(a - mu)^2 / (2 s^2) has its maximum at mu, the Delta ln L = 0.5 crossings at
mu -+ s and the 2.0 crossings at mu -+ 2s; two such profiles centred at mu1, mu2 with equal s share a
best common age at the mean and 2 Delta ln L = (mu1 - mu2)^2 / (2 s^2).
The crossings are linear interpolations on a 0.05-dex grid: on a parabola that costs <= 0.005 dex."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/validation/isochrone_phases"))
phases = pytest.importorskip("phases")


def _prof(mu, s, c=0.0, grid=np.round(np.arange(5.7, 7.0001, 0.05), 3)):
    return {"loga": grid.tolist(), "loglike": (c - ((grid - mu) ** 2) / (2 * s * s)).tolist()}


def test_profile_summary_gaussian():
    s = phases.profile_summary(_prof(6.32, 0.1))
    assert s["loga_hat"] == pytest.approx(6.32, abs=1e-9)
    assert s["lo1"] == pytest.approx(6.22, abs=0.005)
    assert s["hi1"] == pytest.approx(6.42, abs=0.005)
    assert s["lo2"] == pytest.approx(6.12, abs=0.008)
    assert not (s["open_lo1"] or s["open_hi1"])


def test_profile_summary_open_edge():
    s = phases.profile_summary(_prof(5.6, 0.1))
    assert s["at_grid_edge"] and s["open_lo1"]


def test_one_age_vs_two_gaussians():
    r = phases.one_age_vs_two(_prof(6.10, 0.1, 3.0), _prof(6.50, 0.1, -5.0))
    # different constants: a ratio that drops either window's maximum cannot pass
    assert r["delta_loga"] == pytest.approx(-0.40, abs=1e-9)
    assert r["loga_shared"] == pytest.approx(6.30, abs=1e-9)
    assert r["two_dlnL"] == pytest.approx(8.0, rel=1e-6)
