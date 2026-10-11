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


def _nested_ok(s):
    return s["lo2"] <= s["lo1"] <= s["loga_hat"] <= s["hi1"] <= s["hi2"]


# mu between nodes and s from far below to above the 0.05-dex node spacing. Before the fix, the
# Delta ln L crossings were interpolated around the best NODE while loga_hat is the parabola vertex,
# and the threshold llmax - 0.5 could sit above every node (vertex 0.51-3.83 above the best node in
# 17 of 140 profiles of synth_T1/T2.json): loga_hat fell outside [lo1, hi1], or [lo1, hi1] came out
# open at both grid edges while [lo2, hi2] was 0.005 dex wide. Not covered here: with an excess
# > 0.5 the new interval excludes the best evaluated node (14 of 112 synthetic intervals).
@pytest.mark.parametrize("s", [0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
@pytest.mark.parametrize("mu", [6.40, 6.405, 6.4125, 6.425, 6.435, 6.445])
def test_profile_summary_estimator_inside_nested_intervals(mu, s):
    p = _prof(mu, s)
    r = phases.profile_summary(p)
    assert _nested_ok(r), r
    # the Gaussian is far from the grid edges: no interval may be declared open
    assert not any(r[f"open_{x}{t}"] for x in ("lo", "hi") for t in "12"), r


def test_profile_summary_known_failure_case():
    # mu = 6.435, s = 0.01: parabola vertex 1.125 above the best node (6.45). Before the fix:
    # [lo1, hi1] = [5.70, 7.00] open both sides, [lo2, hi2] closed and narrow.
    r = phases.profile_summary(_prof(6.435, 0.01))
    assert _nested_ok(r), r
    assert not (r["open_lo1"] or r["open_hi1"])
    # the crossings cannot be resolved by the grid: no node other than the best lies inside
    assert r["unresolved1"]


def test_profile_summary_resolved_flag_on_wide_profile():
    r = phases.profile_summary(_prof(6.32, 0.1))
    assert not r["unresolved1"] and not r["unresolved2"]


def test_profile_summary_sawtooth_real_shape():
    # T2 replica 1, `ms` arm a (synth_T2.json): node values relative to the best node 6.45
    a = np.round(np.arange(6.30, 6.6001, 0.05), 3)
    ll = np.array([-11.02, -2.43, -1.81, 0.0, -6.93, -10.08, -11.6])
    r = phases.profile_summary({"loga": a.tolist(), "loglike": ll.tolist()})
    assert _nested_ok(r), r


def test_profile_summary_unresolved_when_only_best_node_inside():
    # peak on a node (T2's truth 6.40 is one): the best node is inside [lo1, hi1], no other is
    r = phases.profile_summary(_prof(6.40, 0.01))
    assert r["lo1"] <= 6.40 <= r["hi1"]
    assert r["unresolved1"] and r["unresolved2"]
