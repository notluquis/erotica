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
    assert not ok and "no draw visited it" in text


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
