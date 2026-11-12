"""The beta-binomial dispersion of the PMS-classifier gate-noise simulations
(``tools/validation/pms_classifier/power_sim.py``, R113-01).

Closed form: Beta(p a, (1 - p) a) has mean p and variance p (1 - p) / (a + 1); with the concentration
of ``beta_concentration`` the coefficient of variation is the requested one. The formula that stood until
2026-11-12 (no 1/p) gives CV 3.16 at p = 0.05, 4.08 at p = 0.03 and 1.83 at p = 0.15 for cv = 1: red."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/validation/pms_classifier"))
import power_sim  # noqa: E402


@pytest.mark.parametrize("p", [0.03, 0.05, 0.15])
@pytest.mark.parametrize("cv", [0.5, 1.0])
def test_beta_has_the_declared_cv_exact_moments(p, cv):
    a = power_sim.beta_concentration(p, cv)
    assert np.sqrt((1 - p) / (p * (a + 1))) == pytest.approx(cv, rel=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("p", [0.03, 0.05, 0.15])
@pytest.mark.parametrize("cv", [0.5, 1.0])
def test_beta_draw_has_the_declared_mean_and_cv(p, cv):
    # the draw the simulations use: 2e6 draws, MC sd of the CV well below 0.02
    x = power_sim.cluster_fpr(p, cv, 2_000_000, np.random.default_rng(0))
    assert x.mean() == pytest.approx(p, rel=0.01)
    assert x.std() / x.mean() == pytest.approx(cv, abs=0.02)


def test_zero_cv_is_the_binomial_world():
    assert np.all(power_sim.cluster_fpr(0.05, 0.0, 10, np.random.default_rng(0)) == 0.05)


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_degenerate_p_is_the_binomial_world(p):
    assert np.all(power_sim.cluster_fpr(p, 1.0, 10, np.random.default_rng(0)) == p)
