"""The single-star draw of the interval experiment's generator (``tools/validation/isochrone_intervals/
iv_common._true_stars``) against the density the likelihood integrates (R113-03).

Closed form: two segments of equal CMD length with IMF weights ``w = [0.9, 0.1]``. The likelihood's
continuous deposit (``IsochroneFitter._segments``, decisions.md 2026-09-24 (later)) puts point
densities rho = (0.9, 0.5, 0.1) per unit length, so the segments carry 0.9 - 0.4 t and 0.5 - 0.4 t per
unit t: masses 0.7 and 0.3, mean positions (0.45 - 0.4/3)/0.7 = 0.45238 and (0.25 - 0.4/3)/0.3 =
0.38889, CDF (c0 t + c1 t^2 / 2) / M. The generator that stood until 2026-11-12 drew [0.9, 0.1] and
t uniform (mean 0.5): red on all three."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/validation/isochrone_intervals"))
iv = pytest.importorskip("iv_common")


class _Toy:
    """Only what ``_true_stars`` and ``IsochroneFitter._segments`` read: a three-point track
    G = 10, 11, 12 at constant colour, no binaries (``alpha = beta = 0`` and zero binary weight)."""

    alpha = 0.0
    beta = 0.0

    def _deposit(self, met, loga, dm, Av, xp=np):
        return {
            "G": np.array([10.0, 11.0, 12.0]),
            "col": np.array([1.0, 1.0, 1.0]),
            "w_single": np.array([0.9, 0.1]),
            "m_b": np.array([1.0]),
            "w_bin": [np.array([0.0])],
            "Gq": np.zeros((2, 1)),
            "cq": np.zeros((2, 1)),
        }

    def _segments(self, met, loga, dm, Av, xp=np):
        from erotica.analysis._isochrone import IsochroneFitter

        return IsochroneFitter._segments(self, met, loga, dm, Av, xp)


C0 = np.array([0.9, 0.5])
C1 = np.array([-0.4, -0.4])
MASS = C0 + 0.5 * C1  # 0.7, 0.3


def test_likelihood_coefficients_are_the_closed_form():
    # the oracle's premise, read from the likelihood code itself
    Ag, Ac, Bg, Bc, c0, c1 = iv._single_segments(_Toy(), 0, 0, 0, 0)
    np.testing.assert_allclose(c0, C0, rtol=1e-12)
    np.testing.assert_allclose(c1, C1, rtol=1e-12)


def test_singles_follow_the_likelihood_density():
    n = 200_000
    G, C = iv._true_stars(_Toy(), 0, 0, 0, 0, n, np.random.default_rng(1))
    assert np.all((G >= 10) & (G <= 12)) and np.allclose(C, 1.0)
    seg0 = G < 11
    # segment masses: binomial sd at n = 2e5 is 0.001
    assert seg0.mean() == pytest.approx(MASS[0], abs=0.005)
    for k, sel in ((0, seg0), (1, ~seg0)):
        t = G[sel] - (10.0 + k)
        mean = (C0[k] / 2 + C1[k] / 3) / MASS[k]
        assert t.mean() == pytest.approx(mean, abs=0.005)

        def cdf(x, k=k):
            x = np.clip(x, 0, 1)
            return (C0[k] * x + C1[k] * x**2 / 2) / MASS[k]

        assert stats.kstest(t, cdf).pvalue > 1e-3


@pytest.mark.parametrize(
    "c0, c1, u, t",
    [(1.0, 0.0, 0.3, 0.3), (0.0, 2.0, 0.25, 0.5), (0.9, -0.4, 1.0, 1.0), (0.9, -0.4, 0.0, 0.0)],
)
def test_linear_t_limits(c0, c1, u, t):
    assert float(iv._linear_t(np.array([c0]), np.array([c1]), np.array([u]))[0]) == pytest.approx(t)
