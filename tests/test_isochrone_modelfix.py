"""The arms of ``tools/validation/isochrone_modelfix/aniso.py`` (hub finding
``isochrone-model-fix.md``): the per-star G-colour covariance of the segment density, the
exponential field density, and arm I against the parent fitter. Oracles: ``scipy`` (bivariate normal,
quadrature) and ``erotica.analysis._isochrone`` itself. The last test needs the local MIST files and is
skipped where they are absent (CI)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/validation/isochrone_modelfix"))
# A plain import, not importorskip: aniso needs only numpy and erotica, so a failure to import it
# is a broken tree (e.g. isochrone_intervals/iv_common.py missing), not an absent optional extra.
import aniso as A  # noqa: E402


def _seg_args():
    xg, xc = np.array([1.0, 1.5]), np.array([2.0, 1.7])
    sg, sc = np.array([0.3, 0.1]), np.array([0.2, 0.15])
    Ag, Ac, Bg, Bc = (
        np.array([1.2, 0.9]),
        np.array([1.9, 1.6]),
        np.array([2.0, 0.9]),
        np.array([2.5, 1.6]),
    )
    return xg, xc, sg, sc, Ag, Ac, Bg, Bc


def test_zero_covariance_is_the_package_segment_density():
    from erotica.analysis._isochrone import _segment_density

    xg, xc, sg, sc, Ag, Ac, Bg, Bc = _seg_args()
    a = _segment_density(xg, xc, sg, sc, Ag, Ac, Bg, Bc, np, first_moment=True)
    b = A._seg_density_cov(xg, xc, sg, sc, 0.0, Ag, Ac, Bg, Bc, np, first_moment=True)
    np.testing.assert_allclose(a[0], b[0], rtol=1e-14)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-14)


def test_covariance_matches_bivariate_normal_and_quadrature():
    from scipy.stats import multivariate_normal as mvn

    xg, xc, sg, sc, Ag, Ac, Bg, Bc = _seg_args()
    s12 = np.array([0.04, 0.005])  # per star; each covariance positive definite
    d = A._seg_density_cov(xg, xc, sg, sc, s12, Ag, Ac, Bg, Bc, np)
    cov = [[sg[0] ** 2, s12[0]], [s12[0], sc[0] ** 2]]
    # segment (star 0, segment 0): average of the bivariate normal along it
    ts = np.linspace(0, 1, 4001)
    q = np.mean(
        [
            mvn([Ag[0] + t * (Bg[0] - Ag[0]), Ac[0] + t * (Bc[0] - Ac[0])], cov).pdf([xg[0], xc[0]])
            for t in ts
        ]
    )
    np.testing.assert_allclose(d[0, 0], q, rtol=2e-4)
    # zero-length segment (segment 1): the bivariate normal itself
    np.testing.assert_allclose(d[0, 1], mvn([Ag[1], Ac[1]], cov).pdf([xg[0], xc[0]]), rtol=1e-12)


def _bare(obs_mag, area):
    f = A.WidthFitter.__new__(A.WidthFitter)
    f._obs_mag = np.asarray(obs_mag, float)
    f._box_area = float(area)
    return f


@pytest.mark.parametrize("beta", [0.0, 0.8, -1.3])
def test_field_density_is_normalised_and_uniform_at_beta_zero(beta):
    lo, hi, span_c = 14.0, 19.0, 2.0
    g = np.linspace(lo, hi, 20001)
    f = _bare(g, (hi - lo) * span_c)
    d = f.field_density(beta)
    np.testing.assert_allclose(np.trapezoid(d, g) * span_c, 1.0, rtol=1e-6)
    if beta == 0.0:
        np.testing.assert_allclose(d, 1.0 / f._box_area, rtol=1e-12)


# The path mist() reads ($EROTICA_NGC6383_DATA or the checkout), so the skip and the use agree.
from common import MIST12 as MIST  # noqa: E402


@pytest.mark.skipif(not MIST.exists(), reason="local MIST v1.2 files absent")
def test_arm_I_is_the_parent_likelihood():
    r = A.selftest()
    assert r["max_abs_diff"] < 1e-10


class _LinearIso:
    """Two [Fe/H] x two age nodes, each with G, BP, RP linear in mass: the bilinear blend at any
    (feh, loga) inside is the same line, so the generator's band values have a closed form."""

    _met_values = [-0.5, 0.0]
    _loga_values = [6.0, 6.5]
    mass = np.linspace(0.1, 3.0, 300)

    @staticmethod
    def line(m):
        return np.array([12.0 - 2.0 * m, 12.6 - 1.7 * m, 11.4 - 2.2 * m])

    def get_isochrone_eep(self, feh, loga):
        G, BP, RP = self.line(self.mass)
        return None, self.mass[::-1].copy(), G[::-1].copy(), BP[::-1].copy(), RP[::-1].copy()


def test_independent_generator_bands_are_the_closed_form():
    """C3' generator (``kw_c3.ind_true_stars``): with bands linear in mass, a single sits on the
    line and a binary is the flux sum of the line at m1 and at max(q m1, lowest mass), plus dm and
    k A_V. Oracle: the line and the flux sum written here, not ``_mag_combine``."""
    import kw_c3

    iso, kG, kBP, kRP, dm, Av = _LinearIso(), 0.84, 1.08, 0.63, 10.2, 0.5
    rng = np.random.default_rng(7)
    for alpha, beta in [(0.0, 0.0), (1.0, 0.0), (0.3, 0.4)]:
        G, col, m1, isb, q = kw_c3.ind_true_stars(
            iso, kG, kBP, kRP, -0.2, 6.3, dm, Av, 4000, rng, alpha, beta, detail=True
        )
        if (alpha, beta) == (0.0, 0.0):
            assert not isb.any()
        elif (alpha, beta) == (1.0, 0.0):
            assert isb.all()
        else:
            assert 0.05 < isb.mean() < 0.95
        b1 = _LinearIso.line(m1)
        b2 = _LinearIso.line(np.maximum(q * m1, _LinearIso.mass[0]))
        pair = -2.5 * np.log10(10 ** (-0.4 * b1) + 10 ** (-0.4 * b2))
        bands = np.where(isb[None, :], pair, b1)
        np.testing.assert_allclose(G, bands[0] + dm + kG * Av, rtol=0, atol=1e-9)
        np.testing.assert_allclose(col, bands[1] - bands[2] + (kBP - kRP) * Av, rtol=0, atol=1e-9)
