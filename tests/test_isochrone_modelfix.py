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
A = pytest.importorskip("aniso")


def _seg_args():
    xg, xc = np.array([1.0, 1.5]), np.array([2.0, 1.7])
    sg, sc = np.array([0.3, 0.1]), np.array([0.2, 0.15])
    Ag, Ac, Bg, Bc = np.array([1.2, 0.9]), np.array([1.9, 1.6]), np.array([2.0, 0.9]), np.array([2.5, 1.6])
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
    q = np.mean([mvn([Ag[0] + t * (Bg[0] - Ag[0]), Ac[0] + t * (Bc[0] - Ac[0])], cov).pdf([xg[0], xc[0]])
                 for t in ts])
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


MIST = Path("/Users/notluquis/erotica/data/test/NGC6383/MIST/UBVRIplus")


@pytest.mark.skipif(not MIST.exists(), reason="local MIST v1.2 files absent")
def test_arm_I_is_the_parent_likelihood():
    r = A.selftest()
    assert r["max_abs_diff"] < 1e-10
