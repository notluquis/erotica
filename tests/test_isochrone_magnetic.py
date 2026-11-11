"""Closed-form oracles for the HR-shift transplant (``tools/validation/isochrone_magnetic/transplant.py``).

A toy grid whose bolometric correction is LINEAR in log Teff (BC_b = c_b * (log Teff - 3.6)) and
independent of log g: the transplant of a shift (dlogL, dlogTeff) then has the exact answer
M_b' = M_b - 2.5 dlogL - c_b dlogTeff. Three properties:

* a zero shift is the identity, bit for bit (the differential application);
* a uniform shift gives the closed form above;
* above the source's top mass the shift decays linearly to zero over ``TAPER``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/validation/isochrone_magnetic"))
tp = pytest.importorskip("transplant")

from erotica.analysis.grids.base import GridNode, IsochroneGrid  # noqa: E402

C = {"G": 2.0, "BP": 3.0, "RP": 1.0}


def _toy_grid():
    g = IsochroneGrid()
    g.bands = ("G", "BP", "RP")
    rng = np.random.default_rng(1)
    for feh in (-0.5, 0.0):
        for a in (6.0, 6.1, 6.2, 6.3):
            m = np.linspace(0.1, 3.0, 60)
            t = 3.45 + 0.3 * (m - 0.1) / 2.9 + 0.01 * (a - 6.0) + 0.002 * rng.standard_normal(60)
            lum = -1.0 + 2.0 * (m - 0.1) / 2.9 - 0.5 * (a - 6.0)
            mbol = tp.MBOL_SUN - 2.5 * lum
            mags = {b: mbol - C[b] * (t - 3.6) for b in g.bands}
            g._add(feh, a, GridNode(np.arange(60.0), m, mags, {"logte": t, "logl": lum}), 0.01)
    return g


def _src(dl, dt, top=1.3):
    ages = np.array([6.0, 6.3])
    ms = np.round(np.arange(0.1, top + 1e-9, 0.05), 3)
    return {
        "loga": ages,
        "mass": ms,
        "dlogl": np.full((2, len(ms)), dl),
        "dlogte": np.full((2, len(ms)), dt),
        "source": "toy",
        "mass_top": top,
    }


def test_zero_shift_is_identity_bit_for_bit():
    g = _toy_grid()
    g2, st = tp.transplant(g, None)
    for f, a in g.nodes():
        for b in g.bands:
            assert np.array_equal(g2.node(f, a).mags[b], g.node(f, a).mags[b])
    assert st["shifted"] == 0


def test_uniform_shift_matches_closed_form_iso_mode():
    """``iso`` extrapolates linearly, so on a linear BC it is exact at every point below the top mass.
    The ``table`` mode is NOT held to this: outside its Delaunay hull it takes the nearest point, and on
    this toy grid that loses up to 0.025 mag of the shift at the hull edges (measured 2026-10-12) --
    which is why the finding requires the two modes to agree before an E3 age is read."""
    g = _toy_grid()
    dl, dt = -0.05, -0.02
    g2, _ = tp.transplant(g, _src(dl, dt), bc_mode="iso")
    for f, a in g.nodes():
        n0, n1 = g.node(f, a), g2.node(f, a)
        s = n0.mass <= 1.3
        for b in g.bands:
            want = n0.mags[b][s] - 2.5 * dl - C[b] * dt
            assert n1.mags[b][s] == pytest.approx(want, abs=2e-3)


def test_shift_tapers_to_zero_above_top_mass():
    src = _src(-0.1, -0.03, top=1.3)
    m = np.array([1.0, 1.3, 1.45, 1.6, 2.0])
    v = tp.shift_at(src, "dlogl", m, 6.1)
    assert v == pytest.approx([-0.1, -0.1, -0.05, 0.0, 0.0], abs=1e-9)
