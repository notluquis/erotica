"""MIST v1.2 with the HR shift of a spotted (SPOTS) or magnetic (Feiden 2016) model transplanted onto
it (hub finding ``agent-findings/isochrone-ms-pms-mechanism.md`` §0.2, E3).

WHY THIS EXISTS
---------------
The question is whether starspots or magnetic inhibition of convection raise the PMS age of NGC 6383
to the turn-on age. Neither model can be fitted to the Gaia CMD as it ships:

* SPOTS (Somers+2020, Zenodo 3593339) carries Gaia DR2 magnitudes from empirical MS colour tables,
  -99 below 0.45-0.65 Msun once f > 0, and starts at log t 6.0. Its native safe window in G is ~0.6
  mag wide, and on the lambda Ori oracle its CMD colour-Teff relation alone moved the age by
  -0.355 dex against MIST (``isochrone-grids-implementation.md`` §4).
* Feiden's public magnetic isochrones (github.com/gfeiden/MagneticUpperSco, MIT) have no magnitudes.

So each model is used only for what it is about: the change of (log L, log Teff) AT FIXED MASS AND
AGE between its spotted/magnetic version and its own standard version. That shift is added to every
MIST point, and the new magnitudes come from MIST's own bolometric corrections, interpolated in
(log Teff, log g) over the MIST points of the same [Fe/H]. The colour-Teff relation is then the same
in the standard and the shifted grid: the difference is interior physics only.

Every choice and the process it models (methodology §A.1):

* **Differential application.** M_b' = M_b + (Mbol' - Mbol) - (BC_b(T', g') - BC_b(T, g)), with the
  BC difference from ONE interpolator: a zero shift is the identity exactly (V2), and the BC
  interpolation error enters only through its gradient (V1 measures the error itself, held out).
* **log g** from mass, L and Teff (logg_sun 4.438, Teff_sun 5772 K), the same formula before and after.
* **Out of the models' domains** (choices, not physics; measured through the ``ms`` control):
  below the youngest model age (log t 6.0) the shift of log t 6.0; above the most massive model mass
  (SPOTS 1.30, Feiden 1.70 Msun) the shift of that mass, tapered linearly to zero over +0.3 Msun.
* **Two-temperature photometry** (SPOTS only, ``two_temp=True``): a spotted star is a surface at
  T_hot with a fraction f at T_cool = 0.8 T_hot. Its band flux is the area-weighted sum of the two,
  each with MIST's BC at its own temperature. The 2T - 1T magnitude difference of the SPOTS star is
  added on top of the 1T transplant. Temperatures outside the BC table's hull use the nearest point
  and are counted (``stats``).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".cache/erotica-grids"
SPOTS_DIR = CACHE / "spots"
FEIDEN_DIR = CACHE / "feiden16"
LOGG_SUN, LOGT_SUN, MBOL_SUN = 4.438, np.log10(5772.0), 4.74
TAPER = 0.3  # Msun over which the shift of the top model mass decays to zero


def logg_of(mass, logte, logl):
    return LOGG_SUN + np.log10(mass) + 4.0 * (logte - LOGT_SUN) - logl


# --------------------------------------------------------------------------------------------
# Source models: shifts on a (log t, mass) lattice
# --------------------------------------------------------------------------------------------


def _read_spots(f: float) -> dict:
    fp = SPOTS_DIR / f"f{int(round(f * 100)):03d}.isoc"
    names, rows = None, []
    for line in fp.read_text().splitlines():
        s = line.strip()
        if s.startswith("##"):
            tok = s.lstrip("#").split()
            if tok[:2] == ["logAge", "Mass"]:
                names = tok
            continue
        if s and names is not None:
            rows.append([float(v) for v in s.split()])
    a = np.asarray(rows, float)
    return {n: a[:, i] for i, n in enumerate(names)}


def _lattice(cols: dict, keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, dict]:
    ages = np.unique(np.round(cols["logAge"], 4))
    masses = np.unique(np.round(cols["Mass"], 4))
    out = {k: np.full((len(ages), len(masses)), np.nan) for k in keys}
    ia = np.searchsorted(ages, np.round(cols["logAge"], 4))
    im = np.searchsorted(masses, np.round(cols["Mass"], 4))
    for k in keys:
        out[k][ia, im] = cols[k]
    return ages, masses, out


def spots_shift(f: float) -> dict:
    """Delta log L, Delta log Teff (SPOTS f - SPOTS 0) and, for the 2T variant, the SPOTS star's
    log T_hot, log T_cool, log Teff, log g and fspot, on SPOTS' (log t, mass) lattice."""
    s0, s1 = _read_spots(0.0), _read_spots(f)
    keys = ("log(L/Lsun)", "log(Teff)", "log(T_hot)", "log(T_cool)", "log(g)", "Fspot")
    a0, m0, L0 = _lattice(s0, keys)
    a1, m1, L1 = _lattice(s1, keys)
    assert np.array_equal(a0, a1) and np.array_equal(m0, m1)
    return {
        "loga": a0,
        "mass": m0,
        "dlogl": L1["log(L/Lsun)"] - L0["log(L/Lsun)"],
        "dlogte": L1["log(Teff)"] - L0["log(Teff)"],
        "thot": L1["log(T_hot)"],
        "tcool": L1["log(T_cool)"],
        "teff": L1["log(Teff)"],
        "logg": L1["log(g)"],
        "fspot": L1["Fspot"],
        "source": f"SPOTS f={f} - f=0 (Zenodo 3593339 v1.0)",
        "mass_top": float(m0.max()),
    }


def _read_feiden(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def feiden_shift(mgrid=np.round(np.arange(0.09, 1.7001, 0.01), 3)) -> dict:
    """Delta log L, Delta log Teff (Feiden magnetic B_eq - standard), both interpolated in mass onto
    ``mgrid`` at each age (the magnetic files skip masses where a model failed: interpolation in
    mass bridges those gaps, declared)."""
    mag = sorted((FEIDEN_DIR / "mag").glob("*.iso"))
    ages, dl, dt = [], [], []
    for fm in mag:
        myr = float(fm.name.split("_")[1].replace("myr", ""))
        fs = FEIDEN_DIR / "std" / fm.name.replace("_magBeq", "")
        M, S = _read_feiden(fm), _read_feiden(fs)
        lo = max(M[:, 0].min(), S[:, 0].min())
        hi = min(M[:, 0].max(), S[:, 0].max())
        row_l = np.interp(mgrid, M[:, 0], M[:, 3]) - np.interp(mgrid, S[:, 0], S[:, 3])
        row_t = np.interp(mgrid, M[:, 0], M[:, 1]) - np.interp(mgrid, S[:, 0], S[:, 1])
        out = (mgrid < lo) | (mgrid > hi)
        row_l[out] = np.nan
        row_t[out] = np.nan
        ages.append(np.log10(myr * 1e6))
        dl.append(row_l)
        dt.append(row_t)
    o = np.argsort(ages)
    return {
        "loga": np.round(np.asarray(ages)[o], 4),
        "mass": mgrid,
        "dlogl": np.asarray(dl)[o],
        "dlogte": np.asarray(dt)[o],
        "source": "Feiden 2016 magnetic B_eq - standard (github gfeiden/MagneticUpperSco 9a9d16d)",
        "mass_top": float(mgrid.max()),
    }


def _fill_mass(a: np.ndarray) -> np.ndarray:
    """NaN-fill along mass by nearest valid value (edges of a row's mass range)."""
    a = a.copy()
    for i in range(a.shape[0]):
        ok = np.isfinite(a[i])
        if ok.any() and not ok.all():
            idx = np.where(ok)[0]
            a[i] = np.interp(np.arange(a.shape[1]), idx, a[i, idx])
    return a


def shift_at(src: dict, key: str, mass: np.ndarray, loga: float) -> np.ndarray:
    """Bilinear in (log t, mass) on the source lattice; log t clamped to the lattice, mass clamped
    below, and above the top mass the top-mass value times a linear taper to zero over TAPER."""
    from scipy.interpolate import RegularGridInterpolator

    tab = _fill_mass(src[key])
    f = RegularGridInterpolator((src["loga"], src["mass"]), tab, bounds_error=False, fill_value=None)
    a = float(np.clip(loga, src["loga"][0], src["loga"][-1]))
    m = np.clip(mass, src["mass"][0], src["mass"][-1])
    v = f(np.column_stack([np.full_like(m, a), m]))
    w = np.clip(1.0 - (mass - src["mass_top"]) / TAPER, 0.0, 1.0)
    w[mass <= src["mass_top"]] = 1.0
    return v * w if key in ("dlogl", "dlogte") else v


# --------------------------------------------------------------------------------------------
# MIST's own bolometric corrections in (log Teff, log g)
# --------------------------------------------------------------------------------------------


class BCTable:
    """BC_b = Mbol - M_b over the MIST points of one [Fe/H], interpolated linearly in
    (log Teff, log g) (Delaunay); outside the hull, the nearest point (counted)."""

    def __init__(self, grid, feh: float, bands, exclude_loga: float | None = None):
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        X, Y = [], []
        for f, a in grid.nodes():
            if abs(f - feh) > 1e-6 or (exclude_loga is not None and abs(a - exclude_loga) < 1e-6):
                continue
            n = grid.node(f, a)
            t, lum = n.extra["logte"], n.extra["logl"]
            g = logg_of(n.mass, t, lum)
            mbol = MBOL_SUN - 2.5 * lum
            X.append(np.column_stack([t, g]))
            Y.append(np.column_stack([mbol - n.mags[b] for b in bands]))
        X, Y = np.vstack(X), np.vstack(Y)
        _, u = np.unique(np.round(X, 5), axis=0, return_index=True)
        X, Y = X[u], Y[u]
        self.lin = LinearNDInterpolator(X, Y)
        self.near = NearestNDInterpolator(X, Y)
        self.bands = tuple(bands)
        self.n_outside = 0
        self.n_total = 0

    def __call__(self, logte, logg):
        q = np.column_stack([np.ravel(logte), np.ravel(logg)])
        v = self.lin(q)
        bad = ~np.all(np.isfinite(v), axis=1)
        if bad.any():
            v[bad] = self.near(q[bad])
        self.n_outside += int(bad.sum())
        self.n_total += len(q)
        return v


# --------------------------------------------------------------------------------------------
# The transplanted grid
# --------------------------------------------------------------------------------------------


class IsoBC:
    """Robustness alternative to :class:`BCTable` (deviation after V1, §0.3 of the finding): BC as a
    function of log Teff alone along ONE node's own points with M <= 2 Msun, linear, extrapolated
    linearly from the two coolest/hottest points. No log g dependence, no hull."""

    def __init__(self, node, bands, mmax: float = 2.0):
        s = node.mass <= mmax
        t, lum = node.extra["logte"][s], node.extra["logl"][s]
        o = np.argsort(t)
        self.t = t[o]
        mbol = MBOL_SUN - 2.5 * lum[o]
        self.y = np.column_stack([mbol - node.mags[b][s][o] for b in bands])
        _, u = np.unique(np.round(self.t, 6), return_index=True)
        self.t, self.y = self.t[u], self.y[u]
        self.n_outside = 0
        self.n_total = 0

    def __call__(self, logte, logg=None):
        x = np.ravel(logte)
        out = np.empty((len(x), self.y.shape[1]))
        for i in range(self.y.shape[1]):
            v = np.interp(x, self.t, self.y[:, i])
            lo, hi = x < self.t[0], x > self.t[-1]
            s0 = (self.y[1, i] - self.y[0, i]) / (self.t[1] - self.t[0])
            s1 = (self.y[-1, i] - self.y[-2, i]) / (self.t[-1] - self.t[-2])
            v[lo] = self.y[0, i] + s0 * (x[lo] - self.t[0])
            v[hi] = self.y[-1, i] + s1 * (x[hi] - self.t[-1])
            out[:, i] = v
        self.n_outside += int(np.sum((x < self.t[0]) | (x > self.t[-1])))
        self.n_total += len(x)
        return out


def transplant(
    grid, src: dict | None, two_temp: bool = False, name: str = "", bc_mode: str = "table"
) -> tuple:
    """A copy of ``grid`` (MIST v1.2 nodes) with ``src``'s HR shift applied at every point.
    ``src=None`` returns an identity copy through the same code path (V2). Returns (grid, stats)."""
    bands = grid.bands
    tables = {f: BCTable(grid, f, bands) for f in grid.feh_nodes} if bc_mode == "table" else {}
    g2 = copy.copy(grid)
    g2._nodes = {}
    stats = {"points": 0, "shifted": 0, "bc_outside": 0, "bc_total": 0, "two_temp_outside": 0}
    for f, a in grid.nodes():
        n = grid.node(f, a)
        m, t, lum = n.mass, n.extra["logte"], n.extra["logl"]
        if src is None:
            dl = np.zeros_like(m)
            dt = np.zeros_like(m)
        else:
            dl = shift_at(src, "dlogl", m, a)
            dt = shift_at(src, "dlogte", m, a)
        t2, l2 = t + dt, lum + dl
        g, gg2 = logg_of(m, t, lum), logg_of(m, t2, l2)
        bc = tables[f] if bc_mode == "table" else IsoBC(n, bands)
        b1, b2 = bc(t, g), bc(t2, gg2)
        dmbol = -2.5 * dl
        mags = {b: n.mags[b] + dmbol - (b2[:, i] - b1[:, i]) for i, b in enumerate(bands)}
        if two_temp and src is not None and "thot" in src:
            th = shift_at(src, "thot", m, a)
            tc = shift_at(src, "tcool", m, a)
            te = shift_at(src, "teff", m, a)
            lg = shift_at(src, "logg", m, a)
            fs = shift_at(src, "fspot", m, a)
            n0 = bc.n_outside
            bh, bcool, be = bc(th, lg), bc(tc, lg), bc(te, lg)
            stats["two_temp_outside"] += bc.n_outside - n0
            if bc_mode != "table":
                stats["bc_outside"] += bc.n_outside
                stats["bc_total"] += bc.n_total
            # magnitudes up to a common constant: m = -2.5 log(area * T^4) - BC
            m1t = -2.5 * (4 * te)[:, None] - be
            fh = (1 - fs)[:, None] * 10 ** (4 * th)[:, None] * 10 ** (0.4 * bh)
            fc = fs[:, None] * 10 ** (4 * tc)[:, None] * 10 ** (0.4 * bcool)
            m2t = -2.5 * np.log10(fh + fc)
            d2 = m2t - m1t
            # only where the source mass applies (taper the correction like the shift)
            w = np.clip(1.0 - (m - src["mass_top"]) / TAPER, 0.0, 1.0)
            w[m <= src["mass_top"]] = 1.0
            for i, b in enumerate(bands):
                mags[b] = mags[b] + w * d2[:, i]
        from erotica.analysis.grids.base import GridNode

        extra = dict(n.extra)
        extra["logte"], extra["logl"] = t2, l2
        g2._add(f, a, GridNode(n.eep, m, mags, extra), grid._z.get(round(float(f), 6)))
        stats["points"] += len(m)
        stats["shifted"] += int(np.sum((np.abs(dl) > 0) | (np.abs(dt) > 0)))
        if bc_mode != "table" and not two_temp:
            stats["bc_outside"] += bc.n_outside
            stats["bc_total"] += bc.n_total
    if bc_mode == "table":
        stats["bc_outside"] = int(sum(t.n_outside for t in tables.values()))
        stats["bc_total"] = int(sum(t.n_total for t in tables.values()))
    g2.name = f"{grid.name} + {name or (src['source'] if src else 'identity')}"
    return g2, stats


def v1_bc_holdout(grid, holdout=(6.0, 6.3, 6.6), mlo=0.1, mhi=1.5) -> dict:
    """V1: BC interpolation error at held-out age nodes, on the points of the PMS window masses."""
    out = {}
    bands = grid.bands
    for f in grid.feh_nodes:
        for a in holdout:
            if not grid.has(f, a):
                continue
            bc = BCTable(grid, f, bands, exclude_loga=a)
            n = grid.node(f, a)
            s = (n.mass >= mlo) & (n.mass <= mhi)
            t, lum = n.extra["logte"][s], n.extra["logl"][s]
            g = logg_of(n.mass[s], t, lum)
            pred = MBOL_SUN - 2.5 * lum[:, None] - bc(t, g)
            eg = pred[:, 0] - n.mags[bands[0]][s]
            ec = (pred[:, 1] - pred[:, 2]) - (n.mags[bands[1]][s] - n.mags[bands[2]][s])
            out[f"{f:+.2f}@{a}"] = {
                "n": int(s.sum()),
                "G_p68": float(np.percentile(np.abs(eg), 68)),
                "G_p99": float(np.percentile(np.abs(eg), 99)),
                "col_p68": float(np.percentile(np.abs(ec), 68)),
                "col_p99": float(np.percentile(np.abs(ec), 99)),
                "outside": bc.n_outside,
            }
    return out
