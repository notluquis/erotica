#!/usr/bin/env python3
"""NGC 6383, phase by phase, with each grid (hub finding ``isochrone-ms-vs-pms.md`` §0.5). Run only
after ``synth_phases.py`` (pre-registered order: the null before the real ages).

Per grid (MIST v1.2, MIST v2.5, PARSEC v1.2S):

* ``global``: the 254 members, ``common.search`` with the parallax dm prior, [Fe/H] free over the
  grid's nodes (PARSEC's local file starts at [M/H] -0.18: declared, not hidden).
* ``global_feh0``: the same with [Fe/H] fixed at 0 (arm ``c``): the only configuration in which the
  three grids are compared at the same composition, so the difference between them is the grid's
  colour / physics and not a metallicity it chose.
* per window (``phases.WINDOWS``), profiles of log t: arm ``a`` ([Fe/H], dm, A_V fixed at ``global``),
  arm ``b`` (A_V free per window), arm ``c`` (fixed at ``global_feh0``).
* ``counts``: model-predicted members per G slice and brighter than the brightest member (G 8.80),
  at ``global``.
* ``gencheck`` (MIST v1.2 only): 40 synthetic samples drawn at ``global`` against the real counts and
  colour quantiles per window, the generator's predictive check per window (§A.1.3).

Output: ``real_<grid>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phases import (  # noqa: E402
    PRI_PLX,
    SAMPLE,
    WINDOWS,
    full_fitter,
    mist,
    one_age_vs_two,
    search,
    window_counts,
)
from synth_phases import phase_profiles, synth_table  # noqa: E402

CACHE = Path.home() / ".cache/erotica-grids"
NGC = Path("/Users/notluquis/erotica/data/test/NGC6383")


def grid_and_pri(name: str):
    from erotica.analysis import grids as G

    if name == "mist12":
        return mist(), dict(PRI_PLX)
    if name == "mist25":
        g = G.MISTGrid(
            CACHE / "mist_v2.5/UBVRIplus_afe0_vvcrit0.4",
            bands=("Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"),
            loga_range=(5.9, 7.1),
        ).select(feh=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25])
        return g, {**PRI_PLX, "loga_range": (5.9, 7.0)}
    if name == "parsec":
        g = G.PARSECGrid(NGC / "PARSEC/gaiaedr3", loga_range=(5.9, 7.1), max_mass=20.0)
        return g, {**PRI_PLX, "loga_range": (5.9, 7.0)}
    raise ValueError(name)


def bands_of(name: str):
    return (
        ("Gmag", "G_BPmag", "G_RPmag")
        if name == "parsec"
        else (
            "Gaia_G_EDR3",
            "Gaia_BP_EDR3",
            "Gaia_RP_EDR3",
        )
    )


def fitter_on(grid, name, data, pri, window=None):
    from erotica.analysis._isochrone import IsochroneFitter

    b = bands_of(name)
    f = IsochroneFitter(grid=grid, magnitude=b[0], color=b[1:], **pri)
    if window is None:
        f.setup(data, prob_threshold=0.0)
    else:
        lo, hi = window
        hi = float(np.max(np.asarray(data["Gmag"], float))) + 1e-6 if hi is None else hi
        f.setup(data, prob_threshold=0.0, mag_window=(lo, hi))
    return f


def search_fixed_feh(f, feh: float, age_stride: int = 2, polish: int = 4) -> dict:
    """``common.search`` with [Fe/H] held at ``feh`` (any value inside the grid's nodes, not only a
    node): lattice over every ``age_stride``-th age node optimising (dm, A_V) at the wide width, then
    the ``polish`` best polished on (log t, dm, A_V, sigma_int, f_bg). Parallax dm prior included."""
    from scipy.optimize import minimize

    t0 = time.time()
    fl = f._compiled_loglike(None)
    lo = np.array([f.loga_range[0], f.dm_range[0], f.Av_range[0], 1e-4, 1e-4])
    hi = np.array([f.loga_range[1], f.dm_range[1], f.Av_range[1], 0.5, 0.5])

    def fn(y5):
        y = np.r_[feh, y5]
        v, g = fl(y)
        z = (y[2] - f.dm_mu) / f.dm_sigma
        g = g.copy()
        g[2] -= z / f.dm_sigma
        return v - 0.5 * z * z, g[1:]

    ages = [a for a in f._node_loga if lo[0] <= a <= hi[0]][::age_stride]
    cands = []
    for a in ages:

        def nll2(y, a=float(a)):
            v, g = fn(np.array([a, y[0], y[1], 0.05, 0.02]))
            return -v, -g[1:3]

        r = minimize(
            nll2,
            [f.dm_mu, 1.0],
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(lo[1:3], hi[1:3], strict=True)),
        )
        cands.append((float(r.fun), float(a), *map(float, r.x)))
    cands.sort(key=lambda c: c[0])
    pol = []
    for c in cands[:polish]:
        y0 = np.clip(np.array([*c[1:], 0.03, 0.02]), lo, hi)
        r = minimize(
            lambda y: tuple(-v for v in fn(y)),
            y0,
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(lo, hi, strict=True)),
        )
        pol.append((-float(r.fun), np.asarray(r.x, float)))
    pol.sort(key=lambda p: -p[0])
    ll, b = pol[0]
    m = dict(zip(("loga", "dm", "Av", "sigma_int", "f_bg"), b.tolist(), strict=True))
    m = {"feh": float(feh), **m}
    bounds = {"loga": f.loga_range, "dm": f.dm_range, "Av": f.Av_range}
    at = [k for k, (l_, h_) in bounds.items() if min(m[k] - l_, h_ - m[k]) < 1e-3 * (h_ - l_)]
    return {
        "mode": m,
        "logpost": ll,
        "at_bound": at,
        "n": int(f._N_obs),
        "seconds": round(time.time() - t0, 1),
    }


def gencheck(grid, data, theta, n_draw=40, seed=7) -> dict:
    """Synthetic samples at ``theta`` against the real per-window counts and colour quantiles."""
    f = full_fitter(grid, data)
    g_real = np.asarray(data["Gmag"], float)
    c_real = np.asarray(data["G_BPmag"], float) - np.asarray(data["G_RPmag"], float)
    rng = np.random.default_rng(seed)
    rows = {w: {"n": [], "c50": []} for w in WINDOWS}
    for _ in range(n_draw):
        dm = theta["dm"]
        syn = synth_table(f, {**theta}, dm, rng)
        G = np.asarray(syn["Gmag"])
        C = np.asarray(syn["G_BPmag"])
        nbg = rng.binomial(len(G), theta["f_bg"])  # field stars: uniform in the members' box
        if nbg:
            G[:nbg] = rng.uniform(g_real.min(), g_real.max(), nbg)
            C[:nbg] = rng.uniform(c_real.min(), c_real.max(), nbg)
        for w, (lo, hi) in WINDOWS.items():
            m = (G >= lo) & (G < (hi or 99))
            rows[w]["n"].append(int(m.sum()))
            rows[w]["c50"].append(float(np.median(C[m])) if m.any() else np.nan)
    out = {}
    for w, (lo, hi) in WINDOWS.items():
        m = (g_real >= lo) & (g_real < (hi or 99))
        n = np.array(rows[w]["n"])
        c = np.array(rows[w]["c50"])
        out[w] = {
            "real_n": int(m.sum()),
            "synth_n_p16_50_84": np.percentile(n, [16, 50, 84]).tolist(),
            "real_colour_median": float(np.median(c_real[m])),
            "synth_colour_median_p16_50_84": np.nanpercentile(c, [16, 50, 84]).tolist(),
        }
    return out


def main() -> None:
    from astropy.table import QTable, Table

    import erotica

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, choices=["mist12", "mist25", "parsec"])
    a = ap.parse_args()
    grid, pri = grid_and_pri(a.grid)
    data = QTable(Table.read(SAMPLE))
    path = HERE / f"real_{a.grid}.json"
    out = {
        "erotica_file": erotica.__file__,
        "grid": a.grid,
        "grid_name": grid.name,
        "priors": {k: list(v) if isinstance(v, tuple) else v for k, v in pri.items()},
        "windows": {k: list(v) for k, v in WINDOWS.items()},
    }

    def save():
        path.write_text(json.dumps(out, indent=1, default=float) + "\n")

    f = fitter_on(grid, a.grid, data, pri)
    out["global"] = search(f, verbose=False)
    print("global", out["global"]["mode"], out["global"]["at_bound"], flush=True)
    out["counts"] = window_counts(f, out["global"]["mode"])
    zlo, zhi = f._met_bounds()
    feh0 = float(np.clip(0.0, zlo, zhi))
    out["global_feh0"] = search_fixed_feh(f, feh0)
    print("global feh0", out["global_feh0"]["mode"], flush=True)
    out["counts_feh0"] = window_counts(f, out["global_feh0"]["mode"])
    del f
    save()
    if a.grid == "mist12":
        out["gencheck"] = gencheck(grid, data, out["global"]["mode"])
        save()

    def prof(shared, arms):
        res = {}
        for w in ("ms", "pms", "ums", "to"):
            res.update(_one(w, shared, arms))
        return res

    def _one(w, shared, arms):
        from phases import profile_loga, profile_summary

        f = fitter_on(grid, a.grid, data, pri, WINDOWS[w])
        r = {"n": int(f._N_obs)}
        for arm in arms:
            free = ("sigma_int", "f_bg") if arm == "a" else ("Av", "sigma_int", "f_bg")
            p = profile_loga(f, shared, free=free)
            r[arm] = {"profile": p, "summary": profile_summary(p)}
        return {w: r}

    sh = {k: out["global"]["mode"][k] for k in ("feh", "dm", "Av")}
    out["phases"] = prof(sh, ("a", "b"))
    save()
    sh0 = {k: out["global_feh0"]["mode"][k] for k in ("feh", "dm", "Av")}
    out["phases_feh0"] = prof(sh0, ("a",))
    out["tension"] = {
        "a": one_age_vs_two(
            out["phases"]["ms"]["a"]["profile"], out["phases"]["pms"]["a"]["profile"]
        ),
        "b": one_age_vs_two(
            out["phases"]["ms"]["b"]["profile"], out["phases"]["pms"]["b"]["profile"]
        ),
        "c": one_age_vs_two(
            out["phases_feh0"]["ms"]["a"]["profile"], out["phases_feh0"]["pms"]["a"]["profile"]
        ),
    }
    save()
    print("done", flush=True)


if __name__ == "__main__":
    _ = phase_profiles  # imported for the shared definition of the arms
    main()
