#!/usr/bin/env python3
"""Drivers for the hub finding ``agent-findings/isochrone-ms-pms-mechanism.md`` (pre-registered in §0
there before any of these ran). One task per call, one JSON per task in ``out/``:

* ``where``       E1a (per-star l_i(log t) on the 254 at the global mode) + E1b (log t profile per
                  fine G slice, shared parameters fixed at the global mode, as arm ``a``).
* ``where_null``  E1b on the T1 synthetic replicates (same seeds 5000+r, same generator as
                  ``isochrone_phases/synth_phases.py``), shared parameters at the TRUTH.
* ``mech``        E2: R_V = 3.48, no binaries, f_bg = 0 in ``ms``.
* ``e3``          E3: MIST v1.2 with a transplanted HR shift (``transplant.py``) in arm a / s / g.

Shared conventions with ``isochrone_phases``: windows, profile grid, ``profile_loga`` and the fixed
``profile_summary`` (R33-01) are imported from there, never copied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "isochrone_phases"))
from phases import (  # noqa: E402
    EDR3,
    PRI_PLX,
    SAMPLE,
    WINDOWS,
    LOGA_PROFILE,
    mist,
    profile_loga,
    profile_summary,
)

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
REAL = json.loads((HERE.parent / "isochrone_phases/real_mist12.json").read_text())
GLOBAL = REAL["global"]["mode"]  # feh -0.472, loga 6.023, dm 10.224, Av 0.661, sigma 0.153, fbg 0.170
SHARED_A = {k: GLOBAL[k] for k in ("feh", "dm", "Av")}
SHARED_S = {"feh": 0.0, "dm": 10.223, "Av": GLOBAL["Av"]}
# Post-hoc arms (finding §0.5, declared before reading them): A_V at the 3D-map foreground level,
# 0.92 = midpoint of the 0.81-1.04 of isochrone-external-constraints.md §1 point 3 (members at d_c).
AV_MAP = 0.92
SHARED_M = {"feh": GLOBAL["feh"], "dm": GLOBAL["dm"], "Av": AV_MAP}
SHARED_SM = {"feh": 0.0, "dm": 10.223, "Av": AV_MAP}
SLICES = [8.80, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, None]  # §0.2 E1b
T1 = {"feh": -0.47, "loga": 6.02, "Av": 0.66, "sigma_int": 0.15, "f_bg": 0.0}
FEH3 = [-0.5, -0.25, 0.0]


def fitter(grid, data, window=None, **kw):
    from erotica.analysis._isochrone import IsochroneFitter

    f = IsochroneFitter(grid=grid, magnitude=EDR3[0], color=EDR3[1:], **PRI_PLX, **kw)
    if window is None:
        f.setup(data, prob_threshold=0.0)
    else:
        lo, hi = window
        hi = float(np.max(np.asarray(data["Gmag"], float))) + 1e-6 if hi is None else hi
        f.setup(data, prob_threshold=0.0, mag_window=(lo, hi))
    return f


def real_data():
    from astropy.table import QTable, Table

    return QTable(Table.read(SAMPLE))


def prof(f, shared, free=("sigma_int", "f_bg")):
    p = profile_loga(f, shared, free=free)
    return {"profile": p, "summary": profile_summary(p), "n": int(f._N_obs)}


def best_node(p):
    a, ll = np.asarray(p["loga"]), np.asarray(p["loglike"])
    return float(a[int(np.argmax(ll))])


def save(name, obj):
    import erotica

    obj = {"erotica_file": erotica.__file__, "pid": os.getpid(), **obj}
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float) + "\n")


def refit_global(f, feh, dm, start):
    """(log t, A_V, sigma_int, f_bg) re-fitted at fixed [Fe/H] and dm: lattice over every second age
    node with A_V optimised (sigma 0.05, f_bg 0.02), then the 3 best polished on the four. Never
    seeded at a truth: the lattice covers the whole prior (§A.1.3)."""
    from scipy.optimize import minimize

    fl = f._compiled_loglike(None)
    lo = np.array([f.loga_range[0], f.Av_range[0], 1e-4, 1e-4])
    hi = np.array([f.loga_range[1], f.Av_range[1], 0.5, 0.5])

    def nll(y):
        v, g = fl(np.array([feh, y[0], dm, y[1], y[2], y[3]]))
        return -v, -g[[1, 3, 4, 5]]

    cands = []
    for a in [x for x in f._node_loga if lo[0] <= x <= hi[0]][::2]:

        def n1(y, a=float(a)):
            v, g = nll([a, y[0], 0.05, 0.02])
            return v, g[1:2]

        r = minimize(n1, [start["Av"]], jac=True, method="L-BFGS-B", bounds=[(lo[1], hi[1])])
        cands.append((float(r.fun), float(a), float(r.x[0])))
    cands.sort()
    pol = []
    for c in cands[:3]:
        y0 = np.clip([c[1], c[2], 0.1, 0.05], lo, hi)
        r = minimize(nll, y0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi, strict=True)))
        pol.append((float(r.fun), r.x.tolist()))
    pol.sort()
    x = pol[0][1]
    return {
        "mode": {"feh": feh, "loga": x[0], "dm": dm, "Av": x[1], "sigma_int": x[2], "f_bg": x[3]},
        "loglike": -pol[0][0],
        "lattice_best": cands[:3],
    }


# --------------------------------------------------------------------------------------------


def task_where():
    data = real_data()
    grid = mist()
    out = {"global": GLOBAL, "slices": SLICES}
    # E1a: per-star l_i(log t) on the 254 at the global mode
    f = fitter(grid, data)
    G = np.asarray(data["Gmag"], float)
    C = np.asarray(data["G_BPmag"], float) - np.asarray(data["G_RPmag"], float)
    L = []
    for a in LOGA_PROFILE:
        L.append(
            f._star_loglike(
                GLOBAL["feh"], float(a), GLOBAL["dm"], GLOBAL["Av"], GLOBAL["sigma_int"],
                GLOBAL["f_bg"], np,
            )
        )
    L = np.asarray(L)  # (n_age, N)
    ids = [int(x) for x in np.asarray(data["source_id"])] if "source_id" in data.colnames else []
    extra = {}
    for c in ("probability_hdbscan", "parallax", "ruwe"):
        if c in data.colnames:
            extra[c] = np.asarray(data[c], float).tolist()
    out["per_star"] = {
        "loga": LOGA_PROFILE.tolist(),
        "G": G.tolist(),
        "bp_rp": C.tolist(),
        "source_id": ids,
        "loglike": L.T.tolist(),
        **extra,
    }
    save("where", out)
    # E1b: profiles per slice
    res = {}
    for lo, hi in zip(SLICES[:-1], SLICES[1:], strict=True):
        fw = fitter(grid, data, (lo, hi))
        key = f"{lo:g}-{hi if hi else 'faint'}"
        res[key] = prof(fw, SHARED_A)
        print("where", key, res[key]["n"], round(res[key]["summary"]["loga_hat"], 3), flush=True)
        out["slice_profiles"] = res
        save("where", out)


def task_where_null(reps):
    from synth_phases import DM_MU, DM_SD, synth_table

    grid = mist()
    gen = fitter(grid, real_data())
    out = {"truth": T1, "reps": []}
    for r in reps:
        rng = np.random.default_rng(5000 + r)
        dm = float(rng.normal(DM_MU, DM_SD))
        syn = synth_table(gen, T1, dm, rng)
        sh = {"feh": T1["feh"], "dm": dm, "Av": T1["Av"]}
        rep = {"rep": r, "dm": dm, "slices": {}}
        for lo, hi in zip(SLICES[:-1], SLICES[1:], strict=True):
            key = f"{lo:g}-{hi if hi else 'faint'}"
            g = np.asarray(syn["Gmag"], float)
            hh = float(g.max()) + 1e-6 if hi is None else hi
            if int(((g >= lo) & (g < hh)).sum()) < 3:
                rep["slices"][key] = None
                continue
            fw = fitter(grid, syn, (lo, hi))
            p = prof(fw, sh)
            rep["slices"][key] = {"n": p["n"], "summary": p["summary"], "best_node": best_node(p["profile"])}
        out["reps"].append(rep)
        print("null rep", r, {k: (round(v["summary"]["loga_hat"] - T1["loga"], 3) if v else None) for k, v in rep["slices"].items()}, flush=True)
        save(f"where_null_{reps[0]}_{reps[-1]}", out)


def task_mech(which):
    data = real_data()
    grid = mist()
    out = {"which": which}
    if which in ("rv", "bin"):
        kw = {"Rv": 3.48} if which == "rv" else {"alpha": 0.0, "beta": 0.0}
        f = fitter(grid, data, **kw)
        g = refit_global(f, GLOBAL["feh"], GLOBAL["dm"], GLOBAL)
        out["global"] = g
        print(which, "global", g["mode"], flush=True)
        save(f"mech_{which}", out)
        del f
        sh = {k: g["mode"][k] for k in ("feh", "dm", "Av")}
        for w in ("ms", "pms"):
            out[w] = prof(fitter(grid, data, WINDOWS[w], **kw), sh)
            print(which, w, round(out[w]["summary"]["loga_hat"], 3), flush=True)
            save(f"mech_{which}", out)
    elif which == "bg":
        fw = fitter(grid, data, WINDOWS["ms"])
        out["ms"] = prof(fw, {**SHARED_A, "f_bg": 1e-6}, free=("sigma_int",))
        print("bg ms", round(out["ms"]["summary"]["loga_hat"], 3), flush=True)
        save("mech_bg", out)


def make_grid(src_name, bc_mode="table"):
    from transplant import feiden_shift, spots_shift, transplant

    base = mist(feh=FEH3)
    if src_name == "identity":
        return transplant(base, None, name="identity")
    if src_name == "feiden":
        return transplant(base, feiden_shift(), name="Feiden16 mag", bc_mode=bc_mode)
    two = src_name.endswith("_2T")
    f = int(src_name.replace("spots", "").replace("_2T", "")) / 100
    return transplant(base, spots_shift(f), two_temp=two, name=src_name, bc_mode=bc_mode)


def task_e3(src_name, arm, windows, bc_mode):
    data = real_data()
    t0 = time.time()
    grid, stats = make_grid(src_name, bc_mode)
    tag = f"e3_{src_name}_{arm}" + ("" if bc_mode == "table" else f"_{bc_mode}")
    path = OUT / f"{tag}.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    out.update({"src": src_name, "arm": arm, "bc_mode": bc_mode, "grid": grid.name, "stats": stats})
    if arm == "a":
        sh = SHARED_A
    elif arm == "s":
        sh = SHARED_S
    elif arm == "m":
        sh = SHARED_M
    elif arm == "sm":
        sh = SHARED_SM
    elif arm == "g":
        if "global" not in out:
            f = fitter(grid, data)
            out["global"] = refit_global(f, GLOBAL["feh"], GLOBAL["dm"], GLOBAL)
            save(tag, out)
            del f
            print(tag, "global", out["global"]["mode"], flush=True)
        sh = {k: out["global"]["mode"][k] for k in ("feh", "dm", "Av")}
    out["shared"] = sh
    for w in windows:
        if w in out:
            continue
        out[w] = prof(fitter(grid, data, WINDOWS[w]), sh)
        print(tag, w, round(out[w]["summary"]["loga_hat"], 3), round(time.time() - t0), flush=True)
        save(tag, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["where", "where_null", "mech", "e3"])
    ap.add_argument("--reps", default="0-7")
    ap.add_argument("--which", default="rv")
    ap.add_argument("--src", default="spots34")
    ap.add_argument("--arm", default="a")
    ap.add_argument("--windows", default="pms,ms")
    ap.add_argument("--bc", default="table")
    a = ap.parse_args()
    if a.task == "where":
        task_where()
    elif a.task == "where_null":
        r0, r1 = map(int, a.reps.split("-"))
        task_where_null(list(range(r0, r1 + 1)))
    elif a.task == "mech":
        task_mech(a.which)
    else:
        task_e3(a.src, a.arm, a.windows.split(","), a.bc)


if __name__ == "__main__":
    main()
