#!/usr/bin/env python3
r"""2D profile maps of the C1 model on NGC 6383, with and without prior, and the modes they show.

WHY THIS EXISTS
---------------
Task 2 of the hub finding ``isochrone-model-screen.md`` (§0.5): is the posterior of C1 multimodal --
is there a mode at solar Z and older age? C1's NUTS chains started from one search mode, so their
corner plot cannot answer it. Profiles over pairs (loga-dm, loga-met, met-A_V, loga-A_V), the other
parameters maximised, on a grid that passes C1's bounds (loga from 5.5, Z 0.002-0.04, A_V 0-3).

WHAT WOULD MAKE "UNIMODAL" AN ARTEFACT
--------------------------------------
A grid swept by continuation from one start follows one branch. Each point is therefore the best of
TWO families: one sweep warm-started from the young / low-Z corner, the other, in reverse order,
from the old / solar corner. Modes = local maxima of the map, then polished in all six parameters;
their "volume" is the Laplace evidence in the optimiser's coordinates (an approximation, said so).

Run: ``OMP_NUM_THREADS=1 NPROC=1 nice -n 19 python maps.py [--n 10]`` -> ``maps.json``, resumable.
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
from screen import (  # noqa: E402
    FACTORS,
    compile_logpost,
    laplace,
    make_fitter,
    maximise,
    real_table,
    to_u,
)

PAIRS = {"loga-dm": (1, 2), "loga-met": (1, 0), "met-Av": (0, 3), "loga-Av": (1, 3)}
RANGES = {0: (np.log10(0.002), np.log10(0.04)), 1: (5.5, 7.0), 2: (9.5, 10.7), 3: (0.0, 3.0)}
YOUNG = dict(met=0.005, loga=6.0, dm=10.25, Av=0.6, sigma_int=0.15, f_bg=0.15)
OLD = dict(met=0.017, loga=6.6, dm=10.25, Av=1.3, sigma_int=0.10, f_bg=0.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    cfg = {k: -1 for k in FACTORS}
    cfg["loga_lo"] = 1
    f = make_fitter(cfg, real_table())
    bnd = [
        RANGES[0],
        RANGES[1],
        RANGES[2],
        RANGES[3],
        (np.log(2e-3), np.log(0.5)),
        (-8.0, 0.0),
        (0.0, 0.0),
    ]
    path = HERE / "maps.json"
    out = json.loads(path.read_text()) if path.exists() else {"n": a.n, "maps": {}, "modes": {}}
    for prior in (True, False):
        fn = compile_logpost(f, cfg, with_prior=prior)
        tag = "post" if prior else "like"
        for pname, (i, j) in PAIRS.items():
            key = f"{pname}|{tag}"
            if key in out["maps"]:
                continue
            t0 = time.time()
            xi = np.linspace(*RANGES[i], a.n)
            xj = np.linspace(*RANGES[j], a.n)
            order = [(p, q if p % 2 == 0 else a.n - 1 - q) for p in range(a.n) for q in range(a.n)]
            val = np.full((2, a.n, a.n), -np.inf)
            par = np.zeros((2, a.n, a.n, 7))
            for fam, (start, seq) in enumerate(((YOUNG, order), (OLD, order[::-1]))):
                x = np.array(to_u(start), float)
                x[6] = 0.0
                for p, q in seq:
                    x[i], x[j] = xi[p], xj[q]
                    v, x, _ = maximise(fn, x, bnd, fixed={i: xi[p], j: xj[q]}, maxiter=150)
                    val[fam, p, q] = v
                    par[fam, p, q] = x
            best = val.max(0)
            which = val.argmax(0)
            out["maps"][key] = dict(
                axes=[int(i), int(j)],
                xi=xi.tolist(),
                xj=xj.tolist(),
                value=best.tolist(),
                family_gap=(val[0] - val[1]).tolist(),
                family=which.tolist(),
                params=np.take_along_axis(par, which[None, ..., None], 0)[0].tolist(),
                seconds=round(time.time() - t0, 1),
            )
            path.write_text(json.dumps(out) + "\n")
            print(key, round(float(best.max()), 2), out["maps"][key]["seconds"], flush=True)
        # modes: local maxima of every map of this prior, polished in 6D
        if tag in out["modes"]:
            continue
        cands = []
        for key, m in out["maps"].items():
            if not key.endswith(tag):
                continue
            V = np.array(m["value"])
            P = np.array(m["params"])
            for p in range(a.n):
                for q in range(a.n):
                    nb = V[max(p - 1, 0) : p + 2, max(q - 1, 0) : q + 2]
                    if V[p, q] >= nb.max() and np.isfinite(V[p, q]):
                        cands.append((key, p, q, P[p, q]))
        modes = []
        for key, p, q, x0 in cands:
            v, x, _ = maximise(fn, x0, bnd, maxiter=400)
            if any(np.abs(np.array(mm["u"])[:4] - x[:4]).max() < 0.02 for mm in modes):
                continue
            lap, ok = laplace(fn, x, bnd)
            logZ = v + 0.5 * (lap["logdet_cov"] + lap["n_free"] * np.log(2 * np.pi)) if ok else None
            at_b = {
                n: bool(min(abs(x[k] - bnd[k][0]), abs(x[k] - bnd[k][1])) < 1e-3)
                for k, n in enumerate(["lz", "loga", "dm", "Av"])
            }
            modes.append(
                dict(
                    from_map=key,
                    cell=[p, q],
                    value=v,
                    u=x.tolist(),
                    Z=10 ** x[0],
                    laplace_ok=ok,
                    log_volume_laplace=logZ,
                    at_bound=at_b,
                )
            )
        modes.sort(key=lambda m: -m["value"])
        out["modes"][tag] = modes
        path.write_text(json.dumps(out) + "\n")
        print(
            tag,
            json.dumps(
                [
                    (
                        round(m["value"], 2),
                        [round(c, 3) for c in m["u"][:4]],
                        m["log_volume_laplace"],
                    )
                    for m in modes
                ]
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
