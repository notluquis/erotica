#!/usr/bin/env python3
"""Experiment 1 of the hub finding ``isochrone-colour-mechanism.md`` (pre-registered there, §0.2):
does a colour-T_eff relation different from MIST's, fitted with MIST, produce a low [Fe/H] and a
young age on a solar-metallicity, 2.5 Myr cluster?

Generator: the grid layer's own forward model (``IsochroneFitter._draw_stars``: Chabrier IMF,
Offner binaries, the real sample's error model and completeness cut), N = 254, on MIST v1.2 with the
BP magnitude moved so that BP-RP changes by Delta(T_eff) at fixed T_eff (``common.shifted``; G and RP
untouched). Truth: [Fe/H] 0, log t 6.40, A_V 1.0, sigma_int 0.03, dm ~ N(10.223, 0.026) per replicate.
Fit: unmodified MIST v1.2, ``common.search`` (coarse lattice + polish), priors ``common.PRI_PLX``.

Arms: ``control`` (Delta = 0), ``lori`` (the lambda Ori offset measured by ``lori_bprp_offset.py``),
``bhac15`` and ``spots0`` (those grids' BP-RP minus MIST's at fixed T_eff, log t 6.4).

WHAT WOULD FALSIFY IT (pre-registered): ``lori`` median Delta[Fe/H] > -0.10 or Delta log t > -0.05
kills the mechanism in this form; ``control`` off the truth by > 0.10 in either makes the test
uninterpretable.

Output: ``colour_synth_<arm>.json`` (one entry per replicate, rewritten after each).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CACHE, EDR3, PRI_PLX, SAMPLE, mist, search, shifted  # noqa: E402

TRUTH = {"feh": 0.0, "loga": 6.40, "Av": 1.0, "sigma_int": 0.03, "dm_mu": 10.223, "dm_sd": 0.026}
N_STARS = 254
LT_HOT = np.log10(6300.0)


def dcol_lori():
    d = json.loads((HERE / "lori_bprp_offset.json").read_text())["nodes"]["6.5"]["bins"]
    x = np.array([b["logte_mid"] for b in d])
    y = np.array([b["median"] for b in d])

    def f(lt):
        lt = np.asarray(lt, float)
        v = np.interp(lt, x, y)  # constant below the coolest bin
        taper = np.clip(1 - (lt - x[-1]) / (LT_HOT - x[-1]), 0, 1)
        return np.where(lt > x[-1], y[-1] * taper, v)

    return f, {"x": x.tolist(), "y": y.tolist(), "taper_to_logte": LT_HOT}


def dcol_grid(name):
    from erotica.analysis import grids as G

    if name == "bhac15":
        g, b = G.BHAC15Grid(CACHE / "bhac15", loga_range=(5.6, 7.5)), ("G", "G_BP", "G_RP")
    else:
        g, b = G.SPOTSGrid(CACHE / "spots", 0.0, loga_range=(5.9, 7.1)), ("G_mag", "BP_mag", "RP_mag")
    a = float(g.loga_nodes[np.argmin(np.abs(g.loga_nodes - TRUTH["loga"]))])
    n = g.node(float(g.feh_nodes[0]), a)
    o = np.argsort(n.extra["logte"])
    lt_alt, c_alt = n.extra["logte"][o], (n.mags[b[1]] - n.mags[b[2]])[o]
    m = mist(feh=[0.0], loga_range=(6.3, 6.5)).node(0.0, 6.4)
    om = np.argsort(m.extra["logte"])
    lt_m, c_m = m.extra["logte"][om], (m.mags[EDR3[1]] - m.mags[EDR3[2]])[om]
    xs = np.linspace(max(lt_alt.min(), lt_m.min()), min(lt_alt.max(), lt_m.max()), 200)
    ys = np.interp(xs, lt_alt, c_alt) - np.interp(xs, lt_m, c_m)

    def f(lt):
        lt = np.asarray(lt, float)
        v = np.interp(lt, xs, ys)
        taper = np.clip(1 - (lt - xs[-1]) / 0.05, 0, 1)
        return np.where(lt > xs[-1], ys[-1] * taper, v)

    return f, {"alt_node_loga": a, "logte_range": [float(xs[0]), float(xs[-1])],
               "delta_at": {str(t): float(f(np.log10(t))) for t in (3200, 3500, 4000, 4500, 5000)}}


def main() -> None:
    from astropy.table import QTable, Table

    import erotica
    from erotica.analysis._isochrone import IsochroneFitter

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["control", "lori", "bhac15", "spots0"])
    ap.add_argument("--reps", type=int, default=6)
    a = ap.parse_args()

    if a.arm == "control":
        dcol, info = (lambda lt: 0.0 * np.asarray(lt, float)), {}
    elif a.arm == "lori":
        dcol, info = dcol_lori()
    else:
        dcol, info = dcol_grid(a.arm)
    base = mist()
    truth_grid = shifted(base, dcol, a.arm)
    real = QTable(Table.read(SAMPLE))
    gen = IsochroneFitter(grid=truth_grid, magnitude=EDR3[0], color=EDR3[1:], **PRI_PLX)
    gen.setup(real, prob_threshold=0.0)
    out = {"erotica_file": erotica.__file__, "arm": a.arm, "truth": TRUTH, "delta": info,
           "priors": {k: list(v) if isinstance(v, tuple) else v for k, v in PRI_PLX.items()},
           "reps": []}
    path = HERE / f"colour_synth_{a.arm}.json"
    for r in range(a.reps):
        rng = np.random.default_rng(1000 + r)  # same seeds in every arm: paired replicates
        dm = float(rng.normal(TRUTH["dm_mu"], TRUTH["dm_sd"]))
        G, C = gen._draw_stars(TRUTH["feh"], TRUTH["loga"], dm, TRUTH["Av"], TRUTH["sigma_int"],
                               N_STARS, rng)
        syn = QTable({"Gmag": G, "G_BPmag": C, "G_RPmag": np.zeros_like(G),
                      "e_Gmag": gen._e_mag_fn(G), "e_G_BPmag": gen._e_col_fn(G),
                      "e_G_RPmag": np.zeros_like(G), "e_BP_RP": gen._e_col_fn(G),
                      "probability_hdbscan": np.ones_like(G)})
        f = IsochroneFitter(grid=base, magnitude=EDR3[0], color=EDR3[1:], **PRI_PLX)
        f.setup(syn, prob_threshold=0.0)
        res = search(f, verbose=False)
        res["truth_dm"] = dm
        res["d_feh"] = res["mode"]["feh"] - TRUTH["feh"]
        res["d_loga"] = res["mode"]["loga"] - TRUTH["loga"]
        res["d_Av"] = res["mode"]["Av"] - TRUTH["Av"]
        out["reps"].append(res)
        print(a.arm, r, json.dumps({k: res[k] for k in ("d_feh", "d_loga", "d_Av", "at_bound",
                                                          "seconds")}, default=float), flush=True)
        path.write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
