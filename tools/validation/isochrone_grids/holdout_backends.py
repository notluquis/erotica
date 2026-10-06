#!/usr/bin/env python3
"""Held-out-node validation of every grid backend, and of the pseudo-EEP schemes against MIST.

WHY THIS EXISTS
---------------
Each backend of the grid layer is interpolated between its nodes by the fitter. Whether that
interpolation is right is a property of the grid, separate from whether NUTS converges
(methodology §A.1.2): a coarse emulator in this programme passed every convergence check and was
biased 22 sigma. So before any backend fits a cluster, this measures, for every interior age node
in log t 6.0-7.0 at [Fe/H] = 0 (and for [Fe/H] nodes at log t 6.5), the error of the isochrone
interpolated from the neighbours at TWICE the native step, in G and colour (mass-matched and
orthogonal) and as the log-age offset a fit would infer from interpolation alone
(``erotica.analysis.grids.holdout``).

It also decides "reuse vs write" for the PARSEC pseudo-EEP: the three candidate parametrisations
(this package's arc-length one, ezpadova's index-within-label, ASteCA's mass quantiles) are applied
to MIST v1.2 -- whose native EEPs are the reference -- and scored against the SAME native isochrones.

WHAT WOULD FALSIFY THE CONCLUSIONS
----------------------------------
* "arc-length pseudo-EEP is adequate": its MIST hold-out error exceeding the native-EEP one by more
  than the native error itself (i.e. > 2x), or its inferred log-age offset exceeding 0.02 dex where
  native's does not.
* "a backend is usable at NGC 6383's ages": a median inferred |dlog t| > 0.05 dex (the size of the
  E1 falsifier in the hub landscape) at 2x step.

Output: ``holdout_backends.json`` (full per-node rows). Inputs are local caches; a missing grid is
recorded as missing, not skipped silently.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))
NGC = Path.home() / "erotica/data/test/NGC6383"
CACHE = Path.home() / ".cache/erotica-grids"
AGES = (6.0, 7.0)


def main() -> None:
    import erotica
    from erotica.analysis import grids as G

    res: dict = {"erotica_file": erotica.__file__, "ages": AGES, "runs": {}, "missing": []}
    t0 = time.time()

    def run(key, grid, bands, window, truth=None, feh_nodes=()):
        r = G.holdout_grid(
            grid,
            bands,
            feh=0.0 if 0.0 in grid.feh_nodes else None,
            loga_range=AGES,
            mass_window=window,
            truth=truth,
        )
        r["sigma_floor"] = G.recommended_sigma_floor(r)
        r["feh_holdout"] = [G.holdout_feh_node(grid, f, 6.5, bands, window) for f in feh_nodes]
        res["runs"][key] = r
        m = r["median_over_nodes"]
        print(
            f"{key:28s} n={r['n_nodes']:2d} step={r['native_step_median']:.3f} "
            f"dG68={m['dG_p68']:.4f} dc68={m['dcol_p68']:.4f} orth68={m['orth_p68']:.4f} "
            f"|dloga| med={r['dloga_inferred']['median_abs']:.4f} max={r['dloga_inferred']['max_abs']:.4f}"
            f"  [{time.time() - t0:.0f}s]",
            flush=True,
        )

    mb = ("Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3")
    p12 = NGC / "MIST/UBVRIplus"
    if p12.exists():
        m12 = G.MISTGrid(p12, loga_range=(5.9, 7.1))
        sub = m12.select(feh=[-0.5, -0.25, 0.0, 0.25, 0.5])
        run("MIST v1.2 native", sub, mb, (0.1, 7.0), feh_nodes=(-0.25, 0.25))
        for scheme in ("arclength", "index", "massquantile"):
            run(f"MIST v1.2 {scheme}", G.regrid(sub, scheme), mb, (0.1, 7.0), truth=sub)
    else:
        res["missing"].append(str(p12))
    p25 = CACHE / "mist_v2.5/UBVRIplus_afe0_vvcrit0.4"
    if p25.exists():
        m25 = G.MISTGrid(p25, loga_range=(5.9, 7.1)).select(feh=[-0.5, -0.25, 0.0, 0.25, 0.5])
        run("MIST v2.5 native", m25, mb, (0.1, 7.0), feh_nodes=(-0.25, 0.25))
    else:
        res["missing"].append(str(p25))
    pp = NGC / "PARSEC/gaiaedr3"
    if pp.exists():
        pb = ("Gmag", "G_BPmag", "G_RPmag")
        raw = G.PARSECGrid(pp, scheme="raw", loga_range=(5.9, 7.1), max_mass=20.0)
        for scheme in ("arclength", "index", "massquantile"):
            g = G.PARSECGrid(pp, scheme=scheme, loga_range=(5.9, 7.1), max_mass=20.0)
            feh0 = float(g.feh_nodes[0])
            r = G.holdout_grid(g, pb, feh=feh0, loga_range=AGES, mass_window=(0.1, 7.0), truth=raw)
            r["sigma_floor"] = G.recommended_sigma_floor(r)
            r["feh_holdout"] = [G.holdout_feh_node(g, float(g.feh_nodes[1]), 6.5, pb, (0.1, 7.0))]
            res["runs"][f"PARSEC v1.2S {scheme}"] = r
            m = r["median_over_nodes"]
            print(
                f"PARSEC {scheme:21s} n={r['n_nodes']:2d} orth68={m['orth_p68']:.4f} "
                f"dG68={m['dG_p68']:.4f} |dloga| med={r['dloga_inferred']['median_abs']:.4f}",
                flush=True,
            )
    else:
        res["missing"].append(str(pp))
    pb15 = CACHE / "bhac15"
    if pb15.exists():
        run("BHAC15", G.BHAC15Grid(pb15, loga_range=(5.6, 7.5)), ("G", "G_BP", "G_RP"), (0.1, 1.4))
    else:
        res["missing"].append(str(pb15))
    ps = CACHE / "spots"
    if ps.exists():
        for f in (0.0, 0.17, 0.34):
            run(
                f"SPOTS f={f}",
                G.SPOTSGrid(ps, f, loga_range=(5.9, 7.1)),
                ("G_mag", "BP_mag", "RP_mag"),
                (0.1, 1.3),
            )
    else:
        res["missing"].append(str(ps))
    res["seconds"] = round(time.time() - t0, 1)
    (HERE / "holdout_backends.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
