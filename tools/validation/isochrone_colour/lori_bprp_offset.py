#!/usr/bin/env python3
"""Observed, dereddened Gaia BP-RP of the lambda Ori stars (Cao+22) minus MIST v1.2 BP-RP at the
star's spectroscopic T_eff -- the colour-T_eff offset of D3 (``isochrone_grids/lambda_ori_colour_check.py``,
measured there in J-Ks), now in the bands of NGC 6383. It is the size of the colour offset injected by
``colour_synth.py`` in its ``lori`` arm.

Process and limits: A_V is Cao's, built from PARSEC colours on Pecaut & Mamajek tables (D3's caveat:
not independent of a colour table). CCM89 coefficients at the band effective wavelengths of the
fitter (the same law NGC 6383 is fitted with). Class III only (no disc excess), RUWE < 1.4.
Output ``lori_bprp_offset.json``: binned median in log T_eff with bootstrap SE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CACHE, EDR3, mist  # noqa: E402

BINS = np.log10([3000, 3300, 3600, 3900, 4200, 4600, 5200, 6500])


def main() -> None:
    from astropy.table import Table

    from erotica.analysis._isochrone import _ccm89

    t = Table.read(CACHE / "oracles/lambda_ori/lambda_ori_2mass_dr3.ecsv")
    f = lambda c: np.asarray(c.filled(np.nan) if hasattr(c, "filled") else c, float)  # noqa: E731
    lt, av = f(t["logTeff"]), f(t["Av"])
    bp, rp, ruwe = f(t["BPmag"]), f(t["RPmag"]), f(t["RUWE"])
    ok = (np.asarray(t["Class"]) == "III") & np.isfinite(lt + av + bp + rp) & (ruwe < 1.4)
    kbp, krp = _ccm89(5182.6), _ccm89(7825.1)
    c0 = (bp - rp - (kbp - krp) * av)[ok]
    lt = lt[ok]
    out = {"n": int(ok.sum()), "k_bp_minus_k_rp": kbp - krp, "nodes": {}}
    g = mist(feh=[0.0], loga_range=(6.0, 7.0))
    rng = np.random.default_rng(0)
    for a in (6.3, 6.5):
        n = g.node(0.0, a)
        o = np.argsort(n.extra["logte"])
        cm = np.interp(lt, n.extra["logte"][o], (n.mags[EDR3[1]] - n.mags[EDR3[2]])[o])
        r = c0 - cm
        rows = []
        for lo, hi in zip(BINS[:-1], BINS[1:], strict=True):
            s = (lt >= lo) & (lt < hi)
            if s.sum() < 5:
                continue
            boots = [np.median(rng.choice(r[s], s.sum())) for _ in range(2000)]
            rows.append({"logte_mid": float(np.median(lt[s])), "n": int(s.sum()),
                         "median": float(np.median(r[s])), "se_boot": float(np.std(boots))})
        out["nodes"][str(a)] = {"all_median": float(np.median(r)), "bins": rows}
        print(a, json.dumps(out["nodes"][str(a)]), flush=True)
    (HERE / "lori_bprp_offset.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
