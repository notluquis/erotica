#!/usr/bin/env python3
"""D3 (post-unblinding) of ``lambda_ori_oracle.py``: observed, dereddened J-Ks of the Cao+22 Class III
stars minus each grid's J-Ks at the star's published log T_eff, on the node nearest log t 6.5.

A grid-dependent offset here is a colour-T_eff difference that the CMD fit has to absorb in age, dm
or A_V -- the candidate mechanism for the primary failing in (J, J-Ks) while passing in the HR
diagram (D2). Output: ``lambda_ori_oracle_D3_colour.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
    import lambda_ori_oracle as lo
    from astropy.table import Table

    q, _ = lo.sample()
    grids = lo.grids()
    t = Table.read(lo.DATA)
    cls = np.asarray(t["Class"])
    has = ~np.asarray(getattr(t["logAge"], "mask", np.zeros(len(t), bool)))
    t = t[(cls == "III") & has]
    log_teff = np.asarray(t["logTeff"], float)
    col = np.asarray(q["J0"] - q["K0"], float)
    out = {}
    for name, (g, b) in grids.items():
        a = float(g.loga_nodes[np.argmin(np.abs(g.loga_nodes - 6.5))])
        n = g.node(0.0, a)
        o = np.argsort(n.extra["logte"])
        lt, cc = n.extra["logte"][o], (n.mags[b[1]] - n.mags[b[2]])[o]
        inside = (log_teff >= lt.min()) & (log_teff <= lt.max())
        r = col[inside] - np.interp(log_teff[inside], lt, cc)
        out[name] = {
            "node_loga": a,
            "n": int(inside.sum()),
            "median_obs_minus_model_JK": float(np.median(r)),
            "p16": float(np.percentile(r, 16)),
            "p84": float(np.percentile(r, 84)),
        }
        print(name, out[name], flush=True)
    (HERE / "lambda_ori_oracle_D3_colour.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
