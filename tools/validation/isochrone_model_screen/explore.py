#!/usr/bin/env python3
r"""POST-HOC, exploratory arms (not in the pre-registration of ``isochrone-model-screen.md`` §0):
written after reading the per-star residuals (§3.1), which showed the faint-BP stars (BP > 20.3,
the BP flux-bias regime of Riello+2021 §9.2) and the outer stars (r > 20') bluer than the C1
isochrone. Each arm refits two configurations (row 0 = C1 with A_V widened; row 0 + plx) on a
subsample, to see whether [M/H] or loga move. Labelled exploratory: they were chosen by looking.

Run: ``python explore.py`` -> ``explore.json``.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from screen import FACTORS, fit, make_fitter, real_table  # noqa: E402

t = real_table()
bp = np.asarray(t["G_BPmag"], float)
ra, de = np.asarray(t["ra"], float), np.asarray(t["dec"], float)
r = np.hypot((ra - np.median(ra)) * np.cos(np.deg2rad(np.median(de))), de - np.median(de)) * 60
subs = {
    "bp_le_20.3": bp <= 20.3,
    "r_lt_20": r < 20,
    "bp_le_20.3_and_r_lt_20": (bp <= 20.3) & (r < 20),
}
base = {k: -1 for k in FACTORS}
path = HERE / "explore.json"
out = json.loads(path.read_text()) if path.exists() else {}
for name, m in subs.items():
    for cname, cfg in (("row0", base), ("row0+plx", dict(base, plx=1))):
        if f"{name}|{cname}" in out:
            continue
        res = fit(make_fitter(cfg, t[m]), cfg)
        out[f"{name}|{cname}"] = dict(
            N=int(m.sum()),
            params=res["params"],
            prof=res["loga_profile68"],
            at_bound=res["at_bound"],
        )
        path.write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(
            name, cname, int(m.sum()), json.dumps(res["params"]), res["loga_profile68"], flush=True
        )

# Second post-hoc pass, written after the first: dropping BP > 20.3 also moves the completeness
# cut G_lim from 20.66 to 18.98 (no member fainter than 18.98 has BP <= 20.3), so the arm above
# cannot tell "BP-biased colours" from "the faint end". A sequence of G cuts separates them: if Z
# rises only when the 8 red BP > 20.3 stars with G < 18.98 go, it is their colours; if it rises
# smoothly with the G cut, it is the faint end (MIST low-mass PMS colours, or colour-dependent
# incompleteness that the G-only completeness term does not model).
G = np.asarray(t["Gmag"], float)
for gc in (20.0, 19.5, 18.98, 18.5, 18.0):
    m = G <= gc
    cfg = dict(base, plx=1)
    res = fit(make_fitter(cfg, t[m]), cfg)
    out[f"G_le_{gc}|row0+plx"] = dict(
        N=int(m.sum()), params=res["params"], prof=res["loga_profile68"], at_bound=res["at_bound"]
    )
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("G<=", gc, int(m.sum()), json.dumps(res["params"]), res["loga_profile68"], flush=True)
