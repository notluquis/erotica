#!/usr/bin/env python3
r"""POST-HOC (after ``explore.py``): on ALL 254 members with the parallax likelihood (row 0 + plx),
is the old / solar solution that appears when G > 18.98 is cut a local maximum, how far below the
young one is it, and which stars pay the difference (per-star log L by G and by BP)?

Run: ``python explore_modes.py`` -> ``explore_modes.json``.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from screen import (  # noqa: E402
    FACTORS,
    bounds,
    compile_logpost,
    make_fitter,
    maximise,
    real_table,
    to_u,
)

t = real_table()
cfg = dict({k: -1 for k in FACTORS}, plx=1)
f = make_fitter(cfg, t)
fn = compile_logpost(f, cfg)
bnd = bounds(f, cfg)
ex = json.loads((HERE / "explore.json").read_text())
starts = {"young": ex["G_le_20.0|row0+plx"]["params"], "old": ex["G_le_18.98|row0+plx"]["params"]}
out = {}
ll = {}
for k, p in starts.items():
    # the old solution is evaluated with (Z, loga, dm, A_V) held at its G <= 18.98 values and only
    # sigma_int, f_bg re-optimised: freed, the optimiser slides off it (it is not a local maximum
    # of the 254-star posterior; measured, first run of this script)
    x0 = np.array(to_u(p), float)
    fix = {i: x0[i] for i in range(4)} if k == "old" else None
    v, x, _ = maximise(fn, x0, bnd, fixed=fix, maxiter=400)
    met, loga, dm, Av, s, fb = (
        10 ** x[0],
        x[1],
        x[2],
        x[3],
        float(np.exp(x[4])),
        float(1 / (1 + np.exp(-x[5]))),
    )
    ll[k] = np.asarray(f.star_ll(met, loga, dm, Av, s, fb, 0.0, np))
    out[k] = dict(
        logpost=v,
        met=met,
        mh=float(np.log10(met / 0.0142)),
        loga=loga,
        dm=dm,
        Av=Av,
        sigma_int=s,
        f_bg=fb,
        loglike=float(ll[k].sum()),
    )
G = np.asarray(t["Gmag"], float)
bp = np.asarray(t["G_BPmag"], float)
d = ll["old"] - ll["young"]
out["delta_logpost_old_minus_young"] = out["old"]["logpost"] - out["young"]["logpost"]
out["delta_loglike_by_G"] = {
    f"{lo}-{hi}": [float(d[(G >= lo) & (G < hi)].sum()), int(((G >= lo) & (G < hi)).sum())]
    for lo, hi in ((8, 12), (12, 14), (14, 16), (16, 18), (18, 18.98), (18.98, 19.5), (19.5, 21))
}
out["delta_loglike_bp_gt_20.3"] = [float(d[bp > 20.3].sum()), int((bp > 20.3).sum())]
out["five_largest_star_deltas"] = sorted(
    [(float(d[i]), float(G[i]), float(bp[i] - np.asarray(t["G_RPmag"])[i])) for i in range(len(d))]
)[:5]
(HERE / "explore_modes.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))

# decomposition: log F (the completeness normalisation, the same for every star) against the
# density term, at both solutions, with f_bg = 0
lim = f._mag_lim
dec = {}
for k in ("young", "old"):
    o = out[k]
    a = np.asarray(f.star_ll(o["met"], o["loga"], o["dm"], o["Av"], o["sigma_int"], 0.0, 0.0, np))
    f._mag_lim = 99.0
    b = np.asarray(f.star_ll(o["met"], o["loga"], o["dm"], o["Av"], o["sigma_int"], 0.0, 0.0, np))
    f._mag_lim = lim
    dec[k] = dict(
        logF=float(np.median(b - a)), logF_spread=float(np.ptp(b - a)), sum_log_dens=float(b.sum())
    )
dec["N_times_dlogF_old_minus_young"] = len(G) * (dec["old"]["logF"] - dec["young"]["logF"])
out["decomposition_fbg0"] = dec
(HERE / "explore_modes.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(dec, indent=1))

# which stars the uniform field component absorbs at each solution: responsibility of the field
resp = {}
for k in ("young", "old"):
    o = out[k]
    full = np.asarray(
        f.star_ll(o["met"], o["loga"], o["dm"], o["Av"], o["sigma_int"], o["f_bg"], 0.0, np)
    )
    pf = (o["f_bg"] / f._box_area) / np.exp(full)
    resp[k] = {
        f"{lo}-{hi}": [
            round(float(pf[(G >= lo) & (G < hi)].sum()), 2),
            int(((G >= lo) & (G < hi)).sum()),
        ]
        for lo, hi in ((8, 12), (12, 14), (14, 16), (16, 18), (18, 18.98), (18.98, 21))
    }
    resp[k]["total_expected_field"] = round(float(pf.sum()), 2)
out["field_responsibility_by_G"] = resp
(HERE / "explore_modes.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(resp, indent=1))
