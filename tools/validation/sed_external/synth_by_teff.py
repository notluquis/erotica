"""Post hoc breakdown of the synthetic arms by TRUE T_eff (declared post hoc in the finding §2):
the pre-registered bins were in G, but the truth T_eff is drawn independently of the star's G, so
G bins mix the hot regime (where broad-band colours are degenerate with A_V) with the cool one."""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
z = np.load(HERE / "synth_perstar.npz")
cols = list(z["columns"])
TB = [(3.52, 3.65), (3.65, 3.75), (3.75, 3.85), (3.85, 4.0), (4.0, 4.31)]
out = {}
for key in [k for k in z.files if k != "columns"]:
    a = z[key]
    c = {n: a[:, cols.index(n)] for n in cols}
    o = {}
    for lo, hi in TB:
        s = (c["lt"] >= lo) & (c["lt"] < hi)
        if s.sum() < 5:
            continue
        dav = c["av50"][s] - c["av"][s]
        dlt = c["logt50"][s] - c["lt"][s]
        o[f"T{10**lo:.0f}-{10**hi:.0f}"] = dict(
            n=int(s.sum()), bias_av=round(float(np.median(dav)), 3),
            p16_84_dav=[round(float(x), 3) for x in np.percentile(dav, [16, 84])],
            bias_logt=round(float(np.median(dlt)), 4),
            cov_av=round(float(np.mean((c["av16"][s] <= c["av"][s]) & (c["av"][s] <= c["av84"][s]))), 3),
            cov_logt=round(float(np.mean((c["logt16"][s] <= c["lt"][s]) & (c["lt"][s] <= c["logt84"][s]))), 3),
            med_sd_av=round(float(np.median(c["sd_av"][s])), 3),
            med_sd_logt=round(float(np.median(c["sd_logt"][s])), 4),
            frac_sd_av_le_015=round(float(np.mean(c["sd_av"][s] <= 0.15)), 3))
    out[key] = o
(HERE / "synth_by_teff.json").write_text(json.dumps(out, indent=1))
for k, o in out.items():
    print(k)
    for b, v in o.items():
        print(f"   {b:12s} n={v['n']:4d} biasAV={v['bias_av']:+.3f} [{v['p16_84_dav'][0]:+.2f},{v['p16_84_dav'][1]:+.2f}] "
              f"biaslogT={v['bias_logt']:+.4f} covAV={v['cov_av']:.2f} covT={v['cov_logt']:.2f} "
              f"sdAV={v['med_sd_av']:.3f} sdlogT={v['med_sd_logt']:.4f} f(sdAV<=.15)={v['frac_sd_av_le_015']:.2f}")
