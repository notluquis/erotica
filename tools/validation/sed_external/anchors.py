"""A_V of members with a spectroscopic type (T_eff restricted to the type's EEM range +- one subtype,
or Aidelman+2018 / Rauw+2010 T_eff ranges), per star with its posterior width, MIST and EEM, and the
hierarchical level over them. Declared in the finding as an addition to the pre-registration: §0.5
pre-registered the hierarchical level over 'usable' stars, of which there are none."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data, sedgrid as sg
from run_real import spt_teff
from compare import hier

t, mag, err, have, flags = data.load()
sid = np.asarray(t["source_id"], np.int64)
anc = spt_teff()
grids = {"mist": sg.mist_fine(0.0), "eem": sg.eem_fine(0.0)[:2]}
rows, marg = [], {}
for config in ("a", "b"):
    for model, (logt, bc) in grids.items():
        use_all = data.mask(have, config); s = data.sigma(err)
        for i, sd in enumerate(sid):
            if sd not in anc or use_all[i].sum() < 2:
                continue
            lo, hi, lab = anc[sd]
            if model == "eem" and hi > 10 ** logt[-1]:
                continue
            sel = (logt >= np.log10(lo)) & (logt <= np.log10(hi))
            if sel.sum() == 0:
                sel = np.zeros_like(sel); sel[np.argmin(np.abs(logt - np.log10(0.5 * (lo + hi))))] = True
            p, c2 = sg.fit_star(bc[sel], mag[i], s[i], use_all[i])
            sm = sg.summarise(p, logt[sel])
            marg.setdefault(f"{model}_{config}", {})[int(sd)] = p.sum(0)
            rows.append(dict(source_id=int(sd), label=lab, config=config, model=model, G=float(mag[i, 0]),
                             tlo=lo, thi=hi, av16=round(sm["av16"], 3), av50=round(sm["av50"], 3),
                             av84=round(sm["av84"], 3), sd_av=round(sm["sd_av"], 3), chi2=round(c2, 2),
                             nb=int(use_all[i].sum()), irx=bool(flags["irx_flag"][i])))
out = {"rows": rows}
def clean(lab):  # dwarfs/subgiants with a type, no peculiar ('p'), no giant/bright giant, no emission
    sp = lab.split(":", 1)[1]
    return not any(x in sp for x in ("III", "II", "p", "e", "em"))
for key, m in marg.items():
    ids = list(m)
    labs = {r["source_id"]: r["label"] for r in rows}
    out[f"hier_{key}_all"] = hier([m[k] for k in ids], sg.AV)
    cl = [k for k in ids if clean(labs[k])]
    out[f"hier_{key}_clean"] = hier([m[k] for k in cl], sg.AV) if len(cl) > 2 else None
(HERE / "anchors.json").write_text(json.dumps(out, indent=1))
for r in rows:
    if r["model"] == "mist":
        print(f"{r['config']} {r['source_id']} {r['label']:24s} G={r['G']:5.2f} AV={r['av50']:.2f} [{r['av16']:.2f},{r['av84']:.2f}] sd={r['sd_av']:.3f} chi2={r['chi2']:.1f}/{r['nb']}")
for r in rows:
    if r["model"] == "eem":
        print(f"EEM {r['config']} {r['source_id']} {r['label']:24s} AV={r['av50']:.2f} sd={r['sd_av']:.3f}")
for k, v in out.items():
    if k.startswith("hier"):
        print(k, v)
