#!/usr/bin/env python3
r"""Summary of the screen: main effects on the real-data MAP, and per configuration and truth the
bias, RMSE and coverage on synthetic clusters; the pre-registered gate and ranking (hub finding
``isochrone-model-screen.md`` §0.3-§0.4). Main effects on synthetics are computed **per replicate**
(every configuration fits the same synthetic data set, so the contrast is paired) and their SE is
the spread over replicates / sqrt(R).

Run: ``python summarize.py`` -> ``summary.json`` (seconds, no fitting).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FACTORS = ["zprior", "diffred", "loga_lo", "plx", "imf", "ext", "cstar"]
Z_SUN = 0.0142


def key(cfg):
    return tuple(cfg[f] for f in FACTORS)


def main():
    real = json.loads((HERE / "real.json").read_text())
    out = {"real": {}, "synth": {}}
    rows = real["configs"]
    tab = []
    for r in rows:
        p = r["params"]
        tab.append(
            dict(
                run=r["run"],
                cfg=key(r["cfg"]),
                loga=p["loga"],
                mh=p["mh"],
                dm=p["dm"],
                Av=p["Av"],
                sigma_int=p["sigma_int"],
                sigma_Av=p["sigma_Av"],
                logpost=r["logpost"],
                prof=r["loga_profile68"],
                at_bound=r["at_bound"],
            )
        )
    out["real"]["table"] = sorted(tab, key=lambda t: t["run"])
    me = {}
    for i, fct in enumerate(FACTORS):
        hi = [t for t in tab if t["cfg"][i] > 0]
        lo = [t for t in tab if t["cfg"][i] < 0]
        me[fct] = {
            q: float(np.mean([t[q] for t in hi]) - np.mean([t[q] for t in lo]))
            for q in ("loga", "mh", "dm", "Av")
        }
    out["real"]["main_effects"] = me
    # synthetic
    truths = {}
    for p in sorted(HERE.glob("synth_T*_*of*.json")):
        d = json.loads(p.read_text())
        tname = p.name.split("_")[1]
        truths.setdefault(tname, {"truth": d["truth"], "reps": []})["reps"] += d["reps"]
    for tname, d in truths.items():
        tr = d["truth"]
        mh_t = np.log10(tr["met"] / Z_SUN)
        by = {}
        for r in d["reps"]:
            p = r["params"]
            lo, hi = r["loga_profile68"]
            by.setdefault(key(r["cfg"]), {})[r["rep"]] = dict(
                loga=p["loga"] - tr["loga"],
                dm=p["dm"] - tr["dm"],
                mh=p["mh"] - mh_t,
                cover=float(lo <= tr["loga"] <= hi),
                bound=float(r["at_bound"]["loga"]),
                dlp_truth=r["logpost"] - r["logpost_at_truth_other_at_map"],
            )
        per = {}
        for k, reps in by.items():
            R = len(reps)
            v = {
                q: np.array([reps[j][q] for j in reps])
                for q in ("loga", "dm", "mh", "cover", "bound", "dlp_truth")
            }
            per[str(k)] = dict(
                R=R,
                **{f"bias_{q}": float(v[q].mean()) for q in ("loga", "dm", "mh")},
                **{
                    f"se_{q}": float(v[q].std(ddof=1) / np.sqrt(R)) if R > 1 else None
                    for q in ("loga", "dm", "mh")
                },
                rmse_loga=float(np.sqrt((v["loga"] ** 2).mean())),
                cover_loga=int(v["cover"].sum()),
                frac_loga_at_bound=float(v["bound"].mean()),
                median_dlp_map_minus_truth=float(np.median(v["dlp_truth"])),
            )
        # paired main effects per replicate
        reps_all = sorted({j for reps in by.values() for j in reps})
        mes = {}
        for i, fct in enumerate(FACTORS):
            for q in ("loga", "dm", "mh"):
                vals = []
                for j in reps_all:
                    hi = [by[k][j][q] for k in by if k[i] > 0 and j in by[k]]
                    lo = [by[k][j][q] for k in by if k[i] < 0 and j in by[k]]
                    if len(hi) == 8 and len(lo) == 8:
                        vals.append(np.mean(np.abs(hi)) - np.mean(np.abs(lo)))
                vals = np.array(vals)
                mes.setdefault(fct, {})[f"d_absbias_{q}"] = [
                    float(vals.mean()),
                    float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None,
                    len(vals),
                ]
            for lev, name in ((1, "hi"), (-1, "lo")):
                c = [by[k][j]["cover"] for k in by if k[i] == lev for j in by[k]]
                mes[fct][f"cover_loga_{name}"] = [int(np.sum(c)), len(c)]
        out["synth"][tname] = dict(truth=tr, per_config=per, main_effects=mes)
    # gate and ranking
    if len(truths) == 2:
        gate = {}
        for k in out["synth"]["T1"]["per_config"]:
            a, b = out["synth"]["T1"]["per_config"][k], out["synth"]["T2"]["per_config"].get(k)
            if b is None:
                continue
            ok = (
                abs(a["bias_loga"]) <= 0.05
                and abs(b["bias_loga"]) <= 0.05
                and abs(a["bias_dm"]) <= 0.05
                and abs(b["bias_dm"]) <= 0.05
                and a["cover_loga"] + b["cover_loga"] >= 7
            )
            gate[k] = dict(
                passes=bool(ok),
                rmse_mean=0.5 * (a["rmse_loga"] + b["rmse_loga"]),
                cover=a["cover_loga"] + b["cover_loga"],
                n=a["R"] + b["R"],
            )
        out["gate"] = dict(
            sorted(gate.items(), key=lambda kv: (not kv[1]["passes"], kv[1]["rmse_mean"]))
        )
    (HERE / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out["real"]["main_effects"], indent=1))


if __name__ == "__main__":
    main()
