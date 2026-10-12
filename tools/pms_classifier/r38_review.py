"""R38, comprobaciones de la revision adversarial (subagente con adversarial-referee, 2026-10-12), re-ejecutadas aca. POST HOC.
 (M2a) ¿diffAV50 de HR23 es el ancho al rojo del CMD de los miembros? Spearman entre el IQR por cumulo de DeltaA_V(BP-RP) y diffAV50,
       en los 115 viejos de A u B (M_G,0 +1..+7, >= 10 estrellas con raiz).
 (M2b) control de binarias con la linea base de cada cumulo: FP del pool (diffAV50 >= 1) con dB - mediana(dB de los no FP del cumulo)
       > 1,03 (maximo de una binaria q = 1, r38_posthoc P3).
Corre: OMP_NUM_THREADS=2 ~/miniforge3/envs/erotica/bin/python r38_review.py > r38_review_measured.txt 2> cache/r38_review.log
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from xgboost import XGBClassifier

import exp2_common as E
import exp2_grid as GR
import r38_common as R
from r38_oos_data import load_oos
from train_validate import f

C = Path("cache")
th = np.array(json.load(open(C / "exp2_selected.json"))["th"])
d = GR.prepare()
m = d["m"]
G, BP, RP = f(m["Gmag"]), f(m["BPmag"]), f(m["RPmag"])
dav = f(m["diffAV50"])
sel = d["old"] & (d["mg0"] >= 1) & (d["mg0"] < 7)
iqr, dv = [], []
for c in np.unique(d["grp"][sel]):
    s = sel & (d["grp"] == c)
    v = R.delta_av((BP - RP)[s], G[s], np.median(f(m["MOD50"])[s]), np.median(f(m["AV50"])[s]), "BPRP", np.median(d["logt"][s]))
    v = v[np.isfinite(v)]
    if len(v) >= 10:
        iqr.append(np.percentile(v, 75) - np.percentile(v, 25)); dv.append(np.median(dav[s]))
iqr, dv = np.array(iqr), np.array(dv)
r = spearmanr(iqr, dv)
print(f"[M2a] {len(iqr)} viejos: Spearman IQR(DeltaA_V) vs diffAV50 {r.statistic:+.3f} p {r.pvalue:.1e}; IQR mediano por diffAV50: " +
      "  ".join(f"{lo}-{hi}: {np.median(iqr[(dv >= lo) & (dv < hi)]):.2f} (n {np.sum((dv >= lo) & (dv < hi))})"
                for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 9)) if np.sum((dv >= lo) & (dv < hi))))
o = load_oos()
mdl = XGBClassifier()
mdl.load_model(C / "exp2_model_T0P1K0F0D1.json")
dec = E.decide(mdl.predict_proba(o["X"])[:, 1], o["mg0"], th)
mo = o["m"]
Go, BPo, RPo = f(mo["Gmag"]), f(mo["BPmag"]), f(mo["RPmag"])
s0 = o["old"] & (o["dav"] >= 1.0) & (o["mg0"] >= 1) & (o["mg0"] < 7)
rel = np.full(len(Go), np.nan)
for c in np.unique(o["grp"][s0]):
    s = s0 & (o["grp"] == c)
    v = R.delta_av((BPo - RPo)[s], Go[s], np.median(o["ph"]["mod"][s]), np.median(o["av"][s]), "BPRP", np.median(o["logt"][s]))
    base = np.nanmedian(v[~dec[s]]) if np.any(np.isfinite(v[~dec[s]])) else np.nan
    rel[s] = v - base
for nm, mk in (("FP", s0 & dec), ("no FP", s0 & ~dec)):
    v = rel[mk & np.isfinite(rel)]
    print(f"[M2b] {nm}: n {len(v)}; dB - linea base del cumulo > 1,03 en {np.mean(v > 1.03):.2f}")
