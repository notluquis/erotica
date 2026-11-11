"""POST HOC (escrito despues de ver exp2_confirm_measured.txt; no cambia ningun veredicto de §2.4). Pregunta: el sintetico de §2.2
dio FPR 0,294 en +7..+12, ¿es el clasificador o el generador? El generador muestrea uniforme en M_G,0, asi que a la MOD y A_V de NGC 6383
pone estrellas hasta G ~ 23, mas debiles que el limite de Gaia y que cualquier miembro real; su ruido de BP/RP crece sin tope ahi.
Chequeo predictivo del generador (skill astro-inference, §A.1.3): el sintetico se recorta al rango de G de los 254 miembros reales.
Corre: SAGITTA_SRC=~/.local/lib/sagitta-1.2.3 ~/miniforge3/envs/erotica/bin/python exp2_posthoc.py > exp2_posthoc_measured.txt
"""
import json
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

import apply_ngc6383 as A
import exp2_common as E
from exp2_confirm import feats, synth

C = Path(__file__).parent / "cache"
cj = json.load(open(C / "exp2_confirm.json"))
cfg, th, key = cj["cfg"], np.array(cj["th"]), cj["key"]
mdl = XGBClassifier()
mdl.load_model(C / f"exp2_model_{key}.json")
mg0 = A.G - A.MOD - 0.789 * A.AV
k83 = E.tramo_index(mg0)
print(f"[H0] G de los 254 miembros: max {np.nanmax(A.G):.2f}, p95 {np.nanpercentile(A.G, 95):.2f}; en +7..+12 (n {np.sum(k83 == 3)}): "
      f"G {np.nanmin(A.G[k83 == 3]):.2f}..{np.nanmax(A.G[k83 == 3]):.2f}, M_G,0 max {np.nanmax(mg0):.2f}")
gmax = float(np.nanmax(A.G))
for kind in ("ms", "pms"):
    for binary in (False, True):
        ob = synth(kind, 4000, A.MOD, A.AV, binary)
        X = feats(cfg, ob["G"], ob["BP"], ob["RP"], ob["J"], ob["H"], ob["Ks"], A.MOD, A.AV)
        dd = E.decide(mdl.predict_proba(X)[:, 1], X[:, 0], th)
        k = E.tramo_index(X[:, 0])
        inr = ob["G"] <= gmax
        lab = "FPR MS 9,0" if kind == "ms" else "recall PMS 6,6"
        print(f"  {lab:14s} {'bin q=1' if binary else 'simples'} todo: " +
              "  ".join(f"{nm} {np.mean(dd[k == i]):.3f}" for i, nm in enumerate(E.TRAMO_NAMES)) +
              f" || G <= {gmax:.2f}: " + "  ".join(
                  f"{nm} {np.mean(dd[(k == i) & inr]):.3f} (n {np.sum((k == i) & inr)})" if np.any((k == i) & inr) else f"{nm} -"
                  for i, nm in enumerate(E.TRAMO_NAMES)))
