"""Recall con IC (pedido de la tarea, no estaba en exp2_confirm): (a) en la celda tipo NGC 6383 (log t < 6,8; 700-1600 pc; A_V 0,6-2,0),
OOF, contra la etiqueta de §0.2, global y por tramo, con bootstrap por cumulo; elegida (T0P1K0F0D1), modelo del F2 y Sagitta (0,6, U ->
token, de oof.npz); (b) en NGC 6530 por tramo con Clopper-Pearson, modelo final elegido.
Corre: SAGITTA_SRC=~/.local/lib/sagitta-1.2.3 ~/miniforge3/envs/erotica/bin/python exp2_recall_cell.py > exp2_recall_cell_measured.txt
"""
import json
from pathlib import Path

import numpy as np
from astropy.table import Table
from xgboost import XGBClassifier

import core
import exp2_common as E
import exp2_grid as GR
from exp2_confirm import feats
from train_validate import cp, f

C = Path("cache")
RNG = np.random.default_rng(611)
d = GR.prepare()
o = np.load(C / "oof.npz", allow_pickle=True)
oofs = np.load(C / "exp2_oof.npz")
sel = json.load(open(C / "exp2_selected.json"))
th = np.array(sel["th"])
av = f(d["m"]["AV50"])
cell = (d["logt"] < 6.8) & (d["dist"] >= 700) & (d["dist"] < 1600) & (av >= 0.6) & (av < 2.0) & d["y0"]
k = E.tramo_index(d["mg0"])
arms = {"elegida T0P1K0F0D1": E.decide(oofs["T0P1K0F0"].astype(float), d["mg0"], th),
        "modelo del F2 (0,5)": oofs["T0P0K0F0"] >= 0.5, "Sagitta (0,6) U->token": o["sag"] >= 0.6}


def boot(pr, g, R=1000):
    ug, inv = np.unique(g, return_inverse=True)
    n = np.bincount(inv).astype(float)
    x = np.bincount(inv, weights=pr.astype(float))
    v = [x[i].sum() / n[i].sum() for i in (RNG.integers(0, len(ug), len(ug)) for _ in range(R))]
    return np.percentile(v, [2.5, 97.5])


print(f"[a] celda tipo NGC 6383, etiqueta 1: {cell.sum()} estrellas en {len(np.unique(d['grp'][cell]))} cumulos (recall contra la etiqueta propia)")
for nm, pr in arms.items():
    lo, hi = boot(pr[cell], d["grp"][cell])
    row = [f"  {nm:24s} global {pr[cell].mean():.3f} [{lo:.3f}; {hi:.3f}]"]
    for i, t in enumerate(E.TRAMO_NAMES):
        s = cell & (k == i)
        if s.sum() >= 10 and len(np.unique(d["grp"][s])) >= 3:
            l2, h2 = boot(pr[s], d["grp"][s])
            row.append(f"{t} {pr[s].mean():.3f} [{l2:.3f}; {h2:.3f}] (n {s.sum()})")
    print(" | ".join(row))

cl = Table.read(C / "hr23_clusters_sel.ecsv")
r = cl[cl["Name"] == "NGC_6530"][0]
MOD, AV, LOGT = float(r["MOD50"]), float(r["AV50"]), float(r["logAge50"])
g = Table.read(C / "ngc6530_dr3.ecsv")
q6 = np.array([str(x) if (x is not None and not np.ma.is_masked(x) and len(str(x)) == 3) else "XXX" for x in g["ph_qual"]])
J, H, K = (core.nir_clean(f(g[c]), [s[i] for s in q6]) for i, c in enumerate(("j_m", "h_m", "ks_m")))
G, BP, RP = f(g["Gmag"]), f(g["BPmag"]), f(g["RPmag"])
ok = np.isfinite(G) & np.isfinite(BP) & np.isfinite(RP)
ydef = core.label(G, MOD, AV, LOGT)
mdl = XGBClassifier()
mdl.load_model(C / f"exp2_model_{sel['chosen']}.json")
X = feats(sel["cfg"], G, BP, RP, J, H, K, MOD, AV)
dec = E.decide(mdl.predict_proba(X)[:, 1], X[:, 0], th)
k6 = E.tramo_index(X[:, 0])
print("[b] NGC 6530 (Wright+2019, etiqueta PMS; muestra preseleccionada en el CMD, R37-05), elegida, Clopper-Pearson 95 %")
for i, t in enumerate(E.TRAMO_NAMES):
    s = ok & ydef & (k6 == i)
    x, n = int(dec[s].sum()), int(s.sum())
    print(f"  {t:8s} {x}/{n} = {x / max(n, 1):.3f} {np.round(cp(x, n), 3).tolist()}")
