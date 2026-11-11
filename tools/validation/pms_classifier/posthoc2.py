"""POST HOC (escrito despues de ver [0]-[10] y [P1]-[P3]; nada de esto estaba prerregistrado). Revision de la lectura:

 [Q1] incertidumbre TOTAL del censo propio de NGC 6383: B = 30 modelos reentrenados remuestreando cumulos de entrenamiento con
      reemplazo; en cada uno, una realizacion de la calibracion CAL (parametrica + fotometrica) y A_V, MOD ~ N(HR23 50, (84-16)/2).
      Componentes por separado: solo modelo; solo calibracion (ya en apply: [200; 203]); solo A_V/MOD.
 [Q2] punto de operacion: umbral propio t* que iguala el FPR de Sagitta(0,6) en los cumulos viejos fuera de particion; a t*, recall
      en NGC 6530 y censo de NGC 6383. (La ventaja de F1 se midio a FPR distintos: 0,097 contra 0,034.)
 [Q3] de donde sale el FPR en viejos: por Prob de HR23 (>= 0,95 contra 0,7-0,95: contaminacion de membresia) y con model_young,
      que nunca vio logAge >= 8,3 (prior de edades).
 [Q4] recall de NGC 6530 por tramo de M_G,0 y reponderado a la distribucion de M_G,0 de los 241 PMS-por-definicion de NGC 6383
      (§K.1.41 P3: la muestra GES no tiene la distribucion de masas de los 254).
 [Q5] Sagitta en su convencion (2MASS `U` tal como viene listada, no -> token): FPR en viejos, celda de NGC 6383, NGC 6530.
Env: erotica. Corre: SAGITTA_SRC=... OMP_NUM_THREADS=4 nice -n 19 ~/miniforge3/envs/erotica/bin/python posthoc2.py > posthoc2_measured.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table
from xgboost import XGBClassifier

import apply_ngc6383 as A
import core
import sagitta_ref as S
from train_validate import build, cp, f, load

C = Path(__file__).parent / "cache"
RNG = np.random.default_rng(41)
o = np.load(C / "oof.npz", allow_pickle=True)
m = load()
m = m[m["role"] == "train"]
X, y, ph = build(m)
assert np.all(np.asarray(m["GaiaDR3"], np.int64) == o["sid"])
grp = o["grp"].astype(str)
MG0 = X[:, core.FEATURES.index("MG0")]
old = (o["logt"] >= 8.5) & (o["dist"] >= 600) & (o["dist"] <= 2500) & ~o["y"]
cell = (o["logt"] < 6.8) & (o["dist"] >= 700) & (o["dist"] < 1600) & (o["av"] >= 0.6) & (o["av"] < 2.0)
h = Table.read(C / "ngc6383_hr23.ecsv")[0]
sAV, sMOD = (float(h["AV84"]) - float(h["AV16"])) / 2, (float(h["MOD84"]) - float(h["MOD16"])) / 2


def census(model, nir, av=A.AV, mod=A.MOD, th=0.5):
    X83 = core.features(A.G, A.BP, A.RP, nir["J"], nir["H"], nir["Ks"], mod, av)
    return int(np.sum(model.predict_proba(X83)[:, 1] >= th))


# ------------------------------------------------------------------ [Q2] punto de operacion
fpr_sag = np.mean(o["sag"][old] >= .6)
ts = np.linspace(0.5, 0.99, 491)
fprs = np.array([np.mean(o["oof"][old] >= t) for t in ts])
tstar = float(ts[np.argmin(np.abs(fprs - fpr_sag))])
print(f"[Q2] FPR viejos Sagitta(0,6) {fpr_sag:.4f}; umbral propio que lo iguala t* = {tstar:.3f} (FPR {np.mean(o['oof'][old] >= tstar):.4f})")
yng = cell & o["y"]
print(f"  celda NGC 6383, recall contra la etiqueta: propio(t*) {np.mean(o['oof'][yng] >= tstar):.3f}, Sagitta(0,6) "
      f"{np.mean(o['sag'][yng] >= .6):.3f}")
mall = XGBClassifier()
mall.load_model(C / "model_all.json")
myng = XGBClassifier()
myng.load_model(C / "model_young.json")
g = Table.read(C / "ngc6530_dr3.ecsv")
cl = Table.read(C / "hr23_clusters_sel.ecsv")
r6530 = cl[cl["Name"] == "NGC_6530"][0]
q = np.array([str(x) if (x is not None and not np.ma.is_masked(x) and len(str(x)) == 3) else "XXX" for x in g["ph_qual"]])
Jr, Hr, Kr = f(g["j_m"]), f(g["h_m"]), f(g["ks_m"])
J = core.nir_clean(Jr, [s[0] for s in q])
H = core.nir_clean(Hr, [s[1] for s in q])
K = core.nir_clean(Kr, [s[2] for s in q])
G6, BP6, RP6, plx6 = f(g["Gmag"]), f(g["BPmag"]), f(g["RPmag"]), f(g["Plx"])
ok6 = np.isfinite(G6) & np.isfinite(BP6) & np.isfinite(RP6)
X6 = core.features(G6, BP6, RP6, J, H, K, float(r6530["MOD50"]), float(r6530["AV50"]))
y6 = core.label(G6, float(r6530["MOD50"]), float(r6530["AV50"]), float(r6530["logAge50"])) & ok6
p6 = mall.predict_proba(X6)[:, 1]
from astropy.coordinates import SkyCoord  # noqa: E402
import astropy.units as u  # noqa: E402
gal6 = SkyCoord(f(g["RAdeg"]) * u.deg, f(g["DEdeg"]) * u.deg).galactic
s6, _ = S.sagitta_pms(gal6.l.deg, gal6.b.deg, plx6, G6, BP6, RP6, J, H, K)
s6U, _ = S.sagitta_pms(gal6.l.deg, gal6.b.deg, plx6, G6, BP6, RP6, Jr, Hr, Kr)
k1, k2, n6 = int(np.sum(p6[y6] >= tstar)), int(np.sum(s6[y6] >= .6)), int(y6.sum())
a, b = (p6[y6] >= tstar).astype(float), (s6[y6] >= .6).astype(float)
d = [np.mean(a[i] - b[i]) for i in (RNG.integers(0, n6, n6) for _ in range(2000))]
print(f"  NGC 6530 (etiqueta PMS, n {n6}): propio(t*) {k1 / n6:.3f} {np.round(cp(k1, n6), 3)}; Sagitta(0,6) {k2 / n6:.3f}; "
      f"diferencia {np.mean(a - b):+.3f} [{np.percentile(d, 2.5):+.3f}; {np.percentile(d, 97.5):+.3f}]")
print(f"  NGC 6383 (CAL) a t*: propio {census(mall, A.CAL, th=tstar)} (a 0,5: {census(mall, A.CAL)})")

# ------------------------------------------------------------------ [Q3] origen del FPR en viejos
prob = f(m["Prob"])
print("[Q3] FPR propio(0,5) en viejos, por Prob de HR23 y por tramo de M_G,0")
for lo, hi in ((-5, 5), (5, 7), (7, 12)):
    s = old & (MG0 >= lo) & (MG0 < hi)
    hi_p, lo_p = s & (prob >= 0.95), s & (prob < 0.95)
    print(f"  M_G,0 {lo:+d} a {hi:+d}: Prob >= 0,95 {np.mean(o['oof'][hi_p] >= .5):.3f} (n {hi_p.sum()}) | Prob 0,7-0,95 "
          f"{np.mean(o['oof'][lo_p] >= .5):.3f} (n {lo_p.sum()})")
py_old = myng.predict_proba(X[old])[:, 1]
print(f"  model_young (nunca vio logAge >= 8,3) en viejos: FPR {np.mean(py_old >= .5):.4f}; model_all fuera de particion "
      f"{np.mean(o['oof'][old] >= .5):.4f}")
for lo, hi in ((-5, 5), (5, 7), (7, 12)):
    s = (MG0[old] >= lo) & (MG0[old] < hi)
    print(f"    M_G,0 {lo:+d} a {hi:+d}: model_young {np.mean(py_old[s] >= .5):.3f}")

# ------------------------------------------------------------------ [Q4] NGC 6530 por tramo y reponderado a NGC 6383
mg6 = X6[:, core.FEATURES.index("MG0")]
G83 = A.G
mg83 = G83 - A.MOD - core.EXT["G"] * A.AV
lab83 = core.label(G83, A.MOD, A.AV, float(h["logAge50"]))
bins = ((-5, 3), (3, 5), (5, 7), (7, 12))
print("[Q4] recall en NGC 6530 (etiqueta PMS) por tramo de M_G,0; peso = fraccion de los 241 PMS-por-definicion de NGC 6383")
rw = {"propio(0,5)": 0.0, "propio(t*)": 0.0, "Sagitta(0,6)": 0.0, "Sagitta U listada(0,6)": 0.0}
for lo, hi in bins:
    s = y6 & (mg6 >= lo) & (mg6 < hi)
    w = np.mean((mg83[lab83] >= lo) & (mg83[lab83] < hi))
    vals = {"propio(0,5)": np.mean(p6[s] >= .5), "propio(t*)": np.mean(p6[s] >= tstar), "Sagitta(0,6)": np.mean(s6[s] >= .6),
            "Sagitta U listada(0,6)": np.mean(s6U[s] >= .6)}
    for k in rw:
        rw[k] += w * (vals[k] if s.any() else 0)
    print(f"  M_G,0 {lo:+d} a {hi:+d}: n6530 {s.sum():3d}, peso NGC 6383 {w:.3f}: " + ", ".join(f"{k} {v:.3f}" for k, v in vals.items()))
print("  reponderado a NGC 6383: " + ", ".join(f"{k} {v:.3f}" for k, v in rw.items()))

# ------------------------------------------------------------------ [Q5] Sagitta con U listada
print("[Q5] Sagitta con las magnitudes 2MASS tal como vienen listadas (U incluidas; NaN -> token solo sin contraparte)")
mm = m[old]
pU, _ = S.sagitta_pms(f(mm["GLON"]), f(mm["GLAT"]), f(mm["Plx"]), ph["G"][old], ph["BP"][old], ph["RP"][old],
                      f(mm["j_m"]), f(mm["h_m"]), f(mm["ks_m"]))
k = int(np.sum(pU >= .6))
print(f"  viejos: FPR Sagitta(0,6) U listada {k / old.sum():.4f} {np.round(cp(k, old.sum()), 4)} (U -> token: {fpr_sag:.4f})")
mc = m[yng]
pUc, _ = S.sagitta_pms(f(mc["GLON"]), f(mc["GLAT"]), f(mc["Plx"]), ph["G"][yng], ph["BP"][yng], ph["RP"][yng],
                       f(mc["j_m"]), f(mc["h_m"]), f(mc["ks_m"]))
print(f"  celda NGC 6383 recall Sagitta(0,6) U listada {np.mean(pUc >= .6):.3f} (U -> token {np.mean(o['sag'][yng] >= .6):.3f})")
k = int(np.sum(s6U[y6] >= .6))
print(f"  NGC 6530 recall Sagitta(0,6) U listada {k / n6:.3f} {np.round(cp(k, n6), 3)}; todos con G,BP,RP "
      f"{np.mean(s6U[ok6] >= .6):.3f}")

# ------------------------------------------------------------------ [Q1] incertidumbre total
print("[Q1] censo propio NGC 6383, N(P >= 0,5): B = 30 modelos con cumulos remuestreados")
ug = np.unique(grp)
idx = {c: np.where(grp == c)[0] for c in ug}
tot, mod_only, avmod_only = [], [], []
for bb in range(30):
    pick = RNG.choice(ug, len(ug))
    s = np.concatenate([idx[c] for c in pick])
    mdl = core.fit(X[s], y[s], seed=1000 + bb)
    mod_only.append(census(mdl, A.CAL))
    nir = A.corrected(A.cal_draw(True), True)
    av, mod = A.AV + RNG.normal(0, sAV), A.MOD + RNG.normal(0, sMOD)
    tot.append(census(mdl, nir, av=av, mod=mod))
    avmod_only.append(census(mall, A.CAL, av=av, mod=mod))
for k, v in (("solo modelo", mod_only), ("solo A_V y MOD (sigma de HR23)", avmod_only), ("total", tot)):
    v = np.array(v)
    print(f"  {k:32s} mediana {np.median(v):.0f} [2,5; 97,5 %] = [{np.percentile(v, 2.5):.0f}; {np.percentile(v, 97.5):.0f}] "
          f"sd {v.std():.1f}")
print(f"  (sigma A_V {sAV:.3f}, sigma MOD {sMOD:.3f}; mezcla de edades aparte: model_young {census(myng, A.CAL)} contra "
      f"model_all {census(mall, A.CAL)})")
