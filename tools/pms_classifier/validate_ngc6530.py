"""Validacion externa en NGC 6530 (Wright+2019: miembros jovenes por GES y rayos X) y ablacion de NIR. Prerregistro §0.4, §0.6.

Ningun cumulo a < 2 deg de NGC 6530 entro al entrenamiento (fetch_training.py). Para Sagitta, NGC 6530 es IN-SAMPLE (Laguna en la
Table 1 de McBride+2021): el control lo favorece.
Env: erotica. Corre: SAGITTA_SRC=... OMP_NUM_THREADS=4 nice -n 19 ~/miniforge3/envs/erotica/bin/python validate_ngc6530.py \
     > validate_ngc6530_measured.txt
QUE MIDE (fijado antes de correr):
 [6] recall de los dos clasificadores sobre los miembros de Wright con DR3, y sobre los que la etiqueta llama PMS
     (M_G,0 > M_G,on a la edad HR23 de NGC 6530); en el dominio de McBride aparte. Diferencia pareada propio(0,5) - Sagitta(0,6)
     con bootstrap de estrellas B = 2000. F1: falla si el IC 95 % de la diferencia queda entero bajo -0,05.
 [7] ablacion de NIR (F3): a cada estrella de NGC 6530 se le copia el patron de faltantes J/H/Ks de un miembro de NGC 6383 al azar
     del mismo tramo de G (1 mag; entrada 2MASS A-E de la tabla limpia), 50 sorteos; fraccion que cambia de clase. Falla si > 0,15.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table
from xgboost import XGBClassifier

import core
import sagitta_ref as S
from train_validate import cp, f

C = Path(__file__).parent / "cache"
RNG = np.random.default_rng(6530)

cl = Table.read(C / "hr23_clusters_sel.ecsv")
r = cl[cl["Name"] == "NGC_6530"][0]
MOD, AV, LOGT = float(r["MOD50"]), float(r["AV50"]), float(r["logAge50"])
g = Table.read(C / "ngc6530_dr3.ecsv")
w = Table.read(C / "wright2019.ecsv")
print(f"NGC 6530 (HR23): MOD50 {MOD:.3f}, AV50 {AV:.3f}, logAge50 {LOGT:.3f}, M_G,on {core.mg_turn_on(LOGT):.3f}")
print(f"Wright+2019: {len(w)} filas; con DR3 por X-Match (<= 0,5'', |dG| < 0,3) {len(g)}")
from astropy.coordinates import SkyCoord
import astropy.units as u
GAL = SkyCoord(f(g["RAdeg"]) * u.deg, f(g["DEdeg"]) * u.deg).galactic

q = np.array([str(x) if (x is not None and not np.ma.is_masked(x) and len(str(x)) == 3) else "XXX" for x in g["ph_qual"]])
J = core.nir_clean(f(g["j_m"]), [s[0] for s in q])
H = core.nir_clean(f(g["h_m"]), [s[1] for s in q])
K = core.nir_clean(f(g["ks_m"]), [s[2] for s in q])
G, BP, RP, plx = f(g["Gmag"]), f(g["BPmag"]), f(g["RPmag"]), f(g["Plx"])
ok = np.isfinite(G) & np.isfinite(BP) & np.isfinite(RP)
print(f"con G, BP, RP: {ok.sum()}; con J/H/Ks validas (A-E): {np.isfinite(J).sum()}/{np.isfinite(H).sum()}/{np.isfinite(K).sum()}")
X = core.features(G, BP, RP, J, H, K, MOD, AV)
ydef = core.label(G, MOD, AV, LOGT)
with np.errstate(invalid="ignore"):
    exc, bprp = f(g["E(BP/RP)"]), BP - RP
    dom = ((f(g["RUWE"]) < 1.4) & (exc > 1 + 0.015 * bprp ** 2) & (exc < 1.3 + 0.06 * bprp ** 2)
           & (f(g["RFG"]) > 10) & (f(g["RFBP"]) > 10)
           & (f(g["RFRP"]) > 10) & (plx > 0.2) & ((plx / f(g["e_Plx"]) > 10) | (f(g["e_Plx"]) < 0.1)))

models = {}
for name in ("all", "young"):
    mm = XGBClassifier()
    mm.load_model(C / f"model_{name}.json")
    models[name] = mm
p = {k: mm.predict_proba(X)[:, 1] for k, mm in models.items()}
ps, _ = S.sagitta_pms(GAL.l.deg, GAL.b.deg, plx, G, BP, RP, J, H, K)


def rec(sel, pr, th):
    k, n = int(np.sum(pr[sel] >= th)), int(sel.sum())
    lo, hi = cp(k, n)
    return f"{k}/{n} = {k / max(n, 1):.3f} [{lo:.3f}; {hi:.3f}]"


print("[6] recall (fraccion clasificada PMS) en miembros jovenes confirmados de Wright+2019")
for tag, sel in (("todos con G, BP, RP", ok), ("etiqueta PMS (M_G,0 > M_G,on)", ok & ydef),
                 ("etiqueta PMS, dominio McBride", ok & ydef & dom), ("etiqueta PMS, fuera del dominio", ok & ydef & ~dom),
                 ("etiqueta PMS, con J H Ks", ok & ydef & np.isfinite(J) & np.isfinite(H) & np.isfinite(K)),
                 ("etiqueta PMS, sin NIR", ok & ydef & ~np.isfinite(J) & ~np.isfinite(H) & ~np.isfinite(K))):
    print(f"  {tag:34s} propio(0,5) {rec(sel, p['all'], .5)} | propio-joven(0,5) {rec(sel, p['young'], .5)} | "
          f"Sagitta(0,6) {rec(sel, ps, .6)} | Sagitta(0,5) {rec(sel, ps, .5)}")
print(f"  etiqueta: {np.sum(ok & ~ydef)} de {ok.sum()} miembros de Wright quedan SOBRE el turn-on (no PMS por la etiqueta)")
sel = np.where(ok & ydef)[0]
a, b = (p["all"][sel] >= .5).astype(float), (ps[sel] >= .6).astype(float)
d = [np.mean(a[s] - b[s]) for s in (RNG.integers(0, len(sel), len(sel)) for _ in range(2000))]
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"  diferencia pareada propio(0,5) - Sagitta(0,6), etiqueta PMS: {np.mean(a - b):+.3f} [{lo:+.3f}; {hi:+.3f}]  "
      f"-> F1 {'FALLA' if hi < -0.05 else 'no falla'}")
print(f"  Sigma P (propio) sobre etiqueta PMS: {p['all'][sel].sum():.1f} de {len(sel)}")

# [7] ablacion de NIR con el patron de NGC 6383
t83 = Table.read(Path.home() / "phd/agent-findings/scripts/virac2_ngc6383_members_clean.ecsv")
q83 = np.array([str(x) if (x is not None and not np.ma.is_masked(x) and len(str(x)) == 3) else "XXX" for x in t83["tmass_ph_qual"]])
miss83 = np.stack([~np.isfinite(core.nir_clean(f(t83[f"{bd}_2mass"]), [s[i] for s in q83]))
                   for i, bd in enumerate(("J", "H", "Ks"))], 1)
G83 = f(t83["G"])
print(f"[7] patron de NGC 6383 (2MASS A-E de la tabla limpia): falta J/H/Ks {miss83.mean(0).round(3)}; falta todo "
      f"{np.all(miss83, 1).mean():.3f}")
gb = np.clip(np.floor(G), 8, 20)
gb83 = np.clip(np.floor(G83), 8, 20)
base = p["all"] >= .5
fl, fl_pms = [], []
for _ in range(50):
    Jm, Hm, Km = J.copy(), H.copy(), K.copy()
    for i in np.where(ok)[0]:
        pool = np.where(gb83 == gb[i])[0]
        if not len(pool):
            pool = np.argsort(np.abs(G83 - G[i]))[:5]
        mk = miss83[RNG.choice(pool)]
        if mk[0]:
            Jm[i] = np.nan
        if mk[1]:
            Hm[i] = np.nan
        if mk[2]:
            Km[i] = np.nan
    pa = models["all"].predict_proba(core.features(G, BP, RP, Jm, Hm, Km, MOD, AV))[:, 1] >= .5
    fl.append(np.mean(pa[ok] != base[ok]))
    fl_pms.append(np.mean(pa[ok & ydef]) - np.mean(base[ok & ydef]))
print(f"  propio: fraccion que cambia de clase {np.mean(fl):.3f} (sd {np.std(fl):.3f}; max {np.max(fl):.3f}) -> F3 "
      f"{'FALLA' if np.mean(fl) > 0.15 else 'no falla'}; cambio del recall en etiqueta PMS {np.mean(fl_pms):+.3f}")
