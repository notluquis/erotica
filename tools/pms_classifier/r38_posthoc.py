"""R38, POST HOC (despues de ver r38_oos_eval y r38_generalize; nada de esto estaba prerregistrado ni cambia las reglas de §4):
 (P1) guardia "realmente fuera de muestra": source_id del pool contra el entrenamiento completo y contra los 254 de P01 (debe dar 0;
      con uno metido a proposito, 1).
 (P2) FPR del modelo final por tramo en el pool a la diffAV50 de NGC 6383 (2,0): diffAV50 en [1,5; 2,5] y >= 2, con cota bootstrap.
 (P3) prueba de dos planos (BP-RP y G-Ks) sobre los FP y no FP del pool con diffAV50 >= 1: ¿el corrimiento es extincion (dK ~ dB)?
      Control: un desplazamiento VERTICAL de binaria (-0,75 mag, q = 1) aplicado a la MS de MIST da (dB, dK) que no coinciden.
Env: erotica. Corre: cd ~/erotica-wt-pms/tools/pms_classifier && OMP_NUM_THREADS=2 ~/miniforge3/envs/erotica/bin/python r38_posthoc.py \
     > r38_posthoc_measured.txt 2> cache/r38_posthoc.log
"""
import json
from pathlib import Path

import numpy as np
from astropy.table import Table
from scipy.stats import spearmanr
from xgboost import XGBClassifier

import core
import exp2_common as E
import r38_common as R
from r38_oos_data import load_oos
from train_validate import f

C = Path("cache")
th = np.array(json.load(open(C / "exp2_selected.json"))["th"])
o = load_oos()
sid_pool = np.asarray(o["m"]["GaiaDR3"], np.int64)
sid_tr = np.asarray(Table.read(C / "hr23_members.ecsv")["GaiaDR3"], np.int64)
sid_p01 = np.asarray(Table.read(Path.home() / "phd/agent-findings/scripts/virac2_ngc6383_members_clean.ecsv")["GaiaDR3"], np.int64)
print("[P1] pool contra entrenamiento completo (hr23_members, train + val):", np.isin(sid_pool, sid_tr).sum(),
      "| contra los 254 de P01:", np.isin(sid_pool, sid_p01).sum(),
      "| guardia con uno metido a proposito:", np.isin(np.append(sid_pool, sid_tr[0]), sid_tr).sum(),
      "| cumulos del pool en el entrenamiento:", np.isin(np.unique(o["grp"]), np.unique(np.asarray(Table.read(C / "hr23_members.ecsv")["Name"]).astype(str))).sum())

mdl = XGBClassifier()
mdl.load_model(C / "exp2_model_T0P1K0F0D1.json")
dec = E.decide(mdl.predict_proba(o["X"])[:, 1], o["mg0"], th)
old, grp, k = o["old"], o["grp"], E.tramo_index(o["mg0"])


def bup(pr, g, R_=2000, seed=1):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(g, return_inverse=True)
    n = np.bincount(inv).astype(float); x = np.bincount(inv, weights=pr.astype(float))
    return np.percentile([x[i].sum() / n[i].sum() for i in (rng.integers(0, len(ug), len(ug)) for _ in range(R_))], 97.5)


print("[P2] FPR del modelo final en el pool a la diffAV50 de NGC 6383 (2,0)")
for nm, cs in (("diffAV50 1,5-2,5", (o["dav"] >= 1.5) & (o["dav"] <= 2.5)), ("diffAV50 >= 2", o["dav"] >= 2.0)):
    s = old & cs
    print(f"  {nm}: cumulos {len(np.unique(grp[s]))}, global {dec[s].sum()}/{s.sum()} = {dec[s].mean():.4f} cota {bup(dec[s], grp[s]):.4f}")
    for i, t in enumerate(E.TRAMO_NAMES):
        si = s & (k == i)
        if si.sum():
            print(f"    {t}: {dec[si].sum()}/{si.sum()} = {dec[si].mean():.4f} cota boot {bup(dec[si], grp[si]):.4f} (cumulos {len(np.unique(grp[si]))})")

print("[P3] dos planos sobre el pool con diffAV50 >= 1 (M_G,0 +1..+7; isocrona, MOD y AV50 de HR23 de cada cumulo)")
m = o["m"]
G, BP, RP = f(m["Gmag"]), f(m["BPmag"]), f(m["RPmag"])
K = o["ph"]["K"]
sel = old & (o["dav"] >= 1.0) & (o["mg0"] >= 1) & (o["mg0"] < 7)
dB, dK = np.full(len(G), np.nan), np.full(len(G), np.nan)
for c in np.unique(grp[sel]):
    s = sel & (grp == c)
    mod, av, lt = np.median(o["ph"]["mod"][s]), np.median(o["av"][s]), np.median(o["logt"][s])
    dB[s] = R.delta_av((BP - RP)[s], G[s], mod, av, "BPRP", lt)
    dK[s] = R.delta_av((G - K)[s], K[s], mod, av, "GK", lt)
for nm, msk in (("FP", sel & dec), ("no FP", sel & ~dec)):
    b = msk & np.isfinite(dB) & np.isfinite(dK)
    r = spearmanr(dB[b], dK[b])
    print(f"  {nm:6s} n {msk.sum():5d}, con las dos raices {b.sum():5d}: mediana dB {np.median(dB[b]):+.2f} dK {np.median(dK[b]):+.2f}; "
          f"Spearman {r.statistic:.3f}; mediana dK - dB {np.median(dK[b] - dB[b]):+.2f} IQR [{np.percentile(dK[b] - dB[b], 25):+.2f}; "
          f"{np.percentile(dK[b] - dB[b], 75):+.2f}]; |dK - dB| <= 0,3 en {np.mean(np.abs(dK[b] - dB[b]) <= 0.3):.2f}")
# control de binarias: MS de MIST log t 9,0, puntos con M_G 3..7, 0,75 mag mas brillantes, sin enrojecer (MOD 10, A_V 1)
mi = core._read_mist()
z = (np.abs(mi["log10_isochrone_age_yr"] - 9.0) < 1e-6) & (mi["phase"] == 0) & (mi["Gaia_G_EDR3"] > 3) & (mi["Gaia_G_EDR3"] < 7)
ext = core.EXT
for tag, dv, sh in (("binaria q=1 (-0,75 mag vertical)", 0.0, -0.75), ("enrojecida +1,0 A_V", 1.0, 0.0)):
    Gs = mi["Gaia_G_EDR3"][z] + 10 + ext["G"] * (1 + dv) + sh
    BPs = mi["Gaia_BP_EDR3"][z] + 10 + ext["BP"] * (1 + dv) + sh
    RPs = mi["Gaia_RP_EDR3"][z] + 10 + ext["RP"] * (1 + dv) + sh
    Ks = mi["2MASS_Ks"][z] + 10 + ext["Ks"] * (1 + dv) + sh
    b_ = R.delta_av(BPs - RPs, Gs, 10.0, 1.0, "BPRP", 9.0)
    k_ = R.delta_av(Gs - Ks, Ks, 10.0, 1.0, "GK", 9.0)
    ok = np.isfinite(b_) & np.isfinite(k_)
    print(f"  CONTROL {tag}: n {ok.sum()}, mediana dB {np.median(b_[ok]):+.2f} dK {np.median(k_[ok]):+.2f}; mediana dK - dB "
          f"{np.median(k_[ok] - b_[ok]):+.2f}; |dK - dB| <= 0,3 en {np.mean(np.abs(k_[ok] - b_[ok]) <= 0.3):.2f}; max dB {np.max(b_[ok]):+.2f}")
    if dv == 0.0:
        bmax = np.max(b_[ok])
print(f"  la prueba de dos planos NO separa binaria de enrojecimiento (los dos controles coinciden); lo que separa es el TAMANO: FP con dB > "
      f"{bmax:.2f} (el maximo de una binaria q = 1): " + ", ".join(
          f"{nm} {np.mean(dB[m_ & np.isfinite(dB)] > bmax):.2f}" for nm, m_ in (("FP", sel & dec), ("no FP", sel & ~dec))))
