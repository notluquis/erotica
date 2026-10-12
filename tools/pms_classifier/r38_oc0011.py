"""R38-02, pruebas (a)-(d) del prerregistro (hub agent-findings/pms-classifier-non-circular.md §4.1) sobre las 320 estrellas de
OC_0011 en B. Requiere cache/r38_fold*.json (r38_folds.py, con su control) y el cubo de Vergely+2022 en cache/vergely/.
Env: erotica. Corre: cd ~/erotica-wt-pms/tools/pms_classifier && OMP_NUM_THREADS=2 ~/miniforge3/envs/erotica/bin/python r38_oc0011.py \
     > r38_oc0011_measured.txt
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from xgboost import XGBClassifier

import core
import exp2_common as E
import exp2_grid as GR
import r38_common as R
from train_validate import f

C = Path("cache")
d = GR.prepare()
th = np.array(json.load(open(C / "exp2_selected.json"))["th"])
oof = np.load(C / "exp2_oof.npz")["T0P1K0F0"].astype(float)
dec = E.decide(oof, d["mg0"], th)
m = d["m"]
s = d["grp"] == "OC_0011"
fp = dec[s]
G, BP, RP = f(m["Gmag"])[s], f(m["BPmag"])[s], f(m["RPmag"])[s]
q = np.array([str(x) if (x is not None and not np.ma.is_masked(x)) else "XXX" for x in m["ph_qual"][s]])
q = np.array([z if len(z) == 3 else "XXX" for z in q])
K = core.nir_clean(f(m["ks_m"])[s], [z[2] for z in q])
mod, av, lt = (float(np.median(f(m[c])[s])) for c in ("MOD50", "AV50", "logAge50"))
mg0 = d["mg0"][s]
print(f"OC_0011 en B: {s.sum()} estrellas, FP {fp.sum()}; HR23 MOD {mod:.3f} AV50 {av:.3f} log t {lt:.3f}; Ks valida en {np.isfinite(K).sum()}")

print("=== (a) DeltaA_V por estrella hasta la MS de MIST, dos planos (estrellas con M_G,0 HR23 en +1..+7)")
sel = (mg0 >= 1) & (mg0 < 7)
dB = R.delta_av(BP - RP, G, mod, av, "BPRP", lt)
dK = R.delta_av(G - K, K, mod, av, "GK", lt)
for nm, msk in (("todas", sel), ("FP", sel & fp), ("no FP", sel & ~fp)):
    v = dB[msk]
    v = v[np.isfinite(v)]
    print(f"  {nm:6s} n {msk.sum():3d}, con raiz BP-RP {len(v):3d}: mediana {np.median(v):+.2f} IQR [{np.percentile(v, 25):+.2f}; "
          f"{np.percentile(v, 75):+.2f}] (ancho {np.percentile(v, 75) - np.percentile(v, 25):.2f})")
both = sel & np.isfinite(dB) & np.isfinite(dK)
rho = spearmanr(dB[both], dK[both])
print(f"  dos planos, n {both.sum()}: Spearman {rho.statistic:.3f} (p {rho.pvalue:.1e}); mediana de dK - dB {np.median(dK[both] - dB[both]):+.2f} "
      f"IQR [{np.percentile(dK[both] - dB[both], 25):+.2f}; {np.percentile(dK[both] - dB[both], 75):+.2f}]")
for nm, msk in (("FP", both & fp), ("no FP", both & ~fp)):
    print(f"    {nm}: n {msk.sum()}, mediana dB {np.median(dB[msk]):+.2f}, dK {np.median(dK[msk]):+.2f}")
w = np.isfinite(dB) & sel
mw = mannwhitneyu(dB[w & fp], dB[w & ~fp])
print(f"  FP - no FP en dB: diferencia de medianas {np.median(dB[w & fp]) - np.median(dB[w & ~fp]):+.2f}; Mann-Whitney p {mw.pvalue:.1e}")
for kG in (0.759, 0.819):
    v = R.delta_av(BP - RP, G, mod, av, "BPRP", lt, kG=kG)[sel]
    v = v[np.isfinite(v)]
    print(f"  sensibilidad A_G/A_V = {kG}: mediana {np.median(v):+.2f} IQR ancho {np.percentile(v, 75) - np.percentile(v, 25):.2f}")
ib = np.percentile(dB[w], 75) - np.percentile(dB[w], 25)
print(f"  LECTURA (regla §4.1 a): planos coinciden: {rho.statistic >= 0.5 and abs(np.median(dK[both] - dB[both])) <= 0.3}; "
      f"IQR {ib:.2f} -> {'H1' if ib < 0.5 else ('H2 si FP > no FP por > 0,5' if ib >= 1.0 else 'mixto')}")

print("=== (b) astrometria FP contra no FP")
for c in ("Plx", "pmRA", "pmDE", "Prob", "RUWE"):
    a, bb = f(m[c])[s][fp], f(m[c])[s][~fp]
    a, bb = a[np.isfinite(a)], bb[np.isfinite(bb)]
    se = np.hypot(1.2533 * np.std(a) / np.sqrt(len(a)), 1.2533 * np.std(bb) / np.sqrt(len(bb)))
    print(f"  {c:5s} FP mediana {np.median(a):.4f}  no FP {np.median(bb):.4f}  diferencia {np.median(a) - np.median(bb):+.4f} "
          f"({(np.median(a) - np.median(bb)) / se:+.1f} SE); Mann-Whitney p {mannwhitneyu(a, bb).pvalue:.2e}")
gl, gb = float(np.median(f(m["GLON"])[s])), float(np.median(f(m["GLAT"])[s]))

print("=== (c) A_V del mapa 3D (Vergely+2022, 25 pc)")
dist = float(np.median(d["dist"][s]))
avm = R.vergely_av(gl, gb, dist)[0]
avm_n = R.vergely_av(gl, gb, dist * 0.9)[0], R.vergely_av(gl, gb, dist * 1.1)[0]
print(f"  l {gl:.3f} b {gb:.3f} d {dist:.0f} pc: A0 mapa {avm:.2f} (d +-10 %: {avm_n[0]:.2f}, {avm_n[1]:.2f}); HR23 AV50 {av:.2f}")
# mapa por estrella (posicion propia, distancia del cumulo): dispersion de la linea de vision dentro del cumulo
lst, bst = f(m["GLON"])[s], f(m["GLAT"])[s]
avs = R.vergely_av(lst, bst, np.full(s.sum(), dist))
print(f"  mapa en la posicion de cada miembro: mediana {np.nanmedian(avs):.2f}, rango 5-95 % [{np.nanpercentile(avs, 5):.2f}; "
      f"{np.nanpercentile(avs, 95):.2f}]; FP {np.nanmedian(avs[fp]):.2f} no FP {np.nanmedian(avs[~fp]):.2f}")

print("=== (d) re-prediccion con el modelo de su particion (control: reproduce el OOF)")
fold = np.load(C / "r38_folds.npz")["fold"]
k = int(np.unique(fold[s])[0])
mdl = XGBClassifier()
mdl.load_model(C / f"r38_fold{k}.json")
J = core.nir_clean(f(m["j_m"])[s], [z[0] for z in q])
H = core.nir_clean(f(m["h_m"])[s], [z[1] for z in q])


def nfp(a):
    X = core.features(G, BP, RP, J, H, K, mod, a)
    return int(np.sum(E.decide(mdl.predict_proba(X)[:, 1], X[:, 0], th))), X


n0, X0 = nfp(av)
print(f"  particion {k}; control: |p - OOF| max {np.max(np.abs(mdl.predict_proba(X0)[:, 1] - oof[s])):.1e}; FP con A_V HR23 {n0}")
shift = float(np.nanmedian(dB[w]))
for nm, a in (("(ii) A_V mapa", avm), ("(iii) AV50 + mediana dB (H1)", av + shift)):
    print(f"  {nm} = {a:.2f}: FP {nfp(a)[0]} de {s.sum()}  (H1 se sostiene si <= 40)")
for a in np.arange(2.0, 5.01, 0.5):
    print(f"    barrido A_V {a:.1f}: FP {nfp(a)[0]}")

print("=== (e) ¿subgigantes? magnitud de los FP respecto del turn-off de MIST a la edad, MOD y A_V de HR23")
mi = core._read_mist()
ages = np.unique(mi["log10_isochrone_age_yr"])
t0 = ages[np.argmin(np.abs(ages - lt))]
gto = mi["Gaia_G_EDR3"][(mi["log10_isochrone_age_yr"] == t0) & (mi["phase"] == 0)].min() + mod + 0.789 * av
for dg in (0.5, 1.0, 2.0):
    print(f"  G turn-off {gto:.2f}: FP con G > G_TO + {dg}: {np.sum(fp & (G > gto + dg))} de {fp.sum()}; no FP: "
          f"{np.sum(~fp & (G > gto + dg))} de {(~fp).sum()}")
