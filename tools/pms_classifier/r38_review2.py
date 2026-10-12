"""R38-02, cierre (post hoc, pedido por la tarea y por la revision R41): (K) OC_0011 en catalogos alternativos (Cantat-Gaudin+2020,
Dias+2021, Hao+2022) por cono de 0,5 deg con row_limit explicito; (T) ¿la edad puede quitar FP? core.features no recibe la edad, y la
etiqueta usa M_G,on(logAge) contra M_G,0; (B) HR23 - mapa (Vergely+2022) por AV50 en los 115 viejos; (R) base de la RGB de MIST a la
edad/MOD/A_V de HR23 de OC_0011; (X) C* de Riello+2021 (Tabla 2 y ec. 21, verificadas en arXiv:2012.01916 el 2026-10-12) desde los
flujos FG, FBP, FRP de HR23: fraccion > 3 sigma en FP y no FP de los viejos (OOF de la elegida) y en OC_0011.
Env: erotica. Corre: OMP_NUM_THREADS=2 ~/miniforge3/envs/erotica/bin/python r38_review2.py > r38_review2_measured.txt 2> cache/r38_review2.log
"""
import inspect
import json
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.stats import fisher_exact

import core
import exp2_common as E
import exp2_grid as GR
import r38_common as R
from train_validate import f

C = Path("cache")
th = np.array(json.load(open(C / "exp2_selected.json"))["th"])
d = GR.prepare()
m = d["m"]
oof = np.load(C / "exp2_oof.npz")["T0P1K0F0"].astype(float)
dec = E.decide(oof, d["mg0"], th)
oc = d["grp"] == "OC_0011"

print("[K] OC_0011 (272,9928; -22,9714; plx 0,615; pm 0,473, 0,794) en catalogos alternativos, cono 0,5 deg")
from astroquery.vizier import Vizier
V = Vizier(row_limit=-1, columns=["**"])
pos = SkyCoord(272.9928421 * u.deg, -22.97135671 * u.deg)
for nm, cat in (("Cantat-Gaudin+2020", "J/A+A/640/A1"), ("Dias+2021", "J/MNRAS/504/356"), ("Hao+2022", "J/A+A/660/A4")):
    try:
        res = V.query_region(pos, radius=0.5 * u.deg, catalog=cat)
    except Exception as e:  # noqa: BLE001
        print(f"  {nm} ({cat}): ERROR {type(e).__name__}: {str(e)[:120]}")
        continue
    if len(res) == 0:
        print(f"  {nm} ({cat}): 0 filas en ninguna tabla")
    for key in res.keys():
        t = res[key]
        cols = [c for c in t.colnames if c.lower() in ("cluster", "name", "plx", "pmra", "pmde", "agenn", "avnn", "dmnn", "age", "av",
                                                          "logage", "dist", "e_age", "e_av", "_r", "pmrax", "pmdex", "plx_", "ra_icrs", "de_icrs",
                                                          "logt", "a_v", "rv", "ebv", "e(b-v)", "dmode", "dist50", "n")]
        print(f"  {nm} {key}: {len(t)} filas; " + " | ".join(
            "; ".join(f"{c}={row[c]}" for c in cols) for row in t[:5]))

print("[T] ¿la edad mueve los FP de OC_0011?")
print(f"  core.features recibe la edad: {'logt' in inspect.signature(core.features).parameters or 'logage' in inspect.signature(core.features).parameters}")
lt16, lt84 = (float(np.median(f(m[c])[oc])) for c in ("logAge16", "logAge84"))
fpmg = d["mg0"][oc & dec]
for lt in (lt16, lt84):
    print(f"  M_G,on(log t {lt:.2f}) = {float(core.mg_turn_on(lt)):.2f}; M_G,0 maximo de los FP {fpmg.max():.2f} -> "
          f"{'ninguna edad del rango cambia sus etiquetas (todas 0)' if float(core.mg_turn_on(lt)) > fpmg.max() else 'la etiqueta podria cambiar'}")

print("[B] HR23 AV50 - A0 del mapa (Vergely+2022) por AV50, en los 115 viejos")
L, Bg = f(m["GLON"]), f(m["GLAT"])
rows = []
for c in np.unique(d["grp"][d["old"]]):
    s = d["grp"] == c
    rows.append((np.median(f(m["AV50"])[s]), R.vergely_av(np.median(L[s]), np.median(Bg[s]), np.median(d["dist"][s]))[0]))
rows = np.array(rows)
for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4)):
    s = (rows[:, 0] >= lo) & (rows[:, 0] < hi) & np.isfinite(rows[:, 1])
    print(f"  AV50 {lo}-{hi}: {s.sum():3d} cumulos, mediana HR23 - mapa {np.median(rows[s, 0] - rows[s, 1]):+.2f}")

print("[R] base de la RGB (punto mas debil de phase 2) de MIST a la edad, MOD y A_V de HR23 de OC_0011")
mi = core._read_mist()
ages = np.unique(mi["log10_isochrone_age_yr"])
lt = float(np.median(f(m["logAge50"])[oc])); t0 = ages[np.argmin(np.abs(ages - lt))]
mod, av = float(np.median(f(m["MOD50"])[oc])), float(np.median(f(m["AV50"])[oc]))
z = (mi["log10_isochrone_age_yr"] == t0) & (mi["phase"] == 2)
k = np.argmax(mi["Gaia_G_EDR3"][z])
gt = mi["Gaia_G_EDR3"][(mi["log10_isochrone_age_yr"] == t0) & (mi["phase"] == 0)].min() + mod + 0.789 * av
print(f"  log t {t0:.2f}: G = {mi['Gaia_G_EDR3'][z][k] + mod + 0.789 * av:.2f}, BP-RP = "
      f"{mi['Gaia_BP_EDR3'][z][k] - mi['Gaia_RP_EDR3'][z][k] + 0.413 * av:.2f}; G del turn-off {gt:.2f} (corte de (e): {gt + 1:.2f})")

print("[X] C* (Riello+2021) > 3 sigma")
FG, FBP, FRP = f(m["FG"]), f(m["FBP"]), f(m["FRP"])
G, x = f(m["Gmag"]), f(m["BPmag"]) - f(m["RPmag"])
Cx = (FBP + FRP) / FG
poly = np.where(x < 0.5, 1.154360 + 0.033772 * x + 0.032277 * x ** 2,
                np.where(x < 4.0, 1.162004 + 0.011464 * x + 0.049255 * x ** 2 - 0.005879 * x ** 3, 1.057572 + 0.140537 * x))
cs = (Cx - poly) / (0.0059898 + 8.817481e-12 * G ** 7.618399)
hi = cs > 3
for nm, s in (("viejos A u B", d["old"]), ("OC_0011", oc)):
    a, b = np.sum(hi[s & dec] & np.isfinite(cs[s & dec])), np.sum(np.isfinite(cs[s & dec]))
    c_, d_ = np.sum(hi[s & ~dec] & np.isfinite(cs[s & ~dec])), np.sum(np.isfinite(cs[s & ~dec]))
    print(f"  {nm}: FP {a}/{b} = {a / b:.3f}; no FP {c_}/{d_} = {c_ / d_:.3f}; Fisher p {fisher_exact([[a, b - a], [c_, d_ - c_]]).pvalue:.2g}")

print("[K2] OC_0011 con la A_V de Hao+2022 (A_G = 3,78 -> A_V = A_G/0,789 con Wang & Chen; edad 9,10 +- 0,21): FP con el modelo de su particion")
from astroquery.vizier import Vizier as _V
from xgboost import XGBClassifier
hao = _V(row_limit=-1, columns=["**"]).query_constraints(catalog="J/A+A/660/A4/table1", Cluster="OC-0011")[0]
avh = float(hao["AG"][0]) / core.EXT["G"]
fold = np.load(C / "r38_folds.npz")["fold"]
mdl = XGBClassifier(); mdl.load_model(C / f"r38_fold{int(np.unique(fold[oc])[0])}.json")
q = np.array([str(z) if (z is not None and not np.ma.is_masked(z)) else "XXX" for z in m["ph_qual"][oc]])
q = np.array([z if len(z) == 3 else "XXX" for z in q])
J, H, K = (core.nir_clean(f(m[c])[oc], [z[i] for z in q]) for i, c in enumerate(("j_m", "h_m", "ks_m")))
Go, BPo, RPo = f(m["Gmag"])[oc], f(m["BPmag"])[oc], f(m["RPmag"])[oc]
for a in (av, avh):
    X = core.features(Go, BPo, RPo, J, H, K, mod, a)
    dd = E.decide(mdl.predict_proba(X)[:, 1], X[:, 0], th)
    print(f"  A_V {a:.2f}: FP {dd.sum()} de {oc.sum()} (de los 153 FP con HR23 siguen {np.sum(dd & dec[oc])})")
