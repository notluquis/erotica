"""Aplicacion a NGC 6383 (254 miembros de P01) del clasificador no circular, con las mismas entradas que dieron el censo de Sagitta
125 [103; 141] (~/phd/agent-findings/ngc6383-sagitta-virac2.md §4.2). Prerregistro §0.5: umbral 0,5, cifra N(P >= 0,5) y Sigma P.

Entradas (tabla limpia ~/phd/agent-findings/scripts/virac2_ngc6383_members_clean.ecsv; paralaje y G/BP/RP del CDS de P01, solo para
Sagitta):
  T2M   2MASS con ph_qual A-E (U y sin contraparte -> NaN).
  BEST  VIRAC2 -> 2MASS o 2MASS, regla de la tabla limpia, SIN correccion de campo (Sagitta: 138).
  CAL   BEST con el residuo VIRAC2->2MASS - 2MASS medido por tramo (Sagitta: 120); porte de `corrected()` de
        ~/phd/agent-findings/scripts/ngc6383_sagitta_virac2_attrib.py. Bootstrap B = 200: PARAMETRICO, cada mediana de tramo
        ~ N(mediana, SE) de su seccion [B] (el no parametrico remuestreaba las estrellas de campo, cuyas cachés ya no estan) +
        fotometrico como alli.
ORACULO DEL PORTE: Sagitta(0,6) debe dar 138 en BEST y 120 en CAL determinista. Si no, el porte no vale.
NGC 6383: MOD50, AV50 de HR23 (consultados en TAPVizieR y guardados en cache/ngc6383_hr23.ecsv); barridos AV +-0,3 y MOD +-0,1.
Env: erotica. Corre: SAGITTA_SRC=... OMP_NUM_THREADS=4 nice -n 19 ~/miniforge3/envs/erotica/bin/python apply_ngc6383.py \
     > apply_ngc6383_measured.txt
"""
from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from xgboost import XGBClassifier

import core
import sagitta_ref as S
from train_validate import f

C = Path(__file__).parent / "cache"
RNG = np.random.default_rng(6383)
BANDS = ("J", "H", "Ks")
CJ, CH, CK = -0.031, 0.015, -0.006          # GF2018 eqs. 7-9 invertidas, como el hub
SATB = {"J": 12.25, "H": 12.25, "Ks": 12.50}
# [B] de ngc6383_sagitta_virac2_attrib_measured.txt: (centro del tramo, mediana, SE); solo tramos con n >= 5, como binned()
CALB = {"J": [(12.675, 0.041, 0.010), (13.25, 0.039, 0.008), (13.75, 0.012, 0.007), (14.25, 0.006, 0.005), (14.75, 0.001, 0.011)],
        "H": [(12.675, 0.023, 0.006), (13.25, -0.008, 0.008), (13.75, -0.062, 0.038)],
        "Ks": [(12.8, 0.011, 0.006), (13.25, 0.021, 0.025)]}

hp = C / "ngc6383_hr23.ecsv"
if not hp.exists():
    import pyvo
    s = pyvo.dal.TAPService("https://tapvizier.cds.unistra.fr/TAPVizieR/tap")
    s.run_sync("""SELECT Name, dist50, MOD16, MOD50, MOD84, AV16, AV50, AV84, diffAV50, logAge50 FROM "J/A+A/673/A114/clusters"
                  WHERE Name = 'NGC_6383'""", maxrec=10).to_table().write(hp, overwrite=True)
h = Table.read(hp)[0]
MOD, AV = float(h["MOD50"]), float(h["AV50"])
print(f"NGC 6383 (HR23): dist50 {float(h['dist50']):.1f} pc, MOD50 {MOD:.3f} [{float(h['MOD16']):.3f}; {float(h['MOD84']):.3f}], "
      f"AV50 {AV:.3f} [{float(h['AV16']):.3f}; {float(h['AV84']):.3f}], diffAV50 {float(h['diffAV50']):.2f} "
      f"(logAge50 {float(h['logAge50']):.2f}: NO se usa)")

tab = Table.read(Path.home() / "phd/agent-findings/scripts/virac2_ngc6383_members_clean.ecsv")
tab = tab[np.argsort(np.asarray(tab["GaiaDR3"], np.int64))]
cds = Table.read(Path.home() / "paper-ngc6383-aa52082-24/cds_final/ngc6383_members.ecsv")
cds = cds[cds["Ref"] == 1]
cds = cds[np.argsort(np.asarray(cds["GaiaDR3"], np.int64))]
assert np.all(np.asarray(cds["GaiaDR3"], np.int64) == np.asarray(tab["GaiaDR3"], np.int64))
N = len(tab)
G, BP, RP = f(tab["G"]), f(tab["BP"]), f(tab["RP"])
plx = f(cds["Plx"])
gal = SkyCoord(f(cds["RAdeg"]) * u.deg, f(cds["DEdeg"]) * u.deg).galactic
AVS = S.predict_av(gal.l.deg, gal.b.deg, plx)
qual = np.array([str(x) if x and not np.ma.is_masked(x) else "" for x in tab["tmass_ph_qual"]])
Q = {b: np.array([q[i] if len(q) == 3 else "" for q in qual]) for i, b in enumerate(BANDS)}
M2 = {b: f(tab[f"{b}_2mass"]) for b in BANDS}
E2 = {b: f(tab[f"e{b}_2mass"]) for b in BANDS}
MV = {b: f(tab[f"{b}_vista"]) for b in BANDS}
EV = {b: f(tab[f"e{b}_vista"]) for b in BANDS}
fbl = f(tab["f_blend_2as"])
amb = np.where(f(tab["Ks_n_amb"]) < 0, False, f(tab["Ks_n_amb"]) / np.maximum(f(tab["Ks_n_det"]), 1) >= 0.1)


def to_2mass(JV, HV, KV):
    c = (JV - KV) / (1 + CJ - CK)
    return {"J": JV - CJ * c, "H": HV - CH * c, "Ks": KV - CK * c}


def usableV(tr):
    out = {}
    for b in BANDS:
        sat = (np.isfinite(MV[b]) & (MV[b] < SATB[b] + 0.1)) | (
            np.isfinite(M2[b]) & ~np.isin(Q[b], ["U", ""]) & (M2[b] < SATB[b]) & ((fbl < 0.1) | ~np.isfinite(MV[b])))
        out[b] = np.isfinite(tr[b]) & ~sat & ~amb
    return out


def best(tr):
    ok = usableV(tr)
    return {b: np.where(ok[b], tr[b], np.where(np.isfinite(M2[b]) & np.isin(Q[b], list("ABCDE")), M2[b], np.nan)) for b in BANDS}


def corrected(cal, phot):
    mv = {b: MV[b] + (RNG.normal(0, 1, N) * np.sqrt(np.clip(EV[b] ** 2 - 0.03 ** 2, 0, None)) if phot else 0) for b in BANDS}
    tr = to_2mass(mv["J"], mv["H"], mv["Ks"])
    out = {b: tr[b] - np.interp(tr[b], cal[b][0], cal[b][1]) for b in BANDS}
    nir = best(out)
    if phot:
        ok = usableV(out)
        for b in BANDS:
            e = np.where(np.isfinite(E2[b]), E2[b], 0.1)
            nir[b] = np.where(~ok[b] & np.isfinite(nir[b]), M2[b] + RNG.normal(0, 1, N) * e, nir[b])
    return nir


def cal_draw(param):
    return {b: (np.array([x[0] for x in CALB[b]]),
                np.array([x[1] + (RNG.normal(0, x[2]) if param else 0.0) for x in CALB[b]])) for b in BANDS}


models = {}
for name in ("all", "young"):
    mm = XGBClassifier()
    mm.load_model(C / f"model_{name}.json")
    models[name] = mm


def own(nir, name="all", mod=MOD, av=AV):
    return models[name].predict_proba(core.features(G, BP, RP, nir["J"], nir["H"], nir["Ks"], mod, av))[:, 1]


def sag(nir):
    return S.predict_pms(plx, AVS, G, BP, RP, nir["J"], nir["H"], nir["Ks"])


T2M = {b: np.where(np.isfinite(M2[b]) & np.isin(Q[b], list("ABCDE")), M2[b], np.nan) for b in BANDS}
TR0 = to_2mass(MV["J"], MV["H"], MV["Ks"])
BEST = best(TR0)
CAL = corrected(cal_draw(False), False)
print(f"entradas: con J/H/Ks validas T2M {[int(np.isfinite(T2M[b]).sum()) for b in BANDS]}, "
      f"BEST {[int(np.isfinite(BEST[b]).sum()) for b in BANDS]}")
print("[8] ORACULO DEL PORTE (Sagitta >= 0,6): "
      f"BEST {int(np.sum(sag(BEST) >= .6))} (esperado 138), CAL {int(np.sum(sag(CAL) >= .6))} (esperado 120), "
      f"T2M {int(np.sum(sag(T2M) >= .6))}")

print("[9] censo de NGC 6383 (umbral 0,5 congelado; Sigma P)")
for tag, nir in (("T2M", T2M), ("BEST", BEST), ("CAL", CAL)):
    for name in ("all", "young"):
        p = own(nir, name)
        print(f"  {tag:5s} propio-{name:5s} N(P>=0,5) {int(np.sum(p >= .5)):3d}  N(P>=0,6) {int(np.sum(p >= .6)):3d}  "
              f"Sigma P {p.sum():6.1f}   | Sagitta N(>=0,6) {int(np.sum(sag(nir) >= .6))}")
pc = own(CAL)
lab = core.label(G, MOD, AV, float(h["logAge50"]))
print(f"  referencia: la etiqueta de §0.2 a la edad HR23 de NGC 6383 llama PMS a {lab.sum()} de {N} (M_G,0 > M_G,on); "
      f"no es un censo: es lo que la definicion dice que deberia salir con eficiencia 1")

B = 200
res = {"own": [], "sag": [], "own_young": []}
fr = np.zeros(N)
for _ in range(B):
    nir = corrected(cal_draw(True), True)
    p = own(nir)
    res["own"].append(int(np.sum(p >= .5)))
    res["own_young"].append(int(np.sum(own(nir, "young") >= .5)))
    res["sag"].append(int(np.sum(sag(nir) >= .6)))
    fr += p >= .5
fr /= B
for k, v in res.items():
    v = np.array(v)
    print(f"  bootstrap CAL B={B} {k:10s} mediana {np.median(v):.0f} [2,5; 16; 84; 97,5 %] = "
          f"[{np.percentile(v, 2.5):.0f}; {np.percentile(v, 16):.0f}; {np.percentile(v, 84):.0f}; {np.percentile(v, 97.5):.0f}]")
print(f"  propio: PMS en >= 95 % / >= 50 % de las realizaciones: {np.sum(fr >= .95)} / {np.sum(fr >= .5)}")

print("[10] sensibilidad (CAL determinista, propio-all, N(P>=0,5))")
for dav in (-0.3, 0.0, 0.3):
    print("  " + "  ".join(f"AV{dav:+.1f} MOD{dm:+.1f}: {int(np.sum(own(CAL, mod=MOD + dm, av=AV + dav) >= .5))}"
                           for dm in (-0.1, 0.0, 0.1)))
for dj in (-0.03, 0.03):
    nir = {b: v.copy() for b, v in CAL.items()}
    nir["J"] = nir["J"] + dj
    print(f"  J {dj:+.2f} mag: propio {int(np.sum(own(nir) >= .5))}, Sagitta {int(np.sum(sag(nir) >= .6))}")
nonir = {b: np.full(N, np.nan) for b in BANDS}
print(f"  sin NIR (solo Gaia): propio {int(np.sum(own(nonir) >= .5))}")
ps_cal = sag(CAL) >= .6
po = pc >= .5
print(f"  acuerdo por estrella CAL: ambos PMS {np.sum(po & ps_cal)}, solo propio {np.sum(po & ~ps_cal)}, "
      f"solo Sagitta {np.sum(~po & ps_cal)}, ninguno {np.sum(~po & ~ps_cal)}")
np.savez_compressed(C / "ngc6383_pred.npz", sid=np.asarray(tab["GaiaDR3"], np.int64), p_cal=pc, p_frac=fr, sag_cal=sag(CAL))
