"""R38 §4.4, brazo S2 (prerregistrado bajo H2): umbrales por tramo calibrados (FPR <= 0,03, como D1) SOLO en los viejos de A con
diffAV50 en [1,0; 3,0], las condiciones de NGC 6383 (diffAV50 2,0). Se corre porque r38_posthoc [P2] midio en el pool, a diffAV50
1,5-2,5, FPR 0,069 (cota 0,114) en +5..+7 con los umbrales de S0. Mismo modelo; cambian los umbrales. Criterios G1-G4 de §2.4 en B y
en el pool; censo de NGC 6383 por tramo con el intervalo de §2.5 con el procedimiento y la semilla de exp2_census (6384), S0 y S2 pareados sorteo a sorteo, umbrales de cada
sorteo recalculados en los cumulos de A fuera de bolsa con diffAV50 en [1; 3]. S0 (control) al lado.
Env: erotica. Corre: cd ~/erotica-wt-pms/tools/pms_classifier && SAGITTA_SRC=~/.local/lib/sagitta-1.2.3 OMP_NUM_THREADS=4 nice -n 10 \
     ~/miniforge3/envs/erotica/bin/python r38_s2.py > r38_s2_measured.txt 2> cache/r38_s2.log
"""
import json
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.stats import beta
from xgboost import XGBClassifier

import core
import exp2_common as E
import exp2_grid as GR
from r38_oos_data import load_oos
from train_validate import f

C = Path("cache")
sel = json.load(open(C / "exp2_selected.json"))
th0 = np.array(sel["th"])
cfg = dict(T="T0", P="P1", K="K0", F="F0", D="D1")
d = GR.prepare()
dav_d = f(d["m"]["diffAV50"])
oof = np.load(C / "exp2_oof.npz")["T0P1K0F0"].astype(float)
Asub = d["A"] & (dav_d >= 1.0) & (dav_d <= 3.0)
th2 = E.tramo_thresholds(oof, d["mg0"], Asub, target_fpr=0.03)
print(f"[S2-0] A con diffAV50 1-3: {len(np.unique(d['grp'][Asub]))} cumulos, {Asub.sum()} estrellas; umbrales S0 {th0.tolist()} -> S2 {th2.tolist()}")


def cp(x, n):
    return (beta.ppf(0.025, x, n - x + 1) if x > 0 else 0.0), (beta.ppf(0.975, x + 1, n - x) if x < n else 1.0)


def bup(pr, g, R=2000, seeds=10):
    out = []
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        ug, inv = np.unique(g, return_inverse=True)
        n = np.bincount(inv).astype(float); x = np.bincount(inv, weights=pr.astype(float))
        out.append(np.percentile([x[i].sum() / n[i].sum() for i in (rng.integers(0, len(ug), len(ug)) for _ in range(R))], 97.5))
    return np.median(out)


def table(tag, dec, s, grp, k):
    x, n = int(dec[s].sum()), int(s.sum())
    up = bup(dec[s], grp[s])
    print(f"  {tag:34s} global {x}/{n} = {x / n:.4f} CP {cp(x, n)[1]:.4f} boot {up:.4f} -> G1 {'pasa' if max(cp(x, n)[1], up) <= 0.10 else 'FALLA'} | "
          + "  ".join(f"{nm} {dec[s & (k == i)].mean():.3f} ({bup(dec[s & (k == i)], grp[s & (k == i)], 1000, 3):.3f})"
                      for i, nm in enumerate(E.TRAMO_NAMES) if (s & (k == i)).sum()), flush=True)


kd = E.tramo_index(d["mg0"])
print("[S2-1] mitad B (OOF), FPR y cota bootstrap (mediana de 10 semillas); por tramo FPR (cota)")
for nm, th in (("S0", th0), ("S2", th2)):
    dec = E.decide(oof, d["mg0"], th)
    table(f"{nm} B", dec, d["B"], d["grp"], kd)
    table(f"{nm} B diffAV50 1-3", dec, d["B"] & (dav_d >= 1) & (dav_d <= 3), d["grp"], kd)
o = load_oos()
mdl = XGBClassifier()
mdl.load_model(C / "exp2_model_T0P1K0F0D1.json")
po = mdl.predict_proba(o["X"])[:, 1]
ko = E.tramo_index(o["mg0"])
print("[S2-2] pool fuera de muestra, modelo final")
for nm, th in (("S0", th0), ("S2", th2)):
    dec = E.decide(po, o["mg0"], th)
    table(f"{nm} pool", dec, o["old"], o["grp"], ko)
    table(f"{nm} pool diffAV50 1,5-2,5", dec, o["old"] & (o["dav"] >= 1.5) & (o["dav"] <= 2.5), o["grp"], ko)
    table(f"{nm} pool diffAV50 >= 2", dec, o["old"] & (o["dav"] >= 2.0), o["grp"], ko)

# G3 / G4 con el modelo final
import sagitta_ref as S
cl = Table.read(C / "hr23_clusters_sel.ecsv")
r = cl[cl["Name"] == "NGC_6530"][0]
MOD6, AV6, LOGT6 = float(r["MOD50"]), float(r["AV50"]), float(r["logAge50"])
g = Table.read(C / "ngc6530_dr3.ecsv")
GAL = SkyCoord(f(g["RAdeg"]) * u.deg, f(g["DEdeg"]) * u.deg).galactic
q6 = np.array([str(x) if (x is not None and not np.ma.is_masked(x) and len(str(x)) == 3) else "XXX" for x in g["ph_qual"]])
J, H, K = (core.nir_clean(f(g[c]), [s[i] for s in q6]) for i, c in enumerate(("j_m", "h_m", "ks_m")))
G, BP, RP, plx = f(g["Gmag"]), f(g["BPmag"]), f(g["RPmag"]), f(g["Plx"])
ok = np.isfinite(G) & np.isfinite(BP) & np.isfinite(RP)
sel6 = ok & core.label(G, MOD6, AV6, LOGT6)
X6 = core.features(G, BP, RP, J, H, K, MOD6, AV6)
p6 = mdl.predict_proba(X6)[:, 1]
s6, _ = S.sagitta_pms(GAL.l.deg, GAL.b.deg, plx, G, BP, RP, J, H, K)
k6 = E.tramo_index(X6[:, 0])
RNG = np.random.default_rng(6530)
print(f"[S2-3] G3 NGC 6530 (n {sel6.sum()}), recall y diferencia pareada contra Sagitta(0,6) U->token")
for nm, th in (("S0", th0), ("S2", th2)):
    d6 = E.decide(p6, X6[:, 0], th)
    idx = np.where(sel6)[0]
    a, b = d6[idx].astype(float), (s6[idx] >= .6).astype(float)
    dd = [np.mean(a[s] - b[s]) for s in (RNG.integers(0, len(idx), len(idx)) for _ in range(2000))]
    lo, hi = np.percentile(dd, [2.5, 97.5])
    rt = [float(np.mean(d6[sel6 & (k6 == i)])) for i in range(4)]
    print(f"  {nm}: recall {d6[sel6].mean():.3f}; por tramo {np.round(rt, 3).tolist()}; reponderado a NGC 6383 {np.dot(E.W6383, rt):.3f}; "
          f"diferencia {np.mean(a - b):+.3f} [{lo:+.3f}; {hi:+.3f}] -> G3 {'pasa' if not hi < -0.05 else 'FALLA'}")
t83 = Table.read(Path.home() / "phd/agent-findings/scripts/virac2_ngc6383_members_clean.ecsv")
q83 = np.array([str(x) if (x is not None and not np.ma.is_masked(x) and len(str(x)) == 3) else "XXX" for x in t83["tmass_ph_qual"]])
miss83 = np.stack([~np.isfinite(core.nir_clean(f(t83[f"{bd}_2mass"]), [s[i] for s in q83])) for i, bd in enumerate(("J", "H", "Ks"))], 1)
G83 = f(t83["G"])
gb, gb83 = np.clip(np.floor(G), 8, 20), np.clip(np.floor(G83), 8, 20)
d6_2 = E.decide(p6, X6[:, 0], th2)
fl = []
for _ in range(30):
    Jm, Hm, Km = J.copy(), H.copy(), K.copy()
    for i in np.where(ok)[0]:
        pool = np.where(gb83 == gb[i])[0]
        if not len(pool):
            pool = np.argsort(np.abs(G83 - G[i]))[:5]
        mk = miss83[RNG.choice(pool)]
        Jm[i], Hm[i], Km[i] = (np.nan if mk[0] else Jm[i]), (np.nan if mk[1] else Hm[i]), (np.nan if mk[2] else Km[i])
    Xa = core.features(G, BP, RP, Jm, Hm, Km, MOD6, AV6)
    fl.append(np.mean(E.decide(mdl.predict_proba(Xa)[:, 1], Xa[:, 0], th2)[ok] != d6_2[ok]))
print(f"[S2-4] G4 ablacion de NIR con S2: cambia de clase {np.mean(fl):.3f} (max {np.max(fl):.3f}) -> {'pasa' if np.mean(fl) <= 0.15 else 'FALLA'}")

# censo
import apply_ngc6383 as AP
from exp2_census import fit_sub, per_tramo, sAV, sMOD
print("[S2-5] censo de NGC 6383 por tramo, determinista (modelo final)")
for tag, nir in (("T2M", AP.T2M), ("BEST", AP.BEST), ("CAL", AP.CAL)):
    for nm, th in (("S0", th0), ("S2", th2)):
        n, tot = per_tramo(mdl, cfg, th, nir)
        print(f"  {tag:5s} {nm}: " + "  ".join(f"{t} {a}/{b_}" for t, a, b_ in zip(E.TRAMO_NAMES, n, tot)))
print("[S2-6] intervalo, B = 30 mismo procedimiento y semilla que exp2_census (6384); S0 y S2 pareados en cada sorteo: umbrales de cada sorteo en A fuera de bolsa (S0) o en A fuera "
      "de bolsa con diffAV50 1-3 (S2)", flush=True)
RNG2 = np.random.default_rng(6384)
ug = np.unique(d["grp"])
idx = {c: np.where(d["grp"] == c)[0] for c in ug}
tot0, tot2 = [], []
for bb in range(30):
    pick = RNG2.choice(ug, len(ug))
    s = np.concatenate([idx[c] for c in pick])
    mb = fit_sub(d, cfg, s, seed=2000 + bb)
    oob = d["A"] & ~np.isin(d["grp"], pick)
    pA = np.zeros(len(d["y0"]))
    pA[oob] = mb.predict_proba(d["X"][oob])[:, 1]
    t0_ = E.tramo_thresholds(pA, d["mg0"], oob, target_fpr=0.03)
    o2 = oob & (dav_d >= 1) & (dav_d <= 3)
    t2_ = E.tramo_thresholds(pA, d["mg0"], o2, target_fpr=0.03)
    nir = AP.corrected(AP.cal_draw(True), True)
    av, mod = AP.AV + RNG2.normal(0, sAV), AP.MOD + RNG2.normal(0, sMOD)
    tot0.append(per_tramo(mb, cfg, t0_, nir, av, mod)[0])
    tot2.append(per_tramo(mb, cfg, t2_, nir, av, mod)[0])
    print(f"    b {bb}: S0 {t0_.round(2).tolist()} {tot0[-1].tolist()} | S2 (oob 1-3: {len(np.unique(d['grp'][o2]))} cum) "
          f"{t2_.round(2).tolist()} {tot2[-1].tolist()}", flush=True)
for nm, v in (("S0", np.array(tot0)), ("S2", np.array(tot2))):
    print(f"  {nm}: " + "  ".join(f"{t} {np.median(v[:, i]):.0f} [{np.percentile(v[:, i], 2.5):.0f}; {np.percentile(v[:, i], 97.5):.0f}]"
                               for i, t in enumerate(E.TRAMO_NAMES)) +
          f" | -5..+3 y +5..+7 {np.median(v[:, [0, 2]].sum(1)):.0f} [{np.percentile(v[:, [0, 2]].sum(1), 2.5):.0f}; "
          f"{np.percentile(v[:, [0, 2]].sum(1), 97.5):.0f}] | -5..+7 {np.median(v[:, :3].sum(1)):.0f} [{np.percentile(v[:, :3].sum(1), 2.5):.0f}; "
          f"{np.percentile(v[:, :3].sum(1), 97.5):.0f}]")
json.dump(dict(th0=th0.tolist(), th2=th2.tolist()), open(C / "r38_s2.json", "w"))
