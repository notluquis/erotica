"""R38-08 (prerregistro hub agent-findings/pms-classifier-non-circular.md §4.3): el MODELO FINAL (entrenado con todo,
cache/exp2_model_T0P1K0F0D1.json) con los umbrales del OOF, aplicado a los 353 cumulos viejos que no entraron a ningun entrenamiento.
Nada se ajusta en el pool. Ademas: la cota de G1 en A/B como distribucion (200 particiones, 40 semillas del bootstrap).
Env: erotica. Corre: cd ~/erotica-wt-pms/tools/pms_classifier && OMP_NUM_THREADS=4 nice -n 10 ~/miniforge3/envs/erotica/bin/python \
     r38_oos_eval.py > r38_oos_eval_measured.txt 2> cache/r38_oos_eval.log
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta
from xgboost import XGBClassifier

import core
import exp2_common as E
import exp2_grid as GR
from r38_oos_data import load_oos

C = Path("cache")
sel = json.load(open(C / "exp2_selected.json"))
th = np.array(sel["th"])
print(f"configuracion {sel['chosen']}, umbrales {th.tolist()}")
o = load_oos()
old, grp, mg0 = o["old"], o["grp"], o["mg0"]
k = E.tramo_index(mg0)
mdl = XGBClassifier()
mdl.load_model(C / "exp2_model_T0P1K0F0D1.json")
p = mdl.predict_proba(o["X"])[:, 1]
dec = E.decide(p, mg0, th)


def cp(x, n):
    return (beta.ppf(0.025, x, n - x + 1) if x > 0 else 0.0), (beta.ppf(0.975, x + 1, n - x) if x < n else 1.0)


def boot_up(pr, g, R, seed):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(g, return_inverse=True)
    n = np.bincount(inv).astype(float)
    x = np.bincount(inv, weights=pr.astype(float))
    v = [x[i].sum() / n[i].sum() for i in (rng.integers(0, len(ug), len(ug)) for _ in range(R))]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def block(name, s, seeds=50, R=2000):
    x, n = int(dec[s].sum()), int(s.sum())
    lo, hi = cp(x, n)
    ups = np.array([boot_up(dec[s], grp[s], R, sd)[1] for sd in range(seeds)])
    los = np.array([boot_up(dec[s], grp[s], R, sd)[0] for sd in range(3)])
    print(f"  {name:28s} {x:5d}/{n:5d} = {x / n:.4f} CP [{lo:.4f}; {hi:.4f}] | boot cota 97,5 % sobre {seeds} semillas: "
          f"min {ups.min():.4f} q05 {np.percentile(ups, 5):.4f} med {np.median(ups):.4f} q95 {np.percentile(ups, 95):.4f} "
          f"max {ups.max():.4f} frac>0,10 {np.mean(ups > 0.10):.2f} (cota inf ~{np.median(los):.4f}) | cumulos {len(np.unique(grp[s]))}",
          flush=True)
    return dict(x=x, n=n, cp=[lo, hi], up_med=float(np.median(ups)), frac=float(np.mean(ups > 0.10)))


print("[O1] modelo final en el pool (solo etiqueta 0)")
res = {"all": block("todos", old)}
for i, nm in enumerate(E.TRAMO_NAMES):
    res[nm] = block(nm, old & (k == i))
g1 = res["all"]["cp"][1] <= 0.10 and res["all"]["up_med"] <= 0.10
print(f"  G1-OOS {'PASA' if g1 else 'FALLA'} (regla §4.3: mediana de la cota bootstrap y cota CP <= 0,10)")
print("  G2-OOS por tramo (cota bootstrap mediana <= 0,10): " + "  ".join(
    f"{nm} {'si' if res[nm]['up_med'] <= 0.10 else 'no'}" for nm in E.TRAMO_NAMES))
cond = old & (o["av"] >= 0.6) & (o["av"] <= 2.0) & (o["dav"] >= 1.0) & (o["dav"] <= 3.0)
print("[O2] subconjunto en condiciones de NGC 6383 (AV50 0,6-2,0; diffAV50 1,0-3,0)")
block("condiciones NGC 6383", cond, seeds=20)
for i, nm in enumerate(E.TRAMO_NAMES):
    block("  cond. " + nm, cond & (k == i), seeds=20)
print("[O3] FPR del pool por AV50 y por diffAV50 (agregado)")
for lo_, hi_ in ((0, 1), (1, 2), (2, 3), (3, 4)):
    s = old & (o["av"] >= lo_) & (o["av"] < hi_)
    print(f"  AV50 {lo_}-{hi_}: cumulos {len(np.unique(grp[s])):3d} FPR {dec[s].mean():.4f}")
for lo_, hi_ in ((0, 1), (1, 2), (2, 3), (3, 9)):
    s = old & (o["dav"] >= lo_) & (o["dav"] < hi_)
    print(f"  diffAV50 {lo_}-{hi_}: cumulos {len(np.unique(grp[s])):3d} FPR {dec[s].mean():.4f}")
ug, inv = np.unique(grp[old], return_inverse=True)
fpc = np.bincount(inv, weights=dec[old].astype(float))
nc = np.bincount(inv)
top = np.argsort(-fpc)[:5]
print("  cumulos con mas FP: " + "; ".join(
    f"{ug[j]} {int(fpc[j])}/{nc[j]} AV50 {np.median(o['av'][old][inv == j]):.2f} diffAV50 {np.median(o['dav'][old][inv == j]):.2f}"
    for j in top) + f"; los 5 suman {int(fpc[top].sum())} de {int(fpc.sum())}")

print("[O4] semilla del modelo final: 5 reentrenamientos con todo (random_state de XGBoost y sorteo de la aumentacion de NIR)")
d = GR.prepare()
y, stab = d["tg"]["T0"]
w = E.config_weights(dict(T="T0", P="P1", K="K0", F="F0"), y, stab, d["dobs"], d["dth"], d["logt"])
fprs = []
for j in range(5):
    mj = core.fit(d["X"], y, seed=777 + j, params=dict(core.XGB_PARAMS, random_state=777 + j), sample_weight=w)
    dj = E.decide(mj.predict_proba(o["X"])[:, 1], mg0, th)
    fprs.append([dj[old].mean()] + [dj[old & (k == i)].mean() for i in range(4)])
    print(f"  semilla {777 + j}: global {fprs[-1][0]:.4f} por tramo " + " ".join(f"{v:.4f}" for v in fprs[-1][1:]), flush=True)
fprs = np.array(fprs)
print(f"  rango global [{fprs[:, 0].min():.4f}; {fprs[:, 0].max():.4f}] (modelo guardado {res['all']['x'] / res['all']['n']:.4f})")

print("[O5] cota de G1 en B como DISTRIBUCION: 200 particiones A/B (umbrales recalculados en A) y bootstrap R = 1000")
oofp = np.load(C / "exp2_oof.npz")["T0P1K0F0"].astype(float)
uo = np.unique(d["grp"][d["old"]])
rng = np.random.default_rng(99)
ups, fprB, ths = [], [], []
for it in range(200):
    a = set(rng.permutation(uo)[: len(uo) // 2])
    A = d["old"] & np.isin(d["grp"], list(a))
    B = d["old"] & ~A
    t_ = E.tramo_thresholds(oofp, d["mg0"], A, target_fpr=0.03)
    dd = E.decide(oofp, d["mg0"], t_)
    ups.append(boot_up(dd[B], d["grp"][B], 1000, it)[1])
    fprB.append(dd[B].mean())
    ths.append(t_)
ups, fprB = np.array(ups), np.array(fprB)
print(f"  cota boot de G1: q05 {np.percentile(ups, 5):.4f} q50 {np.median(ups):.4f} q95 {np.percentile(ups, 95):.4f} max {ups.max():.4f}; "
      f"frac > 0,10: {np.mean(ups > 0.10):.3f}")
print(f"  FPR en B: q05 {np.percentile(fprB, 5):.4f} q50 {np.median(fprB):.4f} q95 {np.percentile(fprB, 95):.4f}")
ths = np.array(ths)
print("  umbrales por tramo (q05/q50/q95): " + "  ".join(
    f"{nm} {np.percentile(ths[:, i], 5):.2f}/{np.median(ths[:, i]):.2f}/{np.percentile(ths[:, i], 95):.2f}" for i, nm in enumerate(E.TRAMO_NAMES)))
json.dump({kk: v for kk, v in res.items()}, open(C / "r38_oos_eval.json", "w"), indent=1)
