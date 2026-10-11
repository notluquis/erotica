"""Segundo experimento, paso 1: la grilla de 24 entrenamientos (T x P x K x F) con CV por cumulo, y la SELECCION de §2.3 usando
SOLO la mitad de ajuste (A) de los viejos y los jovenes OOF. No mira la mitad B ni NGC 6530.

Prerregistro: ~/phd/agent-findings/pms-classifier-non-circular.md §2 (hub 42b93a6; codigo d426c01).
Env: erotica. Corre (4 hilos, segundo plano):
  cd ~/erotica-wt-pms/tools/pms_classifier && OMP_NUM_THREADS=4 nice -n 19 ~/miniforge3/envs/erotica/bin/python exp2_grid.py \
      > exp2_grid_measured.txt 2> cache/exp2_grid.log
Salida: exp2_grid_measured.txt (versionado); cache/exp2_oof.npz (OOF de las 24), cache/exp2_selected.json.
CONTROL: T0P0K0F0 debe dar 1402 FP de 14405 en los viejos (el F2); si no, la grilla no vale y el script se detiene.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

import core
import exp2_common as E
from train_validate import build, f, load

C = Path(__file__).parent / "cache"


def prepare():
    m = load()
    m = m[m["role"] == "train"]
    X, y0, ph = build(m)
    grp = np.asarray(m["Name"]).astype(str)
    logt, dist = f(m["logAge50"]), f(m["dist50"])
    mg0 = X[:, core.FEATURES.index("MG0")]
    tg, dth = E.targets(ph["G"], ph["mod"], ph["av"], f(m["logAge16"]), logt, f(m["logAge84"]), mg0)
    assert np.array_equal(tg["T0"][0], y0)
    X1 = E.extra_features(X)
    dobs = X1[:, -2]
    old = (logt >= 8.5) & (dist >= 600) & (dist <= 2500) & ~y0
    A, B = E.old_split(grp, old)
    young = (logt < 6.8) & y0
    return dict(m=m, X=X, X1=X1, y0=y0, tg=tg, dth=dth, dobs=dobs, grp=grp, logt=logt, dist=dist, mg0=mg0, old=old, A=A, B=B,
                young=young, sid=np.asarray(m["GaiaDR3"], np.int64))


CFGS = [dict(T=T, P=P, K=K, F=F) for T, P, K, F in itertools.product(("T0", "T1"), ("P0", "P1"), ("K0", "K1", "K2"), ("F0", "F1"))]


def tag(c):
    return "".join(c[k] for k in ("T", "P", "K", "F"))


def nfactors(c):
    return sum(c[k][1] != "0" for k in ("T", "P", "K", "F"))


def oof_for(d, c):
    X = d["X1"] if c["F"] == "F1" else d["X"]
    y, stab = d["tg"][c["T"]]
    w = E.config_weights(c, y, stab, d["dobs"], d["dth"], d["logt"])
    trivial = c["P"] == "P0" and c["K"] == "K0"
    oof = np.full(len(y), np.nan)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, d["grp"])):
        mdl = fit_any(X[tr], y[tr], seed=20261011 + k, w=None if trivial else w[tr], nfeat=X.shape[1])
        oof[te] = mdl.predict_proba(X[te])[:, 1]
    return oof, w


def fit_any(X, y, seed, w, nfeat):
    """core.fit con la aumentacion de NIR aplicada tambien a las features nuevas que dependen de Ks (DZK)."""
    if nfeat == len(core.FEATURES):
        return core.fit(X, y, seed=seed, sample_weight=w)
    from xgboost import XGBClassifier
    rng = np.random.default_rng(seed)
    Xd = X.copy()
    kdrop = rng.random(len(X)) < 0.25
    for col in [core.FEATURES.index(c) for c in core.NIR_FEATURES] + [nfeat - 1]:
        Xd[kdrop, col] = np.nan
    mdl = XGBClassifier(**core.XGB_PARAMS)
    mdl.fit(Xd, y.astype(int), sample_weight=w)
    return mdl


def evaluate(d, p, decided_fn):
    """FPR por tramo en A, recall por tramo en jovenes (§0.2), recall ponderado a NGC 6383."""
    k = E.tramo_index(d["mg0"])
    pr = decided_fn(p)
    fprA = [float(np.mean(pr[d["A"] & (k == i)])) for i in range(len(E.TRAMOS))]
    rec = [float(np.mean(pr[d["young"] & (k == i)])) for i in range(len(E.TRAMOS))]
    return dict(fprA=fprA, fprA_all=float(np.mean(pr[d["A"]])), rec=rec, recw=float(np.dot(E.W6383, rec)))


if __name__ == "__main__":
    t0 = time.time()
    d = prepare()
    print(f"[G0] {len(d['y0'])} estrellas; viejos {d['old'].sum()} (A {d['A'].sum()}, B {d['B'].sum()}); jovenes PMS (log t < 6,8) "
          f"{d['young'].sum()}")
    for T in ("T0", "T1"):
        y, stab = d["tg"][T]
        print(f"  {T}: PMS {y.sum()} ({y.mean():.3f}); estabilidad < 1 en {np.mean(stab < 1):.3f}")
    y1 = d["tg"]["T1"][0]
    k2drop = d["y0"] & np.isfinite(d["dth"]) & (d["dobs"] < d["dth"] - E.KAPPA)
    print(f"  K2 quita (T0) {k2drop.sum()} etiquetas 1; P1 ESS de Kish {E.kish(E.age_weights(d['logt'], 1.0)):.0f} de {len(y1)}")
    oofs, summ = {}, {}
    for c in CFGS:
        oof, w = oof_for(d, c)
        oofs[tag(c)] = oof.astype(np.float32)
        if tag(c) == "T0P0K0F0":
            fp = int(np.sum(oof[d["old"]] >= 0.5))
            print(f"  CONTROL T0P0K0F0: FP en viejos {fp} de {d['old'].sum()} (F2: 1402 de 14405)", flush=True)
            if fp != 1402:
                sys.exit("CONTROL FALLA: la grilla no reproduce el modelo del F2")
        r0 = evaluate(d, oof, lambda p: p >= 0.5)
        th = E.tramo_thresholds(oof, d["mg0"], d["A"], target_fpr=0.03)
        r1 = evaluate(d, oof, lambda p, th=th: E.decide(p, d["mg0"], th))
        summ[tag(c) + "D0"] = dict(cfg=dict(c, D="D0"), th=[0.5] * 4, **r0)
        summ[tag(c) + "D1"] = dict(cfg=dict(c, D="D1"), th=th.tolist(), **r1)
        print(f"  {tag(c)} ESS {E.kish(w):.0f} ({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(C / "exp2_oof.npz", sid=d["sid"], **oofs)

    print("[G-sel] por configuracion: FPR en A por tramo (-5..+3, +3..+5, +5..+7, +7..+12), FPR A global, recall jovenes por tramo, "
          "recall ponderado a NGC 6383; umbrales (D1)")
    elig = []
    for key, r in summ.items():
        ok = all(r["fprA"][i] <= 0.10 for i in (1, 2, 3))
        if ok:
            elig.append(key)
        print(f"  {key:12s} {'ELEGIBLE' if ok else '        '} FPR_A {np.round(r['fprA'], 3).tolist()} glob {r['fprA_all']:.3f} | "
              f"rec {np.round(r['rec'], 3).tolist()} pond {r['recw']:.3f} | th {np.round(r['th'], 2).tolist()}")
    print(f"  tamano de la busqueda: {len(summ)} configuraciones; elegibles {len(elig)}")
    if not elig:
        print("  NINGUNA ELEGIBLE: el falsificador de §2.4 se cumple (el arreglo no sirve)")
        sys.exit(0)
    best = max(summ[k]["recw"] for k in elig)
    tied = [k for k in elig if summ[k]["recw"] >= best - 0.005]
    chosen = min(tied, key=lambda k: (nfactors(summ[k]["cfg"]) + (summ[k]["cfg"]["D"] == "D1"), -summ[k]["recw"]))
    print(f"  ELEGIDA: {chosen} (recall ponderado {summ[chosen]['recw']:.3f}; empatadas a 0,005: {tied})")
    json.dump(dict(chosen=chosen, **summ[chosen]), open(C / "exp2_selected.json", "w"), indent=1)
    json.dump(summ, open(C / "exp2_summary.json", "w"), indent=1)
    print(f"  ({time.time() - t0:.0f} s)")
