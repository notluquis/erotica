"""Entrenamiento + validacion cruzada por cumulo del clasificador PMS no circular, con Sagitta como control en las mismas estrellas.

Prerregistro: ~/phd/agent-findings/pms-classifier-non-circular.md §0.3-§0.4. Entradas: cache/ de fetch_training.py.
Env: ~/miniforge3/envs/erotica (xgboost 3.4.1, scikit-learn 1.9.1, torch 2.12.0). Corre (<= 4 nucleos):
  cd ~/erotica-wt-pms/tools/pms_classifier && SAGITTA_SRC=<sdist extraido> OMP_NUM_THREADS=4 nice -n 19 \
      ~/miniforge3/envs/erotica/bin/python train_validate.py > train_validate_measured.txt
Salida: train_validate_measured.txt (versionado), cache/oof.npz, cache/model_all.json, cache/model_young.json.

QUE MIDE (fijado antes de correr):
 [0] el conjunto: estrellas, cumulos, fraccion PMS; GUARDIA: los 254 de P01 no estan en el entrenamiento (cruce por source_id = 0).
 [1] ruido de etiqueta: fraccion que cambia de etiqueta con logAge16 / logAge84 en vez de logAge50.
 [2] CV GroupKFold(5) por cumulo: AUC, recall, precision, FPR al umbral 0,5; por estrato de edad, distancia y A_V; la celda de
     NGC 6383 (logAge < 6,8; 700-1600 pc; AV 0,6-2,0); IC: Clopper-Pearson (estrellas) y bootstrap por cumulo (B = 300).
     Calibracion: fraccion observada de PMS por bin de P.
 [3] Sagitta en las MISMAS estrellas, umbrales 0,6 y 0,5, contra la misma etiqueta. La etiqueta es la de ESTE clasificador, no la
     de Sagitta (Kounkel+2020 + corte de CMD): esta comparacion favorece al propio por construccion. La verdad independiente es [4] y
     NGC 6530 (validate_ngc6530.py).
 [4] FPR en miembros de cumulos viejos (logAge >= 8,5; 600-2500 pc) con etiqueta 0 (sobre la secuencia: no PMS es cierto salvo
     binarias/contaminacion), fuera de particion. F2 del prerregistro: falla si la cota superior del IC 95 % > 0,10.
 [5] prior de la mezcla de edades: modelo entrenado solo con logAge < 8,3 (para aplicarlo a NGC 6383 aparte).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table, join
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import core

HERE = Path(__file__).parent
C = HERE / "cache"
TH = 0.5
RNG = np.random.default_rng(20261011)


def f(x):
    return np.ma.filled(np.ma.asarray(x).astype(float), np.nan)


def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def load():
    cl = Table.read(C / "hr23_clusters_sel.ecsv")
    m = Table.read(C / "hr23_members.ecsv")
    tm = Table.read(C / "tmass.ecsv")
    tm.rename_column("source_id", "GaiaDR3")
    m = join(m, cl["Name", "MOD50", "AV50", "diffAV50", "dist50", "logAge16", "logAge50", "logAge84", "role"], keys="Name")
    m = join(m, tm["GaiaDR3", "ph_qual", "j_m", "h_m", "ks_m"], keys="GaiaDR3", join_type="left")
    return m


def build(m):
    q = np.array([str(x) if (x is not None and not np.ma.is_masked(x)) else "XXX" for x in m["ph_qual"]])
    q = np.array([s if len(s) == 3 else "XXX" for s in q])
    J = core.nir_clean(f(m["j_m"]), [s[0] for s in q])
    H = core.nir_clean(f(m["h_m"]), [s[1] for s in q])
    K = core.nir_clean(f(m["ks_m"]), [s[2] for s in q])
    G, BP, RP = f(m["Gmag"]), f(m["BPmag"]), f(m["RPmag"])
    mod, av = f(m["MOD50"]), f(m["AV50"])
    X = core.features(G, BP, RP, J, H, K, mod, av)
    y = core.label(G, mod, av, f(m["logAge50"]))
    return X, y, dict(G=G, BP=BP, RP=RP, J=J, H=H, K=K, mod=mod, av=av)


def metrics(p, y, th=TH):
    pr = p >= th
    tp, fn = np.sum(pr & y), np.sum(~pr & y)
    fp, tn = np.sum(pr & ~y), np.sum(~pr & ~y)
    auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan
    return dict(n=len(y), npos=int(y.sum()), auc=auc, rec=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1),
                prec=tp / max(tp + fp, 1), tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn))


def cluster_boot(p, y, grp, key, B=300, th=TH):
    """IC 95 % de una metrica remuestreando cumulos (las estrellas de un cumulo no son independientes)."""
    ug, inv = np.unique(grp, return_inverse=True)
    idx = [np.where(inv == i)[0] for i in range(len(ug))]
    vals = []
    for _ in range(B):
        s = np.concatenate([idx[i] for i in RNG.integers(0, len(ug), len(ug))])
        vals.append(metrics(p[s], y[s], th)[key])
    v = np.array(vals, float)
    return np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)


def line(tag, p, y, grp, th=TH):
    r = metrics(p, y, th)
    rl, rh = cp(r["tp"], r["tp"] + r["fn"])
    fl, fh = cp(r["fp"], r["fp"] + r["tn"])
    if len(np.unique(grp)) >= 5:
        brl, brh = cluster_boot(p, y, grp, "rec", th=th)
        bfl, bfh = cluster_boot(p, y, grp, "fpr", th=th)
        bs = f" | boot cumulos rec [{brl:.3f}; {brh:.3f}] fpr [{bfl:.3f}; {bfh:.3f}]"
    else:
        bs = ""
    print(f"  {tag:34s} n {r['n']:6d} (PMS {r['npos']:6d}, {len(np.unique(grp)):4d} cum) AUC {r['auc']:.3f}  "
          f"rec {r['rec']:.3f} [{rl:.3f}; {rh:.3f}]  FPR {r['fpr']:.3f} [{fl:.3f}; {fh:.3f}]  prec {r['prec']:.3f}{bs}",
          flush=True)


if __name__ == "__main__":
    t0 = time.time()
    m = load()
    m = m[m["role"] == "train"]
    X, y, ph = build(m)
    grp = np.asarray(m["Name"]).astype(str)
    logt, dist, av = f(m["logAge50"]), f(m["dist50"]), f(m["AV50"])
    ok = np.isfinite(X[:, 0]) & np.isfinite(X[:, 1])  # G y BP-RP presentes
    print(f"[0] miembros HR23 Prob>=0,7 en cumulos de entrenamiento: {len(m)}; con G y BP-RP: {ok.sum()}; "
          f"cumulos {len(np.unique(grp))}")
    m, X, y, grp, logt, dist, av = m[ok], X[ok], y[ok], grp[ok], logt[ok], dist[ok], av[ok]
    ph = {k: v[ok] for k, v in ph.items()}
    sid = np.asarray(m["GaiaDR3"], np.int64)
    p01 = Table.read(Path.home() / "phd/agent-findings/scripts/virac2_ngc6383_members_clean.ecsv")
    ids254 = np.asarray(p01["GaiaDR3"], np.int64)
    inter = np.intersect1d(sid, ids254)
    print(f"    GUARDIA: miembros de P01 (254) en el entrenamiento: {len(inter)}  (debe ser 0)")
    mut = np.intersect1d(np.append(sid, ids254[0]), ids254)
    print(f"    mutacion de la guardia (se agrega 1 de los 254 a proposito): {len(mut)}  (debe ser 1: la guardia ve)")
    if len(inter):
        sys.exit("FALLA la guardia de exclusion")
    nir = np.isfinite(X[:, core.FEATURES.index("MKs0")])
    print(f"    PMS por etiqueta: {y.sum()} ({y.mean():.3f}); con Ks valida {nir.mean():.3f}; "
          f"edad logAge50 [p5 p50 p95] {np.percentile(logt, [5, 50, 95]).round(2)}")

    # [1] ruido de etiqueta
    y16 = core.label(ph["G"], ph["mod"], ph["av"], f(m["logAge16"]))
    y84 = core.label(ph["G"], ph["mod"], ph["av"], f(m["logAge84"]))
    flip = (y16 != y) | (y84 != y)
    print(f"[1] ruido de etiqueta: cambian con logAge16 o 84: {flip.sum()} ({flip.mean():.3f}); "
          f"en logAge50 < 6,8: {flip[logt < 6.8].mean():.3f}")

    # [2] CV por cumulo
    gkf = GroupKFold(n_splits=5)
    oof = np.full(len(y), np.nan)
    for k, (tr, te) in enumerate(gkf.split(X, y, grp)):
        mdl = core.fit(X[tr], y[tr], seed=20261011 + k)
        oof[te] = mdl.predict_proba(X[te])[:, 1]
        print(f"    particion {k}: train {len(tr)} test {len(te)} ({time.time() - t0:.0f} s)", flush=True)
    print("[2] CV GroupKFold(5) por cumulo, umbral 0,5 (contra la etiqueta de §0.2)")
    line("todos", oof, y, grp)
    strata = {
        "logAge < 6,8": logt < 6.8, "6,8-7,3": (logt >= 6.8) & (logt < 7.3), "7,3-7,8": (logt >= 7.3) & (logt < 7.8),
        "7,8-8,5": (logt >= 7.8) & (logt < 8.5), ">= 8,5": logt >= 8.5,
        "d < 700 pc": dist < 700, "d 700-1600": (dist >= 700) & (dist < 1600), "d >= 1600": dist >= 1600,
        "AV < 0,6": av < 0.6, "AV 0,6-2,0": (av >= 0.6) & (av < 2.0), "AV >= 2,0": av >= 2.0,
        "sin NIR": ~nir, "con NIR": nir,
    }
    for k, s in strata.items():
        line(k, oof[s], y[s], grp[s])
    cell = (logt < 6.8) & (dist >= 700) & (dist < 1600) & (av >= 0.6) & (av < 2.0)
    print("  -- celda de NGC 6383 (logAge < 6,8; 700-1600 pc; AV 0,6-2,0):")
    line("celda NGC 6383", oof[cell], y[cell], grp[cell])
    line("celda NGC 6383, sin NIR", oof[cell & ~nir], y[cell & ~nir], grp[cell & ~nir])
    print("  calibracion (todos): bin de P -> fraccion PMS observada (n)")
    edges = np.linspace(0, 1, 11)
    row = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (oof >= lo) & (oof < hi + (hi == 1))
        row.append(f"{lo:.1f}-{hi:.1f}: {y[s].mean():.3f} ({s.sum()})")
    print("    " + "  ".join(row))
    row = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = cell & (oof >= lo) & (oof < hi + (hi == 1))
        row.append(f"{lo:.1f}-{hi:.1f}: {y[s].mean() if s.any() else np.nan:.3f} ({s.sum()})")
    print("    celda NGC 6383: " + "  ".join(row))

    # [3] Sagitta en las mismas estrellas
    import sagitta_ref as S
    plx = f(m["Plx"])
    ps, _ = S.sagitta_pms(f(m["GLON"]), f(m["GLAT"]), plx, ph["G"], ph["BP"], ph["RP"], ph["J"], ph["H"], ph["K"])
    print(f"[3] Sagitta 1.2.3 en las mismas estrellas (NaN -> token), contra la etiqueta de §0.2 ({time.time() - t0:.0f} s)")
    for th in (0.6, 0.5):
        line(f"Sagitta >= {th} todos", ps, y, grp, th)
        line(f"Sagitta >= {th} celda NGC 6383", ps[cell], y[cell], grp[cell], th)
    line("propio >= 0,5 celda NGC 6383", oof[cell], y[cell], grp[cell])

    # [4] FPR en cumulos viejos
    old = (logt >= 8.5) & (dist >= 600) & (dist <= 2500) & ~y
    print(f"[4] FPR en miembros de cumulos viejos (logAge >= 8,5; 600-2500 pc; etiqueta 0): n {old.sum()}, "
          f"{len(np.unique(grp[old]))} cumulos")
    for tag, p, th in (("propio >= 0,5", oof, 0.5), ("Sagitta >= 0,6", ps, 0.6), ("Sagitta >= 0,5", ps, 0.5)):
        k = int(np.sum(p[old] >= th))
        lo, hi = cp(k, old.sum())
        blo, bhi = cluster_boot(p[old], y[old], grp[old], "fpr", th=th)
        print(f"  {tag:16s} FP {k} de {old.sum()} = {k / old.sum():.4f} [{lo:.4f}; {hi:.4f}]; boot cumulos [{blo:.4f}; {bhi:.4f}]")
    oldfaint = old & (ph["G"] > 17)
    k = int(np.sum(oof[oldfaint] >= 0.5))
    print(f"  propio, sólo G > 17: {k} de {oldfaint.sum()} = {k / max(oldfaint.sum(), 1):.4f} {cp(k, oldfaint.sum())}")

    # [5] modelos finales: todos, y solo logAge < 8,3 (prior de mezcla de edades)
    mdl = core.fit(X, y)
    mdl.save_model(C / "model_all.json")
    yg = logt < 8.3
    mdl_y = core.fit(X[yg], y[yg])
    mdl_y.save_model(C / "model_young.json")
    print(f"[5] modelos finales guardados: todos ({len(y)} estrellas), logAge < 8,3 ({yg.sum()}); "
          f"sin semillas de NIR: aumentacion 25 % ({time.time() - t0:.0f} s)")
    np.savez_compressed(C / "oof.npz", sid=sid, oof=oof, sag=ps, y=y, logt=logt, dist=dist, av=av, grp=grp, nir=nir)
