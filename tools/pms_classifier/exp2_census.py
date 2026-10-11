"""Segundo experimento, paso 3: censo PMS de NGC 6383 con la configuracion elegida (§2.5) y la sensibilidad a las `U` de 2MASS (§2.6).

Sólo suma los tramos ESTABLECIDOS por exp2_confirm.py (G2); los demas se reportan aparte. Intervalo: B = 30 reentrenamientos con
cumulos de entrenamiento remuestreados; en cada uno, si la decision es D1, los umbrales por tramo se recalculan sobre los cumulos de la
mitad A que quedaron FUERA de ese remuestreo (fuera de bolsa); una realizacion de la calibracion CAL (parametrica + fotometrica) y
A_V, MOD ~ N(HR23 50, (84-16)/2). Sensibilidades (prior alfa, delta, kappa) fuera del IC, con umbrales recalculados en A con un modelo
entrenado sin los cumulos de A.
Prerregistro: ~/phd/agent-findings/pms-classifier-non-circular.md §2.5-§2.6 (hub 42b93a6).
Env: erotica. Corre:
  cd ~/erotica-wt-pms/tools/pms_classifier && SAGITTA_SRC=~/.local/lib/sagitta-1.2.3 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 nice -n 5 \
      ~/miniforge3/envs/erotica/bin/python exp2_census.py > exp2_census_measured.txt 2> cache/exp2_census.log
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.table import Table
from xgboost import XGBClassifier

import apply_ngc6383 as A
import core
import exp2_common as E
import exp2_grid as GR
from diag_identifiability import delta_th
from exp2_confirm import feats
from train_validate import f

C = Path(__file__).parent / "cache"
RNG = np.random.default_rng(6384)
h = Table.read(C / "ngc6383_hr23.ecsv")[0]
sAV, sMOD = (float(h["AV84"]) - float(h["AV16"])) / 2, (float(h["MOD84"]) - float(h["MOD16"])) / 2


def weights_for(d, cfg):
    y, stab = d["tg"][cfg["T"]]
    w = E.config_weights(cfg, y, stab, d["dobs"], d["dth"], d["logt"])
    trivial = cfg["P"] == "P0" and cfg["K"] == "K0" and cfg.get("alpha") is None
    return y, (None if trivial else w)


def Xof(d, cfg):
    return d["X1"] if cfg["F"] == "F1" else d["X"]


def fit_sub(d, cfg, s, seed):
    X = Xof(d, cfg)
    y, w = weights_for(d, cfg)
    return GR.fit_any(X[s], y[s], seed=seed, w=None if w is None else w[s], nfeat=X.shape[1])


def thresholds_A(d, cfg, mdl, Asel):
    if cfg["D"] == "D0":
        return np.full(len(E.TRAMOS), 0.5)
    p = mdl.predict_proba(Xof(d, cfg)[Asel])[:, 1]
    full = np.zeros(len(d["y0"]))
    full[Asel] = p
    return E.tramo_thresholds(full, d["mg0"], Asel, target_fpr=0.03)


def per_tramo(mdl, cfg, th, nir, av=A.AV, mod=A.MOD):
    X = feats(cfg, A.G, A.BP, A.RP, nir["J"], nir["H"], nir["Ks"], mod, av)
    dec = E.decide(mdl.predict_proba(X)[:, 1], X[:, 0], th)
    k = E.tramo_index(X[:, 0])
    return np.array([int(np.sum(dec & (k == i))) for i in range(len(E.TRAMOS))]), np.array(
        [int(np.sum(k == i)) for i in range(len(E.TRAMOS))])


if __name__ == "__main__":
    cj = json.load(open(C / "exp2_confirm.json"))
    key, cfg, th = cj["key"], cj["cfg"], np.array(cj["th"])
    est = np.array([cj["established"][nm] for nm in E.TRAMO_NAMES])
    print(f"[N0] configuracion {key}; veredicto §2.4 {'SIRVE' if cj['verdict'] else 'NO SIRVE'}; tramos establecidos "
          f"{[nm for nm, v in zip(E.TRAMO_NAMES, est) if v]}; umbrales {th.round(2).tolist()}")
    d = GR.prepare()
    mdl = XGBClassifier()
    mdl.load_model(C / f"exp2_model_{key}.json")

    print("[N1] censo determinista (CAL), por tramo de M_G,0: PMS / estrellas del tramo")
    for tag, nir in (("T2M", A.T2M), ("BEST", A.BEST), ("CAL", A.CAL)):
        n, tot = per_tramo(mdl, cfg, th, nir)
        sag = A.sag(nir) >= 0.6
        X = core.features(A.G, A.BP, A.RP, nir["J"], nir["H"], nir["Ks"], A.MOD, A.AV)
        k = E.tramo_index(X[:, 0])
        ns = [int(np.sum(sag & (k == i))) for i in range(len(E.TRAMOS))]
        print(f"  {tag:5s} elegida " + "  ".join(f"{nm} {a}/{b}" for nm, a, b in zip(E.TRAMO_NAMES, n, tot)) +
              f" | establecidos {int(n[est].sum())} | total {int(n.sum())} || Sagitta(0,6) " +
              "  ".join(f"{nm} {a}" for nm, a in zip(E.TRAMO_NAMES, ns)) + f" | en establecidos {int(np.sum(np.array(ns)[est]))}")
    lab = core.label(A.G, A.MOD, A.AV, float(h["logAge50"]))
    mg0 = A.G - A.MOD - core.EXT["G"] * A.AV
    dth = delta_th(mg0, np.full(len(mg0), float(h["logAge50"])))
    sep = lab & ~(np.isfinite(dth) & (dth <= E.DELTA))
    print(f"  referencia: §0.2 a la edad HR23 llama PMS a {lab.sum()} de {len(lab)}; con T1 (Delta_th > {E.DELTA}) quedan {sep.sum()} "
          f"(salen {int(lab.sum() - sep.sum())}); por tramo T1: " + "  ".join(
              f"{nm} {int(np.sum(sep & (E.tramo_index(mg0) == i)))}" for i, nm in enumerate(E.TRAMO_NAMES)))

    print("[N2] intervalo (B = 30): modelo (cumulos remuestreados, umbrales en A fuera de bolsa) + CAL + A_V/MOD", flush=True)
    ug = np.unique(d["grp"])
    idx = {c: np.where(d["grp"] == c)[0] for c in ug}
    tot, mod_only, avmod_only, cal_only = [], [], [], []
    for bb in range(30):
        pick = RNG.choice(ug, len(ug))
        s = np.concatenate([idx[c] for c in pick])
        mb = fit_sub(d, cfg, s, seed=2000 + bb)
        oob = d["A"] & ~np.isin(d["grp"], pick)
        thb = thresholds_A(d, cfg, mb, oob)
        nir = A.corrected(A.cal_draw(True), True)
        av, mod = A.AV + RNG.normal(0, sAV), A.MOD + RNG.normal(0, sMOD)
        tot.append(per_tramo(mb, cfg, thb, nir, av, mod)[0])
        mod_only.append(per_tramo(mb, cfg, thb, A.CAL)[0])
        avmod_only.append(per_tramo(mdl, cfg, th, A.CAL, av, mod)[0])
        cal_only.append(per_tramo(mdl, cfg, th, nir)[0])
        print(f"    b {bb}: oob A {len(np.unique(d['grp'][oob]))} cum, umbrales {thb.round(2).tolist()}, total {tot[-1].tolist()}",
              flush=True)
    for nm, v in (("solo modelo", mod_only), ("solo calibracion", cal_only), ("solo A_V y MOD", avmod_only), ("TOTAL", tot)):
        v = np.array(v)
        es = v[:, est].sum(1) if est.any() else np.zeros(len(v))
        print(f"  {nm:18s} establecidos mediana {np.median(es):.0f} [2,5; 97,5 %] = [{np.percentile(es, 2.5):.0f}; "
              f"{np.percentile(es, 97.5):.0f}] (68 %: [{np.percentile(es, 16):.0f}; {np.percentile(es, 84):.0f}]) | por tramo " +
              "  ".join(f"{t} {np.median(v[:, i]):.0f} [{np.percentile(v[:, i], 2.5):.0f}; {np.percentile(v[:, i], 97.5):.0f}]"
                        for i, t in enumerate(E.TRAMO_NAMES)))

    print("[N3] sensibilidad fuera del IC (CAL determinista; umbrales recalculados en A con un modelo entrenado sin A)", flush=True)
    notA = ~d["A"]
    sens = [("alpha 0,5", dict(cfg, alpha=0.5))] if cfg["P"] == "P1" else [("alpha 1 (P1)", dict(cfg, P="P1"))]
    sens += [("alpha 0 (P0)", dict(cfg, P="P0", alpha=None))]
    base_delta, base_kappa = E.DELTA, E.KAPPA
    for nm, c in sens + [("referencia (misma config)", dict(cfg))]:
        mA = fit_sub(d, c, notA, seed=3000)
        thA = thresholds_A(d, c, mA, d["A"])
        mf = fit_sub(d, c, np.ones(len(d["y0"]), bool), seed=3001)
        n, _ = per_tramo(mf, c, thA, A.CAL)
        print(f"  {nm:26s} umbrales {thA.round(2).tolist()} | " + "  ".join(f"{t} {a}" for t, a in zip(E.TRAMO_NAMES, n)) +
              f" | establecidos {int(n[est].sum())}", flush=True)
    for par, vals in (("delta", (0.25, 0.75)), ("kappa", (0.75, 1.5))):
        if (par == "delta" and cfg["T"] != "T1") or (par == "kappa" and cfg["K"] != "K2"):
            continue
        for v in vals:
            E.DELTA, E.KAPPA = (v, base_kappa) if par == "delta" else (base_delta, v)
            d2 = GR.prepare()
            mA = fit_sub(d2, cfg, ~d2["A"], seed=3000)
            thA = thresholds_A(d2, cfg, mA, d2["A"])
            mf = fit_sub(d2, cfg, np.ones(len(d2["y0"]), bool), seed=3001)
            n, _ = per_tramo(mf, cfg, thA, A.CAL)
            print(f"  {par} {v:<22} umbrales {thA.round(2).tolist()} | " + "  ".join(f"{t} {a}" for t, a in zip(E.TRAMO_NAMES, n)) +
                  f" | establecidos {int(n[est].sum())}", flush=True)
        E.DELTA, E.KAPPA = base_delta, base_kappa

    print("[N4] las `U` de 2MASS: censo en las entradas T2M (2MASS de la tabla limpia), 2 clasificadores x 2 codificaciones")
    T2L = {b: np.where(np.isfinite(A.M2[b]) & (A.Q[b] != ""), A.M2[b], np.nan) for b in A.BANDS}
    nU = {b: int(np.sum(A.Q[b] == "U")) for b in A.BANDS}
    print(f"  estrellas con `U` por banda (J, H, Ks): {nU}; con alguna `U`: "
          f"{int(np.sum((A.Q['J'] == 'U') | (A.Q['H'] == 'U') | (A.Q['Ks'] == 'U')))} de {len(A.G)}")
    for nm, nir in (("U -> faltante", A.T2M), ("U como valor", T2L)):
        n, _ = per_tramo(mdl, cfg, th, nir)
        n0 = int(np.sum(A.own(nir) >= 0.5))
        print(f"  {nm:14s} elegida: " + "  ".join(f"{t} {a}" for t, a in zip(E.TRAMO_NAMES, n)) +
              f" | establecidos {int(n[est].sum())} total {int(n.sum())} || modelo del F2 (0,5) {n0} || Sagitta(0,6) "
              f"{int(np.sum(A.sag(nir) >= 0.6))}")
