"""Nucleo del clasificador PMS no circular: etiqueta por isocrona, features desenrojecidas por cumulo, modelo XGBoost.

Prerregistro: ~/phd/agent-findings/pms-classifier-non-circular.md §0. Cada pieza, con el proceso que modela:

- ETIQUETA (§0.2). PMS = masa inicial bajo la de llegada a la ZAMS a la edad del cumulo. MIST v1.2, [Fe/H] = 0, v/vcrit = 0,4
  (UBVRIplus, en disco en ~/erotica/data/test/NGC6383/MIST). M_G,on(t) = Gaia_G_EDR3 del punto mas masivo con phase = -1; la
  estrella es PMS si M_G,0 > M_G,on(t). A edades sin puntos phase = -1, M_G,on = +inf (nada es PMS). Interpolada en log t.
- FEATURES (§0.3). Distancia y A_V DEL CUMULO (HR23 MOD50, AV50), no de la estrella; sin paralaje ni posicion: el modelo ve la
  posicion en el CMD absoluto desenrojecido. La extincion diferencial de cada cumulo queda como ruido real en el entrenamiento.
  Cocientes A_lambda/A_V: Wang & Chen 2019 (arXiv:1904.04575, Tabla 3; verificados 2026-10-11).
- FALTANTES. NaN nativo de XGBoost. 2MASS con ph_qual fuera de A-E -> NaN. Aumentacion: al 25 % de las filas de entrenamiento se
  les borra el NIR entero al azar, independiente de la magnitud (en NGC 6383 el NIR falta por posicion/crowding, no por profundidad).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

EXT = {"G": 0.789, "BP": 1.002, "RP": 0.589, "J": 0.243, "H": 0.131, "Ks": 0.078}
FEATURES = ["MG0", "BPRP0", "GRP0", "MKs0", "GKs0", "JH0", "HKs0"]
NIR_FEATURES = ["MKs0", "GKs0", "JH0", "HKs0"]
MIST = Path.home() / "erotica/data/test/NGC6383/MIST/UBVRIplus/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd"
XGB_PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, tree_method="hist", n_jobs=4, eval_metric="logloss", random_state=20261011)


def _read_mist(path=MIST):
    rows, cols = [], None
    for line in open(path):
        if line.startswith("# EEP"):
            cols = line[1:].split()
        elif line.startswith("#") or not line.strip():
            continue
        else:
            rows.append([float(x) for x in line.split()])
    a = np.array(rows)
    return {c: a[:, i] for i, c in enumerate(cols)}


def turn_on_table(path=MIST):
    """(log t, M_G,on, M_on) por isocrona de la grilla. M_G,on = +inf si no hay puntos PMS."""
    m = _read_mist(path)
    out = []
    for t in np.unique(m["log10_isochrone_age_yr"]):
        s = m["log10_isochrone_age_yr"] == t
        pms = s & (m["phase"] == -1)
        if pms.any():
            k = np.where(pms)[0][np.argmax(m["initial_mass"][pms])]
            out.append((t, m["Gaia_G_EDR3"][k], m["initial_mass"][k]))
        else:
            out.append((t, np.inf, 0.0))
    return np.array(out)


_TON = None


def mg_turn_on(logt):
    """M_G,on interpolado en log t (inf donde la grilla ya no tiene PMS)."""
    global _TON
    if _TON is None:
        _TON = turn_on_table()
    t, mg = _TON[:, 0], _TON[:, 1]
    fin = np.isfinite(mg)
    tmax = t[fin].max()
    logt = np.asarray(logt, float)
    out = np.interp(logt, t[fin], mg[fin])
    return np.where(logt > tmax, np.inf, out)


def nir_clean(mag, qual_char):
    """Una banda 2MASS vale si ph_qual en A-E; si no, NaN."""
    q = np.asarray(qual_char).astype(str)
    return np.where(np.isin(q, list("ABCDE")) & np.isfinite(mag), mag, np.nan)


def features(G, BP, RP, J, H, Ks, mod, av):
    A = {k: v * np.asarray(av, float) for k, v in EXT.items()}
    f = {
        "MG0": G - mod - A["G"],
        "BPRP0": (BP - RP) - (A["BP"] - A["RP"]),
        "GRP0": (G - RP) - (A["G"] - A["RP"]),
        "MKs0": Ks - mod - A["Ks"],
        "GKs0": (G - Ks) - (A["G"] - A["Ks"]),
        "JH0": (J - H) - (A["J"] - A["H"]),
        "HKs0": (H - Ks) - (A["H"] - A["Ks"]),
    }
    return np.stack([np.asarray(f[k], float) for k in FEATURES], 1)


def label(G, mod, av, logt):
    return (np.asarray(G) - mod - EXT["G"] * np.asarray(av)) > mg_turn_on(logt)


def drop_nir(X, frac, rng):
    """Borra el NIR entero de una fraccion `frac` de filas, al azar e independiente de la magnitud."""
    X = X.copy()
    k = rng.random(len(X)) < frac
    for c in NIR_FEATURES:
        X[k, FEATURES.index(c)] = np.nan
    return X


def fit(X, y, frac_drop=0.25, seed=20261011, params=None):
    from xgboost import XGBClassifier
    rng = np.random.default_rng(seed)
    m = XGBClassifier(**(params or XGB_PARAMS))
    m.fit(drop_nir(X, frac_drop, rng), y.astype(int))
    return m
