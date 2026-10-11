"""Segundo experimento (prerregistro: ~/phd/agent-findings/pms-classifier-non-circular.md §2). Piezas comunes de la grilla.

Cada factor, con el proceso que modela (§2.2 del prerregistro):
  T objetivo   T0 = etiqueta de §0.2 (masa bajo la de llegada a la ZAMS a la edad del cumulo).
               T1 = PMS SEPARABLE: T0 y ademas desplazamiento TEORICO sobre la ZAMS Delta_th > DELTA (0,5 mag) a la edad y la
                    M_G,0 de la estrella. Estimando nuevo: "PMS que la fotometria puede ver"; una K de 100 Myr a 0,1 mag de la
                    ZAMS es PMS por T0 y no por T1. Delta_th NaN (fuera de la tabla de la isocrona) -> se queda con T0.
  P prior edad P0 = mezcla del catalogo (lo que hizo el F2).  P1 = uniforme en log t sobre el soporte [6,4; 9,6): peso por estrella
               (1/N_bin)^ALPHA con bins de 0,2 dex (9,2-9,6 juntos: 9 cumulos), ALPHA = 1; sensibilidad ALPHA = 0,5.
  K etiquetas  K0 = nada.  K1 = peso por estabilidad: fraccion de {logAge16, 50, 84} que da la misma etiqueta del objetivo.
               K2 = K1 y peso 0 a las etiquetas 1 cuya posicion observada contradice la isocrona del cumulo:
                    Delta_obs < Delta_th - KAPPA (1,0 mag): ni binaria ni extincion diferencial la bajan tanto [I]; son
                    contaminantes o una edad de catalogo equivocada. Es SELECCION EN LAS FEATURES: se declara su costo (recall
                    de PMS reales mal desenrojecidas) y lo mide NGC 6530.
  F features   F0 = las 7 de §0.3.  F1 = + altura sobre la ZAMS en (BP-RP)_0 y en (G-Ks)_0 (Delta_obs en G y en Ks).
  D decision   D0 = umbral 0,5.  D1 = umbral por tramo de M_G,0 elegido en la mitad de AJUSTE de los cumulos viejos para FPR <= 0,05
               (umbral max 0,99); se aplica tal cual en la mitad CONFIRMATORIA, NGC 6530 y NGC 6383.
"""
from __future__ import annotations

import numpy as np

import core
from diag_identifiability import delta_th, mg_zams

DELTA, KAPPA = 0.5, 1.0
TRAMOS = [(-5.0, 3.0), (3.0, 5.0), (5.0, 7.0), (7.0, 12.0)]
TRAMO_NAMES = ["-5..+3", "+3..+5", "+5..+7", "+7..+12"]
# fraccion de los 241 PMS-por-definicion de NGC 6383 en cada tramo (posthoc2_measured.txt [Q4], "peso NGC 6383")
W6383 = np.array([0.091, 0.232, 0.444, 0.232])
SPLIT_SEED = 20261012
FEATURES_F1 = core.FEATURES + ["DZG", "DZK"]


def _zams_k():
    m = core._read_mist()
    z = m["EEP"] == 202
    c = m["Gaia_G_EDR3"][z] - m["2MASS_Ks"][z]
    k = m["2MASS_Ks"][z]
    o = np.argsort(c)
    return c[o], k[o]


ZKC, ZKM = _zams_k()


def extra_features(X):
    """Altura observada sobre la ZAMS (positivo = mas brillante que la ZAMS a ese color), en G y en Ks."""
    mg0, bprp0 = X[:, core.FEATURES.index("MG0")], X[:, core.FEATURES.index("BPRP0")]
    mk0, gk0 = X[:, core.FEATURES.index("MKs0")], X[:, core.FEATURES.index("GKs0")]
    dzg = mg_zams(bprp0) - mg0
    dzk = np.interp(gk0, ZKC, ZKM, left=np.nan, right=np.nan) - mk0
    return np.column_stack([X, dzg, dzk])


def tramo_index(mg0):
    k = np.full(len(mg0), -1)
    for i, (lo, hi) in enumerate(TRAMOS):
        k[(mg0 >= lo) & (mg0 < hi)] = i
    return k


def targets(G, mod, av, logt16, logt50, logt84, mg0):
    """Etiquetas T0 y T1 a las tres edades; devuelve dict {T: (y50, estabilidad)} y Delta_th a logAge50."""
    out = {}
    dth = {}
    for tag, lt in (("16", logt16), ("50", logt50), ("84", logt84)):
        dth[tag] = delta_th(mg0, lt)
    for T in ("T0", "T1"):
        ys = []
        for tag, lt in (("16", logt16), ("50", logt50), ("84", logt84)):
            y = core.label(G, mod, av, lt)
            if T == "T1":
                d = dth[tag]
                y = y & ~(np.isfinite(d) & (d <= DELTA))
            ys.append(y)
        stab = (ys[0] == ys[1]).astype(float) / 3 + 1 / 3 + (ys[2] == ys[1]).astype(float) / 3
        out[T] = (ys[1], stab)
    return out, dth["50"]


def age_weights(logt, alpha):
    if alpha == 0:
        return np.ones(len(logt))
    edges = np.array([6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.6])
    idx = np.clip(np.digitize(logt, edges) - 1, 0, len(edges) - 2)
    n = np.bincount(idx, minlength=len(edges) - 1).astype(float)
    w = (1.0 / np.maximum(n[idx], 1)) ** alpha
    return w / w.mean()


def kish(w):
    return w.sum() ** 2 / np.sum(w ** 2)


def config_weights(cfg, y, stab, dobs, dth, logt):
    w = age_weights(logt, 1.0 if cfg["P"] == "P1" else 0.0)
    if cfg.get("alpha") is not None:
        w = age_weights(logt, cfg["alpha"])
    if cfg["K"] in ("K1", "K2"):
        w = w * stab
    if cfg["K"] == "K2":
        w = w * ~(y & np.isfinite(dth) & (dobs < dth - KAPPA))
    return w


def old_split(grp, old):
    """Mitades de AJUSTE (A) y CONFIRMATORIA (B) de los cumulos viejos, por cumulo, semilla fija."""
    u = np.unique(grp[old])
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(u)
    a = set(perm[: len(u) // 2])
    inA = np.array([g in a for g in grp]) & old
    inB = old & ~inA
    return inA, inB


def tramo_thresholds(p, mg0, sel, target_fpr=0.05, tmax=0.99):
    """Umbral por tramo: el menor t en una grilla fina con FPR <= target en `sel` (todas etiqueta 0)."""
    k = tramo_index(mg0)
    th = np.full(len(TRAMOS), 0.5)
    grid = np.round(np.arange(0.5, tmax + 1e-9, 0.01), 2)
    for i in range(len(TRAMOS)):
        s = sel & (k == i)
        if s.sum() == 0:
            continue
        for t in grid:
            th[i] = t
            if np.mean(p[s] >= t) <= target_fpr:
                break
    return th


def decide(p, mg0, th):
    k = tramo_index(mg0)
    t = np.where(k >= 0, th[np.clip(k, 0, None)], 0.5)
    return p >= t
