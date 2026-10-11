"""Diagnostico previo al segundo prerregistro (corrido ANTES de escribirlo y antes de entrenar nada nuevo; no esta ciego: F2 y los
post hoc ya se vieron). Pregunta: en el tramo debil, ¿la etiqueta de §0.2 es identificable con la fotometria, o la fija la edad del
cumulo?

Mide, sin entrenar:
 [D1] desplazamiento TEORICO sobre la ZAMS de cada estrella con etiqueta 1: Delta_th = M_G,ZAMS(color de la isocrona del cumulo a su
      M_G,0) - M_G,0 (positivo = sobre la ZAMS). ZAMS = puntos EEP 202 de MIST v1.2 (la misma grilla de la etiqueta), uno por masa.
      Fraccion de etiquetas 1 con Delta_th < delta, por tramo de M_G,0 y por log t del cumulo.
 [D2] tasa base de la etiqueta 1 entre estrellas OBSERVADAS cerca de la ZAMS (|Delta_obs| < 0,5 mag, Delta_obs con su (BP-RP)_0),
      por tramo, bajo tres priors de edad: la mezcla del catalogo, uniforme en log t sobre [6,6; 9,4] y sobre [6,6; 9,0]. Es lo que
      un clasificador sin edad puede devolver ahi como maximo de informacion: si queda cerca de 0,5, decide el prior.
Env: erotica. Corre: cd ~/erotica-wt-pms/tools/pms_classifier && ~/miniforge3/envs/erotica/bin/python diag_identifiability.py \
     > diag_identifiability_measured.txt
"""
from __future__ import annotations

import numpy as np

import core
from train_validate import build, f, load

TRAMOS = [(-5, 1), (1, 3), (3, 5), (5, 7), (7, 12)]


def zams_locus():
    m = core._read_mist()
    z = m["EEP"] == 202
    c = m["Gaia_BP_EDR3"][z] - m["Gaia_RP_EDR3"][z]
    g = m["Gaia_G_EDR3"][z]
    o = np.argsort(c)
    return c[o], g[o]


ZC, ZG = zams_locus()


def mg_zams(bprp0):
    return np.interp(bprp0, ZC, ZG, left=np.nan, right=np.nan)


def iso_pms_tables():
    """Por edad de la grilla: (M_G, BP-RP) de los puntos PMS (phase = -1), ordenados por M_G."""
    m = core._read_mist()
    out = {}
    for t in np.unique(m["log10_isochrone_age_yr"]):
        s = (m["log10_isochrone_age_yr"] == t) & (m["phase"] == -1)
        if s.sum() < 3:
            continue
        g, c = m["Gaia_G_EDR3"][s], m["Gaia_BP_EDR3"][s] - m["Gaia_RP_EDR3"][s]
        o = np.argsort(g)
        out[round(float(t), 2)] = (g[o], c[o])
    return out


ISO = iso_pms_tables()
TGRID = np.array(sorted(ISO))


def delta_th(mg0, logt):
    """Desplazamiento teorico sobre la ZAMS a la edad del cumulo (NaN si la isocrona no tiene PMS a esa M_G)."""
    mg0, logt = np.asarray(mg0, float), np.asarray(logt, float)
    out = np.full(len(mg0), np.nan)
    near = TGRID[np.abs(TGRID[None, :] - logt[:, None]).argmin(1)]
    for t in np.unique(near):
        s = near == t
        g, c = ISO[t]
        col = np.interp(mg0[s], g, c, left=np.nan, right=np.nan)
        out[s] = mg_zams(col) - mg0[s]
    return out


if __name__ == "__main__":
    m = load()
    m = m[m["role"] == "train"]
    X, y, ph = build(m)
    logt = f(m["logAge50"])
    mg0, bprp0 = X[:, 0], X[:, 1]
    dth = delta_th(mg0, logt)
    dobs = mg_zams(bprp0) - mg0
    print(f"ZAMS (EEP 202): {len(ZC)} puntos, (BP-RP)_0 {ZC.min():.2f}..{ZC.max():.2f}; isocronas con PMS: {len(TGRID)} edades")
    print(f"[D1] Delta_th de las etiquetas 1 (n {y.sum()}; Delta_th finito en {np.isfinite(dth[y]).sum()})")
    lbins = [(6.0, 6.8), (6.8, 7.3), (7.3, 7.8), (7.8, 8.2), (8.2, 8.5), (8.5, 9.5)]
    for lo, hi in TRAMOS:
        s = y & (mg0 >= lo) & (mg0 < hi)
        cells = []
        for a, b in lbins:
            k = s & (logt >= a) & (logt < b)
            if k.sum() >= 20:
                d = dth[k]
                cells.append(f"lt {a}-{b}: n {k.sum():5d} med {np.nanmedian(d):+.2f} <0,5 {np.mean(d < 0.5):.2f}")
        print(f"  M_G,0 {lo:+d} a {hi:+d}: " + " | ".join(cells))
    for dl in (0.25, 0.5, 0.75, 1.0):
        print(f"  delta {dl}: etiquetas 1 con Delta_th < delta (o NaN): {np.mean(~(dth[y] >= dl)):.3f}")

    print("[D2] tasa base de la etiqueta 1 entre estrellas observadas cerca de la ZAMS (|Delta_obs| < 0,5), por prior de edad")
    hist_edges = np.arange(6.0, 10.01, 0.2)

    def wts(lo, hi):
        if lo is None:
            return np.ones(len(logt))
        w = np.zeros(len(logt))
        idx = np.digitize(logt, hist_edges)
        for i in np.unique(idx):
            s = idx == i
            c = 0.5 * (hist_edges[max(i - 1, 0)] + hist_edges[min(i, len(hist_edges) - 1)])
            if lo <= c <= hi:
                w[s] = 1.0 / s.sum()
        return w

    priors = {"catalogo": wts(None, None), "uniforme log t [6,6; 9,4]": wts(6.6, 9.4), "uniforme log t [6,6; 9,0]": wts(6.6, 9.0)}
    for lo, hi in TRAMOS:
        s = (mg0 >= lo) & (mg0 < hi) & (np.abs(dobs) < 0.5)
        row = []
        for k, w in priors.items():
            ww = w[s]
            row.append(f"{k}: {np.sum(ww * y[s]) / max(np.sum(ww), 1e-300):.3f}")
        print(f"  M_G,0 {lo:+d} a {hi:+d} (n {s.sum():6d}): " + "  ".join(row))
    edad = [(6.0, 7.0), (7.0, 7.6), (7.6, 8.0), (8.0, 8.4), (8.4, 10.0)]
    print("  composicion por edad del tramo +7 a +12 cerca de la ZAMS (catalogo): " + ", ".join(
        f"lt {a}-{b}: {np.sum((mg0 >= 7) & (np.abs(dobs) < .5) & (logt >= a) & (logt < b))} "
        f"(PMS {np.mean(y[(mg0 >= 7) & (np.abs(dobs) < .5) & (logt >= a) & (logt < b)]):.2f})" for a, b in edad))

    print("[D3] etiquetas 1 cuya posicion observada contradice la isocrona del cumulo (Delta_obs < Delta_th - kappa), por tramo;"
          " y tasa base cerca de la ZAMS (|Delta_obs| < 0,5) tras: objetivo separable (Delta_th > delta) y/o quitar esas etiquetas")
    for kap in (0.75, 1.0, 1.5):
        inc = y & (dobs < dth - kap)
        print(f"  kappa {kap}: " + ", ".join(
            f"{lo:+d}..{hi:+d}: {np.sum(inc & (mg0 >= lo) & (mg0 < hi))}/{np.sum(y & (mg0 >= lo) & (mg0 < hi))}" for lo, hi in TRAMOS))
    for dl in (0.0, 0.5):
        for kap in (None, 1.0):
            ysep = y & (dth > dl) if dl > 0 else y.copy()
            keep = ~(y & (dobs < dth - kap)) if kap is not None else np.ones(len(y), bool)
            row = []
            for lo, hi in TRAMOS[2:]:
                s = keep & (mg0 >= lo) & (mg0 < hi) & (np.abs(dobs) < 0.5)
                for k, w in priors.items():
                    if k.startswith("uniforme log t [6,6; 9,0]"):
                        continue
                    ww = w[s]
                    row.append(f"{lo:+d}..{hi:+d} {k[:9]} {np.sum(ww * ysep[s]) / max(np.sum(ww), 1e-300):.3f}")
            print(f"  delta {dl} kappa {kap}: " + "  ".join(row))
