"""D1 arm (2b): unbinned, q-marginalised binaries (exact along q), free intrinsic jitter."""

import json
import sys
import time

import numpy as np

HERE = __import__("pathlib").Path(__file__).resolve().parent
from d1_proto import (
    AGES,
    KC,
    KG,
    NODES,
    ZS,
    Data,
    Phi,
    _chabrier2014_weights,
    _mag_combine,
    _seg_density,
    bracket,
    synthetic_cluster,
)
from scipy.optimize import minimize

QN = np.array([0.0, 0.3, 0.5, 0.65, 0.75, 0.85, 0.93, 1.0])


def gamma_dk(m):
    # D&K (2013) power-law index, as the generator uses it (step function)
    return np.where(
        m <= 0.1,
        4.2,
        np.where(m <= 0.6, 0.4, np.where(m <= 1.4, 0.3, np.where(m <= 6.5, -0.5, 0.0))),
    )


_bc = {}


def node_q(key, e_lo, e_hi):
    ck = (key, e_lo, e_hi)
    if ck not in _bc:
        eep, mass, G, col, BP, RP = NODES[key]
        rows = []
        for q in QN:
            M2 = np.clip(q * mass, mass[0], None)
            G2, BP2, RP2 = (np.interp(M2, mass, a) for a in (G, BP, RP))
            Gb = _mag_combine(G, G2) if q > 0 else _mag_combine(G, G2)  # q=0 -> 0.1 Msun companion
            cb = _mag_combine(BP, BP2) - _mag_combine(RP, RP2)
            rows += [Gb, cb]
        ax = np.arange(e_lo, e_hi + 1)
        _bc[ck] = np.stack([np.interp(ax, eep, a) for a in [mass, G, col] + rows])
    return _bc[ck]


def iso_q(met, loga, zs=ZS, ages=AGES):
    i, wz = bracket(np.log10(zs), np.log10(met))
    j, wa = bracket(ages, loga)
    keys = [(zs[i], ages[j]), (zs[i + 1], ages[j]), (zs[i], ages[j + 1]), (zs[i + 1], ages[j + 1])]
    e_lo = min(NODES[k][0][0] for k in keys)
    e_hi = max(NODES[k][0][-1] for k in keys)
    A = [node_q(k, e_lo, e_hi) for k in keys]
    X = (1 - wz) * (1 - wa) * A[0] + wz * (1 - wa) * A[1] + (1 - wz) * wa * A[2] + wz * wa * A[3]
    mass, G, col = X[0], X[1], X[2]
    Gq, cq = X[3::2], X[4::2]  # (nq, n)
    return mass, G, col, Gq, cq


def loglike(D, th, binaries=True, zs=ZS, ages=AGES):
    met, loga, dm, Av, sint = th
    mass, G, col, Gq, cq = iso_q(met, loga, zs, ages)
    w = _chabrier2014_weights(mass)
    bp = np.clip(0.09 + 0.94 / (1 + 1.4 / mass), 0, 1) * float(binaries)
    sG = np.sqrt(D.eG**2 + sint**2)
    sC = np.sqrt(D.eC**2 + sint**2)
    sh_m, sh_c = dm + KG * Av, KC * Av
    Ga, ca = G + sh_m, col + sh_c
    ws = 0.5 * (w[1:] * (1 - bp[1:]) + w[:-1] * (1 - bp[:-1]))
    f = _seg_density(D.G, D.c, sG, sC, Ga[:-1], ca[:-1], Ga[1:], ca[1:]) @ ws
    em, _ = D.sig(Ga)
    pin = Phi((D.Glim - Ga) / np.sqrt(em**2 + sint**2))
    F = np.sum(ws * 0.5 * (pin[1:] + pin[:-1]))
    if binaries:
        g1 = gamma_dk(mass) + 1.0
        Fq = QN[:, None] ** g1[None]  # CDF of q at the nodes, per primary
        wq = np.diff(Fq, axis=0) * (w * bp)[None]  # (nq-1, n)
        Gb, cb = Gq + sh_m, cq + sh_c
        for k in range(len(QN) - 1):
            f = f + _seg_density(D.G, D.c, sG, sC, Gb[k], cb[k], Gb[k + 1], cb[k + 1]) @ wq[k]
            Gm = 0.5 * (Gb[k] + Gb[k + 1])
            em, _ = D.sig(Gm)
            F = F + np.sum(wq[k] * Phi((D.Glim - Gm) / np.sqrt(em**2 + sint**2)))
    return float(np.sum(np.log(np.maximum(f, 1e-300))) - len(D.G) * np.log(F))


LO = np.array([ZS[0] * 1.001, 6.0, 9.5, 0.5, 0.002])
HI = np.array([ZS[-1] * 0.999, 7.0, 10.7, 2.0, 0.3])


def mle(D, starts, **kw):
    def nll(x):
        if np.any(x < LO) or np.any(x > HI):
            return 1e30
        return -loglike(D, x, **kw)

    best = None
    for s in starts:
        r = minimize(
            nll, s, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-4, maxiter=3000)
        )
        r = minimize(
            nll, r.x, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-5, maxiter=3000)
        )
        if best is None or r.fun < best.fun:
            best = r
    return best.x, -best.fun


if __name__ == "__main__":
    batch = sys.argv[1]
    cfg = {"singles": (10.47, 1.10, False), "binaries": (10.45, 1.24, True)}[batch]
    dm, Av, b = cfg
    ages_out = AGES[~np.isclose(AGES, 6.55)]
    res = []
    for seed in (1, 2, 3):
        t0 = time.time()
        D = Data(synthetic_cluster(0.014286, 6.55, dm, Av, seed, binaries=b))
        truth = np.array([0.014286, 6.55, dm, Av, 0.01])
        row = {"seed": seed, "ll_truth": loglike(D, truth, binaries=b)}
        for tag, kw in (("full", {}), ("age_node_out", {"ages": ages_out})):
            xh, lh = mle(D, [truth, np.array([0.0143, 6.5, 10.3, 1.25, 0.05])], binaries=b, **kw)
            row[tag] = {"mle": np.round(xh, 5).tolist(), "ll": round(lh, 2)}
        row["s"] = round(time.time() - t0)
        print(batch, row, flush=True)
        res.append(row)
        json.dump(res, open(HERE / f"d1_2b_{batch}.json", "w"), indent=1)
