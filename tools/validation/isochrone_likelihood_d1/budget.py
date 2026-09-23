"""Approximation-error budget of the forward model, at masses visible in the NGC 6383 window."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/notluquis/erotica/tools/validation")
from isochrone_nuts_convergence import MIST

from erotica.analysis._isochrone import (
    _Q_NODES,
    MISTIsochrones,
    _ccm89,
    _chabrier2014_xi,
    _mag_combine,
)

iso = MISTIsochrones(MIST)
kG = _ccm89(6390.7)
# visible: apparent G < 20.66 at the brightest prior shift (dm 9.5, Av 0.5) -> G_abs < 20.66 - 9.5 - kG*0.5
Gcut = 20.66 - 9.5 - kG * 0.5


def on_axis(z, a, ax):
    e, m, G, BP, RP = iso.get_isochrone_eep(z, a)
    return (
        np.interp(ax, e, m),
        np.interp(ax, e, G),
        np.interp(ax, e, BP - RP),
        (ax >= e.min()) & (ax <= e.max()),
    )


def loo(z0, z1, z, a0, a1, a, wz_log=True):
    e_all = iso.get_isochrone_eep(z, a)[0]
    ax = e_all.astype(float)
    A = {k: on_axis(*k, ax) for k in [(z0, a0), (z1, a0), (z0, a1), (z1, a1)]}
    wz = 0.0 if z0 == z1 else (np.log10(z) - np.log10(z0)) / (np.log10(z1) - np.log10(z0))
    wa = 0.0 if a0 == a1 else (a - a0) / (a1 - a0)
    W = {}
    for k, v in (
        ((z0, a0), (1 - wz) * (1 - wa)),
        ((z1, a0), wz * (1 - wa)),
        ((z0, a1), (1 - wz) * wa),
        ((z1, a1), wz * wa),
    ):
        W[k] = W.get(k, 0.0) + v
    G = sum(W[k] * A[k][1] for k in W)
    C = sum(W[k] * A[k][2] for k in W)
    ok = np.all([A[k][3] for k in W], axis=0)
    _, mt, Gt, BPt, RPt = iso.get_isochrone_eep(z, a)
    vis = ok & (Gt < Gcut)
    # compare at fixed EEP (same phase) -- residual in G and colour; weight by IMF mass step
    w = _chabrier2014_xi(np.maximum(mt, 1e-3)) * np.abs(np.gradient(mt))
    w = w[vis] / w[vis].sum()
    dG, dC = (G - Gt)[vis], (C - (BPt - RPt))[vis]

    def wq(x, p):
        o = np.argsort(np.abs(x))
        c = np.cumsum(w[o])
        return float(np.abs(x)[o][np.searchsorted(c, p)])

    return dict(
        wq50G=wq(dG, 0.5),
        wq68G=wq(dG, 0.68),
        wq90G=wq(dG, 0.9),
        wq68C=wq(dC, 0.68),
        wq90C=wq(dC, 0.9),
        rmsG=float(np.sqrt(np.sum(w * dG**2))),
        rmsC=float(np.sqrt(np.sum(w * dC**2))),
        p95G=float(np.percentile(np.abs(dG), 95)),
        p95C=float(np.percentile(np.abs(dC), 95)),
        n=int(vis.sum()),
    )


out = {}
Z = iso._met_values
A_ = iso._loga_values
for a in (6.15, 6.35, 6.55, 6.75, 6.95):
    out[f"age_loo_0.10dex_a{a}"] = loo(
        0.014286, 0.014286, 0.014286, round(a - 0.05, 2), round(a + 0.05, 2), a
    )
out["z_loo_0.5dex_Z0.0143_a6.55"] = loo(0.008033, 0.025404, 0.014286, 6.55, 6.55, 6.55)
out["z_loo_0.5dex_Z0.0080_a6.55"] = loo(0.004518, 0.014286, 0.008033, 6.55, 6.55, 6.55)
# q linearisation, at the node (Z 0.014286, a 6.55): exact combined photometry at q between nodes vs linear
e, m, G, BP, RP = iso.get_isochrone_eep(0.014286, 6.55)
ms = np.argsort(m, kind="stable")
qf = np.linspace(0, 1, 201)


def comb(q):
    m2 = np.clip(q * m, m.min(), None)
    G2, B2, R2 = (np.interp(m2, m[ms], x[ms]) for x in (G, BP, RP))
    return _mag_combine(G, G2), _mag_combine(BP, B2) - _mag_combine(RP, R2)


nodesG = np.array([comb(q)[0] for q in _Q_NODES])
nodesC = np.array([comb(q)[1] for q in _Q_NODES])
dG, dC = [], []
for q in qf:
    j = min(np.searchsorted(_Q_NODES, q, side="right") - 1, len(_Q_NODES) - 2)
    t = (q - _Q_NODES[j]) / (_Q_NODES[j + 1] - _Q_NODES[j])
    g, c = comb(q)
    dG.append((nodesG[j] + t * (nodesG[j + 1] - nodesG[j]) - g)[G < Gcut])
    dC.append((nodesC[j] + t * (nodesC[j + 1] - nodesC[j]) - c)[G < Gcut])
dGa = np.array(dG)
from erotica.analysis._isochrone import _dk_gamma

vis = G < Gcut
wm = (_chabrier2014_xi(np.maximum(m, 1e-3)) * np.abs(np.gradient(m)))[vis]
g1 = _dk_gamma(m[vis]) + 1
wqq = np.array([wm * (g1 * np.maximum(q, 1e-6) ** (g1 - 1)) for q in qf])  # IMF x f(q) density
wqq = (wqq / wqq.sum()).ravel()
aG = np.abs(dGa).ravel()
o = np.argsort(aG)
c = np.cumsum(wqq[o])
aC = np.abs(np.array(dC)).ravel()
oc = np.argsort(aC)
cc = np.cumsum(wqq[oc])
qlin_w = {f"wq{int(p * 100)}G": float(aG[o][np.searchsorted(c, p)]) for p in (0.5, 0.68, 0.9, 0.99)}
qlin_w.update({f"wq{int(p * 100)}C": float(aC[oc][np.searchsorted(cc, p)]) for p in (0.68, 0.9)})
dG, dC = np.abs(np.concatenate(dG)), np.abs(np.concatenate(dC))
bad = np.argwhere(np.abs(dGa) > 0.2)
print(
    "q-lin >0.2 mag: n",
    len(bad),
    "of",
    dGa.size,
    "q values",
    np.unique(qf[bad[:, 0]])[:10],
    "masses",
    np.unique(np.round(m[G < Gcut][bad[:, 1]], 2))[:20],
)
out["q_linearisation"] = dict(
    rmsG=float(np.sqrt(np.mean(dG**2))),
    rmsC=float(np.sqrt(np.mean(dC**2))),
    p95G=float(np.percentile(dG, 95)),
    p95C=float(np.percentile(dC, 95)),
)
out["q_linearisation_imf_x_fq_weighted"] = qlin_w
out["Gabs_visible_cut"] = Gcut
print(json.dumps(out, indent=1))
json.dump(out, open("budget.json", "w"), indent=1)
