"""Shared pieces of the interval experiment (hub finding ``isochrone-intervals-nuts.md``).

Every choice carries the process it models (methodology §A.1):

* **Model**: the one of the external-constraints X1 fit (``isochrone_external/real_fits.py``), i.e.
  ``IsochroneFitter`` on the grid layer, MIST v1.2 [Fe/H] nodes -1.0...+0.5 (uniform in [Fe/H]),
  log t U(5.7, 7.0), A_V U(0, 3), dm ~ N(10.223, 0.026) truncated to [9.5, 10.7] (the members' mean
  parallax with the 10 uas floor), sigma_int HalfNormal(0.05), f_bg Beta(1, 19). Unchanged: this
  experiment is about the *estimator* (MAP + Laplace against a sampled posterior), not the model.
* **Data**: ``members254_gaia.ecsv`` (P01's 254 reference members, Gaia DR3 photometry; rebuilt by
  ``rebuild_sample.py``, X1 reproduced to the third decimal).
* **Generator** (``draw_synthetic``): the fitter's own forward model (``_deposit``: Chabrier IMF,
  Offner binaries with Duchene & Kraus q, EEP interpolation), with
  - the **real catalogue's heteroscedastic errors**: each synthetic star takes (e_G, e_col) of a real
    member drawn among the 3 nearest in G to its *true* G (Gaia errors scatter at fixed G; the
    fitter's own ``_draw_stars`` uses a polynomial in G instead);
  - noise variance e^2 + SIGMA_FLOOR^2 + sigma_int^2 in G and in colour, as the likelihood assumes;
  - a **field component**: Binomial(N, f_bg) stars uniform over the real members' CMD box, the same
    process as the likelihood's f_bg / A term;
  - the completeness cut at the faintest real member (G 20.66), as ``_draw_stars``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "isochrone_colour"))
sys.path.insert(0, str(HERE.parent / "isochrone_external"))
from common import EDR3, PRI_PLX, mist, search  # noqa: E402

SAMPLE = HERE / "members254_gaia.ecsv"
PARAMS = ("feh", "loga", "dm", "Av", "sigma_int", "f_bg")
X1 = {  # isochrone-external-constraints.md §6, reproduced on SAMPLE (commit 4a519ef)
    "feh": -0.4723,
    "loga": 6.0232,
    "dm": 10.2239,
    "Av": 0.6614,
    "sigma_int": 0.1527,
    "f_bg": 0.1699,
}


def real_table():
    from astropy.table import QTable, Table

    return QTable(Table.read(SAMPLE))


def fitter(data):
    from erotica.analysis._isochrone import IsochroneFitter

    f = IsochroneFitter(grid=mist(), magnitude=EDR3[0], color=EDR3[1:], **PRI_PLX)
    f.setup(data, prob_threshold=0.0)
    return f


def table(G, C, eG, eC):
    from astropy.table import QTable

    return QTable(
        {
            "Gmag": G,
            "G_BPmag": C,
            "G_RPmag": np.zeros_like(G),
            "e_Gmag": eG,
            "e_G_BPmag": eC,
            "e_G_RPmag": np.zeros_like(G),
            "e_BP_RP": eC,
            "probability_hdbscan": np.ones_like(G),
        }
    )


def _true_stars(gen, feh, loga, dm, Av, n, rng):
    """``n`` noiseless model stars (G, colour) from the generator fitter's forward model, the same
    draw as ``IsochroneFitter._draw_stars`` before its noise."""
    from erotica.analysis._isochrone import _Q_NODES, _dk_gamma

    d = gen._deposit(feh, loga, dm, Av, np)
    g1 = _dk_gamma(d["m_b"]) + 1.0
    w_single = d["w_single"]
    p = np.concatenate([w_single, sum(d["w_bin"])])
    p = np.clip(p, 0, None) / np.clip(p, 0, None).sum()
    k = rng.choice(len(p), size=n, p=p)
    single = k < len(w_single)
    G = np.empty(n)
    C = np.empty(n)
    ks, t = k[single], rng.uniform(size=int(single.sum()))
    G[single] = d["G"][ks] + t * (d["G"][ks + 1] - d["G"][ks])
    C[single] = d["col"][ks] + t * (d["col"][ks + 1] - d["col"][ks])
    kb = k[~single] - len(w_single)
    q = rng.uniform(size=kb.size) ** (1.0 / g1[kb])
    j = np.clip(np.searchsorted(_Q_NODES, q, side="right") - 1, 0, len(_Q_NODES) - 2)
    fq = (q - _Q_NODES[j]) / (_Q_NODES[j + 1] - _Q_NODES[j])
    G[~single] = d["Gq"][j, kb] + fq * (d["Gq"][j + 1, kb] - d["Gq"][j, kb])
    C[~single] = d["cq"][j, kb] + fq * (d["cq"][j + 1, kb] - d["cq"][j, kb])
    return G, C


def _real_errors(Greal, eGreal, eCreal, Gt, rng, k=3):
    order = np.argsort(Greal)
    gs = Greal[order]
    pos = np.clip(np.searchsorted(gs, Gt), 0, len(gs) - 1)
    out = np.empty(Gt.size, int)
    for i, (p, g) in enumerate(zip(pos, Gt, strict=True)):
        lo, hi = max(0, p - k), min(len(gs), p + k)
        cand = np.arange(lo, hi)
        near = cand[np.argsort(np.abs(gs[cand] - g))[:k]]
        out[i] = order[rng.choice(near)]
    return eGreal[out], eCreal[out]


def draw_synthetic(gen, truth: dict, n: int, rng, real=None):
    """One synthetic catalogue of ``n`` observed stars (see the module docstring). Returns the
    QTable and a dict with the number of field stars."""
    real = real_table() if real is None else real
    Greal = np.asarray(real["Gmag"], float)
    Creal = np.asarray(real["G_BPmag"], float) - np.asarray(real["G_RPmag"], float)
    eGr = np.asarray(real["e_Gmag"], float)
    eCr = np.asarray(real["e_BP_RP"], float)
    mag_lim = float(Greal.max())
    n_bg = int(rng.binomial(n, truth["f_bg"]))
    n_cl = n - n_bg
    s2 = gen.SIGMA_FLOOR**2 + truth["sigma_int"] ** 2
    Gs, Cs, eGs, eCs = [], [], [], []
    got = 0
    for _ in range(200):
        if got >= n_cl:
            break
        Gt, Ct = _true_stars(gen, truth["feh"], truth["loga"], truth["dm"], truth["Av"], 4 * n, rng)
        eG, eC = _real_errors(Greal, eGr, eCr, Gt, rng)
        Go = Gt + rng.normal(size=Gt.size) * np.sqrt(eG**2 + s2)
        Co = Ct + rng.normal(size=Gt.size) * np.sqrt(eC**2 + s2)
        keep = Go <= mag_lim
        Gs.append(Go[keep])
        Cs.append(Co[keep])
        eGs.append(eG[keep])
        eCs.append(eC[keep])
        got += int(keep.sum())
    G = np.concatenate(Gs)[:n_cl]
    C = np.concatenate(Cs)[:n_cl]
    eG = np.concatenate(eGs)[:n_cl]
    eC = np.concatenate(eCs)[:n_cl]
    Gb = rng.uniform(Greal.min(), Greal.max(), n_bg)
    Cb = rng.uniform(Creal.min(), Creal.max(), n_bg)
    eGb, eCb = _real_errors(Greal, eGr, eCr, Gb, rng)
    G, C = np.concatenate([G, Gb]), np.concatenate([C, Cb])
    eG, eC = np.concatenate([eG, eGb]), np.concatenate([eC, eCb])
    return table(G, C, eG, eC), {"n_bg": n_bg, "n_cluster": n_cl}


def laplace(f, x):
    """The control estimator's width: ``av_synth.laplace`` (sd from -H^-1 of log L + dm prior)."""
    from av_synth import laplace as _lap

    return _lap(f, x)


def laplace_cov(f, x):
    """Full -H^-1 (6 x 6) of the same log posterior, for the ridge direction of the starts."""
    fl = f._compiled_loglike(None)

    def grad(y):
        v, g = fl(y)
        g = np.array(g, float).copy()
        g[2] -= (y[2] - f.dm_mu) / f.dm_sigma**2
        return g

    x = np.asarray(x, float)
    h = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4])
    H = np.empty((6, 6))
    for i in range(6):
        e = np.zeros(6)
        e[i] = h[i]
        H[i] = (grad(x + e) - grad(x - e)) / (2 * h[i])
    H = 0.5 * (H + H.T)
    return np.linalg.inv(-H)


def control_fit(f) -> dict:
    """The old estimator, exactly as ``av_synth.run_fit``: ``search`` + Laplace sd."""
    res = search(f, verbose=False)
    m = res["mode"]
    x = [m[k] for k in PARAMS]
    res["laplace_sd"] = laplace(f, x)
    try:
        res["laplace_cov"] = laplace_cov(f, x).tolist()
    except np.linalg.LinAlgError:
        res["laplace_cov"] = None
    return res


# loga offsets of the four chain starts along the age-[Fe/H]-dm ridge: twice the MAP scatter
# measured off the nodes (0.085 dex, ctl-off), so the starts straddle the ridge instead of sitting
# within 3 local sd (~0.01 dex) of one point of it (find_start), which would let R-hat certify
# mixing within one point of the ridge (methodology §A.1.2)
START_DLOGA = (-0.17, -0.06, 0.06, 0.17)


def chain_starts(f, ctrl: dict, rng) -> dict:
    """Overdispersed starts: the mode moved along the Laplace covariance's regression line of
    (feh, dm, Av) on log t (the ridge, to first order) so that log t moves by ``START_DLOGA``; clipped strictly
    inside the priors. Nuisance (sigma_int, f_bg) at the mode times U(0.7, 1.4). Also the seed of
    the inverse mass matrix: the Laplace marginal sd (wider than find_start's conditional ones)."""
    m = ctrl["mode"]
    x = np.array([m[k] for k in PARAMS[:4]])
    lo = np.array([f._met_bounds()[0], f.loga_range[0], f.dm_range[0], f.Av_range[0]])
    hi = np.array([f._met_bounds()[1], f.loga_range[1], f.dm_range[1], f.Av_range[1]])
    v = None
    if ctrl.get("laplace_cov") is not None:
        S = np.asarray(ctrl["laplace_cov"])[:4, :4]
        # the Gaussian ridge line: E[x | log t] moves by S[:, loga] / S[loga, loga] per unit log t
        if np.all(np.isfinite(S)) and S[1, 1] > 0:
            v = S[:, 1] / S[1, 1]
    if v is None:  # no usable covariance: move log t alone
        v = np.array([0.0, 1.0, 0.0, 0.0])
    margin = 1e-3 * (hi - lo)
    starts = []
    for d in START_DLOGA:
        s = np.clip(x + d * v, lo + margin, hi - margin)
        st = dict(zip(PARAMS[:4], s.tolist(), strict=True))
        st["sigma_int"] = float(max(m["sigma_int"], 1e-3) * rng.uniform(0.7, 1.4))
        st["f_bg"] = float(np.clip(m["f_bg"], 1e-3, 0.5) * rng.uniform(0.7, 1.4))
        starts.append(st)
    sd = {
        k: (ctrl["laplace_sd"][k] if np.isfinite(ctrl["laplace_sd"][k]) else 0.05)
        for k in PARAMS[:4]
    }
    return {"mode": m, "local_sd": sd, "chain_starts": starts, "ridge_vector": v.tolist()}


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float) + "\n")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
