"""Grid-posterior SED fit for T_eff and A_V per star, with no distance and no isochrone.

Model: m_b = m_bol - BC_b(T_eff, log g, [Fe/H], A_V) + eps_b, eps_b ~ N(0, sigma_b^2 + s_b^2).
m_bol has a flat prior and is marginalised analytically: for Gaussian errors the marginal likelihood
is exp(-chi2_min(m_bol)/2) times a factor that depends only on the weights, which are fixed per star,
so the grid posterior is exp(-chi2/2) with m_bol at its weighted best value.

Two model families share the same extinction (MIST's, integrated over the spectrum, R_V = 3.1):
- ``mist``: MIST v1 bolometric corrections (C3K/ATLAS atmospheres) — the same atmospheres that give
  the colours of the MIST isochrones.
- ``eem``: empirical dwarf intrinsic colours of Pecaut & Mamajek (EEM table v2022.04.16) relative to
  G, plus A_b(T, g, A_V) = BC_b(A_V = 0) - BC_b(A_V) from MIST. Independent of MIST atmospheres for
  the intrinsic colour, not for the extinction term.

Priors (pre-registered, agent-findings/isochrone-sed-external.md §0.3): flat in log T_eff on
[T_lo, 50 000] K (step 0.004 dex), flat in A_V on [0, 6] (step 0.01), flat over log g nodes
{3.0, 3.5, 4.0, 4.5, 5.0}, [Fe/H] fixed at a MIST node.
"""

from __future__ import annotations

import functools
import io
import tarfile
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
DL = HERE / "dl"
BC_URL = "https://mist.science/BC_tables/v1/UBVRIplus.txz"
BC_SHA256 = "524dc5d0fe7c9993ce9d07d32063fda8ce3dad09139739fff14868e89ae622c4"
EEM_URL = "https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt"
EEM_SHA256 = "1de2edeec17bb3346e0e4e70b999de5ee29947df38474e64cddb7cfacc164b7f"

BANDS = ["G", "BP", "RP", "J", "H", "Ks"]
MIST_COLS = ["Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3", "2MASS_J", "2MASS_H", "2MASS_Ks"]
LOGG = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
AV = np.round(np.arange(0.0, 6.0001, 0.01), 4)
FEH_FILE = {0.0: "fehp000", -0.25: "fehm025"}


def _sha(b: bytes) -> str:
    import hashlib

    return hashlib.sha256(b).hexdigest()


def _fetch(url, name, sha):
    DL.mkdir(exist_ok=True)
    p = DL / name
    if not p.exists():
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        p.write_bytes(r.content)
    got = _sha(p.read_bytes())
    if got != sha:
        raise SystemExit(f"{name}: sha256 {got} != pinned {sha}")
    return p


@functools.lru_cache(maxsize=4)
def load_mist_bc(feh=0.0):
    """BC[iT, ig, iA, band] on the native MIST nodes, plus the node vectors."""
    p = _fetch(BC_URL, "UBVRIplus.txz", BC_SHA256)
    with tarfile.open(p, "r:xz") as tf:
        member = [m for m in tf.getmembers() if m.name.endswith(f"{FEH_FILE[feh]}.UBVRIplus")][0]
        txt = tf.extractfile(member).read().decode()
    header = [ln for ln in txt.splitlines() if ln.startswith("#") and "Teff" in ln][0]
    names = header.lstrip("#").split()
    a = np.loadtxt(io.StringIO(txt), comments="#")
    T, g, A = np.unique(a[:, 0]), np.unique(a[:, 1]), np.unique(a[:, 3])
    if T.size * g.size * A.size != a.shape[0]:
        raise SystemExit("MIST BC grid is not regular")
    order = np.lexsort((a[:, 3], a[:, 1], a[:, 0]))
    a = a[order]
    idx = [names.index(c) for c in MIST_COLS]
    bc = a[:, idx].reshape(T.size, g.size, A.size, len(idx))
    return T, g, A, bc


def _interp_axis(x_nodes, y, x_new, axis):
    """Linear interpolation of y along ``axis`` from x_nodes to x_new (x_new inside the node range)."""
    i = np.clip(np.searchsorted(x_nodes, x_new) - 1, 0, x_nodes.size - 2)
    t = (x_new - x_nodes[i]) / (x_nodes[i + 1] - x_nodes[i])
    y0 = np.take(y, i, axis=axis)
    y1 = np.take(y, i + 1, axis=axis)
    shape = [1] * y.ndim
    shape[axis] = -1
    t = t.reshape(shape)
    return y0 * (1 - t) + y1 * t


def mist_fine(feh=0.0, t_lo=2800.0, t_hi=50000.0, dlogt=0.004):
    """BC on the fine grid: (nT, nA, ng, nb), log T linear in log, A_V linear, g on LOGG nodes."""
    T, g, A, bc = load_mist_bc(feh)
    gi = [int(np.argmin(np.abs(g - x))) for x in LOGG]
    if not np.allclose(g[gi], LOGG):
        raise SystemExit("log g node missing in MIST BC")
    bc = bc[:, gi]  # (T, 5, A, b)
    logt = np.arange(np.log10(t_lo), np.log10(t_hi) + 1e-9, dlogt)
    f = _interp_axis(np.log10(T), bc, logt, axis=0)  # (nT, 5, A, b)
    f = _interp_axis(A, f, AV, axis=2)  # (nT, 5, nA, b)
    return logt, np.transpose(f, (0, 2, 1, 3)).copy()  # (nT, nA, ng, nb)


@functools.lru_cache(maxsize=1)
def load_eem():
    p = _fetch(EEM_URL, "EEM.txt", EEM_SHA256)
    rows = []
    started = False
    for ln in p.read_text().splitlines():
        if ln.startswith("#SpT"):
            started = True
            continue
        if not started or ln.startswith("#") or not ln.strip():
            if started and ln.startswith("# UPDATES"):
                break
            continue
        rows.append(ln.split())
    head = ("SpT Teff logT BCv logL Mbol R_Rsun Mv B-V Bt-Vt G-V Bp-Rp G-Rp M_G b-y U-B V-Rc V-Ic "
            "V-Ks J-H H-Ks").split()

    def col(name):
        k = head.index(name)
        out = []
        for r in rows:
            try:
                out.append(float(r[k]))
            except (ValueError, IndexError):
                out.append(np.nan)
        return np.array(out)

    out = {n: col(n) for n in ("Teff", "G-V", "Bp-Rp", "G-Rp", "V-Ks", "J-H", "H-Ks")}
    out["SpT"] = np.array([r[0] for r in rows])
    return out


def eem_fine(feh=0.0, dlogt=0.004):
    """Pseudo-BC from EEM intrinsic colours + MIST extinction. Returns logt, BC (nT, nA, ng, nb)."""
    e = load_eem()
    ok = np.all([np.isfinite(e[k]) for k in e if k != "SpT"], axis=0)
    T = e["Teff"][ok]
    # colours relative to G (intrinsic): b - G
    bg = {
        "G": np.zeros(T.size),
        "BP": e["Bp-Rp"][ok] - e["G-Rp"][ok],
        "RP": -e["G-Rp"][ok],
        "Ks": -e["V-Ks"][ok] - e["G-V"][ok],
    }
    bg["H"] = bg["Ks"] + e["H-Ks"][ok]
    bg["J"] = bg["H"] + e["J-H"][ok]
    o = np.argsort(T)
    T = T[o]
    logT = np.log10(T)
    if np.any(np.diff(logT) <= 0):
        keep = np.concatenate([[True], np.diff(logT) > 0])
        T, logT = T[keep], logT[keep]
        o = o[keep]
    logt = np.arange(max(logT[0], np.log10(2800.0)), logT[-1] + 1e-9, dlogt)  # MIST BC starts at 2500 K
    c0 = np.stack([np.interp(logt, logT, bg[b][o]) for b in BANDS], axis=-1)  # (nT, nb)
    lt_m, bcm = mist_fine(feh, t_lo=10 ** logt[0], t_hi=10 ** logt[-1] * 1.0001, dlogt=dlogt)
    n = min(lt_m.size, logt.size)
    logt, c0, bcm = logt[:n], c0[:n], bcm[:n]
    ext = bcm[:, :1] - bcm  # A_b(T, A_V, g) = BC(A_V=0) - BC(A_V)  (nT, nA, ng, nb)
    bc = -(c0[:, None, None, :]) - ext  # m_b = m0 - BC  =>  m_b = G0 + (b-G)_0 + A_b
    return logt, bc, (float(10 ** logt[0]), float(10 ** logt[-1]))


def truth_mags(model, logt, logg, av, feh=0.0):
    """Noise-free magnitudes (up to a constant) at continuous (log T, log g, A_V), trilinear in the
    native MIST nodes (log T, log g, A_V); ``eem`` takes the intrinsic colour from the EEM table and
    the extinction from MIST at the same point."""
    T, g, A, bc = load_mist_bc(feh)
    lT = np.log10(T)
    x = _interp_axis(lT, bc, np.atleast_1d(logt), 0)[0]  # (g, A, b)
    x = _interp_axis(g, x, np.atleast_1d(logg), 0)[0]  # (A, b)
    bc_av = _interp_axis(A, x, np.atleast_1d(av), 0)[0]
    bc_0 = x[0]
    if model == "mist":
        return -bc_av
    e = load_eem()
    ok = np.all([np.isfinite(e[k]) for k in e if k != "SpT"], axis=0)
    Te = e["Teff"][ok]
    o = np.argsort(Te)
    lte = np.log10(Te[o])
    bgk = {"G": 0 * lte, "BP": (e["Bp-Rp"][ok] - e["G-Rp"][ok])[o], "RP": -e["G-Rp"][ok][o],
           "Ks": (-e["V-Ks"][ok] - e["G-V"][ok])[o]}
    bgk["H"] = bgk["Ks"] + e["H-Ks"][ok][o]
    bgk["J"] = bgk["H"] + e["J-H"][ok][o]
    c0 = np.array([np.interp(logt, lte, bgk[b]) for b in BANDS])
    return c0 + (bc_0 - bc_av)


def cstar(G, bp_rp, C):
    """Riello+2021 corrected BP/RP flux excess and its 1-sigma scatter (coefficients as in
    agent-findings/scripts/e6_bp_excess_blends.py, read from the paper's Table 2)."""
    x = np.asarray(bp_rp, float)
    f = np.full_like(x, np.nan)
    m1, m2, m3 = x < 0.5, (x >= 0.5) & (x < 4.0), x >= 4.0
    f[m1] = 1.154360 + 0.033772 * x[m1] + 0.032277 * x[m1] ** 2
    f[m2] = 1.162004 + 0.011464 * x[m2] + 0.049255 * x[m2] ** 2 - 0.005879 * x[m2] ** 3
    f[m3] = 1.057572 + 0.140537 * x[m3]
    cs = np.asarray(C, float) - f
    sig = 0.0059898 + 8.817481e-12 * np.asarray(G, float) ** 7.618399
    return cs, sig


def fit_star(bc, m, sig, use):
    """Grid posterior for one star. bc: (nT, nA, ng, nb); m, sig: (nb,); use: bool (nb,).
    Returns the posterior over (nT, nA) marginalised over g, and chi2_min."""
    b = np.flatnonzero(use)
    w = 1.0 / sig[b] ** 2
    r = m[b] + bc[..., b]  # = m_bol estimates per band
    mb = (r * w).sum(-1) / w.sum()
    chi2 = ((r - mb[..., None]) ** 2 * w).sum(-1)  # (nT, nA, ng)
    cmin = chi2.min()
    p = np.exp(-0.5 * (chi2 - cmin)).sum(-1)  # flat over g nodes
    return p / p.sum(), float(cmin)


def summarise(p, logt, av=AV):
    pt, pa = p.sum(1), p.sum(0)

    def q(x, w, qs):
        c = np.cumsum(w)
        return np.interp(qs, c / c[-1], x)

    lt16, lt50, lt84 = q(logt, pt, [0.16, 0.5, 0.84])
    a16, a50, a84 = q(av, pa, [0.16, 0.5, 0.84])
    LT, A = np.meshgrid(logt, av, indexing="ij")
    mt, ma = (p * LT).sum(), (p * A).sum()
    st, sa = np.sqrt((p * (LT - mt) ** 2).sum()), np.sqrt((p * (A - ma) ** 2).sum())
    rho = ((p * (LT - mt) * (A - ma)).sum() / (st * sa)) if st > 0 and sa > 0 else np.nan
    far = float(pa[np.abs(av - a50) > 0.5].sum())
    edge_t = float(pt[:3].sum() + pt[-3:].sum())
    return dict(logt16=lt16, logt50=lt50, logt84=lt84, sd_logt=st, av16=a16, av50=a50, av84=a84,
                sd_av=sa, rho=rho, far_mass=far, edge_t=edge_t)
