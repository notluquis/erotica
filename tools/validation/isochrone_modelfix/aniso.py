#!/usr/bin/env python3
"""The six arms of the hub finding ``isochrone-model-fix.md`` §0.2: width shape {I, V, VR} x window
{F, W}. Each choice and the process it models (methodology §A.1):

* **I** (control): Sigma_int = sigma_int^2 diag(1, 1), the current ``IsochroneFitter``.
* **V**: Sigma_int = diag(sigma_G^2, sigma_c^2). sigma_G is a luminosity spread at fixed T_eff (age or
  accretion-history spread, variability, unmodelled multiplicity); sigma_c a colour spread at fixed L.
* **VR**: V plus differential extinction A_V,i ~ N(A_V, tau^2) with its full G-colour covariance
  tau^2 k k^T, k = (k_G, k_col) of the fitter's extinction layer.
* **F**: all stars, completeness cut at the faintest member. **W**: 14.0 <= G <= 19.0 through
  ``setup(mag_window=...)`` (exact conditional likelihood, both window edges in F(theta)).

The field component (uniform over the window's CMD box) and everything else is the fitter's own.
``WidthFitter`` only replaces the per-star covariance; with tau = 0 and sigma_G = sigma_c it is the
parent's likelihood exactly (checked in ``selftest``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "isochrone_intervals"))
import iv_common as C  # noqa: E402

from erotica.analysis._isochrone import IsochroneFitter, _erf_np  # noqa: E402

FULL = ("feh", "loga", "dm", "Av", "sG", "sC", "tau", "f_bg", "k", "beta")
DEFAULTS = {"sC": None, "tau": 0.0, "k": 1.0, "beta": 0.0}
ARM_FREE = {  # free width parameters of each width arm (the rest tied/fixed in ``expand``)
    "I": ("sG",),
    "V": ("sG", "sC"),
    "VR": ("sG", "sC", "tau"),
    # stage 2 (§0.6): isotropic width plus error scale k and/or exponential field in G
    "K": ("sG", "k"),
    "B": ("sG", "beta"),
    "KB": ("sG", "k", "beta"),
}
ISOTROPIC = ("I", "K", "B", "KB")
WINDOW = {"F": None, "W": (14.0, 19.0)}
WBOUNDS = {"sG": (1e-4, 2.0), "sC": (1e-4, 0.5), "tau": (1e-4, 2.0), "f_bg": (1e-4, 0.5),
           "k": (1.0, 6.0), "beta": (-2.0, 3.0)}


def _seg_density_cov(xg, xc, sg, sc, s12, Ag, Ac, Bg, Bc, xp=np, smear=None, first_moment=False):
    """``erotica.analysis._isochrone._segment_density`` with a per-star G-colour covariance ``s12``
    (scalar or (N,)) added to the star's covariance. Same algebra otherwise."""
    s11 = (sg**2)[:, None]
    s22 = (sc**2)[:, None]
    s12m = (s12 + 0.0 * sg)[:, None]
    if smear is not None:
        vg, vc = smear
        s11 = s11 + (vg**2 / 12.0)[None, :]
        s22 = s22 + (vc**2 / 12.0)[None, :]
        s12m = s12m + (vg * vc / 12.0)[None, :]
    l11 = xp.sqrt(s11)
    l21 = s12m / l11
    l22 = xp.sqrt(s22 - l21**2)
    ag = (Ag[None, :] - xg[:, None]) / l11
    ac = ((Ac[None, :] - xc[:, None]) - l21 * ag) / l22
    dg = (Bg - Ag)[None, :] / l11
    dc = ((Bc - Ac)[None, :] - l21 * dg) / l22
    L2 = dg**2 + dc**2
    long_ = L2 > 1e-8
    L2s = xp.where(long_, L2, 1.0)
    L = xp.sqrt(L2s)
    t0 = -(ag * dg + ac * dc) / L2s
    perp2 = (ag + t0 * dg) ** 2 + (ac + t0 * dc) ** 2
    erf = xp.erf if xp is not np else _erf_np
    cdf_diff = 0.5 * (erf(L * (1 - t0) / np.sqrt(2.0)) - erf(-L * t0 / np.sqrt(2.0)))
    I0 = np.sqrt(2 * np.pi) / L * cdf_diff
    seg = xp.exp(-0.5 * perp2) * I0
    point = xp.exp(-0.5 * (ag**2 + ac**2))
    norm = 1.0 / (2 * np.pi * l11 * l22)
    J0 = xp.where(long_, seg, point) * norm
    if not first_moment:
        return J0
    I1 = t0 * I0 + (xp.exp(-0.5 * L2s * t0**2) - xp.exp(-0.5 * L2s * (1 - t0) ** 2)) / L2s
    J1 = xp.where(long_, xp.exp(-0.5 * perp2) * I1, 0.5 * point) * norm
    return J0, J1


class WidthFitter(IsochroneFitter):
    """``IsochroneFitter`` with the anisotropic intrinsic covariance of arms V / VR."""

    def star_loglike8(self, met, loga, dm, Av, sG, sC, tau, f_bg, xp=np):
        return self.star_loglike10(met, loga, dm, Av, sG, sC, tau, f_bg, 1.0, 0.0, xp)

    def field_density(self, beta, xp=np):
        """Field density per unit CMD area at each star: exp(beta (G - G_lo)) normalised over the
        window's G span, uniform in colour over the box; beta = 0 is the parent's 1 / box_area."""
        lo = float(self._obs_mag.min())
        span = max(float(np.ptp(self._obs_mag)), 1e-3)
        span_c = self._box_area / span
        small = xp.abs(beta) < 1e-4
        bs = xp.where(small, 1.0, beta)
        norm = xp.where(small, span, (xp.exp(bs * span) - 1.0) / bs)
        return xp.exp(beta * (self._obs_mag - lo)) / norm / span_c

    def star_loglike10(self, met, loga, dm, Av, sG, sC, tau, f_bg, k, beta, xp=np):
        fl2 = self.SIGMA_FLOOR**2
        kG, kc = self._kG, self._k_col1
        xg, xc = self._obs_mag, self._obs_col
        sg = xp.sqrt((k * self._e_obs_mag) ** 2 + fl2 + sG**2 + (kG * tau) ** 2)
        sc = xp.sqrt((k * self._e_obs_col) ** 2 + fl2 + sC**2 + (kc * tau) ** 2)
        s12 = kG * kc * tau**2
        single, binary = self._segments(met, loga, dm, Av, xp)
        Ag, Ac, Bg, Bc, c0, c1 = single
        J0, J1 = _seg_density_cov(xg, xc, sg, sc, s12, Ag, Ac, Bg, Bc, xp, first_moment=True)
        dens = xp.dot(J0, c0) + xp.dot(J1, c1)
        if binary is not None:
            Ag, Ac, Bg, Bc, w, vg, vc = binary
            dens = dens + xp.dot(
                _seg_density_cov(xg, xc, sg, sc, s12, Ag, Ac, Bg, Bc, xp, smear=(vg, vc)), w
            )
        # completeness / window: only the G variance matters
        sGeff2 = sG**2 + (kG * tau) ** 2
        F = self._detected_fraction(met, loga, dm, Av, xp.sqrt(sGeff2), xp, segs=(single, binary))
        return xp.log((1 - f_bg) * dens / F + f_bg * self.field_density(beta, xp))

    def compiled8(self, arm: str):
        """(log L, grad) over the arm's free parameter vector (see ``free_names``)."""
        import pytensor
        import pytensor.tensor as pt

        names = free_names(arm)
        v = {n: pt.dscalar(n) for n in names}
        full = expand_sym(arm, v, pt)
        ll = pt.sum(self._star_weights * self.star_loglike10(*[full[k] for k in FULL], xp=pt))
        fn = pytensor.function([v[n] for n in names], [ll, *pytensor.grad(ll, [v[n] for n in names])])
        return lambda x: (lambda out: (float(out[0]), np.array(out[1:], float)))(fn(*x))


def free_names(arm: str) -> tuple:
    return ("feh", "loga", "dm", "Av", *ARM_FREE[arm], "f_bg")


def _fill(arm, d, zero):
    out = dict(d)
    if arm in ISOTROPIC:
        out["sC"] = d["sG"]
    for k_, v in DEFAULTS.items():
        if k_ not in out:
            out[k_] = zero + v
    return out


def expand_sym(arm, v, pt=None):
    return _fill(arm, v, 0.0 * v["sG"])


def expand(arm: str, x) -> dict:
    return _fill(arm, dict(zip(free_names(arm), [float(t) for t in x], strict=True)), 0.0)


def make_fitter(data, window: str) -> WidthFitter:
    from common import EDR3, PRI_PLX, mist

    f = WidthFitter(grid=mist(), magnitude=EDR3[0], color=EDR3[1:], **PRI_PLX)
    f.setup(data, prob_threshold=0.0, mag_window=WINDOW[window])
    return f


def search8(f: WidthFitter, arm: str, feh_lattice=(-1.0, -0.5, 0.0, 0.5), age_stride=2, polish=4,
            verbose=True) -> dict:
    """``common.search`` generalised to the arm's parameter vector: (dm, A_V) optimised at every
    (feh, log t) lattice point at the wide width (0.05 mag, tau 0, f_bg 0.02), then the ``polish``
    best polished on every free parameter (L-BFGS-B, bounded). Never seeded at a truth."""
    from scipy.optimize import minimize

    t0 = time.time()
    names = free_names(arm)
    fl = f.compiled8(arm)
    zlo, zhi = f._met_bounds()
    blo = {"feh": zlo + 1e-9, "loga": f.loga_range[0], "dm": f.dm_range[0], "Av": f.Av_range[0]}
    bhi = {"feh": zhi - 1e-9, "loga": f.loga_range[1], "dm": f.dm_range[1], "Av": f.Av_range[1]}
    for k, (a, b) in WBOUNDS.items():
        blo[k], bhi[k] = a, b
    lo = np.array([blo[n] for n in names])
    hi = np.array([bhi[n] for n in names])
    wide = {"sG": 0.05, "sC": 0.05, "tau": 1e-4, "f_bg": 0.02, "k": 1.0, "beta": 0.0}

    def fn(y):
        v, g = fl(y)
        z = (y[2] - f.dm_mu) / f.dm_sigma
        g = g.copy()
        g[2] -= z / f.dm_sigma
        return v - 0.5 * z * z, g

    ages = [a for a in f._node_loga if lo[1] <= a <= hi[1]][::age_stride]
    fehs = [x for x in feh_lattice if zlo <= x <= zhi]
    tail = [wide[n] for n in names[4:]]
    cands = []
    for z in fehs:
        for a in ages:

            def nll2(y, z=float(z), a=float(a)):
                v, g = fn(np.array([z, a, y[0], y[1], *tail]))
                return -v, -g[2:4]

            r = minimize(nll2, [f.dm_mu, 1.0], jac=True, method="L-BFGS-B",
                         bounds=list(zip(lo[2:4], hi[2:4], strict=True)))
            cands.append((float(r.fun), float(z), float(a), *map(float, r.x)))
    cands.sort(key=lambda c: c[0])
    if verbose:
        print(f"  [{arm}] lattice {len(cands)} t={time.time() - t0:.0f}s", flush=True)
    start_tail = {"sG": 0.03, "sC": 0.03, "tau": 0.03, "f_bg": 0.02, "k": 1.5, "beta": 0.5}

    def nll(y):
        v, g = fn(y)
        return -v, -g

    pol = []
    for c in cands[:polish]:
        y0 = np.clip(np.array([*c[1:], *[start_tail[n] for n in names[4:]]]), lo, hi)
        r = minimize(nll, y0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi, strict=True)))
        pol.append((-float(r.fun), np.asarray(r.x, float)))
        if verbose:
            print(f"  [{arm}] polish nfev={r.nfev} lp={-r.fun:.2f} t={time.time() - t0:.0f}s", flush=True)
    pol.sort(key=lambda p: -p[0])
    lp, b = pol[0]
    at = [n for n, x, a, z in zip(names, b, lo, hi, strict=True) if min(x - a, z - x) < 1e-3 * (z - a)]
    return {"arm": arm, "x": b.tolist(), "mode": expand(arm, b), "logpost": lp, "loglike": fl(b)[0],
            "runner_up": [(p[0], p[1].tolist()) for p in pol[1:]], "at_bound": at,
            "n": int(f._N_obs), "seconds": round(time.time() - t0, 1)}


def draw(f: WidthFitter, gen_full, th: dict, n: int, rng, real, window):
    """Synthetic catalogue at ``th`` (a FULL dict) for the arm's width shape and window: the
    generator of ``iv_common.draw_synthetic`` (forward model of the full-window fitter ``gen_full``,
    real per-star errors by nearest G), with intrinsic noise N(0, diag(sG^2, sC^2)) plus a common
    extinction offset dA ~ N(0, tau) moving (k_G dA, k_col dA), the window [lo, hi] on observed G
    (hi = faintest real member for F) and the field uniform over the window's real CMD box."""
    Greal = np.asarray(real["Gmag"], float)
    Creal = np.asarray(real["G_BPmag"], float) - np.asarray(real["G_RPmag"], float)
    eGr = np.asarray(real["e_Gmag"], float)
    eCr = np.asarray(real["e_BP_RP"], float)
    lo, hi = (Greal.min() - 10.0, Greal.max()) if window is None else window
    inw = (Greal >= lo) & (Greal <= hi)
    Gw, Cw = Greal[inw], Creal[inw]
    kk = th.get("k", 1.0)
    beta = th.get("beta", 0.0)
    n_bg = int(rng.binomial(n, th["f_bg"]))
    n_cl = n - n_bg
    fl2 = f.SIGMA_FLOOR**2
    kG, kc = f._kG, f._k_col1
    Gs, Cs, eGs, eCs = [], [], [], []
    got = 0
    for _ in range(400):
        if got >= n_cl:
            break
        Gt, Ct = C._true_stars(gen_full, th["feh"], th["loga"], th["dm"], th["Av"], 4 * n, rng)
        eG, eC = C._real_errors(Greal, eGr, eCr, Gt, rng)
        eG, eC = kk * eG, kk * eC
        dA = rng.normal(size=Gt.size) * th["tau"]
        Go = Gt + kG * dA + rng.normal(size=Gt.size) * np.sqrt(eG**2 + fl2 + th["sG"] ** 2)
        Co = Ct + kc * dA + rng.normal(size=Gt.size) * np.sqrt(eC**2 + fl2 + th["sC"] ** 2)
        keep = (Go >= lo) & (Go <= hi)
        Gs.append(Go[keep])
        Cs.append(Co[keep])
        eGs.append(eG[keep])
        eCs.append(eC[keep])
        got += int(keep.sum())
    G = np.concatenate(Gs)[:n_cl]
    Cc = np.concatenate(Cs)[:n_cl]
    span = Gw.max() - Gw.min()
    u = rng.uniform(size=n_bg)
    if abs(beta) < 1e-4:
        Gb = Gw.min() + u * span
    else:  # inverse CDF of the truncated exponential
        Gb = Gw.min() + np.log1p(u * np.expm1(beta * span)) / beta
    Cb = rng.uniform(Cw.min(), Cw.max(), n_bg)
    return np.concatenate([G, Gb]), np.concatenate([Cc, Cb]), n_cl


def selftest() -> dict:
    """Arm I of WidthFitter == parent likelihood at X1 (full window), per star."""
    real = C.real_table()
    f = make_fitter(real, "F")
    x = C.X1
    a = f._star_loglike(x["feh"], x["loga"], x["dm"], x["Av"], x["sigma_int"], x["f_bg"], np)
    b = f.star_loglike8(x["feh"], x["loga"], x["dm"], x["Av"], x["sigma_int"], x["sigma_int"], 0.0,
                        x["f_bg"], np)
    return {"max_abs_diff": float(np.max(np.abs(a - b))), "sum_parent": float(a.sum())}


if __name__ == "__main__":
    print(selftest())


def build_model8(f: WidthFitter, arm: str):
    """PyMC model of an arm, for NUTS: the parent's ``build_model`` priors on the shared
    parameters (MIST [Fe/H] uniform, log t, A_V uniform, dm truncated normal from the parallax,
    sigma_int HalfNormal(0.05), f_bg Beta(1, 19)) plus, per arm: k ~ U(1, 6) (the error scale; >= 1
    because the hypothesis is under-estimated DR3 errors, upper end = search bound), beta ~ U(-2, 3),
    sigma_c ~ HalfNormal(0.05), tau ~ HalfNormal(0.3) (the anchors' tau, isochrone-sed-external §2).
    Variable names: feh (the parent's ``_met_name``), loga, dm, Av, sigma_int (= sG), f_bg, and k / beta / sigma_c / tau."""
    import pymc as pm
    import pytensor.tensor as pt

    from erotica.analysis._isochrone import _require_pymc

    _require_pymc()
    zlo, zhi = f._met_bounds()  # as the parent: [Fe/H] on the grid layer
    free = ARM_FREE[arm]
    with pm.Model() as model:
        met = pm.Uniform(f._met_name, lower=zlo, upper=zhi)
        loga = pm.Uniform("loga", lower=f.loga_range[0], upper=f.loga_range[1])
        dm = pm.TruncatedNormal("dm", mu=f.dm_mu, sigma=f.dm_sigma, lower=f.dm_range[0],
                                upper=f.dm_range[1])
        Av = pm.Uniform("Av", lower=f.Av_range[0], upper=f.Av_range[1])
        sG = pm.HalfNormal("sigma_int", sigma=0.05)
        f_bg = pm.Beta("f_bg", alpha=1.0, beta=19.0)
        v = {"sG": sG}
        if "sC" in free:
            v["sC"] = pm.HalfNormal("sigma_c", sigma=0.05)
        if "tau" in free:
            v["tau"] = pm.HalfNormal("tau", sigma=0.3)
        if "k" in free:
            v["k"] = pm.Uniform("k", lower=1.0, upper=6.0)
        if "beta" in free:
            v["beta"] = pm.Uniform("beta", lower=-2.0, upper=3.0)
        full = _fill(arm, v, 0.0 * sG)
        w = f._star_weights

        def _logp(value, met, loga, dm, Av, sG, sC, tau, f_bg, k, beta):
            return w * f.star_loglike10(met, loga, dm, Av, sG, sC, tau, f_bg, k, beta, pt)

        pm.CustomDist("y", met, loga, dm, Av, full["sG"], full["sC"], full["tau"], f_bg,
                      full["k"], full["beta"], logp=_logp, observed=np.zeros(f._N_obs))
    return model
