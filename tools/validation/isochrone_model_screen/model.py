r"""Model variants for the isochrone screen of NGC 6383 (hub finding ``isochrone-model-screen.md``).

WHY THIS EXISTS
---------------
C1 (``erotica`` 96ce4c0) gave loga 6.07 with Z = 0.005 and A_V 0.64 on a ridge whose position the
dm prior fixes (hub finding ``isochrone-c1-age-review.md``). Seven proposed changes to the model are
screened together as a 2^(7-3) fractional factorial instead of one by one. This module is the model
side: ``ScreenFitter`` is ``IsochroneFitter`` with each change as a switch, and with every switch off
it **is** the package model (``tests`` in ``screen.py check`` assert log L equal to 1e-9).

THE SWITCHES -- each with the physical process it models (methodology §A.1)
-------------------------------------------------------------------------
``zprior``   prior on [M/H] = log10(Z/0.0142) (MIST v1.2 / Asplund+09 scale), Normal(mu, sd), from
             the radial [Fe/H] gradient of open clusters at R_gc 7.19 kpc (no spectroscopic value of
             NGC 6383 exists: hub findings e4/e7). Off: uniform in linear Z, i.e. p([M/H]) ∝ 10^[M/H].
             The Jacobian 1/Z is included so the prior is Normal *in [M/H]*.
``diffred``  differential reddening: each star's A_V scatters around the cluster value with sd
             ``sigma_Av``; in the CMD that is a rank-1 covariance sigma_Av^2 k k^T along the local
             reddening vector k = (k_G, k_BP - k_RP), added to every segment's covariance (and to the
             G width of the completeness integral). Off: only the isotropic ``sigma_int``.
``loga_lo``  lower bound of the age prior (6.0 in C1, P01's range; 5.5 here reaches below it).
``plx``      the members' Gaia parallaxes as a likelihood of the distance: N(plx_obs | 10^(2-dm/5),
             sigma_plx), sigma_plx with the 10.3 µas angular-covariance floor (Maíz Apellániz+2021,
             methodology §J.2); dm uniform on ``dm_range``. Off: C1's N(10.3, 0.2) prior on dm.
``imf``      ``"c05"``: the package's xi (m_c 0.20, sigma 0.55, m0 1, slope 2.35 = Chabrier 2005
             *individual-object* IMF, labelled "Chabrier 2014 system"); ``"chc14"``: what ASteCA's
             ``chabrier_2014`` implements (n_c 11, m_c 0.18, sigma 0.579, m0 = n_c m_c = 1.98, x 1.35;
             Chabrier, Hennebelle & Charlot 2014 Table 2, MW row), the IMF P01 and the thesis cite.
``ext``      ``"ccm"``: CCM89+O'Donnell at fixed effective wavelengths (constant k); ``"fitz19"``: the
             ESA Gaia EDR3 extinction law (Fitzpatrick+2019 curve; Riello+2021) -- k as a cubic in the
             intrinsic (BP-RP)_0 of each EEP point and in A0 = A_V, ``Fitz19_EDR3_MainSequence.csv``
             row BPRP, X clipped to its validity range [-0.06, 2.5].
``cstar``    data cut, applied in ``screen.py``: drop |C*| > 3 sigma_C*(G) (Riello+2021 eq. 18/21),
             stars whose BP/RP fluxes are inconsistent with G (blends, nebular background). Not a
             model switch; the completeness function does not model this cut [I: ~10 % of stars].
"""

from __future__ import annotations

from typing import Any

import numpy as np

from erotica.analysis._isochrone import (
    _Q_NODES,
    IsochroneFitter,
    _chabrier2014_xi,
    _dk_gamma,
    _erf_np,
)

Z_SUN = 0.0142
# Fitz19_EDR3_MainSequence.csv, X = (BP-RP)_0 rows; columns a1..a10 for
# k = a1 + a2 X + a3 X^2 + a4 X^3 + a5 A + a6 A^2 + a7 A^3 + a8 A X + a9 A X^2 + a10 X A^2
FITZ19 = {
    "G": (
        0.99597,
        -0.15973,
        0.012238,
        0.00090727,
        -0.037716,
        0.0015135,
        -2.5236e-5,
        0.011452,
        -0.00093691,
        -0.00026030,
    ),
    "BP": (
        1.15363,
        -0.081401,
        -0.036013,
        0.019214,
        -0.022398,
        0.00084056,
        -1.3102e-5,
        0.0066012,
        -0.00088225,
        -0.00011122,
    ),
    "RP": (
        0.66321,
        -0.017985,
        0.00049377,
        -0.0026799,
        -0.0065142,
        3.3018e-5,
        1.5789e-6,
        -7.9801e-5,
        0.00025568,
        1.1048e-5,
    ),
}
FITZ19_XRANGE = (-0.06, 2.5)


def fitz19_k(band: str, X: Any, A0: Any, xp: Any = np) -> Any:
    a = FITZ19[band]
    X = xp.clip(X, *FITZ19_XRANGE)
    A = xp.clip(A0, 0.01, 20.0)
    return (
        a[0]
        + a[1] * X
        + a[2] * X**2
        + a[3] * X**3
        + a[4] * A
        + a[5] * A**2
        + a[6] * A**3
        + a[7] * A * X
        + a[8] * A * X**2
        + a[9] * X * A**2
    )


def chc14_xi(m: Any, xp: Any = np) -> Any:
    """ASteCA ``chabrier_2014``: lognormal (m_c 0.18, sigma^2 = log10(n_c)/(x ln 10)) below
    m0 = n_c m_c, Salpeter-like power law m^-(1+x) above, continuous at m0. dN/dm, unnormalised."""
    nc, mc, x = 11.0, 0.18, 1.35
    m0 = nc * mc
    sig = float(np.sqrt(np.log10(nc) / (x * np.log(10.0))))
    xi_ln_m0 = float(np.exp(-0.5 * (np.log10(m0 / mc) / sig) ** 2) / m0)
    scale = xi_ln_m0 / m0 ** (-(1 + x))
    lm = xp.log10(m)
    xi_ln = xp.exp(-0.5 * ((lm - np.log10(mc)) / sig) ** 2) / m
    return xp.where(m < m0, xi_ln, scale * m ** (-(1 + x)))


def seg_density(xg, xc, sg, sc, Ag, Ac, Bg, Bc, xp, c11, c22, c12, first_moment=False):
    """``_segment_density`` of the package with an extra per-segment covariance (c11, c22, c12)."""
    s11 = (sg**2)[:, None] + c11[None, :]
    s22 = (sc**2)[:, None] + c22[None, :]
    s12 = 0.0 * (sg**2)[:, None] + c12[None, :]
    l11 = xp.sqrt(s11)
    l21 = s12 / l11
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


class ScreenFitter(IsochroneFitter):
    """``IsochroneFitter`` with the screen's switches (see module docstring)."""

    def __init__(self, *a, imf: str = "c05", ext: str = "ccm", diffred: bool = False, **kw):
        super().__init__(*a, **kw)
        self.imf = imf
        self.ext = ext
        self.diffred = diffred

    def _xi(self, m, xp):
        return _chabrier2014_xi(m, xp) if self.imf == "c05" else chc14_xi(m, xp)

    def _kvec(self, X0, Av, xp):
        """(k_G, k_col) at intrinsic colour X0 (array) -- constant for ``ccm``."""
        if self.ext == "ccm":
            return self._kG + 0.0 * X0, self._k_col1 + 0.0 * X0
        kG = fitz19_k("G", X0, Av, xp)
        kc = fitz19_k("BP", X0, Av, xp) - fitz19_k("RP", X0, Av, xp)
        return kG, kc

    # -- copy of IsochroneFitter._deposit with the IMF and extinction switches -----------------
    def _deposit(self, met, loga, dm, Av, xp=np):
        X = self._interp_isochrone(met, loga, xp)
        mass = xp.maximum(X[0], 1e-9)
        kG, kc = self._kvec(X[2], Av, xp)
        kGq, kcq = self._kvec(X[4::2], Av, xp)
        G, col = X[1] + dm + kG * Av, X[2] + kc * Av
        Gq, cq = X[3::2] + dm + kGq * Av, X[4::2] + kcq * Av
        m_mid = 0.5 * (mass[1:] + mass[:-1])
        w_seg = self._xi(m_mid, xp) * (mass[1:] - mass[:-1])
        w_norm = xp.sum(w_seg)
        w_seg = w_seg / w_norm
        K = self.BINARY_SUBSAMPLE
        t = (np.arange(K) + 0.5) / K
        n_e = self._nodes.shape[-1]
        idx = np.unique(np.r_[np.arange(0, n_e, self.BINARY_STRIDE), n_e - 1])

        def _sub(A):
            a0, a1 = A[..., :-1], A[..., 1:]
            out = a0[..., None] + (a1 - a0)[..., None] * t
            return out.reshape((-1,)) if A.ndim == 1 else out.reshape((A.shape[0], -1))

        def _step(A):
            d = (A[..., 1:] - A[..., :-1]) / K
            out = d[..., None] + 0.0 * t
            return out.reshape((-1,)) if A.ndim == 1 else out.reshape((A.shape[0], -1))

        mass_c, Gq_c, cq_c = mass[idx], Gq[:, idx], cq[:, idx]
        w_c = self._xi(0.5 * (mass_c[1:] + mass_c[:-1]), xp) * (mass_c[1:] - mass_c[:-1])
        w_c = w_c / w_norm
        m_b = _sub(mass_c)
        b_sub = xp.clip(self.alpha + self.beta / (1.0 + 1.4 / m_b), 0.0, 1.0)
        w_rep = (w_c[:, None] * np.ones(K)).reshape((-1,)) / K
        b = xp.clip(self.alpha + self.beta / (1.0 + 1.4 / mass), 0.0, 1.0)
        g1 = _dk_gamma(m_b, xp, smooth=True) + 1.0
        cdf = [m_b * 0.0] + [m_b * 0.0 + 1.0 if q == 1.0 else q**g1 for q in _Q_NODES[1:]]
        pq = [cdf[k + 1] - cdf[k] for k in range(len(_Q_NODES) - 1)]
        return dict(
            G=G,
            col=col,
            Gq=_sub(Gq_c),
            cq=_sub(cq_c),
            vq=_step(Gq_c),
            vc=_step(cq_c),
            m_b=m_b,
            w_single=w_seg * (1 - 0.5 * (b[1:] + b[:-1])),
            w_bin=[w_rep * b_sub * p for p in pq],
            kG=kG,
            kc=kc,
            kGq=_sub(kGq[:, idx]),
            kcq=_sub(kcq[:, idx]),
        )

    def _segments(self, met, loga, dm, Av, xp=np):
        d = self._deposit(met, loga, dm, Av, xp)
        G, col, Gq, cq = d["G"], d["col"], d["Gq"], d["cq"]
        w = d["w_single"]
        ell = xp.sqrt((G[1:] - G[:-1]) ** 2 + (col[1:] - col[:-1]) ** 2 + 1e-18)
        zero = w[:1] * 0.0
        wl = xp.concatenate([zero, w, zero])
        ll = xp.concatenate([zero, ell, zero])
        rho = (wl[:-1] + wl[1:]) / (ll[:-1] + ll[1:])
        c0 = ell * rho[:-1]
        c1 = ell * (rho[1:] - rho[:-1])
        scale = xp.sum(w) / xp.sum(c0 + 0.5 * c1)
        ks = (0.5 * (d["kG"][1:] + d["kG"][:-1]), 0.5 * (d["kc"][1:] + d["kc"][:-1]))
        single = (G[:-1], col[:-1], G[1:], col[1:], c0 * scale, c1 * scale, ks)
        if not (self.alpha or self.beta):
            return single, None
        nq = len(d["w_bin"])
        on = 1.0 if self.BINARY_SMEAR else 0.0
        kb = (
            (0.5 * (d["kGq"][:-1] + d["kGq"][1:])).reshape((-1,)),
            (0.5 * (d["kcq"][:-1] + d["kcq"][1:])).reshape((-1,)),
        )
        binary = (
            Gq[:-1].reshape((-1,)),
            cq[:-1].reshape((-1,)),
            Gq[1:].reshape((-1,)),
            cq[1:].reshape((-1,)),
            xp.stack(d["w_bin"]).reshape((-1,)),
            on * (0.5 * (d["vq"][:nq] + d["vq"][1:])).reshape((-1,)),
            on * (0.5 * (d["vc"][:nq] + d["vc"][1:])).reshape((-1,)),
            kb,
        )
        return single, binary

    def star_ll(self, met, loga, dm, Av, sigma_int, f_bg, sigma_Av=0.0, xp=np):
        """Per-star log-likelihood ``(N,)``; with ``sigma_Av = 0`` and both switches off it is
        the package's ``_star_loglike`` (checked in ``screen.py check``)."""
        s2 = self.SIGMA_FLOOR**2 + sigma_int**2
        xg, xc = self._obs_mag, self._obs_col
        sg = xp.sqrt(self._e_obs_mag**2 + s2)
        sc = xp.sqrt(self._e_obs_col**2 + s2)
        single, binary = self._segments(met, loga, dm, Av, xp)
        Ag, Ac, Bg, Bc, c0, c1, (kGs, kcs) = single
        v2 = sigma_Av**2
        J0, J1 = seg_density(
            xg,
            xc,
            sg,
            sc,
            Ag,
            Ac,
            Bg,
            Bc,
            xp,
            v2 * kGs**2,
            v2 * kcs**2,
            v2 * kGs * kcs,
            first_moment=True,
        )
        dens = xp.dot(J0, c0) + xp.dot(J1, c1)
        # completeness: G width of each segment, its own reddening spread included
        P, T = self._p_observed(Ag, Bg, s2 + v2 * kGs**2, xp, first_moment=True)
        F = xp.sum(c0 * P + c1 * T)
        if binary is not None:
            Ag, Ac, Bg, Bc, w, vg, vc, (kGb, kcb) = binary
            dens = dens + xp.dot(
                seg_density(
                    xg,
                    xc,
                    sg,
                    sc,
                    Ag,
                    Ac,
                    Bg,
                    Bc,
                    xp,
                    vg**2 / 12 + v2 * kGb**2,
                    vc**2 / 12 + v2 * kcb**2,
                    vg * vc / 12 + v2 * kGb * kcb,
                ),
                w,
            )
            F = F + xp.sum(w * self._p_observed(Ag, Bg, s2 + v2 * kGb**2, xp))
        return xp.log((1 - f_bg) * dens / F + f_bg / self._box_area)
