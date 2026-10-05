r"""Multiband isochrone likelihood: Gaia (G, BP-RP) plus near-infrared magnitudes, per star.

WHY THIS EXISTS
---------------
:class:`~erotica.analysis.IsochroneFitter` fits the Gaia CMD, one magnitude and one colour. A
0.05 mag error in the model's BP alone moves log t by ~0.10 dex and [Fe/H] by ~0.07 on a cluster
like NGC 6383 (hub finding ``isochrone-multiband-nir.md`` §3.3), and the measured colour-table
offset of lambda Ori pushes a solar, 2.5 Myr synthetic cluster to [Fe/H] -0.17
(``isochrone-colour-mechanism.md``). Near-infrared bands (2MASS, or VISTA/VIRAC2 brought to the
2MASS system) add independent measurements with a long colour baseline. This module is that
likelihood; whether it actually reduces the colour bias is measured, not assumed
(``tools/validation/isochrone_multiband/``).

THE MODEL
---------
Observation of star :math:`i` in the coordinates :math:`c = (G,\ BP-RP,\ m_{b_1}, \dots)`
(the two Gaia coordinates of the 2D fitter, then one magnitude per near-infrared band):

.. math::
   \ln\mathcal{L} = \sum_i \omega_i \ln\!\left[(1-f_{\rm bg})\,\frac{\sum_k w_k\,d_{ik}}{F(\theta)}
   + \frac{f_{\rm bg}}{\prod_{c \in \mathcal{C}_i} \Delta_c}\right],\qquad
   d_{ik} = \int_0^1 \mathcal{N}_{n_i}\!\big(x_i;\ A_k + t(B_k - A_k),\ \Sigma_{ik}\big)\,dt,

with :math:`\mathcal{C}_i` the coordinates star :math:`i` has (:math:`n_i` of them),
:math:`\Delta_c` the observed span of coordinate :math:`c`, and

.. math::
   \Sigma_{ik} = \mathrm{diag}\big(e_{ic}^2 + \sigma_{{\rm floor},c}^2 + \sigma_{\rm int}^2\big)
   + \sigma_{A_V}^2\,\bar k_k \bar k_k^\top + \tfrac{1}{12}\,v_k v_k^\top .

Each piece and the process it models (methodology §A.1):

* **Segment integral, closed form in n dimensions.** With :math:`D_i` the diagonal part and
  :math:`U_k` the (at most two) rank-1 vectors, whiten by :math:`D_i` and use Woodbury,
  :math:`(I + \tilde U\tilde U^\top)^{-1} = I - \tilde U (I + \tilde U^\top\tilde U)^{-1}\tilde U^\top`,
  :math:`|I + \tilde U\tilde U^\top| = |I + \tilde U^\top \tilde U|`. The exponent is then the
  quadratic :math:`a + 2bt + ct^2` in the segment parameter and the integral is the one of
  :func:`~erotica.analysis._isochrone._segment_density` with :math:`L^2 = c`,
  :math:`t^* = -b/c`, :math:`\rho^2 = a - b^2/c`. Exact for a straight segment; cost linear in
  the number of coordinates (no n x n Cholesky per star and segment). Error of n photometric bands.
* **Missing bands are marginalised**: a band a star lacks has precision 0, which removes its row
  and column from :math:`\Sigma_{ik}` (the marginal of a Gaussian), and its span from the
  background. This is right when the band is missing for a reason independent of the model
  magnitude: saturation (VIRAC2 J, H < 12.25, Ks < 12.5, replaced by 2MASS when it exists), no
  cross-match, a blend. A **non-detection** (fainter than a limit) would need a censoring factor
  :math:`\Phi((m_{\rm lim}-m)/\sigma)` instead; it is **not implemented**, because no member of
  NGC 6383 lacks a band for that reason (hub finding ``isochrone-multiband-implementation.md``).
* **Differential extinction** (``sigma_av``): each star's :math:`A_V` scatters around the cluster
  value with sd :math:`\sigma_{A_V}`; in the coordinates that is the rank-1 covariance along the
  segment's reddening vector :math:`\bar k_k` (mean of its two ends). Off by default. In the Gaia
  CMD alone :math:`\sigma_{A_V}` is degenerate with :math:`\sigma_{\rm int}` along one direction;
  with the near infrared, where :math:`k` is 3-7 times smaller, it is measured.
* **Binary sheet** spread along the primary-mass step :math:`v_k` (the :math:`v v^\top/12` of the
  2D fitter), now in every coordinate.
* **Extinction per band and per point** (``ext``): ``"ccm"`` = CCM89 + O'Donnell 94 at fixed
  effective wavelengths, constant :math:`k_c` (the 2D fitter's law); ``"fitz19"`` = the ESA Gaia
  EDR3 extinction law (Fitzpatrick et al. 2019, ``2019ApJ...886..108F``; Riello et al. 2021,
  ``2021A&A...649A...3R``): :math:`k_b = A_b/A_0` as a cubic in the **intrinsic** colour
  :math:`X = (BP-RP)_0` of each EEP point and in :math:`A_0 = A_V`, for G, BP, RP and 2MASS J, H,
  Ks (``Fitz19_EDR3_MainSequence.csv``). The coefficient is evaluated at every point of the
  isochrone before the segments are formed, so the extinction varies along the track **inside**
  the segment integral. :math:`X` is clipped to the table's validity range [-0.06, 2.5]; cooler
  pre-main-sequence points take the value at 2.5 [I: the law is calibrated on main-sequence
  SEDs]. In the near infrared the two laws differ a lot: :math:`k_{Ks}` 0.117 (CCM89 at 2.16 µm)
  against ~0.19 (Fitzpatrick+19 through the 2MASS passband).
* **Zero point between instruments** (``zp``): one offset per near-infrared band, added to the
  model magnitude of the stars whose magnitude in that band came from another instrument than the
  model's system (here: VIRAC2 transformed to 2MASS with Gonzalez-Fernandez et al. 2018,
  ``2018MNRAS.474.5459G``). Prior :math:`\mathcal{N}(0, 0.02)` from that paper's stated calibration
  precision ("better than ~2 % in YJHKs"). A global parameter of the cluster, not per star.
* **Floor and intrinsic width per coordinate**, not per band: the Gaia coordinates get
  ``SIGMA_FLOOR`` and :math:`\sigma_{\rm int}` exactly as in the 2D fitter (so a colour gets one
  floor, not two), each near-infrared band its own ``nir_floor`` and the same :math:`\sigma_{\rm int}`.
* **Completeness** :math:`F(\theta)`: the cut is in G only, as in the 2D fitter, with the G width
  of each segment widened by :math:`\sigma_{A_V}^2 \bar k_G^2`. Missing near-infrared bands do not
  enter :math:`F` (missing at random, above).
* **The IMF weight along each segment** is spread with the 2D fitter's continuous linear density,
  with segment lengths measured in the Gaia (G, BP-RP) plane: a parametrisation of how the weight is
  laid along the track, which keeps the Gaia-only case identical to the 2D fitter.

CONTROL (the construction oracle)
---------------------------------
With no near-infrared band, ``ext="ccm"`` and ``sigma_av`` off, this is the 2D likelihood:
``tests/test_isochrone_multiband.py`` holds the two to rounding in NumPy and in the compiled
PyTensor graph (value and gradient), with binaries, an off-node point, f_bg > 0 and a two-sided
window. The code path is the n-dimensional one, never a dispatch to the 2D function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.table import QTable

from .._membership import COLUMNA_ISOCRONA, select_by_probability
from ._isochrone import (
    _Q_NODES,
    IsochroneFitter,
    _ccm89,
    _chabrier2014_xi,
    _dk_gamma,
    _erf_np,
    _error_poly,
    _GridAdapter,
    _mag_combine,
)

# ESA Gaia EDR3 extinction law, main-sequence table, X = (BP-RP)_0 rows ("BPRP"):
# https://www.cosmos.esa.int/web/gaia/edr3-extinction-law, file Fitz19_EDR3_extinctionlawcoefficients.zip
# (sha256 828ee1ce022ea7fa55c1a804389ea4e4a34522eb18064cf06915d41463eabc3b, downloaded 2026-10-05),
# Fitz19_EDR3_MainSequence.csv. k = a1 + a2 X + a3 X^2 + a4 X^3 + a5 A + a6 A^2 + a7 A^3
# + a8 A X + a9 A X^2 + a10 X A^2, with A = A0. kG, kBP, kRP as in the screen's ``fitz19``.
FITZ19_BPRP = {
    "G": (
        0.995969721536602,
        -0.159726460302015,
        0.0122380738156057,
        0.00090726555099859,
        -0.0377160263914123,
        0.00151347495244888,
        -2.52364537395142e-05,
        0.0114522658102451,
        -0.000936914989014318,
        -0.000260296774134201,
    ),
    "BP": (
        1.15363197483424,
        -0.0814012991657388,
        -0.036013023976704,
        0.0192143585568966,
        -0.022397548243016,
        0.000840562680547171,
        -1.31018008013549e-05,
        0.00660124080271006,
        -0.000882247501989453,
        -0.000111215755291684,
    ),
    "RP": (
        0.66320787941067,
        -0.0179847164933981,
        0.000493769449961458,
        -0.00267994405695751,
        -0.00651422146709376,
        3.30179903473159e-05,
        1.57894227641527e-06,
        -7.9800898337247e-05,
        0.000255679812110045,
        1.10476584967393e-05,
    ),
    "J": (
        0.340345410744913,
        -0.00150278241491212,
        -0.000664573369992718,
        0.000417668208427758,
        -0.000156212937136542,
        1.82216155475468e-07,
        2.33928990110555e-09,
        3.41610470017311e-06,
        -2.13961419094946e-07,
        4.78655838716295e-08,
    ),
    "H": (
        0.25505435103948,
        0.000238094101040351,
        -0.0009480815739429,
        0.000286492979762418,
        -6.84496601067957e-05,
        -2.40939063887753e-09,
        7.2663178194737e-10,
        1.87920964708149e-07,
        1.01009401360882e-06,
        1.11538757526755e-08,
    ),
    "K": (
        0.19404852029171,
        -0.000259572240803642,
        0.000495770920767284,
        -0.00026750735610992,
        -2.92327041145282e-05,
        8.01889842444919e-09,
        1.22296149058604e-10,
        6.4675963409363e-08,
        1.50135637674828e-07,
        2.70438735386413e-09,
    ),
}
FITZ19_XRANGE = (-0.06, 2.5)


def fitz19_k(band: str, X: Any, A0: Any, xp: Any = np) -> Any:
    """:math:`A_b/A_0` of the ESA EDR3 law (see :data:`FITZ19_BPRP`) at intrinsic colour ``X``."""
    a = FITZ19_BPRP[band]
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


@dataclass(frozen=True)
class NIRBand:
    """One near-infrared band: model column, observed columns, extinction and zero-point group.

    ``group_column`` / ``group_values``: a star's magnitude in this band is shifted by this band's
    zero-point parameter when ``row[group_column]`` is one of ``group_values`` (the stars measured
    by the other instrument). ``None`` = no zero-point parameter for this band.
    """

    model: str
    obs: str
    err: str
    effl: float
    fitz19: str
    group_column: str | None = None
    group_values: tuple[str, ...] = ()


#: 2MASS J, H, Ks (MIST/PARSEC column names), observed as ``*_best`` of the VIRAC2 + 2MASS member
#: table; VIRAC2-sourced magnitudes get the zero-point. lambda_eff from Cohen et al. 2003
#: (``2003AJ....126.1090C``): 1.235, 1.662, 2.159 µm.
TWOMASS_BEST = (
    NIRBand("2MASS_J", "J_best", "eJ_best", 12350.0, "J", "src_J", ("VIRAC2->2MASS",)),
    NIRBand("2MASS_H", "H_best", "eH_best", 16620.0, "H", "src_H", ("VIRAC2->2MASS",)),
    NIRBand("2MASS_Ks", "Ks_best", "eKs_best", 21590.0, "K", "src_Ks", ("VIRAC2->2MASS",)),
)


def _column(t: QTable, name: str) -> np.ndarray:
    """A float column with masked entries as NaN (``np.asarray`` of a masked column returns the
    fill value, 0 -- the bug that produced Ks = 0.0 in the VIRAC2 preprocessing)."""
    c = t[name]
    if hasattr(c, "filled"):
        c = c.filled(np.nan)
    return np.asarray(c, dtype=float)


def _segment_density_nd(
    X: Any,
    sw: Any,
    A: list,
    B: list,
    U: list,
    xp: Any = np,
    first_moment: bool = False,
) -> Any:
    r"""Density of each star from a uniform deposit along each segment, in n coordinates.

    ``X``: list of n observed coordinates, each ``(N,)`` (any value where missing);
    ``sw``: list of n ``(N,)`` square roots of the diagonal precision (0 where the star lacks the
    coordinate); ``A``, ``B``: lists of n ``(S,)`` segment ends; ``U``: list of rank-1 vectors, each
    a list of n ``(S,)`` arrays (at most two vectors). Returns ``(N, S)`` (and the first moment in
    the segment parameter with ``first_moment``). Formula: module docstring. The normalisation
    :math:`(2\pi)^{-n_i/2}\prod_c s_{ic}\,|I + \tilde U^\top\tilde U|^{-1/2}` uses each star's own
    :math:`n_i`.
    """
    n = len(X)
    r = len(U)
    yy = yd = dd = 0.0
    uy = [0.0] * r
    ud = [0.0] * r
    uu = [[0.0] * r for _ in range(r)]
    n_i = 0.0
    log_sw = 0.0
    for c in range(n):
        s = sw[c][:, None]
        y = (A[c][None, :] - X[c][:, None]) * s
        d = (B[c] - A[c])[None, :] * s
        yy = yy + y**2
        yd = yd + y * d
        dd = dd + d**2
        us = [Uj[c][None, :] * s for Uj in U]
        for j in range(r):
            uy[j] = uy[j] + us[j] * y
            ud[j] = ud[j] + us[j] * d
            for k in range(j, r):
                uu[j][k] = uu[j][k] + us[j] * us[k]
        n_i = n_i + xp.where(sw[c] > 0, 1.0, 0.0)
        log_sw = log_sw + xp.log(xp.where(sw[c] > 0, sw[c], 1.0))
    a, b, cc = yy, yd, dd
    if r == 1:
        det = 1.0 + uu[0][0]
        a = a - uy[0] ** 2 / det
        b = b - uy[0] * ud[0] / det
        cc = cc - ud[0] ** 2 / det
    elif r == 2:
        c00, c11, c01 = 1.0 + uu[0][0], 1.0 + uu[1][1], uu[0][1]
        det = c00 * c11 - c01**2
        i00, i11, i01 = c11 / det, c00 / det, -c01 / det

        def q(p: Any, s: Any) -> Any:
            return i00 * p[0] * s[0] + i11 * p[1] * s[1] + i01 * (p[0] * s[1] + p[1] * s[0])

        a = a - q(uy, uy)
        b = b - q(uy, ud)
        cc = cc - q(ud, ud)
    elif r > 2:
        raise ValueError("at most two rank-1 covariance terms")
    long_ = cc > 1e-8
    cs = xp.where(long_, cc, 1.0)  # safe value: no NaN in the branch not taken, nor its gradient
    L = xp.sqrt(cs)
    t0 = -b / cs
    perp2 = a - b**2 / cs
    erf = xp.erf if xp is not np else _erf_np
    cdf_diff = 0.5 * (erf(L * (1 - t0) / np.sqrt(2.0)) - erf(-L * t0 / np.sqrt(2.0)))
    I0 = np.sqrt(2 * np.pi) / L * cdf_diff
    seg = xp.exp(-0.5 * perp2) * I0
    point = xp.exp(-0.5 * a)
    lognorm = (log_sw - 0.5 * n_i * np.log(2 * np.pi))[:, None]
    if r:
        lognorm = lognorm - 0.5 * xp.log(det)
    norm = xp.exp(lognorm)
    J0 = xp.where(long_, seg, point) * norm
    if not first_moment:
        return J0
    I1 = t0 * I0 + (xp.exp(-0.5 * cs * t0**2) - xp.exp(-0.5 * cs * (1 - t0) ** 2)) / cs
    J1 = xp.where(long_, xp.exp(-0.5 * perp2) * I1, 0.5 * point) * norm
    return J0, J1


class _MultibandAdapter(_GridAdapter):
    """The grid's node arrays for (G, BP, RP) plus the near-infrared bands."""

    def __init__(self, grid: Any, gaia: tuple[str, str, str], nir: tuple[str, ...]) -> None:
        super().__init__(grid, *gaia)
        missing = [b for b in nir if b not in grid.bands]
        if missing:
            raise ValueError(f"{grid.name} does not provide bands {missing}; it has {grid.bands}")
        self.nir = nir

    def get_bands(self, feh: float, loga: float) -> tuple[np.ndarray, np.ndarray, list]:
        nd = self.grid.node(feh, loga)
        return nd.eep, nd.mass, [nd.mags[b] for b in (*self.bands, *self.nir)]


class MultibandIsochroneFitter(IsochroneFitter):
    r""":class:`IsochroneFitter` in the coordinates (G, BP-RP, near-infrared magnitudes).

    See the module docstring for the likelihood. Grid path only (``grid=``), since the legacy MIST
    reader keeps three bands.

    Parameters (beyond :class:`IsochroneFitter`'s)
    ----------------------------------------------
    nir_bands : sequence of :class:`NIRBand`, default ``()``
        The near-infrared bands. Empty = the Gaia-only control.
    ext : {"ccm", "fitz19"}, default "ccm"
        Extinction law (module docstring).
    sigma_av : bool, default False
        Fit the differential-extinction width :math:`\sigma_{A_V}` (rank-1 covariance).
    zp_sigma : float, default 0.02
        Prior sd of each near-infrared zero point, in mag (Gonzalez-Fernandez+2018).
    nir_floor : float, optional
        Forward-model floor of the near-infrared coordinates; default the Gaia ``SIGMA_FLOOR``
        (the EEP-interpolation error, the same order in every band).

    Parameter vector everywhere: ``(met, loga, dm, Av, sigma_int, f_bg, sigma_av, zp_1, ...)``,
    ``sigma_av`` present only if fitted, one ``zp`` per band with a group column.
    """

    LIKELIHOOD_VERSION = "unbinned-eep-multiband-v1"

    def __init__(
        self,
        *,
        grid: Any,
        nir_bands: tuple[NIRBand, ...] | list[NIRBand] = (),
        ext: str = "ccm",
        sigma_av: bool = False,
        zp_sigma: float = 0.02,
        nir_floor: float | None = None,
        **kw: Any,
    ) -> None:
        if grid is None:
            raise ValueError("the multiband fitter needs a grid backend (grid=...)")
        if ext not in ("ccm", "fitz19"):
            raise ValueError(f"ext must be 'ccm' or 'fitz19', not {ext!r}")
        super().__init__(grid=grid, **kw)
        self.nir_bands = tuple(nir_bands)
        self.ext = ext
        self.fit_sigma_av = bool(sigma_av)
        self.zp_sigma = float(zp_sigma)
        self.nir_floor = float(self.SIGMA_FLOOR if nir_floor is None else nir_floor)
        self.zp_bands = [b for b in self.nir_bands if b.group_column is not None]

    # -- bookkeeping ---------------------------------------------------------------------------

    @property
    def n_coord(self) -> int:
        return 2 + len(self.nir_bands)

    @property
    def param_names(self) -> list[str]:
        names = [self._met_name, "loga", "dm", "Av", "sigma_int", "f_bg"]
        if self.fit_sigma_av:
            names.append("sigma_av")
        return names + [f"zp_{b.model}" for b in self.zp_bands]

    def _split(self, theta: Any) -> tuple:
        """``(met, loga, dm, Av, sigma_int, f_bg, sigma_av, zp list)`` from a parameter vector."""
        met, loga, dm, Av, s, f = (theta[j] for j in range(6))
        i = 6
        sav = 0.0
        if self.fit_sigma_av:
            sav = theta[6]
            i = 7
        zp = [theta[i + j] for j in range(len(self.zp_bands))]
        return met, loga, dm, Av, s, f, sav, zp

    # -- setup -------------------------------------------------------------------------------

    def setup(
        self,
        cluster_data: QTable,
        *,
        prob_threshold: float = 0.6,
        probability_column: str = COLUMNA_ISOCRONA,
        precompute_grid: bool = True,
        pms_column: str | None = None,
        pms_max: float = 0.5,
        ms_weight: float = 1.0,
        mag_window: tuple[float, float] | None = None,
    ) -> None:
        """As :meth:`IsochroneFitter.setup`, plus the near-infrared columns of the same members.

        A band is missing for a star when its magnitude or error is not finite or the error is not
        positive (no error model fills it: a missing band is marginalised, not imputed).
        """
        super().setup(
            cluster_data,
            prob_threshold=prob_threshold,
            probability_column=probability_column,
            precompute_grid=False,
            pms_column=pms_column,
            pms_max=pms_max,
            ms_weight=ms_weight,
            mag_window=mag_window,
        )
        self._isochs = _MultibandAdapter(
            self.grid,
            (self.magnitude, *self.color),
            tuple(b.model for b in self.nir_bands),
        )
        members = select_by_probability(cluster_data, probability_column, prob_threshold)
        if mag_window is not None:
            g_all = np.asarray(members[self.obs_columns[0]], dtype=float)
            members = members[(g_all >= mag_window[0]) & (g_all <= mag_window[1])]
        if len(members) != self._N_obs:
            raise RuntimeError("member selection differs from the parent's")
        self._nir_obs, self._nir_err, self._nir_has, self._nir_group = [], [], [], []
        self._nir_ecoef, self._nir_span = [], []
        for b in self.nir_bands:
            m, e = _column(members, b.obs), _column(members, b.err)
            has = np.isfinite(m) & np.isfinite(e) & (e > 0)
            self._nir_has.append(has)
            self._nir_obs.append(np.where(has, m, 0.0))
            self._nir_err.append(np.where(has, e, 1.0))
            if b.group_column is not None:
                g = np.asarray(members[b.group_column]).astype(str)
                self._nir_group.append(np.isin(g, b.group_values) & has)
            else:
                self._nir_group.append(np.zeros(len(members), bool))
            self._nir_ecoef.append(_error_poly(m[has], e[has]) if has.sum() >= 1 else None)
            self._nir_span.append(max(float(np.ptp(m[has])), 1e-3) if has.any() else 1.0)
        # background density over each star's own coordinates: 1 / prod of the spans it has
        span_g = max(float(np.ptp(self._obs_mag)), 1e-3)
        span_c = max(float(np.ptp(self._obs_col)), 1e-3)
        log_area = np.full(self._N_obs, np.log(span_g) + np.log(span_c))
        for has, sp in zip(self._nir_has, self._nir_span, strict=True):
            log_area = log_area + np.where(has, np.log(sp), 0.0)
        self._log_area = log_area
        self._k_nir_ccm = [_ccm89(b.effl, self.Rv) for b in self.nir_bands]
        if precompute_grid:
            self._build_nodes()

    # -- node table --------------------------------------------------------------------------

    def _build_nodes(self) -> None:
        """Rows per node: mass; the n coordinates; then the n coordinates of the unresolved pair
        at each mass-ratio node (companion mass clipped to the node's lowest mass), all on one
        integer EEP axis, clamped outside each node's EEPs like :meth:`IsochroneFitter._build_nodes`."""
        iso = self._isochs
        if iso is None:
            raise RuntimeError("Call setup() first.")
        zs = np.asarray(iso._met_values, float)
        ages = np.asarray(iso._loga_values, float)
        lo, hi = self.loga_range
        a0 = int(np.clip(np.searchsorted(ages, lo, side="right") - 1, 0, len(ages) - 1))
        a1 = int(np.clip(np.searchsorted(ages, hi, side="left"), 0, len(ages) - 1))
        ages = ages[a0 : a1 + 1]
        if lo < ages[0] - 1e-9 or hi > ages[-1] + 1e-9:
            raise ValueError(f"loga_range {self.loga_range} outside [{ages[0]}, {ages[-1]}]")

        def coords(mags: list) -> list:
            G, BP, RP, *nir = mags
            return [G, BP - RP, *nir]

        per_node = {}
        e_lo, e_hi = np.inf, -np.inf
        for z in zs:
            for a in ages:
                eep, mass, mags = iso.get_bands(float(z), float(a))
                ms = np.argsort(mass, kind="stable")
                rows = [mass, *coords(mags)]
                for q in _Q_NODES:
                    m2 = np.clip(q * mass, mass.min(), None)
                    comp = [np.interp(m2, mass[ms], x[ms]) for x in mags]
                    rows += coords([_mag_combine(x, y) for x, y in zip(mags, comp, strict=True)])
                per_node[(z, a)] = (eep, rows)
                e_lo, e_hi = min(e_lo, eep.min()), max(e_hi, eep.max())
        axis = np.arange(int(e_lo), int(e_hi) + 1, dtype=float)
        n_rows = 1 + self.n_coord * (1 + len(_Q_NODES))
        T = np.empty((len(zs), len(ages), n_rows, len(axis)))
        for i, z in enumerate(zs):
            for j, a in enumerate(ages):
                eep, rows = per_node[(z, a)]
                for r, x in enumerate(rows):
                    T[i, j, r] = np.interp(axis, eep, x)
        if np.any(np.diff(T[:, :, 0], axis=-1) < 0):
            raise ValueError("initial mass decreases along EEP in some isochrone node")
        self._set_nodes(T, zs, ages)
        print(f"  EEP node table: {T.shape} (feh x age x rows x EEP)")

    def save_grid(self, path: Any) -> None:  # pragma: no cover - the 2D cache format
        raise NotImplementedError("the multiband node table is rebuilt, not cached")

    # -- extinction --------------------------------------------------------------------------

    def _kvec(self, col0: Any, Av: Any, xp: Any) -> list:
        """:math:`k_c` of each coordinate at the intrinsic colours ``col0`` (any shape)."""
        if self.ext == "ccm":
            return [self._kG + 0.0 * col0, self._k_col1 + 0.0 * col0] + [
                k + 0.0 * col0 for k in self._k_nir_ccm
            ]
        kG = fitz19_k("G", col0, Av, xp)
        kc = fitz19_k("BP", col0, Av, xp) - fitz19_k("RP", col0, Av, xp)
        return [kG, kc] + [fitz19_k(b.fitz19, col0, Av, xp) for b in self.nir_bands]

    # -- forward model -----------------------------------------------------------------------

    def _deposit_nd(self, met: Any, loga: Any, dm: Any, Av: Any, xp: Any = np) -> dict:
        """Apparent coordinates of the singles and of the binary sheet, and their weights; the
        n-coordinate version of :meth:`IsochroneFitter._deposit` (same weights, same sub-sampling
        of the sheet)."""
        n = self.n_coord
        X = self._interp_isochrone(met, loga, xp)
        mass = xp.maximum(X[0], 1e-9)
        is_mag = [1.0, 0.0] + [1.0] * len(self.nir_bands)
        single0 = [X[1 + c] for c in range(n)]
        k_s = self._kvec(single0[1], Av, xp)
        single = [single0[c] + is_mag[c] * dm + k_s[c] * Av for c in range(n)]
        nq = len(_Q_NODES)
        pair0 = [[X[1 + n * (1 + j) + c] for c in range(n)] for j in range(nq)]
        k_p = [self._kvec(pair0[j][1], Av, xp) for j in range(nq)]
        pair = [
            xp.stack([pair0[j][c] + is_mag[c] * dm + k_p[j][c] * Av for j in range(nq)])
            for c in range(n)
        ]  # per coordinate: (n_q, n_eep)
        kpair = [xp.stack([k_p[j][c] for j in range(nq)]) for c in range(n)]

        m_mid = 0.5 * (mass[1:] + mass[:-1])
        w_seg = _chabrier2014_xi(m_mid, xp) * (mass[1:] - mass[:-1])
        w_norm = xp.sum(w_seg)
        w_seg = w_seg / w_norm
        K = self.BINARY_SUBSAMPLE
        t = (np.arange(K) + 0.5) / K
        n_e = self._nodes.shape[-1]
        idx = np.unique(np.r_[np.arange(0, n_e, self.BINARY_STRIDE), n_e - 1])

        def _sub(A: Any) -> Any:
            a0, a1 = A[..., :-1], A[..., 1:]
            out = a0[..., None] + (a1 - a0)[..., None] * t
            return out.reshape((-1,)) if A.ndim == 1 else out.reshape((A.shape[0], -1))

        def _step(A: Any) -> Any:
            d = (A[..., 1:] - A[..., :-1]) / K
            out = d[..., None] + 0.0 * t
            return out.reshape((-1,)) if A.ndim == 1 else out.reshape((A.shape[0], -1))

        mass_c = mass[idx]
        w_c = _chabrier2014_xi(0.5 * (mass_c[1:] + mass_c[:-1]), xp) * (mass_c[1:] - mass_c[:-1])
        w_c = w_c / w_norm
        m_b = _sub(mass_c)
        b_sub = xp.clip(self.alpha + self.beta / (1.0 + 1.4 / m_b), 0.0, 1.0)
        w_rep = (w_c[:, None] * np.ones(K)).reshape((-1,)) / K
        b = xp.clip(self.alpha + self.beta / (1.0 + 1.4 / mass), 0.0, 1.0)
        g1 = _dk_gamma(m_b, xp, smooth=True) + 1.0
        cdf = [m_b * 0.0] + [m_b * 0.0 + 1.0 if q == 1.0 else q**g1 for q in _Q_NODES[1:]]
        pq = [cdf[k + 1] - cdf[k] for k in range(nq - 1)]
        return dict(
            single=single,
            k_single=k_s,
            pair=[_sub(p[:, idx]) for p in pair],
            k_pair=[_sub(k[:, idx]) for k in kpair],
            v_pair=[_step(p[:, idx]) for p in pair],
            m_b=m_b,
            w_single=w_seg * (1 - 0.5 * (b[1:] + b[:-1])),
            w_bin=[w_rep * b_sub * p for p in pq],
        )

    def _segments_nd(self, met: Any, loga: Any, dm: Any, Av: Any, xp: Any = np) -> tuple:
        """``(single, binary)``: singles ``(A, B, c0, c1, kbar)`` with the 2D fitter's continuous
        linear density (lengths in the Gaia plane); binary pieces ``(A, B, w, v, kbar)`` or
        ``None``. A, B, v, kbar are lists over coordinates."""
        d = self._deposit_nd(met, loga, dm, Av, xp)
        n = self.n_coord
        S = d["single"]
        w = d["w_single"]
        G, col = S[0], S[1]
        ell = xp.sqrt((G[1:] - G[:-1]) ** 2 + (col[1:] - col[:-1]) ** 2 + 1e-18)
        zero = w[:1] * 0.0
        wl = xp.concatenate([zero, w, zero])
        ll = xp.concatenate([zero, ell, zero])
        rho = (wl[:-1] + wl[1:]) / (ll[:-1] + ll[1:])
        c0 = ell * rho[:-1]
        c1 = ell * (rho[1:] - rho[:-1])
        scale = xp.sum(w) / xp.sum(c0 + 0.5 * c1)
        ks = d["k_single"]
        single = (
            [S[c][:-1] for c in range(n)],
            [S[c][1:] for c in range(n)],
            c0 * scale,
            c1 * scale,
            [0.5 * (ks[c][1:] + ks[c][:-1]) for c in range(n)],
        )
        if not (self.alpha or self.beta):
            return single, None
        nq = len(d["w_bin"])
        on = 1.0 if self.BINARY_SMEAR else 0.0
        P, Kp, V = d["pair"], d["k_pair"], d["v_pair"]
        binary = (
            [P[c][:-1].reshape((-1,)) for c in range(n)],
            [P[c][1:].reshape((-1,)) for c in range(n)],
            xp.stack(d["w_bin"]).reshape((-1,)),
            [on * (0.5 * (V[c][:nq] + V[c][1:])).reshape((-1,)) for c in range(n)],
            [(0.5 * (Kp[c][:-1] + Kp[c][1:])).reshape((-1,)) for c in range(n)],
        )
        return single, binary

    # -- likelihood --------------------------------------------------------------------------

    def _obs_lists(self, sigma_int: Any, zp: list, xp: Any) -> tuple[list, list]:
        """Observed coordinates (shifted by the zero points) and the square-root precisions."""
        s2g = self.SIGMA_FLOOR**2 + sigma_int**2
        s2n = self.nir_floor**2 + sigma_int**2
        X = [self._obs_mag, self._obs_col]
        sw = [
            1.0 / xp.sqrt(self._e_obs_mag**2 + s2g),
            1.0 / xp.sqrt(self._e_obs_col**2 + s2g),
        ]
        zi = 0
        for j, b in enumerate(self.nir_bands):
            x = self._nir_obs[j]
            if b.group_column is not None:
                x = x - zp[zi] * self._nir_group[j]
                zi += 1
            X.append(x)
            sw.append(self._nir_has[j] / xp.sqrt(self._nir_err[j] ** 2 + s2n))
        return X, sw

    def _star_loglike_nd(self, theta: Any, xp: Any = np) -> Any:
        """Per-star log-likelihood ``(N,)`` at the parameter vector ``theta`` (no weights)."""
        met, loga, dm, Av, sigma_int, f_bg, sav, zp = self._split(theta)
        X, sw = self._obs_lists(sigma_int, zp, xp)
        single, binary = self._segments_nd(met, loga, dm, Av, xp)
        A, B, c0, c1, ks = single
        U = [[sav * k for k in ks]] if self.fit_sigma_av else []
        J0, J1 = _segment_density_nd(X, sw, A, B, U, xp, first_moment=True)
        dens = xp.dot(J0, c0) + xp.dot(J1, c1)
        if binary is not None:
            A, B, w, v, kb = binary
            Ub = [[vc / np.sqrt(12.0) for vc in v]]
            if self.fit_sigma_av:
                Ub.append([sav * k for k in kb])
            dens = dens + xp.dot(_segment_density_nd(X, sw, A, B, Ub, xp), w)
        F = self._detected_fraction_nd(sigma_int, sav, single, binary, xp)
        return xp.log((1 - f_bg) * dens / F + f_bg * xp.exp(-self._log_area))

    def _detected_fraction_nd(
        self, sigma_int: Any, sav: Any, single: Any, binary: Any, xp: Any
    ) -> Any:
        s2 = self.SIGMA_FLOOR**2 + sigma_int**2
        A, B, c0, c1, ks = single
        P, T = self._p_observed(A[0], B[0], s2 + sav**2 * ks[0] ** 2, xp, first_moment=True)
        F = xp.sum(c0 * P + c1 * T)
        if binary is not None:
            A, B, w, _, kb = binary
            F = F + xp.sum(w * self._p_observed(A[0], B[0], s2 + sav**2 * kb[0] ** 2, xp))
        return F

    def loglike_theta(self, theta: Any) -> float:
        """Total (weighted) log-likelihood at the parameter vector (NumPy)."""
        v = self._star_loglike_nd(np.asarray(theta, float), np)
        return float(np.sum(self._star_weights * v))

    def loglike(
        self,
        met: float,
        loga: float,
        dm: float,
        Av: float,
        sigma_int: float = 0.0,
        f_bg: float = 0.0,
        sigma_av: float = 0.0,
        zp: tuple[float, ...] | None = None,
    ) -> float:
        theta = [met, loga, dm, Av, sigma_int, f_bg]
        if self.fit_sigma_av:
            theta.append(sigma_av)
        theta += list(zp) if zp is not None else [0.0] * len(self.zp_bands)
        return self.loglike_theta(theta)

    def log_prior_extra(self, theta: Any) -> float:
        """Log prior of ``dm`` (truncated normal, up to a constant) and of the zero points: the
        terms a MAP adds to the likelihood (as ``tools/validation/isochrone_colour/common.search``)."""
        z = (theta[2] - self.dm_mu) / self.dm_sigma
        lp = -0.5 * z * z
        n0 = 7 if self.fit_sigma_av else 6
        for j in range(len(self.zp_bands)):
            lp -= 0.5 * (theta[n0 + j] / self.zp_sigma) ** 2
        return float(lp)

    def _compiled_loglike_nd(self, mode: str | None = None) -> Any:
        """Compiled ``theta -> (log L, d log L / d theta)`` over :attr:`param_names`."""
        import pytensor
        import pytensor.tensor as pt

        v = pt.dvector("theta")
        ll = pt.sum(self._star_weights * self._star_loglike_nd(v, pt))
        fn = pytensor.function([v], [ll, pytensor.grad(ll, v)], mode=mode)
        return lambda x: (lambda o: (float(o[0]), np.asarray(o[1], float)))(
            fn(np.asarray(x, float))
        )

    def _compiled_loglike(self, mode: str | None = None) -> Any:
        """The base class's six-parameter interface (``sigma_av`` = 0, zero points 0)."""
        f = self._compiled_loglike_nd(mode)
        extra = len(self.param_names) - 6

        def six(x: Any) -> tuple:
            v, g = f(np.r_[np.asarray(x, float), np.zeros(extra)])
            return v, g[:6]

        return six

    # -- PyMC model --------------------------------------------------------------------------

    def build_model(self) -> Any:
        """PyMC model with the parent's priors plus ``sigma_av ~ HalfNormal(0.3)`` (if fitted;
        the 0.3 is the lower end of the screen's measured diffred range, not a fit) and
        ``zp_b ~ Normal(0, zp_sigma)``."""
        from ._isochrone import _require_pymc

        pm = _require_pymc()
        import pytensor.tensor as pt

        met_min, met_max = self._met_bounds()
        weights = self._star_weights
        with pm.Model() as model:
            if len(self._node_logz) > 1:
                met = pm.Uniform(self._met_name, lower=met_min, upper=met_max)
            else:
                met = pm.Deterministic(self._met_name, pt.as_tensor_variable(met_min))
            loga = pm.Uniform("loga", lower=self.loga_range[0], upper=self.loga_range[1])
            dm = pm.TruncatedNormal(
                "dm",
                mu=self.dm_mu,
                sigma=self.dm_sigma,
                lower=self.dm_range[0],
                upper=self.dm_range[1],
            )
            Av = pm.Uniform("Av", lower=self.Av_range[0], upper=self.Av_range[1])
            sigma_int = pm.HalfNormal("sigma_int", sigma=0.05)
            f_bg = pm.Beta("f_bg", alpha=1.0, beta=19.0)
            parts = [met, loga, dm, Av, sigma_int, f_bg]
            if self.fit_sigma_av:
                parts.append(pm.HalfNormal("sigma_av", sigma=0.3))
            for b in self.zp_bands:
                parts.append(pm.Normal(f"zp_{b.model}", mu=0.0, sigma=self.zp_sigma))
            theta = pt.stack([pt.as_tensor_variable(p) for p in parts])
            pm.Potential("loglike", pt.sum(weights * self._star_loglike_nd(theta, pt)))
        return model

    # -- generator ---------------------------------------------------------------------------

    def draw_stars_nd(
        self,
        theta: Any,
        n: int,
        rng: np.random.Generator,
        *,
        nir_pattern: dict | None = None,
    ) -> dict:
        """``n`` observed stars from the forward model (the n-coordinate :meth:`_draw_stars`).

        Errors: Gaia from the fitted error models at G; each near-infrared band from a quadratic
        log-error fit to the members' own errors in that band, at the model magnitude (clipped to
        the observed range). ``nir_pattern``, if given, is ``{"has": [(n_real,) bool per band],
        "group": [...], "G": (n_real,)}`` of the real members: each synthetic star takes the
        missing-band and zero-point-group pattern of the real star at the same rank in G, so the
        synthetic sample has the real selection in the bands (methodology §A.1.3).
        Returns a dict of columns (``Gmag``, ``BP_RP``, ``e_Gmag``, ``e_BP_RP`` and per band
        ``obs``/``err``/group) ready for :meth:`setup`.
        """
        met, loga, dm, Av, sigma_int, _, sav, zp = self._split(np.asarray(theta, float))
        d = self._deposit_nd(met, loga, dm, Av, np)
        g1 = _dk_gamma(d["m_b"]) + 1.0
        w_single = d["w_single"]
        p = np.concatenate([w_single, sum(d["w_bin"])])
        p = np.clip(p, 0, None) / np.clip(p, 0, None).sum()
        nc = self.n_coord
        out: list[list] = [[] for _ in range(nc)]
        s2g = self.SIGMA_FLOOR**2 + sigma_int**2
        s2n = self.nir_floor**2 + sigma_int**2
        lo, hi = float(self._obs_mag.min()), float(self._obs_mag.max())
        got = 0
        for _ in range(100):
            if got >= n:
                break
            k = rng.choice(len(p), size=4 * n, p=p)
            single = k < len(w_single)
            ks, t = k[single], rng.uniform(size=int(single.sum()))
            kb = k[~single] - len(w_single)
            q = rng.uniform(size=kb.size) ** (1.0 / g1[kb])
            j = np.clip(np.searchsorted(_Q_NODES, q, side="right") - 1, 0, len(_Q_NODES) - 2)
            f = (q - _Q_NODES[j]) / (_Q_NODES[j + 1] - _Q_NODES[j])
            true = []
            for c in range(nc):
                v = np.empty(k.size)
                S, P = d["single"][c], d["pair"][c]
                v[single] = S[ks] + t * (S[ks + 1] - S[ks])
                v[~single] = P[j, kb] + f * (P[j + 1, kb] - P[j, kb])
                true.append(v)
            if sav:
                da = rng.normal(size=k.size) * sav
                kv = self._kvec(true[1], Av, np)
                true = [true[c] + kv[c] * da for c in range(nc)]
            Gc = np.clip(true[0], lo, hi)
            eg = np.sqrt(self._e_mag_fn(Gc) ** 2 + s2g)
            ec = np.sqrt(self._e_col_fn(Gc) ** 2 + s2g)
            obs = [true[0] + rng.normal(size=k.size) * eg, true[1] + rng.normal(size=k.size) * ec]
            errs = [self._e_mag_fn(Gc), self._e_col_fn(Gc)]
            for jb in range(len(self.nir_bands)):
                coef = self._nir_ecoef[jb]
                has = self._nir_has[jb]
                m_lo, m_hi = (
                    float(self._nir_obs[jb][has].min()),
                    float(self._nir_obs[jb][has].max()),
                )
                mc = np.clip(true[2 + jb], m_lo, m_hi)
                e = 10.0 ** np.polynomial.polynomial.polyval(mc, coef)
                obs.append(true[2 + jb] + rng.normal(size=k.size) * np.sqrt(e**2 + s2n))
                errs.append(e)
            keep = obs[0] <= self._mag_lim
            if self._mag_bright is not None:
                keep &= obs[0] >= self._mag_bright
            for c in range(nc):
                out[c].append(np.c_[obs[c][keep], errs[c][keep]])
            got += int(keep.sum())
        arr = [np.concatenate(o)[:n] for o in out]
        cols = {
            "Gmag": arr[0][:, 0],
            "BP_RP": arr[1][:, 0],
            "e_Gmag": arr[0][:, 1],
            "e_BP_RP": arr[1][:, 1],
        }
        m = len(arr[0])
        if nir_pattern is not None:
            rank_real = np.argsort(np.argsort(nir_pattern["G"]))
            order_real = np.argsort(nir_pattern["G"])
            rank_syn = np.argsort(np.argsort(cols["Gmag"]))
            pick = order_real[
                np.clip(
                    np.round(rank_syn * (len(rank_real) - 1) / max(m - 1, 1)).astype(int),
                    0,
                    len(rank_real) - 1,
                )
            ]
        zi = 0
        for jb, b in enumerate(self.nir_bands):
            mag, err = arr[2 + jb][:, 0].copy(), arr[2 + jb][:, 1].copy()
            grp = np.zeros(m, bool)
            if nir_pattern is not None:
                has = np.asarray(nir_pattern["has"][jb], bool)[pick]
                grp = np.asarray(nir_pattern["group"][jb], bool)[pick]
                mag[~has], err[~has] = np.nan, np.nan
            if b.group_column is not None:
                mag = mag + zp[zi] * grp  # a nonzero true zero point shifts the group's stars
                zi += 1
            cols[b.obs], cols[b.err] = mag, err
            if b.group_column is not None:
                cols[b.group_column] = np.where(grp, b.group_values[0], "ref")
        return cols


def synthetic_table(cols: dict) -> QTable:
    """A member table for :meth:`MultibandIsochroneFitter.setup` from :meth:`draw_stars_nd`
    output: BP-RP is stored as ``G_BPmag`` with ``G_RPmag`` = 0 and ``e_BP_RP`` as its error (the
    convention of the colour experiment)."""
    t = QTable({k: v for k, v in cols.items() if k not in ("BP_RP", "e_BP_RP")})
    t["G_BPmag"] = cols["BP_RP"]
    t["G_RPmag"] = np.zeros_like(cols["BP_RP"])
    t["e_G_BPmag"] = cols["e_BP_RP"]
    t["e_G_RPmag"] = np.zeros_like(cols["BP_RP"])
    t["e_BP_RP"] = cols["e_BP_RP"]
    t["probability_hdbscan"] = np.ones(len(cols["Gmag"]))
    return t


__all__ = [
    "FITZ19_BPRP",
    "MultibandIsochroneFitter",
    "NIRBand",
    "TWOMASS_BEST",
    "fitz19_k",
    "synthetic_table",
]
