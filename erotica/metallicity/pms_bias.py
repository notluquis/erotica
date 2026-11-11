"""Empirical bias of ASPCAP [Fe/H] in young late-G/K (pre-main-sequence) stars.

ASPCAP (APOGEE DR17) returns [Fe/H] too low for young K stars. In the calibration sample (104 stars
in 12 clusters younger than ~10 Myr) the median of ``fe_h - anchor``, with the Pleiades + alpha Per
G-dwarf anchor at +0.021, is -0.17 dex in the 82 stars without infrared excess (Ks - [4.5] < 0.4)
and -0.45 dex in the 22 with it. This module models that deficit as

    delta = fe_h - anchor
          = a + b * (logg - 4) + c * (teff - 4800) / 500 + d * max(Ks - [4.5], 0),

where ``logg`` is **ASPCAP's** surface gravity and ``Ks - [4.5]`` the mid-infrared excess (Spitzer
IRAC [4.5]; WISE W2 where [4.5] is missing). The correction to apply is ``-delta``:
``fe_h_corrected = fe_h_aspcap - delta``.

Domain
------
T_eff 4000-5500 K, ASPCAP log g 3.0-4.8, Ks - [4.5] from -0.2 to 2 mag, ASPCAP DR17 (DR19 gives the
same values star by star). Outside it :func:`predict_bias` warns and flags the star.

What the model is not
---------------------
It is **empirical**. Its terms are not mechanisms: the log g term does not mean that surface
gravity biases the abundance. In the hub's young-K sample the infrared excess correlates with
ASPCAP's log g (rho = -0.78, N 122) and not with the physical log g from the Gaia luminosity
(+0.02), and, inside a
cluster at fixed T_eff, [Fe/H] follows ASPCAP's log g (+0.26 dex per dex in stars without excess)
and not the physical one. Whatever veils the spectrum lowers ASPCAP's log g and [Fe/H] together.

**So the regressor is the symptom, and the model must not be applied blindly.** It is valid only
with the log g that ASPCAP returned for the same spectrum; a log g from isochrones, asteroseismology
or another pipeline is a different variable and gives a wrong correction. That is why
``logg_source`` is a required argument of :func:`predict_bias`.

A second limit: the slope ``b`` is learned mostly from stars *with* a disc at low log g, where
ASPCAP's log g carries the veiling. No calibration star without excess (Ks - [4.5] < 0.4) has log g
below 3.507, so a star without excess at log g < 3.45 is an extrapolation and is flagged.

Calibration and uncertainty
---------------------------
**Default: the hierarchical posterior (author decision D26-64, 2026-11-11).** The calibration is the
Bayesian random-effects model M1 of the hub finding ``agent-findings/pms-bias-hierarchical-2026-11-10.md``:

    y_i = fe_h_i - anchor = x_i . beta + tau * z_c[i] + eps_i,   z_c ~ N(0, 1),
    eps_i ~ N(0, sqrt(e_i^2 + sigma^2)),

with ``e_i`` ASPCAP's ``fe_h_err``, ``beta`` = (a, b, c, d) ~ N(0, (0.3, 0.5, 0.3, 0.5)), ``tau`` and
``sigma`` ~ HalfNormal(0.10). ``tau`` is the between-cluster scatter: the true [Fe/H] difference between
young clusters mixed with the part of the bias proper to a cluster, which these data cannot separate.
Its posterior is 0.043 [68 %: 0.031, 0.060]; the old moment estimate was 0.028, and it entered the
prediction as if known. For a star of a **new** cluster the prediction is the posterior predictive

    delta = x . beta + u_new + r,   u_new ~ N(0, tau),   r ~ N(0, sqrt(sigma_ref^2 + anchor_err^2)),

whose mean and standard deviation :func:`predict_bias` takes in closed form from the shipped draws:
``sigma = sqrt(Var(x . beta) + E[tau^2] + sigma_ref^2 + anchor_err^2)``. ``r`` is the true metallicity
offset of young associations from the Pleiades / alpha Per anchor, which these data cannot measure
(it is absorbed in ``a``); ``u_new`` partly overlaps it, so keeping both is conservative. Stars of one
cluster share ``u_new`` and ``r``: :func:`predictive_draws` returns the joint draws, so a cluster mean
does not get ``1/sqrt(n)`` on the shared terms. The per-star ``sigma`` (0.061) is **not** in the
prediction; it enters when stars are averaged.

The posterior ships with the package (``data/pms_bias_hierarchical_posterior.npz``, 2 nutpie chains x
4000 draws of beta, tau, sigma, float32), so applying the correction needs neither PyMC nor a sampler.
:func:`fit_hierarchical_model` refits it (PyMC + nutpie, both core dependencies). Validation in the hub
finding: SBC passes (p >= 0.040 in 7 quantities), leave-one-cluster-out predictive coverage 0.952 at
95 % and 0.750 at 68 % over the 104 stars, RMS 0.077 (the OLS's 0.076).

**Student-t arm** (author decision D26-63: declared, not the default). One disc star of NGC 2264
(2M06410934+0956081, ASPCAP log g 3.31 against 4.48 physical) makes the within-cluster residuals
heavy-tailed; a Student-t likelihood discounts it, the log g slope rises from +0.21 to +0.29 and the
low-log g NGC 6383 member moves from -0.323 to -0.376. ``load_posterior("student_t")`` gives it.

**The OLS model stays** as :data:`CALIBRATED_MODEL` / :func:`fit_bias_model` (the control): ordinary
least squares, ``sigma_e`` (0.061) and ``tau`` (0.028) by moments, and
``sigma = sqrt(x^T C x + tau^2 + sigma_ref^2 + anchor_err^2)``, with ``C`` the cluster-aware
(sandwich) covariance of the coefficients: the offset is shared by the stars of a cluster, so more stars
per cluster do not average it away. Pass ``model=CALIBRATED_MODEL`` for the least-squares numbers:
0.061 and 0.065 for the NGC 6383 members against 0.073 and 0.076 (the prototype, with tau treated as
per-star noise in ``C``, gave 0.060 and 0.063). Leaving one cluster out, both give a per-star RMS of 0.076-0.077 dex against 0.137 for
a constant.

Provenance: the author's private research log (APOGEE PMS-bias analysis of 2026-11-09 and its
prototype fit; hierarchical model of 2026-11-10, draws sha256 f290b5a6...); author decisions D26-52
(2026-11-09) to move it into EROTICA, D26-63 and D26-64. Everything needed to reproduce the numbers
ships here: the 104-star table in ``tests/data/``, the posterior in ``data/`` and the refits in
``tests/test_pms_bias.py``.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, field
from importlib import resources
from typing import NamedTuple

import numpy as np

__all__ = [
    "ANCHOR_FE_H",
    "ANCHOR_FE_H_ERR",
    "CALIBRATED_MODEL",
    "SIGMA_REF",
    "BiasModel",
    "BiasPrediction",
    "HierarchicalPosterior",
    "PMSBiasDomainWarning",
    "fit_bias_model",
    "fit_hierarchical_model",
    "leave_one_cluster_out",
    "load_posterior",
    "predict_bias",
    "predictive_draws",
    "select_calibration_sample",
]

#: [Fe/H] of the G-dwarf anchor (Pleiades + alpha Per, ASPCAP DR17) and its error.
ANCHOR_FE_H = 0.021
ANCHOR_FE_H_ERR = 0.005
#: True metallicity spread between young associations and the anchor clusters (hub finding
#: 2026-11-06, amendment 1).
SIGMA_REF = 0.05

#: Calibration domain. The excess range is the one of the calibration sample.
TEFF_RANGE = (4000.0, 5500.0)
LOGG_RANGE = (3.0, 4.8)
EXCESS_RANGE = (-0.2, 2.0)
#: "No excess" is Ks - [4.5] < 0.4, the split used in the hub findings
#: (b9-hd159176-fate-undecided-best-practice-2026-11-06.md sec. 4.2 and
#: scripts/apogee_pms_bias_noexcess_2026_11_09.py). Below log g 3.45 such a star is outside the
#: calibrated region (lowest calibration star without excess: log g 3.507).
NO_EXCESS_BELOW = 0.4
EXTRAPOLATION_LOGG = 3.45

_VALID_LOGG_SOURCES = ("aspcap_dr17", "aspcap_dr19")
_MIN_STARS_PER_CLUSTER = 3


class PMSBiasDomainWarning(UserWarning):
    """A star lies outside the calibration domain of the PMS [Fe/H] bias model."""


@dataclass(frozen=True)
class BiasModel:
    """Calibrated coefficients of the bias model.

    ``beta`` = (a, b, c, d) and ``cov`` its covariance, scaled by ``sigma_e**2 + tau**2``.
    """

    beta: np.ndarray
    cov: np.ndarray
    sigma_e: float
    tau: float
    n_stars: int
    n_clusters: int
    anchor: float = ANCHOR_FE_H


class BiasPrediction(NamedTuple):
    """Output of :func:`predict_bias`. ``delta`` is what ASPCAP under-reports; subtract it."""

    delta: np.ndarray
    sigma: np.ndarray
    extrapolated: np.ndarray
    out_of_domain: np.ndarray


#: Priors of the hierarchical model M1 (hub finding sec. 1.2): sd of beta = (a, b, c, d), and the
#: HalfNormal scales of tau and sigma. Each scale is argued from a physical range there.
BETA_PRIOR_SD = (0.3, 0.5, 0.3, 0.5)
TAU_PRIOR_SCALE = 0.10
SIGMA_PRIOR_SCALE = 0.10
_LIKELIHOODS = ("normal", "student_t")
_POSTERIOR_FILE = "pms_bias_hierarchical_posterior.npz"


@dataclass(frozen=True)
class HierarchicalPosterior:
    """Posterior draws of the hierarchical bias model.

    ``beta`` (n, 4) = (a, b, c, d); ``tau`` the between-cluster scatter; ``sigma`` the per-star scatter
    beyond ASPCAP's ``fe_h_err``. ``likelihood`` is ``"normal"`` (M1, the default) or ``"student_t"``.
    ``diagnostics`` holds what :func:`fit_hierarchical_model` measured (empty for the shipped draws,
    whose diagnostics are in the hub finding: R-hat <= 1.0016, ESS bulk >= 1528, 1 divergence in 8000).
    """

    beta: np.ndarray
    tau: np.ndarray
    sigma: np.ndarray
    likelihood: str = "normal"
    anchor: float = ANCHOR_FE_H
    n_stars: int = 104
    n_clusters: int = 12
    diagnostics: dict = field(default_factory=dict, compare=False)

    @property
    def beta_mean(self) -> np.ndarray:
        return np.asarray(self.beta, float).mean(axis=0)

    @property
    def beta_cov(self) -> np.ndarray:
        return np.cov(np.asarray(self.beta, float), rowvar=False)

    @property
    def tau2_mean(self) -> float:
        """E[tau^2]: the variance a new cluster's offset adds, with tau's uncertainty integrated."""
        return float(np.mean(np.asarray(self.tau, float) ** 2))


@functools.cache
def load_posterior(likelihood="normal"):
    """The shipped posterior: ``"normal"`` (M1, default of :func:`predict_bias`) or ``"student_t"``.

    Draws from the hub run (``scripts/pms_bias_hierarchical_2026_11_10.py fit``, nutpie, 2 chains x
    4000 draws after 2000 tuning, seed 11), stored as float32.
    """
    if likelihood not in _LIKELIHOODS:
        raise ValueError(f"likelihood={likelihood!r}; use one of {_LIKELIHOODS}")
    ref = resources.files(__package__).joinpath("data", _POSTERIOR_FILE)
    with resources.as_file(ref) as path, np.load(path) as z:
        b, t, s = (np.array(z[f"{likelihood}_{q}"], float) for q in ("beta", "tau", "sigma"))
    for a in (b, t, s):
        a.setflags(write=False)
    return HierarchicalPosterior(beta=b, tau=t, sigma=s, likelihood=likelihood)


# OLS control (the default before D26-64; see the module docstring). ``beta``, ``sigma_e`` and ``tau``
# are the hub prototype's; ``cov`` is the cluster-aware sandwich (PR #61 review, 2026-11-11), which the
# prototype's (X^T X)^-1 (sigma_e^2 + tau^2) understated (sd of a 0.0131 -> 0.0162, of b 0.0284 -> 0.0355).
# Embedded calibration: full-precision copy of the prototype fit in the author's private research log
# (2026-11-09), fitted on the 104 stars of tests/data/apogee_dr17_young_k_pms_bias.ecsv.
# tests/test_pms_bias.py refits them and fails if they drift.
CALIBRATED_MODEL = BiasModel(
    beta=np.array(
        [
            -0.18090877702474475,
            0.18251503390151047,
            0.04068476651176977,
            -0.12350123453595746,
        ]
    ),
    cov=np.array(
        [
            [
                0.0002617246946871321,
                -9.877650805538386e-05,
                7.119297723336772e-05,
                -0.00020272234246809968,
            ],
            [
                -9.877650805538389e-05,
                0.0012595932279156204,
                0.0003644717114584356,
                0.0008207611758321151,
            ],
            [
                7.119297723336765e-05,
                0.0003644717114584355,
                0.00021221451784365477,
                0.0002357235427862836,
            ],
            [
                -0.00020272234246809968,
                0.0008207611758321151,
                0.00023572354278628374,
                0.0009453464226551864,
            ],
        ]
    ),
    sigma_e=0.06050772077911833,
    tau=0.028302879141487953,
    n_stars=104,
    n_clusters=12,
)


def _design(teff, logg, excess):
    teff, logg, excess = np.broadcast_arrays(
        np.asarray(teff, float), np.asarray(logg, float), np.asarray(excess, float)
    )
    return np.stack(
        [np.ones_like(teff), logg - 4.0, (teff - 4800.0) / 500.0, np.clip(excess, 0.0, None)],
        axis=-1,
    )


def _numeric(table, name):
    """A float column with masked entries as NaN. ``np.asarray`` on an astropy ``MaskedColumn``
    returns the values *under* the mask, so a masked measurement would enter as an observation."""
    return np.ma.filled(np.ma.asarray(table[name], dtype=float), np.nan)


def _columns(table):
    def col(name):
        return _numeric(table, name)

    return (
        col("teff"),
        col("logg"),
        col("ks_minus_45"),
        col("fe_h"),
        np.asarray(table["Name"]).astype(str),
    )


def select_calibration_sample(table):
    """Boolean mask of the calibration selection.

    Finite ``ks_minus_45`` and ``fe_h``, ``logg >= 3.0``, and clusters (column ``Name``) keeping at
    least three such stars. On the vendored table this gives 104 stars in 12 clusters.
    """
    teff, logg, excess, fe_h, name = _columns(table)
    ok = np.isfinite(excess) & np.isfinite(fe_h) & (logg >= LOGG_RANGE[0])
    big = [c for c in np.unique(name[ok]) if (ok & (name == c)).sum() >= _MIN_STARS_PER_CLUSTER]
    return ok & np.isin(name, big)


def _check_sample(X, name):
    """What the fit needs to be defined: a design of full rank (4 coefficients, so >= 4 stars and an
    excess column that is not all zero), >= 2 clusters (tau is a between-cluster variance) and one
    cluster with >= 2 stars (sigma_e is a within-cluster variance)."""
    rank = np.linalg.matrix_rank(X) if X.size else 0
    if rank < X.shape[1]:
        raise ValueError(
            f"design matrix has rank {rank} < {X.shape[1]} ({X.shape[0]} stars): the coefficients "
            "(a, b, c, d) are not identified; at least 4 stars spanning log g, T_eff and excess are needed"
        )
    clusters, counts = np.unique(name, return_counts=True)
    if clusters.size < 2:
        raise ValueError(
            f"{clusters.size} cluster(s): tau, the between-cluster scatter, needs >= 2"
        )
    if not np.any(counts > 1):
        raise ValueError(
            "no cluster has more than one star: the within-cluster scatter sigma_e is undefined"
        )


def _fit(teff, logg, excess, y, name):
    X = _design(teff, logg, excess)
    _check_sample(X, name)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    clusters = np.unique(name)
    means = np.array([resid[name == c].mean() for c in clusters])
    counts = np.array([(name == c).sum() for c in clusters])
    sigma_e = np.sqrt(
        np.mean(
            [
                np.var(resid[name == c], ddof=1)
                for c, n in zip(clusters, counts, strict=True)
                if n > 1
            ]
        )
    )
    tau2 = max(0.0, np.var(means, ddof=1) - np.mean(sigma_e**2 / counts))
    # Cluster-aware ("sandwich") covariance: the residual covariance is sigma_e^2 on the diagonal plus
    # tau^2 between stars of the same cluster, because the offset is shared. Treating tau as per-star
    # noise, (X^T X)^-1 (sigma_e^2 + tau^2), let more stars per cluster average the offset away.
    same = name[:, None] == name[None, :]
    V = sigma_e**2 * np.eye(y.size) + tau2 * same
    A_inv = np.linalg.inv(X.T @ X)
    cov = A_inv @ X.T @ V @ X @ A_inv
    return beta, cov, float(sigma_e), float(np.sqrt(tau2)), clusters.size


def fit_bias_model(table, *, anchor=ANCHOR_FE_H, select=True):
    """Fit the bias model to a table with columns Name, teff, logg, fe_h, ks_minus_45.

    Parameters
    ----------
    table : astropy Table, pandas DataFrame or mapping of columns
        ``logg`` and ``fe_h`` must be ASPCAP's, from the same spectrum.
    anchor : float
        [Fe/H] of the reference; the fitted quantity is ``fe_h - anchor``.
    select : bool
        Apply :func:`select_calibration_sample` first.
    """
    teff, logg, excess, fe_h, name = _columns(table)
    if select:
        keep = select_calibration_sample(table)
        teff, logg, excess, fe_h, name = (v[keep] for v in (teff, logg, excess, fe_h, name))
    beta, cov, sigma_e, tau, n_clusters = _fit(teff, logg, excess, fe_h - anchor, name)
    return BiasModel(beta, cov, sigma_e, tau, int(fe_h.size), int(n_clusters), float(anchor))


def leave_one_cluster_out(table, *, anchor=ANCHOR_FE_H, select=True):
    """Validate by leaving one cluster out, against a constant (the training median).

    Returns a dict with ``rms_star`` and ``mean_bias`` (per-star prediction errors),
    ``rms_cluster`` (RMS of the per-cluster mean error), ``baseline_rms_star`` (same split,
    predictor = median of the training ``fe_h - anchor``), ``n_stars`` and ``n_clusters``.
    """
    teff, logg, excess, fe_h, name = _columns(table)
    if select:
        keep = select_calibration_sample(table)
        teff, logg, excess, fe_h, name = (v[keep] for v in (teff, logg, excess, fe_h, name))
    if np.unique(name).size < 3:
        raise ValueError(
            f"{np.unique(name).size} cluster(s): leaving one out needs >= 2 left to estimate tau, "
            "so >= 3 clusters"
        )
    y = fe_h - anchor
    errs, cluster_errs, base = [], [], []
    for c in np.unique(name):
        out = name == c
        beta, *_ = _fit(teff[~out], logg[~out], excess[~out], y[~out], name[~out])
        e = y[out] - _design(teff[out], logg[out], excess[out]) @ beta
        errs.append(e)
        cluster_errs.append(e.mean())
        base.append(y[out] - np.median(y[~out]))
    errs, base, cluster_errs = np.concatenate(errs), np.concatenate(base), np.array(cluster_errs)
    return {
        "rms_star": float(np.sqrt(np.mean(errs**2))),
        "mean_bias": float(errs.mean()),
        "rms_cluster": float(np.sqrt(np.mean(cluster_errs**2))),
        "baseline_rms_star": float(np.sqrt(np.mean(base**2))),
        "n_stars": int(y.size),
        "n_clusters": int(cluster_errs.size),
    }


def _check_source(logg_source):
    if logg_source not in _VALID_LOGG_SOURCES:
        raise ValueError(
            f"logg_source={logg_source!r}: the PMS [Fe/H] bias model is calibrated on ASPCAP's own "
            f"log g, which carries the bias; use one of {_VALID_LOGG_SOURCES} or do not apply it."
        )


def _domain(teff, logg, excess, stacklevel=3):
    inside = (
        (teff >= TEFF_RANGE[0])
        & (teff <= TEFF_RANGE[1])
        & (logg >= LOGG_RANGE[0])
        & (logg <= LOGG_RANGE[1])
        & (excess >= EXCESS_RANGE[0])
        & (excess <= EXCESS_RANGE[1])
    )
    out_of_domain = ~inside
    if np.any(out_of_domain):
        warnings.warn(
            f"{int(np.sum(out_of_domain))} star(s) outside the calibration domain "
            f"(T_eff {TEFF_RANGE} K, ASPCAP log g {LOGG_RANGE}, Ks - [4.5] {EXCESS_RANGE}) "
            "or with a missing input; the PMS [Fe/H] bias there is an extrapolation.",
            PMSBiasDomainWarning,
            stacklevel=stacklevel,
        )
    extrapolated = (excess < NO_EXCESS_BELOW) & (logg < EXTRAPOLATION_LOGG)
    return out_of_domain, extrapolated


def predict_bias(
    teff,
    logg,
    ks_minus_45,
    *,
    logg_source,
    model=None,
    sigma_ref=SIGMA_REF,
    anchor_err=ANCHOR_FE_H_ERR,
):
    """ASPCAP [Fe/H] deficit ``delta`` of a young K star, and its uncertainty.

    The corrected metallicity is ``fe_h_aspcap - delta``. Vectorized: the three inputs broadcast.

    **Do not use blindly.** The regressor is ASPCAP's log g, which is itself the symptom: the
    veiling that lowers [Fe/H] lowers ASPCAP's log g with it, so the model is empirical and its
    terms are not mechanisms. It holds only for the log g that ASPCAP DR17 (or DR19, which agrees
    star by star) returned for the same spectrum; ``logg_source`` must say so.

    Parameters
    ----------
    teff, logg : array_like
        ASPCAP T_eff (K) and log g (cgs). Domain 4000-5500 K and 3.0-4.8.
    ks_minus_45 : array_like
        Ks - [4.5] (IRAC; W2 if [4.5] is missing), mag. Negative values count as no excess.
    logg_source : {"aspcap_dr17", "aspcap_dr19"}
        Required. Anything else raises ``ValueError``.
    model : HierarchicalPosterior or BiasModel, optional
        Defaults to ``load_posterior("normal")``, the hierarchical model M1 (since D26-64).
        ``load_posterior("student_t")`` is the declared Student-t arm; :data:`CALIBRATED_MODEL` the
        OLS control (the pre-D26-64 default).
    sigma_ref, anchor_err : float
        Reference-metallicity spread and anchor error added in quadrature.

    Returns
    -------
    BiasPrediction
        ``delta`` (posterior predictive mean for a star of a new cluster) and ``sigma`` (its standard
        deviation: parameters (+) the new cluster's offset with tau integrated over its posterior (+)
        sigma_ref (+) anchor_err; no per-star scatter). For a :class:`BiasModel`, ``sigma`` uses
        the moment ``tau`` as known. ``extrapolated`` (no excess, Ks - [4.5] < 0.4, and log g < 3.45:
        no calibration star supports it) and ``out_of_domain`` (outside the ranges above, or a
        missing input; a :class:`PMSBiasDomainWarning` is emitted).
        Stars of one cluster share the offset and ``r``: average them with :func:`predictive_draws`.
    """
    _check_source(logg_source)
    m = load_posterior("normal") if model is None else model
    teff, logg, excess = np.broadcast_arrays(
        np.asarray(teff, float), np.asarray(logg, float), np.asarray(ks_minus_45, float)
    )
    X = _design(teff, logg, excess)
    if isinstance(m, HierarchicalPosterior):
        delta = X @ m.beta_mean
        var_par = np.einsum("...i,ij,...j->...", X, m.beta_cov, X)
        var_cluster = m.tau2_mean
    elif isinstance(m, BiasModel):
        delta = X @ m.beta
        var_par = np.einsum("...i,ij,...j->...", X, m.cov, X)
        var_cluster = m.tau**2
    else:
        raise TypeError(
            f"model must be a HierarchicalPosterior or a BiasModel, not {type(m).__name__}"
        )
    sigma = np.sqrt(var_par + var_cluster + sigma_ref**2 + anchor_err**2)
    out_of_domain, extrapolated = _domain(teff, logg, excess)
    return BiasPrediction(delta, sigma, extrapolated, out_of_domain)


def predictive_draws(
    teff,
    logg,
    ks_minus_45,
    *,
    logg_source,
    model=None,
    rng=None,
    sigma_ref=SIGMA_REF,
    anchor_err=ANCHOR_FE_H_ERR,
):
    """Joint posterior-predictive draws of ``delta`` for stars of **one** new cluster.

    One draw per posterior draw: ``x . beta + tau z_new + r``, with ``z_new ~ N(0, 1)`` and
    ``r ~ N(0, sqrt(sigma_ref^2 + anchor_err^2))`` **shared** by every star passed, as for members
    of the same cluster. Use these to average stars (the shared terms do not shrink with ``n``) or for
    quantiles: the predictive is a scale mixture over ``tau``, not exactly Gaussian.

    Returns an array of shape ``(n_draws,) + broadcast shape of the inputs``. ``model`` must be a
    :class:`HierarchicalPosterior` (default ``load_posterior("normal")``); ``rng`` a seed or
    ``numpy.random.Generator``. Domain warnings as :func:`predict_bias`.
    """
    _check_source(logg_source)
    m = load_posterior("normal") if model is None else model
    if not isinstance(m, HierarchicalPosterior):
        raise TypeError(
            "predictive_draws needs a HierarchicalPosterior (draws), not an OLS BiasModel"
        )
    rng = np.random.default_rng(rng)
    teff, logg, excess = np.broadcast_arrays(
        np.asarray(teff, float), np.asarray(logg, float), np.asarray(ks_minus_45, float)
    )
    _domain(teff, logg, excess)
    X = _design(teff, logg, excess)
    beta, tau = np.asarray(m.beta, float), np.asarray(m.tau, float)
    n = tau.size
    shared = tau * rng.standard_normal(n) + rng.normal(0.0, np.hypot(sigma_ref, anchor_err), n)
    fixed = np.einsum("ki,...i->k...", beta, X)
    return fixed + shared.reshape((n,) + (1,) * (X.ndim - 1))


def fit_hierarchical_model(
    table,
    *,
    likelihood="normal",
    anchor=ANCHOR_FE_H,
    select=True,
    tau_scale=TAU_PRIOR_SCALE,
    draws=4000,
    tune=2000,
    chains=2,
    random_seed=11,
    mode_search=False,
):
    """Refit the hierarchical model with PyMC + nutpie (JAX backend), never PyMC's own NUTS.

    ``table`` needs Name, teff, logg, fe_h, ks_minus_45 and **fe_h_err** (ASPCAP's, the floor of the
    per-star error). Priors as :data:`BETA_PRIOR_SD`, ``tau ~ HalfNormal(tau_scale)``,
    ``sigma ~ HalfNormal(0.10)``; non-centred cluster offsets. ``likelihood="student_t"`` adds
    ``nu ~ Gamma(2, 0.1)``. Defaults reproduce the hub run (2 chains x 4000 draws after 2000 tuning;
    ``cores=1``, the RAM rule of the programme).

    ``mode_search`` (off by default) runs :func:`erotica.analysis.inference.attach_mode_search`. The hub
    ran a 10-start search on M1 and found one mode; in the non-centred space the joint mode sits at
    tau ~ 0.15 with small z (the funnel), which is not the marginal mode of tau, so read its report
    on (a, b, c, d, sigma), not on tau.

    Returns a :class:`HierarchicalPosterior` whose ``diagnostics`` holds max R-hat, min bulk and tail
    ESS (``az.rhat`` / ``az.ess`` per parameter), divergences, draws and, with ``mode_search``, the
    report. Check them; this function does not gate.
    """
    if likelihood not in _LIKELIHOODS:
        raise ValueError(f"likelihood={likelihood!r}; use one of {_LIKELIHOODS}")
    import arviz as az

    from ..analysis.inference import SamplingConfig, _require_pymc, _sample

    pm = _require_pymc()
    teff, logg, excess, fe_h, name = _columns(table)
    err = _numeric(table, "fe_h_err")
    if select:
        keep = select_calibration_sample(table)
        teff, logg, excess, fe_h, name, err = (
            v[keep] for v in (teff, logg, excess, fe_h, name, err)
        )
    if not np.all(np.isfinite(err)):
        raise ValueError("fe_h_err must be finite for every star fitted")
    clusters, cid = np.unique(name, return_inverse=True)
    X = _design(teff, logg, excess)
    _check_sample(X, name)
    y = fe_h - anchor
    with pm.Model() as model:
        beta = pm.Normal("beta", 0.0, sigma=np.asarray(BETA_PRIOR_SD), shape=4)
        tau = pm.HalfNormal("tau", tau_scale)
        sig = pm.HalfNormal("sigma", SIGMA_PRIOR_SCALE)
        z = pm.Normal("z", 0.0, 1.0, shape=clusters.size)
        mu = pm.math.dot(X, beta) + tau * z[cid]
        s = pm.math.sqrt(err**2 + sig**2)
        if likelihood == "student_t":
            nu = pm.Gamma("nu", 2.0, 0.1)
            pm.StudentT("y", nu=nu, mu=mu, sigma=s, observed=y)
        else:
            pm.Normal("y", mu=mu, sigma=s, observed=y)
    config = SamplingConfig(
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        extra_kwargs={"cores": 1},
        mode_search=SamplingConfig().mode_search if mode_search else None,
    )
    trace = _sample(pm, model, config)
    names = ["beta", "tau", "sigma"] + (["nu"] if likelihood == "student_t" else [])
    rhat = az.rhat(trace, var_names=names)
    ess_b = az.ess(trace, var_names=names, method="bulk")
    ess_t = az.ess(trace, var_names=names, method="tail")
    diag = {
        "rhat_max": max(float(np.max(rhat[n].values)) for n in names),
        "ess_bulk_min": min(float(np.min(ess_b[n].values)) for n in names),
        "ess_tail_min": min(float(np.min(ess_t[n].values)) for n in names),
        "divergences": int(trace.sample_stats["diverging"].values.sum()),
        "n_draws": int(trace.sample_stats["diverging"].values.size),
    }
    if mode_search:
        from ..analysis.inference import mode_search_summary

        diag["mode_search"] = mode_search_summary(trace)

    def flat(v):
        v = trace.posterior[v].values
        return v.reshape(-1, *v.shape[2:])

    return HierarchicalPosterior(
        beta=flat("beta"),
        tau=flat("tau"),
        sigma=flat("sigma"),
        likelihood=likelihood,
        anchor=float(anchor),
        n_stars=int(y.size),
        n_clusters=int(clusters.size),
        diagnostics=diag,
    )
