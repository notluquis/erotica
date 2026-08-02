"""Bayesian inference helpers with lazy optional dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy import units as u

from .kinematics import projected_velocity_values, radial_velocity_values
from .units import quantity_values


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling knobs shared by PyMC models."""

    draws: int = 2_000
    tune: int = 1_000
    target_accept: float = 0.9
    chains: int | None = None
    random_seed: int | None = None
    nuts_sampler: str = "pymc"
    progressbar: bool = False
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParallaxPriors:
    """Scale-free priors for the cluster parallax model.

    Every value is a **fixed constant, independent of the data being fit**. The
    default path in :func:`fit_parallax_model` previously centred a
    ``Uniform(0.5x, 1.5x)`` on ``nanmean(parallax)`` *of the data*, and put a
    ``HalfNormal(sigma=nanstd(parallax))`` on the spread -- the data used twice.
    WHERE THE NUMBERS COME FROM
    ---------------------------
    Set from **Hunt & Reffert (2024)**, `2024A&A...686A..42H`, 5647 open clusters — an external
    catalogue, never the cluster being fitted. Regenerate with ``tools/validation/fetch_hr24.py``.
    Percentiles across that census:

    ========================  =========  =========  =========  =========
    quantity                       0.5%        16%     median      99.5%
    ========================  =========  =========  =========  =========
    parallax (mas)                0.087      0.235      0.406      3.564
    distance (kpc)                0.278      1.094      2.259      8.084
    ``pmRA`` (mas/yr)           -11.617     -4.796     -1.794      4.956
    ``pmDE`` (mas/yr)           -10.602     -3.778     -1.265      7.024
    radial velocity (km/s)      -94.559    -29.142      0.000    126.823
    ``pmRA`` dispersion           0.036      0.067      0.097      0.696
    parallax dispersion           0.009      0.023      0.037      0.136
    ========================  =========  =========  =========  =========

    ``mu_upper = 10`` mas covers the census with room to spare — the richest cluster parallax is
    21.2 mas but the 99.5th percentile is 3.6, and a cluster nearer than 100 pc is not what this
    pipeline is for. ``sigma_scale = 0.05`` mas sits between the census median cluster parallax
    dispersion (0.037) and its 84th percentile (0.056), so it is weakly informative about a real
    internal spread rather than about measurement error.

    .. important::
       ``zero_point_scale = 0.0103`` mas is **not** a free choice: it is the systematic floor on the
       Gaia parallax zero point measured by **Maíz Apellániz, Pantaleoni González & Barbá (2021,
       A&A 649, A13,** ``2021A&A...649A..13M``**)**, 10.3 µas. Attributing it to Vasiliev &
       Baumgardt is a mis-citation this project made once and corrected.


    Attributes
    ----------
    mu_lower, mu_upper : float
        Support for the mean cluster parallax, in mas. The default ``(0, 10)``
        spans every Galactic open cluster: 10 mas is 100 pc, closer than any
        known OC, and 0 is the distant limit. Wide enough to be uninformative,
        bounded enough to keep the sampler on the physical branch.
    sigma_scale : float
        Half-normal scale for the **intrinsic** parallax spread of the cluster,
        in mas. 0.05 mas at 1 kpc is a depth of ~50 pc, comfortably larger than
        any bound cluster, and the half-normal tail permits more.
    """

    mu_lower: float = 0.0
    mu_upper: float = 10.0
    sigma_scale: float = 0.05
    #: Prior width (mas) on the **residual** Gaia parallax zero-point, shared by all
    #: members. 0.0103 mas is the systematic floor of Maiz Apellaniz, Pantaleoni
    #: Gonzalez & Barba (2021, A&A 649, A13, `2021A&A...649A..13M`), whose abstract
    #: states: "The angular covariance at zero separation is estimated to be
    #: 106 microarcsec^2, yielding a minimum (systematic) uncertainty for EDR3
    #: parallaxes of 10.3 microarcsec for individual stars or compact stellar
    #: clusters." (An earlier version of this comment credited Vasiliev &
    #: Baumgardt 2021 for that number; that was wrong.) Riess et al. measure a residual
    #: zp = -3 +/- 4 uas in open clusters. Lindegren et al. (2021,
    #: `2021A&A...649A...4L`) say of their own correction that it is "not perfect"
    #: and its use is "at the researcher's discretion" -- so treating the corrected
    #: parallax as exact is not supported by the correction's own source.
    zero_point_scale: float = 0.0103


@dataclass(frozen=True)
class DistancePriors:
    """Scale-free priors for the hierarchical distance model. Fixed constants.
    WHERE THE NUMBERS COME FROM
    ---------------------------
    Set from **Hunt & Reffert (2024)**, `2024A&A...686A..42H`, 5647 open clusters — an external
    catalogue, never the cluster being fitted. Regenerate with ``tools/validation/fetch_hr24.py``.
    Percentiles across that census:

    ========================  =========  =========  =========  =========
    quantity                       0.5%        16%     median      99.5%
    ========================  =========  =========  =========  =========
    parallax (mas)                0.087      0.235      0.406      3.564
    distance (kpc)                0.278      1.094      2.259      8.084
    ``pmRA`` (mas/yr)           -11.617     -4.796     -1.794      4.956
    ``pmDE`` (mas/yr)           -10.602     -3.778     -1.265      7.024
    radial velocity (km/s)      -94.559    -29.142      0.000    126.823
    ``pmRA`` dispersion           0.036      0.067      0.097      0.696
    parallax dispersion           0.009      0.023      0.037      0.136
    ========================  =========  =========  =========  =========

    ``mu_lower = 0.05`` kpc and ``mu_upper = 20`` kpc bracket a census that runs 0.28 to 12.9 kpc,
    with the 99.5th percentile at 8.1. The bounds are deliberately loose: they exist to keep the
    sampler in a physical region, not to express belief.

    .. warning::
       These are **uniform** bounds, so unlike the half-Cauchy scales elsewhere in the package they
       do not degrade gracefully if wrong. A cluster genuinely beyond 20 kpc would be silently pushed
       to the boundary rather than pulling the posterior. Check the posterior is not against a bound
       before quoting a distance.


    Attributes
    ----------
    mu_lower, mu_upper : float
        Support (kpc) for the mean cluster distance. ``(0.05, 20)`` spans every
        Galactic open cluster with room to spare.
    sigma_scale : float
        Half-normal scale (kpc) for the cluster's **intrinsic** line-of-sight
        depth. 0.05 kpc is 50 pc, larger than any bound cluster; the tail allows more.
    """

    mu_lower: float = 0.05
    mu_upper: float = 20.0
    sigma_scale: float = 0.05


@dataclass(frozen=True)
class VelocityPriors:
    """Scale-free priors for the velocity model. Fixed constants.
    WHERE THE NUMBERS COME FROM
    ---------------------------
    Set from **Hunt & Reffert (2024)**, `2024A&A...686A..42H`, 5647 open clusters — an external
    catalogue, never the cluster being fitted. Regenerate with ``tools/validation/fetch_hr24.py``.
    Percentiles across that census:

    ========================  =========  =========  =========  =========
    quantity                       0.5%        16%     median      99.5%
    ========================  =========  =========  =========  =========
    parallax (mas)                0.087      0.235      0.406      3.564
    distance (kpc)                0.278      1.094      2.259      8.084
    ``pmRA`` (mas/yr)           -11.617     -4.796     -1.794      4.956
    ``pmDE`` (mas/yr)           -10.602     -3.778     -1.265      7.024
    radial velocity (km/s)      -94.559    -29.142      0.000    126.823
    ``pmRA`` dispersion           0.036      0.067      0.097      0.696
    parallax dispersion           0.009      0.023      0.037      0.136
    ========================  =========  =========  =========  =========

    ``mu_scale = 50`` km/s is a half-Cauchy scale against a census whose radial velocities run from
    −94.6 to +126.8 km/s at the 0.5/99.5 percentiles; a half-Cauchy at 50 puts its 95th percentile
    at 635 km/s, so the tail comfortably admits the 664 km/s extreme without the bulk of the prior
    sitting out there.

    ``sigma_scale = 2.0`` km/s is the internal velocity dispersion scale. **This one is not derived
    from the census**: the catalogue's ``s_RV`` column has a median of 9.6 km/s, but that is
    dominated by measurement scatter in samples with few RV members, not by the true dispersion of a
    bound open cluster, which is of order 1 km/s. The value encodes that physical expectation, and
    is flagged rather than dressed up as a measurement.


    Attributes
    ----------
    mu_scale : float
        Standard deviation (km/s) of the zero-centred Normal prior on the mean
        projected velocity.
    sigma_scale : float
        Half-normal scale (km/s) for the cluster's **internal** velocity
        dispersion. Open clusters sit near 0.3-1 km/s; the previous
        ``Uniform(0, 40)`` was about **80x too wide**, which in the low-dispersion
        regime leaves the posterior prior-dominated rather than data-dominated.
        A half-normal at 2 km/s is weakly informative and still admits an
        unusually hot cluster through its tail.
    """

    mu_scale: float = 50.0
    sigma_scale: float = 2.0


@dataclass(frozen=True)
class ProperMotionPriors:
    """Scale-free priors for the 2D proper-motion model. Fixed constants.

    The previous defaults centred ``mu_RA``/``mu_Dec`` on ``nanmedian`` of the
    data and scaled every width by ``nanstd`` of the data -- the data used twice.
    WHERE THE NUMBERS COME FROM
    ---------------------------
    Set from **Hunt & Reffert (2024)**, `2024A&A...686A..42H`, 5647 open clusters — an external
    catalogue, never the cluster being fitted. Regenerate with ``tools/validation/fetch_hr24.py``.
    Percentiles across that census:

    ========================  =========  =========  =========  =========
    quantity                       0.5%        16%     median      99.5%
    ========================  =========  =========  =========  =========
    parallax (mas)                0.087      0.235      0.406      3.564
    distance (kpc)                0.278      1.094      2.259      8.084
    ``pmRA`` (mas/yr)           -11.617     -4.796     -1.794      4.956
    ``pmDE`` (mas/yr)           -10.602     -3.778     -1.265      7.024
    radial velocity (km/s)      -94.559    -29.142      0.000    126.823
    ``pmRA`` dispersion           0.036      0.067      0.097      0.696
    parallax dispersion           0.009      0.023      0.037      0.136
    ========================  =========  =========  =========  =========

    ``mu_scale = 20`` mas/yr as a half-Cauchy scale covers a census whose proper motions run to about
    ±11 mas/yr at the 0.5/99.5 percentiles in both axes, with a maximum of 104. ``sigma_scale = 1.0``
    mas/yr is again generous against the census median dispersion of 0.097 mas/yr and its 99.5th
    percentile of 0.70 — deliberately, since a dispersion prior that is too tight is the failure that
    matters here, biasing every cluster toward being colder than it is.


    Attributes
    ----------
    mu_scale : float
        Standard deviation (mas/yr) of the zero-centred Normal prior on the mean
        cluster proper motion. 20 mas/yr comfortably covers Galactic open
        clusters, which rarely exceed a few mas/yr beyond the local standard.
    sigma_scale : float
        Half-normal scale (mas/yr) for the cluster's **intrinsic** velocity
        dispersion. 1 mas/yr at 1 kpc is ~4.7 km/s, well above any bound open
        cluster, and the tail permits more.
    """

    mu_scale: float = 20.0
    sigma_scale: float = 1.0


@dataclass(frozen=True)
class DistanceFitResult:
    mu_r_mean: float
    std_r_mean: float
    mu_r_std: float
    std_r_std: float
    trace: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParallaxFitResult:
    mu_parallax_mean: float
    sigma_parallax_mean: float
    mu_parallax_std: float
    sigma_parallax_std: float
    trace: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProperMotionFitResult:
    results: dict[str, float]
    trace: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VelocityFitResult:
    mu_v_mean: float
    std_v_mean: float
    mu_v_std: float
    std_v_std: float
    trace: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _require_pymc():
    try:
        import pymc as pm
    except ImportError as exc:
        raise ImportError(
            "PyMC is required for erotica.analysis.inference. Install the 'bayes' extra."
        ) from exc
    return pm


def _sample(pm, model, config: SamplingConfig):
    kwargs = dict(
        draws=config.draws,
        tune=config.tune,
        target_accept=config.target_accept,
        progressbar=config.progressbar,
        random_seed=config.random_seed,
        nuts_sampler=config.nuts_sampler,
        **config.extra_kwargs,
    )
    if config.chains is not None:
        kwargs["chains"] = config.chains
    with model:
        return pm.sample(**kwargs)


def distance_model(
    data,
    *,
    distance_column: str = "r_med_geo",
    parallax_column: str = "parallax",
    distance_lo_column: str | None = None,
    distance_hi_column: str | None = None,
    prior_type: str = "uniform",
    priors: DistancePriors | None = None,
    return_trace: bool = False,
    sampling: SamplingConfig | None = None,
) -> DistanceFitResult:
    r"""Fit the hierarchical cluster-distance model.

    Parameters
    ----------
    distance_lo_column, distance_hi_column : str, optional
        Bailer-Jones lower and upper bounds (e.g. ``r_lo_geo`` / ``r_hi_geo``,
        the 16th and 84th percentiles). **Strongly recommended.** When given, each
        star's catalogue distance is treated as a *measurement* of a latent true
        distance rather than as exact:

        .. math::

            r^{\mathrm{true}}_i &\sim \mathrm{Gamma}(\mu_r,\ \sigma_r) \\
            r^{\mathrm{obs}}_i  &\sim \mathcal{N}(r^{\mathrm{true}}_i,\ \sigma_i)

        with :math:`\sigma_i = (r_{\mathrm{hi}} - r_{\mathrm{lo}})/2`. Without
        them ``std_r`` absorbs the Bailer-Jones uncertainties as if they were
        cluster depth, and those are large: a 10% distance error at 1 kpc is
        100 pc, several times any real cluster.
    priors : DistancePriors, optional
        Scale-free priors. Fixed constants, **not** functions of `data`.

    Notes
    -----
    Before 2026-07-27 the prior was ``Uniform(0.5x, 1.5x)`` centred on
    ``nanmean([nanmean(1/parallax), nanmean(distances)])`` **of the data being
    fit**, with ``HalfNormal(sigma=nanstd(distances))`` on the spread -- the data
    used twice, twice over. `parallax_column` is retained for API compatibility
    and is no longer read.
    """
    pm = _require_pymc()
    sampling = sampling or SamplingConfig()
    priors = priors or DistancePriors()
    distances = np.asarray(quantity_values(data[distance_column], u.kpc), dtype=float)

    errors = None
    if (distance_lo_column is None) != (distance_hi_column is None):
        raise ValueError("Give both distance_lo_column and distance_hi_column, or neither.")
    if distance_lo_column is not None:
        lo = np.asarray(quantity_values(data[distance_lo_column], u.kpc), dtype=float)
        hi = np.asarray(quantity_values(data[distance_hi_column], u.kpc), dtype=float)
        if lo.shape != distances.shape or hi.shape != distances.shape:
            raise ValueError("Distance bounds must match the distances in length.")
        if np.any(hi <= lo):
            raise ValueError("distance_hi_column must exceed distance_lo_column for every star.")
        errors = (hi - lo) / 2.0
        if not np.all(np.isfinite(errors)) or np.any(errors <= 0):
            raise ValueError("Derived per-star distance errors must be finite and positive.")

    with pm.Model() as model:
        if prior_type == "uniform":
            mu_r = pm.Uniform("mu_r", lower=priors.mu_lower, upper=priors.mu_upper)
        elif prior_type == "normal":
            mid = 0.5 * (priors.mu_lower + priors.mu_upper)
            mu_r = pm.TruncatedNormal(
                "mu_r", lower=priors.mu_lower, upper=priors.mu_upper, mu=mid, sigma=1
            )
        else:
            raise ValueError("prior_type must be 'uniform' or 'normal'.")
        std_r = pm.HalfNormal("std_r", sigma=priors.sigma_scale)
        if errors is None:
            pm.Gamma("r", mu=mu_r, sigma=std_r, observed=distances)
        else:
            true_r = pm.Gamma("r_true", mu=mu_r, sigma=std_r, shape=distances.size)
            pm.Normal("r", mu=true_r, sigma=errors, observed=distances)
    trace = _sample(pm, model, sampling)
    return DistanceFitResult(
        mu_r_mean=float(trace.posterior["mu_r"].mean()),
        std_r_mean=float(trace.posterior["std_r"].mean()),
        mu_r_std=float(trace.posterior["mu_r"].std()),
        std_r_std=float(trace.posterior["std_r"].std()),
        trace=trace if return_trace else None,
        metadata={
            "backend": sampling.nuts_sampler,
            "variables": ["mu_r", "std_r"],
            "error_aware": errors is not None,
            "prior": "scale-free",
        },
    )


def fit_parallax_model(
    data,
    *,
    parallax_column: str = "parallax",
    parallax_error_column: str | None = None,
    prior_distance=None,
    prior_type: str = "uniform",
    priors: ParallaxPriors | None = None,
    zero_point: bool = False,
    return_trace: bool = False,
    sampling: SamplingConfig | None = None,
) -> ParallaxFitResult:
    r"""Fit a Gaussian cluster parallax model.

    Parameters
    ----------
    parallax_error_column : str, optional
        Column holding the per-star parallax uncertainty (mas). **Strongly
        recommended.** When given, the likelihood becomes

        .. math:: \varpi_i \sim \mathcal{N}\!\left(\mu_\varpi,\;
                  \sqrt{\sigma_{\mathrm{int}}^2 + \sigma_{\varpi,i}^2}\right)

        so ``sigma_parallax`` is the cluster's **intrinsic** parallax spread.
        Without it, every star is treated as measured exactly and
        ``sigma_parallax`` absorbs the measurement scatter as well, which for a
        Gaia sample is usually the larger of the two -- the cluster then appears
        deeper than it is. The distinction matters because the intrinsic spread,
        not the observed one, is what a physical depth or a virial estimate wants.
    prior_distance : Quantity or float, optional
        An independent distance estimate to centre the prior on. Supplying one is
        the *informative* path and is not data-dependent, since it comes from
        outside this sample.
    priors : ParallaxPriors, optional
        Scale-free priors used when `prior_distance` is not supplied. Defaults are
        fixed constants; they are **not** derived from `data`.
    zero_point : bool, default False
        Add a **residual parallax zero-point** as a nuisance parameter shared by
        all members, with width `priors.zero_point_scale`. Gaia's published
        zero-point correction is, in its authors' words, "not perfect", and its
        residual is spatially correlated on the scale of a cluster -- so every
        member carries essentially the *same* leftover offset, which no amount of
        averaging removes.

        The offset is exactly degenerate with ``mu_parallax`` for a single
        cluster, and that is deliberate: the degeneracy widens the reported
        uncertainty on the mean parallax by the systematic floor in quadrature
        rather than leaving it out. **Enable it for any published mean parallax.**

    Notes
    -----
    Before 2026-07-27 the default path built ``Uniform(0.5x, 1.5x)`` around
    ``nanmean(parallax)`` of the sample and a ``HalfNormal(sigma=nanstd(parallax))``
    on the spread -- both functions of the data being fit. That is the data used
    twice; see the decision log.
    """
    pm = _require_pymc()
    sampling = sampling or SamplingConfig()
    priors = priors or ParallaxPriors()
    parallax_values = quantity_values(data[parallax_column], u.mas)

    errors = None
    if parallax_error_column is not None:
        errors = np.asarray(quantity_values(data[parallax_error_column], u.mas), dtype=float)
        if errors.shape != np.shape(parallax_values):
            raise ValueError("parallax_error_column must have the same length as the parallaxes.")
        if not np.all(np.isfinite(errors)) or np.any(errors <= 0):
            raise ValueError("Per-star parallax errors must all be finite and positive.")

    if prior_distance is not None:
        prior_parallax = prior_distance.to(u.mas, equivalencies=u.parallax()).value if hasattr(prior_distance, "to") else 1 / float(prior_distance)
        lower, upper = 0.5 * prior_parallax, 1.5 * prior_parallax
    else:
        prior_parallax = 0.5 * (priors.mu_lower + priors.mu_upper)
        lower, upper = priors.mu_lower, priors.mu_upper

    with pm.Model() as model:
        if prior_type == "uniform":
            mu_parallax = pm.Uniform("mu_parallax", lower=lower, upper=upper)
        elif prior_type == "normal":
            mu_parallax = pm.TruncatedNormal(
                "mu_parallax", lower=lower, upper=upper, sigma=1, mu=prior_parallax
            )
        else:
            raise ValueError("prior_type must be 'uniform' or 'normal'.")
        sigma_parallax = pm.HalfNormal("sigma_parallax", sigma=priors.sigma_scale)
        total = (
            sigma_parallax
            if errors is None
            else pm.math.sqrt(sigma_parallax**2 + errors**2)  # intrinsic (+) measurement
        )
        if zero_point:
            # One nuisance offset shared by every member. It is exactly degenerate
            # with mu_parallax for a single cluster -- that is the point: the
            # degeneracy is what propagates the correlated systematic into the
            # reported uncertainty instead of hiding it.
            zp = pm.Normal("zero_point", mu=0.0, sigma=priors.zero_point_scale)
            centre = mu_parallax + zp
        else:
            centre = mu_parallax
        pm.Normal("observed_parallax", mu=centre, sigma=total, observed=parallax_values)
    trace = _sample(pm, model, sampling)
    return ParallaxFitResult(
        mu_parallax_mean=float(trace.posterior["mu_parallax"].mean()),
        sigma_parallax_mean=float(trace.posterior["sigma_parallax"].mean()),
        mu_parallax_std=float(trace.posterior["mu_parallax"].std()),
        sigma_parallax_std=float(trace.posterior["sigma_parallax"].std()),
        trace=trace if return_trace else None,
        metadata={
            "backend": sampling.nuts_sampler,
            "variables": ["mu_parallax", "sigma_parallax"],
            "error_aware": errors is not None,
            "zero_point_nuisance": bool(zero_point),
            "prior": "informative" if prior_distance is not None else "scale-free",
        },
    )


def proper_motion_2d_gaussian(
    pm_ra,
    pm_dec,
    *,
    pm_ra_error=None,
    pm_dec_error=None,
    pm_ra_dec_corr=None,
    priors: ProperMotionPriors | None = None,
    return_trace: bool = False,
    sampling: SamplingConfig | None = None,
) -> ProperMotionFitResult:
    r"""Fit a correlated 2D Gaussian proper-motion model.

    Parameters
    ----------
    pm_ra_error, pm_dec_error : array-like, optional
        Per-star proper-motion uncertainties (mas/yr). **Strongly recommended.**
        When given, each star gets its own total covariance

        .. math:: \Sigma_i = \Sigma_{\mathrm{int}} + C_i

        so ``sigma_RA``/``sigma_Dec`` measure the cluster's **intrinsic**
        velocity dispersion rather than the dispersion plus Gaia's measurement
        scatter. For an open cluster the measurement term usually dominates, so
        without this the fitted dispersion is mostly an error bar, and any
        virial mass or crossing time derived from it is inflated.
    pm_ra_dec_corr : array-like, optional
        Per-star ``pmra_pmdec_corr`` from Gaia. Ignoring it treats an error
        ellipse as if it were axis-aligned; Gaia's proper-motion correlations
        are routinely |rho| > 0.3.
    priors : ProperMotionPriors, optional
        Scale-free priors. Defaults are constants, **not** functions of the data.

    Notes
    -----
    Before 2026-07-27 the priors were centred on ``nanmedian`` of the data with
    widths from ``nanstd`` of the data, and the per-star covariance was ignored
    entirely even though :mod:`erotica.core._error_aware` already builds it.
    """
    pm = _require_pymc()
    sampling = sampling or SamplingConfig()
    priors = priors or ProperMotionPriors()
    pm_ra_values = np.asarray(quantity_values(pm_ra, u.mas / u.yr), dtype=float)
    pm_dec_values = np.asarray(quantity_values(pm_dec, u.mas / u.yr), dtype=float)
    observed = np.stack([pm_ra_values, pm_dec_values], axis=1)

    per_star = None
    if (pm_ra_error is None) != (pm_dec_error is None):
        raise ValueError("Give both pm_ra_error and pm_dec_error, or neither.")
    if pm_ra_error is not None:
        e_ra = np.asarray(quantity_values(pm_ra_error, u.mas / u.yr), dtype=float)
        e_dec = np.asarray(quantity_values(pm_dec_error, u.mas / u.yr), dtype=float)
        if e_ra.shape != pm_ra_values.shape or e_dec.shape != pm_dec_values.shape:
            raise ValueError("Proper-motion errors must match the proper motions in length.")
        if not (np.all(np.isfinite(e_ra)) and np.all(np.isfinite(e_dec))):
            raise ValueError("Per-star proper-motion errors must all be finite.")
        if np.any(e_ra <= 0) or np.any(e_dec <= 0):
            raise ValueError("Per-star proper-motion errors must all be positive.")
        rho = (
            np.zeros_like(e_ra)
            if pm_ra_dec_corr is None
            else np.asarray(quantity_values(pm_ra_dec_corr), dtype=float)
        )
        if np.any(np.abs(rho) >= 1.0):
            raise ValueError("pm_ra_dec_corr must lie strictly inside (-1, 1).")
        off = rho * e_ra * e_dec
        per_star = np.empty((e_ra.size, 2, 2), dtype=float)
        per_star[:, 0, 0] = e_ra**2
        per_star[:, 1, 1] = e_dec**2
        per_star[:, 0, 1] = per_star[:, 1, 0] = off

    with pm.Model() as model:
        mu_ra = pm.Normal("mu_RA", mu=0.0, sigma=priors.mu_scale)
        mu_dec = pm.Normal("mu_Dec", mu=0.0, sigma=priors.mu_scale)
        sigma_ra = pm.HalfNormal("sigma_RA", sigma=priors.sigma_scale)
        sigma_dec = pm.HalfNormal("sigma_Dec", sigma=priors.sigma_scale)
        # corr is sampled as tanh(z) rather than Uniform(-1, 1). At |corr| = 1 the
        # covariance is singular, and a hard uniform boundary lets NUTS propose
        # arbitrarily close to it -- with a per-star covariance added the Cholesky
        # then fails and the worker dies with EOFError. tanh keeps |corr| < 1
        # strictly, is smooth everywhere, and needs no boundary correction.
        corr_z = pm.Normal("corr_z", mu=0.0, sigma=1.0)
        corr = pm.Deterministic("corr", pm.math.tanh(corr_z))
        cov = pm.math.stack(
            [[sigma_ra**2, corr * sigma_ra * sigma_dec], [corr * sigma_ra * sigma_dec, sigma_dec**2]]
        )
        total = cov if per_star is None else cov + per_star  # broadcasts to (n, 2, 2)
        pm.MvNormal(
            "obs", mu=pm.math.stack([mu_ra, mu_dec]), cov=total, observed=observed
        )
    trace = _sample(pm, model, sampling)
    metadata = {
        "backend": sampling.nuts_sampler,
        "error_aware": per_star is not None,
        "correlation_used": pm_ra_dec_corr is not None,
        "prior": "scale-free",
    }
    results = {
        "mu_RA_mean": float(trace.posterior["mu_RA"].mean()),
        "mu_Dec_mean": float(trace.posterior["mu_Dec"].mean()),
        "sigma_RA_mean": float(trace.posterior["sigma_RA"].mean()),
        "sigma_Dec_mean": float(trace.posterior["sigma_Dec"].mean()),
        "corr_mean": float(trace.posterior["corr"].mean()),
        "mu_RA_std": float(trace.posterior["mu_RA"].std()),
        "mu_Dec_std": float(trace.posterior["mu_Dec"].std()),
        "sigma_RA_std": float(trace.posterior["sigma_RA"].std()),
        "sigma_Dec_std": float(trace.posterior["sigma_Dec"].std()),
        "corr_std": float(trace.posterior["corr"].std()),
    }
    return ProperMotionFitResult(
        results=results,
        trace=trace if return_trace else None,
        metadata=metadata | {"variables": list(results)},
    )


def velocity_model(
    values,
    *,
    distance=None,
    errors=None,
    priors: VelocityPriors | None = None,
    return_trace: bool = False,
    sampling: SamplingConfig | None = None,
) -> VelocityFitResult:
    r"""Fit a Gaussian velocity model.

    Parameters
    ----------
    errors : array-like, optional
        Per-star velocity uncertainties (km/s). **Strongly recommended** — without
        them ``std_v`` is the observed scatter, not the cluster's internal
        velocity dispersion, and any virial mass or crossing time built on it is
        inflated.
    priors : VelocityPriors, optional
        Scale-free priors. Fixed constants, **not** functions of `values`.

    Notes
    -----
    Before 2026-07-27 this model had all three defects the other three carried:
    ``mu_v`` was centred on ``nanmean`` of the data, the dispersion prior was
    ``Uniform(0, 40)`` km/s — **roughly 80x too wide for a ~0.5 km/s quantity**,
    so in the low-dispersion regime the posterior was prior-dominated — and the
    per-star velocity uncertainties never entered the likelihood.
    """
    pm = _require_pymc()
    sampling = sampling or SamplingConfig()
    priors = priors or VelocityPriors()
    if hasattr(values, "colnames"):
        if "projected_velocity" not in values.colnames:
            raise ValueError("Table input must contain a 'projected_velocity' column.")
        values = values["projected_velocity"]
    velocity_values, _, _ = projected_velocity_values(values, distance=distance)
    velocity_values = np.asarray(velocity_values, dtype=float)

    per_star = None
    if errors is not None:
        per_star = np.asarray(quantity_values(errors, u.km / u.s), dtype=float)
        if per_star.shape != velocity_values.shape:
            raise ValueError("Velocity errors must match the velocities in length.")
        if not np.all(np.isfinite(per_star)) or np.any(per_star <= 0):
            raise ValueError("Per-star velocity errors must all be finite and positive.")

    with pm.Model() as model:
        mu_v = pm.Normal("mu_v", mu=0.0, sigma=priors.mu_scale)
        std_v = pm.HalfNormal("std_v", sigma=priors.sigma_scale)
        total = std_v if per_star is None else pm.math.sqrt(std_v**2 + per_star**2)
        pm.Normal("observed_velocity", mu=mu_v, sigma=total, observed=velocity_values)
    trace = _sample(pm, model, sampling)
    return VelocityFitResult(
        mu_v_mean=float(trace.posterior["mu_v"].mean()),
        std_v_mean=float(trace.posterior["std_v"].mean()),
        mu_v_std=float(trace.posterior["mu_v"].std()),
        std_v_std=float(trace.posterior["std_v"].std()),
        trace=trace if return_trace else None,
        metadata={
            "backend": sampling.nuts_sampler,
            "variables": ["mu_v", "std_v"],
            "error_aware": per_star is not None,
            "prior": "scale-free",
        },
    )


def radial_velocity_model(
    radial_velocity,
    *,
    return_trace: bool = False,
    sampling: SamplingConfig | None = None,
) -> VelocityFitResult:
    """Fit a Gaussian radial-velocity model."""
    if hasattr(radial_velocity, "colnames"):
        if "radial_velocity" not in radial_velocity.colnames:
            raise ValueError("Table input must contain a 'radial_velocity' column.")
        radial_velocity = radial_velocity["radial_velocity"]
    return velocity_model(
        radial_velocity_values(radial_velocity) * u.km / u.s,
        return_trace=return_trace,
        sampling=sampling,
    )


def _dataclass_result_to_dict(result) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in result.__dict__.items()
        if key not in {"trace", "metadata"} and value is not None
    }
    if result.trace is not None:
        payload["trace"] = result.trace
    return payload


class ClusterInferenceAnalyzer:
    """Probability-thresholded Bayesian summaries for a cluster source table."""

    def __init__(
        self,
        data,
        *,
        probability_column: str = "probability",
        sampling: SamplingConfig | None = None,
    ) -> None:
        self.data = data
        self.probability_column = probability_column
        self.sampling = sampling or SamplingConfig()

    def select(self, probability_threshold: float | None = None):
        """Return data above an optional probability threshold."""
        if probability_threshold is None:
            return self.data
        return self.data[self.data[self.probability_column] >= probability_threshold]

    @staticmethod
    def _normalise_thresholds(values) -> np.ndarray:
        thresholds = np.asarray(values, dtype=float)
        return np.where(thresholds > 1, thresholds / 100, thresholds)

    def distance_and_parallax_by_probability(
        self,
        probability_thresholds=(0.5, 0.6, 0.7, 0.8),
        *,
        return_trace: bool = False,
        progressbar: bool | None = None,
        distance_prior=None,
        fractional_parallax_error_max: float = 0.1,
        distance_column: str = "r_med_geo",
        parallax_column: str = "parallax",
        parallax_error_column: str = "parallax_error",
    ) -> dict[str, list[Any]]:
        """Fit distance and parallax models for each probability threshold."""
        thresholds = self._normalise_thresholds(probability_thresholds)
        sampling = self.sampling
        if progressbar is not None:
            sampling = SamplingConfig(**{**sampling.__dict__, "progressbar": progressbar})
        results: dict[str, list[Any]] = {
            "mu_r_mean": [],
            "std_r_mean": [],
            "mu_r_std": [],
            "std_r_std": [],
            "mu_parallax_mean": [],
            "sigma_parallax_mean": [],
            "mu_parallax_std": [],
            "sigma_parallax_std": [],
        }
        traces = []
        for threshold in thresholds:
            subset = self.select(float(threshold))
            parallax = quantity_values(subset[parallax_column], u.mas)
            parallax_error = quantity_values(subset[parallax_error_column], u.mas)
            # Only positive parallaxes are physical distance estimators. Requiring
            # parallax > 0 (not just abs(error/parallax)) excludes negative parallaxes,
            # which would otherwise pass the ratio cut, and guards against div-by-zero.
            positive_parallax = parallax > 0
            fractional_error = np.divide(
                parallax_error,
                parallax,
                out=np.full_like(parallax, np.inf),
                where=positive_parallax,
            )
            useful = subset[positive_parallax & (fractional_error <= fractional_parallax_error_max)]
            distance_result = distance_model(
                useful,
                distance_column=distance_column,
                parallax_column=parallax_column,
                return_trace=return_trace,
                sampling=sampling,
            )
            parallax_result = fit_parallax_model(
                useful,
                parallax_column=parallax_column,
                prior_distance=distance_prior,
                return_trace=return_trace,
                sampling=sampling,
            )
            distance_payload = _dataclass_result_to_dict(distance_result)
            parallax_payload = _dataclass_result_to_dict(parallax_result)
            for key in ("mu_r_mean", "std_r_mean", "mu_r_std", "std_r_std"):
                results[key].append(distance_payload[key])
            for key in (
                "mu_parallax_mean",
                "sigma_parallax_mean",
                "mu_parallax_std",
                "sigma_parallax_std",
            ):
                results[key].append(parallax_payload[key])
            if return_trace:
                traces.extend([distance_result.trace, parallax_result.trace])
        if return_trace:
            results["traces"] = traces
        return results

    def proper_motion_by_probability(
        self,
        probability_thresholds=(0.5, 0.6, 0.7, 0.8),
        *,
        pmra_column: str = "pmra",
        pmdec_column: str = "pmdec",
        return_trace: bool = False,
        return_pmdist: bool = False,
        return_pmprob: bool = False,
        progressbar: bool | None = None,
    ):
        """Fit proper-motion Gaussian models for each probability threshold."""
        thresholds = self._normalise_thresholds(probability_thresholds)
        sampling = self.sampling
        if progressbar is not None:
            sampling = SamplingConfig(**{**sampling.__dict__, "progressbar": progressbar})
        stats = []
        distances = []
        pm_probabilities = []
        for threshold in thresholds:
            subset = self.select(float(threshold))
            fit = proper_motion_2d_gaussian(
                subset[pmra_column],
                subset[pmdec_column],
                return_trace=return_trace,
                sampling=sampling,
            )
            row = {"probability": float(threshold), **fit.results}
            if return_trace:
                row["trace"] = fit.trace
            stats.append(row)
            if return_pmdist:
                center_distance = np.sqrt(
                    (subset[pmra_column] - fit.results["mu_RA_mean"] * u.mas / u.yr) ** 2
                    + (subset[pmdec_column] - fit.results["mu_Dec_mean"] * u.mas / u.yr) ** 2
                )
                payload = {"probability": float(threshold), "distancepm": 1 / center_distance}
                distances.append(payload)
                if return_pmprob:
                    pm_probabilities.append((1 / center_distance) * subset[self.probability_column])
        if return_pmdist and return_pmprob:
            return stats, distances, pm_probabilities
        if return_pmdist:
            return stats, distances
        return stats

    def projected_velocity_by_probability(
        self,
        probability_thresholds=(0.5, 0.6, 0.7, 0.8),
        *,
        distance=None,
        return_trace: bool = False,
        progressbar: bool | None = None,
    ):
        """Fit projected-velocity models for each probability threshold."""
        thresholds = self._normalise_thresholds(probability_thresholds)
        sampling = self.sampling
        if progressbar is not None:
            sampling = SamplingConfig(**{**sampling.__dict__, "progressbar": progressbar})
        rows = []
        for threshold in thresholds:
            subset = self.select(float(threshold))
            result = velocity_model(
                subset,
                distance=distance,
                return_trace=return_trace,
                sampling=sampling,
            )
            rows.append({"probability": float(threshold), "model_results": _dataclass_result_to_dict(result)})
        return rows

    def radial_velocity_by_probability(
        self,
        probability_thresholds=(0.5, 0.6, 0.7, 0.8),
        *,
        return_trace: bool = False,
        progressbar: bool | None = None,
    ):
        """Fit radial-velocity models for each probability threshold."""
        thresholds = self._normalise_thresholds(probability_thresholds)
        sampling = self.sampling
        if progressbar is not None:
            sampling = SamplingConfig(**{**sampling.__dict__, "progressbar": progressbar})
        finite = self.data[np.isfinite(quantity_values(self.data["radial_velocity"], u.km / u.s))]
        analyzer = ClusterInferenceAnalyzer(
            finite,
            probability_column=self.probability_column,
            sampling=sampling,
        )
        rows = []
        for threshold in thresholds:
            subset = analyzer.select(float(threshold))
            result = radial_velocity_model(subset, return_trace=return_trace, sampling=sampling)
            rows.append(
                {
                    "probability": float(threshold),
                    "model_results": _dataclass_result_to_dict(result),
                    "len_data": len(subset),
                }
            )
        return rows


def FitProperMotion2DGaussian(pm_RA, pm_Dec, return_trace=False, progressbar=False):
    """Legacy-compatible alias for :func:`proper_motion_2d_gaussian`."""
    fit = proper_motion_2d_gaussian(
        pm_RA,
        pm_Dec,
        return_trace=return_trace,
        sampling=SamplingConfig(progressbar=progressbar),
    )
    payload = {"results": fit.results}
    if return_trace:
        payload["trace"] = fit.trace
    return payload


def parallax_determination(data, prob_thresholds=(50, 60, 70, 80), return_trace=False, progressbar=False, **kwargs):
    """Legacy-compatible non-plotting parallax/distance threshold summary."""
    kwargs.pop("savefig", None)
    kwargs.pop("paper_single", None)
    kwargs.pop("parallax_hist", None)
    distance_prior = kwargs.pop("distance_prior", None)
    analyzer = ClusterInferenceAnalyzer(data, sampling=SamplingConfig(progressbar=progressbar))
    return analyzer.distance_and_parallax_by_probability(
        prob_thresholds,
        return_trace=return_trace,
        progressbar=progressbar,
        distance_prior=distance_prior,
        **kwargs,
    )


def pm_determination(data, savefig=None, prob_number=(50, 60, 70, 80), progressbar=False, **kwargs):
    """Legacy-compatible non-plotting proper-motion threshold summary."""
    del savefig
    kwargs.pop("paper_single", None)
    analyzer = ClusterInferenceAnalyzer(data, sampling=SamplingConfig(progressbar=progressbar))
    return analyzer.proper_motion_by_probability(prob_number, progressbar=progressbar, **kwargs)


def velocity_determination(
    data,
    prob_thresholds=(50, 60, 70, 80),
    return_trace=False,
    progressbar=False,
    savefig=None,
    paper_single=False,
    distance=None,
):
    """Legacy-compatible non-plotting projected-velocity threshold summary."""
    del savefig, paper_single
    analyzer = ClusterInferenceAnalyzer(data, sampling=SamplingConfig(progressbar=progressbar))
    return analyzer.projected_velocity_by_probability(
        prob_thresholds,
        distance=distance,
        return_trace=return_trace,
        progressbar=progressbar,
    )


def rv_determination(data, prob_thresholds=(50, 60, 70, 80), return_trace=False, progressbar=False):
    """Legacy-compatible radial-velocity threshold summary."""
    analyzer = ClusterInferenceAnalyzer(data, sampling=SamplingConfig(progressbar=progressbar))
    return analyzer.radial_velocity_by_probability(
        prob_thresholds,
        return_trace=return_trace,
        progressbar=progressbar,
    )


__all__ = [
    "ClusterInferenceAnalyzer",
    "DistanceFitResult",
    "FitProperMotion2DGaussian",
    "ParallaxFitResult",
    "ProperMotionFitResult",
    "SamplingConfig",
    "VelocityFitResult",
    "distance_model",
    "fit_parallax_model",
    "parallax_determination",
    "pm_determination",
    "proper_motion_2d_gaussian",
    "radial_velocity_model",
    "rv_determination",
    "velocity_determination",
    "velocity_model",
]
