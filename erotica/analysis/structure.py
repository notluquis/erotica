"""Cluster structural measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, angular_separation
from astropy.table import QTable
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from .units import linear_size, quantity_values


@dataclass(frozen=True)
class CenterFitResult:
    """Result of a sky-position KDE center fit."""

    ra: u.Quantity
    dec: u.Quantity
    bandwidth: float
    density_peak: float


@dataclass(frozen=True)
class KingProfileResult:
    """Lightweight King-profile parameter container."""

    core_radius: u.Quantity
    tidal_radius: u.Quantity
    background: float | None = None


@dataclass(frozen=True)
class RadialDensityProfile:
    """Radial surface-density profile derived from a source table."""

    radius: u.Quantity
    density: np.ndarray
    density_error: np.ndarray
    counts: np.ndarray
    distances: u.Quantity
    method: str


def _coord_column(table: QTable, column: str, unit: u.UnitBase) -> u.Quantity:
    values = table[column]
    if hasattr(values, "to"):
        return values.to(unit)
    return values * unit


def half_mass_radius(
    data: QTable,
    center,
    *,
    mass_column: str = "mass",
    mass_error_column: str = "mass_std",
    ra_column: str = "ra",
    dec_column: str = "dec",
    distance=None,
):
    """Calculate angular and optionally linear half-mass radius."""
    if isinstance(center, SkyCoord):
        center_coord = center
    else:
        center_coord = SkyCoord(ra=center[0], dec=center[1], frame="icrs", unit="deg")

    separations = angular_separation(
        _coord_column(data, ra_column, u.deg),
        _coord_column(data, dec_column, u.deg),
        center_coord.ra,
        center_coord.dec,
    ).to(u.arcmin)
    order = np.argsort(separations)
    masses = data[mass_column][order]
    cumulative_mass = np.nancumsum(masses)
    half_total_mass = np.nanmax(cumulative_mass) / 2
    cumulative_values = quantity_values(cumulative_mass, u.Msun)
    idx = int(np.searchsorted(cumulative_values, half_total_mass.to(u.Msun).value))
    radius = separations[order][idx].to(u.arcmin)

    if mass_error_column in data.colnames:
        mass_errors = data[mass_error_column][order]
        cumulative_mass_error = np.sqrt(np.nancumsum(mass_errors**2))
        radius_error = cumulative_mass_error[idx] / cumulative_mass[idx] * radius
    else:
        radius_error = np.nan * u.arcmin

    if distance is None:
        return radius, radius_error.to(u.arcmin)

    linear_radius = linear_size(radius, distance)
    linear_error = linear_size(radius + radius_error, distance) - linear_radius
    return radius, linear_radius, radius_error.to(u.arcmin), linear_error.to(u.pc)


def calculate_half_light_radius(
    data,
    center=None,
    *legacy_args,
    magnitude_column: str = "Gmag",
    magnitude_error_column: str = "e_Gmag",
    ra_column: str = "ra",
    dec_column: str = "dec",
    distance=None,
):
    """Calculate the half-light radius from cumulative flux.

    New API accepts a table and a sky center. The legacy notebook API
    ``(Gmag, Gmag_err, distance_to_center, distance_to_cluster=None)`` is also
    supported and returns the original dictionary keys.
    """
    if legacy_args:
        if len(legacy_args) > 2:
            raise TypeError("Legacy call accepts Gmag, Gmag_err, distance_to_center, [distance].")
        gmag = data
        gmag_err = center
        distance_to_center = legacy_args[0]
        distance_to_cluster = legacy_args[1] if len(legacy_args) == 2 else distance
        mag_values = quantity_values(gmag, u.mag)
        mag_errors = quantity_values(gmag_err, u.mag)
        separations = distance_to_center
        luminosity = 10 ** (-0.4 * (mag_values - np.nanmin(mag_values)))
        luminosity_err = luminosity * np.log(10) * 0.4 * mag_errors
        order = np.argsort(separations)
        sorted_separations = separations[order]
        cumulative = np.nancumsum(luminosity[order])
        cumulative_err = np.sqrt(np.nancumsum(luminosity_err[order] ** 2))
        idx = int(np.argmax(cumulative >= cumulative[-1] / 2))
        radius = sorted_separations[idx]
        radius_err = cumulative_err[idx] / cumulative[idx] * radius
        results = {"R_h": radius, "R_h_error": radius_err}
        if distance_to_cluster is not None:
            linear_radius = linear_size(radius.to(u.arcmin), distance_to_cluster)
            upper = linear_size((radius + radius_err).to(u.arcmin), distance_to_cluster)
            lower = linear_size((radius - radius_err).to(u.arcmin), distance_to_cluster)
            results["R_h_linear"] = linear_radius
            results["R_h_linear_error"] = (upper - lower) / 2
            results["R_h_linear_bounds"] = (lower, upper)
        return results

    if isinstance(center, SkyCoord):
        center_coord = center
    else:
        center_coord = SkyCoord(ra=center[0], dec=center[1], frame="icrs", unit="deg")
    separations = angular_separation(
        _coord_column(data, ra_column, u.deg),
        _coord_column(data, dec_column, u.deg),
        center_coord.ra,
        center_coord.dec,
    ).to(u.arcmin)
    order = np.argsort(separations)
    flux = 10 ** (quantity_values(data[magnitude_column][order], u.mag) / -2.5)
    cumulative_flux = np.nancumsum(flux)
    idx = int(np.searchsorted(cumulative_flux, np.nanmax(cumulative_flux) / 2))
    radius = separations[order][idx].to(u.arcmin)
    if distance is None:
        return radius
    return radius, linear_size(radius, distance)


def center_determination(
    data: QTable,
    *,
    ra_column: str = "ra",
    dec_column: str = "dec",
    ra_error_column: str = "ra_error",
    dec_error_column: str = "dec_error",
    bandwidths: np.ndarray | None = None,
    kernels=("gaussian",),
    grid_ra=None,
    grid_dec=None,
    weights=None,
    return_density: bool = False,
    return_grids: bool = False,
    return_bestparams: bool = False,
):
    """Estimate a cluster center with 2D KDE over sky coordinates."""
    ra = quantity_values(data[ra_column], u.deg)
    dec = quantity_values(data[dec_column], u.deg)
    coords = np.vstack([ra, dec]).T
    finite = np.all(np.isfinite(coords), axis=1)
    coords = coords[finite]
    if len(coords) == 0:
        raise ValueError("No finite coordinates available for center determination.")
    if bandwidths is None:
        if ra_error_column in data.colnames and dec_error_column in data.colnames:
            min_bandwidth = np.nanmean(
                [
                    np.nanmean(quantity_values(data[ra_error_column], u.deg)),
                    np.nanmean(quantity_values(data[dec_error_column], u.deg)),
                ]
            )
            max_bandwidth = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), min_bandwidth)
            bandwidths = np.linspace(max(min_bandwidth, 1e-4), max(max_bandwidth, 1e-3), 60)
        else:
            span = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))
            bandwidths = np.linspace(max(span / 100, 1e-4), max(span / 8, 1e-3), 20)
    grid = GridSearchCV(KernelDensity(), {"bandwidth": bandwidths, "kernel": list(kernels)})
    if weights is None:
        grid.fit(coords)
    else:
        grid.fit(coords, sample_weight=np.asarray(weights, dtype=float))
    kde = grid.best_estimator_
    if grid_ra is None or grid_dec is None:
        pad_ra = np.ptp(coords[:, 0]) / 8
        pad_dec = np.ptp(coords[:, 1]) / 8
        ra_grid = np.linspace(np.nanmin(coords[:, 0]) - pad_ra, np.nanmax(coords[:, 0]) + pad_ra, 200)
        dec_grid = np.linspace(np.nanmin(coords[:, 1]) - pad_dec, np.nanmax(coords[:, 1]) + pad_dec, 200)
        xx, yy = np.meshgrid(ra_grid, dec_grid)
    else:
        xx, yy = grid_ra, grid_dec
    sample = np.vstack([xx.ravel(), yy.ravel()]).T
    density = np.exp(kde.score_samples(sample)).reshape(xx.shape)
    peak = np.unravel_index(np.argmax(density), density.shape)
    result = CenterFitResult(
        ra=xx[peak] * u.deg,
        dec=yy[peak] * u.deg,
        bandwidth=float(grid.best_params_["bandwidth"]),
        density_peak=float(density[peak]),
    )
    if return_density or return_grids or return_bestparams:
        ra_error = np.nanmean(quantity_values(data[ra_error_column], u.deg)) if ra_error_column in data.colnames else 0.0
        dec_error = np.nanmean(quantity_values(data[dec_error_column], u.deg)) if dec_error_column in data.colnames else 0.0
        bw = result.bandwidth * u.deg
        payload = {
            "center_coords": (result.ra, result.dec),
            "center_coords_error": (
                np.sqrt(ra_error**2 + result.bandwidth**2) * u.deg,
                np.sqrt(dec_error**2 + result.bandwidth**2) * u.deg,
            ),
        }
        if return_density:
            payload["density"] = density
        if return_grids:
            payload["grid_ra"] = xx
            payload["grid_dec"] = yy
        if return_bestparams:
            payload["best_params"] = {**grid.best_params_, "bandwidth_quantity": bw}
        return payload
    return result


def density_annulus_calculator_width(
    data,
    center,
    width,
    *,
    ra_column="ra",
    dec_column="dec",
    return_radius_gen: bool = False,
):
    """Calculate annulus densities using fixed radial bin width."""
    center_coord = center if isinstance(center, SkyCoord) else SkyCoord(center[0], center[1], unit="deg")
    distances = angular_separation(
        _coord_column(data, ra_column, u.deg),
        _coord_column(data, dec_column, u.deg),
        center_coord.ra,
        center_coord.dec,
    ).to(u.arcmin)
    width = width.to(u.arcmin) if hasattr(width, "to") else width * u.arcmin
    max_radius = np.nanmax(distances.value)
    edges = np.unique(
        np.concatenate(
            [
                np.arange(0, max_radius / 3, 0.5 * width.value),
                np.arange(max_radius / 3, 2 * max_radius / 3, width.value),
                np.arange(2 * max_radius / 3, 1.1 * max_radius, 2 * width.value),
            ]
        )
    )
    densities = []
    density_errors = []
    radii = []
    counts = []
    for inner, outer in zip(edges[:-1], edges[1:]):
        mask = (distances >= inner * u.arcmin) & (distances < outer * u.arcmin)
        count = int(np.count_nonzero(mask))
        area = np.pi * (outer**2 - inner**2)
        if count == 0:
            continue
        counts.append(count)
        densities.append(count / area)
        density_errors.append(np.sqrt(count) / area)
        radii.append((inner + outer) / 2)
    result = {
        "radius": np.asarray(radii) * u.arcmin,
        "density": np.asarray(densities),
        "counts": np.asarray(counts),
        "edges": edges * u.arcmin,
        "density_annulus": densities,
        "density_errors_annulus": density_errors,
        "d_center": distances,
    }
    if return_radius_gen:
        result["radius_gen"] = radii
    return result


def density_annulus_calculator_equip(data, center, *, ra_column="ra", dec_column="dec", return_radius_gen=False):
    """Calculate annulus densities with approximately equal-count radial bins."""
    center_coord = center if isinstance(center, SkyCoord) else SkyCoord(center[0], center[1], unit="deg")
    distances = angular_separation(
        _coord_column(data, ra_column, u.deg),
        _coord_column(data, dec_column, u.deg),
        center_coord.ra,
        center_coord.dec,
    ).to(u.arcmin)
    n = len(data)
    if n < 2:
        raise ValueError("At least two sources are required for annulus densities.")
    n_bins = max(1, int(2 * n ** (2 / 5)))
    sorted_distances = np.sort(distances.value)
    edges = np.interp(
        np.linspace(0, len(sorted_distances) - 1, n_bins + 1),
        np.arange(len(sorted_distances)),
        sorted_distances,
    )
    densities = []
    density_errors = []
    radii = []
    counts = []
    for inner, outer in zip(edges[:-1], edges[1:]):
        if outer <= inner:
            continue
        mask = (distances.value >= inner) & (distances.value < outer)
        count = int(np.count_nonzero(mask))
        area = np.pi * (outer**2 - inner**2)
        counts.append(count)
        densities.append(count / area)
        density_errors.append(np.sqrt(count) / area)
        radii.append((inner + outer) / 2)
    result = {
        "radius": np.asarray(radii) * u.arcmin,
        "density": np.asarray(densities),
        "counts": np.asarray(counts),
        "edges": edges * u.arcmin,
        "density_annulus": densities,
        "density_errors_annulus": density_errors,
        "d_center": distances,
    }
    if return_radius_gen:
        result["radius_gen"] = radii
    return result


def radial_density_profile(data, center, *, method="equip", width=None, **kwargs) -> RadialDensityProfile:
    """Return a typed radial-density profile for a cluster source table."""
    if method == "equip":
        raw = density_annulus_calculator_equip(data, center, return_radius_gen=True, **kwargs)
    elif method in {"width", "fixed_width", "kde_width", "kde_wdith"}:
        if width is None:
            raise ValueError("width is required for fixed-width radial density profiles.")
        raw = density_annulus_calculator_width(data, center, width, return_radius_gen=True, **kwargs)
        method = "width"
    else:
        raise ValueError("method must be 'equip' or 'width'.")
    return RadialDensityProfile(
        radius=np.asarray(raw["radius_gen"], dtype=float) * u.arcmin,
        density=np.asarray(raw["density_annulus"], dtype=float),
        density_error=np.asarray(raw["density_errors_annulus"], dtype=float),
        counts=np.asarray(raw["counts"], dtype=int),
        distances=raw["d_center"],
        method=method,
    )


def king_profile(radius, *args, core_radius=None, tidal_radius=None, background=0.0, amplitude=1.0):
    """Projected King profile shape."""
    r = quantity_values(radius)
    if len(args) == 4:
        amplitude, background, core_radius, tidal_radius = args
    elif len(args) == 2:
        core_radius, tidal_radius = args
    elif args:
        raise TypeError("king_profile accepts either (radius, rc, rt) or legacy (radius, k, b, rc, rt).")
    if core_radius is None or tidal_radius is None:
        raise TypeError("core_radius and tidal_radius are required.")
    rc = core_radius.to(u.arcmin).value if hasattr(core_radius, "to") else float(core_radius)
    rt = tidal_radius.to(u.arcmin).value if hasattr(tidal_radius, "to") else float(tidal_radius)
    profile = (1 / np.sqrt(1 + (r / rc) ** 2) - 1 / np.sqrt(1 + (rt / rc) ** 2)) ** 2
    return amplitude * profile + background


@dataclass(frozen=True)
class KingPriors:
    """Scale-free priors for the unbinned King fit.

    Every value is a **fixed constant, independent of the data being fit**. That
    is the point: the binned :func:`RDP_bayesian` derives all four of its prior
    bounds from ``nanstd``/``nanmin``/``nanmax`` of the observed densities and
    radii, which is the data used twice and was raised by the P01 referee
    ("0.8 Tmax, 1.5 Tmax ... appear arbitrary"). Half-Cauchy priors are the form
    used by Olivares et al. (2018, A&A 612, A70) for exactly this reason: they
    are scale-free and heavy-tailed, so an order-of-magnitude error in the scale
    costs little.

    .. warning::
       The half-Cauchy is built as ``pm.HalfStudentT(nu=1, sigma=...)``, **not**
       ``pm.HalfCauchy(beta=...)``, which they are mathematically identical to.
       The cause is in **PyTensor**, not PyMC: ``pytensor/link/numba/dispatch/
       random.py`` implements ``CauchyRV`` as ``(loc + z) / scale`` instead of
       ``loc + scale * z``, so numba-backed draws get location ``loc/scale`` and
       scale ``1/scale``. numba is the *default* linker; the scipy and JAX paths
       are correct, as is ``logp``. NUTS reads ``logp``, so posteriors are
       unaffected -- but ``sample_prior_predictive`` reads the draws, so every
       prior-predictive check built on ``HalfCauchy`` is silently wrong.
       ``HalfStudentT`` is correct in both paths. Full analysis and a one-line
       patch: ``tools/validation/pytensor_cauchy_bug_report.md``.

    Attributes
    ----------
    r_c_scale, r_t_scale : float
        Half-Cauchy scales (arcmin) for the core radius and for the *increment*
        ``R_t - R_c``. Fitting the increment rather than ``R_t`` enforces
        ``R_t > R_c`` by construction instead of by a data-dependent bound.
    k_scale, b_scale : float
        Half-Cauchy scales for the cluster amplitude and the background level,
        in stars per square arcmin.
    """

    r_c_scale: float = 5.0
    r_t_scale: float = 20.0
    k_scale: float = 1.0
    b_scale: float = 1.0


def king_expected_count(k, b, R_c, R_t, field_radius, *, xp=np):
    r"""Expected number of stars inside a circular field, in closed form.

    Evaluates :math:`\Lambda = \int_0^{R_f} 2\pi r\,\Sigma(r)\,\mathrm{d}r` for the
    King profile with an additive background. This is the normalisation of the
    unbinned point-process likelihood, so it is evaluated at every leapfrog step;
    a closed form avoids quadrature inside the PyTensor graph.

    Expanding :math:`(u - c)^2` with :math:`u = (1 + (r/R_c)^2)^{-1/2}` and
    :math:`c = R_c/\sqrt{R_c^2 + R_t^2}` gives three elementary integrals:

    .. math::

        I(R) = \frac{R_c^2}{2}\ln\!\left(1 + \frac{R^2}{R_c^2}\right)
             - 2cR_c\left(\sqrt{R_c^2 + R^2} - R_c\right)
             + \frac{c^2 R^2}{2}

    and :math:`\Lambda = 2\pi k\,I(\min(R_t, R_f)) + \pi b R_f^2`. The cluster
    term stops at ``R_t`` (the profile is zero beyond it) or at the field edge,
    whichever comes first; the background covers the whole field.

    Parameters
    ----------
    k, b, R_c, R_t : float or tensor
        King amplitude, background, core radius and tidal radius.
    field_radius : float
        Radius of the circular selection footprint, same units as `R_c`.
    xp : module, optional
        Array namespace supplying ``log``, ``sqrt`` and ``minimum``. Defaults to
        NumPy; pass ``pymc.math`` to build a symbolic graph.

    Returns
    -------
    float or tensor
        The expected count.

    Notes
    -----
    Verified against :func:`scipy.integrate.quad` to a relative error below
    ``1e-9`` across four decades of parameter space; see
    ``tests/test_structure.py::test_king_normalisation_matches_quadrature``.
    """
    c = R_c / xp.sqrt(R_c**2 + R_t**2)
    upper = xp.minimum(R_t, field_radius)
    integral = (
        (R_c**2 / 2.0) * xp.log(1.0 + (upper / R_c) ** 2)
        - 2.0 * c * R_c * (xp.sqrt(R_c**2 + upper**2) - R_c)
        + c**2 * upper**2 / 2.0
    )
    return 2.0 * np.pi * k * integral + np.pi * b * field_radius**2


def king_expected_count_weighted(
    k, b, R_c, R_t, field_radius, completeness, *, xp=np, nodes: int = 256
):
    r"""Expected *detected* count when completeness varies with radius.

    Evaluates :math:`\Lambda = \int_0^{R_f} 2\pi r\,\Sigma(r)\,\bar{S}(r)\,\mathrm{d}r`,
    where :math:`\bar{S}(r)` is the survey's detection probability averaged over the
    annulus at radius `r`. With a non-constant :math:`\bar{S}` there is no closed form,
    so a fixed Gauss-Legendre rule is used: the nodes and the completeness at them are
    computed **once**, in NumPy, and the sampler only evaluates a weighted sum over
    them. Nothing data-dependent is re-evaluated inside the graph.

    Parameters
    ----------
    completeness : callable or array-like
        Either ``S(r)``, called once on the quadrature nodes, or an array already
        aligned with them from a previous call using the same `nodes` and
        `field_radius`.
    nodes : int, default 256
        Gauss-Legendre order. This integrates the King form itself to machine
        precision; the practical limit is how smooth :math:`\bar{S}(r)` is.

    Returns
    -------
    float or tensor
        Expected number of *detected* stars.

    See Also
    --------
    king_expected_count : the closed form, valid when completeness is uniform.
    """
    x, w = np.polynomial.legendre.leggauss(int(nodes))
    r_nodes = 0.5 * field_radius * (x + 1.0)  # map [-1, 1] -> [0, R_f]
    w_nodes = 0.5 * field_radius * w
    s_nodes = np.asarray(
        completeness(r_nodes) if callable(completeness) else completeness, dtype=float
    )
    if s_nodes.shape != r_nodes.shape:
        raise ValueError(
            f"completeness has shape {s_nodes.shape}, expected {r_nodes.shape} "
            f"(nodes={nodes}). Pass a callable, or an array aligned with the same "
            "`nodes` and `field_radius`."
        )
    if np.any(~np.isfinite(s_nodes)) or np.any(s_nodes < 0) or np.any(s_nodes > 1):
        raise ValueError("completeness must be finite and within [0, 1]: it is a probability.")

    core = 1.0 / xp.sqrt(1.0 + (r_nodes / R_c) ** 2)
    edge = 1.0 / xp.sqrt(1.0 + (R_t / R_c) ** 2)
    switch = getattr(xp, "switch", None)
    sigma = (
        switch(r_nodes <= R_t, k * (core - edge) ** 2 + b, b)
        if switch is not None
        else np.where(r_nodes <= R_t, k * (core - edge) ** 2 + b, b)
    )
    return xp.sum(w_nodes * 2.0 * np.pi * r_nodes * sigma * s_nodes)


def king_unbinned(
    radii,
    *,
    field_radius,
    completeness=None,
    priors: KingPriors | None = None,
    tidal_prior: tuple[float, float] | None = None,
    sampling=None,
    progressbar: bool = False,
    return_trace: bool = True,
):
    r"""Fit a King profile to stellar radii directly, without binning.

    Models the sky positions as an inhomogeneous Poisson point process with
    intensity :math:`\lambda(r) = 2\pi r\,\Sigma(r)`, giving the log-likelihood

    .. math:: \log L = \sum_i \log \lambda(r_i) - \Lambda ,

    the continuous form of the Cash (1979) statistic. There are no bins, so
    there is no binning choice for a referee to question, and no shared scatter
    parameter: a point process has no nuisance ``sigma``.

    Parameters
    ----------
    radii : array-like or Quantity
        Angular distance of every star from the cluster centre. Must be the
        **complete** sample inside `field_radius` -- the normalisation integral
        assumes the footprint is that full disc.
    field_radius : float or Quantity
        Radius of the circular selection footprint, in arcmin.
    completeness : callable or array-like, optional
        Radial detection probability :math:`\bar{S}(r)` of the survey. When given,
        the fit models the *detected* intensity
        :math:`\lambda(r) = 2\pi r\,\Sigma(r)\,\bar{S}(r)` and the recovered
        ``k``/``R_c``/``R_t`` describe the **true** cluster rather than the
        observed one. Build it from :mod:`erotica.selection`, which wraps the
        Gaia DR3 selection function of Cantat-Gaudin et al. (2023). Because
        :math:`\sum_i \log \bar{S}(r_i)` does not depend on any parameter it
        cancels from the log-likelihood, so only the normalisation changes.
    priors : KingPriors, optional
        Scale-free priors. The defaults are constants, not functions of `radii`.
    tidal_prior : tuple of float, optional
        ``(mu, sigma)`` in arcmin for a physically motivated prior on ``R_t``,
        e.g. the Jacobi radius from
        :func:`~erotica.analysis.dynamics.tidal_radius_prior`. When given, it
        replaces the scale-free half-Cauchy on the ``R_t - R_c`` increment.
    sampling : SamplingConfig, optional
        Sampler settings. Defaults to 2000 draws, 1000 tuning.
    return_trace : bool, default True
        Keep the full posterior in the result. **Defaults to True**, unlike the
        binned fits: collapsing a posterior to two scalars on exit is what stops
        uncertainty reaching the derived dynamical quantities.

    Returns
    -------
    dict
        Posterior medians and standard deviations for ``k``, ``b``, ``R_c`` and
        ``R_t``, plus ``king_trace`` and the ``field_radius`` actually used.

    Notes
    -----
    Why unbinned rather than a per-annulus Poisson likelihood: the package's
    default binner (``method="equip"``) uses **equal-count** annuli, so the count
    per bin is fixed by construction and the *area* is what varies. The Poisson
    dispersion index ``Var(N_i)/E(N_i)`` is then approximately ``1/n_bins``
    -- about 0.04 at the 25 bins the published fit uses -- rather than the 1.0 a
    Poisson likelihood asserts. See
    ``tools/validation/king_binning_likelihood.py`` and the decision log.

    The ``R_t`` recovered by any King fit is the cluster's *time-averaged* tidal
    radius, not its perigalactic one (Küpper et al. 2010, MNRAS 407, 2241).
    """
    from .inference import SamplingConfig, _sample

    try:
        import pymc as pm
    except ImportError as exc:
        raise ImportError("PyMC is required for king_unbinned. Install the 'bayes' extra.") from exc

    r = np.asarray(quantity_values(radii, u.arcmin), dtype=float)
    r = r[np.isfinite(r) & (r > 0)]
    field_radius = float(quantity_values(field_radius, u.arcmin))
    if r.size < 10:
        raise ValueError("At least ten stars are required for an unbinned King fit.")
    if r.max() > field_radius:
        raise ValueError(
            f"{int((r > field_radius).sum())} stars lie outside field_radius="
            f"{field_radius:g} arcmin. The normalisation integral assumes the sample is "
            "complete within that disc, so this would bias the fit."
        )

    priors = priors or KingPriors()
    sampling = sampling or SamplingConfig(draws=2_000, tune=1_000, progressbar=progressbar)

    with pm.Model() as model:
        R_c = pm.HalfStudentT("R_c", nu=1, sigma=priors.r_c_scale)
        if tidal_prior is None:
            R_t = pm.Deterministic(
                "R_t", R_c + pm.HalfStudentT("dR", nu=1, sigma=priors.r_t_scale)
            )
        else:
            mu, sigma = tidal_prior
            R_t = pm.Deterministic(
                "R_t", R_c + pm.TruncatedNormal("dR", mu=mu, sigma=sigma, lower=0.0)
            )
        k = pm.HalfStudentT("k", nu=1, sigma=priors.k_scale)
        b = pm.HalfStudentT("b", nu=1, sigma=priors.b_scale)

        core = 1.0 / pm.math.sqrt(1.0 + (r / R_c) ** 2)
        edge = 1.0 / pm.math.sqrt(1.0 + (R_t / R_c) ** 2)
        surface = pm.math.switch(r <= R_t, k * (core - edge) ** 2 + b, b)
        log_intensity = pm.math.log(2.0 * np.pi * r * surface)
        if completeness is None:
            expected = king_expected_count(k, b, R_c, R_t, field_radius, xp=pm.math)
        else:
            expected = king_expected_count_weighted(
                k, b, R_c, R_t, field_radius, completeness, xp=pm.math
            )
        pm.Potential("point_process", pm.math.sum(log_intensity) - expected)

    trace = _sample(pm, model, sampling)
    results = {
        "field_radius": field_radius * u.arcmin,
        "n_stars": int(r.size),
        "completeness_corrected": completeness is not None,
    }
    for name, unit in (("k", None), ("b", None), ("R_c", u.arcmin), ("R_t", u.arcmin)):
        arr = np.asarray(trace.posterior[name].values, dtype=float)
        median, std = float(np.nanmedian(arr)), float(np.nanstd(arr))
        results[f"{name}_median"] = median * unit if unit else median
        results[f"{name}_std"] = std * unit if unit else std
    if return_trace:
        results["king_trace"] = trace
    return results


def _summarize_king_trace(trace):
    # NOTE: the point estimate reported for each King parameter is the posterior
    # *median* (robust to the skewed, prior-bounded R_t/R_c marginals). It is
    # exposed under two key families:
    #   ``*_median`` -- correctly named accessors (prefer these in new code);
    #   ``*_mean``   -- deprecated back-compat aliases holding the SAME median
    #                   value, kept so existing readers (paper figure notebooks,
    #                   ``graph_king``) reproduce identical numbers. Do not
    #                   "fix" these to arithmetic means -- that would silently
    #                   shift published paper quantities.
    values = {}
    for key in ("k", "b", "R_c", "R_t", "sigma"):
        arr = np.asarray(trace.posterior[key].values, dtype=float)
        median = float(np.nanmedian(arr))
        values[f"{key}_median"] = median
        values[f"{key}_mean"] = median  # deprecated alias, equals the median
        values[f"{key}_std"] = float(np.nanstd(arr))
    k_median = values["k_median"]
    b_median = values["b_median"]
    rc_median = values["R_c_median"]
    rt_median = values["R_t_median"]
    b_std = values["b_std"]
    bg_level = b_median + 3 * b_std
    return {
        "k_median": k_median,
        "b_median": b_median,
        "R_c_median": rc_median * u.arcmin,
        "R_t_median": rt_median * u.arcmin,
        "k_mean": k_median,  # deprecated alias of k_median
        "b_mean": b_median,  # deprecated alias of b_median
        "R_c_mean": rc_median * u.arcmin,  # deprecated alias of R_c_median
        "R_t_mean": rt_median * u.arcmin,  # deprecated alias of R_t_median
        "k_std": values["k_std"],
        "b_std": b_std,
        "R_c_std": values["R_c_std"],
        "R_t_std": values["R_t_std"],
        "king_std": values["sigma_median"],
        "bg_level": bg_level,
        "C": float(np.log(rt_median / rc_median)),
        "d_c": float(1 + k_median / bg_level) if bg_level else np.nan,
        "r_lim": float(rc_median * np.sqrt(k_median / (3 * b_std) - 1)) if b_std > 0 else np.nan,
    }


def RDP_bayesian(
    density_annulus,
    radius_gen,
    *,
    return_trace: bool = False,
    progressbar: bool = False,
    d_center=None,
    priors: bool = False,
    priors_parameters=None,
    return_priors=None,
    sampling=None,
):
    """Fit a King radial-density profile with PyMC."""
    from .inference import SamplingConfig, _sample

    try:
        import pymc as pm
    except ImportError as exc:
        raise ImportError("PyMC is required for RDP_bayesian. Install the 'bayes' extra.") from exc

    density_values = quantity_values(density_annulus)
    radius_values = quantity_values(radius_gen, u.arcmin)
    d_center_values = quantity_values(d_center, u.arcmin) if d_center is not None else radius_values
    finite = np.isfinite(density_values) & (density_values > 0) & np.isfinite(radius_values)
    density_values = density_values[finite]
    radius_values = radius_values[finite]
    if len(density_values) < 3:
        raise ValueError("At least three positive density bins are required.")
    sampling = sampling or SamplingConfig(draws=2_000, tune=1_000, progressbar=progressbar)
    upper_radius = float(1.5 * np.nanmax(d_center_values))
    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", sigma=max(float(np.nanstd(density_values)), 1e-6))
        b = pm.Uniform("b", lower=0, upper=max(2 * float(np.nanmin(density_values)), 1e-6))
        k = pm.Uniform("k", lower=b, upper=max(2 * float(np.nanmax(density_values)), 1e-6))
        R_c = pm.Uniform("R_c", lower=1e-6, upper=max(0.8 * float(np.nanmax(d_center_values)), 1e-5))
        R_t = pm.Uniform("R_t", lower=R_c, upper=max(upper_radius, 1e-4))
        r = np.asarray(radius_values, dtype=float)
        king = pm.Deterministic(
            "king",
            pm.math.switch(
                r <= R_t,
                k
                * (
                    (1 / pm.math.sqrt(1 + (r / R_c) ** 2))
                    - (1 / pm.math.sqrt(1 + (R_t / R_c) ** 2))
                )
                ** 2
                + b,
                b,
            ),
        )
        pm.Normal("obs_density", mu=king, sigma=sigma, observed=density_values)
    trace = _sample(pm, model, sampling)
    results = _summarize_king_trace(trace)
    if return_trace:
        results["king_trace"] = trace
    if return_priors:
        results["priors_results"] = dict(priors_parameters or {}, priors_used=bool(priors))
    return results


def RDP_bayesian_log_space(
    density_annulus,
    radius_gen,
    *,
    return_trace: bool = False,
    progressbar: bool = False,
    d_center=None,
    sampling=None,
):
    """Log-space variant of :func:`RDP_bayesian` for sparse radial profiles."""
    from .inference import SamplingConfig, _sample

    try:
        import pymc as pm
    except ImportError as exc:
        raise ImportError("PyMC is required for RDP_bayesian_log_space.") from exc

    density_values = quantity_values(density_annulus)
    radius_values = quantity_values(radius_gen, u.arcmin)
    d_center_values = quantity_values(d_center, u.arcmin) if d_center is not None else radius_values
    finite = np.isfinite(density_values) & (density_values > 0) & np.isfinite(radius_values) & (radius_values > 0)
    density_values = density_values[finite]
    radius_values = radius_values[finite]
    if len(density_values) < 3:
        raise ValueError("At least three positive density bins are required.")
    sampling = sampling or SamplingConfig(draws=2_000, tune=1_000, progressbar=progressbar)
    with pm.Model() as model:
        log_sigma = pm.Normal("log_sigma", mu=np.log(max(np.nanstd(np.log(density_values)), 1e-3)), sigma=1)
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))
        log_b = pm.Uniform("log_b", lower=np.log(1e-8), upper=np.log(max(0.5 * np.nanmax(density_values), 1e-7)))
        b = pm.Deterministic("b", pm.math.exp(log_b))
        log_k = pm.Uniform("log_k", lower=log_b, upper=np.log(max(2 * np.nanmax(density_values), 1e-6)))
        k = pm.Deterministic("k", pm.math.exp(log_k))
        log_R_c = pm.Uniform("log_R_c", lower=np.log(1e-4), upper=np.log(max(0.8 * np.nanmax(d_center_values), 1e-3)))
        R_c = pm.Deterministic("R_c", pm.math.exp(log_R_c))
        log_R_t = pm.Uniform("log_R_t", lower=log_R_c, upper=np.log(max(2 * np.nanmax(d_center_values), 1e-3)))
        R_t = pm.Deterministic("R_t", pm.math.exp(log_R_t))
        r = np.asarray(radius_values, dtype=float)
        king = pm.math.switch(
            r <= R_t,
            k
            * (
                (1 / pm.math.sqrt(1 + (r / R_c) ** 2))
                - (1 / pm.math.sqrt(1 + (R_t / R_c) ** 2))
            )
            ** 2
            + b,
            b,
        )
        pm.Normal("obs_log_density", mu=pm.math.log(king), sigma=sigma, observed=np.log(density_values))
    trace = _sample(pm, model, sampling)
    results = _summarize_king_trace(trace)
    if return_trace:
        results["king_trace"] = trace
    return results


class ClusterStructureAnalyzer:
    """Stateful structural analysis facade for a cluster source table.

    The methods are intentionally thin: they keep table selection and column
    naming in one place while delegating numerical work to the module-level
    functions, which remain easy to test independently.
    """

    def __init__(
        self,
        data: QTable,
        *,
        probability_column: str = "probability",
        ra_column: str = "ra",
        dec_column: str = "dec",
    ) -> None:
        self.data = data
        self.probability_column = probability_column
        self.ra_column = ra_column
        self.dec_column = dec_column

    def select(self, probability_threshold: float | None = None) -> QTable:
        """Return the table subset above ``probability_threshold``."""
        if probability_threshold is None:
            return self.data
        if self.probability_column not in self.data.colnames:
            raise ValueError(f"Missing probability column {self.probability_column!r}.")
        return self.data[self.data[self.probability_column] >= probability_threshold]

    def center(self, probability_threshold: float | None = None, **kwargs):
        """Estimate the KDE center for an optional probability-selected subset."""
        return center_determination(
            self.select(probability_threshold),
            ra_column=self.ra_column,
            dec_column=self.dec_column,
            **kwargs,
        )

    def centers(self, probability_thresholds=(0.5, 0.6, 0.7, 0.8), **kwargs):
        """Estimate centers for several probability thresholds."""
        return [self.center(threshold, **kwargs) for threshold in probability_thresholds]

    def half_mass_radius(self, center, probability_threshold: float | None = None, **kwargs):
        """Calculate the half-mass radius for a selected subset."""
        return half_mass_radius(
            self.select(probability_threshold),
            center,
            ra_column=self.ra_column,
            dec_column=self.dec_column,
            **kwargs,
        )

    def half_light_radius(self, center, probability_threshold: float | None = None, **kwargs):
        """Calculate the half-light radius for a selected subset."""
        return calculate_half_light_radius(
            self.select(probability_threshold),
            center,
            ra_column=self.ra_column,
            dec_column=self.dec_column,
            **kwargs,
        )

    def radial_density_profile(
        self,
        center,
        *,
        probability_threshold: float | None = None,
        method: str = "equip",
        width=None,
        **kwargs,
    ) -> RadialDensityProfile:
        """Build a radial-density profile for a selected subset."""
        return radial_density_profile(
            self.select(probability_threshold),
            center,
            method=method,
            width=width,
            ra_column=self.ra_column,
            dec_column=self.dec_column,
            **kwargs,
        )

    def fit_king_profile(
        self,
        profile: RadialDensityProfile,
        *,
        log_space: bool = False,
        **kwargs,
    ):
        """Fit a King profile to a :class:`RadialDensityProfile`."""
        fitter = RDP_bayesian_log_space if log_space else RDP_bayesian
        return fitter(
            profile.density,
            profile.radius,
            d_center=profile.distances,
            **kwargs,
        )


__all__ = [
    "CenterFitResult",
    "ClusterStructureAnalyzer",
    "KingProfileResult",
    "RadialDensityProfile",
    "RDP_bayesian",
    "RDP_bayesian_log_space",
    "calculate_half_light_radius",
    "center_determination",
    "density_annulus_calculator_equip",
    "density_annulus_calculator_width",
    "half_mass_radius",
    "king_profile",
    "radial_density_profile",
]
