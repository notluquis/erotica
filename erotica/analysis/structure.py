"""Cluster structural measurements."""

from __future__ import annotations

import warnings
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
        ra_grid = np.linspace(
            np.nanmin(coords[:, 0]) - pad_ra, np.nanmax(coords[:, 0]) + pad_ra, 200
        )
        dec_grid = np.linspace(
            np.nanmin(coords[:, 1]) - pad_dec, np.nanmax(coords[:, 1]) + pad_dec, 200
        )
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
        ra_error = (
            np.nanmean(quantity_values(data[ra_error_column], u.deg))
            if ra_error_column in data.colnames
            else 0.0
        )
        dec_error = (
            np.nanmean(quantity_values(data[dec_error_column], u.deg))
            if dec_error_column in data.colnames
            else 0.0
        )
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
    center_coord = (
        center if isinstance(center, SkyCoord) else SkyCoord(center[0], center[1], unit="deg")
    )
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
    for inner, outer in zip(edges[:-1], edges[1:], strict=True):
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


def density_annulus_calculator_equip(
    data, center, *, ra_column="ra", dec_column="dec", return_radius_gen=False
):
    """Calculate annulus densities with approximately equal-count radial bins."""
    center_coord = (
        center if isinstance(center, SkyCoord) else SkyCoord(center[0], center[1], unit="deg")
    )
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
    for inner, outer in zip(edges[:-1], edges[1:], strict=True):
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


def radial_density_profile(
    data, center, *, method="equip", width=None, **kwargs
) -> RadialDensityProfile:
    """Return a typed radial-density profile for a cluster source table.

    Thin typed wrapper over :func:`density_annulus_calculator_equip` and
    :func:`density_annulus_calculator_width`, which return loosely-keyed dicts.

    Parameters
    ----------
    data : QTable
        Source table carrying the sky-position columns named by `ra_column` and
        `dec_column` (default ``"ra"``/``"dec"``, forwarded via ``**kwargs``).
    center : SkyCoord or sequence of float
        Cluster centre. A bare ``(ra, dec)`` pair is interpreted as **degrees**;
        pass a :class:`~astropy.coordinates.SkyCoord` to be explicit.
    method : {"equip", "width"}, default "equip"
        ``"equip"`` uses approximately equal-count annuli whose edges are
        interpolated from the sorted radii, with
        ``n_bins = max(1, int(2 * n ** 0.4))`` -- so the bin count grows with
        sample size and is not a free choice. ``"width"`` uses fixed-width
        annuli that widen outwards (half `width` in the inner third of the
        field, `width` in the middle third, twice `width` beyond).
        ``"fixed_width"``, ``"kde_width"`` and the misspelled ``"kde_wdith"``
        are accepted aliases of ``"width"``, kept for older call sites; the
        returned ``method`` field is normalised to ``"width"`` for all of them.
    width : Quantity or float, optional
        Base annulus width for ``method="width"``, taken as arcmin when a plain
        float. Required for that method and ignored by ``"equip"``.
    **kwargs
        Forwarded to the annulus calculator -- in practice `ra_column` and
        `dec_column`.

    Returns
    -------
    RadialDensityProfile
        ``radius`` -- annulus midpoints (arcmin Quantity);
        ``density`` -- surface density in **stars per square arcmin**, a plain
        ``ndarray`` with no unit attached;
        ``density_error`` -- the Poisson root-N error on the same scale,
        ``sqrt(count) / area``;
        ``counts`` -- integer star count per annulus;
        ``distances`` -- every star's angular separation from `center` (arcmin
        Quantity, one entry per row of `data`, **not** per annulus);
        ``method`` -- the normalised method name actually used.

    Raises
    ------
    ValueError
        If `method` is unrecognised, if ``method="width"`` and `width` is
        ``None``, or (from the ``"equip"`` calculator) if `data` has fewer than
        two rows.

    Notes
    -----
    The two methods differ in how they treat empty annuli, which shows up as a
    difference in array length: ``"width"`` **drops** annuli containing no
    stars, while ``"equip"`` keeps them and records a density of zero. Do not
    assume ``len(counts)`` equals the nominal bin count.

    These binned profiles are for plotting and for the legacy binned fits. The
    fits this package recommends -- :func:`king_unbinned` and
    :func:`eff_unbinned` -- take the raw radii and never bin, precisely so that
    the choice made here cannot reach a parameter estimate.
    """
    if method == "equip":
        raw = density_annulus_calculator_equip(data, center, return_radius_gen=True, **kwargs)
    elif method in {"width", "fixed_width", "kde_width", "kde_wdith"}:
        if width is None:
            raise ValueError("width is required for fixed-width radial density profiles.")
        raw = density_annulus_calculator_width(
            data, center, width, return_radius_gen=True, **kwargs
        )
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
    r"""Projected King profile shape, plus an additive ``background``.

    .. math:: \Sigma(r) = \begin{cases}
                  k\left[\frac{1}{\sqrt{1+(r/r_c)^2}}
                       - \frac{1}{\sqrt{1+(r_t/r_c)^2}}\right]^2 + b
                  & r \le r_t \\
                  b & r > r_t
              \end{cases}

    .. important::
       **The** ``+ b`` **is not King's.** King (1962, AJ 67, 471) Eq. (14) is the bracket
       squared and nothing more. His background was handled *outside* the formula: subtracted
       from the counts where a wide-field plate reached past the cluster, and where it did
       not, chosen "so as to make the outermost points satisfy the empirical law". He also
       warns against exactly the misreading the extra term invites — *"the second term in
       brackets in Eq. (14) could be replaced by a single constant; it is written in this
       more complicated form in order to show the role of* :math:`r_t`" — i.e. the constant
       already inside the bracket is the **truncation** constant, not a background.

       The additive form is folk practice with no primary citation. Seleznev (2016, MNRAS
       456, 3757) introduces it as *"in order to take into account stellar background, this
       formula is supplemented by stellar background density* :math:`F_b` *as a constant
       addition"*, citing nobody.

       King identified the :math:`r_t \leftrightarrow` background degeneracy himself in the
       same paper, criticising Wallenquist for having *"underestimated the radii of the
       clusters and consequently chosen incorrect values for the background densities."*

       On a Gaia membership-selected sample ``b`` is **not** field contamination. See
       ``docs/design-notes/king_model_validity.md`` for the evidence and
       ``kb/methods/king-background-term.md`` in the research hub for the grounded node.

    Parameters
    ----------
    radius : Quantity or array-like
        Angular distance(s) from the cluster centre. A Quantity is **converted**
        to arcmin; a plain array is **assumed** to already be arcmin. Passing
        degrees as a plain array is silently wrong -- measured at a factor 480
        at :math:`r = 20`, which is why the conversion here is explicit.
    *args : float or Quantity, optional
        Legacy positional forms, kept for older call sites. Two positionals are
        read as ``(core_radius, tidal_radius)``; four as
        ``(amplitude, background, core_radius, tidal_radius)`` -- note that the
        four-argument order is ``k, b, R_c, R_t``, amplitude first, which is not
        the keyword order below. Any other non-zero count raises. Prefer the
        keywords.
    core_radius : float or Quantity
        :math:`r_c`, in arcmin when a plain float. Required (by keyword or
        positionally); ``None`` raises ``TypeError``.
    tidal_radius : float or Quantity
        :math:`r_t`, in arcmin when a plain float. Required on the same terms.
        The cluster term is **truncated** at :math:`r_t`: beyond it the return
        value is the background `b` alone, not zero and not a continuation of
        the formula. Evaluating the bracket past :math:`r_t` would send it
        negative and the square would make the profile *rise* again -- with
        ``rc=5, rt=20, k=1`` the untruncated expression gives 0.0206 at
        :math:`r = 50` against 0.0419 at :math:`r = 10`, i.e. half the central
        shape re-appearing outside the cluster. The truncation here matches
        :func:`_king_model`, :func:`_king_corona_model` and
        :func:`RDP_bayesian`, which all switch to `b` at :math:`r > R_t`, and
        the normalisation in :func:`king_expected_count`, which caps the
        cluster term at :math:`\min(R_t, R_f)`.
    background : float, default 0.0
        The additive :math:`b`, in the same surface-density units as
        `amplitude`. See the warning above: on a membership-selected sample this
        is not field contamination.
    amplitude : float, default 1.0
        The scale :math:`k`. The default of 1.0 makes the return value the
        dimensionless **shape**, which is what the plotting helpers want.

    Returns
    -------
    ndarray
        :math:`\Sigma(r)`, same shape as `radius`, equal to `background` for
        :math:`r > r_t`. Dimensionless with the default `amplitude` and
        `background`; otherwise it carries whatever units those two were given,
        since they are applied without conversion.

    Raises
    ------
    TypeError
        If the positional count is neither 0, 2 nor 4, or if `core_radius` or
        `tidal_radius` is missing.

    Notes
    -----
    The model is **empirical**: King calls Eq. (14) *"merely a convenient fitting formula"*.
    The dynamical King model is King (1966, AJ 71, 64), a different object with a
    concentration parameter :math:`W_0`. There is no "King 1964".
    """
    # Explicit unit: quantity_values(x) with no target strips whatever unit x carries.
    # Passing degrees where arcmin is meant returns silently wrong values -- measured, a
    # 480x error at r = 20. Plain arrays pass through unchanged and are taken as arcmin.
    r = quantity_values(radius, u.arcmin)
    if len(args) == 4:
        amplitude, background, core_radius, tidal_radius = args
    elif len(args) == 2:
        core_radius, tidal_radius = args
    elif args:
        raise TypeError(
            "king_profile accepts either (radius, rc, rt) or legacy (radius, k, b, rc, rt)."
        )
    if core_radius is None or tidal_radius is None:
        raise TypeError("core_radius and tidal_radius are required.")
    rc = core_radius.to(u.arcmin).value if hasattr(core_radius, "to") else float(core_radius)
    rt = tidal_radius.to(u.arcmin).value if hasattr(tidal_radius, "to") else float(tidal_radius)
    # The truncation is not cosmetic: past r_t the bracket goes negative and the square
    # sends the profile back UP, so an untruncated evaluation puts more light at 50' than
    # at 10' for rc=5, rt=20. `np.where` here is the same switch _king_model,
    # _king_corona_model and RDP_bayesian already apply, and the value outside is `b`, not
    # zero -- the background does not stop at the tidal radius.
    bracket = 1 / np.sqrt(1 + (r / rc) ** 2) - 1 / np.sqrt(1 + (rt / rc) ** 2)
    # `[()]` keeps a scalar input returning a scalar, as it did before the truncation
    # was added; np.where would otherwise hand back a 0-d array.
    profile = np.where(r <= rt, bracket**2, 0.0)[()]
    return amplitude * profile + background


# ---------------------------------------------------------------------------
# Model builders. Shared by the fit functions and by compare_radial_profiles,
# so a change to a prior or a likelihood cannot drift between the two.
# ---------------------------------------------------------------------------


def _king_model(pm, r, field_radius, priors, tidal_prior, completeness):
    with pm.Model() as model:
        R_c = pm.HalfStudentT("R_c", nu=1, sigma=priors.r_c_scale)
        if tidal_prior is None:
            R_t = pm.Deterministic("R_t", R_c + pm.HalfStudentT("dR", nu=1, sigma=priors.r_t_scale))
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
    return model


def _eff_model(pm, r, field_radius, priors, gamma, completeness=None):
    with pm.Model() as model:
        a = pm.HalfStudentT("a", nu=1, sigma=priors.a_scale)
        k = pm.HalfStudentT("k", nu=1, sigma=priors.k_scale)
        b = pm.HalfStudentT("b", nu=1, sigma=priors.b_scale)
        if gamma is None:
            g = pm.TruncatedNormal("gamma", mu=priors.gamma_mu, sigma=priors.gamma_sigma, lower=0.1)
        else:
            g = pm.Deterministic("gamma", pm.math.constant(float(gamma)))
        surface = k * (1.0 + (r / a) ** 2) ** (-g / 2.0) + b
        log_intensity = pm.math.log(2.0 * np.pi * r * surface)
        if completeness is None:
            expected = eff_expected_count(k, b, a, g, field_radius, xp=pm.math)
        else:
            # Same Gauss-Legendre rule as king_expected_count_weighted, so an
            # S(r) built for one profile can be passed straight to the other.
            x, w = np.polynomial.legendre.leggauss(256)
            r_nodes = 0.5 * field_radius * (x + 1.0)
            w_nodes = 0.5 * field_radius * w
            s_nodes = np.asarray(
                completeness(r_nodes) if callable(completeness) else completeness, dtype=float
            )
            if s_nodes.shape != r_nodes.shape:
                raise ValueError(
                    f"completeness has shape {s_nodes.shape}, expected {r_nodes.shape}"
                )
            if np.any(~np.isfinite(s_nodes)) or np.any(s_nodes < 0) or np.any(s_nodes > 1):
                raise ValueError("completeness must be finite and within [0, 1].")
            sig = k * (1.0 + (r_nodes / a) ** 2) ** (-g / 2.0) + b
            expected = pm.math.sum(w_nodes * 2.0 * np.pi * r_nodes * sig * s_nodes)
        pm.Potential("point_process", pm.math.sum(log_intensity) - expected)
    return model


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

    WHERE THE NUMBERS COME FROM
    ---------------------------
    Set from **Hunt & Reffert (2024)**, `2024A&A...686A..42H`, 5647 open clusters, never from the
    cluster in hand. Regenerate with ``tools/validation/fetch_hr24.py``. Angular radii in arcmin
    (their columns are published in **degrees** — see the units warning in :class:`EFFPriors`):

    ========  =======  =======  =======  =======
    quantity      16%   median      84%      99%
    ========  =======  =======  =======  =======
    ``r_c``      2.61     7.83    20.14    92.24
    ``r_t``      8.48    15.03    37.02   236.23
    ``r_J``     11.05    16.14    27.12    85.24
    ========  =======  =======  =======  =======

    A half-Cauchy with scale ``s`` has median ``s``, 95th percentile ``12.7 s``, 99th ``63.7 s``. The
    defaults sit **below** the catalogue medians so the prior does not push compact clusters outward,
    while the heavy tail keeps it weakly informative — the reason for the half-Cauchy form, following
    Olivares et al. (2018, A&A 612, A70), Gelman (2006) and Polson & Scott (2012).

    Attributes
    ----------
    r_c_scale, r_t_scale : float
        Half-Cauchy scales (arcmin) for the core radius and for the *increment* ``R_t - R_c``.
        Fitting the increment rather than ``R_t`` enforces ``R_t > R_c`` by construction instead of by
        a data-dependent bound. Defaults 5.0 and 20.0 against catalogue medians of 7.83′ and 15.03′.

        .. warning::
           **For NGC 6383 this prior determines the answer.** With a scale-free prior the posterior
           SD on ``R_t`` is ~7000′; with a Jacobi-informed prior at ±50% it returns 59.1 ± 11.4′.
           Restricting the fit to stars inside ``r_J`` does not help — ``R_t`` is still unconstrained
           at 85.9 ± 2661′. The measured reason is that **the footprint does not contain the object**:
           fitting the corona model gives ``P(R_2 > field) = 1.000``. Any ``R_t`` quoted must name its
           prior. See ``docs/design-notes/king_model_validity.md``.
    k_scale, b_scale : float
        Half-Cauchy scales for the cluster amplitude and the background level, in stars per square
        arcmin. **Neither is derived** — they are order-unity because surface densities in this
        geometry are order unity, not because a catalogue was consulted. The point-process
        normalisation makes the fit largely insensitive to ``k`` (it is set by the total count), which
        is an argument for why it matters little, not a derivation.

        .. important::
           ``b`` is **not** field contamination on a membership-selected sample. On NGC 6383 it
           accounts for **56% of the sample** against a target–decoy false-discovery proportion of
           a **median of 2.8%** (mean 10.5%, p90 30.8% over 40 realizations — the distribution is
           tail-dominated, so quote the median). What it absorbs is the corona, exactly as Seleznev (2016, MNRAS
           456, 3757) states: *"the cluster corona … is perceived by the approximation algorithm as
           part of the stellar background."* Report the fraction attributed to ``b`` alongside
           ``k``, ``R_c`` and ``R_t``, or the profile of a minority of the data is being presented as
           the profile of the cluster.
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

    model = _king_model(pm, r, field_radius, priors, tidal_prior, completeness)

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


@dataclass(frozen=True)
class EFFPriors:
    """Scale-free priors for the EFF profile. Fixed constants, not data-derived.

    WHERE THE NUMBERS COME FROM
    ---------------------------
    "Independent of the data being fitted" is necessary but not sufficient: the *values*
    also need a stated provenance, or they are arbitrary constants with a good excuse.
    These are set from **Hunt & Reffert (2024)**, `2024A&A...686A..42H`, an external
    catalogue of 5647 open clusters, never from the cluster in hand. Regenerate with
    ``tools/validation/fetch_hr24.py``. Their angular radii, converted to arcmin:

    ========  =======  =======  =======  =======
    quantity      16%   median      84%      99%
    ========  =======  =======  =======  =======
    ``r_c``      2.61     7.83    20.14    92.24
    ``r_t``      8.48    15.03    37.02   236.23
    ``r_tot``   13.98    25.43    54.81   362.22
    ========  =======  =======  =======  =======

    .. warning::
       Those columns are published in **degrees**, not arcmin. Reading them as arcmin
       understates every radius by 60x and would have set ``a_scale`` two orders of
       magnitude too small. Caught by checking the implied physical radius against the
       catalogue's own ``rcpc`` column, which agrees to the digit once the units are right.

    A half-Cauchy with scale ``s`` has median ``s``, 95th percentile ``12.7 s`` and 99th
    ``63.7 s``, so choosing ``s`` near the catalogue median makes the prior *weakly*
    informative: it covers the observed 99th percentile at well under its own 99th, and
    its tail is heavy enough that an order-of-magnitude error in the scale costs little.
    That heavy tail is the reason for the half-Cauchy form, following Olivares et al.
    (2018, A&A 612, A70), Gelman (2006) on weakly informative scale priors, and Polson &
    Scott (2012) on the half-Cauchy as a default.

    Attributes
    ----------
    a_scale : float
        Half-Cauchy scale (arcmin) for the EFF scale radius. Default 5.0, against a
        catalogue median core radius of 7.83'. Deliberately *below* the median so the
        prior does not push small clusters outward.
    k_scale, b_scale : float
        Half-Cauchy scales for amplitude and background, stars per square arcmin.
        **These two are the least defensible values here** and are flagged as such:
        they are order-unity because surface densities in this geometry are order unity,
        not because a catalogue was consulted. The point-process normalisation makes the
        fit largely insensitive to them (``k`` is determined by the total count), but that
        is an argument for why it does not matter much, not a derivation.
    gamma_mu, gamma_sigma : float
        Normal prior on the asymptotic slope, truncated positive. ``gamma = 4`` is the
        Plummer profile and ``gamma = 2`` is the untruncated-King limit. Observed values:
        Elson, Fall & Freeman (1987, ApJ 323, 54) report 2.2 <~ gamma <~ 3.2 for ten young
        LMC clusters; Mackey & Gilmore (2003a) get 2.01-3.79 (median 2.59) for 22 LMC
        clusters younger than 3e8 yr. So ``gamma_mu = 3.0`` sits inside the observed range
        and ``gamma_sigma = 2.0`` spans it several times over.

        .. important::
           **The convention matters and the literature is split.** This is the *surface*
           convention, ``Sigma ~ (1 + (r/a)^2)^(-gamma/2)``, used by EFF87 and Mackey &
           Gilmore. McLaughlin & van der Marel (2005) use a 3D convention in which
           ``gamma_MvdM = gamma_EFF + 1``. Comparing gamma across papers without checking
           which is meant will appear to shift every value by exactly 1.

        .. warning::
           This prior is **not** neutral where the data are weak. Measured in
           ``tools/validation/eff_gamma_bias.py``: at a footprint-to-scale ratio of 2 --
           the regime a quarter of the Hunt & Reffert census sits in -- the recovered
           ``gamma`` is biased by **+1.6**, and true values of 2.00 / 2.32 / 3.00 come back
           as 3.58 / 3.78 / 4.21. Widening the prior 2.5x does not remove it, so it is not
           simple prior pull; but in that regime the number reported is not a measurement.
    """

    a_scale: float = 5.0
    k_scale: float = 1.0
    b_scale: float = 1.0
    gamma_mu: float = 3.0
    gamma_sigma: float = 2.0


def eff_surface_density(radius, *, k, b, a, gamma):
    r"""EFF (Elson, Fall & Freeman 1987) surface density with a background.

    .. math:: \Sigma(r) = k\left(1 + (r/a)^2\right)^{-\gamma/2} + b

    Unlike King, this has **no tidal cutoff**: it declines as a power law
    :math:`r^{-\gamma}` forever. That is the honest model for a cluster whose
    truncation radius cannot be located in the data.

    Parameters
    ----------
    radius : Quantity or array-like
        Angular distance(s) from the centre. A Quantity is converted to arcmin;
        a plain array is assumed to already be arcmin.
    k : float
        Central amplitude, in the caller's surface-density units.
    b : float
        Additive background, same units as `k`. The caveats in
        :func:`king_profile` about what ``b`` is *not* apply here too.
    a : float or Quantity
        Scale radius :math:`a`, in arcmin when a plain number; a Quantity is
        **converted**, on the same convention as `radius` and as
        :func:`king_profile` applies to `core_radius` and `tidal_radius`.
        (Before 2026-08-04 this argument was not unit-aware and a Quantity
        raised :class:`~astropy.units.UnitConversionError` -- at ``1.0 +
        (r/a)**2``, the *first* addition, because `radius` had already been
        stripped to a bare array, so it failed for any angular unit even with
        ``b = 0``. The docstring blamed the ``+ b`` and was wrong about it.)
    gamma : float
        Power-law slope :math:`\gamma`. The projected density falls as
        :math:`r^{-\gamma}` at large radius. ``gamma=4`` is the Plummer profile.

    Returns
    -------
    ndarray
        :math:`\Sigma(r)`, same shape as `radius`, in the units of `k` and `b`.

    Notes
    -----
    Unbounded total mass is the price of no cutoff: the enclosed count
    integrates to a finite value over a finite field only, which is why
    :func:`eff_expected_count` always takes a `field_radius`.
    """
    r = quantity_values(radius, u.arcmin)
    # Same two-line convention as king_profile's rc/rt: convert a Quantity, take a
    # plain number as arcmin. Without it `r / a` carries 1/arcmin and the failure is
    # at `1.0 + (r/a)**2`, before `b` is ever reached.
    a = a.to(u.arcmin).value if hasattr(a, "to") else float(a)
    return k * (1.0 + (r / a) ** 2) ** (-gamma / 2.0) + b


def eff_expected_count(k, b, a, gamma, field_radius, *, xp=np):
    r"""Expected count inside a circular field for the EFF profile, in closed form.

    With :math:`z = (R_f/a)^2` and :math:`e = 1 - \gamma/2`,

    .. math:: \Lambda = \pi k a^2 \frac{(1+z)^{e} - 1}{e} + \pi b R_f^2 .

    The expression is singular at :math:`\gamma = 2` (where the integral is
    logarithmic), which is squarely inside the plausible range, so it is
    evaluated as :math:`\log(1+z)\cdot\mathrm{expm1}(x)/x` with
    :math:`x = e\log(1+z)`; that ratio tends to 1 as :math:`x \to 0` and is
    replaced by its series there. Both branches stay finite, so the gradient
    does not pick up a NaN from the unused one.

    Parameters
    ----------
    k, b : float or tensor
        EFF amplitude and additive background, in surface-density units.
    a : float or tensor
        Scale radius, in the **same units as** `field_radius`.
    gamma : float or tensor
        Power-law slope. The :math:`\gamma = 2` singularity is handled, so this
        may be sampled across it.
    field_radius : float or tensor
        Radius of the circular selection footprint, same units as `a`.
    xp : module, optional
        Array namespace supplying ``log``, ``exp`` and ``abs``, and optionally
        ``switch``. Defaults to NumPy, which takes the :func:`numpy.where`
        branch; pass ``pymc.math`` (which does provide ``switch``) to build a
        symbolic graph.

    Returns
    -------
    float or tensor
        The expected count :math:`\Lambda`.

        This function is **unit-agnostic**: it performs no conversion. With
        plain floats it returns a plain float, and `a` and `field_radius` must
        already share a unit -- passing arcmin for one and degrees for the other
        is silently wrong. With Astropy Quantities it propagates them and
        returns an area-dimensioned Quantity (``arcmin2`` for arcmin inputs),
        so the caller is responsible for `k` carrying the reciprocal units if a
        dimensionless count is wanted.

    Notes
    -----
    Verified against :func:`scipy.integrate.quad` to machine precision across
    :math:`\gamma \in [2, 5]`, including :math:`\gamma = 2` exactly; see
    ``tests/test_structure.py::test_eff_normalisation_matches_quadrature``.
    """
    z = (field_radius / a) ** 2
    log1pz = xp.log(1.0 + z)
    x = (1.0 - gamma / 2.0) * log1pz
    small = xp.abs(x) < 1e-6
    safe_x = xp.switch(small, 1.0, x) if hasattr(xp, "switch") else np.where(small, 1.0, x)
    series = 1.0 + x / 2.0 + x * x / 6.0
    expm1 = xp.exp(x) - 1.0
    ratio = (
        xp.switch(small, series, expm1 / safe_x)
        if hasattr(xp, "switch")
        else np.where(small, series, expm1 / safe_x)
    )
    return np.pi * k * a**2 * log1pz * ratio + np.pi * b * field_radius**2


def eff_unbinned(
    radii,
    *,
    field_radius,
    completeness=None,
    gamma=None,
    priors: EFFPriors | None = None,
    sampling=None,
    progressbar: bool = False,
    return_trace: bool = True,
):
    r"""Fit an EFF profile to stellar radii, without binning.

    Same inhomogeneous-Poisson point process as :func:`king_unbinned`, with
    :math:`\Sigma(r)` replaced by the EFF form. Because EFF has no tidal cutoff
    it has nothing to be unconstrained about, which is the point: for NGC 6383
    the King ``R_t`` is not locatable even inside the Jacobi radius.

    Parameters
    ----------
    radii : array-like or Quantity
        Angular distance of every star from the centre. Must be the **complete**
        sample inside `field_radius`, as in :func:`king_unbinned`. Non-finite
        and non-positive entries are dropped; at least ten must remain.
    field_radius : float or Quantity
        Radius of the circular selection footprint, in arcmin.
    completeness : callable or array-like, optional
        Radial detection probability, exactly as in :func:`king_unbinned`. Note
        that the selection acting on a sample is not only the survey's: the
        pipeline's own cuts induce their own :math:`\bar{S}(r)`, and for
        NGC 6383 that one is ~35x larger than Gaia's (see the decision log).
    gamma : float, optional
        Fix the slope instead of fitting it. ``gamma=4`` is the **Plummer**
        profile, so ``eff_unbinned(..., gamma=4.0)`` fits Plummer. When fixed,
        ``gamma`` is still recorded as a PyMC ``Deterministic``, so it appears
        in the trace and in the results below with zero spread -- a
        ``gamma_std`` of 0.0 means "fixed", not "converged perfectly".
    priors : EFFPriors, optional
        Scale-free priors. The defaults are constants, not functions of `radii`.
    sampling : SamplingConfig, optional
        Sampler settings. Defaults to 2000 draws, 1000 tuning.
    progressbar : bool, default False
        Only used when `sampling` is not given, to build the default config.
    return_trace : bool, default True
        Keep the full posterior in the result. Defaults to True for the same
        reason as :func:`king_unbinned`: collapsing to scalars on exit is what
        stops uncertainty reaching the derived quantities.

    Returns
    -------
    dict
        ``field_radius`` -- the value actually used (arcmin Quantity);
        ``n_stars`` -- radii surviving the finite/positive filter;
        ``model`` -- ``"plummer"`` when ``gamma == 4.0`` was passed, else
        ``"eff"``;
        ``completeness_corrected`` -- whether an :math:`\bar{S}(r)` was applied;
        ``gamma_fixed`` -- the `gamma` argument verbatim, ``None`` when fitted;
        ``k_median``/``k_std`` and ``b_median``/``b_std`` -- amplitude and
        background, plain floats in surface-density units;
        ``a_median``/``a_std`` -- scale radius as an **arcmin Quantity**;
        ``gamma_median``/``gamma_std`` -- slope, dimensionless;
        ``eff_trace`` -- the full :class:`arviz.InferenceData`, present only
        when `return_trace`.

        The point estimates are posterior **medians** with **standard
        deviations**, not means with credible intervals; for an asymmetric
        posterior the two differ and the trace is the thing to quote from.

    Raises
    ------
    ImportError
        If PyMC is not installed (the ``bayes`` extra).
    ValueError
        If fewer than ten usable radii remain, or if any radius exceeds
        `field_radius` -- the normalisation integral assumes the sample is
        complete inside that disc, so a star outside it would bias the fit.
    """
    from .inference import SamplingConfig, _sample

    try:
        import pymc as pm
    except ImportError as exc:
        raise ImportError("PyMC is required for eff_unbinned. Install the 'bayes' extra.") from exc

    r = np.asarray(quantity_values(radii, u.arcmin), dtype=float)
    r = r[np.isfinite(r) & (r > 0)]
    field_radius = float(quantity_values(field_radius, u.arcmin))
    if r.size < 10:
        raise ValueError("At least ten stars are required for an unbinned EFF fit.")
    if r.max() > field_radius:
        raise ValueError(
            f"{int((r > field_radius).sum())} stars lie outside field_radius="
            f"{field_radius:g} arcmin; the normalisation assumes completeness inside that disc."
        )

    priors = priors or EFFPriors()
    sampling = sampling or SamplingConfig(draws=2_000, tune=1_000, progressbar=progressbar)

    model = _eff_model(pm, r, field_radius, priors, gamma, completeness)

    trace = _sample(pm, model, sampling)
    results = {
        "field_radius": field_radius * u.arcmin,
        "n_stars": int(r.size),
        "model": "plummer" if gamma == 4.0 else "eff",
        "completeness_corrected": completeness is not None,
        "gamma_fixed": gamma,
    }
    for name, unit in (("k", None), ("b", None), ("a", u.arcmin), ("gamma", None)):
        arr = np.asarray(trace.posterior[name].values, dtype=float)
        median, std = float(np.nanmedian(arr)), float(np.nanstd(arr))
        results[f"{name}_median"] = median * unit if unit else median
        results[f"{name}_std"] = std * unit if unit else std
    if return_trace:
        results["eff_trace"] = trace
    return results


@dataclass(frozen=True)
class CoronaPriors:
    r"""Scale-free priors for the King-plus-corona model.

    Same discipline as :class:`KingPriors`: fixed constants, independent of the data.

    WHERE THE NUMBERS COME FROM — AND WHERE THEY DO NOT
    ---------------------------------------------------
    ``r_c_scale``, ``r_t_scale`` and ``k_scale`` follow :class:`KingPriors`, set from Hunt & Reffert
    (2024). The two corona parameters have **no catalogue to be set from**, and that is stated rather
    than hidden:

    * ``r_2_scale`` (default 60′) is anchored on HR24's ``r_tot`` — median **25.4′**, 84th percentile
      54.8′, 99th 362′ — which is the closest published proxy for a corona extent, though `r_tot` is
      defined by *contrast against the field* (Hunt & Reffert 2023) rather than by a physical edge, so
      it measures the footprint as much as the object. The scale is set generously above the median
      because the quantity being inferred is expected to exceed it.
    * ``delta_scale`` (default 0.01, stars per cubic arcmin) is **not derived**. It is chosen so the
      implied corona count over a typical field is order-100, i.e. the same order as the flat
      background it replaces. There is no independent measurement of open-cluster corona space
      densities to anchor it.

    .. important::
       **This model is deliberately near-degenerate with a flat background, and the prior does not
       resolve that — the data must.** As :math:`R_2 \to \infty`, :math:`2\delta_f\sqrt{R_2^2-r^2}
       \to 2\delta_f R_2`, a constant. On NGC 6383 the two models sit at ``2 ln B = -2.27``, i.e.
       indistinguishable, and the fit reports why: ``P(R_2 > 70' field) = 1.000``, ``R_2`` median
       176.6′ = 55 pc with a 95% interval of [96.8, 895]′, unbounded above. **An unconstrained
       ``R_2`` posterior is the result, not a failure** — it says the corona does not fit inside the
       footprint. The model is identifiable when it does: injecting a corona at ``R_2 = 35'`` in a 70′
       field gives ``2 ln B = -16.18`` in its favour (``tests/test_structure.py``).
    """

    r_c_scale: float = 5.0
    r_t_scale: float = 30.0
    k_scale: float = 10.0
    r_2_scale: float = 60.0
    delta_scale: float = 0.01


def corona_surface_density(radius, *, delta_f, R_2):
    r"""Projected surface density of a uniform sphere — the cluster corona.

    .. math:: \Delta\Sigma(r) = 2\,\delta_f\sqrt{R_2^2 - r^2}\quad (r < R_2),\ 0\ \text{beyond}

    From Danilov & Putkov (2012, Astron. Rep. 56, 609) as used by Seleznev (2016,
    MNRAS 456, 3757), who fits a King core **plus** this corona and states the
    reason it is needed:

        *"the King model does not have an extended corona, and the cluster corona
        … is perceived by the approximation algorithm as part of the stellar
        background."*

    That is the mechanism this repository measured independently on NGC 6383,
    where the flat background term accounts for **56% of a membership-selected
    sample** against a target–decoy false-discovery proportion whose median is 2.8%. See
    ``docs/design-notes/king_model_validity.md``.

    .. important::
       Note the near-degeneracy this model exists to expose: for
       :math:`R_2 \gg r`, :math:`\Delta\Sigma \to 2\delta_f R_2`, a **constant**.
       So a corona much larger than the field is indistinguishable from a flat
       background, and an unconstrained posterior on ``R_2`` is itself the
       answer — it says the corona does not fit inside the footprint.

    Parameters
    ----------
    delta_f : float or tensor
        Space density of the corona, in stars per unit volume.
    R_2 : float or Quantity
        Corona radius, in arcmin when a plain number; a Quantity is
        **converted**, on the same convention as `radius`. (Before 2026-08-04
        this argument was not unit-aware and a Quantity raised
        :class:`~astropy.units.UnitConversionError` inside
        ``np.maximum(R_2**2 - r**2, 0.0)``, because `radius` had already been
        stripped to a bare array -- the identical defect
        :func:`eff_surface_density` carried on ``a``.)
    """
    # Explicit unit: quantity_values(x) with no target strips whatever unit x carries.
    # Passing degrees where arcmin is meant returns silently wrong values -- measured, a
    # 480x error at r = 20. Plain arrays pass through unchanged and are taken as arcmin.
    r = quantity_values(radius, u.arcmin)
    # Same two-line convention as king_profile's rc/rt.
    R_2 = R_2.to(u.arcmin).value if hasattr(R_2, "to") else float(R_2)
    # The clamp at zero *is* the truncation: beyond R_2 the radicand is negative,
    # so clipping it sends the density to zero. An explicit `np.where(r < R_2, ...)`
    # on top is dead code that reads as if it were doing the work.
    return 2.0 * delta_f * np.sqrt(np.maximum(R_2**2 - r**2, 0.0))


def corona_expected_count(delta_f, R_2, field_radius, *, xp=np):
    r"""Expected corona count inside a circular field, in closed form.

    .. math:: \Lambda = \frac{4\pi\delta_f}{3}
              \left[R_2^3 - \left(R_2^2 - \min(R_f, R_2)^2\right)^{3/2}\right]

    When the field encloses the corona this reduces to
    :math:`\frac{4}{3}\pi R_2^3 \delta_f`, i.e. volume times density, which is the
    check that the algebra is right.

    Verified against :func:`scipy.integrate.quad` to a relative error below
    ``1e-10`` across regimes including ``R_2 > field_radius`` and
    ``R_2 == field_radius``.
    """
    upper = xp.minimum(field_radius, R_2)
    remainder = xp.maximum(R_2**2 - upper**2, 0.0)
    return (4.0 * np.pi * delta_f / 3.0) * (R_2**3 - remainder**1.5)


def _king_corona_model(pm, r, field_radius, priors, tidal_prior, completeness):
    """King core plus a uniform-sphere corona, replacing the flat background."""
    if completeness is not None:
        raise NotImplementedError(
            "completeness weighting is not implemented for the corona model; the "
            "weighted normalisation would need its own quadrature."
        )
    with pm.Model() as model:
        R_c = pm.HalfStudentT("R_c", nu=1, sigma=priors.r_c_scale)
        if tidal_prior is None:
            R_t = pm.Deterministic("R_t", R_c + pm.HalfStudentT("dR", nu=1, sigma=priors.r_t_scale))
        else:
            mu, sigma = tidal_prior
            R_t = pm.Deterministic(
                "R_t", R_c + pm.TruncatedNormal("dR", mu=mu, sigma=sigma, lower=0.0)
            )
        k = pm.HalfStudentT("k", nu=1, sigma=priors.k_scale)
        R_2 = pm.HalfStudentT("R_2", nu=1, sigma=priors.r_2_scale)
        delta_f = pm.HalfStudentT("delta_f", nu=1, sigma=priors.delta_scale)

        core = 1.0 / pm.math.sqrt(1.0 + (r / R_c) ** 2)
        edge = 1.0 / pm.math.sqrt(1.0 + (R_t / R_c) ** 2)
        cluster = pm.math.switch(r <= R_t, k * (core - edge) ** 2, 0.0)
        corona = pm.math.switch(
            r < R_2, 2.0 * delta_f * pm.math.sqrt(pm.math.maximum(R_2**2 - r**2, 0.0)), 0.0
        )
        surface = cluster + corona
        log_intensity = pm.math.log(2.0 * np.pi * r * surface)
        expected = king_expected_count(
            k, 0.0, R_c, R_t, field_radius, xp=pm.math
        ) + corona_expected_count(delta_f, R_2, field_radius, xp=pm.math)
        pm.Potential("point_process", pm.math.sum(log_intensity) - expected)
    return model


def compare_radial_profiles(
    radii,
    *,
    field_radius,
    models=("king", "eff", "plummer"),
    king_priors: KingPriors | None = None,
    eff_priors: EFFPriors | None = None,
    corona_priors: CoronaPriors | None = None,
    tidal_prior: tuple[float, float] | None = None,
    completeness=None,
    draws: int = 2000,
    chains: int = 4,
    random_seed: int | None = None,
):
    r"""Compare radial-profile families by Bayes factor.

    Each model is fit with **sequential Monte Carlo**, which returns a log
    marginal likelihood directly; the Bayes factor is the difference. This is the
    approach Olivares et al. (2018, `2018A&A...612A..70O`) use to compare
    King/GKing/EFF/GDP for the Pleiades, and it is the right tool here because
    the models are **not nested** — EFF has no ``R_t`` at all, so a likelihood-ratio
    test does not apply.

    Marginal likelihood is used rather than LOO/WAIC because the point-process
    likelihood is a single :class:`~pymc.Potential` over the whole field, not a
    sum of exchangeable per-star terms: :math:`\Lambda` couples every star, so
    there is no clean pointwise decomposition to leave one out of.

    Parameters
    ----------
    models : sequence of str
        Any of ``"king"``, ``"eff"``, ``"plummer"`` (EFF with :math:`\gamma` fixed
        to 4).
    draws, chains, random_seed
        SMC settings. SMC needs more draws than NUTS for a stable evidence.

    Returns
    -------
    dict
        ``log_marginal_likelihood`` per model, ``best``, ``log_bayes_factor``
        relative to the best model, and ``interpretation`` on the Kass & Raftery
        (1995) scale applied to :math:`2\ln B`.

    Notes
    -----
    Evidence is prior-sensitive by construction — that is what "marginal" means.
    Comparing models whose priors are on different footings is meaningless, so
    both families here use the same scale-free half-Cauchy form and the same
    background prior. Report the priors alongside any Bayes factor.
    """
    from scipy.special import logsumexp

    try:
        import pymc as pm
    except ImportError as exc:
        raise ImportError(
            "PyMC is required for compare_radial_profiles. Install the 'bayes' extra."
        ) from exc

    r = np.asarray(quantity_values(radii, u.arcmin), dtype=float)
    r = r[np.isfinite(r) & (r > 0)]
    field_radius = float(quantity_values(field_radius, u.arcmin))
    if r.max() > field_radius:
        raise ValueError("Stars lie outside field_radius; the normalisation assumes completeness.")
    king_priors = king_priors or KingPriors()
    eff_priors = eff_priors or EFFPriors()

    builders = {
        "king": lambda: _king_model(pm, r, field_radius, king_priors, tidal_prior, completeness),
        "eff": lambda: _eff_model(pm, r, field_radius, eff_priors, None, completeness),
        "plummer": lambda: _eff_model(pm, r, field_radius, eff_priors, 4.0, completeness),
        "king_corona": lambda: _king_corona_model(
            pm, r, field_radius, corona_priors or CoronaPriors(), tidal_prior, completeness
        ),
    }
    unknown = set(models) - set(builders)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}; choose from {sorted(builders)}")

    log_ml: dict[str, float] = {}
    log_ml_sd: dict[str, float] = {}
    for name in models:
        with builders[name]():
            idata = pm.sample_smc(
                draws=draws, chains=chains, random_seed=random_seed, progressbar=False
            )
        # sample_stats["log_marginal_likelihood"] is (chain, stage), NaN-padded at
        # the front because chains need different numbers of stages. The estimate
        # is the last finite entry of each chain. Taking `.ravel()[-1]` looks
        # right and is not: with ragged padding it can land on a NaN, and it
        # silently discards every chain but one.
        raw = np.asarray(idata.sample_stats["log_marginal_likelihood"])
        # PyMC 6.1.0 returns this in two different layouts depending on whether
        # the chains agreed on a stage count: either a (chain, stage) array
        # NaN-padded at the front, or a 1-D object array of per-chain lists.
        rows = raw if (raw.dtype == object and raw.ndim == 1) else np.atleast_2d(raw)
        finals = []
        for row in rows:
            vals = np.asarray(row, dtype=float).ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size:
                finals.append(vals[-1])
        per_chain = np.asarray(finals, dtype=float)
        if per_chain.size == 0:
            raise RuntimeError(f"SMC returned no finite marginal likelihood for model {name!r}")
        # Average the evidence, not its logarithm.
        log_ml[name] = float(logsumexp(per_chain) - np.log(per_chain.size))
        log_ml_sd[name] = float(np.std(per_chain))

    best = max(log_ml, key=log_ml.get)
    log_bf = {m: log_ml[m] - log_ml[best] for m in log_ml}

    def _verdict(two_log_b: float) -> str:
        a = abs(two_log_b)
        if a < 2:
            return "not worth more than a bare mention"
        if a < 6:
            return "positive"
        if a < 10:
            return "strong"
        return "very strong"

    # Chain-to-chain scatter in log Z. A Bayes factor smaller than this is noise,
    # not evidence -- SMC evidence estimates are far less stable than posteriors.
    noise = max(log_ml_sd.values())
    return {
        "n_stars": int(r.size),
        "field_radius": field_radius * u.arcmin,
        "log_marginal_likelihood": log_ml,
        "log_marginal_likelihood_chain_sd": log_ml_sd,
        "resolvable": {m: abs(log_bf[m]) > 2.0 * noise for m in log_bf},
        "best": best,
        "log_bayes_factor_vs_best": log_bf,
        "interpretation": {m: _verdict(2.0 * log_bf[m]) for m in log_bf},
        "scale": "Kass & Raftery (1995), applied to 2 ln B",
    }


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
    r"""Fit a King radial-density profile with PyMC. **Superseded — see the warning.**

    .. deprecated:: 2026-08-02
       Kept reachable **only** to reproduce results published before 2026-08-02.
       New work must use :func:`king_unbinned`. Calling this emits a
       :class:`UserWarning`.

    .. warning::
       **This likelihood is mis-specified by a measured factor of ~25, and its
       priors are derived from the data being fitted.**

       *The likelihood.* It puts a Normal on **binned** surface densities with a
       single shared scatter ``sigma``. The package's default binner
       (``method="equip"``) uses **equal-count** annuli, so the count per bin is
       fixed by construction and the *area* is what varies. For
       :math:`n \sim \mathrm{Poisson}(\Lambda)` split into ``n_bins``
       equal-count annuli, :math:`\mathrm{Var}(N_i) = \Lambda/n_\mathrm{bins}^2`
       while :math:`E(N_i) = \Lambda/n_\mathrm{bins}`, so the Poisson dispersion
       index is :math:`1/n_\mathrm{bins}`. Measured over 400 realizations of a
       known King point process it is **0.045** at the 25 bins the published fit
       used, against the 1.0 a count likelihood asserts -- a **~25x**
       mis-specification. Oracle: ``tools/validation/king_binning_likelihood.py``.

       *The priors.* All four bounds come from ``nanstd`` / ``nanmin`` /
       ``nanmax`` **of the observed densities and radii** -- the data used twice,
       which the P01 referee raised in those words ("*appear arbitrary*"). In
       particular ``R_t ~ Uniform(R_c, 1.5 * max(d_center))`` **truncates** the
       ``R_t`` posterior rather than letting the data turn it over.

    :func:`king_unbinned` replaces both defects at once with an inhomogeneous
    Poisson point process, :math:`\log L = \sum_i \log \lambda(r_i) - \Lambda`
    (the continuous Cash 1979 statistic, `1979ApJ...228..939C`), and the
    scale-free :class:`KingPriors`. There is no binned-Poisson middle ground on
    offer and none ever shipped here: a per-annulus Poisson likelihood on
    equal-count bins would assert the same false ``Var = E`` and is the *other*
    way to get this wrong.

    Notes
    -----
    See ``docs/design-notes/decisions.md`` (2026-07-27, "the King fit is now
    unbinned") for the full measurement, and
    ``tools/validation/king_unbinned_delta.py`` for what the change does to the
    published NGC 6383 numbers.
    """
    from .inference import SamplingConfig, _sample

    warnings.warn(
        "RDP_bayesian fits a Normal likelihood to *binned* densities with "
        "data-derived prior bounds. On equal-count annuli its Poisson dispersion "
        "index is ~0.045 against the 1.0 asserted (a ~25x mis-specification, "
        "measured in tools/validation/king_binning_likelihood.py), and every prior "
        "bound is a function of the data being fitted. Use "
        "erotica.analysis.structure.king_unbinned instead; this path is retained "
        "only to reproduce results published before 2026-08-02.",
        UserWarning,
        stacklevel=2,
    )

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
        R_c = pm.Uniform(
            "R_c", lower=1e-6, upper=max(0.8 * float(np.nanmax(d_center_values)), 1e-5)
        )
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
        # ⚠ DECLARADO, NO ARREGLADO — 2026-08-26.
        #
        # Esto devuelve la entrada: `priors_parameters` entra y sale, y `priors_used` sólo dice si
        # se pasó algo. Los priors de King de esta función son Uniforms hardcodeados y estos tres
        # argumentos NUNCA entran en el modelo PyMC. Era un release-blocker.
        #
        # No se arregla, y la razón es que arreglarlo sería trabajo sobre una ruta muerta: esta
        # función está **deprecada desde 2026-08-02** por una verosimilitud mal especificada por un
        # factor ~25, y `king_unbinned` la reemplaza — con `KingPriors` libre de datos y
        # `tidal_prior=(mu, sigma)`, que **sí** entra en el modelo (`_king_model` construye
        # `R_t = R_c + TruncatedNormal(mu, sigma, lower=0)` cuando se le da). O sea la capacidad que
        # estos argumentos prometían y no cumplían **existe y funciona en la ruta nueva**.
        #
        # Se conserva la firma por compatibilidad con el código que reprodujo P01. Quien la llame ya
        # recibe el `UserWarning` de deprecación, que es el sitio donde este aviso importa.
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
    """Log-space variant of :func:`RDP_bayesian` for sparse radial profiles.

    .. deprecated:: 2026-08-02
       Carries **the same two defects** as :func:`RDP_bayesian` -- a Gaussian
       likelihood on binned densities, and prior bounds derived from the data
       being fitted -- and taking the logarithm changes neither. Working in log
       space also makes the Gaussian assumption *less* defensible in the sparse
       outer bins, where the density is small and its log is strongly skewed.
       Use :func:`king_unbinned`. Calling this emits a :class:`UserWarning`.
    """
    from .inference import SamplingConfig, _sample

    warnings.warn(
        "RDP_bayesian_log_space has the same defects as RDP_bayesian -- a Gaussian "
        "likelihood on binned densities and data-derived prior bounds -- which "
        "taking a logarithm does not fix. Use "
        "erotica.analysis.structure.king_unbinned instead; this path is retained "
        "only to reproduce results published before 2026-08-02.",
        UserWarning,
        stacklevel=2,
    )

    try:
        import pymc as pm
    except ImportError as exc:
        raise ImportError("PyMC is required for RDP_bayesian_log_space.") from exc

    density_values = quantity_values(density_annulus)
    radius_values = quantity_values(radius_gen, u.arcmin)
    d_center_values = quantity_values(d_center, u.arcmin) if d_center is not None else radius_values
    finite = (
        np.isfinite(density_values)
        & (density_values > 0)
        & np.isfinite(radius_values)
        & (radius_values > 0)
    )
    density_values = density_values[finite]
    radius_values = radius_values[finite]
    if len(density_values) < 3:
        raise ValueError("At least three positive density bins are required.")
    sampling = sampling or SamplingConfig(draws=2_000, tune=1_000, progressbar=progressbar)
    with pm.Model() as model:
        log_sigma = pm.Normal(
            "log_sigma", mu=np.log(max(np.nanstd(np.log(density_values)), 1e-3)), sigma=1
        )
        sigma = pm.Deterministic("sigma", pm.math.exp(log_sigma))
        log_b = pm.Uniform(
            "log_b", lower=np.log(1e-8), upper=np.log(max(0.5 * np.nanmax(density_values), 1e-7))
        )
        b = pm.Deterministic("b", pm.math.exp(log_b))
        log_k = pm.Uniform(
            "log_k", lower=log_b, upper=np.log(max(2 * np.nanmax(density_values), 1e-6))
        )
        k = pm.Deterministic("k", pm.math.exp(log_k))
        log_R_c = pm.Uniform(
            "log_R_c", lower=np.log(1e-4), upper=np.log(max(0.8 * np.nanmax(d_center_values), 1e-3))
        )
        R_c = pm.Deterministic("R_c", pm.math.exp(log_R_c))
        log_R_t = pm.Uniform(
            "log_R_t", lower=log_R_c, upper=np.log(max(2 * np.nanmax(d_center_values), 1e-3))
        )
        R_t = pm.Deterministic("R_t", pm.math.exp(log_R_t))
        r = np.asarray(radius_values, dtype=float)
        king = pm.math.switch(
            r <= R_t,
            k
            * ((1 / pm.math.sqrt(1 + (r / R_c) ** 2)) - (1 / pm.math.sqrt(1 + (R_t / R_c) ** 2)))
            ** 2
            + b,
            b,
        )
        pm.Normal(
            "obs_log_density", mu=pm.math.log(king), sigma=sigma, observed=np.log(density_values)
        )
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
        """Bind a source table and its column names.

        Parameters
        ----------
        data : QTable
            Cluster source table. Held by reference and never copied, so later
            edits to the caller's table are seen by every method here.
        probability_column : str, default "probability"
            Membership-probability column consulted by :meth:`select` and by
            every method taking a `probability_threshold`. Nothing is validated
            at construction: a wrong name raises only when a threshold is
            actually applied, since ``probability_threshold=None`` never touches
            the column.
        ra_column, dec_column : str, default "ra", "dec"
            Sky-position columns, read as **degrees** by
            :func:`center_determination` and the radial-profile helpers.

        Notes
        -----
        Constructing this object runs no numerics. Every method delegates to the
        module-level function of the same name, adding only the probability
        selection and the column names, so results are identical to calling
        those functions directly.
        """
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
        method: str = "unbinned",
        field_radius=None,
        log_space: bool = False,
        **kwargs,
    ):
        r"""Fit a King profile to a :class:`RadialDensityProfile`.

        Parameters
        ----------
        profile : RadialDensityProfile
            Used for its ``distances`` attribute -- the **per-star** angular
            separations from the centre -- when ``method="unbinned"``, and for
            its binned ``density`` / ``radius`` when ``method="binned"``.
        method : {"unbinned", "binned"}, default "unbinned"
            ``"unbinned"`` fits :func:`king_unbinned`: an inhomogeneous Poisson
            point process on the individual stellar radii,
            :math:`\log L = \sum_i \log \lambda(r_i) - \Lambda`, with the
            scale-free :class:`KingPriors`.

            ``"binned"`` routes to :func:`RDP_bayesian` (or
            :func:`RDP_bayesian_log_space` when `log_space`), which is retained
            only for reproducing results published before 2026-08-02 and warns
            when called.
        field_radius : float or Quantity, **required for** ``method="unbinned"``
            Radius of the circular selection footprint, in arcmin. There is no
            default, deliberately: the point-process normalisation
            :math:`\Lambda = \int_0^{R_f} 2\pi r\,\Sigma(r)\,dr` assumes the
            sample is **complete inside that disc**, and inferring it from
            ``max(profile.distances)`` would let a footprint assumption be made
            silently by the code instead of stated by the caller. For a
            membership-selected list the footprint is whatever the selection
            carved out and is *not* a disc -- see
            ``docs/design-notes/king_model_validity.md``.
        log_space : bool, default False
            Only meaningful for ``method="binned"``.

        Notes
        -----
        **What changed on 2026-08-02.** The default was ``RDP_bayesian``: a
        Normal likelihood on equal-count binned densities whose Poisson
        dispersion index is **0.045** against the 1.0 such a likelihood asserts
        (measured over 400 realizations in
        ``tools/validation/king_binning_likelihood.py``, a ~25x
        mis-specification), with all four prior bounds taken from ``nanstd`` /
        ``nanmin`` / ``nanmax`` of the data being fitted. :func:`king_unbinned`
        existed, was correct, and was simply not what this method called.

        Passing ``method="binned"`` restores the old behaviour exactly.
        """
        if method == "binned":
            fitter = RDP_bayesian_log_space if log_space else RDP_bayesian
            return fitter(
                profile.density,
                profile.radius,
                d_center=profile.distances,
                **kwargs,
            )
        if method != "unbinned":
            raise ValueError("method must be 'unbinned' or 'binned'.")
        if log_space:
            raise ValueError(
                "log_space applies only to method='binned'. The unbinned point "
                "process has no binned densities to take a logarithm of."
            )
        if field_radius is None:
            raise ValueError(
                "field_radius is required for method='unbinned': the normalisation "
                "integral assumes the sample is complete inside that disc, so the "
                "footprint must be stated rather than inferred from the data. Pass "
                "the selection radius, or method='binned' for the legacy fit."
            )
        return king_unbinned(profile.distances, field_radius=field_radius, **kwargs)


__all__ = [
    "CoronaPriors",
    "corona_surface_density",
    "corona_expected_count",
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
