"""Dynamical and Galactic-structure helpers."""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.constants import G
from astropy.coordinates import Galactic, SkyCoord

from .units import angular_size, ensure_units, estimate_cluster_mass, quantity_values

#: Galactocentric distance of the Sun, R_0.
#:
#: **Single source of truth.** Before 2026-07-27 this module carried two different
#: values — ``8.125 kpc`` in :func:`calculate_galactic_mass` and ``8.3 kpc`` in
#: :func:`calculate_galactocentric_distance` and :func:`calculate_hill_radius` — even
#: though all three feed the same Hill-radius chain. They are now unified here.
#:
#: The VALUE is unchanged by that de-duplication (8.125 kpc, as in
#: ``calculate_galactic_mass``), so this is not a re-calibration.
#:
#: NOTE: 8.125 kpc does not correspond to a verified published measurement; its
#: provenance in this codebase is unknown. The best current geometric determination is
#: R_0 = 8178 +/- 13(stat) +/- 22(sys) pc (GRAVITY Collaboration 2019, A&A 625, L10,
#: bibcode 2019A&A...625L..10G, doi:10.1051/0004-6361/201935656). Adopting 8.178 kpc is a
#: science decision, not a bug fix -- it shifts every galactocentric and Hill-radius
#: number -- and is deferred to an explicit call. See docs/design-notes/decisions.md.
#: Override per call with the ``solar_radius`` keyword.
SOLAR_RADIUS: u.Quantity = 8.125 * u.kpc


def calculate_galactic_mass(
    radius,
    radius_err=None,
    *,
    solar_radius: u.Quantity = SOLAR_RADIUS,
    model: str = "legacy_power_law",
):
    """Approximate enclosed Galactic mass inside ``radius``.

    ``model='legacy_power_law'`` preserves the thesis/paper helper
    ``2e8 * (R/30 pc)**1.2``. ``model='solar_scaled'`` keeps the earlier package
    approximation ``1e11 * (R/R_sun)**2``. If ``radius_err`` is supplied, the
    function returns ``(mass, mass_err)``.
    """
    radius = ensure_units(radius, u.kpc)
    if model == "legacy_power_law":
        radius_pc = radius.to(u.pc)
        mass = 2e8 * (radius_pc / (30 * u.pc)) ** 1.2 * u.Msun
        if radius_err is None:
            return mass.to(u.Msun)
        radius_err = ensure_units(radius_err, u.pc)
        derivative = 1.2 * 2e8 * (radius_pc / (30 * u.pc)) ** 0.2 / (30 * u.pc) * u.Msun
        return mass.to(u.Msun), (derivative * radius_err).to(u.Msun)
    if model == "solar_scaled":
        solar_radius = ensure_units(solar_radius, u.kpc)
        mass = 1e11 * (radius / solar_radius) ** 2 * u.Msun
        if radius_err is None:
            return mass.to(u.Msun)
        radius_err = ensure_units(radius_err, u.kpc)
        derivative = 2e11 * radius / solar_radius**2 * u.Msun
        return mass.to(u.Msun), (derivative * radius_err).to(u.Msun)
    raise ValueError("model must be 'legacy_power_law' or 'solar_scaled'.")


def calculate_galactocentric_distance(
    first=None,
    second=None,
    third=None,
    *legacy_pos,
    distance=None,
    ra=None,
    dec=None,
    l=None,
    b=None,
    ra_err=0 * u.deg,
    dec_err=0 * u.deg,
    distance_err=0 * u.kpc,
    solar_radius=SOLAR_RADIUS,
):
    """Calculate Galactocentric radius.

    Supports both package-style Galactic inputs ``(distance, l, b)`` and the
    legacy notebook call ``(ra, dec, distance=...)``. The equatorial form returns
    ``(radius, radius_err)``; the Galactic-coordinate form returns only radius.
    """
    solar_radius = ensure_units(solar_radius, u.kpc)
    if legacy_pos:
        if len(legacy_pos) > 3:
            raise TypeError("Expected at most ra_err, dec_err, and distance_err.")
        distance = third
        if len(legacy_pos) >= 1:
            ra_err = legacy_pos[0]
        if len(legacy_pos) >= 2:
            dec_err = legacy_pos[1]
        if len(legacy_pos) >= 3:
            distance_err = legacy_pos[2]

    equatorial_positional = False
    if distance is None and ra is None and dec is None and third is not None:
        try:
            ensure_units(third, u.kpc)
        except Exception:
            equatorial_positional = False
        else:
            equatorial_positional = True

    # Five call forms have to be told apart, and the Galactic branch further down
    # is only reachable if we decline the equatorial one here:
    #
    #   1. (dist, l, b)                positional Galactic
    #   2. (ra, dec, dist)             legacy positional equatorial
    #   3. (ra=, dec=, distance=)      keyword equatorial
    #   4. (ra, dec, distance=)        positional pair + keyword distance
    #                                  -- used by calculate_hill_radius(center=...)
    #   5. (distance=, l=, b=)         keyword Galactic
    #
    # Before 2026-07-27 the guard fired on ``distance is not None`` alone, so form
    # 5 -- the keyword Galactic call the docstring advertises -- fell into the
    # equatorial path with ``ra=None`` and raised inside SkyCoord. Only form 1
    # reached the Galactic branch.
    galactic_kw = l is not None and b is not None
    galactic_pos = (
        distance is None
        and ra is None
        and dec is None
        and third is not None
        and not equatorial_positional
    )
    if not (galactic_kw or galactic_pos):
        ra = first if ra is None else ra
        dec = second if dec is None else dec
        distance = third if distance is None else distance
        dist = ensure_units(distance, u.kpc)
        dist_err = ensure_units(distance_err, u.kpc)
        coord = SkyCoord(ra=ra, dec=dec, distance=dist, frame="icrs").transform_to(Galactic)
        l_rad = coord.l.radian
        b_rad = coord.b.radian
        radius = np.sqrt(
            solar_radius**2 + dist**2 - 2 * solar_radius * dist * np.cos(l_rad) * np.cos(b_rad)
        ).to(u.kpc)
        d_radius_ddist = (dist - solar_radius * np.cos(l_rad) * np.cos(b_rad)) / radius
        d_radius_dra = (solar_radius * dist * np.sin(l_rad) * np.cos(b_rad)) / radius
        d_radius_ddec = (solar_radius * dist * np.cos(l_rad) * np.sin(b_rad)) / radius
        ra_err_rad = ensure_units(ra_err, u.deg).to(u.rad)
        dec_err_rad = ensure_units(dec_err, u.deg).to(u.rad)
        radius_err = np.sqrt(
            (d_radius_ddist * dist_err) ** 2
            + ((d_radius_dra * ra_err_rad) ** 2).to(
                u.kpc**2, equivalencies=u.dimensionless_angles()
            )
            + ((d_radius_ddec * dec_err_rad) ** 2).to(
                u.kpc**2, equivalencies=u.dimensionless_angles()
            )
        )
        return radius, radius_err.to(u.kpc)

    cluster_distance = first if distance is None else distance
    l = second if l is None else l
    b = third if b is None else b
    dist = ensure_units(cluster_distance, u.kpc)
    coords = SkyCoord(l=l, b=b, distance=dist, frame=Galactic)
    l_rad = coords.l.radian
    b_rad = coords.b.radian
    return np.sqrt(
        solar_radius**2 + dist**2 - 2 * solar_radius * dist * np.cos(l_rad) * np.cos(b_rad)
    ).to(u.kpc)


def posterior_summary(samples, *, credible_mass: float = 0.68):
    """Median and equal-tailed credible interval of a derived quantity.

    The point of propagating a posterior into `dynamics` is that the answer is a
    *distribution*. This reduces one to the three numbers a table needs, without
    pretending the distribution was Gaussian.

    Parameters
    ----------
    samples : array-like or Quantity
        Draws of the derived quantity, e.g. the output of a dynamics function
        called on posterior samples. Units are preserved.
    credible_mass : float, default 0.68
        Central probability mass of the interval.

    Returns
    -------
    dict
        ``median``, ``lower``, ``upper`` (the interval *bounds*, not offsets),
        ``minus``, ``plus`` (offsets from the median), and ``n``.

    Examples
    --------
    >>> import numpy as np
    >>> s = posterior_summary(np.arange(1001.0))
    >>> round(s["median"]), round(s["lower"]), round(s["upper"])
    (500, 160, 840)
    """
    values = np.asarray(quantity_values(samples), dtype=float)
    unit = getattr(samples, "unit", None)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite samples to summarize.")
    tail = 0.5 * (1.0 - float(credible_mass))
    lo, med, hi = np.percentile(finite, [100 * tail, 50.0, 100 * (1.0 - tail)])
    out = {
        "median": med, "lower": lo, "upper": hi,
        "minus": med - lo, "plus": hi - med, "n": int(finite.size),
    }
    if unit is not None:
        out = {k: (v * unit if k != "n" else v) for k, v in out.items()}
    return out


def calculate_hill_radius(
    data=None,
    cluster_distance=None,
    l=None,
    b=None,
    *,
    distance=None,
    distance_err=None,
    center=None,
    galactocentric_distance=None,
    galactocentric_distance_err=None,
    galactic_mass=None,
    galactic_mass_err=None,
    cluster_mass=None,
    cluster_mass_err=None,
    mass_column: str | None = None,
    magnitude_column: str = "Gmag",
    return_galdist=False,
    return_cluster_mass=False,
    return_linear_size=False,
    return_galactic_mass=False,
    solar_radius=SOLAR_RADIUS,
):
    """Estimate Hill radius and optional propagated uncertainty."""
    dist = ensure_units(distance if distance is not None else cluster_distance, u.kpc)
    dist_err = ensure_units(0 * u.kpc if distance_err is None else distance_err, u.kpc)
    if galactocentric_distance is None:
        if center is not None:
            galactocentric_distance, galactocentric_distance_err = calculate_galactocentric_distance(
                center[0], center[1], distance=dist, distance_err=dist_err,
                solar_radius=solar_radius,
            )
        else:
            galactocentric_distance = calculate_galactocentric_distance(
                dist, l, b, solar_radius=solar_radius
            )
            galactocentric_distance_err = 0 * u.kpc
    galactocentric_distance = ensure_units(galactocentric_distance, u.kpc)
    galactocentric_distance_err = ensure_units(
        0 * u.kpc if galactocentric_distance_err is None else galactocentric_distance_err,
        u.kpc,
    )
    if galactic_mass is None:
        # calculate_galactic_mass returns a bare Quantity when no error is given
        # and a (mass, err) tuple when one is. Unpacking unconditionally -- as both
        # call sites did before 2026-07-27 -- breaks the DEFAULT path, where
        # galactocentric_distance_err is None. NOTE: an audit established that only
        # tidal_radius_prior's copy was actually reachable; calculate_hill_radius
        # always supplies a non-None error before reaching it, so its guard here is
        # defensive rather than a fix.
        computed = calculate_galactic_mass(
            galactocentric_distance, galactocentric_distance_err
        )
        if isinstance(computed, tuple):
            galactic_mass, galactic_mass_err = computed
        else:
            galactic_mass, galactic_mass_err = computed, None
    else:
        galactic_mass = ensure_units(galactic_mass, u.Msun)
        galactic_mass_err = ensure_units(0 * u.Msun if galactic_mass_err is None else galactic_mass_err, u.Msun)
    if cluster_mass is None:
        if mass_column and data is not None and mass_column in data.colnames:
            cluster_mass = np.nansum(data[mass_column]).to(u.Msun)
        else:
            cluster_mass = estimate_cluster_mass(data[magnitude_column] if hasattr(data, "colnames") else data, dist)
        cluster_mass_err = 0 * u.Msun
    cluster_mass = ensure_units(cluster_mass, u.Msun)
    cluster_mass_err = ensure_units(0 * u.Msun if cluster_mass_err is None else cluster_mass_err, u.Msun)

    radius = (galactocentric_distance * (cluster_mass / (3 * galactic_mass)) ** (1 / 3)).to(u.pc)
    d_radius_d_galdist = (cluster_mass / (3 * galactic_mass)) ** (1 / 3)
    d_radius_d_cluster_mass = galactocentric_distance / (
        3 ** (4 / 3) * galactic_mass ** (1 / 3) * cluster_mass ** (2 / 3)
    )
    d_radius_d_galactic_mass = -(
        galactocentric_distance * cluster_mass ** (1 / 3)
    ) / (3 ** (4 / 3) * galactic_mass ** (4 / 3))
    radius_err = np.sqrt(
        (d_radius_d_galdist * galactocentric_distance_err) ** 2
        + (d_radius_d_cluster_mass * cluster_mass_err) ** 2
        + (d_radius_d_galactic_mass * galactic_mass_err) ** 2
    ).to(u.pc)
    angular = angular_size(radius, dist).to(u.arcmin)
    angular_err = np.sqrt((radius_err / dist) ** 2 + (radius * dist_err / dist**2) ** 2).to(
        u.arcmin, equivalencies=u.dimensionless_angles()
    )

    results = {"angular_size": angular, "angular_size_err": angular_err}
    if return_linear_size:
        results.update({"linear_size": radius, "linear_size_err": radius_err})
    if return_galdist:
        results.update(
            {
                "galactocentric_distance": galactocentric_distance,
                "galactocentric_distance_err": galactocentric_distance_err,
            }
        )
    if return_cluster_mass:
        results.update({"cluster_mass": cluster_mass, "cluster_mass_err": cluster_mass_err})
    if return_galactic_mass:
        results.update({"galactic_mass": galactic_mass, "galactic_mass_err": galactic_mass_err})
    return results


def grav_bound_radius(
    cluster_mass,
    cluster_mass_err=None,
    *,
    A=15.3 * u.km / u.s / u.kpc,
    A_err=0.4 * u.km / u.s / u.kpc,
    B=-11.9 * u.km / u.s / u.kpc,
    B_err=0.4 * u.km / u.s / u.kpc,
    dispersion=None,
    distance=None,
):
    """Estimate gravitationally bound radius.

    If ``dispersion`` is provided, use ``G M / sigma^2``. Otherwise preserve the
    Oort-constant expression used by the paper notebooks.
    """
    cluster_mass = ensure_units(cluster_mass, u.Msun)
    if dispersion is not None:
        dispersion = ensure_units(dispersion, u.km / u.s)
        radius = (G * cluster_mass / dispersion**2).to(u.pc)
        return radius if distance is None else angular_size(radius, distance).to(u.arcmin)

    cluster_mass_err = ensure_units(0 * u.Msun if cluster_mass_err is None else cluster_mass_err, u.Msun)
    a_b_squared = (A - B) ** 2
    a_b_squared_err = 2 * np.abs(A - B) * np.sqrt(A_err**2 + B_err**2)
    radius = ((G * cluster_mass / (2 * a_b_squared)) ** (1 / 3)).to(u.pc)
    d_radius_dm = (1 / 3) * (radius / cluster_mass)
    d_radius_da_b_squared = (-1 / 3) * (radius / a_b_squared)
    radius_err = np.sqrt(
        (d_radius_dm * cluster_mass_err) ** 2
        + (d_radius_da_b_squared * a_b_squared_err) ** 2
    ).to(u.pc)
    results = {"linear_radius": radius, "linear_radius_err": radius_err}
    if distance is not None:
        results["angular_radius"] = angular_size(radius, distance).to(u.arcmin)
        results["angular_radius_err"] = angular_size(radius_err, distance).to(u.arcmin)
    return results


def tidal_radius_prior(
    cluster_mass,
    galactocentric_distance,
    galactocentric_distance_err=None,
    *,
    galactic_mass=None,
    kind: str = "angular",
    type: str | None = None,
    distance=None,
    distance_err=None,
    return_galactic_mass: bool = False,
    data=None,
):
    """Estimate the tidal-radius prior used by the King-profile workflow."""
    if type is not None:
        kind = type
    if galactic_mass is None:
        # calculate_galactic_mass returns a bare Quantity when no error is given
        # and a (mass, err) tuple when one is. Unpacking unconditionally -- as both
        # call sites did before 2026-07-27 -- breaks the DEFAULT path, where
        # galactocentric_distance_err is None. NOTE: an audit established that only
        # tidal_radius_prior's copy was actually reachable; calculate_hill_radius
        # always supplies a non-None error before reaching it, so its guard here is
        # defensive rather than a fix.
        computed = calculate_galactic_mass(
            galactocentric_distance, galactocentric_distance_err
        )
        if isinstance(computed, tuple):
            galactic_mass, galactic_mass_err = computed
        else:
            galactic_mass, galactic_mass_err = computed, None
    else:
        galactic_mass = ensure_units(galactic_mass, u.Msun)
        galactic_mass_err = None
    if cluster_mass is None:
        cluster_mass = estimate_cluster_mass(data, distance)
    cluster_mass = ensure_units(cluster_mass, u.Msun)
    galactocentric_distance = ensure_units(galactocentric_distance, u.kpc)
    tidal_radius = (cluster_mass / (2 * galactic_mass)) ** (1 / 3) * galactocentric_distance
    results = {}
    if kind in {"linear", "both"}:
        results["linear_size"] = tidal_radius.to(u.pc)
    if kind in {"angular", "both"}:
        results["angular_size"] = angular_size(tidal_radius.to(u.pc), distance).to(u.arcmin)
    if return_galactic_mass:
        results["galactic_mass"] = galactic_mass
        results["galactic_mass_err"] = galactic_mass_err
    return results


def crossing_time(radius, velocity_dispersion) -> u.Quantity:
    """Return a crossing time from radius and velocity dispersion."""
    radius = ensure_units(radius, u.pc)
    velocity_dispersion = ensure_units(velocity_dispersion, u.km / u.s)
    return (radius / velocity_dispersion).to(u.Myr)


def half_mass_relaxation_time(n_stars, half_mass_radius, cluster_mass, *, lambda_value=0.11):
    """Spitzer-style half-mass relaxation time."""
    half_mass_radius = ensure_units(half_mass_radius, u.pc)
    cluster_mass = ensure_units(cluster_mass, u.Msun)
    n_stars = float(n_stars)
    mean_mass = cluster_mass / n_stars
    numerator = 0.138 * np.sqrt(n_stars) * half_mass_radius ** 1.5
    denominator = np.sqrt(G * mean_mass) * np.log(lambda_value * n_stars)
    return (numerator / denominator).to(u.Myr)


def mass_segregation_timescale(reference_mass, stellar_mass, relaxation_time):
    """Mass-segregation timescale ``t_seg = m_ref / m * t_rh``."""
    reference_mass = ensure_units(reference_mass, u.Msun)
    stellar_mass = ensure_units(stellar_mass, u.Msun)
    relaxation_time = ensure_units(relaxation_time, u.Myr)
    return (reference_mass / stellar_mass * relaxation_time).to(u.Myr)


class ClusterDynamicsAnalyzer:
    """Stateful facade for cluster dynamical and Galactic-radius estimates."""

    def __init__(self, data=None, *, distance=None, center=None, mass_column: str | None = "mass") -> None:
        self.data = data
        self.distance = distance
        self.center = center
        self.mass_column = mass_column

    def cluster_mass(self, *, distance=None, magnitude_column: str = "Gmag"):
        """Return a mass estimate from a mass column or the legacy luminosity scaling."""
        if self.data is None:
            raise ValueError("A data table is required to estimate cluster mass.")
        if self.mass_column and self.mass_column in self.data.colnames:
            return np.nansum(self.data[self.mass_column]).to(u.Msun)
        return estimate_cluster_mass(self.data[magnitude_column], distance or self.distance)

    def galactocentric_distance(self, *, center=None, distance=None, **kwargs):
        """Calculate Galactocentric distance for the analyzer center/distance."""
        center = center or self.center
        distance = distance or self.distance
        if center is None or distance is None:
            raise ValueError("center and distance are required.")
        return calculate_galactocentric_distance(center[0], center[1], distance=distance, **kwargs)

    def hill_radius(self, *, distance=None, center=None, cluster_mass=None, **kwargs):
        """Calculate Hill radius with analyzer defaults."""
        return calculate_hill_radius(
            data=self.data,
            distance=distance or self.distance,
            center=center or self.center,
            cluster_mass=cluster_mass,
            mass_column=self.mass_column,
            **kwargs,
        )

    def tidal_radius(self, *, cluster_mass=None, galactocentric_distance=None, distance=None, **kwargs):
        """Calculate the tidal-radius prior with analyzer defaults."""
        if galactocentric_distance is None:
            galactocentric_distance, _ = self.galactocentric_distance(distance=distance or self.distance)
        if cluster_mass is None:
            cluster_mass = self.cluster_mass(distance=distance or self.distance)
        return tidal_radius_prior(
            cluster_mass,
            galactocentric_distance,
            distance=distance or self.distance,
            data=self.data,
            **kwargs,
        )

    def gravitational_bound_radius(self, *, cluster_mass=None, cluster_mass_err=None, distance=None, **kwargs):
        """Calculate the Oort-constant gravitationally bound radius."""
        if cluster_mass is None:
            cluster_mass = self.cluster_mass(distance=distance or self.distance)
        return grav_bound_radius(
            cluster_mass,
            cluster_mass_err,
            distance=distance or self.distance,
            **kwargs,
        )


__all__ = [
    "ClusterDynamicsAnalyzer",
    "calculate_galactic_mass",
    "calculate_galactocentric_distance",
    "calculate_hill_radius",
    "crossing_time",
    "grav_bound_radius",
    "half_mass_relaxation_time",
    "mass_segregation_timescale",
    "tidal_radius_prior",
]
