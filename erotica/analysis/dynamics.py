"""Dynamical and Galactic-structure helpers."""

from __future__ import annotations

import warnings

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
        "median": med,
        "lower": lo,
        "upper": hi,
        "minus": med - lo,
        "plus": hi - med,
        "n": int(finite.size),
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
            galactocentric_distance, galactocentric_distance_err = (
                calculate_galactocentric_distance(
                    center[0],
                    center[1],
                    distance=dist,
                    distance_err=dist_err,
                    solar_radius=solar_radius,
                )
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
        computed = calculate_galactic_mass(galactocentric_distance, galactocentric_distance_err)
        if isinstance(computed, tuple):
            galactic_mass, galactic_mass_err = computed
        else:
            galactic_mass, galactic_mass_err = computed, None
    else:
        galactic_mass = ensure_units(galactic_mass, u.Msun)
        galactic_mass_err = ensure_units(
            0 * u.Msun if galactic_mass_err is None else galactic_mass_err, u.Msun
        )
    if cluster_mass is None:
        if mass_column and data is not None and mass_column in data.colnames:
            cluster_mass = np.nansum(data[mass_column]).to(u.Msun)
        else:
            cluster_mass = estimate_cluster_mass(
                data[magnitude_column] if hasattr(data, "colnames") else data, dist
            )
        cluster_mass_err = 0 * u.Msun
    cluster_mass = ensure_units(cluster_mass, u.Msun)
    cluster_mass_err = ensure_units(
        0 * u.Msun if cluster_mass_err is None else cluster_mass_err, u.Msun
    )

    radius = (galactocentric_distance * (cluster_mass / (3 * galactic_mass)) ** (1 / 3)).to(u.pc)
    d_radius_d_galdist = (cluster_mass / (3 * galactic_mass)) ** (1 / 3)
    d_radius_d_cluster_mass = galactocentric_distance / (
        3 ** (4 / 3) * galactic_mass ** (1 / 3) * cluster_mass ** (2 / 3)
    )
    d_radius_d_galactic_mass = -(galactocentric_distance * cluster_mass ** (1 / 3)) / (
        3 ** (4 / 3) * galactic_mass ** (4 / 3)
    )
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

    cluster_mass_err = ensure_units(
        0 * u.Msun if cluster_mass_err is None else cluster_mass_err, u.Msun
    )
    a_b_squared = (A - B) ** 2
    a_b_squared_err = 2 * np.abs(A - B) * np.sqrt(A_err**2 + B_err**2)
    radius = ((G * cluster_mass / (2 * a_b_squared)) ** (1 / 3)).to(u.pc)
    d_radius_dm = (1 / 3) * (radius / cluster_mass)
    d_radius_da_b_squared = (-1 / 3) * (radius / a_b_squared)
    radius_err = np.sqrt(
        (d_radius_dm * cluster_mass_err) ** 2 + (d_radius_da_b_squared * a_b_squared_err) ** 2
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
        computed = calculate_galactic_mass(galactocentric_distance, galactocentric_distance_err)
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


# Henon's Eq. (15): the velocity-distribution term for a Maxwellian, ln(2/3) - EulerGamma.
_HENON_MAXWELLIAN_TERM = np.log(2.0 / 3.0) - 0.5772156649015329

# The bare 0.4 inside Henon's Eq. (11) is the DOMINANT TERM of his Eq. (6) -- the virial
# coefficient of <v^2> = 0.4 G M / R_h, via Spitzer & Hart (1971) Eq. 3, p_0 = R_h/(0.4 N).
# It is not an independent constant: the two correction terms above are what Henon adds to it.
# See COULOMB_CALIBRATIONS[0.4] for the full provenance.
_SPITZER_DOMINANT_TERM = 0.4


def coulomb_argument_from_mass_function(mass_function, m_min, m_max, *, n_grid=4000):
    r"""Compute the Coulomb-logarithm argument :math:`\gamma` from the mass function.

    .. math:: \ln\gamma = \ln(0.4) + \frac{I_2}{I_1} + \frac{I_3}{I_1}

    Hénon (1975, IAU Symp. 69, 133, `1975IAUS...69..133H`) Eq. (11), with :math:`I_2/I_1` his
    Eq. (15) for a Maxwellian velocity distribution and :math:`I_3/I_1` his Eq. (26), a functional of
    the mass spectrum alone:

    .. math:: \frac{I_3}{I_1} = \frac{\iint h(m_1)h(m_2)\,m_1 m_2^2
                                      \ln\!\frac{2\bar{m}}{m_1+m_2}\,dm_1 dm_2}
                                     {\int h(m_1) m_1 dm_1 \int h(m_2) m_2^2 dm_2}

    **There is a closed form for the mass-function dependence of** :math:`\gamma` **and the
    open-cluster literature does not use it.** Every open-cluster paper found that quotes a Coulomb
    argument at all uses a single constant.

    Parameters
    ----------
    mass_function : callable
        ``h(m)`` — the number of stars per unit mass, up to normalisation. Only its shape matters.
    m_min, m_max : float
        Mass limits in solar masses. For a real cluster ``m_max`` is the **turn-off mass**, not the
        IMF's upper bound, which is why :math:`\gamma` evolves with age.
    n_grid : int
        Log-spaced grid points. 4000 reproduces Hénon's own tables to four decimals.

    Returns
    -------
    dict
        ``gamma``, ``i3_over_i1``, ``mean_mass`` and ``mass_ratio``.

    Notes
    -----
    **Verified against the primary source.** This reproduces Hénon's Table II (continuous
    :math:`h \propto m^{-2}`) and Table III (discrete spectrum) **to four decimal places**, recovers
    the equal-mass limit :math:`\gamma = 0.1497` against his Eq. (19) value of 0.15, and gives 0.0454
    for the :math:`m^{-2.5}` spectrum over a range of 37.5 that Giersz & Heggie (1996) quote Hénon's
    theory as predicting :math:`\approx 0.044` for.

    .. important::
       **Theory and N-body calibration disagree.** For Giersz & Heggie's own mass function this gives
       0.0454 where their N-body fit gives 0.02 — a factor of **2.3**. Use this to understand *how*
       :math:`\gamma` depends on the mass spectrum, not as a drop-in replacement for a calibration.

    .. important::
       **The dependence is strong, and it is age-dependent.** For a Kroupa (2001) IMF truncated at the
       turn-off:

       ==========  =================  =========
       age         :math:`m_{max}`    :math:`\gamma`
       ==========  =================  =========
       1 Myr       100 M☉             **0.0050**
       10 Myr      20 M☉              0.0175
       40 Myr      8 M☉               0.0325
       400 Myr     3 M☉               0.0566
       1 Gyr       2 M☉               **0.0686**
       ==========  =================  =========

       **A factor of 14 across the range.** So a single :math:`\gamma` imposes an age-dependent
       systematic on any ``age / t_rh`` trend — which is precisely the quantity open-cluster
       mass-function studies correlate against.

    Examples
    --------
    >>> kroupa = lambda m: np.where(
    ...     np.asarray(m) < 0.5, np.asarray(m) ** -1.3, 0.5**1.0 * np.asarray(m) ** -2.3
    ... )
    >>> round(coulomb_argument_from_mass_function(kroupa, 0.08, 8.0)["gamma"], 4)
    0.0325
    """
    if not m_max > m_min > 0:
        raise ValueError(f"require 0 < m_min < m_max, got {m_min} and {m_max}")
    log_m = np.linspace(np.log(m_min), np.log(m_max), int(n_grid))
    m = np.exp(log_m)
    # h(m) dm = h(m) m d(log m), so the log-measure is folded in once here.
    weight = np.asarray(mass_function(m), dtype=float) * m
    mean_mass = np.trapezoid(weight * m, log_m) / np.trapezoid(weight, log_m)

    m1, m2 = np.meshgrid(m, m, indexing="ij")
    kernel = np.outer(weight * m, weight * m) * m2
    i3 = np.trapezoid(
        np.trapezoid(kernel * np.log(2.0 * mean_mass / (m1 + m2)), log_m, axis=1), log_m
    ) / (np.trapezoid(weight * m, log_m) * np.trapezoid(weight * m**2, log_m))

    return dict(
        gamma=float(_SPITZER_DOMINANT_TERM * np.exp(_HENON_MAXWELLIAN_TERM + i3)),
        i3_over_i1=float(i3),
        mean_mass=float(mean_mass),
        mass_ratio=float(m_max / m_min),
    )


# Calibration provenance for the Coulomb-logarithm argument. Each entry records the range the value
# was calibrated over, so calling outside it can WARN instead of silently extrapolating -- which is
# this programme's most repeated error (see methodology.md K.1.5).
COULOMB_CALIBRATIONS = {
    0.4: dict(
        source="analytic, equal mass -- DOMINANT TERM ONLY",
        reference="Spitzer & Hart (1971) Eqs. 1-3, on the virial coefficient of Spitzer (1969)",
        bibcode="1971ApJ...164..399S",
        bibcode_origin="1969ApJ...158L.139S",  # Spitzer (1969), source of the 0.4 itself
        n_range=None,  # analytic, so no calibration range in N
        equal_mass=True,
        isolated=None,
        superseded_by=0.15,
        # This 0.4 is NOT a Coulomb-logarithm constant. It is the VIRIAL COEFFICIENT of
        # <v^2> = 0.4 G M / R_h -- Spitzer & Hart (1971) Eq. 2, verbatim from the ADS scan:
        # "the constant 0.4 provides a reasonably good approximation for most systems
        # (Spitzer 1969)". It reaches ln(gamma N) only through their Eq. 3, p_0 = R_h/(0.4 N),
        # where the factor 2 from m_A + m_B = 2m cancels the factor 2 from V^2 = 2 v_m^2 exactly.
        # So gamma = 0.4 is a CHOICE OF CUTOFF (b_max = R_h) plus a structural coefficient, and
        # a different velocity convention gives a different number: CMC uses the BT08 p.361
        # coefficient 0.45 with w^2 = <v^2> and lands on 0.2, a factor 2 from the same algebra.
        note="Dominant term only. Henon (1975) Eq. 11 shows ln(0.4 N) is the leading term of "
        "ln(0.4 N) + ln(V^2/2<V_1^2>) + ln(2<m>/(m_1+m_2)); completing the same calculation "
        "gives 0.15. Legitimate use: reproducing work that adopted it, e.g. Aarseth, Henon & "
        "Wielen (1974) and anything benchmarked against it.",
    ),
    0.15: dict(
        source="analytic, equal mass -- non-dominant terms included",
        reference="Henon (1975) Eq. 19, IAU Symp. 69, 133",
        bibcode="1975IAUS...69..133H",
        n_range=None,
        equal_mass=True,
        isolated=None,
        # Henon's abstract, verbatim from the ADS scan: "a constant usually taken to be of the
        # order of 0.4. A reconsideration of the 'non-dominant terms' leads to a substantially
        # lower value, of the order of 0.15 for equal masses and 0.075 for unequal masses with a
        # typical distribution." Reproduced here as 0.1497 by
        # coulomb_argument_from_mass_function in the equal-mass limit.
        note="The corrected analytic equal-mass value, and the one Tep & Heggie (2026), "
        "2026arXiv260714844T, adopt: with Lambda = 0.15 N, Chandrasekhar theory reproduces "
        "the N-body core-collapse time to 1.026 +/- 0.023 at N = 1e4. Henon's Table I gives "
        "0.108-0.169 across truncated power-law velocity distributions, so the velocity "
        "dependence is weak; the mass-function dependence is not.",
    ),
    0.11: dict(
        source="N-body, equal mass",
        reference="Giersz & Heggie (1994) Paper I, 1994MNRAS.268..257G",
        bibcode="1994MNRAS.268..257G",
        n_range=(250, 1000),
        equal_mass=True,
        isolated=True,
        # Paper I is correct and verified from its abstract: "values for the coefficient gamma in
        # the Coulomb logarithm ln(gamma N) ... These are gamma ~= 0.11". Do not "correct" this to
        # 1994MNRAS.270..298G: Part Two, same authors and year, REDETERMINES gamma after core
        # collapse and is what Tep & Heggie (2026) cite for it. Both are real; Paper I is the
        # origin, and is the one Giersz & Heggie (1996) mean by "the value found in Paper I".
    ),
    0.02: dict(
        source="N-body, multi-mass (power-law MF)",
        reference="Giersz & Heggie (1996), 1996MNRAS.279.1037G",
        bibcode="1996MNRAS.279.1037G",
        n_range=(250, 1000),  # verbatim from their abstract: "from 250 to 1000 stars each"
        equal_mass=False,
        isolated=True,  # verbatim: "the systems are isolated"
    ),
}


def coulomb_calibration_warnings(lambda_value, n_stars, *, tidally_limited=True, equal_mass=False):
    """Reasons the requested Coulomb argument is being used outside its calibration.

    Returns a list of strings, empty if the use is inside the calibrated regime. Separated from
    :func:`half_mass_relaxation_time` so it is testable on its own and so a caller can inspect the
    reasons rather than only catch a warning.
    """
    entry = COULOMB_CALIBRATIONS.get(round(float(lambda_value), 6))
    if entry is None:
        return [f"gamma={lambda_value} has no recorded calibration provenance in this module."]
    problems = []
    if entry.get("superseded_by") is not None:
        problems.append(
            f"gamma={lambda_value} is the DOMINANT-TERM-ONLY value ({entry['reference']}); "
            f"Henon (1975) completed the same calculation including the non-dominant terms and "
            f"obtained {entry['superseded_by']}. Use {lambda_value} only to reproduce work that "
            f"adopted it."
        )
    if entry["n_range"] is not None:
        low, high = entry["n_range"]
        if not (low <= n_stars <= high):
            problems.append(
                f"N={n_stars:g} is outside the calibration range {low}-{high} of "
                f"{entry['reference']}."
            )
    if entry["equal_mass"] and not equal_mass:
        problems.append(
            f"gamma={lambda_value} is an EQUAL-MASS value ({entry['source']}); this cluster has a "
            "mass function."
        )
    if entry["isolated"] and tidally_limited:
        problems.append(
            f"gamma={lambda_value} was calibrated on ISOLATED systems; this cluster is tidally "
            "limited. Giersz & Heggie (1997), 1997MNRAS.286..709G, is the matching setup."
        )
    return problems


def half_mass_relaxation_time(n_stars, half_mass_radius, cluster_mass, *, lambda_value=0.11):
    r"""Half-mass relaxation time.

    .. math:: t_{rh} = \frac{0.138\, N^{1/2} r_h^{3/2}}
                            {\sqrt{G \bar{m}}\; \ln(\gamma N)}

    The arrangement implemented here is **Heggie & Hut (2003) Eq. 9**; Spitzer (1987) gives it as
    Eq. 2-63 (2-62 is the *local* reference time, coefficient 0.065). Note that 0.138, 0.0600 and
    8.9e5 are the **same constant** in different unit systems — ``0.065 × 0.4^1.5 × 8π/3 = 0.13776``
    and ``0.138/ln(10) = 0.05993`` — so pairing a ``log10`` coefficient with a natural log is a
    silent factor of 2.303.

    .. danger::
       **``n_stars`` is the number of BOUND stars, ``M/<m>`` — not the observed member count.**
       Giersz & Heggie (1996) define it that way, and Cordoni et al. (2023) Eq. 5 writes the argument
       as ``M/<m>`` explicitly. Passing an observed member list corrupts **both** terms, because this
       function derives the mean stellar mass as ``M/N``:

       ==========  ===============  ==========  =========================================
       ``N``       implied ``<m>``  ``t_rh``    what it is
       ==========  ===============  ==========  =========================================
       254         3.54 M☉          14.8 Myr    NGC 6383's observed members at 40′ — **wrong**
       687         1.31 M☉          30.8 Myr    ``M/<m>`` with P01's measured mean per system
       1800        0.50 M☉          66.0 Myr    ``M/<m>`` with a Kroupa mean
       ==========  ===============  ==========  =========================================

       A **factor 4.5**, and it never announces itself — the implied ``<m>`` of 3.54 M☉ is the tell,
       since no IMF produces it. The function warns when the implied mean mass is unphysical.

       This also corrects an earlier claim in this codebase. The pole at ``N = 1/γ`` was said to sit
       *inside* the census because the census median observed membership is 61. With ``N`` defined as
       Giersz & Heggie define it, ``N`` for a 900 M☉ cluster is 1500–1800, and the pole survives only
       for very young, very low-mass clusters — which are out of virial equilibrium anyway, so
       ``t_rh`` is undefined there for a different reason.

    .. seealso::
       **Knowledge-graph node**: ``kb/methods/coulomb-logarithm-argument.md`` in the research hub
       carries the full provenance -- every value, what it was calibrated on, what it costs to get
       wrong, and the open questions. The machine-readable form is
       :data:`COULOMB_CALIBRATIONS` in this module. Keep the two in step.

    Parameters
    ----------
    lambda_value : float, default 0.11
        The Coulomb-logarithm argument :math:`\gamma` in :math:`\ln(\gamma N)`. **The default is
        very likely wrong for a Gaia open cluster** — see the warning below.

    .. warning::
       **The default ``lambda_value = 0.11`` is the EQUAL-MASS calibration, and no real cluster has
       equal masses.** The three values in circulation are not interchangeable:

       ==============  ==============================================  =========================
       :math:`\gamma`  what it is                                      source
       ==============  ==============================================  =========================
       0.4             analytic, equal mass, **dominant term only**     Spitzer & Hart (1971)
       0.15            analytic, equal mass, **terms completed**        Hénon (1975) Eq. 19
       **0.11**        N-body calibration, **equal mass**               Giersz & Heggie (1994)
       **0.02**        N-body calibration, **multi-mass**               Giersz & Heggie (1996)
       ==============  ==============================================  =========================

       .. important::
          **0.4 and 0.15 are the same calculation, not two competing conventions.** The 0.4 is not
          a Coulomb constant at all: it is the *virial coefficient* of
          :math:`\langle v^2\rangle = 0.4\,GM/R_h`, which Spitzer & Hart (1971) Eq. 2 attribute to
          Spitzer (1969) with the words *"the constant 0.4 provides a reasonably good approximation
          for most systems"*. It reaches :math:`\ln(\gamma N)` through their Eq. 3,
          :math:`p_0 = R_h/(0.4N)`, once :math:`b_{max}` is *chosen* to be :math:`R_h`.

          Hénon (1975) then showed that :math:`\ln(0.4N)` is only the leading term of
          :math:`\ln(0.4N) + \ln(V^2/2\langle V_1^2\rangle) + \ln(2\langle m\rangle/(m_1+m_2))`, and
          that *"the classical approximation consists in omitting these non-dominant terms."*
          Including them gives **0.15** for equal masses and ≈0.075 for a typical mass spectrum.
          Tep & Heggie (2026), ``2026arXiv260714844T``, adopt 0.15 and show it reproduces the N-body
          core-collapse time to **1.026 ± 0.023** at :math:`N = 10^4`.

          So 0.4 is superseded rather than wrong-in-principle, and it stays in the registry because
          reproducing Aarseth, Hénon & Wielen (1974) — or anything benchmarked against it — requires
          it. The velocity convention alone moves it: CMC pairs the Binney & Tremaine (2008) p. 361
          coefficient 0.45 with :math:`w^2 = \langle v^2\rangle` and reports 0.2, a factor 2 from
          the same algebra.

       Giersz & Heggie (1996) describe the multi-mass value as *"a factor 7 smaller than the value
       found in Paper I for systems with equal masses"*, and Heggie & Hut (2003) p. 142 states the
       distinction plainly. Every Gaia open cluster has a mass function, so 0.11 is certainly the
       wrong branch — the difference is **×5.3 at N = 100** and ×2.0 at N = 1000.

       .. caution::
          **But 0.02 is not simply "the right value" either, and saying so would repeat the
          extrapolation error this codebase already documents.** Giersz & Heggie (1996) state their
          own setup in the abstract: *"The distribution of stellar masses is a power law, and the
          systems are **isolated**"*, with **N from 250 to 1000**.

          A Gaia open cluster is **not isolated** — it is tidally limited — and the census median is
          **N = 61**, below their calibration range. Applying 0.02 here extrapolates in two
          directions at once, which is exactly the failure mode recorded in ``methodology.md``
          K.1.5.

          Paper IV, **Giersz & Heggie (1997, MNRAS 286, 709,** ``1997MNRAS.286..709G``**),
          "Unequal masses with a tidal field"**, is the matching physical setup and is the reference
          to check before adopting any value. Its abstract does not quote :math:`\gamma`; the number
          has to come from the text.

          **Until that is read, treat the Coulomb argument as a stated uncertainty of at least a
          factor of a few, not as a parameter with a known value.**

       The default is left at 0.11 rather than changed silently, because changing it would move every
       previously published relaxation time from this code without a record. Pass
       ``lambda_value=0.02`` for a mass-segregated cluster and say so.

    .. danger::
       **The formula has a pole at** :math:`N = 1/\gamma` **and below it returns a negative time.**
       That is not a rounding artefact: at ``lambda_value=0.11`` this code returned −1.6 Myr for
       N = 5 and −173.7 Myr for N = 9, silently, before the guard below was added.

       The pole sits at N = 9.1 for :math:`\gamma` = 0.11 but at **N = 50** for the physically correct
       :math:`\gamma` = 0.02 — and **40% of the Hunt & Reffert (2024) census has N < 50**, with a
       median of 61. So adopting the correct Coulomb argument puts a large part of the census
       *outside the formula's domain*. That is a real limitation of the expression, not of the
       implementation, and it is raised rather than papered over.

    Raises
    ------
    ValueError
        If ``lambda_value * n_stars <= 1``, where the logarithm is non-positive and the result would
        be infinite or negative.

    Notes
    -----
    A second convention differs in the **prefactor**: Binney & Tremaine (2008) Eq. 7.108 uses 0.17
    rather than 0.138, a flat 23%, and parts of the Gaia cluster literature pair it with
    :math:`\gamma` = 0.1. Compounded across both conventions the worst-case spread at N = 200 is a
    factor of **3.9**. Neither Tarricq et al. (2022) nor Hunt & Reffert (2024) publishes a relaxation
    time, so there is no external catalogue to validate against.
    """
    half_mass_radius = ensure_units(half_mass_radius, u.pc)
    cluster_mass = ensure_units(cluster_mass, u.Msun)
    n_stars = float(n_stars)
    for reason in coulomb_calibration_warnings(lambda_value, n_stars):
        warnings.warn(f"Coulomb argument used outside its calibration: {reason}", stacklevel=2)
    # Posterior samples arrive as arrays; the guard is about the typical value, not each draw.
    implied_mean_mass = float(np.median(np.atleast_1d((cluster_mass / n_stars).to_value(u.Msun))))
    if not 0.1 <= implied_mean_mass <= 2.0:
        warnings.warn(
            f"n_stars={n_stars:g} with a cluster mass of "
            f"{np.median(np.atleast_1d(cluster_mass.to_value(u.Msun))):.0f} Msun implies a mean stellar mass of "
            f"{implied_mean_mass:.2f} Msun, which no IMF produces. n_stars must be the number of "
            "BOUND stars, M/<m>, not an observed member count -- passing the latter corrupts both "
            "the Coulomb logarithm and the mean mass.",
            stacklevel=2,
        )
    argument = lambda_value * n_stars
    if argument <= 1.0:
        raise ValueError(
            f"ln(gamma*N) is non-positive: gamma={lambda_value}, N={n_stars:g}, "
            f"gamma*N={argument:.3f}. The half-mass relaxation time is undefined at or below "
            f"N = 1/gamma = {1 / lambda_value:.1f}; the expression returns a negative time there. "
            "This is a limit of the formula, not of the input."
        )
    mean_mass = cluster_mass / n_stars
    numerator = 0.138 * np.sqrt(n_stars) * half_mass_radius**1.5
    denominator = np.sqrt(G * mean_mass) * np.log(argument)
    return (numerator / denominator).to(u.Myr)


def mass_segregation_timescale(reference_mass, stellar_mass, relaxation_time):
    """Mass-segregation timescale ``t_seg = m_ref / m * t_rh``."""
    reference_mass = ensure_units(reference_mass, u.Msun)
    stellar_mass = ensure_units(stellar_mass, u.Msun)
    relaxation_time = ensure_units(relaxation_time, u.Myr)
    return (reference_mass / stellar_mass * relaxation_time).to(u.Myr)


class ClusterDynamicsAnalyzer:
    """Stateful facade for cluster dynamical and Galactic-radius estimates."""

    def __init__(
        self, data=None, *, distance=None, center=None, mass_column: str | None = "mass"
    ) -> None:
        r"""Bind the table, distance and centre reused by every estimate.

        Parameters
        ----------
        data : QTable, optional
            Cluster source table. Only needed by the methods that sum a mass
            column or fall back to the luminosity estimator;
            :meth:`galactocentric_distance` works without it. Held by reference.
        distance : Quantity or float, optional
            Heliocentric distance to the cluster. Passed through
            :func:`~erotica.analysis.units.ensure_units` with a default of
            **kpc**, so a plain number is read as kpc. Every method accepts a
            per-call override, applied as ``distance or self.distance`` -- note
            that this is a truthiness test, so a distance of exactly zero would
            fall back to the stored value rather than being used.
        center : sequence of Quantity, optional
            ``(ra, dec)`` of the cluster centre. The entries must **carry
            angular units**: they reach :class:`~astropy.coordinates.SkyCoord`
            as ``ra=``/``dec=``, and bare floats raise ``UnitTypeError``
            ("Longitude instances require units equivalent to 'rad'"). This is
            unlike :class:`~erotica.analysis.structure.ClusterStructureAnalyzer`
            and the radial-profile helpers, which do take bare degrees.
        mass_column : str or None, default "mass"
            Column holding per-star masses. :meth:`cluster_mass` sums it with
            ``.to(u.Msun)``, so the column **must carry a mass unit**; a bare
            float column raises. When the column is absent, or when this is
            ``None``, the method silently falls back to the legacy
            mass-luminosity scaling from `magnitude_column` instead -- so the
            returned mass can come from either of two very different estimators
            with nothing in the return value to say which. The luminosity path
            is a diagnostic, not a substitute for isochrone masses.

        Notes
        -----
        Masses are returned in :math:`M_\odot` and radii in pc or kpc as the
        underlying function dictates; all are Astropy Quantities.

        :meth:`galactocentric_distance` returns a ``(radius, radius_err)``
        pair, because binding a `center` selects the equatorial call form of
        :func:`calculate_galactocentric_distance`. The Galactic-coordinate form
        of that function returns a bare radius instead, so results from the two
        are not interchangeable.
        """
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

    def tidal_radius(
        self, *, cluster_mass=None, galactocentric_distance=None, distance=None, **kwargs
    ):
        """Calculate the tidal-radius prior with analyzer defaults."""
        if galactocentric_distance is None:
            galactocentric_distance, _ = self.galactocentric_distance(
                distance=distance or self.distance
            )
        if cluster_mass is None:
            cluster_mass = self.cluster_mass(distance=distance or self.distance)
        return tidal_radius_prior(
            cluster_mass,
            galactocentric_distance,
            distance=distance or self.distance,
            data=self.data,
            **kwargs,
        )

    def gravitational_bound_radius(
        self, *, cluster_mass=None, cluster_mass_err=None, distance=None, **kwargs
    ):
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
