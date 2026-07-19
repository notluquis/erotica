"""Kinematic value transforms and lightweight summaries."""

from __future__ import annotations

import numpy as np
from astropy import units as u

from .units import ensure_units, quantity_values


def pm_amplitude(pmra, pmdec, *, unit: u.UnitBase = u.mas / u.yr) -> u.Quantity:
    """Return total proper-motion amplitude from RA/Dec components."""
    pmra_values = quantity_values(pmra, unit)
    pmdec_values = quantity_values(pmdec, unit)
    return np.sqrt(pmra_values**2 + pmdec_values**2) * unit


def projected_velocity_values(projected_pm, distance=None):
    """Return projected velocity values and labels.

    If ``distance`` is provided, the result is converted from mas/yr to km/s
    using ``v_t = 4.74047 * mu(mas/yr) * d(kpc)``.
    """
    projected_pm_values = quantity_values(projected_pm, u.mas / u.yr)
    if distance is None:
        return projected_pm_values, r"$\mathrm{mas~yr}^{-1}$", "Projected proper motion"

    distance_kpc = ensure_units(distance, u.kpc).value
    velocity_values = 4.74047 * projected_pm_values * distance_kpc
    return velocity_values, r"$\mathrm{km~s}^{-1}$", "Projected velocity"


def radial_velocity_values(radial_velocity, *, unit: u.UnitBase = u.km / u.s) -> np.ndarray:
    """Return radial-velocity values as a float array."""
    return quantity_values(radial_velocity, unit)


__all__ = ["pm_amplitude", "projected_velocity_values", "radial_velocity_values"]
