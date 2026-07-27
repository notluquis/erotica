"""Unit-safe scalar and geometry helpers for EROTICA analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u


def ensure_units(value: Any, default_unit: u.UnitBase) -> u.Quantity:
    """Return ``value`` as an Astropy quantity in ``default_unit``."""
    if hasattr(value, "to"):
        return value.to(default_unit)
    return value * default_unit


def quantity_values(value: Any, unit: u.UnitBase | None = None) -> np.ndarray:
    """Return a finite-friendly float array from quantities, masked arrays, or arrays."""
    if hasattr(value, "to") and unit is not None:
        arr = np.asarray(value.to(unit).value, dtype=float)
    elif hasattr(value, "value"):
        arr = np.asarray(value.value, dtype=float)
    else:
        arr = np.asarray(value, dtype=float)
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return arr


def angular_size(linear_size: Any, distance: Any) -> u.Quantity:
    """Convert a projected linear size to angular size."""
    distance = ensure_units(distance, u.kpc)
    return (linear_size / distance).to(u.deg, equivalencies=u.dimensionless_angles())


def linear_size(angular_size_value: Any, distance: Any) -> u.Quantity:
    """Convert an angular size to projected linear size."""
    distance = ensure_units(distance, u.kpc)
    return (angular_size_value * distance).to(u.pc, equivalencies=u.dimensionless_angles())


def histogram_mode(data: Any, *, bins: str | int = "auto") -> float:
    """Estimate a scalar mode from a histogram bin center."""
    values = quantity_values(data)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Cannot estimate a mode from an empty or all-NaN array.")
    hist, bin_edges = np.histogram(values, bins=bins)
    max_index = int(np.argmax(hist))
    return float(0.5 * (bin_edges[max_index] + bin_edges[max_index + 1]))


def calculate_absolute_magnitude(apparent_mag: Any, distance: Any) -> u.Quantity:
    """Calculate absolute magnitude from apparent magnitude and distance.

    The returned values carry magnitude units so downstream photometric
    functions can remain unit-aware while still accepting plain arrays.
    """
    distance_pc = ensure_units(distance, u.pc).value
    return (quantity_values(apparent_mag, u.mag) - 5 * np.log10(distance_pc / 10)) * u.mag


def estimate_luminosity(absolute_magnitude: Any, solar_mag: float = 4.67) -> np.ndarray:
    """Estimate luminosity in solar units from absolute magnitude."""
    return 10 ** ((solar_mag - quantity_values(absolute_magnitude)) / 2.5)


def estimate_mass_from_luminosity(luminosity: Any) -> np.ndarray:
    """Simple mass-luminosity scaling used by the legacy paper workflow."""
    return np.power(quantity_values(luminosity), 1 / 3.5)


def estimate_cluster_mass(apparent_magnitudes: Any, distance: Any) -> u.Quantity:
    """Estimate observed cluster mass from magnitudes or a table with ``Gmag``.

    This intentionally preserves the lightweight mass-luminosity estimator from
    the paper notebooks. It is useful as a prior/diagnostic, not as a replacement
    for isochrone-based masses.
    """
    if hasattr(apparent_magnitudes, "colnames"):
        if "Gmag" not in apparent_magnitudes.colnames:
            raise ValueError("Table input must contain a 'Gmag' column.")
        apparent_magnitudes = apparent_magnitudes["Gmag"]
    absolute_mag = calculate_absolute_magnitude(apparent_magnitudes, distance)
    luminosity = estimate_luminosity(absolute_mag)
    return np.nansum(estimate_mass_from_luminosity(luminosity)) * u.Msun


__all__ = [
    "angular_size",
    "calculate_absolute_magnitude",
    "ensure_units",
    "estimate_cluster_mass",
    "estimate_luminosity",
    "estimate_mass_from_luminosity",
    "histogram_mode",
    "linear_size",
    "quantity_values",
]
