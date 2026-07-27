"""Constants used by EROTICA data preprocessing utilities."""

from __future__ import annotations

import astropy.units as u

COLUMN_MAPPING = {
    "phot_g_mean_mag": "Gmag",
    "phot_bp_mean_mag": "G_BPmag",
    "phot_rp_mean_mag": "G_RPmag",
    "bp_rp": "BP_RP",
}

ZERO_POINT_REQUIRED_COLUMNS = [
    "parallax",
    "Gmag",
    "nu_eff_used_in_astrometry",
    "pseudocolour",
    "ecl_lat",
    "astrometric_params_solved",
]

PSEUDOCOLOUR_RANGE = (1.24 / u.um, 1.72 / u.um)

PM_CORRECTION_ROWS = (
    (0.0, 9.0, 18.4, 33.8, -11.3),
    (9.0, 9.5, 14.0, 30.7, -19.4),
    (9.5, 10.0, 12.8, 31.4, -11.8),
    (10.0, 10.5, 13.6, 35.7, -10.5),
    (10.5, 11.0, 16.2, 50.0, 2.1),
    (11.0, 11.5, 19.4, 59.9, 0.2),
    (11.5, 11.75, 21.8, 64.2, 1.0),
    (11.75, 12.0, 17.7, 65.6, -1.9),
    (12.0, 12.25, 21.3, 74.8, 2.1),
    (12.25, 12.5, 25.7, 73.6, 1.0),
    (12.5, 12.75, 27.3, 76.6, 0.5),
    (12.75, 13.0, 34.9, 68.9, -2.9),
)

DEFAULT_DROP_COLUMNS = [
    "ra",
    "dec",
    "pmra",
    "pmdec",
    "parallax",
    "Gmag",
    "G_BPmag",
    "G_RPmag",
]

__all__ = [
    "COLUMN_MAPPING",
    "ZERO_POINT_REQUIRED_COLUMNS",
    "PSEUDOCOLOUR_RANGE",
    "PM_CORRECTION_ROWS",
    "DEFAULT_DROP_COLUMNS",
]
