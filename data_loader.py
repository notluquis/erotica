"""Public entry point for EROTICA data loading utilities."""
from __future__ import annotations

from erotica.io._constants import (
    ALIASES,
    FLUX_ERROR_COLUMNS,
    GAIA_DISTANCE_COLUMNS,
    GAIA_PHOTOMETRY_COLUMNS,
    PHOTOMETRIC_SYSTEMS,
    PROPER_MOTION_COLUMNS,
    TMASS_PHOTOMETRY_COLUMNS,
    UNIT_CORRECTIONS,
    WISE_PHOTOMETRY_COLUMNS,
    ZP_COLUMNS,
)
from erotica.io.loader import DataLoader

__all__ = [
    'DataLoader',
    'ALIASES',
    'PHOTOMETRIC_SYSTEMS',
    'GAIA_DISTANCE_COLUMNS',
    'GAIA_PHOTOMETRY_COLUMNS',
    'PROPER_MOTION_COLUMNS',
    'TMASS_PHOTOMETRY_COLUMNS',
    'WISE_PHOTOMETRY_COLUMNS',
    'ZP_COLUMNS',
    'FLUX_ERROR_COLUMNS',
    'UNIT_CORRECTIONS',
]
