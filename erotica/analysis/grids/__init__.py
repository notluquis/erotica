"""Isochrone-grid backends behind one contract (see :mod:`.base` and ``AGENTS.md`` here).

Every backend enters with its control (another grid on the same cluster) and its oracle; the rule
and the measured numbers live in this directory's ``AGENTS.md``.
"""

from .base import GridNode, GridProvenance, IsochroneGrid, common_eep_table, safe_window
from .holdout import holdout_feh_node, holdout_grid, holdout_node, recommended_sigma_floor
from .mist import MISTGrid
from .parsec import PARSECGrid
from .pseudo_eep import regrid

__all__ = [
    "GridNode",
    "GridProvenance",
    "IsochroneGrid",
    "MISTGrid",
    "PARSECGrid",
    "common_eep_table",
    "holdout_feh_node",
    "holdout_grid",
    "holdout_node",
    "recommended_sigma_floor",
    "regrid",
    "safe_window",
]
