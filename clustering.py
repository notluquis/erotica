"""Public entry point for PUMPS clustering utilities."""
from __future__ import annotations

from pumps.core._estimator import (
    FullSplit,
    HDBSCANEstimator,
    compute_relative_validity_from_mst,
)
from pumps.core.clustering import Clustering

# Legacy compatibility alias
HDBSCANClustering = Clustering

__all__ = [
    'Clustering',
    'HDBSCANClustering',  # Legacy alias
    'HDBSCANEstimator',
    'FullSplit',
    'compute_relative_validity_from_mst',
]
