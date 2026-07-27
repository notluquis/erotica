"""Public entry point for EROTICA clustering utilities."""
from __future__ import annotations

from erotica.core._estimator import (
    FullSplit,
    HDBSCANEstimator,
    compute_relative_validity_from_mst,
)
from erotica.core.clustering import Clustering

# Legacy compatibility alias
HDBSCANClustering = Clustering

__all__ = [
    'Clustering',
    'HDBSCANClustering',  # Legacy alias
    'HDBSCANEstimator',
    'FullSplit',
    'compute_relative_validity_from_mst',
]
