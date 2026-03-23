"""Public entry point for COSMIC clustering utilities."""
from __future__ import annotations

from cosmic.core._estimator import (
    FullSplit,
    HDBSCANEstimator,
    compute_relative_validity_from_mst,
)
from cosmic.core.clustering import Clustering

# Legacy compatibility alias
HDBSCANClustering = Clustering

__all__ = [
    'Clustering',
    'HDBSCANClustering',  # Legacy alias
    'HDBSCANEstimator',
    'FullSplit',
    'compute_relative_validity_from_mst',
]
