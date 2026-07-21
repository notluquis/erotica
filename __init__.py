"""Top-level package for the PUMPS project."""

__all__ = [
    'DataLoader',
    'Clustering',
    'HDBSCANEstimator',
    'FullSplit',
    'compute_relative_validity_from_mst',
    'DataPreprocessor',
    'compare_datasets',
    'ClusterAnalyzer',
    'ClusterDynamicsAnalyzer',
    'ClusterFigureBuilder',
    'ClusterInferenceAnalyzer',
    'ClusterStructureAnalyzer',
    'IsochroneFitter',
    'PhotometricMassEstimator',
]


def __getattr__(name):
    if name == "DataLoader":
        from .data_loader import DataLoader

        return DataLoader
    if name in {"Clustering", "HDBSCANEstimator", "FullSplit", "compute_relative_validity_from_mst"}:
        from . import clustering as _clustering

        return getattr(_clustering, name)
    if name == "DataPreprocessor":
        from .data_preprocessor import DataPreprocessor

        return DataPreprocessor
    if name == "compare_datasets":
        from .utils import compare_datasets

        return compare_datasets
    if name == "ClusterAnalyzer":
        from .cluster_analysis import ClusterAnalyzer

        return ClusterAnalyzer
    if name in {
        "ClusterDynamicsAnalyzer",
        "ClusterFigureBuilder",
        "ClusterInferenceAnalyzer",
        "ClusterStructureAnalyzer",
        "IsochroneFitter",
        "PhotometricMassEstimator",
    }:
        from . import pumps as _pkg

        return getattr(_pkg, name)
    raise AttributeError(f"module 'PUMPS' has no attribute {name!r}")
