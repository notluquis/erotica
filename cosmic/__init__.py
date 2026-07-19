"""COSMIC package public API.

This package exposes the main classes from the legacy flat module layout so
existing imports continue to work after we refactor files into subpackages.
"""

_ANALYSIS_EXPORTS = {
    "ClusterAnalyzer",
    "ClusterDynamicsAnalyzer",
    "ClusterFigureBuilder",
    "ClusterInferenceAnalyzer",
    "ClusterStructureAnalyzer",
    "IsochroneFitter",
    "PhotometricMassEstimator",
}

__all__ = [
    "DataLoader",
    "Clustering",
    "DataPreprocessor",
    "compare_datasets",
    *_ANALYSIS_EXPORTS,
]


def __getattr__(name):
    if name == "Clustering":
        from .core.clustering import Clustering

        return Clustering
    if name == "DataLoader":
        from .io.loader import DataLoader

        return DataLoader
    if name == "DataPreprocessor":
        from .preprocess.preprocessor import DataPreprocessor

        return DataPreprocessor
    if name == "compare_datasets":
        from .utils.utils import compare_datasets

        return compare_datasets
    if name in _ANALYSIS_EXPORTS:
        from . import analysis as _analysis

        return getattr(_analysis, name)
    raise AttributeError(f"module 'cosmic' has no attribute {name!r}")
