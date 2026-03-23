"""COSMIC package public API.

This package exposes the main classes from the legacy flat module layout so
existing imports continue to work after we refactor files into subpackages.
"""

from .core.clustering import Clustering
from .io.loader import DataLoader
from .preprocess.preprocessor import DataPreprocessor
from .utils.utils import compare_datasets

__all__ = ["DataLoader", "Clustering", "DataPreprocessor", "compare_datasets"]
