"""HDBSCAN estimator utilities for EROTICA clustering."""

from __future__ import annotations

import hdbscan
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class FullSplit:
    """Single-fold splitter that uses every sample for fitting and scoring."""

    def split(self, X, y=None, groups=None):
        n = X.shape[0]
        yield np.arange(n), np.arange(n)

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1


def compute_relative_validity_from_mst(mst, labels):
    """Recompute HDBSCAN ``relative_validity_`` using a stored MST."""
    sizes = np.bincount(labels + 1)
    noise_size = sizes[0]
    cluster_sizes = sizes[1:]
    total_points = noise_size + cluster_sizes.sum()
    n_clusters = len(cluster_sizes)

    dsc = np.zeros(n_clusters)
    dspc = np.full(n_clusters, np.inf)
    max_dist = 0.0
    min_outlier_dist = np.inf

    if hasattr(mst, "iterrows"):
        iterator = (
            (int(row["from"]), int(row["to"]), row["distance"]) for _, row in mst.iterrows()
        )
    else:
        iterator = ((int(frm), int(to), dist) for frm, to, dist in mst)

    for frm, to, dist in iterator:
        lab_a = labels[frm]
        lab_b = labels[to]
        max_dist = max(max_dist, dist)
        if lab_a == -1 and lab_b == -1:
            continue
        if lab_a == -1 or lab_b == -1:
            min_outlier_dist = min(min_outlier_dist, dist)
            continue
        if lab_a == lab_b:
            dsc[lab_a] = max(dsc[lab_a], dist)
        else:
            dspc[lab_a] = min(dspc[lab_a], dist)
            dspc[lab_b] = min(dspc[lab_b], dist)

    if min_outlier_dist == np.inf:
        min_outlier_dist = max_dist
    correction = 2 * (max_dist if n_clusters > 1 else min_outlier_dist)
    dspc[np.isinf(dspc)] = correction
    V = (dspc - dsc) / np.maximum(dspc, dsc)
    return np.sum(cluster_sizes * V / total_points)


class HDBSCANEstimator(BaseEstimator, ClusterMixin):
    """Sklearn-compatible wrapper around :class:`hdbscan.HDBSCAN`."""

    def __init__(
        self,
        *,
        min_cluster_size: int = 10,
        min_samples: int | None = None,
        persistence_threshold: float = 0.0,
        **hdbscan_kwargs,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.hdbscan_kwargs = hdbscan_kwargs
        self.persistence_threshold = persistence_threshold
        self.model_ = None

    def fit(self, X, y=None):
        params = {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "gen_min_span_tree": True,
            "core_dist_n_jobs": -1,
            **self.hdbscan_kwargs,
        }
        self.model_ = hdbscan.HDBSCAN(**params).fit(X)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self.model_.labels_

    def score(self, X, y=None):
        if self.model_ is None:
            raise RuntimeError("Must call fit() before score().")

        threshold = self.persistence_threshold
        labels = self.model_.labels_.copy()
        persistence = self.model_.cluster_persistence_
        mst = self.model_.minimum_spanning_tree_.to_pandas()[["from", "to", "distance"]].to_numpy()

        if threshold > 0.0:
            for cid, value in enumerate(persistence):
                if value < threshold:
                    labels[labels == cid] = -1

        if threshold <= 0.0:
            return float(self.model_.relative_validity_)
        return float(compute_relative_validity_from_mst(mst, labels))


__all__ = ["FullSplit", "HDBSCANEstimator", "compute_relative_validity_from_mst"]
