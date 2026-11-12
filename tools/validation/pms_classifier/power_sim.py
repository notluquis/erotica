"""Beta-binomial piece shared by ``exp2_power.py`` and ``r38_power.py``: the per-cluster FPR drawn as
Beta(p a, (1 - p) a) with mean p and coefficient of variation cv.

For Beta(p a, (1 - p) a) the variance is p (1 - p) / (a + 1), so CV^2 = (1 - p) / (p (a + 1)) and

    a = (1 - p (1 + cv^2)) / (p cv^2).

Up to 2026-11-12 both scripts used a = (1 - p (1 + cv^2)) / cv^2, without the 1/p (R113-01): at cv = 1
the simulated CV was 3.16 at p = 0.05, 4.08 at p = 0.03 and 1.83 at p = 0.15. a <= 0 has no Beta with
that mean and CV (cv^2 >= (1 - p) / p); then the cluster FPRs are taken equal to p, as exp2_power.py did.
"""
from __future__ import annotations

import numpy as np


def beta_concentration(p: float, cv: float) -> float:
    """Concentration a of Beta(p a, (1 - p) a) with mean ``p`` and coefficient of variation ``cv``."""
    return (1.0 - p * (1.0 + cv**2)) / (p * cv**2)


def cluster_fpr(p: float, cv: float, n_clusters: int, rng) -> np.ndarray:
    """Per-cluster FPR: p for every cluster if ``cv`` is 0, ``p`` is not in (0, 1) or no Beta has that
    mean and CV; else the Beta draw."""
    if cv > 0 and 0 < p < 1:
        a = beta_concentration(p, cv)
        if a > 0:
            return rng.beta(p * a, (1 - p) * a, n_clusters)
    return np.full(n_clusters, p)
