"""Combine proper motions of the same star from two catalogues that are not independent.

The case this exists for is VIRAC2 (VVV/VVVX, Smith et al. 2025, ``2025MNRAS.536.3707S``) together
with Gaia DR3. VIRAC2 calibrates every VISTA image against Gaia DR3 positions propagated to the
VISTA epoch with Gaia's own proper motion and parallax, using Gaia 5-parameter sources with
``ruwe < 1.4`` as references (Smith et al. 2025, Sect. 2.3). Two things follow, and each is a
process this module has to model rather than assume away:

1. **A per-star dependence for reference stars.** The chip transformation is a Chebyshev polynomial
   fitted to the reference stars themselves, so a reference star's VIRAC2 track is pulled towards
   its own Gaia track, Gaia's measurement error included. The size of that pull is not published.
   It is modelled as an *inheritance* fraction ``lam``: ``e_V = lam * e_G + eta`` with ``eta``
   independent of ``e_G``. The cross-covariance of the two errors is then ``X = lam * C_G``.
   ``lam = 0`` is independence; ``lam = 1`` with ``C_V = C_G + Cov(eta)`` means VIRAC2 adds nothing.
2. **A catalogue offset.** VIRAC2 minus Gaia is not zero on average, and in NGC 6383 it differs
   between reference and non-reference stars (measured in the hub finding
   ``agent-findings/virac2-gaia-2mass-calibration.md`` Sect. 2). It enters as ``offset``, which is
   subtracted from the VIRAC2 measurement before combining.

The published precedent combined preliminary VIRAC2 with Gaia EDR3 by an inverse-variance weighted
mean (Pena Ramirez et al. 2022, ``2022MNRAS.513.5799P``, Sect. 2.2), i.e. ``lam = 0`` and no offset.
That is the special case ``combine_proper_motions(..., cross=None, offset=None)``.

What the combination is, exactly: two unbiased measurements ``y_G = mu + e_G`` and
``y_V - offset = mu + e_V`` of the same true proper motion ``mu``, with joint error covariance
``S = [[C_G, X], [X^T, C_V]]``. The generalised-least-squares estimate is

    C_hat = (H^T S^-1 H)^-1,   mu_hat = C_hat H^T S^-1 y,   H = [I; I].

The difference ``d = (y_V - offset) - y_G`` has covariance ``C_G + C_V - X - X^T`` and does **not**
depend on ``mu``. For Gaussian errors ``mu_hat`` and ``d`` are independent, so for every population
whose density is a true-proper-motion density convolved with the per-star errors, the likelihood of
``(y_G, y_V)`` factorises into ``p(mu_hat | population) * p(d)``. ``p(d)`` is the same for every
population: it carries no membership information, but it is where the offset and the overall error
scale are constrained. A population written as a density of *observed* proper motions (a common
empirical field model) does not factorise this way, and mixing the two is the inconsistency to avoid.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "inheritance_cross_covariance",
    "combine_proper_motions",
    "difference_loglike",
]


def _as_stack(a: np.ndarray, d: int = 2) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim == 2 and a.shape == (d, d):
        a = a[None]
    if a.ndim != 3 or a.shape[1:] != (d, d):
        raise ValueError(f"expected (n, {d}, {d}) covariances, got shape {a.shape}")
    return a


def _as_vectors(a: np.ndarray, d: int = 2) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim == 1 and a.shape == (d,):
        a = a[None]
    if a.ndim != 2 or a.shape[1] != d:
        raise ValueError(f"expected (n, {d}) vectors, got shape {a.shape}")
    return a


def inheritance_cross_covariance(C_g: np.ndarray, lam: float | np.ndarray) -> np.ndarray:
    """Cross-covariance ``X = lam * C_G`` of ``e_V = lam * e_G + eta`` (``eta`` independent of ``e_G``).

    ``lam`` may be a scalar or one value per star (e.g. 0 for non-reference stars).
    """
    C_g = _as_stack(C_g)
    lam = np.broadcast_to(np.asarray(lam, dtype=float), (C_g.shape[0],))
    return lam[:, None, None] * C_g


def combine_proper_motions(
    mu_g: np.ndarray,
    C_g: np.ndarray,
    mu_v: np.ndarray,
    C_v: np.ndarray,
    *,
    cross: np.ndarray | None = None,
    offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GLS combination of two measurements of the same proper motion, per star.

    Parameters
    ----------
    mu_g, C_g : (n, 2), (n, 2, 2)
        First catalogue (Gaia) proper motion and its error covariance.
    mu_v, C_v : (n, 2), (n, 2, 2)
        Second catalogue (VIRAC2) proper motion and its error covariance, already inflated if an
        inflation is being applied.
    cross : (n, 2, 2), optional
        Cross-covariance ``E[e_G e_V^T]``. ``None`` means independent errors.
    offset : (2,) or (n, 2), optional
        Systematic offset of the second catalogue, subtracted from ``mu_v``.

    Returns
    -------
    mu_hat, C_hat, d, C_d
        Combined proper motion and covariance; the difference ``(mu_v - offset) - mu_g`` and its
        covariance ``C_g + C_v - X - X^T``.
    """
    mu_g, mu_v = _as_vectors(mu_g), _as_vectors(mu_v)
    C_g, C_v = _as_stack(C_g), _as_stack(C_v)
    n = mu_g.shape[0]
    if not (mu_v.shape[0] == C_g.shape[0] == C_v.shape[0] == n):
        raise ValueError("all inputs must describe the same number of stars")
    X = np.zeros_like(C_g) if cross is None else _as_stack(cross)
    if offset is not None:
        mu_v = mu_v - np.broadcast_to(np.asarray(offset, dtype=float), mu_v.shape)
    S = np.empty((n, 4, 4))
    S[:, :2, :2] = C_g
    S[:, :2, 2:] = X
    S[:, 2:, :2] = np.transpose(X, (0, 2, 1))
    S[:, 2:, 2:] = C_v
    Si = np.linalg.inv(S)
    # H^T S^-1 H = sum of the four 2x2 blocks; H^T S^-1 y = block-row sums applied to y
    A = Si[:, :2, :2] + Si[:, :2, 2:] + Si[:, 2:, :2] + Si[:, 2:, 2:]
    y = np.concatenate([mu_g, mu_v], axis=1)
    Siy = np.einsum("nij,nj->ni", Si, y)
    b = Siy[:, :2] + Siy[:, 2:]
    C_hat = np.linalg.inv(A)
    C_hat = 0.5 * (C_hat + np.transpose(C_hat, (0, 2, 1)))
    mu_hat = np.einsum("nij,nj->ni", C_hat, b)
    d = mu_v - mu_g
    C_d = C_g + C_v - X - np.transpose(X, (0, 2, 1))
    return mu_hat, C_hat, d, C_d


def difference_loglike(d: np.ndarray, C_d: np.ndarray) -> np.ndarray:
    """Per-star Gaussian log-density of the catalogue difference ``d ~ N(0, C_d)``.

    Membership-neutral (the same for every population); use it to constrain the offset and the
    error scale, which the combined proper motion alone cannot see.
    """
    d, C_d = _as_vectors(d), _as_stack(C_d)
    det = C_d[:, 0, 0] * C_d[:, 1, 1] - C_d[:, 0, 1] * C_d[:, 1, 0]
    q = (
        C_d[:, 1, 1] * d[:, 0] ** 2
        - (C_d[:, 0, 1] + C_d[:, 1, 0]) * d[:, 0] * d[:, 1]
        + C_d[:, 0, 0] * d[:, 1] ** 2
    ) / det
    return -0.5 * q - 0.5 * np.log(det) - np.log(2 * np.pi)
