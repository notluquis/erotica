r"""Minimum-spanning-tree mass-segregation diagnostics, null-calibrated.

This module implements the mass-segregation ratio :math:`\Lambda_{\rm MSR}` of
Allison et al. (2009) and its two published per-tree variants, and it reports
significance against the **exact permutation null** rather than against a null of
exactly 1.  The distinction is not cosmetic: see :ref:`the null is not 1
<null-not-one>` below.

Estimators
----------
All three statistics have the same shape.  Let :math:`T(\mathcal{S})` be a
statistic of the minimum spanning tree built on a set :math:`\mathcal{S}` of
:math:`N_{\rm MST}` projected positions, and let :math:`\mathcal{S}_1 \dots
\mathcal{S}_k` be :math:`k` random :math:`N_{\rm MST}`-star subsets of the
cluster.  Then

.. math::
    \Lambda = \frac{\langle T(\mathcal{S}_j) \rangle_j}{T(\mathcal{S}_{\rm massive})}

and the three published choices of :math:`T` are

===================  ==================================  ========================
``statistic``        :math:`T` = ...                     source
===================  ==================================  ========================
``"total"``          :math:`\sum_i e_i`, the total MST    Allison et al. (2009),
                     length.  This is the published       ``2009MNRAS.395.1449A``
                     :math:`\Lambda_{\rm MSR}`.           Eq. (1).
``"median_edge"``    :math:`\mathrm{median}_i(e_i)`,      Maschberger & Clarke
                     the median edge length,              (2011),
                     :math:`\tilde\Lambda`.               ``2011MNRAS.416..541M``
``"geometric_edge"`` :math:`(\prod_i e_i)^{1/n}`, the     Olczak, Spurzem &
                     geometric mean of the edges,         Henning (2011),
                     :math:`\gamma_{\rm MST}`; the ratio  ``2011A&A...532A.119O``
                     is their :math:`\Gamma_{\rm MST}`.   Eqs. (8) and (10).
===================  ==================================  ========================

with :math:`e_i` the :math:`n = N_{\rm MST} - 1` edge lengths of the tree.

**The variants act on the edges of one tree, not on the ensemble of random
lengths.**  Replacing the arithmetic mean *over the k random sets* by a median or
a geometric mean is a different operation from either cited paper, and it does
**not** deliver the outlier robustness they claim: the outlier failure mode
Maschberger & Clarke describe is a single long edge in the *massive* tree,
i.e. in the denominator, which no ensemble average in the numerator can repair.
Attributing the geometric-mean variant to Maschberger & Clarke is a further,
separate error -- it is Olczak et al. (2011).  Both mistakes appear in the
published literature and are the reason this table is here.

`[READ]` Maschberger & Clarke: *"It should be noted that the absolute values of
:math:`\bar\Lambda` and :math:`\tilde\Lambda` are not directly comparable"* -- so
"the signal is unchanged under the variant" is a statement about the **direction**
of the effect, never about the value.

.. _null-not-one:

The null is not 1, and it is available for free
-----------------------------------------------
:math:`\Lambda` is a :math:`k`-sample mean divided by a **single** draw
:math:`T(\mathcal{S}_{\rm massive})`.  By Jensen's inequality
:math:`\mathbb{E}[1/X] > 1/\mathbb{E}[X]`, so under the no-segregation null
:math:`\mathbb{E}[\Lambda] > 1`, with the bias growing as
:math:`\mathrm{Var}(T)/\mathbb{E}[T]^2` -- that is, as :math:`N_{\rm MST}` falls.
Any significance quoted as :math:`(\Lambda - 1)/\sigma_{\rm norm}` is therefore
measured from the wrong origin.

The fix costs nothing.  Under the null hypothesis actually being tested -- *stellar
mass is independent of position* -- the massive subset **is** a uniform random
:math:`N_{\rm MST}`-subset of the observed positions.  So the pool
:math:`\{T_j\}` that the estimator already computes for its own denominator *is*
the null distribution of :math:`T`.  Since :math:`x \mapsto c/x` is monotone
decreasing,

.. math::
    \mathrm{median}(\Lambda \mid H_0) &= \langle T_j\rangle \big/ \mathrm{median}(T_j) \\
    q_{1-\alpha}(\Lambda \mid H_0) &= \langle T_j\rangle \big/ q_{\alpha}(T_j) \\
    p_{\rm one-sided} &= \#\{ T_j \le T_{\rm massive} \} / k

all exactly, with no second Monte Carlo.  ``p_value`` in :class:`SegregationResult`
is that exact permutation p-value; it is what should be quoted.

Measured null (King profile at NGC 6383's :math:`C = 1.43`, :math:`R_c = 1.96'`,
:math:`R_t = 54'`, truncated at its 40 arcmin footprint, :math:`N_{\rm total} = 254`,
600 realisations x 100 draws; ``~/phd/agent-findings/scripts/lambda_msr_king_null.py``,
Monte Carlo errors in the raw output alongside it):

======  ===========  ========  ========
N_MST   null median  null p95  null p99
======  ===========  ========  ========
5       1.031        2.751     4.292
8       1.018        2.063     2.981
10      1.014        1.860     2.574
15      1.005        1.599     2.049
20      1.002        1.464     1.776
======  ===========  ========  ========

**Geometry moves the null far more than N does.**  At :math:`N_{\rm MST} = 5` the
same script measures a null median of **1.230** / p95 **2.732** for an *untruncated*
projected Plummer sphere, **1.058** / **2.111** for a Plummer truncated at the same
footprint, and **1.031** / **2.751** for the King profile -- while dropping
:math:`N_{\rm total}` from 254 to 61 moves the King null median only from 1.031 to
1.024.  The displacement is a property of the **tail of the spatial distribution**,
not of the membership count.  Do not import a null measured on some other cluster's
geometry; compute it from the sample in hand, which is what this module does.

The exact permutation null on the *real* NGC 6383 positions gives 1.024 / 2.59 / 4.46
at :math:`N_{\rm MST} = 5`, agreeing with the King parametric null to a few per cent.

Validity and warnings
---------------------
* Nothing published validates :math:`\Lambda_{\rm MSR}` below :math:`N_{\rm
  total} = 300` (Parker & Goodwin 2015, ``2015MNRAS.449.3381P``, chose
  :math:`N = 300` *"to match the small-N statistics of many young regions"*;
  Allison et al. used :math:`N \sim 900`--1000; Dib et al. 2018,
  ``2018MNRAS.473..849D``, imposed a floor of 40 members).  This module therefore
  warns rather than trusting the estimator blind.
* Olczak et al. recommend :math:`N_{\rm MST} = 10`--20 as the range that
  *"generally provides the clearest signature of mass segregation"*.
* :math:`N_{\rm MST} \gtrsim N_{\rm total}/2` compresses :math:`\Lambda` toward 1
  mechanically, because the massive subset and the reference subsets overlap.
* Parker & Goodwin recommend running :math:`\Sigma_{\rm LDR}` **in tandem**:
  *"they measure different definitions of 'mass segregation' and so should be used
  in tandem"*.  That statistic is not implemented here; the omission is deliberate
  and stated rather than silent.

Complexity
----------
:math:`O(N^2)` once for the pairwise distance matrix, then
:math:`O(k\,N_{\rm MST}^2)` for the reference pool.  **The MST of a subset is not
a subtree of the MST of the full set**, so a single global MST cannot be reused --
only the distance matrix can, and it is.  What *is* shared is more valuable: the
reference pool doubles as the null distribution (see above), so null calibration
is free instead of costing a second nested Monte Carlo.

References
----------
Allison, R. J., et al. 2009, MNRAS, 395, 1449 -- ``2009MNRAS.395.1449A``
Maschberger, Th., & Clarke, C. J. 2011, MNRAS, 416, 541 -- ``2011MNRAS.416..541M``
Olczak, C., Spurzem, R., & Henning, Th. 2011, A&A, 532, A119 -- ``2011A&A...532A.119O``
Parker, R. J., & Goodwin, S. P. 2015, MNRAS, 449, 3381 -- ``2015MNRAS.449.3381P``
Dib, S., Schmeja, S., & Parker, R. J. 2018, MNRAS, 473, 849 -- ``2018MNRAS.473..849D``

Prior art surveyed before writing this (2026-08-02): **ASteCA has no MST or
mass-segregation code at all** (GitHub code search over ``asteca/ASteCA`` for
"spanning" and "segregation" returns zero hits).  ``jakevdp/mst_clustering`` is a
scikit-learn MST *clustering* estimator, unrelated and unmaintained since 2016.
The reusable reference implementation is AMUSE's
``amuse.datamodel.particle_attributes.mass_segregation_ratio``, which is Allison
Eq. (1) verbatim with ``sigma_norm`` errors and ``number_of_random_sets=50``;
``MGJvanGroeningen/gaia_oc_amd`` has a Gaia-OC version using G magnitude as a mass
proxy and 20 random sets.  None of them calibrates the null, and none implements
the Maschberger & Clarke or Olczak variants.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from astropy import units as u
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm

from .units import quantity_values

__all__ = [
    "SegregationValidityWarning",
    "SegregationResult",
    "SegregationProfile",
    "lambda_msr",
    "lambda_msr_profile",
    "mst_edges",
]

class SegregationValidityWarning(UserWarning):
    """Raised when Lambda_MSR is being evaluated outside a regime anyone has validated.

    A dedicated class rather than a bare ``RuntimeWarning`` so a caller can filter
    exactly these -- and so that ``pytest.ini``'s ``error::RuntimeWarning`` rule,
    which exists to catch silent NaNs, is not conflated with a scientific caveat.
    """


Statistic = Literal["total", "median_edge", "geometric_edge"]

_STATISTIC_SOURCE = {
    "total": "Allison et al. 2009, 2009MNRAS.395.1449A, Eq. (1)",
    "median_edge": "Maschberger & Clarke 2011, 2011MNRAS.416..541M",
    "geometric_edge": "Olczak, Spurzem & Henning 2011, 2011A&A...532A.119O, Eqs. (8), (10)",
}

#: Relative deviation of the empirical null median from 1 above which the
#: ``(Lambda - 1)/sigma`` convention is actively misleading and a warning fires.
NULL_MEDIAN_TOLERANCE = 0.02

#: Smallest total membership any published study validated the estimator at
#: (Parker & Goodwin 2015).  Below this the estimator is extrapolated, not tested.
VALIDATED_N_TOTAL = 300


# --------------------------------------------------------------------------------------
# MST primitives
# --------------------------------------------------------------------------------------
def mst_edges(distances: np.ndarray) -> np.ndarray:
    """Edge lengths of the Euclidean minimum spanning tree of a dense distance matrix.

    Uses :func:`scipy.sparse.csgraph.minimum_spanning_tree` (Kruskal on the dense
    graph).  Verified against an independent Prim implementation and against
    hand-computed trees in ``tests/test_segregation.py``.

    Parameters
    ----------
    distances
        ``(k, k)`` symmetric non-negative distance matrix.

    Returns
    -------
    ndarray
        The ``k - 1`` edge lengths, unordered.  Empty for ``k <= 1``.
    """
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError(f"distances must be a square matrix, got {distances.shape}")
    if distances.shape[0] <= 1:
        return np.empty(0)
    tree = minimum_spanning_tree(distances).tocoo()
    return np.asarray(tree.data, dtype=float)


def _tree_statistic(edges: np.ndarray, statistic: Statistic) -> float:
    if statistic == "total":
        return float(edges.sum())
    if statistic == "median_edge":
        return float(np.median(edges))
    if statistic == "geometric_edge":
        # exp(mean(log e)) rather than prod(e)**(1/n): the product underflows for
        # k >~ 30 edges of sub-arcminute length, silently returning 0 and then inf.
        if np.any(edges <= 0):
            raise ValueError("geometric_edge is undefined for zero-length edges (duplicate positions)")
        return float(np.exp(np.mean(np.log(edges))))
    raise ValueError(f"unknown statistic {statistic!r}; expected one of {list(_STATISTIC_SOURCE)}")


def _subset_statistic(distances: np.ndarray, idx: np.ndarray, statistic: Statistic) -> float:
    return _tree_statistic(mst_edges(distances[np.ix_(idx, idx)]), statistic)


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SegregationResult:
    """One :math:`\\Lambda` measurement with its exact permutation calibration.

    Attributes
    ----------
    lam
        :math:`\\Lambda = \\langle T_j \\rangle / T_{\\rm massive}`.
    p_value
        **Exact one-sided permutation p-value** for ``H0: mass independent of
        position``, :math:`(\\#\\{T_j \\le T_{\\rm massive}\\} + 1)/(k + 1)`.  This
        is the number to quote.  Its Monte Carlo error is ``p_value_mcerr``.
    sigma_equivalent
        ``-norm.ppf(p_value)``: the one-sided Gaussian significance the p-value
        corresponds to.  Provided because referees think in sigmas; it is derived
        from ``p_value``, **not** from ``sigma_norm``.
    null_median, null_p95, null_p99
        Exact quantiles of :math:`\\Lambda` under the null for *these* positions.
        ``null_median`` differs from 1 in exactly the regime where the estimator
        misbehaves, so it doubles as a diagnostic.
    sigma_norm
        Allison et al.'s :math:`\\sigma_{\\rm norm}/T_{\\rm massive}`.  Retained for
        comparison with the literature and for reproducing published figures.
        **It is not an uncertainty on a significance** -- it describes the spread
        of the reference sets, and carries no information about the variance of
        the single massive-set draw in the denominator.
    lo_1sigma, hi_1sigma
        Parker & Goodwin (2015) percentile error bars: the 1/6 and 5/6 points of
        the ordered reference statistics, mapped through :math:`\\Lambda`.  They
        replaced :math:`\\sigma_{\\rm norm}` deliberately, *"to prevent a single
        outlying object from heavily influencing the uncertainty"*.
    warnings_raised
        Every validity warning emitted, so a caller can record them in a paper.
    """

    n_mst: int
    n_total: int
    statistic: Statistic
    lam: float
    p_value: float
    p_value_mcerr: float
    sigma_equivalent: float
    null_median: float
    null_p95: float
    null_p99: float
    sigma_norm: float
    lo_1sigma: float
    hi_1sigma: float
    t_massive: u.Quantity
    t_reference_mean: u.Quantity
    n_sets: int
    citation: str
    warnings_raised: tuple[str, ...] = ()
    reference_statistics: np.ndarray = field(default=None, repr=False, compare=False)

    @property
    def naive_sigma(self) -> float:
        """``(Lambda - 1)/sigma_norm`` -- the convention this module exists to replace.

        Exposed only so that a paper can report the difference between it and
        ``sigma_equivalent`` explicitly rather than silently switching conventions.
        """
        return (self.lam - 1.0) / self.sigma_norm

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"Lambda({self.statistic}, N_MST={self.n_mst}) = {self.lam:.2f} "
            f"[null median {self.null_median:.3f}, p95 {self.null_p95:.2f}]  "
            f"p = {self.p_value:.4g} ({self.sigma_equivalent:.2f} sigma); "
            f"naive (L-1)/sigma would read {self.naive_sigma:.2f} sigma"
        )


@dataclass(frozen=True)
class SegregationProfile:
    """A grid of :math:`N_{\\rm MST}` values with the look-elsewhere correction applied.

    ``best_local_p`` is the smallest p-value over the grid and is **not** a valid
    significance on its own: scanning :math:`N_{\\rm MST}` and reporting the best
    point is a multiple-comparisons problem.  ``global_p`` is the corrected value,
    obtained by simulating the whole scan under the null (so the strong correlation
    between neighbouring :math:`N_{\\rm MST}` is accounted for, and the trials
    factor comes out well below the naive number of looks).
    """

    results: dict[int, SegregationResult]
    best_n_mst: int
    best_local_p: float
    global_p: float
    global_p_mcerr: float
    global_sigma_equivalent: float
    trials_factor: float
    n_permutations: int

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"best local p = {self.best_local_p:.4g} at N_MST = {self.best_n_mst}; "
            f"global p = {self.global_p:.4g} +- {self.global_p_mcerr:.4g} "
            f"({self.global_sigma_equivalent:.2f} sigma) after a trials factor of "
            f"{self.trials_factor:.1f} over {len(self.results)} looks"
        )


# --------------------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------------------
def _as_positions(positions: Any) -> tuple[np.ndarray, u.UnitBase]:
    """Return ``(N, 2)`` float positions and the unit they are in.

    A bare array is **arcmin**, the package-wide convention for angular radii
    (see ``erotica/analysis/CLAUDE.md``).  A ``Quantity`` is converted, and a
    ``Quantity`` of the wrong physical type raises ``UnitConversionError`` rather
    than being silently stripped -- the failure mode that cost this package a
    480x error once already.
    """
    if hasattr(positions, "unit"):
        unit = u.arcmin if positions.unit.physical_type == "angle" else positions.unit
        values = quantity_values(positions, unit)
    else:
        unit = u.arcmin
        values = np.asarray(positions, dtype=float)
    values = np.atleast_2d(values)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"positions must be (N, 2) projected coordinates, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("positions contain non-finite values; filter them before calling")
    return values, unit


def _mass_order(masses: Any, n_total: int) -> np.ndarray:
    values = quantity_values(masses, u.Msun if hasattr(masses, "unit") else None)
    values = np.asarray(values, dtype=float).ravel()
    if values.size != n_total:
        raise ValueError(f"masses has {values.size} entries but positions has {n_total}")
    if not np.all(np.isfinite(values)):
        raise ValueError("masses contain non-finite values; filter them before calling")
    return np.argsort(values)[::-1]


def _random_subsets(rng: np.random.Generator, n_total: int, n_mst: int, n_sets: int) -> np.ndarray:
    """``(n_sets, n_mst)`` uniform subsets without replacement.

    ``argsort`` of a uniform row is a uniform random permutation, so its first
    ``n_mst`` entries are a uniform random subset.  Chunk-free: ``n_sets`` here is
    the user's ``n_sets``, which is thousands, not the millions the calibration
    study in ``~/phd/agent-findings/scripts/`` needs.
    """
    return np.argsort(rng.random((n_sets, n_total)), axis=1)[:, :n_mst]


def _check_validity(n_total: int, n_mst: int, n_sets: int, null_median: float,
                    strict: bool) -> tuple[str, ...]:
    msgs: list[str] = []
    if abs(null_median - 1.0) > NULL_MEDIAN_TOLERANCE:
        msgs.append(
            f"the empirical null median of Lambda is {null_median:.3f}, not 1 "
            f"(N_MST={n_mst}): quote p_value/sigma_equivalent, never (Lambda-1)/sigma_norm"
        )
    if n_total < VALIDATED_N_TOTAL:
        msgs.append(
            f"N_total={n_total} is below the smallest membership any published study "
            f"validated Lambda_MSR at (N={VALIDATED_N_TOTAL}, Parker & Goodwin 2015); "
            "the exact permutation p-value remains valid, the literature calibration does not"
        )
    if n_mst < 10:
        msgs.append(
            f"N_MST={n_mst} is below the 10-20 range Olczak et al. (2011) recommend as "
            "giving the clearest signature; the null is strongly right-skewed here"
        )
    if n_mst > n_total / 2:
        msgs.append(
            f"N_MST={n_mst} exceeds half of N_total={n_total}: the massive and reference "
            "subsets overlap heavily and Lambda is mechanically compressed toward 1"
        )
    if n_sets < 500:
        msgs.append(
            f"n_sets={n_sets} is below the 500-1000 Allison et al. (2009) recommend at low "
            "N_MST, and it bounds the smallest resolvable p-value at 1/(n_sets+1)"
        )
    if strict and msgs:
        raise ValueError("; ".join(msgs))
    for m in msgs:
        warnings.warn(m, SegregationValidityWarning, stacklevel=3)
    return tuple(msgs)


# --------------------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------------------
def lambda_msr(
    positions: Any,
    masses: Any,
    n_mst: int = 10,
    *,
    statistic: Statistic = "total",
    n_sets: int = 1000,
    rng: np.random.Generator | int | None = None,
    strict: bool = False,
    _distances: np.ndarray | None = None,
) -> SegregationResult:
    r"""Mass-segregation ratio with an exact permutation p-value.

    Parameters
    ----------
    positions
        ``(N, 2)`` projected positions.  A bare array is **arcmin**; a
        ``Quantity`` is converted (angles to arcmin, lengths kept).
    masses
        ``(N,)`` stellar masses; only their **rank order** is used.
    n_mst
        Size of the "most massive" subset, :math:`N_{\rm MST}`.
    statistic
        ``"total"`` (Allison), ``"median_edge"`` (Maschberger & Clarke) or
        ``"geometric_edge"`` (Olczak).  See the module docstring for which paper
        each belongs to -- the geometric variant is **not** Maschberger & Clarke.
    n_sets
        Number of random reference subsets.  These double as the null sample, so
        this also sets the p-value resolution: the smallest attainable p-value is
        ``1/(n_sets + 1)``.
    rng
        Seed or :class:`numpy.random.Generator`.
    strict
        Raise instead of warning when a validity check fails.

    Returns
    -------
    SegregationResult

    Notes
    -----
    Ranking by a mass that itself depends on the hypothesis under test (e.g. a
    total mass ``m1 + m2`` conditioned on a binary probability, when the claim is
    that binaries are centrally concentrated) makes the ranking and the hypothesis
    non-independent.  Rank by primary mass as a cross-check when that applies.
    """
    pts, unit = _as_positions(positions)
    n_total = len(pts)
    if not 2 <= n_mst < n_total:
        raise ValueError(f"n_mst must satisfy 2 <= n_mst < N_total={n_total}, got {n_mst}")
    order = _mass_order(masses, n_total)
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    distances = squareform(pdist(pts)) if _distances is None else _distances

    t_massive = _subset_statistic(distances, order[:n_mst], statistic)
    if not t_massive > 0:
        raise ValueError("the massive-subset MST has zero length (coincident positions)")

    subsets = _random_subsets(generator, n_total, n_mst, n_sets)
    pool = np.array([_subset_statistic(distances, s, statistic) for s in subsets])

    mean_ref = float(pool.mean())
    lam = mean_ref / t_massive
    # The exact permutation identities.  Because Lambda = mean_ref / T and
    # x -> mean_ref/x is decreasing, upper quantiles of Lambda come from LOWER
    # quantiles of the pool, and P(Lambda >= Lambda_obs) = P(T <= T_obs).
    null_median = mean_ref / float(np.median(pool))
    null_p95 = mean_ref / float(np.quantile(pool, 0.05))
    null_p99 = mean_ref / float(np.quantile(pool, 0.01))
    p_value = float((np.sum(pool <= t_massive) + 1) / (n_sets + 1))
    p_mcerr = float(np.sqrt(max(p_value, 1.0 / n_sets) * (1 - p_value) / n_sets))

    msgs = _check_validity(n_total, n_mst, n_sets, null_median, strict)

    return SegregationResult(
        n_mst=n_mst,
        n_total=n_total,
        statistic=statistic,
        lam=float(lam),
        p_value=p_value,
        p_value_mcerr=p_mcerr,
        sigma_equivalent=float(-norm.ppf(p_value)),
        null_median=null_median,
        null_p95=null_p95,
        null_p99=null_p99,
        sigma_norm=float(pool.std() / t_massive),
        # Parker & Goodwin (2015): the 1/6 and 5/6 points of the ordered reference
        # lengths.  Mapped through Lambda = mean_ref/T the order INVERTS -- the lower
        # bound on Lambda comes from the 5/6 point of T.
        lo_1sigma=float(mean_ref / np.quantile(pool, 5.0 / 6.0)),
        hi_1sigma=float(mean_ref / np.quantile(pool, 1.0 / 6.0)),
        t_massive=t_massive * unit,
        t_reference_mean=mean_ref * unit,
        n_sets=n_sets,
        citation=_STATISTIC_SOURCE[statistic],
        warnings_raised=msgs,
        reference_statistics=pool,
    )


def lambda_msr_profile(
    positions: Any,
    masses: Any,
    n_mst_grid: Sequence[int] = (5, 10, 15, 20, 30, 50),
    *,
    statistic: Statistic = "total",
    n_sets: int = 1000,
    n_permutations: int = 20_000,
    rng: np.random.Generator | int | None = None,
    strict: bool = False,
) -> SegregationProfile:
    r"""Run :func:`lambda_msr` over a grid and apply the look-elsewhere correction.

    Reporting the largest :math:`\Lambda` from a scan over :math:`N_{\rm MST}` and
    quoting its local p-value overstates the significance.  The trials factor is
    **not** the number of grid points, because the looks are strongly correlated
    (the subsets are nested).  It is measured here: ``n_permutations`` random mass
    orderings are drawn, the whole scan is repeated on each, and

    .. math::
        p_{\rm global} = \Pr\big[\min_k p_{\rm local}(k) \le p_{\rm local}^{\rm obs}\big].

    The pairwise distance matrix is computed once and shared across the entire
    grid and every permutation.

    Complexity: :math:`O(N^2) + O\big((k + n_{\rm perm}) \sum_g N_{{\rm MST},g}^2\big)`.
    """
    pts, _ = _as_positions(positions)
    n_total = len(pts)
    order = _mass_order(masses, n_total)
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    distances = squareform(pdist(pts))
    grid = [int(n) for n in n_mst_grid]

    results: dict[int, SegregationResult] = {}
    sorted_pools: dict[int, np.ndarray] = {}
    for n in grid:
        res = lambda_msr(pts, masses, n, statistic=statistic, n_sets=n_sets,
                         rng=generator, strict=strict, _distances=distances)
        results[n] = res
        sorted_pools[n] = np.sort(res.reference_statistics)

    def local_p(n: int, t: np.ndarray | float) -> np.ndarray:
        pool = sorted_pools[n]
        return (np.searchsorted(pool, t, side="right") + 1) / (len(pool) + 1)

    obs_local = {n: float(local_p(n, _subset_statistic(distances, order[:n], statistic)))
                 for n in grid}
    best_n = min(obs_local, key=obs_local.get)
    best_p = obs_local[best_n]

    # Null scan.  The permuted orderings must be NESTED across the grid exactly as
    # the observed mass ranking is -- independent draws per grid point would
    # destroy the correlation that makes the trials factor smaller than len(grid).
    min_p = np.empty(n_permutations)
    chunk = 5_000
    for lo in range(0, n_permutations, chunk):
        hi = min(lo + chunk, n_permutations)
        perms = np.argsort(generator.random((hi - lo, n_total)), axis=1)
        block = np.ones(hi - lo)
        for n in grid:
            stats = np.array([_subset_statistic(distances, p[:n], statistic) for p in perms])
            block = np.minimum(block, local_p(n, stats))
        min_p[lo:hi] = block

    global_p = float((np.sum(min_p <= best_p) + 1) / (n_permutations + 1))
    return SegregationProfile(
        results=results,
        best_n_mst=best_n,
        best_local_p=best_p,
        global_p=global_p,
        global_p_mcerr=float(np.sqrt(global_p * (1 - global_p) / n_permutations)),
        global_sigma_equivalent=float(-norm.ppf(global_p)),
        trials_factor=float(global_p / best_p),
        n_permutations=n_permutations,
    )
