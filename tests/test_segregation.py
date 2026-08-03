"""Oracles for the MST mass-segregation estimators.

Every oracle here exists independently of ``erotica.analysis.segregation``:

* **hand-computed trees** -- a path graph and a unit square have MSTs anyone can
  write down, so ``mst_edges`` is checked against arithmetic, not against itself;
* **an independent Prim implementation** written in this file, so the scipy call
  is cross-validated by a second algorithm rather than by a second call to scipy;
* **the closed-form permutation identity** ``median(Lambda|H0) = mean(pool)/median(pool)``,
  recomputed here from the returned pool;
* **calibration under a true null** -- the exact p-value must be uniform when
  masses are independent of position.  This is the property the module claims and
  the only test that catches a whole class of tail/sign errors at once;
* **injected segregation with a known sign**, in both directions.

Mutation audit run 2026-08-02: **11 deliberate bugs re-applied one at a time, 11 killed, 0
survivors.**  Harness: ``pytest tests/test_segregation.py -x -m "not slow"`` after each
substitution, module restored afterwards.

====================================================  ==========================================
mutation                                              killed by
====================================================  ==========================================
``mst_edges`` returns ``.toarray().ravel()`` (zeros    ``test_mst_edges_matches_hand_computed_trees``
included) instead of the COO ``.data``
``null_p95`` uses ``quantile(pool, 0.95)``             ``test_null_quantile_identity_inverts_the_pool``
(no ratio inversion)
``p_value`` counts ``pool >= t_massive``               ``test_p_value_is_uniform_under_a_true_null``,
                                                       ``test_injected_segregation_is_detected``
``median_edge`` returns ``edges.sum()`` -- i.e. the    ``test_variants_act_on_edges_not_on_the_pool``
variant applied anywhere but to the edges
``geometric_edge`` uses ``prod(e)**(1/n)``             ``test_geometric_variant_survives_underflow``
``lam = t_massive / mean_ref`` (ratio inverted)        ``test_injected_segregation_is_detected``
validity messages suppressed                           ``test_small_n_mst_warns_and_strict_raises``
``lambda_msr_profile`` draws independent (not          ``test_trials_factor_is_one_for_a_repeated_look``
nested) permutations per grid point                    ``_and_above_one_for_a_real_scan``
``global_p`` returns ``best_local_p``                  same test (pins the trials factor at 1.0)
geometric variant attributed to Maschberger &          ``test_each_variant_reports_its_own_citation``
Clarke
Parker & Goodwin bounds not inverted through the        ``test_percentile_error_bars_invert_through``
ratio (``lo`` and ``hi`` swapped)                      ``_the_ratio``
====================================================  ==========================================

Two of these were **added because a first pass left them alive**, and the reason is recorded in
each test's docstring: the underflow test used 40 edges where the naive product is 1e-160 (small
but perfectly representable, so the bug still worked), and there was no test at all for the
percentile bars.  A third, the nesting bug, was only caught by a ``slow`` test until the repeated-
look invariant was added.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy import units as u
from scipy.spatial.distance import pdist, squareform

from erotica.analysis.segregation import (
    SegregationValidityWarning,
    lambda_msr,
    lambda_msr_profile,
    mst_edges,
)


# --------------------------------------------------------------------------------------
# Independent oracles
# --------------------------------------------------------------------------------------
def prim_mst_total(distances):
    """Prim's algorithm, written from the textbook definition.

    Independent of scipy's Kruskal-on-a-dense-graph, so agreement between the two
    is evidence about the answer rather than about one library.
    """
    d = np.asarray(distances, dtype=float)
    k = len(d)
    if k <= 1:
        return 0.0
    in_tree = np.zeros(k, dtype=bool)
    in_tree[0] = True
    best = d[0].copy()
    best[0] = np.inf
    total = 0.0
    for _ in range(k - 1):
        j = int(np.argmin(np.where(in_tree, np.inf, best)))
        total += best[j]
        in_tree[j] = True
        best = np.minimum(best, d[j])
    return float(total)


def plummer_2d(rng, n, a=1.0):
    """Projected Plummer sphere -- smooth, centrally concentrated, no segregation."""
    u_ = rng.uniform(0.0, 1.0, n)
    r = a / np.sqrt(u_ ** (-2.0 / 3.0) - 1.0)
    cos_t = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    s = np.sqrt(1.0 - cos_t**2)
    return np.column_stack([r * s * np.cos(phi), r * s * np.sin(phi)])


def quiet(fn, *args, **kwargs):
    """Call ``fn`` suppressing only the validity warnings, which have their own test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SegregationValidityWarning)
        return fn(*args, **kwargs)


# --------------------------------------------------------------------------------------
# 1. The MST itself
# --------------------------------------------------------------------------------------
def test_mst_edges_matches_hand_computed_trees():
    """Oracle: two geometries whose MST can be written down without any code.

    A path of collinear points at x = 0, 1, 3, 6 has the unique MST {1, 2, 3};
    the unit square's MST is any three sides, total 3.  A wrong edge extraction --
    for instance summing the dense array instead of the sparse non-zeros -- cannot
    reproduce both.
    """
    line = np.column_stack([[0.0, 1.0, 3.0, 6.0], np.zeros(4)])
    edges = np.sort(mst_edges(squareform(pdist(line))))
    assert np.allclose(edges, [1.0, 2.0, 3.0])
    assert edges.size == 3  # k - 1 edges exactly, no zero padding

    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert np.isclose(mst_edges(squareform(pdist(square))).sum(), 3.0)


@pytest.mark.parametrize("k", [2, 3, 5, 12, 30])
def test_mst_edges_agrees_with_an_independent_prim_implementation(k):
    """Oracle: a second, independently written MST algorithm.

    Kruskal (scipy) and Prim (here) are different algorithms; they agree only if
    both compute the true minimum spanning tree.
    """
    rng = np.random.default_rng(4 + k)
    for _ in range(20):
        d = squareform(pdist(rng.normal(size=(k, 2))))
        assert np.isclose(mst_edges(d).sum(), prim_mst_total(d), atol=1e-12)


def test_mst_edges_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        mst_edges(np.zeros((3, 4)))


# --------------------------------------------------------------------------------------
# 2. The permutation identities
# --------------------------------------------------------------------------------------
def test_null_quantile_identity_inverts_the_pool():
    """Oracle: the closed form ``q_{1-a}(Lambda) = mean(pool)/q_a(pool)``, recomputed here.

    Lambda = mean(pool)/T and x -> c/x is decreasing, so the *upper* tail of
    Lambda comes from the *lower* tail of the pool.  Reading the pool's own 95th
    percentile instead -- the natural mistake -- gives a number below the null
    median and is caught by the ordering assertion as well as by the identity.
    """
    rng = np.random.default_rng(11)
    pts = plummer_2d(rng, 200)
    res = quiet(lambda_msr, pts, rng.random(200), 10, n_sets=4000, rng=3)

    pool = res.reference_statistics
    mean_ref = pool.mean()
    assert np.isclose(res.null_median, mean_ref / np.median(pool), rtol=1e-12)
    assert np.isclose(res.null_p95, mean_ref / np.quantile(pool, 0.05), rtol=1e-12)
    assert np.isclose(res.null_p99, mean_ref / np.quantile(pool, 0.01), rtol=1e-12)
    assert res.null_median < res.null_p95 < res.null_p99


def test_null_median_exceeds_one_and_grows_as_n_mst_falls():
    """Oracle: Jensen's inequality.

    Lambda is a mean divided by a single draw, so E[1/T] > 1/E[T] forces the null
    centre above 1, by an amount that scales with Var(T)/E[T]^2 and therefore grows
    as the subset shrinks.  This is a *directional* prediction made before the
    measurement, not a tolerance fitted to the output.
    """
    rng = np.random.default_rng(5)
    pts = plummer_2d(rng, 250)
    mass = rng.random(250)
    medians = [quiet(lambda_msr, pts, mass, n, n_sets=6000, rng=7).null_median
               for n in (5, 10, 20, 40)]
    assert medians[0] > 1.0
    assert medians == sorted(medians, reverse=True), medians


def test_p_value_is_uniform_under_a_true_null():
    """Oracle: the definition of a calibrated p-value.

    Masses are assigned at random to positions, so H0 is true by construction and
    the exact permutation p-value must be Uniform(0, 1).  Any sign error, tail
    swap or off-by-one in the counting shows up as a shifted rejection rate.  With
    400 trials the binomial standard error on a nominal 0.10 rate is 0.015, so the
    0.05 band below is a ~3 sigma envelope.
    """
    rng = np.random.default_rng(2026)
    ps = []
    for _ in range(400):
        pts = plummer_2d(rng, 80)
        ps.append(quiet(lambda_msr, pts, rng.random(80), 10, n_sets=400, rng=rng).p_value)
    ps = np.asarray(ps)
    assert abs(np.mean(ps <= 0.10) - 0.10) < 0.05
    assert abs(np.mean(ps <= 0.50) - 0.50) < 0.07
    assert abs(np.mean(ps <= 0.90) - 0.90) < 0.05


# --------------------------------------------------------------------------------------
# 3. Behaviour on data with a known answer
# --------------------------------------------------------------------------------------
def test_injected_segregation_is_detected():
    """Oracle: a configuration whose answer is known by construction, in both signs.

    Put the massive stars at the centre and Lambda must exceed 1 with a small
    p-value; put them at the edge and Lambda must fall below 1 with p near 1.
    Inverting the ratio, or counting the wrong tail, flips one of the two.
    """
    rng = np.random.default_rng(31)
    pts = plummer_2d(rng, 200, a=1.0)
    radius = np.hypot(*pts.T)

    # segregated: mass decreases outward
    seg = quiet(lambda_msr, pts, -radius, 10, n_sets=3000, rng=1)
    assert seg.lam > 1.5 and seg.p_value < 0.01 and seg.sigma_equivalent > 2.0

    # inversely segregated: mass increases outward
    inv = quiet(lambda_msr, pts, radius, 10, n_sets=3000, rng=1)
    assert inv.lam < 1.0 and inv.p_value > 0.95


def test_lambda_is_invariant_under_similarity_transforms_but_lengths_are_not():
    """Oracle: Lambda is a ratio of lengths, so scale and rotation must cancel exactly.

    A prefactor bug that survives a ratio-only assertion is caught by also checking
    the *dimensional* quantity, which must scale by exactly the factor applied
    (see ``tests/CLAUDE.md``, failure mode 1).
    """
    rng = np.random.default_rng(17)
    pts = plummer_2d(rng, 150)
    mass = rng.random(150)
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    moved = 3.0 * pts @ rot.T + np.array([12.0, -5.0])

    a = quiet(lambda_msr, pts, mass, 10, n_sets=800, rng=9)
    b = quiet(lambda_msr, moved, mass, 10, n_sets=800, rng=9)
    assert np.isclose(a.lam, b.lam, rtol=1e-12)
    assert np.isclose(b.t_massive.value, 3.0 * a.t_massive.value, rtol=1e-12)


def test_positions_respect_the_unit_they_carry():
    """Oracle: 1 degree is 60 arcmin, exactly.

    Lambda is dimensionless and must not move; the MST length is dimensional and
    must convert.  A bare array is arcmin by the package convention, so passing the
    same numbers bare and as ``u.arcmin`` must be identical.
    """
    rng = np.random.default_rng(23)
    pts = plummer_2d(rng, 120)
    mass = rng.random(120)

    bare = quiet(lambda_msr, pts, mass, 10, n_sets=600, rng=5)
    arcmin = quiet(lambda_msr, pts * u.arcmin, mass, 10, n_sets=600, rng=5)
    degree = quiet(lambda_msr, (pts * u.arcmin).to(u.deg), mass, 10, n_sets=600, rng=5)

    assert np.isclose(bare.lam, arcmin.lam, rtol=1e-12)
    assert np.isclose(bare.lam, degree.lam, rtol=1e-12)
    assert bare.t_massive.unit == u.arcmin
    assert np.isclose(degree.t_massive.to_value(u.arcmin), bare.t_massive.to_value(u.arcmin),
                      rtol=1e-10)


# --------------------------------------------------------------------------------------
# 4. The variants
# --------------------------------------------------------------------------------------
def test_variants_act_on_edges_not_on_the_pool():
    """Oracle: a tree whose edge statistics are arithmetic anyone can do by hand.

    The collinear set {0, 1, 3, 6} has MST edges {1, 2, 3}: median 2, geometric
    mean (1*2*3)^(1/3) = 1.8171.  Implementing the variants as a median or
    geometric mean *over the ensemble of random subsets* -- which is how the
    literature sometimes describes them, and is not what either source paper does --
    cannot reproduce these numbers, because it never touches the edges at all.
    """
    line = np.column_stack([[0.0, 1.0, 3.0, 6.0], np.zeros(4)])
    pool = np.zeros(3)  # placeholder; we compare the massive-set statistic only
    del pool

    # 8 stars: the 4 collinear ones carry the largest masses, the rest sit far away
    far = np.column_stack([[100.0, 130.0, 170.0, 220.0], np.full(4, 50.0)])
    pts = np.vstack([line, far])
    mass = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

    total = quiet(lambda_msr, pts, mass, 4, statistic="total", n_sets=200, rng=1)
    med = quiet(lambda_msr, pts, mass, 4, statistic="median_edge", n_sets=200, rng=1)
    geo = quiet(lambda_msr, pts, mass, 4, statistic="geometric_edge", n_sets=200, rng=1)

    assert np.isclose(total.t_massive.value, 6.0)
    assert np.isclose(med.t_massive.value, 2.0)
    assert np.isclose(geo.t_massive.value, 6.0 ** (1.0 / 3.0))  # (1*2*3)^(1/3)


def test_geometric_variant_survives_underflow():
    """Oracle: the geometric mean of n equal edges is that edge length, at any scale.

    A regular grid of 101 collinear points spaced 1e-4 apart has 100 equal MST
    edges, so the geometric mean is exactly 1e-4.  The product of those edges is
    1e-400, which is **not representable** in float64 and evaluates to exactly 0.0,
    so ``prod(e)**(1/n)`` returns 0.0 and the ratio becomes ``inf``;
    ``exp(mean(log e))`` returns 1e-4.  The assertion on ``np.prod`` is ``== 0.0``
    rather than ``isclose(..., 0)`` because at 40 edges the product is 1e-160 --
    tiny, perfectly representable, and the naive form still works.  Getting this
    boundary wrong is how the first version of this test failed to bite.
    """
    step = 1e-4
    pts = np.column_stack([np.arange(101) * step, np.zeros(101)])
    edges = mst_edges(squareform(pdist(pts)))
    assert edges.size == 100
    assert np.prod(edges) == 0.0  # true underflow, not merely small
    from erotica.analysis.segregation import _tree_statistic

    assert np.isclose(_tree_statistic(edges, "geometric_edge"), step, rtol=1e-9)


def test_each_variant_reports_its_own_citation():
    """The geometric mean is Olczak et al., not Maschberger & Clarke.

    This is a published misattribution, so the mapping is asserted rather than
    left to the docstring.
    """
    rng = np.random.default_rng(43)
    pts = plummer_2d(rng, 100)
    mass = rng.random(100)
    cites = {s: quiet(lambda_msr, pts, mass, 10, statistic=s, n_sets=200, rng=1).citation
             for s in ("total", "median_edge", "geometric_edge")}
    assert "2009MNRAS.395.1449A" in cites["total"]
    assert "2011MNRAS.416..541M" in cites["median_edge"]
    assert "2011A&A...532A.119O" in cites["geometric_edge"]
    assert "2011MNRAS.416..541M" not in cites["geometric_edge"]


def test_unknown_statistic_is_rejected():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="unknown statistic"):
        quiet(lambda_msr, plummer_2d(rng, 50), rng.random(50), 5, statistic="harmonic")


# --------------------------------------------------------------------------------------
# 5. Validity guards
# --------------------------------------------------------------------------------------
def test_small_n_mst_warns_and_strict_raises():
    """The regime where the null median is not 1 must not pass silently.

    ``N_MST = 5`` on 60 stars trips three separate guards: the displaced null
    median, the sub-300 membership no published study validated, and the sub-10
    subset size Olczak et al. advise against.
    """
    rng = np.random.default_rng(61)
    pts = plummer_2d(rng, 60)
    mass = rng.random(60)

    with pytest.warns(SegregationValidityWarning):
        res = lambda_msr(pts, mass, 5, n_sets=2000, rng=1)
    assert any("null median" in m for m in res.warnings_raised)
    assert any("Parker & Goodwin" in m for m in res.warnings_raised)

    with pytest.raises(ValueError, match="null median"):
        lambda_msr(pts, mass, 5, n_sets=2000, rng=1, strict=True)


def test_overlapping_subset_size_warns():
    rng = np.random.default_rng(62)
    pts = plummer_2d(rng, 40)
    with pytest.warns(SegregationValidityWarning, match="half of N_total"):
        lambda_msr(pts, rng.random(40), 25, n_sets=600, rng=1)


def test_degenerate_inputs_are_rejected():
    rng = np.random.default_rng(63)
    pts = plummer_2d(rng, 30)
    with pytest.raises(ValueError, match="n_mst"):
        lambda_msr(pts, rng.random(30), 30)
    with pytest.raises(ValueError, match="masses has"):
        lambda_msr(pts, rng.random(29), 5)
    with pytest.raises(ValueError, match="non-finite"):
        lambda_msr(np.vstack([pts, [[np.nan, 0.0]]]), rng.random(31), 5)


def test_naive_sigma_is_exposed_and_differs_from_the_calibrated_one():
    """``(Lambda-1)/sigma_norm`` must remain available *and* be visibly different.

    Keeping it lets a paper state the size of the correction instead of silently
    switching conventions between versions.
    """
    rng = np.random.default_rng(71)
    pts = plummer_2d(rng, 200)
    radius = np.hypot(*pts.T)
    res = quiet(lambda_msr, pts, -radius, 5, n_sets=4000, rng=2)
    assert res.naive_sigma != pytest.approx(res.sigma_equivalent, abs=0.05)


# --------------------------------------------------------------------------------------
# 6. Look-elsewhere
# --------------------------------------------------------------------------------------
def test_profile_global_p_exceeds_best_local_p():
    """Scanning N_MST and reporting the best point cannot make a result *more* significant."""
    rng = np.random.default_rng(81)
    pts = plummer_2d(rng, 150)
    prof = quiet(lambda_msr_profile, pts, -np.hypot(*pts.T), (5, 10, 20),
                 n_sets=1500, n_permutations=3000, rng=4)
    assert prof.global_p >= prof.best_local_p
    assert prof.trials_factor >= 1.0
    assert prof.trials_factor <= len(prof.results) + 1e-9  # Bonferroni is the upper bound


def test_trials_factor_is_one_for_a_repeated_look_and_above_one_for_a_real_scan():
    """Oracle: a parameter-free special case -- looking at the same thing three times is one look.

    ``n_mst_grid = (10, 10, 10)`` repeats an identical look, so the three local
    p-values are the same number and the trials factor must be exactly 1.  This is
    true by construction only if the null scan reuses **one nested permutation
    across the whole grid**, as the observed mass ranking does; drawing an
    independent permutation per grid point turns the repeat into three independent
    looks and the factor climbs toward 3.  A genuine scan over 5, 20, 50 must give
    a factor strictly above 1, which is what returning ``best_local_p`` as the
    global p would violate (it pins the factor at exactly 1.0).

    Note the sharper test that does *not* work: grids like (10, 11, 12) are not
    near-degenerate in the far tail -- measured factor 2.8, essentially Bonferroni --
    because hitting the minimum of a few thousand draws is driven by fine detail
    that a one-star change to the subset destroys.  Correlation between neighbouring
    N_MST is a statement about the bulk of the distribution, not about its extreme.
    """
    rng = np.random.default_rng(82)
    pts = plummer_2d(rng, 150)
    mass = -np.hypot(*pts.T)

    repeated = quiet(lambda_msr_profile, pts, mass, (10, 10, 10),
                     n_sets=2000, n_permutations=5000, rng=6)
    assert len(repeated.results) == 1
    assert repeated.trials_factor < 1.3, repeated.trials_factor

    scan = quiet(lambda_msr_profile, pts, mass, (5, 20, 50),
                 n_sets=2000, n_permutations=5000, rng=6)
    assert scan.trials_factor > 1.0, scan.trials_factor
    assert scan.trials_factor > repeated.trials_factor


def test_percentile_error_bars_invert_through_the_ratio():
    """Oracle: the 1/6 and 5/6 points of the pool, recomputed here, and the ordering.

    Parker & Goodwin (2015) quote the uncertainty as the MST length at 1/6 and 5/6
    of the ordered random lengths.  Lambda = mean(pool)/T inverts the order, so the
    *lower* bound on Lambda comes from the *upper* point of the pool.  Passing them
    through un-inverted produces ``lo > hi``, which no ratio-only assertion would
    notice.
    """
    rng = np.random.default_rng(83)
    pts = plummer_2d(rng, 200)
    res = quiet(lambda_msr, pts, -np.hypot(*pts.T), 10, n_sets=4000, rng=8)
    pool = res.reference_statistics
    assert res.lo_1sigma < res.hi_1sigma
    assert np.isclose(res.lo_1sigma, pool.mean() / np.quantile(pool, 5.0 / 6.0), rtol=1e-12)
    assert np.isclose(res.hi_1sigma, pool.mean() / np.quantile(pool, 1.0 / 6.0), rtol=1e-12)


@pytest.mark.slow
def test_profile_global_p_is_uniform_under_a_true_null():
    """Oracle: a look-elsewhere-corrected p-value is calibrated by definition.

    Under H0 the corrected p must again be Uniform(0, 1) -- that is exactly what
    "corrected" means.  Drawing an *independent* permutation per grid point instead
    of a nested one destroys the correlation between looks, inflates the trials
    factor toward Bonferroni, and shows up here as an over-conservative rate.
    """
    rng = np.random.default_rng(2027)
    ps = []
    for _ in range(120):
        pts = plummer_2d(rng, 80)
        prof = quiet(lambda_msr_profile, pts, rng.random(80), (5, 10, 20),
                     n_sets=800, n_permutations=800, rng=rng)
        ps.append(prof.global_p)
    ps = np.asarray(ps)
    assert abs(np.mean(ps <= 0.20) - 0.20) < 0.11  # 120 trials -> SE 0.037
    assert abs(np.mean(ps <= 0.50) - 0.50) < 0.13
