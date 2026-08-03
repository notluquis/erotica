"""Tests for erotica.analysis._clipping — robust parallax sigma clipping.

This function decides which sources survive the parallax cut, so its behaviour is
science-critical: the P07 thread on CTTS removed by the 2-sigma clip depends on exactly
these semantics. Tests assert on *what it does to the sample*, not merely that it runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import QTable

from erotica.analysis._clipping import DEFAULT_MAXITERS, sigma_clip_parallax

# The tests that exercise generic clip behaviour -- outlier rejection,
# two-sidedness, monotonicity, the preselector, the return arity -- run on the
# RAW basis on purpose. Those invariants belong to "a robust sigma clip" and must
# hold for either basis, and pinning them on the basis whose behaviour was
# published keeps them comparable with the pre-2026-08-02 record. The normalised
# default has its own block at the bottom.
DEFAULTS = dict(
    sigma=2.0,
    use_biweight=True,
    in_place=False,
    mark_label=-1,
    print_results=False,
    return_mask=True,
    preselector_mask=None,
    method="raw",
)

NORMALISED = {**DEFAULTS, "method": "normalised"}


def _table(parallaxes, cluster_labels=None, errors=None):
    n = len(parallaxes)
    if cluster_labels is None:
        cluster_labels = np.zeros(n, dtype=np.int64)
    columns = {
        "parallax": np.asarray(parallaxes, dtype=float),
        "cluster": np.asarray(cluster_labels),
    }
    if errors is not None:
        columns["parallax_error"] = np.broadcast_to(
            np.asarray(errors, dtype=float), (n,)
        ).astype(float)
    return QTable(columns)


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

def test_clips_an_obvious_outlier_and_keeps_the_core():
    """A far-off parallax must be rejected; the tight core must survive intact."""
    core = np.full(30, 1.00)
    core[:15] += 0.01  # a little spread so the scale estimate is non-zero
    core[15:] -= 0.01
    values = np.append(core, 5.0)  # one blatant foreground interloper
    t = _table(values)

    _lo, _hi, _copy, keep, _noise = sigma_clip_parallax(t, cluster=0, **DEFAULTS)
    keep = np.asarray(keep, dtype=bool)

    assert keep[:-1].all(), "tight core members were clipped"
    assert not keep[-1], "the 5 mas outlier survived the clip"


def test_symmetric_sample_is_untouched():
    """With no outliers, a symmetric sample should lose nothing."""
    rng = np.random.default_rng(0)
    t = _table(rng.normal(1.0, 0.02, 200))
    keep = np.asarray(sigma_clip_parallax(t, cluster=0, **DEFAULTS)[3], dtype=bool)
    assert keep.sum() >= 190, f"clipped {200 - keep.sum()} of 200 clean sources"


def test_clip_is_two_sided():
    """Both a foreground and a background outlier must be removed."""
    values = np.concatenate([[0.2], np.full(40, 1.0) + np.linspace(-0.01, 0.01, 40), [3.0]])
    t = _table(values)
    keep = np.asarray(sigma_clip_parallax(t, cluster=0, **DEFAULTS)[3], dtype=bool)
    assert not keep[0] and not keep[-1], "clip is not two-sided"


def test_only_the_requested_cluster_is_affected():
    """Rows belonging to another cluster must never be touched."""
    values = np.concatenate([np.full(20, 1.0), np.full(20, 9.0)])
    labels = np.array([0] * 20 + [1] * 20, dtype=np.int64)
    values[0] = 6.0  # outlier inside cluster 0
    t = _table(values, labels)

    keep = np.asarray(sigma_clip_parallax(t, cluster=0, **DEFAULTS)[3], dtype=bool)
    assert not keep[20:].any(), "rows from cluster 1 were marked as kept"
    assert not keep[0], "the cluster-0 outlier was not clipped"


# ---------------------------------------------------------------------------
# The science-critical knob: sigma
# ---------------------------------------------------------------------------

def test_tighter_sigma_never_keeps_more():
    """Monotonicity: shrinking sigma cannot admit sources a wider cut rejected."""
    rng = np.random.default_rng(3)
    t = _table(np.concatenate([rng.normal(1.0, 0.05, 150), [2.5, 0.1]]))

    kept = {}
    for s in (1.0, 2.0, 3.0):
        opts = {**DEFAULTS, "sigma": s}
        kept[s] = np.asarray(sigma_clip_parallax(t, cluster=0, **opts)[3], dtype=bool)

    assert kept[1.0].sum() <= kept[2.0].sum() <= kept[3.0].sum()
    # and the tighter selection must be a subset of the looser one
    assert not (kept[1.0] & ~kept[2.0]).any()


def test_in_place_marks_rejects_with_the_label():
    """in_place must relabel rejected rows and leave the kept ones alone."""
    values = np.append(np.full(30, 1.0) + np.linspace(-0.01, 0.01, 30), 7.0)
    t = _table(values)
    opts = {**DEFAULTS, "in_place": True, "return_mask": False, "mark_label": -99}

    sigma_clip_parallax(t, cluster=0, **opts)
    labels = np.asarray(t["cluster"])
    assert labels[-1] == -99, "rejected row was not marked"
    assert (labels[:-1] == 0).all(), "surviving rows were relabelled"


# ---------------------------------------------------------------------------
# Guard rails — these raise rather than silently returning nonsense
# ---------------------------------------------------------------------------

def test_missing_cluster_raises():
    with pytest.raises(ValueError, match="zero rows"):
        sigma_clip_parallax(_table(np.ones(10)), cluster=42, **DEFAULTS)


def test_too_few_finite_values_raises():
    """Fewer than three finite parallaxes cannot support a robust scale estimate."""
    t = _table([1.0, np.nan, np.nan, np.nan])
    with pytest.raises(ValueError, match="Not enough finite"):
        sigma_clip_parallax(t, cluster=0, **DEFAULTS)


def test_preselector_mask_length_is_validated():
    t = _table(np.ones(10))
    opts = {**DEFAULTS, "preselector_mask": [True] * 5}
    with pytest.raises(ValueError, match="same length"):
        sigma_clip_parallax(t, cluster=0, **opts)


def test_preselector_restricts_which_rows_can_survive():
    """Rows excluded by the preselector must not be kept, even if their parallax is fine."""
    values = np.full(40, 1.0) + np.linspace(-0.01, 0.01, 40)
    pre = np.ones(40, dtype=bool)
    pre[:10] = False  # these have perfectly good parallaxes but are pre-excluded
    t = _table(values)

    opts = {**DEFAULTS, "preselector_mask": pre}
    keep = np.asarray(sigma_clip_parallax(t, cluster=0, **opts)[3], dtype=bool)
    assert not keep[:10].any(), "pre-excluded rows survived"
    assert keep[10:].sum() > 25, "pre-included core was over-clipped"


def test_return_arity_is_flag_dependent():
    """Characterisation: the return signature changes with the flags.

    This is an API wart worth pinning so a refactor cannot silently change it:
      in_place=True,  return_mask=True  -> (lower, upper, keep, noise)
      in_place=True,  return_mask=False -> (lower, upper)
      in_place=False, return_mask=True  -> (lower, upper, table, keep, noise)
      in_place=False, return_mask=False -> (lower, upper, table)
    """
    values = np.append(np.full(30, 1.0) + np.linspace(-0.01, 0.01, 30), 7.0)

    def run(**over):
        return sigma_clip_parallax(_table(values), cluster=0, **{**DEFAULTS, **over})

    assert len(run(in_place=True, return_mask=True)) == 4
    assert len(run(in_place=True, return_mask=False)) == 2
    assert len(run(in_place=False, return_mask=True)) == 5
    assert len(run(in_place=False, return_mask=False)) == 3


# ---------------------------------------------------------------------------
# The clip is a magnitude-dependent selection function.
#
# Characterisation, not aspiration: this pins CURRENT behaviour so a future
# change to the clip is visible. The oracle is by construction -- every
# simulated star is a true member, so any rejection is a false rejection.
# Per-star errors are drawn to mimic Gaia's magnitude dependence.
#
# Full measurement on the real NGC 6383 errors lives in
# tools/validation/parallax_clip_selection_function.py; see
# docs/design-notes/decisions.md for what it means for P01's faint quartile.
# ---------------------------------------------------------------------------

def _two_precision_populations(rng, n_per=150, small_err=0.03, large_err=0.30):
    """One cluster, one true parallax, two precisions. Every star is a member.

    Any rejection is therefore a *false* rejection by construction, and the
    retention rate per precision class **is** the induced selection function --
    no golden numbers and no model of the errors are involved.
    """
    err = np.concatenate([np.full(n_per, small_err), np.full(n_per, large_err)])
    obs = 0.9 + rng.normal(0, err)
    return obs, err


def test_raw_clip_rejects_large_error_stars_preferentially():
    """A 2-sigma clip on raw parallax is harsher on stars with bigger errors.

    Characterisation of the pre-2026-08-02 default, retained because ``method="raw"``
    is still reachable and its bias must stay visible.
    """
    rng = np.random.default_rng(20260727)
    n_per = 150
    keep_small, keep_large = [], []
    for _ in range(30):
        obs, err = _two_precision_populations(rng, n_per)
        t = _table(obs, errors=err)
        keep = np.asarray(sigma_clip_parallax(t, cluster=0, **DEFAULTS)[3], dtype=bool)
        keep_small.append(keep[:n_per].mean())
        keep_large.append(keep[n_per:].mean())

    r_small, r_large = float(np.mean(keep_small)), float(np.mean(keep_large))
    # Every star is a member, so both rates *should* be ~equal. They are not.
    assert r_small > r_large + 0.15, (
        f"expected the documented precision-dependent bias; got small={r_small:.1%} "
        f"large={r_large:.1%}"
    )


def test_the_default_clip_is_precision_blind():
    """The **shipped default** must not select on precision.

    Same oracle as the raw test above and the same fixture, so the two numbers are
    directly comparable: every star is a true member, so equal retention across
    the two precision classes is the correct answer and any gap is a measured
    selection effect. This calls the real function through its defaults -- not a
    reimplementation of the residual -- so wiring the default back to ``"raw"``,
    or dividing by anything other than the per-star error, makes it fail.
    """
    rng = np.random.default_rng(20260727)
    n_per = 150
    keep_small, keep_large = [], []
    for _ in range(30):
        obs, err = _two_precision_populations(rng, n_per)
        t = _table(obs, errors=err)
        # no `method=` -- the point is what the default does
        keep = np.asarray(
            sigma_clip_parallax(
                t,
                cluster=0,
                sigma=2.0,
                use_biweight=True,
                in_place=False,
                mark_label=-1,
                print_results=False,
                return_mask=True,
                preselector_mask=None,
            )[3],
            dtype=bool,
        )
        keep_small.append(keep[:n_per].mean())
        keep_large.append(keep[n_per:].mean())

    r_small, r_large = float(np.mean(keep_small)), float(np.mean(keep_large))
    assert abs(r_small - r_large) < 0.05, (
        f"the default clip selects on precision; got small={r_small:.1%} large={r_large:.1%}"
    )
    # And it must still be a 2-sigma cut, not an accidental pass-through: a
    # precision-blind clip that keeps everything would also satisfy the line above.
    assert 0.90 < r_small < 0.99, f"retention {r_small:.1%} is not that of a 2-sigma cut"


def test_the_normalised_clip_still_removes_a_genuine_interloper():
    """Precision-blindness must not become blindness.

    A foreground star 10 sigma from the centroid *on its own uncertainty* is an
    outlier under any defensible rule, so the tolerance the normalised basis buys
    a large error must not extend to it.
    """
    core = np.full(60, 1.00) + np.linspace(-0.01, 0.01, 60)
    err = np.full(61, 0.02)
    values = np.append(core, 5.0)  # 200 sigma out on its own error
    t = _table(values, errors=err)

    keep = np.asarray(sigma_clip_parallax(t, cluster=0, **NORMALISED)[3], dtype=bool)
    assert not keep[-1], "the 5 mas interloper survived the normalised clip"
    assert keep[:-1].sum() >= 57, f"over-clipped the core: kept {keep[:-1].sum()} of 60"


def test_the_normalised_clip_needs_the_error_column_and_says_so():
    """Falling back to the raw basis would silently restore the magnitude bias."""
    t = _table(np.linspace(0.9, 1.1, 20))  # no parallax_error column
    with pytest.raises(ValueError, match="parallax_error"):
        sigma_clip_parallax(t, cluster=0, **NORMALISED)


def test_non_positive_errors_are_rejected_rather_than_dividing_by_zero():
    """A zero error would make z infinite and clip the star for being precise."""
    values = np.linspace(0.9, 1.1, 20)
    err = np.full(20, 0.05)
    err[3] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        sigma_clip_parallax(_table(values, errors=err), cluster=0, **NORMALISED)


def test_an_unknown_method_is_refused():
    t = _table(np.linspace(0.9, 1.1, 20), errors=0.05)
    with pytest.raises(ValueError, match="'normalised' or 'raw'"):
        sigma_clip_parallax(t, cluster=0, **{**DEFAULTS, "method": "normalized"})


# ---------------------------------------------------------------------------
# maxiters
#
# astropy iterates the clip, re-estimating centre and scale each pass, so the
# bounds tighten monotonically with the iteration count. Leaving it implicit made
# the published selection depend on an upstream default. The invariant below
# cannot be faked: more iterations can only remove sources, never add them.
# ---------------------------------------------------------------------------

def _heavy_tailed():
    """A Gaussian core plus a ramp of mild outliers.

    The ramp is what makes an iteration test non-vacuous: with a clean core the
    first pass already converges and every ``maxiters`` agrees, so the test could
    not fail for the reason it was written.
    """
    rng = np.random.default_rng(20260802)
    return np.concatenate([rng.normal(1.0, 0.02, 200), np.linspace(1.05, 1.30, 25)])


def test_more_iterations_can_only_remove_sources():
    """Monotone in maxiters, and the tighter selection is a subset of the looser.

    On the RAW basis, where the scale is re-estimated from the survivors and the
    iteration therefore does tighten -- that is the astropy semantics this
    parameter inherits.
    """
    t = _table(_heavy_tailed(), errors=0.02)

    kept = {}
    for iters in (1, 2, 5, None):
        kept[iters] = np.asarray(
            sigma_clip_parallax(t, cluster=0, **{**DEFAULTS, "maxiters": iters})[3], dtype=bool
        )

    assert kept[None].sum() <= kept[5].sum() <= kept[2].sum() <= kept[1].sum()
    assert not (kept[5] & ~kept[1]).any(), "a later iteration re-admitted a clipped source"
    # and the fixture must actually exercise the difference, or the test is vacuous
    assert kept[1].sum() > kept[None].sum()


def test_the_default_maxiters_is_the_astropy_default_that_was_published():
    """Making it explicit must not move the number: the published clip ran 5 passes."""
    assert DEFAULT_MAXITERS == 5
    t = _table(_heavy_tailed(), errors=0.02)

    explicit = np.asarray(
        sigma_clip_parallax(t, cluster=0, **{**DEFAULTS, "maxiters": 5})[3], dtype=bool
    )
    shipped = np.asarray(sigma_clip_parallax(t, cluster=0, **DEFAULTS)[3], dtype=bool)
    np.testing.assert_array_equal(shipped, explicit)
    # ... and 5 is not vacuously the same as everything else here
    other = np.asarray(
        sigma_clip_parallax(t, cluster=0, **{**DEFAULTS, "maxiters": 1})[3], dtype=bool
    )
    assert other.sum() != shipped.sum()


def test_the_normalised_clip_does_not_shrink_its_own_dispersion():
    """Invariant: the normalised selection must be stable under iteration.

    Re-fitting a dispersion inside its own 2-sigma cut is the classic sigma-clip
    shrinkage -- each pass removes the widest survivors, the next pass fits a
    smaller dispersion, and the cut ratchets inward with nothing to stop it. On
    the real NGC 6383 sample that variant drives the fitted intrinsic spread
    0.0443 -> 0.0241 -> 0.0000 mas in three passes.

    The oracle is a fixed point, not a number: because the dispersion is fitted on
    the whole selection rather than on the survivors, the answer at 1, 2, 5 and 20
    iterations must be **identical**. A ratcheting implementation cannot satisfy
    that on data with a tail, and this fixture has one.
    """
    t = _table(_heavy_tailed(), errors=np.concatenate([np.full(200, 0.02), np.full(25, 0.10)]))

    first = None
    for iters in (1, 2, 5, 20, None):
        keep = np.asarray(
            sigma_clip_parallax(t, cluster=0, **{**NORMALISED, "maxiters": iters})[3], dtype=bool
        )
        if first is None:
            first = keep
        np.testing.assert_array_equal(keep, first, err_msg=f"selection moved at maxiters={iters}")
    # and it must be a real cut, not a pass-through that is trivially stable
    assert 0 < first.sum() < len(t)


# ---------------------------------------------------------------------------
# The analyzer facade -- what the pipeline actually calls.
#
# The module-level default above is only load-bearing if the facade forwards it.
# ``ClusterAnalyzer.__init__`` loads a dataset from disk, so the instance is
# built without it: the method under test reads ``self.data`` and
# ``self.selected_cluster`` and nothing else.
# ---------------------------------------------------------------------------

def _bare_analyzer(table):
    from erotica.analysis.analyzer import ClusterAnalyzer

    analyzer = object.__new__(ClusterAnalyzer)
    analyzer.data = table
    analyzer.selected_cluster = 0
    return analyzer


def test_the_analyzer_facade_inherits_the_precision_blind_default():
    """Same oracle as the module-level test, one layer up: every star is a member,
    so a retention gap between the two precision classes is a measured selection
    effect and not noise.
    """
    rng = np.random.default_rng(20260727)
    n_per = 150
    keep_small, keep_large = [], []
    for _ in range(30):
        obs, err = _two_precision_populations(rng, n_per)
        analyzer = _bare_analyzer(_table(obs, errors=err))
        _lo, _hi, keep, _noise = analyzer.sigma_clip_parallax(in_place=True, return_mask=True)
        keep = np.asarray(keep, dtype=bool)
        keep_small.append(keep[:n_per].mean())
        keep_large.append(keep[n_per:].mean())

    r_small, r_large = float(np.mean(keep_small)), float(np.mean(keep_large))
    assert abs(r_small - r_large) < 0.05, (
        f"the facade still selects on precision; small={r_small:.1%} large={r_large:.1%}"
    )
    assert 0.90 < r_small < 0.99


def test_the_analyzer_facade_can_still_reach_the_raw_clip():
    """The published behaviour must stay reproducible through the same entry point."""
    rng = np.random.default_rng(20260727)
    obs, err = _two_precision_populations(rng, 150)
    analyzer = _bare_analyzer(_table(obs, errors=err))
    _lo, _hi, keep, _noise = analyzer.sigma_clip_parallax(
        method="raw", in_place=True, return_mask=True
    )
    keep = np.asarray(keep, dtype=bool)
    assert keep[:150].mean() > keep[150:].mean() + 0.15, "the raw basis lost its known bias"


def test_the_analyzer_facade_inherits_the_published_maxiters():
    """The facade must not silently re-open the 9.2% iteration-count swing.

    Heavy-tailed fixture on purpose: with a clean Gaussian core the first pass
    already converges and any ``maxiters`` would agree, so the test could not
    fail for the reason it was written.
    """
    values = _heavy_tailed()

    def keep_with(**over):
        analyzer = _bare_analyzer(_table(values, errors=0.02))
        _lo, _hi, keep, _noise = analyzer.sigma_clip_parallax(
            method="raw", in_place=True, return_mask=True, **over
        )
        return np.asarray(keep, dtype=bool)

    shipped = keep_with()
    np.testing.assert_array_equal(shipped, keep_with(maxiters=DEFAULT_MAXITERS))
    # the fixture must distinguish the settings, or the equality above proves nothing
    assert keep_with(maxiters=1).sum() > shipped.sum()


def _window_truncated_companions(rng, n=300, err=0.50, half_width=0.05):
    """Stars so imprecise that the survey's own parallax window truncates them.

    NGC 6383 was queried with ``parallax BETWEEN 0.75 AND 1.1``, a window about
    1.2 median errors wide for the faintest magnitude quartile. A star in that
    regime cannot appear in the catalogue with a large normalised residual: its
    ``|z|`` is bounded by the window, not by its membership. This reproduces that
    regime in miniature.
    """
    return 0.9 + rng.uniform(-half_width, half_width, n), np.full(n, err)


def test_the_threshold_is_asserted_at_two_sigma_not_refitted_from_the_sample():
    """Invariant: stars that *cannot* be outliers must not tighten the cut on stars that can.

    Normalisation already fixes the scale of ``z`` at 1, so the threshold is a
    constant. Re-estimating it from the sample instead makes the cut depend on the
    composition of the sample: adding a window-truncated sub-population, whose
    ``|z|`` is bounded near zero by the survey query rather than by membership,
    drags the fitted scale down and clips the well-measured stars for it. On the
    real NGC 6383 sample the refitted scale is 0.71, so a nominal 2-sigma cut acts
    at 1.4 sigma.

    The oracle is invariance, not a number: the retention of the well-measured
    population must not depend on how many stars that cannot be rejected are
    sitting beside it.
    """
    rng = np.random.default_rng(20260802)
    n_a = 300
    err_a = np.full(n_a, 0.02)
    plx_a = 0.9 + rng.normal(0.0, np.hypot(0.04, err_a))  # real depth, precise errors

    alone = np.asarray(
        sigma_clip_parallax(_table(plx_a, errors=err_a), cluster=0, **NORMALISED)[3], dtype=bool
    )

    plx_b, err_b = _window_truncated_companions(rng)
    together = np.asarray(
        sigma_clip_parallax(
            _table(np.concatenate([plx_a, plx_b]), errors=np.concatenate([err_a, err_b])),
            cluster=0, **NORMALISED,
        )[3],
        dtype=bool,
    )[:n_a]

    assert abs(together.mean() - alone.mean()) < 0.03, (
        f"retention of the well-measured population moved from {alone.mean():.1%} to "
        f"{together.mean():.1%} when {len(plx_b)} un-rejectable stars were added"
    )
    # and the fixture must be a real cut on a real population, not a pass-through
    assert 0.90 < alone.mean() < 0.99
