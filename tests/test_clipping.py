"""Tests for erotica.analysis._clipping — robust parallax sigma clipping.

This function decides which sources survive the parallax cut, so its behaviour is
science-critical: the P07 thread on CTTS removed by the 2-sigma clip depends on exactly
these semantics. Tests assert on *what it does to the sample*, not merely that it runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import QTable

from erotica.analysis._clipping import sigma_clip_parallax

DEFAULTS = dict(
    sigma=2.0,
    use_biweight=True,
    in_place=False,
    mark_label=-1,
    print_results=False,
    return_mask=True,
    preselector_mask=None,
)


def _table(parallaxes, cluster_labels=None):
    n = len(parallaxes)
    if cluster_labels is None:
        cluster_labels = np.zeros(n, dtype=np.int64)
    return QTable(
        {"parallax": np.asarray(parallaxes, dtype=float), "cluster": np.asarray(cluster_labels)}
    )


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

def test_raw_clip_rejects_large_error_stars_preferentially():
    """A 2-sigma clip on raw parallax is harsher on stars with bigger errors."""
    rng = np.random.default_rng(20260727)
    n_per = 150
    # Two populations, identical TRUE parallax, differing only in precision.
    small_err, large_err = 0.03, 0.30
    keep_small, keep_large = [], []
    for _ in range(30):
        obs = np.concatenate([
            rng.normal(0.9, small_err, n_per),
            rng.normal(0.9, large_err, n_per),
        ])
        t = _table(obs)
        keep = np.asarray(sigma_clip_parallax(t, cluster=0, **DEFAULTS)[3], dtype=bool)
        keep_small.append(keep[:n_per].mean())
        keep_large.append(keep[n_per:].mean())

    r_small, r_large = float(np.mean(keep_small)), float(np.mean(keep_large))
    # Every star is a member, so both rates *should* be ~equal. They are not.
    assert r_small > r_large + 0.15, (
        f"expected the documented precision-dependent bias; got small={r_small:.1%} "
        f"large={r_large:.1%}"
    )


def test_normalized_residual_clip_is_precision_blind():
    """Clipping on (plx - centre)/err_i retains both populations equally.

    This is the error-aware alternative referenced in the decision log: a large
    uncertainty buys tolerance rather than a rejection.
    """
    from astropy.stats import biweight_location, biweight_scale

    rng = np.random.default_rng(20260727)
    n_per = 150
    small_err, large_err = 0.03, 0.30
    keep_small, keep_large = [], []
    for _ in range(30):
        err = np.concatenate([np.full(n_per, small_err), np.full(n_per, large_err)])
        obs = 0.9 + rng.normal(0, err)
        z = (obs - biweight_location(obs)) / err
        keep = np.abs(z - biweight_location(z)) <= 2.0 * biweight_scale(z)
        keep_small.append(keep[:n_per].mean())
        keep_large.append(keep[n_per:].mean())

    r_small, r_large = float(np.mean(keep_small)), float(np.mean(keep_large))
    assert abs(r_small - r_large) < 0.05, (
        f"normalized clip should be precision-blind; got small={r_small:.1%} large={r_large:.1%}"
    )
