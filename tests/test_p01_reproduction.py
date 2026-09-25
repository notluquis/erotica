"""The P01 (NGC 6383, A&A aa52082-24) membership recipe must still reproduce the published run.

Oracle: the artefact of the submitted run itself, not a re-derivation. The 40 arcmin
``paperfaithful_with_clip_flags.ecsv`` holds the preprocessed catalogue (15 276 sources) together
with the columns the submitted run wrote (``cluster_hdbscan``, ``probability``), and
``paperfaithful_reference_p06.ecsv`` holds the 254-member reference sample, whose ``source_id``
set equals the ``pMember >= 0.6`` rows of the CDS ``table2.dat`` (checked 2026-09-25, hub node
B10). ``summary.json`` carries the published parallax-clip bounds.

Why this exists (hub findings R25-01, R25-02). EROTICA changed two defaults after the submitted
run -- ``selection`` from ``"max_members"`` to ``"max_persistence"`` (e1e1a4c, 2026-08-03) and
``approx_min_span_tree`` from ``True`` to ``False`` (db0fafb, 2026-08-04) -- and each one alone
breaks the reproduction on this catalogue:

    pins                               selected mcs   branch   p>=0.6   shared with the 254
    both (the P01 recipe)                    43         701      254          254
    selection default only                   51         497      259          253
    approx_min_span_tree default only       273         602      418          246

These tests call the real ``search_pseudoprobability`` with the recipe's settings, so they go red
if either pin stops being honoured or the sweep itself drifts.

⚠ **They do not run in CI.** The artefacts live under ``data/test/NGC6383/comments_paper/``,
which git does not track (it is ~730 MB of leftovers from the paper extraction). Without them the
tests skip with the path in the reason. ``EROTICA_P01_RADIUS_DIR`` points them at another
checkout's ``generated/40`` directory (a worktree has no untracked data).

⚠ **The recipe is restated here, not imported**: it lives in the paper repository
(``validation/ngc6383_radius_robustness.py``), which this package cannot depend on. The restated
part is only the arguments of one call plus the final mask; if the paper script changes its
arguments, this test does not follow it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

_DEFAULT_DIR = (
    Path(__file__).resolve().parents[1]
    / "data/test/NGC6383/comments_paper/radius_robustness/generated/40"
)
RADIUS_DIR = Path(os.environ.get("EROTICA_P01_RADIUS_DIR", _DEFAULT_DIR))
FLAGS = RADIUS_DIR / "paperfaithful_with_clip_flags.ecsv"
REFERENCE = RADIUS_DIR / "paperfaithful_reference_p06.ecsv"
SUMMARY = RADIUS_DIR / "summary.json"

pytestmark = pytest.mark.skipif(
    not (FLAGS.exists() and REFERENCE.exists() and SUMMARY.exists()),
    reason=f"P01 40' artefacts not found under {RADIUS_DIR} (untracked data; set EROTICA_P01_RADIUS_DIR)",
)

P01_HDBSCAN_KWARGS = {
    "cluster_selection_method": "leaf",
    "allow_single_cluster": True,
    "core_dist_n_jobs": 1,
}


@pytest.fixture(scope="module")
def published():
    from astropy.table import Table

    table = Table.read(FLAGS)
    reference = Table.read(REFERENCE)
    summary = json.loads(SUMMARY.read_text())
    return table, set(np.asarray(reference["source_id"]).tolist()), summary


def test_the_mcs43_fit_needs_the_approximate_tree(published):
    """The docstring of ``approx_min_span_tree`` once said the switch moves no published NGC 6383
    number. That was measured with ``match_reference_implementation=True``, which forces the exact
    tree; P01 runs with it off, where the switch is live. Positive and negative in one test: the
    approximate tree gives the published labels byte for byte, the exact one does not."""
    from erotica.core.clustering import HDBSCANEstimator

    table, _, summary = published
    X = np.column_stack([np.asarray(table["pmra"]), np.asarray(table["pmdec"])])
    stored = np.asarray(table["cluster_hdbscan"])

    def fit(approx: bool) -> np.ndarray:
        model = HDBSCANEstimator(
            min_cluster_size=int(summary["best_mcs"]),
            algorithm="best",
            metric="euclidean",
            match_reference_implementation=False,
            approx_min_span_tree=approx,
            **P01_HDBSCAN_KWARGS,
        ).fit(X)
        return np.asarray(model.model_.labels_)

    assert np.array_equal(fit(True), stored)
    exact = fit(False)
    label = int(summary["ngc_like_label"])
    overlap = np.count_nonzero((exact == label) & (stored == label))
    assert overlap < np.count_nonzero(stored == label)  # measured 498 of 701


@pytest.mark.slow
def test_the_p01_recipe_reproduces_the_254_published_members(published):
    """Full sweep (290 fits, ~6 min on one core), including the automatic step selector: the
    recipe must select ``min_cluster_size=43`` on its own, write the published columns exactly,
    and give back the 254 ``source_id`` with the published clip bounds."""
    from astropy.table import Table

    from erotica.core import Clustering

    table, reference_ids, summary = published
    data = Table({c: np.asarray(table[c]) for c in ("source_id", "pmra", "pmdec", "parallax")})
    clust = Clustering(data, data[:0])
    clust.search_pseudoprobability(
        columns=["pmra", "pmdec"],
        min_cluster_size_samples=range(10, 300),
        probability_threshold=0.5,
        min_cluster_members=200,
        max_cluster_members=1000,
        select_cluster=False,
        selection="max_members",
        approx_min_span_tree=True,
        match_reference_implementation=False,
        hdbscan_kwargs=P01_HDBSCAN_KWARGS,
    )
    selected = clust.pseudoprobability_selected_
    assert selected["min_cluster_size"] == summary["best_mcs"] == 43
    assert selected["desired_len"] == summary["desired_len"] == 701

    labels = np.asarray(clust.data["cluster_hdbscan"])
    probability = np.asarray(clust.data["probability"], dtype=float)
    assert np.array_equal(labels, np.asarray(table["cluster_hdbscan"]))
    assert np.array_equal(probability, np.asarray(table["probability"], dtype=float))

    parallax = np.asarray(clust.data["parallax"], dtype=float)
    members = (
        (labels == int(summary["ngc_like_label"]))
        & (probability >= 0.6)
        & (parallax >= summary["clip_low"])
        & (parallax <= summary["clip_high"])
    )
    got = set(np.asarray(clust.data["source_id"][members]).tolist())
    assert len(got) == 254
    assert got == reference_ids
