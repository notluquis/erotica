"""Tests for erotica.core.clustering — Clustering and HDBSCANEstimator."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_qtable(n: int = 300, seed: int = 42) -> QTable:
    """Two well-separated clusters in pmra/pmdec + required columns."""
    rng = np.random.default_rng(seed)
    half = n // 2
    pmra = np.concatenate([rng.normal(0.0, 0.3, half), rng.normal(5.0, 0.3, half)])
    pmdec = np.concatenate([rng.normal(0.0, 0.3, half), rng.normal(5.0, 0.3, half)])
    return QTable(
        {
            "pmra": pmra * u.mas / u.yr,
            "pmdec": pmdec * u.mas / u.yr,
            "ra": rng.uniform(263, 264, n) * u.deg,
            "dec": rng.uniform(-32, -31, n) * u.deg,
            "cluster": np.zeros(n, dtype=int),
        }
    )


def _make_contaminated_qtable(n_cluster: int = 40, n_field: int = 400, seed: int = 1) -> QTable:
    """A compact cluster buried in a field ~10x larger — the regime of issue #7.

    Truth is carried in ``is_member`` and deliberately NOT in ``cluster``:
    ``_annotate_pseudoprobability_results`` overwrites ``data["cluster"]``, so a truth
    column by that name would be silently replaced by the algorithm's own answer and any
    purity assertion built on it would compare the algorithm against itself.

    The field is broad enough that HDBSCAN carves more than one structure out of it, and
    at least one of those field structures is LARGER than the injected cluster. That is
    the condition under which :meth:`Clustering._cluster_label_for_size`'s
    "largest non-noise cluster" fallback returns the field.
    """
    rng = np.random.default_rng(seed)
    pmra = np.concatenate([rng.normal(-1.2, 0.25, n_cluster), rng.normal(-3.0, 4.0, n_field)])
    pmdec = np.concatenate([rng.normal(-0.5, 0.25, n_cluster), rng.normal(-3.0, 4.0, n_field)])
    truth = np.concatenate([np.ones(n_cluster, bool), np.zeros(n_field, bool)])
    order = rng.permutation(truth.size)  # row order must carry no membership information
    return QTable(
        {
            "pmra": pmra[order] * u.mas / u.yr,
            "pmdec": pmdec[order] * u.mas / u.yr,
            "is_member": truth[order],
        }
    )


@pytest.fixture
def good_data():
    return _make_qtable()


@pytest.fixture
def bad_data():
    return _make_qtable(n=20, seed=99)


@pytest.fixture
def fitted_clust(good_data, bad_data):
    from erotica.core.clustering import Clustering

    clust = Clustering(good_data, bad_data)
    clust.search_pseudoprobability(
        columns=["pmra", "pmdec"],
        min_cluster_size_samples=range(10, 40),
        min_samples=5,
        probability_threshold=0.5,
    )
    return clust


# ---------------------------------------------------------------------------
# HDBSCANEstimator
# ---------------------------------------------------------------------------


class TestHDBSCANEstimator:
    def test_fit_returns_self(self, good_data):
        from erotica.core._estimator import HDBSCANEstimator

        X = good_data["pmra", "pmdec"].to_pandas().values
        est = HDBSCANEstimator(min_cluster_size=10)
        assert est.fit(X) is est

    def test_labels_length(self, good_data):
        from erotica.core._estimator import HDBSCANEstimator

        X = good_data["pmra", "pmdec"].to_pandas().values
        est = HDBSCANEstimator(min_cluster_size=10).fit(X)
        assert len(est.model_.labels_) == len(good_data)

    def test_predict_after_fit(self, good_data):
        from erotica.core._estimator import HDBSCANEstimator

        X = good_data["pmra", "pmdec"].to_pandas().values
        est = HDBSCANEstimator(min_cluster_size=10).fit(X)
        preds = est.predict(X)
        assert len(preds) == len(good_data)

    def test_predict_before_fit_raises(self):
        from erotica.core._estimator import HDBSCANEstimator

        est = HDBSCANEstimator(min_cluster_size=10)
        with pytest.raises(RuntimeError):
            est.predict(np.zeros((10, 2)))

    def test_score_returns_float(self, good_data):
        from erotica.core._estimator import HDBSCANEstimator

        X = good_data["pmra", "pmdec"].to_pandas().values
        est = HDBSCANEstimator(min_cluster_size=10).fit(X)
        assert isinstance(est.score(X), float)


# ---------------------------------------------------------------------------
# Clustering.__init__
# ---------------------------------------------------------------------------


class TestClusteringInit:
    def test_default_attributes(self, good_data):
        from erotica.core.clustering import Clustering

        clust = Clustering(good_data)
        assert clust.clusterer is None
        assert clust.best_params_ is None
        assert clust.best_score_ is None
        assert clust.combined_data is None
        assert clust.pseudoprobability_results_ is None
        assert clust.pseudoprobability_selected_ is None
        assert clust.pseudoprobability_sweep_track_ is None
        assert clust.legacy_cluster_selection is False

    def test_legacy_cluster_selection_is_a_real_keyword(self, good_data):
        """Reproducing a pre-2026-08-03 result must not require monkey-patching."""
        from erotica.core.clustering import Clustering

        assert Clustering(good_data, legacy_cluster_selection=True).legacy_cluster_selection is True

    def test_invalid_search_method_raises(self, good_data):
        from erotica.core.clustering import Clustering

        with pytest.raises(ValueError):
            Clustering(good_data, search_method="nonexistent")


# ---------------------------------------------------------------------------
# search_pseudoprobability
# ---------------------------------------------------------------------------


class TestSearchPseudoprobability:
    def test_runs_without_error(self, good_data, bad_data):
        from erotica.core.clustering import Clustering

        clust = Clustering(good_data, bad_data)
        clust.search_pseudoprobability(
            columns=["pmra", "pmdec"],
            min_cluster_size_samples=range(10, 30),
            min_samples=5,
        )

    def test_populates_best_params(self, fitted_clust):
        assert fitted_clust.best_params_ is not None
        assert "min_cluster_size" in fitted_clust.best_params_

    def test_populates_best_score(self, fitted_clust):
        assert fitted_clust.best_score_ is not None
        assert np.isfinite(fitted_clust.best_score_)

    def test_populates_pseudoprobability_selected(self, fitted_clust):
        sel = fitted_clust.pseudoprobability_selected_
        assert sel is not None
        assert "min_cluster_size" in sel
        assert "desired_len" in sel
        assert "probability_times" in sel

    def test_populates_sweep_track(self, fitted_clust):
        track = fitted_clust.pseudoprobability_sweep_track_
        assert track is not None
        assert len(track) > 0
        assert "min_cluster_size" in track[0]
        assert "desired_len" in track[0]

    def test_data_has_cluster_column(self, fitted_clust):
        assert "cluster" in fitted_clust.data.colnames

    def test_data_has_probability_columns(self, fitted_clust):
        for col in ("probability_hdbscan", "probability_times", "probability"):
            assert col in fitted_clust.data.colnames

    def test_probability_times_in_0_1(self, fitted_clust):
        pt = np.array(fitted_clust.data["probability_times"])
        assert np.all(pt >= 0.0)
        assert np.all(pt <= 1.0)

    def test_combined_data_populated(self, fitted_clust):
        assert fitted_clust.combined_data is not None

    def test_cluster_label_minus1_for_noise(self, fitted_clust):
        labels = np.array(fitted_clust.data["cluster"])
        assert -1 in labels

    def test_select_cluster_false_keeps_all_labels(self, good_data, bad_data):
        from erotica.core.clustering import Clustering

        clust = Clustering(good_data, bad_data)
        clust.search_pseudoprobability(
            columns=["pmra", "pmdec"],
            min_cluster_size_samples=range(10, 30),
            min_samples=5,
            select_cluster=False,
        )
        labels = np.array(clust.data["cluster"])
        unique = np.unique(labels)
        # with select_cluster=False, raw labels — may have multiple clusters
        assert len(unique) >= 1

    def test_empty_range_raises(self, good_data, bad_data):
        from erotica.core.clustering import Clustering

        clust = Clustering(good_data, bad_data)
        with pytest.raises(RuntimeError):
            clust.search_pseudoprobability(
                columns=["pmra", "pmdec"],
                min_cluster_size_samples=range(500, 500),  # empty
            )

    def test_min_max_cluster_members_filter(self, good_data, bad_data):
        from erotica.core.clustering import Clustering

        clust = Clustering(good_data, bad_data)
        clust.search_pseudoprobability(
            columns=["pmra", "pmdec"],
            min_cluster_size_samples=range(10, 40),
            min_samples=5,
            min_cluster_members=1,
            max_cluster_members=10000,
        )
        assert clust.pseudoprobability_selected_["desired_len"] >= 1


# ---------------------------------------------------------------------------
# get_best_params / save_results
# ---------------------------------------------------------------------------


class TestPublicHelpers:
    def test_get_best_params(self, fitted_clust):
        params = fitted_clust.get_best_params()
        assert isinstance(params, dict)

    def test_save_results_ecsv(self, fitted_clust, tmp_path):
        out = str(tmp_path / "results.ecsv")
        fitted_clust.save_results(out, format="ascii.ecsv")
        import os

        assert os.path.exists(out)

    def test_save_results_before_fit_raises(self, good_data):
        from erotica.core.clustering import Clustering

        clust = Clustering(good_data)
        with pytest.raises(ValueError):
            clust.save_results("/tmp/nope.ecsv")

    def test_clustering_statistics_runs(self, fitted_clust):
        fitted_clust.clustering_statistics()

    def test_get_cluster_summary_returns_table(self, fitted_clust):
        summary = fitted_clust.get_cluster_summary(include_noise=True)
        assert summary is not None


# ---------------------------------------------------------------------------
# plot_mcs_sweep
# ---------------------------------------------------------------------------


class TestPlotMcsSweep:
    def test_runs(self, fitted_clust):
        import matplotlib

        matplotlib.use("Agg")
        fitted_clust.plot_mcs_sweep()

    def test_before_fit_raises(self, good_data):
        import matplotlib

        from erotica.core.clustering import Clustering

        matplotlib.use("Agg")
        clust = Clustering(good_data)
        with pytest.raises(RuntimeError):
            clust.plot_mcs_sweep()

    def test_save_path(self, fitted_clust, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        out = str(tmp_path / "mcs_sweep.pdf")
        fitted_clust.plot_mcs_sweep(save_path=out)
        import os

        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# plot_condensed_tree
# ---------------------------------------------------------------------------


class TestPlotCondensedTree:
    def test_runs(self, fitted_clust):
        import matplotlib

        matplotlib.use("Agg")
        fitted_clust.plot_condensed_tree()

    def test_before_fit_raises(self, good_data):
        import matplotlib

        from erotica.core.clustering import Clustering

        matplotlib.use("Agg")
        clust = Clustering(good_data)
        with pytest.raises(RuntimeError):
            clust.plot_condensed_tree()

    def test_save_path(self, fitted_clust, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        out = str(tmp_path / "condensed_tree.pdf")
        fitted_clust.plot_condensed_tree(save_path=out)
        import os

        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# _build_pseudoprobability / _select_pseudoprobability_result
# ---------------------------------------------------------------------------


class TestStaticHelpers:
    def test_build_pseudoprobability_all_cluster(self):
        from erotica.core.clustering import Clustering

        # 3 sources, 2 iterations, always in cluster
        storage = [[0, 0], [1, 1], [0, 1]]
        pt = Clustering._build_pseudoprobability(storage)
        np.testing.assert_array_equal(pt, [1.0, 1.0, 1.0])

    def test_build_pseudoprobability_all_noise(self):
        from erotica.core.clustering import Clustering

        storage = [[-1, -1], [-1, -1]]
        pt = Clustering._build_pseudoprobability(storage)
        np.testing.assert_array_equal(pt, [0.0, 0.0])

    def test_build_pseudoprobability_mixed(self):
        from erotica.core.clustering import Clustering

        storage = [[0, -1], [-1, -1]]  # first source: 1/2 in cluster
        pt = Clustering._build_pseudoprobability(storage)
        assert pt[0] == pytest.approx(0.5)
        assert pt[1] == pytest.approx(0.0)

    def test_select_max_members(self):
        from erotica.core.clustering import Clustering

        results = [
            {"min_cluster_size": 10, "desired_len": 100, "lambda_value": 5.0},
            {"min_cluster_size": 20, "desired_len": 200, "lambda_value": 3.0},
        ]
        sel = Clustering._select_pseudoprobability_result(results, "max_members")
        assert sel["min_cluster_size"] == 20

    def test_select_max_lambda(self):
        from erotica.core.clustering import Clustering

        results = [
            {"min_cluster_size": 10, "desired_len": 100, "lambda_value": 9.0},
            {"min_cluster_size": 20, "desired_len": 200, "lambda_value": 3.0},
        ]
        sel = Clustering._select_pseudoprobability_result(results, "max_lambda")
        assert sel["min_cluster_size"] == 10

    def test_select_invalid_raises(self):
        from erotica.core.clustering import Clustering

        with pytest.raises(ValueError):
            Clustering._select_pseudoprobability_result([{}], "invalid")

    def test_cluster_label_for_size_exact_match(self):
        from erotica.core.clustering import Clustering

        labels = np.array([0, 0, 0, 1, 1, -1])
        assert Clustering._cluster_label_for_size(labels, 3) == 0

    def test_cluster_label_for_size_no_match_returns_largest(self):
        from erotica.core.clustering import Clustering

        labels = np.array([0, 0, 0, 1, 1, -1])
        # desired_len=99 doesn't exist → return largest cluster (0, size 3)
        assert Clustering._cluster_label_for_size(labels, 99) == 0

    def test_cluster_label_for_size_all_noise(self):
        from erotica.core.clustering import Clustering

        labels = np.array([-1, -1, -1])
        assert Clustering._cluster_label_for_size(labels, 1) == -1


# ---------------------------------------------------------------------------
# Branch selection on a contaminated frame — issue #7
# ---------------------------------------------------------------------------


def _select_on_contaminated_frame(legacy: bool, selection: str = "max_members") -> dict:
    """Drive a contaminated frame through ``search_pseudoprobability`` and score the pick.

    It has to go through ``search_pseudoprobability``: a bare HDBSCAN fit never reaches the
    selector, and it is ``selection='max_members'`` — argmax of the same condensed-tree row
    count across the sweep — that drives ``desired_len`` away from any real cluster size and
    so pushes the size match into its failure case.

    ``selection`` is pinned explicitly rather than left to the default. The default became
    ``"max_persistence"`` on 2026-08-04, and letting these tests follow it would have
    quietly changed what the issue-#7 tests exercise: they assert that the LEGACY label
    selector returns the field, and that only happens under the sweep step ``max_members``
    picks. A test whose premise moves with an unrelated default is not a regression test.
    """
    from erotica.core.clustering import Clustering

    clust = Clustering(_make_contaminated_qtable(), legacy_cluster_selection=legacy)
    clust.search_pseudoprobability(
        columns=["pmra", "pmdec"],
        min_cluster_size_samples=range(10, 40),
        min_samples=5,
        probability_threshold=0.5,
        selection=selection,
    )
    truth = np.asarray(clust.data["is_member"], dtype=bool)
    selected = np.asarray(clust.data["cluster"], dtype=int) != -1
    return {
        "n_selected": int(selected.sum()),
        "purity": float(truth[selected].mean()) if selected.any() else 0.0,
        "recall": float(truth[selected].sum() / truth.sum()),
        "label": int(clust.pseudoprobability_selected_["selected_cluster"]),
        "desired_len": int(clust.pseudoprobability_selected_["desired_len"]),
        "label_sizes": dict(
            zip(
                *[
                    a.tolist()
                    for a in np.unique(
                        np.asarray(clust.data["cluster_hdbscan"], dtype=int), return_counts=True
                    )
                ],
                strict=True,
            )
        ),
        "probability": np.asarray(clust.data["probability"], dtype=float),
    }


@pytest.fixture(scope="module")
def legacy_pick():
    return _select_on_contaminated_frame(legacy=True)


@pytest.fixture(scope="module")
def tree_pick():
    return _select_on_contaminated_frame(legacy=False)


class TestContaminatedBranchSelection:
    """Issue #7: the size-matching selector reports the field as the cluster.

    ``_desired_tree_branch_size`` counts ROWS of the condensed tree; ``_cluster_label_for_size``
    matches that integer against flat-cluster POINT counts. On this frame the two miss by one
    (102 tree rows vs a 101-point cluster) and the fallback hands back the largest non-noise
    cluster, which is field.
    """

    def test_frame_actually_exercises_the_defect(self, legacy_pick):
        """Guard the guard: a frame where the size match happens to succeed proves nothing."""
        sizes = legacy_pick["label_sizes"]
        assert legacy_pick["desired_len"] not in sizes.values(), (
            f"desired_len={legacy_pick['desired_len']} coincides with a real cluster size "
            f"{sizes} — this frame no longer reaches the fallback, so the test below is vacuous"
        )
        assert sum(1 for k in sizes if k != -1) >= 2, (
            f"only one non-noise cluster in {sizes}; 'largest non-noise cluster' is then "
            "trivially the cluster and the defect cannot show"
        )

    def test_legacy_selector_returns_the_field(self, legacy_pick):
        assert legacy_pick["n_selected"] > 0, "nothing selected at all — not the failure mode"
        assert legacy_pick["purity"] < 0.10, (
            f"expected the legacy selector to return the field; got purity "
            f"{legacy_pick['purity']:.3f} over {legacy_pick['n_selected']} sources"
        )

    def test_tree_selector_returns_the_cluster(self, tree_pick):
        assert tree_pick["purity"] > 0.90, (
            f"tree selector purity {tree_pick['purity']:.3f} over {tree_pick['n_selected']} sources"
        )
        assert tree_pick["recall"] > 0.50, f"recall {tree_pick['recall']:.3f}"

    def test_tree_selector_beats_legacy_on_the_same_frame(self, legacy_pick, tree_pick):
        assert tree_pick["label"] != legacy_pick["label"], (
            "both selectors picked the same label, so this frame does not discriminate"
        )
        assert tree_pick["purity"] - legacy_pick["purity"] > 0.50, (
            f"legacy {legacy_pick['purity']:.3f} vs tree {tree_pick['purity']:.3f}"
        )

    def test_selector_cannot_change_the_pseudoprobability(self, legacy_pick, tree_pick):
        """The fix repairs the member list and NOTHING about p-tilde.

        ``data['probability'] = p_HDBSCAN * f_i`` is written before the branch that chooses
        a label, so calibration (ECE/Brier) and any probability-ranked score (AUC, top-K)
        are identical under both selectors. Recorded so nobody credits the selector fix with
        a calibration improvement it cannot produce.
        """
        np.testing.assert_array_equal(legacy_pick["probability"], tree_pick["probability"])


# ---------------------------------------------------------------------------
# Idempotency / reproducibility
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Repeating a search with the same input must reproduce the same result."""

    def test_pseudoprobability_is_idempotent(self, good_data):
        from erotica.core.clustering import Clustering

        out = []
        for _ in range(2):
            clu = Clustering(good_data.copy())
            clu.search_pseudoprobability(
                columns=["pmra", "pmdec"], min_cluster_size_samples=range(10, 40, 5), min_samples=5
            )
            out.append(np.asarray(clu.clusterer.labels_))
        np.testing.assert_array_equal(out[0], out[1])

    def test_pseudoprobability_is_row_order_invariant(self, good_data):
        """Shuffling the input rows must not change WHICH sources are members."""
        from erotica.core.clustering import Clustering

        member_sets = []
        for shuffle in (False, True):
            data = good_data.copy()
            if shuffle:
                data = data[np.random.default_rng(7).permutation(len(data))]
            clu = Clustering(data)
            clu.search_pseudoprobability(
                columns=["pmra", "pmdec"], min_cluster_size_samples=range(10, 40, 5), min_samples=5
            )
            labels = np.asarray(clu.clusterer.labels_)
            pmra = np.round(np.asarray(clu.data["pmra"])[labels != -1], 6)
            pmdec = np.round(np.asarray(clu.data["pmdec"])[labels != -1], 6)
            member_sets.append(set(zip(pmra, pmdec, strict=True)))
        assert member_sets[0] == member_sets[1]

    def test_optuna_search_uses_a_default_space_and_is_seeded(self, good_data):
        """Optuna with no explicit search space must work (not suggest an empty dict)
        and must reproduce, because the sampler is seeded by default."""
        from erotica.core.clustering import Clustering

        best = []
        for i in range(2):
            clu = Clustering(good_data.copy(), search_method="optuna", study_name=f"idem-test-{i}")
            clu.search(columns=["pmra", "pmdec"], n_trials=6, n_jobs=1)
            best.append(clu.best_params_.get("min_cluster_size"))
        assert best[0] is not None, "Optuna produced no hyper-parameters"
        assert best[0] == best[1], "seeded Optuna search did not reproduce"


class TestSweepStepSelection:
    """The rule that picks WHICH sweep step to keep, distinct from which label to return.

    Issue #7 was the label selector. This is its successor defect: even with the label
    resolved correctly from the condensed tree, the sweep-step rule can hand it a fit whose
    densest branch lies inside the field, and then no label choice can be right.

    Measured over 54 benchmark cells (erotica_3d), signed parameter-recovery error at
    p >= 0.5 -- the discriminator, because a rule that selects the field recovers the
    FIELD's centre and dispersion:

        rule              pmra_c    pmdec_c    plx_c    sigma_pm
        max_members       -0.581    -0.834    -0.239      0.505
        max_lambda        -0.006    -0.002     0.000     -0.054
        max_persistence   -0.007     0.001     0.001     -0.012

    Truth-free top-K purity 0.473 / 0.447 / 0.326 and Platt ECE 0.0089 / 0.0178 / 0.0385,
    both favouring max_persistence. On raw p-tilde AUC the three are 0.790 +/- 0.020,
    0.776 +/- 0.018 and 0.689 +/- 0.019: max_members and max_persistence are TIED within
    error, so no AUC win is claimed. Unlike the label selector, the sweep-step rule does
    change p-tilde, because it changes which fit is final.
    """

    def test_unknown_selection_is_rejected(self):
        from erotica.core.clustering import Clustering

        clust = Clustering(_make_contaminated_qtable())
        with pytest.raises(ValueError, match="max_persistence"):
            clust.search_pseudoprobability(
                columns=["pmra", "pmdec"],
                min_cluster_size_samples=range(10, 20),
                selection="largest",
            )

    def test_the_default_is_max_persistence(self):
        """Pins the default, because changing it silently rewrites what every other test means."""
        import inspect

        from erotica.core.clustering import Clustering

        sig = inspect.signature(Clustering.search_pseudoprobability)
        assert sig.parameters["selection"].default == "max_persistence"

    def test_every_documented_rule_runs_and_selects_something(self):
        """A rule that raises or returns nothing is not a rule. max_lambda in particular
        collapses to mcs_range.start, which is degenerate but must still be well-formed."""
        from erotica.core.clustering import Clustering

        for rule in ("max_persistence", "max_members", "max_lambda"):
            clust = Clustering(_make_contaminated_qtable())
            clust.search_pseudoprobability(
                columns=["pmra", "pmdec"],
                min_cluster_size_samples=range(10, 40),
                min_samples=5,
                probability_threshold=0.5,
                selection=rule,
            )
            sel = np.asarray(clust.data["cluster"], dtype=int) != -1
            assert sel.sum() > 0, f"{rule} selected nothing"
            assert "selected_persistence" in clust.pseudoprobability_selected_, (
                f"{rule} did not record selected_persistence, so the rules are not "
                "comparable on the same record"
            )

    def test_persistence_is_scored_on_the_cluster_actually_returned(self):
        """The point of the rule: the score belongs to the SELECTED cluster, not to the
        largest, and not to a row count. A step whose selector returns noise scores zero."""
        from erotica.core.clustering import Clustering

        clust = Clustering(_make_contaminated_qtable())
        clust.search_pseudoprobability(
            columns=["pmra", "pmdec"],
            min_cluster_size_samples=range(10, 40),
            min_samples=5,
            probability_threshold=0.5,
            selection="max_persistence",
        )
        chosen = clust.pseudoprobability_selected_
        persistence = np.asarray(chosen["cluster_persistence"], dtype=float)
        label = int(chosen["selected_label"])
        if label < 0:
            assert chosen["selected_persistence"] == 0.0
        else:
            assert chosen["selected_persistence"] == pytest.approx(float(persistence[label]))


class TestCoincidentRowGuard:
    """Duplicate rows silently destroy ``cluster_persistence_``, which the default rule ranks on.

    Mechanism, from the hdbscan source: >= min_samples identical rows give core distance 0, hence
    mutual reachability 0, hence lambda = INFTY (``_hdbscan_tree.pyx:112``), hence
    ``get_stability_scores`` (``:635``) takes the ``isinf`` branch and returns exactly 1.0 for
    EVERY cluster. ``selection="max_persistence"`` is an argmax over that quantity, so its result
    becomes arbitrary. Cross-matched catalogues produce repeated coordinates as a matter of course.
    """

    @staticmethod
    def _duplicated_table(n_dup: int, seed: int = 3) -> QTable:
        """Two real blobs plus ``n_dup`` byte-identical rows at a third location."""
        rng = np.random.default_rng(seed)
        a = rng.normal([0.0, 0.0], 0.3, (120, 2))
        b = rng.normal([6.0, 6.0], 0.3, (120, 2))
        dup = np.repeat(np.array([[-5.0, 5.0]]), n_dup, axis=0)
        pm = np.vstack([a, b, dup])
        return QTable({"pmra": pm[:, 0] * u.mas / u.yr, "pmdec": pm[:, 1] * u.mas / u.yr})

    def test_the_degeneracy_is_real_before_testing_the_warning(self):
        """Guard the guard: prove hdbscan actually collapses, so the warning is not decoration.

        If a future hdbscan stops returning all-1.0 persistence for coincident rows, this fails
        first and tells us the warning below is now protecting against nothing.
        """
        import hdbscan

        tbl = self._duplicated_table(n_dup=41)
        X = np.column_stack([np.asarray(tbl["pmra"], float), np.asarray(tbl["pmdec"], float)])
        pers = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=40).fit(X).cluster_persistence_
        assert pers.size > 0, "no clusters at all — the fixture stopped exercising the mechanism"
        # The trigger is min_samples+1, not min_samples: the core distance is the distance to the
        # k-th nearest OTHER point (k=min_samples+1 in every fit path), so exactly min_samples
        # coincident rows still leave it non-zero. Measured, and sharp:
        #   n_dup=40 -> max_lambda 3.113, persistence [0.7645 0.5795]
        #   n_dup=41 -> max_lambda inf,   persistence [1. 1. 1.]
        # This assertion caught the guard firing one row early.
        assert np.allclose(pers, 1.0), (
            f"expected the all-1.0 degeneracy from infinite lambda, got {pers}"
        )

    def test_warns_when_duplicates_reach_min_samples(self):
        from erotica.core.clustering import Clustering

        clust = Clustering(self._duplicated_table(n_dup=41))
        with pytest.warns(RuntimeWarning, match="cluster_persistence_ will be exactly 1.0"):
            clust.search_pseudoprobability(
                columns=["pmra", "pmdec"],
                min_cluster_size_samples=range(40, 46),
                min_samples=40,
                probability_threshold=0.5,
            )

    def test_the_warning_names_the_escape_hatch_for_the_default_rule(self):
        """A warning that does not say what to do instead gets ignored."""
        from erotica.core.clustering import Clustering

        info = (
            Clustering._warn_on_coincident_rows.__func__
            if hasattr(Clustering._warn_on_coincident_rows, "__func__")
            else Clustering._warn_on_coincident_rows
        )
        with pytest.warns(RuntimeWarning, match="selection='max_members'"):
            info(
                np.repeat(np.array([[1.0, 2.0]]), 12, axis=0),
                ["pmra", "pmdec"],
                [10],
                10,
                "max_persistence",
            )

    def test_clean_data_does_not_warn(self):
        """The negative control. A guard that always fires is noise, not a guard."""
        from erotica.core.clustering import Clustering

        rng = np.random.default_rng(11)
        X = rng.normal(0.0, 1.0, (400, 2))  # continuous: duplicates have probability zero
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            info = Clustering._warn_on_coincident_rows(
                X, ["pmra", "pmdec"], [10], 10, "max_persistence"
            )
        assert info["max_multiplicity"] == 1
        assert info["degenerate"] is False

    def test_reports_the_diagnostic_for_provenance(self):
        from erotica.core.clustering import Clustering

        with pytest.warns(RuntimeWarning):
            info = Clustering._warn_on_coincident_rows(
                np.repeat(np.array([[1.0, 2.0]]), 12, axis=0),
                ["pmra", "pmdec"],
                [10],
                10,
                "max_persistence",
            )
        assert info["max_multiplicity"] == 12
        assert info["degenerate"] is True
        assert info["n_distinct"] == 1


class TestSoftMembershipOption:
    """``probability_method="soft"`` replaces probabilities_ with the soft column.

    Rationale in the method docstring: probabilities_ is exactly 1.0 for ~84% of an EOM-merged
    cluster, so it cannot rank those points. These tests pin the plumbing, not the advantage --
    the advantage is measured in tools/validation/hdbscan_config_sweep.py.
    """

    def test_rejects_an_unknown_method(self, good_data):
        from erotica.core.clustering import Clustering

        with pytest.raises(ValueError, match="probability_method"):
            Clustering(good_data).search_pseudoprobability(
                columns=["pmra", "pmdec"],
                min_cluster_size_samples=range(10, 15),
                probability_method="bogus",
            )

    def test_soft_produces_a_different_probability_and_records_it(self, good_data, bad_data):
        from erotica.core.clustering import Clustering

        common = dict(
            columns=["pmra", "pmdec"],
            min_cluster_size_samples=range(10, 40),
            min_samples=5,
            probability_threshold=0.5,
        )
        hard = Clustering(good_data.copy(), bad_data)
        hard.search_pseudoprobability(**common, probability_method="hdbscan")
        soft = Clustering(good_data.copy(), bad_data)
        soft.search_pseudoprobability(**common, probability_method="soft")

        assert "probability_soft" in soft.data.colnames, "soft column not recorded"
        assert "probability_soft" not in hard.data.colnames, "hdbscan path must not fabricate it"

        p_hard = np.asarray(hard.data["probability"], dtype=float)
        p_soft = np.asarray(soft.data["probability"], dtype=float)
        assert not np.allclose(p_hard, p_soft), (
            "soft and hdbscan produced identical probabilities — the option is not wired through"
        )
        assert np.all((p_soft >= 0) & (p_soft <= 1)), "probability left [0, 1]"

    def test_the_saturation_this_replaces_is_real(self, good_data, bad_data):
        """Positive control: probabilities_ really is degenerate under EOM on this frame.

        If a future hdbscan stops saturating, the soft option loses its stated justification and
        this fails first, rather than the justification quietly becoming false.
        """
        from erotica.core.clustering import Clustering

        clust = Clustering(good_data.copy(), bad_data)
        clust.search_pseudoprobability(
            columns=["pmra", "pmdec"],
            min_cluster_size_samples=range(10, 40),
            min_samples=5,
            probability_threshold=0.5,
        )
        p = np.asarray(clust.data["probability_hdbscan"], dtype=float)
        in_cluster = np.asarray(clust.data["cluster_hdbscan"], dtype=int) != -1
        frac_saturated = float((p[in_cluster] == 1.0).mean())
        assert frac_saturated > 0.05, (
            f"only {frac_saturated:.3f} of clustered points saturate at exactly 1.0; the "
            "clamp that motivates probability_method='soft' is no longer present"
        )

    def test_falls_back_rather_than_raising_when_soft_is_unavailable(self, good_data):
        """An unavailable soft vector must cost a warning, not the run."""
        from erotica.core.clustering import Clustering

        clust = Clustering(good_data.copy())
        clust.clusterer = object()  # no prediction data, no condensed tree
        with pytest.warns(RuntimeWarning, match="falling back to probabilities_"):
            out = clust._soft_membership_column(selected_label=0, n_rows=10)
        assert out is None

    def test_a_negative_label_yields_no_column(self, good_data):
        from erotica.core.clustering import Clustering

        assert Clustering(good_data.copy())._soft_membership_column(-1, 10) is None
