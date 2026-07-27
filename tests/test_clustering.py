"""Tests for erotica.core.clustering — Clustering and HDBSCANEstimator."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import QTable
from astropy import units as u


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_qtable(n: int = 300, seed: int = 42) -> QTable:
    """Two well-separated clusters in pmra/pmdec + required columns."""
    rng = np.random.default_rng(seed)
    half = n // 2
    pmra  = np.concatenate([rng.normal(0.0, 0.3, half), rng.normal(5.0, 0.3, half)])
    pmdec = np.concatenate([rng.normal(0.0, 0.3, half), rng.normal(5.0, 0.3, half)])
    return QTable({
        "pmra":  pmra  * u.mas / u.yr,
        "pmdec": pmdec * u.mas / u.yr,
        "ra":    rng.uniform(263, 264, n) * u.deg,
        "dec":   rng.uniform(-32, -31, n) * u.deg,
        "cluster": np.zeros(n, dtype=int),
    })


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
        from erotica.core.clustering import Clustering
        import matplotlib
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
        from erotica.core.clustering import Clustering
        import matplotlib
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
            member_sets.append(set(zip(pmra, pmdec)))
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
