"""Tests for cosmic.calibration — membership-probability calibration.

The synthetic design: draw true event rates ``p_true`` and Bernoulli labels
``y ~ Bernoulli(p_true)``. Then

* well-calibrated predictions  = ``p_true``            (observed ≈ predicted),
* mis-calibrated predictions   = ``sqrt(p_true)``      (observed = predicted²,
  i.e. the reliability curve bows below the diagonal — systematic overconfidence).

For the mis-calibrated case isotonic regression should recover the mapping
``v ↦ v²`` and thus undo the distortion.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmic.calibration import (
    CalibrationReport,
    HosmerLemeshowResult,
    ReliabilityDiagram,
    Recalibrator,
    brier_score,
    calibration_report,
    expected_calibration_error,
    fit_isotonic,
    fit_platt,
    hosmer_lemeshow,
    recalibrate,
    reliability_diagram,
)

N = 20_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def calibrated():
    """(predicted, labels) where predicted == true event rate."""
    rng = np.random.default_rng(20240720)
    p_true = rng.uniform(0.02, 0.98, N)
    y = rng.binomial(1, p_true).astype(float)
    return p_true, y


@pytest.fixture(scope="module")
def miscalibrated():
    """(predicted, labels) where predicted = sqrt(true rate) → overconfident."""
    rng = np.random.default_rng(11111)
    p_true = rng.uniform(0.02, 0.98, N)
    y = rng.binomial(1, p_true).astype(float)
    p_pred = np.sqrt(p_true)  # observed rate given prediction v is v**2 < v
    return p_pred, y


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------
class TestReliabilityDiagram:
    def test_calibrated_lies_on_diagonal(self, calibrated):
        p, y = calibrated
        diag = reliability_diagram(p, y, n_bins=10)
        nonempty = diag.counts > 0
        # observed ≈ predicted within Monte-Carlo noise for every populated bin
        assert np.allclose(
            diag.observed_frequency[nonempty], diag.mean_predicted[nonempty], atol=0.05
        )

    def test_miscalibrated_bows_below_diagonal(self, miscalibrated):
        p, y = miscalibrated
        diag = reliability_diagram(p, y, n_bins=10)
        nonempty = diag.counts > 0
        # observed = predicted**2 < predicted → observed strictly below diagonal
        gap = diag.mean_predicted[nonempty] - diag.observed_frequency[nonempty]
        assert np.all(gap > 0)
        assert np.max(gap) > 0.1  # a large, visible miscalibration

    def test_shape_and_counts(self, calibrated):
        p, y = calibrated
        diag = reliability_diagram(p, y, n_bins=10)
        assert isinstance(diag, ReliabilityDiagram)
        assert diag.counts.sum() == N
        assert len(diag.bin_edges) == diag.n_bins + 1

    def test_quantile_strategy_runs(self, calibrated):
        p, y = calibrated
        diag = reliability_diagram(p, y, n_bins=10, strategy="quantile")
        assert diag.counts.sum() == N

    def test_plot_returns_axis(self, calibrated):
        import matplotlib

        matplotlib.use("Agg")
        p, y = calibrated
        diag = reliability_diagram(p, y, n_bins=10)
        ax = diag.plot(show_counts=True)
        assert ax is not None


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------
class TestBrierScore:
    def test_matches_direct_formula(self, calibrated):
        p, y = calibrated
        assert brier_score(p, y) == pytest.approx(float(np.mean((p - y) ** 2)))

    def test_perfect_predictions_zero(self):
        y = np.array([0, 1, 1, 0, 1], dtype=float)
        assert brier_score(y, y) == pytest.approx(0.0)

    def test_calibrated_beats_miscalibrated(self, calibrated, miscalibrated):
        assert brier_score(*calibrated) < brier_score(*miscalibrated)


# ---------------------------------------------------------------------------
# Expected / maximum calibration error
# ---------------------------------------------------------------------------
class TestExpectedCalibrationError:
    def test_calibrated_small(self, calibrated):
        ece, mce = expected_calibration_error(*calibrated, n_bins=10)
        assert ece < 0.03
        assert mce < 0.06

    def test_miscalibrated_large(self, miscalibrated):
        ece, _ = expected_calibration_error(*miscalibrated, n_bins=10)
        assert ece > 0.1

    def test_ordering(self, calibrated, miscalibrated):
        ece_cal, _ = expected_calibration_error(*calibrated)
        ece_mis, _ = expected_calibration_error(*miscalibrated)
        assert ece_cal < ece_mis


# ---------------------------------------------------------------------------
# Hosmer–Lemeshow
# ---------------------------------------------------------------------------
class TestHosmerLemeshow:
    def test_result_type_and_dof(self, calibrated):
        res = hosmer_lemeshow(*calibrated, n_groups=10)
        assert isinstance(res, HosmerLemeshowResult)
        assert res.dof == 8  # n_groups - 2
        assert np.isfinite(res.statistic)
        assert 0.0 <= res.pvalue <= 1.0

    def test_miscalibrated_is_rejected(self, miscalibrated):
        res = hosmer_lemeshow(*miscalibrated, n_groups=10)
        assert res.pvalue < 0.01  # strong evidence against calibration

    def test_calibrated_statistic_much_smaller(self, calibrated, miscalibrated):
        # Direction, not an absolute p-value (a true H-L p-value is ~Uniform and
        # would flake ~5% of the time against a fixed threshold).
        cal = hosmer_lemeshow(*calibrated, n_groups=10)
        mis = hosmer_lemeshow(*miscalibrated, n_groups=10)
        assert cal.statistic < mis.statistic
        assert cal.pvalue > mis.pvalue

    def test_too_few_groups_raises(self, calibrated):
        with pytest.raises(ValueError):
            hosmer_lemeshow(*calibrated, n_groups=2)

    def test_point_mass_at_zero_does_not_crash(self):
        # Real COSMIC p̃ has a spike of exact zeros (field stars, p_HDBSCAN=0).
        # The equal-count grouping must handle the tie-block without blowing up.
        rng = np.random.default_rng(7)
        p_field = np.zeros(5_000)
        y_field = np.zeros(5_000)
        p_cluster = rng.uniform(0.3, 0.98, 5_000)
        y_cluster = rng.binomial(1, p_cluster).astype(float)
        p = np.concatenate([p_field, p_cluster])
        y = np.concatenate([y_field, y_cluster])
        res = hosmer_lemeshow(p, y, n_groups=10)
        assert np.isfinite(res.statistic)
        assert 0.0 <= res.pvalue <= 1.0


# ---------------------------------------------------------------------------
# calibration_report bundle
# ---------------------------------------------------------------------------
class TestCalibrationReport:
    def test_bundle_fields(self, calibrated):
        rep = calibration_report(*calibrated)
        assert isinstance(rep, CalibrationReport)
        assert rep.n_samples == N
        assert isinstance(rep.hosmer_lemeshow, HosmerLemeshowResult)
        assert isinstance(rep.reliability, ReliabilityDiagram)
        assert rep.brier_score == pytest.approx(brier_score(*calibrated))

    def test_report_discriminates(self, calibrated, miscalibrated):
        good = calibration_report(*calibrated)
        bad = calibration_report(*miscalibrated)
        assert good.expected_calibration_error < bad.expected_calibration_error
        assert good.brier_score < bad.brier_score
        assert good.hosmer_lemeshow.statistic < bad.hosmer_lemeshow.statistic


# ---------------------------------------------------------------------------
# Recalibration
# ---------------------------------------------------------------------------
class TestRecalibration:
    def test_isotonic_reduces_ece(self, miscalibrated):
        p, y = miscalibrated
        recal = fit_isotonic(p, y)
        assert isinstance(recal, Recalibrator)
        ece_before, _ = expected_calibration_error(p, y)
        ece_after, _ = expected_calibration_error(recal(p), y)
        assert ece_after < ece_before
        assert ece_after < 0.03  # isotonic recovers v→v², i.e. ≈ calibrated

    def test_platt_reduces_ece(self, miscalibrated):
        p, y = miscalibrated
        recal = fit_platt(p, y)
        ece_before, _ = expected_calibration_error(p, y)
        ece_after, _ = expected_calibration_error(recal(p), y)
        assert ece_after < ece_before

    def test_output_in_unit_interval(self, miscalibrated):
        p, y = miscalibrated
        for method in ("isotonic", "platt"):
            out = recalibrate(p, y, method=method)(p)
            assert out.min() >= 0.0
            assert out.max() <= 1.0
            assert out.shape == p.shape

    def test_isotonic_recovers_square_mapping(self, miscalibrated):
        # predicted v should map back to ≈ v**2 (the true event rate).
        p, y = miscalibrated
        recal = fit_isotonic(p, y)
        grid = np.array([0.2, 0.4, 0.6, 0.8])
        mapped = recal(grid)
        assert np.allclose(mapped, grid**2, atol=0.05)

    def test_calibrated_input_stays_calibrated(self, calibrated):
        # Recalibrating already-good probabilities must not make them worse.
        p, y = calibrated
        ece_before, _ = expected_calibration_error(p, y)
        ece_after, _ = expected_calibration_error(fit_platt(p, y)(p), y)
        assert ece_after < ece_before + 0.01

    def test_single_class_raises(self):
        p = np.array([0.3, 0.4, 0.5])
        y = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError):
            fit_isotonic(p, y)
        with pytest.raises(ValueError):
            fit_platt(p, y)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_probability_out_of_range_raises(self):
        with pytest.raises(ValueError):
            brier_score([0.1, 1.5], [0, 1])

    def test_nonbinary_labels_raise(self):
        with pytest.raises(ValueError):
            brier_score([0.1, 0.9], [0, 2])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            brier_score([0.1, 0.2, 0.3], [0, 1])

    def test_nan_probability_raises(self):
        with pytest.raises(ValueError):
            brier_score([0.1, np.nan], [0, 1])

    def test_bool_labels_accepted(self):
        p = np.array([0.2, 0.8, 0.6, 0.4])
        y = np.array([False, True, True, False])
        assert np.isfinite(brier_score(p, y))


# ---------------------------------------------------------------------------
# Integration convenience (duck-typed, no cosmic import)
# ---------------------------------------------------------------------------
class TestFromClustering:
    def test_pulls_probability_column(self, calibrated):
        from cosmic.calibration import calibration_report_from_clustering

        p, y = calibrated

        class _FakeTable:
            colnames = ["probability"]

            def __getitem__(self, key):
                return p

        class _FakeClustering:
            data = _FakeTable()

        rep = calibration_report_from_clustering(_FakeClustering(), y)
        assert isinstance(rep, CalibrationReport)
        assert rep.n_samples == N

    def test_missing_column_raises(self):
        class _Empty:
            data = None

        with pytest.raises(ValueError):
            from cosmic.calibration import calibration_report_from_clustering

            calibration_report_from_clustering(_Empty(), [0, 1])

