"""Tests for the PUMPS-vs-ASteCA-vs-pyUPMASK benchmark harness.

All tests run on synthetic, co-indexed inputs with NO external clustering
package installed -- they exercise the metrics + report core only. The external
runners (ASteCA / pyUPMASK real execution) are NOT covered here; those must be
verified in the real env (see module docstring).

Verified: this suite was executed against numpy/scipy/scikit-learn/pandas in a
scratchpad (18 passed). Only the import path below changed for the real package.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from pumps.analysis.external.benchmark import (
    AgreementMetrics,
    BenchmarkReport,
    CalibrationMetrics,
    build_report,
    compute_agreement,
    compute_calibration,
    compute_parameter_recovery,
    run_asteca_membership,
    run_pyupmask_membership,
)


# ---------------------------------------------------------------------------
# Synthetic fixture: cluster + field with known membership and calibrated probs
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic():
    rng = np.random.default_rng(20260720)
    n = 3000
    # latent "clusterness" score; sigmoid gives a genuinely calibrated prob
    latent = rng.normal(0.0, 1.5, size=n)
    true_p = 1.0 / (1.0 + np.exp(-latent))
    truth = (rng.random(n) < true_p).astype(int)  # draw membership from calibrated p
    return {"rng": rng, "n": n, "true_p": true_p, "truth": truth, "latent": latent}


# ---------------------------------------------------------------------------
# (d) Calibration
# ---------------------------------------------------------------------------
def test_calibrated_probs_have_low_ece(synthetic):
    calib = compute_calibration(synthetic["true_p"], synthetic["truth"], n_bins=10)
    assert isinstance(calib, CalibrationMetrics)
    assert calib.n == synthetic["n"]
    assert 0.0 <= calib.brier <= 1.0
    # probabilities generated the labels -> well calibrated -> small ECE
    assert calib.ece < 0.05
    assert calib.mce < 0.20


def test_overconfident_probs_have_higher_ece(synthetic):
    good = compute_calibration(synthetic["true_p"], synthetic["truth"], n_bins=10)
    # distort toward 0/1: monotonic but miscalibrated
    bad_p = np.clip(synthetic["true_p"] * 3.0 - 1.0, 0.0, 1.0)
    bad = compute_calibration(bad_p, synthetic["truth"], n_bins=10)
    assert bad.ece > good.ece
    assert bad.brier >= good.brier


def test_calibration_curve_shape_and_empty_bins():
    probs = np.array([0.05, 0.06, 0.95, 0.96])
    truth = np.array([0, 0, 1, 1])
    calib = compute_calibration(probs, truth, n_bins=10)
    assert len(calib.curve.bin_centers) == 10
    assert len(calib.curve.counts) == 10
    assert sum(calib.curve.counts) == 4
    # middle bins are empty -> NaN, not a crash
    assert np.isnan(calib.curve.mean_predicted[5])
    assert calib.curve.counts[5] == 0


def test_calibration_empty_input_is_nan_not_crash():
    calib = compute_calibration(np.array([]), np.array([]), n_bins=5)
    assert calib.n == 0
    assert np.isnan(calib.brier) and np.isnan(calib.ece)


# ---------------------------------------------------------------------------
# (a) Agreement
# ---------------------------------------------------------------------------
def test_identical_probs_perfect_agreement(synthetic):
    p = synthetic["true_p"]
    agr = compute_agreement("A", p, "B", p, threshold=0.5)
    assert isinstance(agr, AgreementMetrics)
    assert agr.jaccard == pytest.approx(1.0)
    assert agr.overlap_coefficient == pytest.approx(1.0)
    assert agr.cohen_kappa == pytest.approx(1.0)
    assert agr.spearman_r == pytest.approx(1.0)
    assert agr.auc_a_scores_b == pytest.approx(1.0)
    assert agr.fp == 0 and agr.fn == 0


def test_independent_probs_low_agreement(synthetic):
    rng = synthetic["rng"]
    a = rng.random(synthetic["n"])
    b = rng.random(synthetic["n"])
    agr = compute_agreement("A", a, "B", b, threshold=0.5)
    # independent -> AUC near chance, kappa near 0 (loose bounds to avoid flakes)
    assert 0.4 < agr.auc_a_scores_b < 0.6
    assert abs(agr.cohen_kappa) < 0.15
    assert abs(agr.spearman_r) < 0.1


def test_agreement_confusion_counts_sum_to_n(synthetic):
    rng = synthetic["rng"]
    a = rng.random(synthetic["n"])
    b = rng.random(synthetic["n"])
    agr = compute_agreement("A", a, "B", b, threshold=0.5)
    assert agr.tp + agr.fp + agr.fn + agr.tn == synthetic["n"]


def test_agreement_single_class_reference_gives_nan_auc():
    # method B says "everyone is a member" -> AUC of A's probs vs B labels is NaN
    a = np.array([0.1, 0.4, 0.6, 0.9])
    b = np.array([0.9, 0.9, 0.9, 0.9])  # all members at 0.5 threshold
    agr = compute_agreement("A", a, "B", b, threshold=0.5)
    # B's labels are single-class (all members) -> ranking them by A's probs is NaN
    assert np.isnan(agr.auc_a_scores_b)
    # A's labels are two-class but B's scores are constant -> sklearn = 0.5 (chance)
    assert agr.auc_b_scores_a == pytest.approx(0.5)


def test_agreement_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_agreement("A", np.zeros(5), "B", np.zeros(6))


# ---------------------------------------------------------------------------
# (b) Parameter recovery
# ---------------------------------------------------------------------------
def test_parameter_recovery_arithmetic():
    truth = {"loga": 7.0, "dm": 10.30, "Av": 1.24}
    recovered = {"loga": 7.10, "dm": 10.10, "Av": 1.24}
    unc = {"loga": 0.20, "dm": 0.10}
    recs = compute_parameter_recovery("ASteCA", recovered, truth, uncertainties=unc)
    by = {r.param: r for r in recs}
    assert by["loga"].abs_error == pytest.approx(0.10)
    assert by["loga"].rel_error == pytest.approx(0.10 / 7.0)
    assert by["loga"].z_score == pytest.approx(0.5)
    assert by["dm"].z_score == pytest.approx(-2.0)
    # Av recovered exactly, no uncertainty -> z is NaN
    assert by["Av"].abs_error == pytest.approx(0.0)
    assert np.isnan(by["Av"].z_score)


def test_parameter_recovery_zero_truth_rel_error_nan():
    recs = compute_parameter_recovery("m", {"x": 0.5}, {"x": 0.0})
    assert np.isnan(recs[0].rel_error)
    assert recs[0].abs_error == pytest.approx(0.5)


def test_parameter_recovery_only_shared_keys():
    recs = compute_parameter_recovery("m", {"a": 1.0}, {"a": 1.0, "b": 2.0})
    assert [r.param for r in recs] == ["a"]


# ---------------------------------------------------------------------------
# Orchestrator / report
# ---------------------------------------------------------------------------
def test_build_report_end_to_end(synthetic):
    true_p = synthetic["true_p"]
    truth = synthetic["truth"]
    rng = synthetic["rng"]
    # three "methods" as noisy monotone views of the same latent membership
    cosmic = np.clip(true_p + rng.normal(0, 0.05, true_p.size), 0, 1)
    asteca = np.clip(true_p + rng.normal(0, 0.10, true_p.size), 0, 1)
    pyupmask = np.clip(true_p + rng.normal(0, 0.15, true_p.size), 0, 1)

    report = build_report(
        "synthetic_cluster",
        {"PUMPS": cosmic, "ASteCA": asteca, "pyUPMASK": pyupmask},
        truth_membership=truth,
        truth_params={"loga": 7.0, "dm": 10.3},
        recovered_params={
            "PUMPS": {"loga": 7.02, "dm": 10.28},
            "ASteCA": {"loga": 6.95, "dm": 10.35},
        },
        param_uncertainties={"PUMPS": {"loga": 0.1, "dm": 0.1}},
        runtimes={"PUMPS": 12.5, "ASteCA": 40.0, "pyUPMASK": 88.0},
        notes=("synthetic smoke test",),
    )
    assert isinstance(report, BenchmarkReport)
    assert report.n_sources == synthetic["n"]
    assert report.truth_available is True
    assert len(report.methods) == 3
    # 3 methods -> 3 unique unordered pairs
    assert len(report.agreements) == 3

    calib_df = report.calibration_table()
    assert set(calib_df["method"]) == {"PUMPS", "ASteCA", "pyUPMASK"}
    # least-noisy method should be best calibrated
    ece = dict(zip(calib_df["method"], calib_df["ece"]))
    assert ece["PUMPS"] <= ece["pyUPMASK"]

    agr_df = report.agreement_table()
    assert len(agr_df) == 3
    rec_df = report.recovery_table()
    assert set(rec_df["method"]) == {"PUMPS", "ASteCA"}
    rt_df = report.runtime_table()
    assert rt_df.set_index("method").loc["PUMPS", "runtime_s"] == 12.5


def test_build_report_without_truth_skips_calibration(synthetic):
    p = synthetic["true_p"]
    report = build_report("no_truth", {"PUMPS": p, "ASteCA": p})
    assert report.truth_available is False
    assert all(m.calibration is None for m in report.methods)
    assert report.calibration_table().empty
    assert len(report.agreements) == 1


def test_build_report_length_mismatch_raises():
    with pytest.raises(ValueError):
        build_report("bad", {"A": np.zeros(5), "B": np.zeros(4)})


# ---------------------------------------------------------------------------
# Lazy-import guards on the external runners (deps absent in this env)
# ---------------------------------------------------------------------------
_HAVE_ASTECA = importlib.util.find_spec("asteca") is not None


@pytest.mark.skipif(_HAVE_ASTECA, reason="asteca installed; import guard not exercised")
def test_asteca_runner_raises_importerror_without_dep():
    with pytest.raises(ImportError, match="ASteCA"):
        run_asteca_membership(cluster_kwargs={}, membership_fn=lambda ac, clu: [0.0])


def test_pyupmask_runner_requires_source():
    with pytest.raises(ValueError, match="results_path"):
        run_pyupmask_membership()


def test_pyupmask_runner_reads_results_csv(tmp_path):
    import pandas as pd

    csv = tmp_path / "pyupmask_out.csv"
    pd.DataFrame({"probability": [0.1, 0.8, 0.95]}).to_csv(csv, index=False)
    probs, secs = run_pyupmask_membership(results_path=str(csv))
    assert np.allclose(probs, [0.1, 0.8, 0.95])
    assert secs >= 0.0

