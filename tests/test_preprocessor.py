"""Value-checking tests for the PUMPS scientific preprocessing corrections.

These exercise the physics/astrometry helpers in ``pumps/preprocess/_helpers.py``:

* ``correct_proper_motion`` -- the Cantat-Gaudin & Brandt (2021) proper-motion
  "spin" correction applied to bright (G < 13) Gaia sources, using the tabulated
  omega vector in :data:`pumps.preprocess._constants.PM_CORRECTION_ROWS`.
* ``add_photometric_errors`` / its nested ``_calculate_mag_error`` -- magnitude
  errors propagated from Gaia flux / flux_error.
* ``split_by_fidelity`` -- high/low fidelity partition at a threshold.
* ``apply_zero_point_correction`` -- Gaia DR3 parallax zero-point correction,
  driven by an injectable ``zpt_module`` (so no external ``gaiadr3-zeropoint``
  package is needed).

IMPORT TRAP: ``import pumps.preprocess`` runs ``preprocess/__init__`` which
imports ``preprocessor`` -> ``from zero_point import zpt``.  The external
``gaiadr3-zeropoint`` package is not installed in the test environment, so that
import raises ``ModuleNotFoundError``.  We dodge it exactly like
``tests/test_io_helpers.py``: stub a bare ``pumps.preprocess`` package in
``sys.modules`` (so the relative ``from ._constants import ...`` still resolves
through ``__path__``) and load ``_helpers`` directly from its file, never
executing the package ``__init__``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

import pumps


def _load_preprocess_helpers():
    """Load ``pumps.preprocess._helpers`` without executing the package
    ``__init__`` (which would trip the missing ``zero_point`` import)."""
    pkg_name = "pumps.preprocess"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(Path(pumps.__file__).parent / "preprocess")]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg
    mod_name = "pumps.preprocess._helpers"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name, Path(pumps.__file__).parent / "preprocess" / "_helpers.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


helpers = _load_preprocess_helpers()
correct_proper_motion = helpers.correct_proper_motion
add_photometric_errors = helpers.add_photometric_errors
split_by_fidelity = helpers.split_by_fidelity
apply_zero_point_correction = helpers.apply_zero_point_correction
PM_CORRECTION_ROWS = helpers.PM_CORRECTION_ROWS


def _pm_row_for(g_mag: float):
    """Return ``(omega_x, omega_y, omega_z)`` [uas/yr] for magnitude *g_mag*
    straight from the tabulated CG&B correction, mirroring the lookup in
    ``correct_proper_motion`` (``gmin <= G < gmax``)."""
    for gmin, gmax, omega_x, omega_y, omega_z in PM_CORRECTION_ROWS:
        if gmin <= g_mag < gmax:
            return omega_x, omega_y, omega_z
    raise AssertionError(f"no PM-correction row for G={g_mag}")


# ---------------------------------------------------------------------------
# 1. correct_proper_motion -- CG&B (2021) bright-source spin correction
# ---------------------------------------------------------------------------
#
# For a bright source (G < 13) the code subtracts, in mas/yr:
#   pmra  -= pmra_corr  / 1000,  pmra_corr  = -sin d cos a wx - sin d sin a wy + cos d wz
#   pmdec -= pmdec_corr / 1000,  pmdec_corr =  sin a wx - cos a wy
# with (wx, wy, wz) the tabulated omega for that magnitude bin (in uas/yr).
#
# At (ra=0, dec=0):   pmra_corr =  wz,   pmdec_corr = -wy   (isolates wz and wy)
# At (ra=90, dec=0):  pmra_corr =  wz,   pmdec_corr =  wx   (isolates wx)


def test_correct_proper_motion_bright_star_ra0_dec0():
    """Bright star at (ra=0, dec=0): shift ties directly to tabulated wz, wy."""
    g_mag = 10.25  # falls in the [10.0, 10.5) bin -> (13.6, 35.7, -10.5)
    wx, wy, wz = _pm_row_for(g_mag)
    assert (wx, wy, wz) == (13.6, 35.7, -10.5)  # guards the constant table itself

    pmra0, pmdec0 = 5.0, -3.0
    table = QTable()
    table["Gmag"] = [g_mag] * u.mag
    table["ra"] = [0.0] * u.deg
    table["dec"] = [0.0] * u.deg
    table["pmra"] = [pmra0] * (u.mas / u.yr)
    table["pmdec"] = [pmdec0] * (u.mas / u.yr)

    correct_proper_motion(table)

    # Expected closed form at (0, 0):
    #   delta_pmra  = -wz / 1000 = +0.0105
    #   delta_pmdec = +wy / 1000 = +0.0357
    exp_pmra = pmra0 - wz / 1000.0
    exp_pmdec = pmdec0 - (-wy) / 1000.0
    assert exp_pmra == pytest.approx(5.0105)
    assert exp_pmdec == pytest.approx(-2.9643)

    assert table["pmra"].to_value(u.mas / u.yr)[0] == pytest.approx(exp_pmra, abs=1e-9)
    assert table["pmdec"].to_value(u.mas / u.yr)[0] == pytest.approx(exp_pmdec, abs=1e-9)

    # The uncorrected values are preserved in *_obs columns.
    assert table["pmra_obs"].to_value(u.mas / u.yr)[0] == pytest.approx(pmra0)
    assert table["pmdec_obs"].to_value(u.mas / u.yr)[0] == pytest.approx(pmdec0)

    # The applied correction magnitude matches the tabulated omega components.
    applied_ra = (pmra0 - table["pmra"].to_value(u.mas / u.yr)[0]) * 1000.0  # uas/yr
    applied_dec = (pmdec0 - table["pmdec"].to_value(u.mas / u.yr)[0]) * 1000.0
    assert applied_ra == pytest.approx(wz, abs=1e-6)       # -10.5 uas/yr
    assert applied_dec == pytest.approx(-wy, abs=1e-6)     # -35.7 uas/yr


def test_correct_proper_motion_covers_omega_x():
    """Bright star at (ra=90, dec=0) isolates the wx component in pmdec.

    Without this case wx is multiplied by sin(ra)=0 / sin(dec)=0 and never
    tested -- a corrupted wx column would slip through silently.
    """
    g_mag = 10.25
    wx, wy, wz = _pm_row_for(g_mag)

    pmra0, pmdec0 = 4.0, 6.0
    table = QTable()
    table["Gmag"] = [g_mag] * u.mag
    table["ra"] = [90.0] * u.deg
    table["dec"] = [0.0] * u.deg
    table["pmra"] = [pmra0] * (u.mas / u.yr)
    table["pmdec"] = [pmdec0] * (u.mas / u.yr)

    correct_proper_motion(table)

    # At (90, 0): pmra_corr = wz, pmdec_corr = wx
    exp_pmra = pmra0 - wz / 1000.0          # 4.0 + 0.0105
    exp_pmdec = pmdec0 - wx / 1000.0        # 6.0 - 0.0136
    assert exp_pmdec == pytest.approx(6.0 - 0.0136)
    assert table["pmra"].to_value(u.mas / u.yr)[0] == pytest.approx(exp_pmra, abs=1e-9)
    assert table["pmdec"].to_value(u.mas / u.yr)[0] == pytest.approx(exp_pmdec, abs=1e-9)

    applied_dec = (pmdec0 - table["pmdec"].to_value(u.mas / u.yr)[0]) * 1000.0  # uas/yr
    assert applied_dec == pytest.approx(wx, abs=1e-6)  # 13.6 uas/yr


def test_correct_proper_motion_faint_star_untouched():
    """A faint star (G >= 13) must be returned exactly unchanged (no bin match)."""
    pmra0, pmdec0 = 7.0, 2.0
    table = QTable()
    table["Gmag"] = [15.0] * u.mag  # > 13 -> excluded from correction
    table["ra"] = [123.4] * u.deg
    table["dec"] = [-45.6] * u.deg
    table["pmra"] = [pmra0] * (u.mas / u.yr)
    table["pmdec"] = [pmdec0] * (u.mas / u.yr)

    correct_proper_motion(table)

    # Subtracting exactly zero -> exact equality, not approx.
    assert table["pmra"].to_value(u.mas / u.yr)[0] == pmra0
    assert table["pmdec"].to_value(u.mas / u.yr)[0] == pmdec0


def test_correct_proper_motion_bright_and_faint_together():
    """Mixed table: bright row corrected, faint row simultaneously untouched."""
    g_bright = 10.25
    _, wy, wz = _pm_row_for(g_bright)
    table = QTable()
    table["Gmag"] = [g_bright, 15.0] * u.mag
    table["ra"] = [0.0, 0.0] * u.deg
    table["dec"] = [0.0, 0.0] * u.deg
    table["pmra"] = [5.0, 7.0] * (u.mas / u.yr)
    table["pmdec"] = [-3.0, 2.0] * (u.mas / u.yr)

    correct_proper_motion(table)

    pmra = table["pmra"].to_value(u.mas / u.yr)
    pmdec = table["pmdec"].to_value(u.mas / u.yr)
    # Bright row shifted; faint row bit-for-bit unchanged.
    assert pmra[0] == pytest.approx(5.0 - wz / 1000.0, abs=1e-9)
    assert pmdec[0] == pytest.approx(-3.0 + wy / 1000.0, abs=1e-9)
    assert pmra[1] == 7.0
    assert pmdec[1] == 2.0


def test_correct_proper_motion_requires_columns():
    """Missing astrometric columns raise a clear ValueError."""
    table = QTable()
    table["Gmag"] = [10.0] * u.mag
    with pytest.raises(ValueError, match="proper motion"):
        correct_proper_motion(table)


# ---------------------------------------------------------------------------
# 2. add_photometric_errors -- magnitude error from flux / flux_error
# ---------------------------------------------------------------------------
#
# The implemented estimator is the asymmetric magnitude error
#   sigma_m = 2.5 * log10(1 + flux_error/flux)
# which agrees with the classic linear propagation (2.5/ln10)*(flux_error/flux)
# only to first order in the (small) flux ratio.  We assert the *implemented*
# form exactly, and separately confirm it tracks the linear form for a small
# ratio.


def _make_flux_table(g_flux, g_err, bp_flux, bp_err, rp_flux, rp_err):
    table = QTable()
    table["phot_g_mean_flux"] = g_flux * (u.electron / u.s)
    table["phot_g_mean_flux_error"] = g_err * (u.electron / u.s)
    table["phot_bp_mean_flux"] = bp_flux * (u.electron / u.s)
    table["phot_bp_mean_flux_error"] = bp_err * (u.electron / u.s)
    table["phot_rp_mean_flux"] = rp_flux * (u.electron / u.s)
    table["phot_rp_mean_flux_error"] = rp_err * (u.electron / u.s)
    return table


def test_add_photometric_errors_analytic_value():
    """e_Gmag equals 2.5*log10(1 + flux_error/flux) for known inputs."""
    table = _make_flux_table(
        g_flux=[1000.0], g_err=[10.0],       # ratio 0.010
        bp_flux=[500.0], bp_err=[10.0],      # ratio 0.020
        rp_flux=[2000.0], rp_err=[10.0],     # ratio 0.005
    )

    created = add_photometric_errors(table)
    assert created == ["e_Gmag", "e_G_BPmag", "e_G_RPmag", "e_BP_RP"]

    exp_g = 2.5 * np.log10(1 + 10.0 / 1000.0)
    exp_bp = 2.5 * np.log10(1 + 10.0 / 500.0)
    exp_rp = 2.5 * np.log10(1 + 10.0 / 2000.0)

    e_g = table["e_Gmag"].to_value(u.mag)[0]
    e_bp = table["e_G_BPmag"].to_value(u.mag)[0]
    e_rp = table["e_G_RPmag"].to_value(u.mag)[0]

    assert e_g == pytest.approx(exp_g)
    assert e_g == pytest.approx(0.010803434456606446)  # concrete implemented value
    assert e_bp == pytest.approx(exp_bp)
    assert e_rp == pytest.approx(exp_rp)

    # BP-RP error is the quadrature sum of the BP and RP magnitude errors.
    exp_bprp = np.sqrt(exp_bp**2 + exp_rp**2)
    assert table["e_BP_RP"].to_value(u.mag)[0] == pytest.approx(exp_bprp)

    # Sanity: for a small ratio the implemented form tracks the classic
    # (2.5/ln10)*ratio propagation to ~1e-4 (they are NOT identical).
    linear_g = (2.5 / np.log(10)) * (10.0 / 1000.0)
    assert e_g == pytest.approx(linear_g, abs=1e-4)
    assert e_g != pytest.approx(linear_g, abs=1e-9)


def test_add_photometric_errors_edge_cases():
    """Zero / negative flux are handled without raising (clipped sentinels).

    * zero flux with positive error -> ratio = +inf -> sigma_m = +inf
    * negative flux                 -> ratio clipped to 0 -> sigma_m = 0
    """
    table = _make_flux_table(
        g_flux=[0.0, -500.0], g_err=[5.0, 5.0],
        bp_flux=[0.0, -500.0], bp_err=[5.0, 5.0],
        rp_flux=[0.0, -500.0], rp_err=[5.0, 5.0],
    )

    add_photometric_errors(table)  # must not raise

    e_g = table["e_Gmag"].to_value(u.mag)
    assert np.isinf(e_g[0])       # zero flux -> infinite error sentinel
    assert e_g[0] > 0
    assert e_g[1] == 0.0          # negative flux -> ratio clipped to 0


def test_add_photometric_errors_requires_columns():
    """Missing flux columns raise a clear ValueError."""
    table = QTable()
    table["phot_g_mean_flux"] = [1000.0] * (u.electron / u.s)
    with pytest.raises(ValueError, match="photometric errors"):
        add_photometric_errors(table)


# ---------------------------------------------------------------------------
# 3. split_by_fidelity -- high/low partition at a strict threshold
# ---------------------------------------------------------------------------


def test_split_by_fidelity_default_threshold():
    """Rows with fidelity strictly > 0.5 are "good"; the boundary 0.5 is "bad"."""
    table = QTable()
    # 0.90 good, 0.51 good, 0.50 bad (strict >), 0.49 bad, 0.00 bad
    table["fidelity_v2"] = np.array([0.90, 0.51, 0.50, 0.49, 0.00])

    good, bad, stats = split_by_fidelity(table)

    assert list(good["fidelity_v2"]) == pytest.approx([0.90, 0.51])
    assert list(bad["fidelity_v2"]) == pytest.approx([0.50, 0.49, 0.00])
    assert len(good) == 2
    assert len(bad) == 3
    assert stats["high_fidelity"] == 2
    assert stats["low_fidelity"] == 3
    assert stats["good_fraction"] == pytest.approx(2 / 5)
    # Boundary value 0.50 must land in the low-fidelity subset (strict >).
    assert 0.50 in list(bad["fidelity_v2"])
    assert 0.50 not in list(good["fidelity_v2"])


def test_split_by_fidelity_custom_column_and_threshold():
    """Custom column name and threshold are honoured."""
    table = QTable()
    table["fidelity"] = np.array([0.95, 0.80, 0.79, 0.10])

    good, bad, stats = split_by_fidelity(
        table, fidelity_column="fidelity", fidelity_threshold=0.8
    )

    # > 0.8 -> only 0.95 (0.80 is NOT strictly greater)
    assert len(good) == 1
    assert len(bad) == 3
    assert good["fidelity"][0] == pytest.approx(0.95)
    assert stats["good_fraction"] == pytest.approx(1 / 4)


def test_split_by_fidelity_missing_column():
    """A missing fidelity column raises a clear ValueError."""
    table = QTable()
    table["something_else"] = np.array([0.1, 0.9])
    with pytest.raises(ValueError, match="fidelity"):
        split_by_fidelity(table)


# ---------------------------------------------------------------------------
# 4. apply_zero_point_correction -- injectable zpt_module
# ---------------------------------------------------------------------------


class _FakeZpt:
    """Stand-in for ``gaiadr3-zeropoint``'s ``zpt`` with a fixed offset [mas]."""

    def __init__(self, offset):
        self.offset = offset
        self.calls = []

    def get_zpt(self, gmag, nu_eff, pseudocolour, ecl_lat, astrometric_params_solved):
        self.calls.append(1)
        return np.full(len(np.atleast_1d(gmag)), self.offset)


def _make_zpt_table(parallax):
    table = QTable()
    table["parallax"] = parallax * u.mas
    table["Gmag"] = [10.0, 11.0] * u.mag
    table["nu_eff_used_in_astrometry"] = [1.5, 1.5] / u.um
    table["pseudocolour"] = [1.5, 1.5] / u.um  # inside PSEUDOCOLOUR_RANGE (1.24-1.72)
    table["ecl_lat"] = [10.0, 20.0] * u.deg
    table["astrometric_params_solved"] = [31, 95]
    return table


def test_apply_zero_point_correction_subtracts_offset():
    """parallax_corrected = parallax_observed - offset, with the fake module."""
    offset = 0.02  # mas
    table = _make_zpt_table([1.0, 2.0])
    fake = _FakeZpt(offset)

    result = apply_zero_point_correction(table, fake)

    assert result["applied"] is True
    assert result["out_of_range"] == 0
    assert fake.calls, "the injected zpt_module.get_zpt was never called"

    # Original parallax stashed unchanged.
    assert table["parallax_observed"].to_value(u.mas) == pytest.approx([1.0, 2.0])
    # zpvals hold the applied offset.
    assert np.asarray(table["zpvals"]) == pytest.approx([offset, offset])
    # Corrected parallax = observed - offset, and keeps mas units.
    assert table["parallax"].unit == u.mas
    assert table["parallax"].to_value(u.mas) == pytest.approx([1.0 - offset, 2.0 - offset])


def test_apply_zero_point_correction_nan_offset_becomes_zero():
    """NaN corrections from the module are treated as zero (parallax unchanged)."""

    class _NaNZpt:
        def get_zpt(self, gmag, nu_eff, pseudocolour, ecl_lat, aps):
            return np.array([np.nan, 0.05])

    table = _make_zpt_table([1.0, 2.0])
    apply_zero_point_correction(table, _NaNZpt())

    zpvals = np.asarray(table["zpvals"])
    assert zpvals[0] == 0.0          # NaN -> 0
    assert zpvals[1] == pytest.approx(0.05)
    parallax = table["parallax"].to_value(u.mas)
    assert parallax[0] == pytest.approx(1.0)          # unchanged
    assert parallax[1] == pytest.approx(2.0 - 0.05)


def test_apply_zero_point_correction_missing_columns_skips():
    """Missing required columns -> correction skipped, not applied, warns."""
    table = QTable()
    table["parallax"] = [1.0, 2.0] * u.mas  # everything else absent

    with pytest.warns(UserWarning, match="zero-point"):
        result = apply_zero_point_correction(table, _FakeZpt(0.02))

    assert result["applied"] is False
    assert "Gmag" in result["missing_columns"]
    # Parallax left untouched; no correction columns created.
    assert table["parallax"].to_value(u.mas) == pytest.approx([1.0, 2.0])
    assert "zpvals" not in table.colnames
