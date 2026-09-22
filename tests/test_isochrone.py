"""Tests for erotica.analysis._isochrone.

Structure
---------
TestCCM89              — extinction law helper
TestChabrier2014       — IMF weights
TestDKMeanQ            — D&K mass-ratio expectation
TestMagCombine         — unresolved-pair magnitude
TestMISTIsochrones     — file parsing, Z extraction, get_isochrone
TestFitErrorModel      — quadratic log-error fit
TestIsochroneFitter    — setup, H_grid ops, save/load, posterior_cmd
TestPyTensorCompat     — tensor dtype / attribute consistency
TestHessFrame          — grid frame, node interpolation, gradient, cache/padding guards
                         (closed-form toy isochrone oracle); two strict xfails pin the open
                         reference-locking defect
"""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import numpy as np
import pytest
from astropy.table import QTable

from erotica.analysis._isochrone import (
    IsochroneFitter,
    MISTIsochrones,
    _ccm89,
    _chabrier2014_weights,
    _dk_mean_q,
    _fit_error_model,
    _mag_combine,
    _smooth2d,
)

_BAYES_EXTRA_MISSING = [
    name for name in ("pymc", "pytensor", "arviz") if importlib.util.find_spec(name) is None
]
requires_bayes_extra = pytest.mark.skipif(
    bool(_BAYES_EXTRA_MISSING),
    reason="requires EROTICA's optional bayes extra: missing " + ", ".join(_BAYES_EXTRA_MISSING),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mist_file(tmp_path: Path, Z: float = 0.0152, n_ages: int = 2) -> Path:
    """Write a minimal valid MIST .iso.cmd file."""
    loga_values = [6.5 + 0.1 * k for k in range(n_ages)]
    masses = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

    # Header block: Zinit parameter line, then value line
    header = textwrap.dedent(f"""\
        # MIST test isochrone
        # Yinit  Zinit  FeH
        #  0.270  {Z:.6f}  0.00
        # EEP initial_mass log10_isochrone_age_yr Gaia_G_EDR3 Gaia_BP_EDR3 Gaia_RP_EDR3
    """)

    rows: list[str] = []
    for loga in loga_values:
        for i, m in enumerate(masses):
            G = 4.0 + 2.0 * i
            BP = 4.5 + 2.0 * i
            RP = 3.8 + 2.0 * i
            rows.append(f"{100 + i} {m:.3f} {loga:.4f} {G:.4f} {BP:.4f} {RP:.4f}")

    fp = tmp_path / "test_feh_p0.00.iso.cmd"
    fp.write_text(header + "\n".join(rows) + "\n")
    return fp


def _make_cluster_data(n: int = 80, rng: np.random.Generator | None = None) -> QTable:
    """Minimal cluster QTable that IsochroneFitter.setup() can consume."""
    if rng is None:
        rng = np.random.default_rng(42)
    mag = rng.uniform(12.0, 18.0, n)
    bp = mag + rng.uniform(0.3, 0.8, n)
    rp = mag - rng.uniform(0.1, 0.4, n)
    return QTable(
        {
            "Gmag": mag,
            "G_BPmag": bp,
            "G_RPmag": rp,
            "e_Gmag": rng.uniform(0.003, 0.05, n),
            "e_G_BPmag": rng.uniform(0.004, 0.06, n),
            "e_G_RPmag": rng.uniform(0.004, 0.06, n),
            "probability_hdbscan": np.ones(n),
        }
    )


# ---------------------------------------------------------------------------
# TestCCM89
# ---------------------------------------------------------------------------


class TestCCM89:
    def test_optical_range(self):
        # V-band (5500 Å): A_V/A_V = 1.0
        assert abs(_ccm89(5500.0) - 1.0) < 0.01

    def test_rv_dependence(self):
        # Higher Rv → lower extinction at optical
        k31 = _ccm89(5500.0, Rv=3.1)
        k51 = _ccm89(5500.0, Rv=5.1)
        assert k31 != k51

    def test_infrared_range(self):
        # IR (22000 Å): A_λ/A_V < 1 for λ > V-band
        assert _ccm89(22000.0) < 1.0

    def test_uv_range(self):
        # UV (2000 Å): higher extinction
        assert _ccm89(2000.0) > 1.0

    def test_gaia_bands_ordering(self):
        # BP (5182 Å) > G (6390 Å) > RP (7825 Å) in A_λ/A_V
        kBP = _ccm89(5182.6)
        kG = _ccm89(6390.7)
        kRP = _ccm89(7825.1)
        assert kBP > kG > kRP


# ---------------------------------------------------------------------------
# TestChabrier2014
# ---------------------------------------------------------------------------


class TestChabrier2014:
    def test_weights_sum_to_one(self):
        mass = np.linspace(0.1, 100.0, 300)
        w = _chabrier2014_weights(mass)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_all_positive(self):
        mass = np.linspace(0.08, 50.0, 200)
        w = _chabrier2014_weights(mass)
        assert np.all(w >= 0)

    def test_low_mass_dominates(self):
        # Chabrier IMF peaks around 0.2 M☉; low-mass stars should carry most weight
        mass = np.linspace(0.1, 100.0, 500)
        w = _chabrier2014_weights(mass)
        low = mass <= 1.0
        high = mass > 1.0
        assert w[low].sum() > w[high].sum()

    def test_minimum_two_stars(self):
        # np.gradient requires ≥2 points; two-star input should work
        mass = np.array([0.5, 1.0])
        w = _chabrier2014_weights(mass)
        assert w.shape == (2,)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_continuity_at_m0(self):
        # Weights should be continuous across m0=1 M☉
        m = np.array([0.99, 1.00, 1.01])
        w = _chabrier2014_weights(m)
        # ratio of adjacent weights should be moderate (< 10× change)
        assert abs(w[1] / w[0] - 1.0) < 2.0
        assert abs(w[2] / w[1] - 1.0) < 2.0


# ---------------------------------------------------------------------------
# TestDKMeanQ
# ---------------------------------------------------------------------------


class TestDKMeanQ:
    def test_output_in_01(self):
        mass = np.array([0.05, 0.3, 1.0, 3.0, 10.0])
        q = _dk_mean_q(mass)
        assert np.all(q >= 0) and np.all(q <= 1)

    def test_mass_ranges(self):
        # γ=4.2 → E[q]=(5.2/6.2)≈0.839 for m≤0.1
        q_vlm = _dk_mean_q(np.array([0.05]))
        assert abs(q_vlm[0] - 5.2 / 6.2) < 1e-9

        # γ=0.4 → E[q]=(1.4/2.4)≈0.583 for 0.1<m≤0.6
        q_lm = _dk_mean_q(np.array([0.3]))
        assert abs(q_lm[0] - 1.4 / 2.4) < 1e-9

        # γ=-0.5 → E[q]=(0.5/1.5)≈0.333 for 1.4<m≤6.5
        q_im = _dk_mean_q(np.array([3.0]))
        assert abs(q_im[0] - 0.5 / 1.5) < 1e-9

    def test_shape_preserved(self):
        mass = np.linspace(0.1, 10.0, 50)
        assert _dk_mean_q(mass).shape == mass.shape


# ---------------------------------------------------------------------------
# TestMagCombine
# ---------------------------------------------------------------------------


class TestMagCombine:
    def test_combined_brighter_than_each(self):
        m1 = np.array([10.0])
        m2 = np.array([10.0])
        comb = _mag_combine(m1, m2)
        assert comb[0] < m1[0] and comb[0] < m2[0]

    def test_equal_mags_offset(self):
        # Two identical stars: combined = m - 2.5*log10(2) ≈ m - 0.7526
        m = 12.0
        expected = m - 2.5 * np.log10(2)
        result = _mag_combine(np.array([m]), np.array([m]))
        assert abs(result[0] - expected) < 1e-10

    def test_shift_property(self):
        # _mag_combine(m1+c, m2+c) == _mag_combine(m1, m2) + c
        m1 = np.array([10.0, 12.0])
        m2 = np.array([11.0, 13.0])
        c = 3.5
        lhs = _mag_combine(m1 + c, m2 + c)
        rhs = _mag_combine(m1, m2) + c
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)

    def test_dominance_by_bright(self):
        # If m2 >> m1 (much fainter), combined ≈ m1
        m1 = np.array([10.0])
        m2 = np.array([25.0])
        comb = _mag_combine(m1, m2)
        assert abs(comb[0] - m1[0]) < 0.01


# ---------------------------------------------------------------------------
# TestMISTIsochrones
# ---------------------------------------------------------------------------


class TestMISTIsochrones:
    def test_load_file(self, tmp_path):
        _make_mist_file(tmp_path, Z=0.0152, n_ages=2)  # escribe el fichero; el path no se usa
        iso = MISTIsochrones(tmp_path)
        assert iso._met_values is not None
        assert len(iso._met_values) == 1  # one Z value
        assert abs(iso._met_values[0] - 0.0152) < 1e-5

    def test_loga_values(self, tmp_path):
        _make_mist_file(tmp_path, Z=0.0152, n_ages=3)  # ídem
        iso = MISTIsochrones(tmp_path)
        assert len(iso._loga_values) == 3

    def test_get_isochrone_returns_tuple(self, tmp_path):
        _make_mist_file(tmp_path, Z=0.0152, n_ages=2)
        iso = MISTIsochrones(tmp_path)
        mass, G, BP, RP = iso.get_isochrone(0.0152, 6.5)
        assert mass.ndim == 1 and len(mass) > 0
        assert G.shape == mass.shape

    def test_get_isochrone_nearest_neighbor(self, tmp_path):
        _make_mist_file(tmp_path, Z=0.0152, n_ages=2)
        iso = MISTIsochrones(tmp_path)
        # Query at a Z that doesn't exist → should return nearest
        mass1, G1, _, _ = iso.get_isochrone(0.0152, 6.5)
        mass2, G2, _, _ = iso.get_isochrone(0.0200, 6.5)  # nearest is 0.0152
        np.testing.assert_array_equal(mass1, mass2)

    def test_z_from_name_fallback(self, tmp_path):
        """File with no Zinit in header → fallback to filename."""
        fp = tmp_path / "feh_p0.00.iso.cmd"
        fp.write_text(
            "# EEP initial_mass log10_isochrone_age_yr Gaia_G_EDR3 Gaia_BP_EDR3 Gaia_RP_EDR3\n"
            "100 0.5 6.5 5.0 5.5 4.8\n"
            "101 1.0 6.5 4.0 4.5 3.8\n"
        )
        iso = MISTIsochrones(tmp_path)
        # Z_sun = 0.0152 * 10^0.00 = 0.0152
        assert abs(iso._met_values[0] - 0.0152) < 1e-4

    def test_no_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MISTIsochrones(tmp_path)

    def test_mass_sorted_ascending(self, tmp_path):
        _make_mist_file(tmp_path, Z=0.0152, n_ages=1)
        iso = MISTIsochrones(tmp_path)
        mass, *_ = iso.get_isochrone(0.0152, 6.5)
        assert np.all(np.diff(mass) > 0)


# ---------------------------------------------------------------------------
# TestFitErrorModel
# ---------------------------------------------------------------------------


class TestFitErrorModel:
    def test_returns_callables(self):
        mag = np.linspace(10, 18, 50)
        e_m = 0.002 * 10 ** (0.3 * (mag - 10))
        e_c = e_m * 1.4
        f_m, f_c = _fit_error_model(mag, e_m, e_c)
        assert callable(f_m) and callable(f_c)

    def test_monotone_increase(self):
        # Fainter stars have larger errors
        mag = np.linspace(12, 20, 100)
        e_m = 0.001 * 10 ** (0.4 * (mag - 12))
        e_c = e_m * 1.3
        f_m, _ = _fit_error_model(mag, e_m, e_c)
        vals = f_m(np.array([13.0, 15.0, 17.0]))
        assert vals[0] < vals[1] < vals[2]

    def test_fallback_on_few_points(self):
        """Fewer than 3 points must give the *median* error, not a fitted curve.

        Oracle: the exact fallback value, ``median([0.01, 0.02]) = 0.015``,
        constant in magnitude. Asserting only ``isfinite`` passed for the whole
        lifetime of the ``mag_ok.sum() < 3`` bug -- the guard summed magnitudes
        (14 + 15 = 29) instead of counting points, so this input silently fitted
        a rank-deficient quadratic through 2 points and returned garbage that
        happened to be finite.
        """
        mag = np.array([14.0, 15.0])
        e_m = np.array([0.01, 0.02])
        e_c = np.array([0.015, 0.025])
        f_m, f_c = _fit_error_model(mag, e_m, e_c)

        probe = np.array([12.0, 16.0, 25.0])
        assert f_m(probe) == pytest.approx(0.015)  # median of e_m
        assert f_c(probe) == pytest.approx(0.020)  # median of e_c
        assert len(np.unique(f_m(probe))) == 1  # genuinely constant

    def test_fallback_on_repeated_magnitudes(self):
        """Three points at one magnitude cannot constrain a quadratic either.

        Rank deficiency is about *distinct* abscissae, not the point count.
        """
        mag = np.full(6, 15.0)
        e_m = np.array([0.01, 0.02, 0.03, 0.01, 0.02, 0.03])
        f_m, _ = _fit_error_model(mag, e_m, e_m * 1.5)
        assert f_m(np.array([13.0, 18.0])) == pytest.approx(np.median(e_m))

    def test_recovers_a_known_quadratic(self):
        """Oracle: coefficients injected by the test, in log10 space."""
        a, b, c = -3.0, 0.15, 0.01
        mag = np.linspace(11.0, 20.0, 60)
        e_m = 10.0 ** (a + b * (mag - 15) + c * (mag - 15) ** 2)
        f_m, _ = _fit_error_model(mag, e_m, e_m * 1.3)
        probe = np.array([12.0, 15.0, 19.0])
        expected = 10.0 ** (a + b * (probe - 15) + c * (probe - 15) ** 2)
        np.testing.assert_allclose(f_m(probe), expected, rtol=1e-6)

    def test_ignores_invalid_errors_when_counting_points(self):
        """Zero and NaN errors are dropped, so what is left decides the branch."""
        mag = np.array([12.0, 13.0, 14.0, 15.0, 16.0])
        e_m = np.array([0.01, 0.0, np.nan, -0.5, 0.02])  # only 2 usable
        f_m, _ = _fit_error_model(mag, e_m, np.full(5, 0.03))
        assert f_m(np.array([14.0])) == pytest.approx(np.nanmedian(e_m))


# ---------------------------------------------------------------------------
# TestIsochroneFitter
# ---------------------------------------------------------------------------


@requires_bayes_extra
class TestIsochroneFitter:
    @pytest.fixture
    def fitter_and_data(self, tmp_path):
        _make_mist_file(tmp_path, Z=0.0152, n_ages=3)
        data = _make_cluster_data(n=60)
        fitter = IsochroneFitter(
            isochs_path=tmp_path,
            loga_range=(6.5, 6.7),
            Av_range=(0.0, 1.0),
            dm_mu=10.0,
            dm_sigma=0.3,
            dm_range=(9.0, 11.0),
            M_met=5,
            M_loga=5,
        )
        fitter.setup(data, prob_threshold=0.5)
        return fitter, data

    def test_setup_sets_obs_arrays(self, fitter_and_data):
        fitter, _ = fitter_and_data
        assert fitter._obs_mag is not None
        assert fitter._obs_col is not None
        assert len(fitter._obs_mag) > 0

    def test_setup_sets_ext_coefs(self, fitter_and_data):
        fitter, _ = fitter_and_data
        assert fitter._kG is not None
        assert fitter._kBP is not None
        assert fitter._kRP is not None
        # BP band has more extinction than G and RP in optical
        assert fitter._kBP > fitter._kG > fitter._kRP

    def test_setup_H_grid_shape(self, fitter_and_data):
        fitter, _ = fitter_and_data
        assert fitter._H_grid is not None
        Nm, Na = 5, 5
        Nb_m, Nb_c = fitter._Nbins
        pm_, pc_ = fitter._pad  # the grid is padded so a shifted model can enter the window
        assert fitter._H_grid.shape == (Nm, Na, Nb_m + 2 * pm_, Nb_c + 2 * pc_)

    def test_interp_H_returns_tensor(self, fitter_and_data):
        import pytensor.tensor as pt

        fitter, _ = fitter_and_data
        met_t = pt.as_tensor_variable(np.float64(0.0152))
        loga_t = pt.as_tensor_variable(np.float64(6.5))
        H = fitter._interp_H(met_t, loga_t)
        result = H.eval()
        Nb_m, Nb_c = fitter._Nbins
        pm_, pc_ = fitter._pad
        assert result.shape == (Nb_m + 2 * pm_, Nb_c + 2 * pc_)
        assert np.all(np.isfinite(result))

    def test_shift_histogram_identity(self, fitter_and_data):
        """At the reference shift the output is the central window of the padded grid.

        The grid is binned in the apparent frame of ``(dmag_ref, dcol_ref)`` on the
        observed window widened by ``pad`` bins; zero offset from that reference must
        return exactly the un-padded centre.
        """
        import pytensor.tensor as pt

        fitter, _ = fitter_and_data
        Nb_m, Nb_c = fitter._Nbins
        pm_, pc_ = fitter._pad
        H_np = np.random.default_rng(0).uniform(0, 1, (Nb_m + 2 * pm_, Nb_c + 2 * pc_))
        shifted = fitter._shift_histogram(
            pt.as_tensor_variable(H_np),
            pt.as_tensor_variable(np.float64(fitter._dmag_ref)),
            pt.as_tensor_variable(np.float64(fitter._dcol_ref)),
        ).eval()
        np.testing.assert_allclose(shifted, H_np[pm_ : pm_ + Nb_m, pc_ : pc_ + Nb_c], atol=1e-12)

    def test_shift_histogram_whole_bin_is_a_row_offset(self, fitter_and_data):
        """A shift of exactly one magnitude bin fainter reads the padded grid one row up."""
        import pytensor.tensor as pt

        fitter, _ = fitter_and_data
        Nb_m, Nb_c = fitter._Nbins
        pm_, pc_ = fitter._pad
        H_np = np.random.default_rng(1).uniform(0, 1, (Nb_m + 2 * pm_, Nb_c + 2 * pc_))
        shifted = fitter._shift_histogram(
            pt.as_tensor_variable(H_np),
            pt.as_tensor_variable(np.float64(fitter._dmag_ref + fitter._binw_mag)),
            pt.as_tensor_variable(np.float64(fitter._dcol_ref)),
        ).eval()
        np.testing.assert_allclose(
            shifted, H_np[pm_ - 1 : pm_ - 1 + Nb_m, pc_ : pc_ + Nb_c], atol=1e-9
        )

    def test_save_load_roundtrip(self, fitter_and_data, tmp_path):
        fitter, data = fitter_and_data
        cache = tmp_path / "hgrid.npz"
        fitter.save_grid(cache)

        fitter2 = IsochroneFitter(
            isochs_path=fitter.isochs_path,
            loga_range=fitter.loga_range,
            Av_range=fitter.Av_range,
            dm_mu=fitter.dm_mu,
            dm_sigma=fitter.dm_sigma,
            dm_range=fitter.dm_range,
            M_met=fitter.M_met,
            M_loga=fitter.M_loga,
        )
        fitter2.setup(data, prob_threshold=0.5, precompute_grid=False)
        fitter2.load_grid(cache)

        np.testing.assert_array_equal(fitter._H_grid, fitter2._H_grid)
        assert fitter2._kG == pytest.approx(fitter._kG)
        assert fitter2._kBP == pytest.approx(fitter._kBP)
        assert fitter2._kRP == pytest.approx(fitter._kRP)

    def test_posterior_cmd_shape(self, fitter_and_data):
        """posterior_cmd should return obs arrays and a list of (mag, col) tuples."""
        import arviz as az

        fitter, _ = fitter_and_data
        # Build a tiny fake idata with 2 draws × 1 chain
        n = 2
        rng = np.random.default_rng(1)
        idata = az.from_dict(
            {
                "posterior": {
                    "met": rng.uniform(0.014, 0.016, (1, n)),
                    "loga": rng.uniform(6.5, 6.7, (1, n)),
                    "dm": rng.uniform(9.8, 10.2, (1, n)),
                    "Av": rng.uniform(0.0, 0.5, (1, n)),
                },
            }
        )
        obs_mag, obs_col, cmds = fitter.posterior_cmd(idata, num_samples=n)
        assert obs_mag.shape == fitter._obs_mag.shape
        assert len(cmds) == n
        for mag_s, col_s in cmds:
            assert mag_s.ndim == 1
            assert col_s.shape == mag_s.shape


# ---------------------------------------------------------------------------
# TestPyTensorCompat
# ---------------------------------------------------------------------------


@requires_bayes_extra
class TestPyTensorCompat:
    """Verify PyTensor tensors are correctly initialized in all code paths."""

    @pytest.fixture
    def base_fitter(self, tmp_path):
        _make_mist_file(tmp_path, Z=0.0152, n_ages=2)
        return IsochroneFitter(
            isochs_path=tmp_path,
            loga_range=(6.5, 6.6),
            Av_range=(0.0, 1.0),
            dm_mu=10.0,
            dm_sigma=0.3,
            dm_range=(9.0, 11.0),
            M_met=4,
            M_loga=4,
        )

    def test_tensors_set_after_setup(self, base_fitter, tmp_path):
        data = _make_cluster_data(n=40)
        base_fitter.setup(data, prob_threshold=0.0)
        assert base_fitter._I_tensor is not None
        assert base_fitter._J_tensor is not None
        assert base_fitter._H_tensor is not None
        assert base_fitter._obs_hess is not None

    def test_tensors_set_after_load_grid(self, base_fitter, tmp_path):
        data = _make_cluster_data(n=40)
        base_fitter.setup(data, prob_threshold=0.0)
        cache = tmp_path / "hgrid.npz"
        base_fitter.save_grid(cache)

        fitter2 = IsochroneFitter(
            isochs_path=base_fitter.isochs_path,
            loga_range=base_fitter.loga_range,
            Av_range=base_fitter.Av_range,
            dm_mu=base_fitter.dm_mu,
            dm_sigma=base_fitter.dm_sigma,
            dm_range=base_fitter.dm_range,
            M_met=base_fitter.M_met,
            M_loga=base_fitter.M_loga,
        )
        fitter2.setup(data, prob_threshold=0.0, precompute_grid=False)
        fitter2.load_grid(cache)

        # All three tensors must be non-None so build_model() doesn't crash
        assert fitter2._I_tensor is not None, "_I_tensor missing after load_grid"
        assert fitter2._J_tensor is not None, "_J_tensor missing after load_grid"
        assert fitter2._H_tensor is not None, "_H_tensor missing after load_grid"

    def test_I_J_tensor_shapes(self, base_fitter, tmp_path):
        data = _make_cluster_data(n=40)
        base_fitter.setup(data, prob_threshold=0.0)
        Nb_m, Nb_c = base_fitter._Nbins
        i_tensor = base_fitter._I_tensor.eval()
        j_tensor = base_fitter._J_tensor.eval()
        assert i_tensor.shape == (Nb_m, 1)
        assert j_tensor.shape == (1, Nb_c)

    def test_I_J_tensor_values(self, base_fitter, tmp_path):
        data = _make_cluster_data(n=40)
        base_fitter.setup(data, prob_threshold=0.0)
        Nb_m, Nb_c = base_fitter._Nbins
        np.testing.assert_array_equal(
            base_fitter._I_tensor.eval()[:, 0],
            np.arange(Nb_m, dtype="float64"),
        )
        np.testing.assert_array_equal(
            base_fitter._J_tensor.eval()[0, :],
            np.arange(Nb_c, dtype="float64"),
        )

    def test_shift_histogram_int64_dtypes(self, base_fitter, tmp_path):
        """Floor-based indices must be int64 to avoid JAX int32 warnings."""
        import pytensor.tensor as pt

        data = _make_cluster_data(n=40)
        base_fitter.setup(data, prob_threshold=0.0)
        Nb_m, Nb_c = base_fitter._Nbins
        H_t = pt.as_tensor_variable(np.zeros((Nb_m, Nb_c)))
        dmag = pt.as_tensor_variable(np.float64(0.1))
        dcol = pt.as_tensor_variable(np.float64(0.05))
        # Building the expression should not raise a dtype error
        result = base_fitter._shift_histogram(H_t, dmag, dcol)
        out = result.eval()
        assert out.dtype == np.float64

    def test_H_tensor_matches_H_grid(self, base_fitter, tmp_path):
        data = _make_cluster_data(n=40)
        base_fitter.setup(data, prob_threshold=0.0)
        # pytensor.shared exposes .get_value() as well as .eval()
        np.testing.assert_array_equal(
            base_fitter._H_tensor.get_value(),
            base_fitter._H_grid,
        )

    def test_obs_hess_shape(self, base_fitter, tmp_path):
        data = _make_cluster_data(n=40)
        base_fitter.setup(data, prob_threshold=0.0)
        Nb_m, Nb_c = base_fitter._Nbins
        assert base_fitter._obs_hess.shape == (Nb_m * Nb_c,)
        assert base_fitter._obs_hess.dtype == np.float64

    def test_build_model_after_load_grid_does_not_crash(self, base_fitter, tmp_path):
        """Regression: _I/_J/_H_tensor were None after load_grid → crash in build_model."""
        data = _make_cluster_data(n=40)
        base_fitter.setup(data, prob_threshold=0.0)
        cache = tmp_path / "hgrid_bm.npz"
        base_fitter.save_grid(cache)

        fitter2 = IsochroneFitter(
            isochs_path=base_fitter.isochs_path,
            loga_range=base_fitter.loga_range,
            Av_range=base_fitter.Av_range,
            dm_mu=base_fitter.dm_mu,
            dm_sigma=base_fitter.dm_sigma,
            dm_range=base_fitter.dm_range,
            M_met=base_fitter.M_met,
            M_loga=base_fitter.M_loga,
        )
        fitter2.setup(data, prob_threshold=0.0, precompute_grid=False)
        fitter2.load_grid(cache)
        model = fitter2.build_model()
        assert model is not None
        # Should expose the four astrophysical parameters
        free_names = {v.name for v in model.free_RVs}
        for param in ("met", "loga", "dm", "Av"):
            assert any(param in n for n in free_names), f"{param} missing from model"


# ---------------------------------------------------------------------------
# TestCCM89Extended  —  wavelength regime boundaries and known values
# ---------------------------------------------------------------------------


class TestCCM89Extended:
    def test_returns_float(self):
        result = _ccm89(5500.0)
        assert isinstance(result, float)

    def test_optical_uv_boundary_continuity(self):
        # x=3.3 (λ≈3030 Å) — boundary between optical polynomial and UV power-law
        lam_near = 1e4 / 3.3
        below = _ccm89(lam_near + 0.5)  # x < 3.3 (optical)
        above = _ccm89(lam_near - 0.5)  # x > 3.3 (UV)
        # Both should be finite and positive
        assert np.isfinite(below) and below > 0
        assert np.isfinite(above) and above > 0
        # And not wildly discontinuous (within 20%)
        assert abs(below / above - 1.0) < 0.20

    def test_uv_fa_fb_boundary(self):
        # x=5.9 activates Fa/Fb corrections in UV regime
        lam_59 = 1e4 / 5.9
        below = _ccm89(lam_59 + 1.0)  # x < 5.9, no Fa/Fb
        above = _ccm89(lam_59 - 1.0)  # x > 5.9, Fa/Fb active
        assert np.isfinite(below) and np.isfinite(above)

    def test_rv_monotone_effect_at_blue(self):
        # At B-band (4400 Å), b > 0 → k = a + b/Rv decreases as Rv increases
        vals = [_ccm89(4400.0, Rv=r) for r in [2.5, 3.1, 4.0, 5.1]]
        assert vals[0] > vals[1] > vals[2] > vals[3]

    def test_all_gaia_coeffs_positive(self):
        for lam in [5182.6, 6390.7, 7825.1]:
            assert _ccm89(lam) > 0

    def test_far_ir_less_than_optical(self):
        k_ir = _ccm89(20000.0)
        k_opt = _ccm89(5500.0)
        assert k_ir < k_opt


# ---------------------------------------------------------------------------
# TestChabrier2014Extended
# ---------------------------------------------------------------------------


class TestChabrier2014Extended:
    def test_exact_m0_boundary(self):
        # At m=1.0 (transition), both branches give the same value by design
        m = np.array([0.9999, 1.0, 1.0001])
        w = _chabrier2014_weights(m)
        assert np.all(w >= 0) and abs(w.sum() - 1.0) < 1e-10

    def test_power_law_regime_weights_decrease(self):
        # On a uniform grid, |Δm| is constant so w ∝ ξ(m) = m^{-2.35}: decreasing
        mass = np.linspace(2.0, 50.0, 200)
        w = _chabrier2014_weights(mass)
        # Monotonically decreasing in the Salpeter regime (sample a few points)
        sample = w[[0, 50, 100, 150, 199]]
        assert np.all(np.diff(sample) < 0)

    def test_output_dtype_float64(self):
        mass = np.linspace(0.1, 10.0, 100)
        w = _chabrier2014_weights(mass)
        assert w.dtype == np.float64

    def test_no_negative_weights(self):
        # Even with extreme mass ranges
        mass = np.logspace(-1, 2, 200)
        w = _chabrier2014_weights(mass)
        assert np.all(w >= 0)


# ---------------------------------------------------------------------------
# TestDKMeanQExtended  —  all five mass regimes
# ---------------------------------------------------------------------------


class TestDKMeanQExtended:
    @pytest.mark.parametrize(
        "mass, gamma",
        [
            (0.05, 4.2),  # m ≤ 0.1
            (0.3, 0.4),  # 0.1 < m ≤ 0.6
            (1.0, 0.3),  # 0.6 < m ≤ 1.4
            (3.0, -0.5),  # 1.4 < m ≤ 6.5
            (10.0, 0.0),  # m > 6.5
        ],
    )
    def test_all_gamma_regimes(self, mass, gamma):
        expected = (gamma + 1.0) / (gamma + 2.0)
        result = _dk_mean_q(np.array([mass]))
        assert abs(result[0] - expected) < 1e-9

    def test_boundary_at_0_1(self):
        # Exactly at the boundary: np.where uses ≤ so 0.1 goes to γ=4.2
        q = _dk_mean_q(np.array([0.1]))
        assert abs(q[0] - 5.2 / 6.2) < 1e-9

    def test_boundary_at_6_5(self):
        # Exactly at 6.5: goes to γ=-0.5
        q = _dk_mean_q(np.array([6.5]))
        assert abs(q[0] - 0.5 / 1.5) < 1e-9

    def test_decreasing_trend_from_vlm_to_im(self):
        # E[q] decreases from VLM (0.839) through IM (0.333) — not monotone at >6.5
        q = _dk_mean_q(np.array([0.05, 0.3, 1.0, 3.0]))
        assert q[0] > q[1] > q[2] > q[3]

    def test_high_mass_q_higher_than_im(self):
        # >6.5 M☉: γ=0.0 → E[q]=0.5, higher than IM (0.333)
        q_im = _dk_mean_q(np.array([3.0]))[0]
        q_hm = _dk_mean_q(np.array([10.0]))[0]
        assert q_hm > q_im


# ---------------------------------------------------------------------------
# TestMagCombineExtended
# ---------------------------------------------------------------------------


class TestMagCombineExtended:
    def test_commutative(self):
        m1 = np.array([10.0, 11.0, 12.0])
        m2 = np.array([11.0, 10.0, 13.0])
        np.testing.assert_allclose(_mag_combine(m1, m2), _mag_combine(m2, m1), atol=1e-12)

    def test_vectorized(self):
        m1 = np.linspace(8, 16, 50)
        m2 = m1 + 1.0
        result = _mag_combine(m1, m2)
        assert result.shape == (50,)
        assert np.all(result < m1)  # combined is brighter than primary

    def test_large_difference_limit(self):
        # When secondary is >> 15 mag fainter, combined ≈ primary (< 0.001 mag error)
        m1 = np.array([12.0])
        m2 = np.array([30.0])
        assert abs(_mag_combine(m1, m2)[0] - m1[0]) < 1e-3


# ---------------------------------------------------------------------------
# TestSmooth2D
# ---------------------------------------------------------------------------


class TestSmooth2D:
    def test_preserves_shape(self):
        H = np.ones((10, 8))
        assert _smooth2d(H).shape == (10, 8)

    def test_non_negative_input_non_negative_output(self):
        H = np.random.default_rng(0).uniform(0, 5, (12, 10))
        assert np.all(_smooth2d(H) >= 0)

    def test_all_zeros_stays_zero(self):
        H = np.zeros((8, 6))
        np.testing.assert_array_equal(_smooth2d(H), np.zeros((8, 6)))

    def test_smoothing_reduces_peak(self):
        H = np.zeros((15, 15))
        H[7, 7] = 100.0
        smoothed = _smooth2d(H)
        # Peak value should decrease after smoothing
        assert smoothed.max() < 100.0
        # Total mass should be approximately conserved (boundary effects aside)
        assert smoothed.sum() <= H.sum() + 1e-6


# ---------------------------------------------------------------------------
# TestFitErrorModelExtended
# ---------------------------------------------------------------------------


class TestFitErrorModelExtended:
    def test_output_always_positive(self):
        mag = np.linspace(12, 20, 100)
        e_m = 0.001 * 10 ** (0.4 * (mag - 12))
        e_c = e_m * 1.3
        f_m, f_c = _fit_error_model(mag, e_m, e_c)
        query = np.linspace(10, 22, 30)
        assert np.all(f_m(query) > 0)
        assert np.all(f_c(query) > 0)

    def test_nan_in_errors_filtered(self):
        mag = np.linspace(12, 20, 50)
        e_m = 0.01 * np.ones(50)
        e_m[::3] = np.nan  # every 3rd point is NaN
        e_c = e_m.copy()
        f_m, f_c = _fit_error_model(mag, e_m, e_c)
        assert np.isfinite(f_m(np.array([15.0]))[0])

    def test_zero_errors_filtered(self):
        mag = np.linspace(12, 20, 50)
        e_m = np.where(np.arange(50) % 2 == 0, 0.01, 0.0)
        e_c = e_m.copy()
        # Should not raise, zero errors are excluded by `e_mag > 0`
        f_m, f_c = _fit_error_model(mag, e_m, e_c)
        assert callable(f_m)

    def test_exactly_three_valid_points(self):
        mag = np.array([13.0, 15.0, 17.0])
        e_m = np.array([0.005, 0.01, 0.02])
        e_c = np.array([0.007, 0.015, 0.03])
        f_m, f_c = _fit_error_model(mag, e_m, e_c)
        # Exactly 3 points: quadratic fit is fully determined
        assert np.isfinite(f_m(np.array([16.0]))[0])


# ---------------------------------------------------------------------------
# TestIsochroneFitterExtended
# ---------------------------------------------------------------------------


@requires_bayes_extra
class TestIsochroneFitterExtended:
    @pytest.fixture
    def setup_fitter(self, tmp_path):
        _make_mist_file(tmp_path, Z=0.0152, n_ages=2)
        fitter = IsochroneFitter(
            isochs_path=tmp_path,
            loga_range=(6.5, 6.6),
            Av_range=(0.0, 1.0),
            dm_mu=10.0,
            dm_sigma=0.3,
            dm_range=(9.0, 11.0),
            M_met=4,
            M_loga=4,
        )
        return fitter

    def test_prob_threshold_filters_members(self, setup_fitter):
        rng = np.random.default_rng(0)
        n = 100
        mag = rng.uniform(12.0, 18.0, n)
        bp = mag + 0.5
        rp = mag - 0.2
        probs = np.concatenate([np.ones(50), np.zeros(50)])
        data = QTable(
            {
                "Gmag": mag,
                "G_BPmag": bp,
                "G_RPmag": rp,
                "e_Gmag": np.full(n, 0.01),
                "e_G_BPmag": np.full(n, 0.012),
                "e_G_RPmag": np.full(n, 0.012),
                "probability_hdbscan": probs,
            }
        )
        setup_fitter.setup(data, prob_threshold=0.5)
        assert setup_fitter._N_obs == 50
        assert len(setup_fitter._obs_mag) == 50

    def test_e_BP_RP_column_used_when_present(self, setup_fitter):
        """e_BP_RP column should be used directly instead of hypot(e_BP, e_RP)."""
        n = 40
        rng = np.random.default_rng(7)
        mag = rng.uniform(12.0, 18.0, n)
        data_separate = QTable(
            {
                "Gmag": mag,
                "G_BPmag": mag + 0.5,
                "G_RPmag": mag - 0.2,
                "e_Gmag": np.full(n, 0.01),
                "e_G_BPmag": np.full(n, 0.02),
                "e_G_RPmag": np.full(n, 0.02),
                "probability_hdbscan": np.ones(n),
            }
        )
        data_combined = QTable(
            {
                "Gmag": mag,
                "G_BPmag": mag + 0.5,
                "G_RPmag": mag - 0.2,
                "e_Gmag": np.full(n, 0.01),
                "e_BP_RP": np.full(n, 0.05),  # explicit combined column
                "probability_hdbscan": np.ones(n),
            }
        )
        f1 = IsochroneFitter(
            isochs_path=setup_fitter.isochs_path,
            loga_range=setup_fitter.loga_range,
            Av_range=setup_fitter.Av_range,
            dm_mu=setup_fitter.dm_mu,
            dm_sigma=setup_fitter.dm_sigma,
            dm_range=setup_fitter.dm_range,
            M_met=4,
            M_loga=4,
        )
        f2 = IsochroneFitter(
            isochs_path=setup_fitter.isochs_path,
            loga_range=setup_fitter.loga_range,
            Av_range=setup_fitter.Av_range,
            dm_mu=setup_fitter.dm_mu,
            dm_sigma=setup_fitter.dm_sigma,
            dm_range=setup_fitter.dm_range,
            M_met=4,
            M_loga=4,
        )
        f1.setup(data_separate, prob_threshold=0.0)
        f2.setup(data_combined, prob_threshold=0.0)
        # Both should set up without errors and have the same obs magnitudes
        np.testing.assert_array_equal(f1._obs_mag, f2._obs_mag)

    def test_k_col1_equals_kBP_minus_kRP(self, setup_fitter):
        data = _make_cluster_data(n=40)
        setup_fitter.setup(data, prob_threshold=0.0)
        assert setup_fitter._k_col1 == pytest.approx(
            setup_fitter._kBP - setup_fitter._kRP, rel=1e-10
        )

    def test_precompute_false_leaves_H_grid_none(self, setup_fitter):
        data = _make_cluster_data(n=40)
        setup_fitter.setup(data, prob_threshold=0.0, precompute_grid=False)
        assert setup_fitter._H_grid is None
        assert setup_fitter._H_tensor is None

    def test_hess_for_isochrone_shape_and_sign(self, setup_fitter):
        data = _make_cluster_data(n=40)
        setup_fitter.setup(data, prob_threshold=0.0)
        iso = setup_fitter._isochs
        mass, G, BP, RP = iso.get_isochrone(0.0152, 6.5)
        H = setup_fitter._hess_for_isochrone(mass, G, BP, RP)
        Nb_m, Nb_c = setup_fitter._Nbins
        pm_, pc_ = setup_fitter._pad
        assert H.shape == (Nb_m + 2 * pm_, Nb_c + 2 * pc_)
        assert np.all(H >= 0)
        assert np.all(np.isfinite(H))

    def test_H_grid_non_negative_finite(self, setup_fitter):
        data = _make_cluster_data(n=40)
        setup_fitter.setup(data, prob_threshold=0.0)
        assert np.all(setup_fitter._H_grid >= 0)
        assert np.all(np.isfinite(setup_fitter._H_grid))

    def test_save_grid_raises_when_no_H_grid(self, setup_fitter, tmp_path):
        with pytest.raises(RuntimeError, match="No H_grid"):
            setup_fitter.save_grid(tmp_path / "never.npz")

    def test_build_model_raises_before_setup(self, setup_fitter):
        with pytest.raises(RuntimeError):
            setup_fitter.build_model()

    def test_build_model_has_correct_free_vars(self, setup_fitter):
        data = _make_cluster_data(n=40)
        setup_fitter.setup(data, prob_threshold=0.0)
        model = setup_fitter.build_model()
        free_names = {v.name for v in model.free_RVs}
        for param in ("met", "loga", "dm", "Av", "log_s", "bg"):
            assert any(param in n for n in free_names), f"'{param}' not in free_RVs"

    def test_posterior_cmd_mag_within_range(self, setup_fitter):
        """All returned magnitudes must lie within the histogram range."""
        import arviz as az

        data = _make_cluster_data(n=40)
        setup_fitter.setup(data, prob_threshold=0.0)
        rng = np.random.default_rng(3)
        idata = az.from_dict(
            {
                "posterior": {
                    "met": rng.uniform(0.014, 0.016, (1, 3)),
                    "loga": rng.uniform(6.5, 6.6, (1, 3)),
                    "dm": rng.uniform(9.5, 10.5, (1, 3)),
                    "Av": rng.uniform(0.0, 0.5, (1, 3)),
                },
            }
        )
        obs_mag, obs_col, cmds = setup_fitter.posterior_cmd(idata, num_samples=3)
        mag_lo, mag_hi = setup_fitter._mag_range
        col_lo, col_hi = setup_fitter._col_range
        for mag_s, col_s in cmds:
            assert np.all(mag_s >= mag_lo) and np.all(mag_s <= mag_hi)
            assert np.all(col_s >= col_lo) and np.all(col_s <= col_hi)

    def test_posterior_cmd_obs_unchanged(self, setup_fitter):
        """posterior_cmd should not mutate _obs_mag / _obs_col."""
        import arviz as az

        data = _make_cluster_data(n=40)
        setup_fitter.setup(data, prob_threshold=0.0)
        obs_before = setup_fitter._obs_mag.copy()
        rng = np.random.default_rng(4)
        idata = az.from_dict(
            {
                "posterior": {
                    "met": rng.uniform(0.014, 0.016, (1, 2)),
                    "loga": rng.uniform(6.5, 6.6, (1, 2)),
                    "dm": rng.uniform(9.8, 10.2, (1, 2)),
                    "Av": rng.uniform(0.0, 0.5, (1, 2)),
                },
            }
        )
        setup_fitter.posterior_cmd(idata, num_samples=2)
        np.testing.assert_array_equal(setup_fitter._obs_mag, obs_before)


# ---------------------------------------------------------------------------
# TestMISTIsochronesExtended
# ---------------------------------------------------------------------------


class TestMISTIsochronesExtended:
    def test_met_age_dict_keys(self, tmp_path):
        _make_mist_file(tmp_path, Z=0.0152, n_ages=2)
        iso = MISTIsochrones(tmp_path)
        d = iso.met_age_dict
        assert "met" in d and "loga" in d
        assert isinstance(d["met"], np.ndarray)
        assert isinstance(d["loga"], np.ndarray)

    def test_multiple_z_files(self, tmp_path):
        """Two files with different Z values → two entries in _met_values."""
        (tmp_path / "a.iso.cmd").write_text(
            "# Yinit  Zinit  FeH\n"
            "#  0.270  0.0100  -0.18\n"
            "# EEP initial_mass log10_isochrone_age_yr Gaia_G_EDR3 Gaia_BP_EDR3 Gaia_RP_EDR3\n"
            "100 0.3 6.5 4.0 4.5 3.8\n"
            "101 0.5 6.5 5.0 5.5 4.8\n"
            "102 1.0 6.5 6.0 6.5 5.8\n"
        )
        (tmp_path / "b.iso.cmd").write_text(
            "# Yinit  Zinit  FeH\n"
            "#  0.270  0.0200  0.12\n"
            "# EEP initial_mass log10_isochrone_age_yr Gaia_G_EDR3 Gaia_BP_EDR3 Gaia_RP_EDR3\n"
            "100 0.3 6.5 4.1 4.6 3.9\n"
            "101 0.5 6.5 5.1 5.6 4.9\n"
            "102 1.0 6.5 6.1 6.6 5.9\n"
        )
        iso = MISTIsochrones(tmp_path)
        assert len(iso._met_values) == 2
        np.testing.assert_allclose(sorted(iso._met_values), [0.010, 0.020], rtol=1e-3)

    def test_negative_feh_filename_fallback(self, tmp_path):
        """[Fe/H]=-0.30 → Z = 0.0152 * 10^(-0.3) ≈ 0.00762."""
        fp = tmp_path / "feh_m0.30.iso.cmd"
        fp.write_text(
            "# EEP initial_mass log10_isochrone_age_yr Gaia_G_EDR3 Gaia_BP_EDR3 Gaia_RP_EDR3\n"
            "100 0.5 6.5 5.0 5.5 4.8\n"
            "101 1.0 6.5 4.0 4.5 3.8\n"
        )
        iso = MISTIsochrones(tmp_path)
        expected_Z = 0.0152 * 10 ** (-0.30)
        assert abs(iso._met_values[0] - expected_Z) < 1e-4

    def test_non_finite_rows_excluded(self, tmp_path):
        """Rows with NaN/Inf magnitudes should not enter the grid."""
        fp = tmp_path / "test.iso.cmd"
        fp.write_text(
            "# Yinit  Zinit  FeH\n"
            "#  0.270  0.0152  0.00\n"
            "# EEP initial_mass log10_isochrone_age_yr Gaia_G_EDR3 Gaia_BP_EDR3 Gaia_RP_EDR3\n"
            "100 0.3 6.5 nan 4.5 3.8\n"  # NaN G → excluded
            "101 0.5 6.5 5.0 5.5 4.8\n"  # good
            "102 1.0 6.5 4.0 4.5 3.8\n"  # good
        )
        iso = MISTIsochrones(tmp_path)
        mass, G, BP, RP = iso.get_isochrone(0.0152, 6.5)
        assert len(mass) == 2  # NaN row excluded
        assert np.all(np.isfinite(G))

    def test_missing_required_column_warns_skips(self, tmp_path):
        """File missing Gaia_G_EDR3 column should produce a warning and be skipped."""
        fp = tmp_path / "bad.iso.cmd"
        fp.write_text(
            "# EEP initial_mass log10_isochrone_age_yr NOT_G NOT_BP NOT_RP\n"
            "100 0.5 6.5 5.0 5.5 4.8\n"
        )
        with pytest.warns(UserWarning, match="Missing columns"):
            with pytest.raises(ValueError):
                MISTIsochrones(tmp_path)  # no valid isochrones remain

    def test_z_from_header_lines_static_method(self):
        """_z_from_header_lines can be called as a static method."""
        lines = ["Yinit  Zinit  FeH", " 0.270  0.0142  -0.03"]
        result = MISTIsochrones._z_from_header_lines(lines)
        assert result is not None
        assert abs(result - 0.0142) < 1e-6

    def test_z_from_header_lines_returns_none_when_absent(self):
        lines = ["EEP initial_mass log10_isochrone_age_yr Gaia_G_EDR3"]
        result = MISTIsochrones._z_from_header_lines(lines)
        assert result is None

    def test_get_isochrone_out_of_range_loga(self, tmp_path):
        """loga outside the grid range → nearest boundary returned."""
        _make_mist_file(tmp_path, Z=0.0152, n_ages=2)
        iso = MISTIsochrones(tmp_path)
        # Ask for loga much larger than any in the grid
        mass, G, BP, RP = iso.get_isochrone(0.0152, 99.9)
        assert len(mass) > 0  # something returned, not an error


def test_isochrone_fitter_exige_los_parametros_del_cumulo():
    """`loga_range`, `dm_mu` y `dm_range` no tienen default, y el error los nombra.

    Hasta 2026-08-26 valian ``(6.0, 7.0)``, ``10.2`` y ``(9.5, 10.7)`` -- la edad y el modulo de
    distancia de **NGC 6383**, horneados en una API general. La consecuencia es medible contra el
    censo Hunt & Reffert 2024: el cumulo mediano esta a 2.259 kpc, o sea DM = 11.77, **fuera** del
    rango por defecto; el percentil 99.5 da 14.54. Un ajuste sobre cualquier otro cumulo se truncaba
    en el borde del prior y devolvia una respuesta segura y equivocada, sin decir nada.

    Se afirma que **faltan los tres**, no que la llamada falle: un `TypeError` cualquiera lo daria
    tambien un argumento mal escrito.
    """
    with pytest.raises(TypeError) as exc:
        IsochroneFitter("cualquier_ruta")
    faltan = str(exc.value)
    for nombre in ("loga_range", "dm_mu", "dm_range"):
        assert nombre in faltan, f"{nombre} deberia ser obligatorio: {faltan}"
    # Y los que SI conservan default no aparecen: su valor no es del cumulo.
    for nombre in ("dm_sigma", "Rv", "Av_range"):
        assert nombre not in faltan, f"{nombre} no deberia ser obligatorio: {faltan}"


def test_el_analyzer_no_reinyecta_los_defaults_que_el_fitter_ya_exige():
    """Los DOS metodos del `Analyzer`, no solo el que se miro.

    El arreglo de 2026-08-26 quito los defaults de NGC 6383 de `IsochroneFitter` y de
    `Analyzer.prepare_isochrone_fitter`, y **dejo vivo `Analyzer.fit_isochrone`**, que construye su
    propio `IsochroneFitter` y le pasaba `loga_range=(6.0, 7.0)`, `dm_mu=10.2` y
    `dm_range=(9.5, 10.7)` desde sus propios defaults. O sea el parametro era obligatorio y el
    metodo de conveniencia lo rellenaba con la respuesta de un cumulo concreto: la correccion
    quedaba anulada justo en la entrada que mas gente usa.

    El guardia original probaba `IsochroneFitter.__init__` y por eso no podia verlo. Este recorre
    la firma de cada metodo del `Analyzer` que construya un fitter.
    """
    import inspect

    from erotica.analysis.analyzer import ClusterAnalyzer

    for nombre_metodo in ("prepare_isochrone_fitter", "fit_isochrone"):
        p = inspect.signature(getattr(ClusterAnalyzer, nombre_metodo)).parameters
        for arg in ("loga_range", "dm_mu", "dm_range"):
            assert p[arg].default is inspect.Parameter.empty, (
                f"{nombre_metodo}.{arg} tiene default {p[arg].default!r}: es un valor de NGC 6383 "
                "reinyectado en una API general"
            )
        # Y la columna de pertenencia esta en la firma, no escondida en el cuerpo.
        assert "probability_column" in p, f"{nombre_metodo} no expone probability_column"


def test_los_ejemplos_de_docstring_pasan_los_argumentos_que_el_metodo_exige():
    """Un ejemplo que ya no corre es documentacion que miente, y la suite no puede verlo.

    `pytest.ini` no corre doctests, asi que un `Examples` que quedo con un argumento de menos pasa
    verde para siempre. Los tres sitios que este guardia habria cazado estaban rotos cuando se
    escribio: el ejemplo de `prepare_isochrone_fitter` y la celda de `PROCESS.ipynb` los rompio
    `1646656` al volver obligatorios los tres parametros del cumulo, y el de `fit_isochrone` lo
    rompio `a539363` haciendo lo mismo un nivel mas arriba -- se le quedo `dm_range`.

    Es §K.1.38 en su forma barata: cuando un hueco sube de nivel, el guardia tiene que mirar a quien
    lo rellena, y un ejemplo de docstring **es** un llamador.
    """
    import inspect

    from erotica.analysis.analyzer import ClusterAnalyzer

    for nombre_metodo in ("prepare_isochrone_fitter", "fit_isochrone"):
        metodo = getattr(ClusterAnalyzer, nombre_metodo)
        firma = inspect.signature(metodo).parameters
        obligatorios = [
            n for n, par in firma.items() if par.default is inspect.Parameter.empty and n != "self"
        ]
        doc = inspect.getdoc(metodo) or ""
        assert ">>>" in doc, f"{nombre_metodo} no trae ejemplo; el guardia no puede pasar en vacio"
        # El ejemplo de ESTE metodo, no el de los que encadena despues. Los parentesis se
        # balancean: cortar en el primer `)` parte `loga_range=(6.0, 7.0)` por la mitad y el
        # guardia falla sobre su propia extraccion en vez de sobre el ejemplo.
        i = doc.index(f"{nombre_metodo}(")
        prof, fin = 0, len(doc)
        for k in range(i, len(doc)):
            if doc[k] == "(":
                prof += 1
            elif doc[k] == ")":
                prof -= 1
                if prof == 0:
                    fin = k + 1
                    break
        llamada = doc[i:fin]
        for arg in obligatorios:
            assert arg in llamada, (
                f"el ejemplo de {nombre_metodo} no pasa {arg!r}, que es obligatorio: "
                f"copiarlo da TypeError. Llamada del ejemplo:\n{llamada}"
            )


# ---------------------------------------------------------------------------
# Hess-grid frame and interpolation (2026-09-22)
# ---------------------------------------------------------------------------
#
# The June 2026 NUTS refit of NGC 6383 gave R-hat 1.5-2.2. Two defects in the precomputed
# Hess grid were measured behind it (hub finding isochrone-nuts-convergence-2026-09.md):
#
# 1. FRAME. The grid was binned in the absolute frame (dm = 0, A_V = 0) on the *apparent*
#    observed window and then shifted by the whole dm + k_G A_V: only stars with G_abs inside
#    the apparent window survived, so the model had zero mass brighter than G = 17 while 129
#    of the 254 members sit there.
# 2. STAIRCASE. Each regular grid point took the nearest file isochrone, so adjacent slices
#    were identical and d loglike / d met was exactly zero at 283 of 300 prior points.
#
# The oracle for both is an analytic toy isochrone family: photometry is a closed-form
# function of (mass, age, Z), so where the stars land at a given (dm, A_V) is known without
# going through the fitter.

_TOY_MASSES = np.geomspace(0.1, 8.0, 120)


def _toy_photometry(mass, loga, Z):
    """Closed-form (G_abs, BP-RP): younger / more metal-rich -> brighter / redder."""
    lm = np.log10(mass)
    G = 4.6 - 6.5 * lm - 2.0 * (7.0 - loga) / (1.0 + mass**2) + 40.0 * (Z - 0.015)
    col = 0.8 - 1.4 * lm + 20.0 * (Z - 0.015) + 0.1 * (7.0 - loga)
    return G, col


def _write_toy_family(tmp_path: Path, Zs, ages) -> Path:
    for k, Z in enumerate(Zs):
        header = textwrap.dedent(f"""\
            # toy isochrone family
            # Yinit  Zinit  FeH
            #  0.270  {Z:.6f}  0.00
            # EEP initial_mass log10_isochrone_age_yr Gaia_G_EDR3 Gaia_BP_EDR3 Gaia_RP_EDR3
        """)
        rows = []
        for loga in ages:
            G, col = _toy_photometry(_TOY_MASSES, loga, Z)
            for i, (m, g, c) in enumerate(zip(_TOY_MASSES, G, col, strict=True)):
                rows.append(f"{i} {m:.6f} {loga:.4f} {g:.6f} {g + 0.6 * c:.6f} {g - 0.4 * c:.6f}")
        (tmp_path / f"toy_{k}.iso.cmd").write_text(header + "\n".join(rows) + "\n")
    return tmp_path


def _toy_stars(n, loga, Z, dm, Av, rng, e=0.01):
    """Star-level draw from the toy family: Chabrier-2014 inverse CDF on a fine mass grid,
    photometry from the closed form (NOT from the isochrone file or the fitter), singles only."""
    m = np.geomspace(0.1, 8.0, 20000)
    ln = np.exp(-0.5 * ((np.log10(m) - np.log10(0.2)) / 0.55) ** 2) / m
    pdf = np.where(m < 1.0, ln, np.exp(-0.5 * (np.log10(0.2) / 0.55) ** 2) * m**-2.35)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(m))])
    mass = np.interp(rng.uniform(size=n), cdf / cdf[-1], m)
    G, col = _toy_photometry(mass, loga, Z)
    kG, kBP, kRP = (_ccm89(lam) for lam in (6390.7, 5182.6, 7825.1))
    Gapp = G + dm + kG * Av + rng.normal(0, e, n)
    BP = G + 0.6 * col + dm + kBP * Av + rng.normal(0, e, n)
    RP = G - 0.4 * col + dm + kRP * Av + rng.normal(0, e, n)
    return QTable(
        {
            "Gmag": Gapp,
            "G_BPmag": BP,
            "G_RPmag": RP,
            "e_Gmag": np.full(n, e),
            "e_G_BPmag": np.full(n, e),
            "e_G_RPmag": np.full(n, e),
            "probability_hdbscan": np.ones(n),
        }
    )


def _toy_fitter(tmp_path, *, M_met=3, M_loga=11, Zs=(0.010, 0.015, 0.020), **kw):
    ages = kw.pop("ages", (6.3, 6.4, 6.5, 6.6, 6.7, 6.8))
    _write_toy_family(tmp_path, Zs, ages)
    args = dict(
        loga_range=(6.3, 6.8),
        Av_range=(0.2, 1.0),
        dm_mu=10.0,
        dm_sigma=0.3,
        dm_range=(9.6, 10.4),
        alpha=0.0,  # single stars only: the toy oracle has no binaries
        beta=0.0,
        M_met=M_met,
        M_loga=M_loga,
    )
    args.update(kw)
    return IsochroneFitter(tmp_path, **args)


@requires_bayes_extra
class TestHessFrame:
    """Where the shifted model Hess puts its mass, against the closed-form toy isochrone."""

    @pytest.mark.parametrize("dm,Av", [(10.0, 0.6), (10.35, 0.95), (9.65, 0.25)])
    def test_model_mass_lands_where_the_isochrone_does(self, tmp_path, dm, Av):
        """Oracle: the toy node's own points, moved by (dm, A_V) in closed form.

        At a grid node (Z = 0.015, loga = 6.5, which the regular grid hits exactly) the
        shifted model Hess must hold the IMF weight of exactly the isochrone points that land
        inside the observed window, at their magnitudes. Checked at the reference shift and
        at both prior corners, because the old defect depended on the size of the shift.

        Mutation (measured 2026-09-22): binning the grid in the absolute frame and shifting
        by the full ``dm + k_G A_V`` (the pre-2026-09-22 code) leaves 0.17-0.28 of the IMF
        weight in the window here, against 0.90-1.00 expected, and all three cases fail.
        """
        import pytensor.tensor as pt

        f = _toy_fitter(tmp_path)
        f.setup(_toy_stars(400, 6.5, 0.015, 10.0, 0.6, np.random.default_rng(3)), prob_threshold=0)
        assert f._met_grid[1] == pytest.approx(0.015) and f._loga_grid[4] == pytest.approx(6.5)

        H = f._shift_histogram(
            f._interp_H(pt.constant(0.015), pt.constant(6.5)),
            pt.constant(dm + f._kG * Av),
            pt.constant(f._k_col1 * Av),
        ).eval()

        mass, G, BP, RP = f._isochs.get_isochrone(0.015, 6.5)
        w = _chabrier2014_weights(mass)
        g_app = G + dm + f._kG * Av
        c_app = BP - RP + f._k_col1 * Av
        (m0, m1), (c0, c1) = f._mag_range, f._col_range
        inside = (g_app >= m0) & (g_app < m1) & (c_app >= c0) & (c_app < c1)
        expected_mass = float(w[inside].sum())
        expected_mean_g = float(np.sum(w[inside] * g_app[inside]) / expected_mass)

        rows = m0 + (np.arange(f._Nbins[0]) + 0.5) * f._binw_mag
        model_mass = float(H.sum())
        model_mean_g = float(np.sum(H.sum(axis=1) * rows) / model_mass) if model_mass else np.inf

        assert expected_mass > 0.4  # the fixture really puts stars in the window
        # the 3x3 smoothing leaks up to one bin of mass across the window edge
        assert model_mass == pytest.approx(expected_mass, abs=0.06)
        assert abs(model_mean_g - expected_mean_g) < 0.5 * f._binw_mag

    def test_grid_interpolates_between_isochrone_nodes(self, tmp_path):
        """Oracle: linearity. Halfway between two metallicity nodes the grid must be the mean
        of the two node Hess diagrams, and no two adjacent slices may be identical.

        Mutation: nearest-node caching (the pre-2026-09-22 code) makes the midpoint equal to
        one node and leaves identical adjacent slices -- zero gradient between them.
        """
        f = _toy_fitter(tmp_path, M_met=5, M_loga=11)  # met grid 0.010, 0.0125, 0.015, ...
        f.setup(_toy_stars(400, 6.5, 0.015, 10.0, 0.6, np.random.default_rng(4)), prob_threshold=0)
        H = f._H_grid
        j = int(np.argmin(np.abs(f._loga_grid - 6.5)))
        assert f._loga_grid[j] == pytest.approx(6.5)
        np.testing.assert_allclose(H[1, j], 0.5 * (H[0, j] + H[2, j]), rtol=0, atol=1e-12)
        for axis in (0, 1):
            same = [
                np.array_equal(np.take(H, k, axis), np.take(H, k + 1, axis))
                for k in range(H.shape[axis] - 1)
            ]
            assert not any(same), f"identical adjacent slices along axis {axis}: {same}"

    def test_loglike_gradient_is_nonzero_in_met_and_loga(self, tmp_path):
        """NUTS needs d loglike / d(met, loga) != 0 between nodes. Measured on NGC 6383 before
        the fix: exactly zero in met at 283 / 300 random prior points, in loga at 133 / 300."""
        import pytensor
        import pytensor.tensor as pt

        f = _toy_fitter(tmp_path, M_met=9, M_loga=21)
        f.setup(_toy_stars(400, 6.5, 0.015, 10.0, 0.6, np.random.default_rng(5)), prob_threshold=0)
        met, loga = pt.dscalar("met"), pt.dscalar("loga")
        lam = pt.maximum(
            f._shift_histogram(f._interp_H(met, loga), 10.0 + f._kG * 0.6, f._k_col1 * 0.6).reshape(
                (-1,)
            ),
            1e-6,
        )
        mu = 400.0 * lam + 0.05
        ll = pt.sum(f._obs_hess * pt.log(mu) - mu)
        grad = pytensor.function([met, loga], pytensor.grad(ll, [met, loga]))
        rng = np.random.default_rng(0)
        zero = np.zeros(2, int)
        for _ in range(50):
            g = grad(rng.uniform(0.0101, 0.0199), rng.uniform(6.31, 6.79))
            zero += np.array([float(x) == 0.0 for x in g])
        assert zero.tolist() == [0, 0]

    def test_stale_grid_cache_is_refused(self, tmp_path):
        """A cache written by the absolute-frame code must not load: its model has no bright
        stars, and nothing downstream would notice."""
        f = _toy_fitter(tmp_path)
        f.setup(_toy_stars(200, 6.5, 0.015, 10.0, 0.6, np.random.default_rng(6)), prob_threshold=0)
        good = tmp_path / "good.npz"
        f.save_grid(good)
        d = dict(np.load(good))
        d.pop("frame_ref")
        d.pop("frame_pad")
        stale = tmp_path / "stale.npz"
        np.savez(stale, **d)
        with pytest.raises(ValueError, match="rebuild"):
            f.load_grid(stale)

    def test_widened_prior_beyond_the_padding_is_refused(self, tmp_path):
        f = _toy_fitter(tmp_path)
        f.setup(_toy_stars(200, 6.5, 0.015, 10.0, 0.6, np.random.default_rng(7)), prob_threshold=0)
        f.set_priors({"dm_range": (7.0, 13.0)})
        with pytest.raises(ValueError, match="padding"):
            f.build_model()


class _HalfBinReference(IsochroneFitter):
    """Identical model; only the grid's internal reference frame moves by half a bin."""

    def _set_reference_frame(self) -> None:
        super()._set_reference_frame()
        self._dmag_ref += 0.5 * self._binw_mag
        self._dcol_ref += 0.5 * self._binw_col
        self._pad = (self._pad[0] + 1, self._pad[1] + 1)


def _toy_loglike(f, met, loga, dm, Av, scale=400.0):
    import pytensor.tensor as pt

    lam = pt.maximum(
        f._shift_histogram(
            f._interp_H(pt.constant(met), pt.constant(loga)), dm + f._kG * Av, f._k_col1 * Av
        ).reshape((-1,)),
        1e-6,
    )
    mu = scale * lam + 0.05
    return float(pt.sum(f._obs_hess * pt.log(mu) - mu).eval())


@requires_bayes_extra
@pytest.mark.xfail(
    strict=True,
    # only the assertion counts as "defect still present": a crash (import, API change) must fail
    raises=AssertionError,
    reason=(
        "OPEN DEFECT (2026-09-22): shifting a precomputed histogram bilinearly is not binning "
        "shifted stars. The likelihood depends on the grid's internal reference: on NGC 6383 a "
        "half-bin move of the reference changes log L by -10.5 / +4.5 at fixed parameters, and "
        "posteriors lock onto the reference. Needs a per-evaluation deposit of shifted isochrone "
        "points (or an unbinned likelihood); see docs/design-notes/isochrone_sampler_fix.md."
    ),
)
def test_likelihood_does_not_depend_on_the_internal_reference_frame(tmp_path):
    """Oracle: invariance. The reference frame is bookkeeping; log L at fixed parameters cannot
    depend on it. Strict xfail, so the day the defect is fixed this turns red and must be
    promoted to a plain test."""
    data = _toy_stars(400, 6.55, 0.0125, 10.25, 0.7, np.random.default_rng(21))
    lls = []
    for cls in (IsochroneFitter, _HalfBinReference):
        (tmp_path / cls.__name__).mkdir()
        f = _toy_fitter(tmp_path / cls.__name__)
        f.__class__ = cls
        f.setup(data, prob_threshold=0)
        lls.append(_toy_loglike(f, 0.0125, 6.55, 10.25, 0.7))
    assert abs(lls[0] - lls[1]) < 0.5, lls


@requires_bayes_extra
@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    # only the assertion counts as "defect still present": a crash (import, API change) must fail
    raises=AssertionError,
    reason=(
        "OPEN DEFECT (2026-09-22): the posterior locks onto the grid's reference (dm_mu, "
        "mean(Av_range)) and onto isochrone metallicity nodes, so an injected truth off the "
        "reference is not recovered. See the reference-invariance test above."
    ),
)
def test_nuts_recovers_an_injected_toy_cluster(tmp_path):
    """Injection-recovery through the real sampler, with the Vehtari gate.

    Oracle: stars drawn star by star from the closed-form toy family (``_toy_stars``), never
    through the Hess grid. Truth is off the isochrone nodes and off the grid's reference. Tolerances are **absolute and tied to the
    truth**, not to the posterior width, and the posterior is checked to be informative first
    (tests/AGENTS.md, failure modes 2 and 4).

    Mutation (measured 2026-09-22): with the absolute-frame grid, ``dm`` and ``A_V`` are
    pulled against the lower walls of their priors and the gate fails.
    """
    f = _toy_fitter(tmp_path, M_met=9, M_loga=26)
    # Off every node (met between 0.010 and 0.015, loga between 6.5 and 6.6) and off the
    # grid's reference (dm_mu = 10.0, mean(Av_range) = 0.6) by about half a magnitude bin, so
    # a posterior stuck on either cannot pass.
    truth = {"met": 0.0125, "loga": 6.55, "dm": 10.25, "Av": 0.7}
    f.setup(
        _toy_stars(
            500, truth["loga"], truth["met"], truth["dm"], truth["Av"], np.random.default_rng(11)
        ),
        prob_threshold=0,
    )
    idata = f.fit(draws=1000, tune=1000, chains=2, cores=1, random_seed=11, progressbar=False)
    import arviz as az

    rhat, ess = az.rhat(idata.posterior), az.ess(idata.posterior)
    post = {p: idata.posterior[p].values.ravel() for p in truth}
    medians = {p: round(float(np.median(post[p])), 4) for p in truth}
    assert int(idata.sample_stats["diverging"].values.sum()) == 0
    for p in truth:
        assert float(rhat[p]) < 1.01, (p, float(rhat[p]), medians)
        assert float(ess[p]) > 400, (p, float(ess[p]), medians)
    # informative: at most half the prior sd, or the recovery below means nothing
    assert np.std(post["loga"]) < 0.5 * 0.5 / np.sqrt(12)
    assert np.std(post["Av"]) < 0.5 * 0.8 / np.sqrt(12)
    assert np.std(post["dm"]) < 0.5 * 0.3
    tol = {"met": 0.002, "loga": 0.1, "dm": 0.12, "Av": 0.08}
    misses = {p: (medians[p], t) for p, t in truth.items() if abs(medians[p] - t) >= tol[p]}
    assert not misses, f"median vs truth outside the absolute tolerance: {misses}"
