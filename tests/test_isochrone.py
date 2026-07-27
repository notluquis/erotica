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
    _chabrier2014_weights,
    _ccm89,
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
    reason="requires EROTICA's optional bayes extra: missing "
    + ", ".join(_BAYES_EXTRA_MISSING),
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
            G  =  4.0 + 2.0 * i
            BP =  4.5 + 2.0 * i
            RP =  3.8 + 2.0 * i
            rows.append(f"{100 + i} {m:.3f} {loga:.4f} {G:.4f} {BP:.4f} {RP:.4f}")

    fp = tmp_path / f"test_feh_p0.00.iso.cmd"
    fp.write_text(header + "\n".join(rows) + "\n")
    return fp


def _make_cluster_data(n: int = 80, rng: np.random.Generator | None = None) -> QTable:
    """Minimal cluster QTable that IsochroneFitter.setup() can consume."""
    if rng is None:
        rng = np.random.default_rng(42)
    mag = rng.uniform(12.0, 18.0, n)
    bp  = mag + rng.uniform(0.3, 0.8, n)
    rp  = mag - rng.uniform(0.1, 0.4, n)
    return QTable({
        "Gmag":             mag,
        "G_BPmag":          bp,
        "G_RPmag":          rp,
        "e_Gmag":           rng.uniform(0.003, 0.05, n),
        "e_G_BPmag":        rng.uniform(0.004, 0.06, n),
        "e_G_RPmag":        rng.uniform(0.004, 0.06, n),
        "probability_hdbscan": np.ones(n),
    })


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
        kG  = _ccm89(6390.7)
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
        low  = mass <= 1.0
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
        c  = 3.5
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
        fp = _make_mist_file(tmp_path, Z=0.0152, n_ages=2)
        iso = MISTIsochrones(tmp_path)
        assert iso._met_values is not None
        assert len(iso._met_values) == 1  # one Z value
        assert abs(iso._met_values[0] - 0.0152) < 1e-5

    def test_loga_values(self, tmp_path):
        fp = _make_mist_file(tmp_path, Z=0.0152, n_ages=3)
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
        # < 3 valid points → constant fallback, should not raise
        mag = np.array([14.0, 15.0])
        e_m = np.array([0.01, 0.02])
        e_c = np.array([0.015, 0.025])
        f_m, f_c = _fit_error_model(mag, e_m, e_c)
        assert np.isfinite(f_m(np.array([16.0]))[0])


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
        assert fitter._H_grid.shape == (Nm, Na, Nb_m, Nb_c)

    def test_interp_H_returns_tensor(self, fitter_and_data):
        import pytensor.tensor as pt
        fitter, _ = fitter_and_data
        met_t  = pt.as_tensor_variable(np.float64(0.0152))
        loga_t = pt.as_tensor_variable(np.float64(6.5))
        H = fitter._interp_H(met_t, loga_t)
        result = H.eval()
        Nb_m, Nb_c = fitter._Nbins
        assert result.shape == (Nb_m, Nb_c)
        assert np.all(np.isfinite(result))

    def test_shift_histogram_identity(self, fitter_and_data):
        """Shifting by (0, 0) should return the original histogram."""
        import pytensor.tensor as pt
        fitter, _ = fitter_and_data
        Nb_m, Nb_c = fitter._Nbins
        H_np = np.random.default_rng(0).uniform(0, 1, (Nb_m, Nb_c))
        H_t  = pt.as_tensor_variable(H_np)
        dmag = pt.as_tensor_variable(np.float64(0.0))
        dcol = pt.as_tensor_variable(np.float64(0.0))
        shifted = fitter._shift_histogram(H_t, dmag, dcol).eval()
        # Zero shift: interior pixels should match (boundary may differ due to valid mask)
        s = 1  # skip outermost ring
        np.testing.assert_allclose(
            shifted[s:-s, s:-s], H_np[s:-s, s:-s], atol=1e-10
        )

    def test_shift_histogram_mass_conservation(self, fitter_and_data):
        """A small shift should not increase total mass significantly."""
        import pytensor.tensor as pt
        fitter, _ = fitter_and_data
        Nb_m, Nb_c = fitter._Nbins
        H_np = np.ones((Nb_m, Nb_c), dtype=float)
        H_t  = pt.as_tensor_variable(H_np)
        bw = min(fitter._binw_mag, fitter._binw_col)
        dmag = pt.as_tensor_variable(np.float64(bw))   # shift by 1 bin
        dcol = pt.as_tensor_variable(np.float64(0.0))
        shifted = fitter._shift_histogram(H_t, dmag, dcol).eval()
        assert shifted.sum() <= H_np.sum() + 1e-6

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
        assert fitter2._kG  == pytest.approx(fitter._kG)
        assert fitter2._kBP == pytest.approx(fitter._kBP)
        assert fitter2._kRP == pytest.approx(fitter._kRP)

    def test_posterior_cmd_shape(self, fitter_and_data):
        """posterior_cmd should return obs arrays and a list of (mag, col) tuples."""
        import arviz as az

        fitter, _ = fitter_and_data
        # Build a tiny fake idata with 2 draws × 1 chain
        n = 2
        rng = np.random.default_rng(1)
        idata = az.from_dict({
            "met":  rng.uniform(0.014, 0.016, (1, n)),
            "loga": rng.uniform(6.5, 6.7, (1, n)),
            "dm":   rng.uniform(9.8, 10.2, (1, n)),
            "Av":   rng.uniform(0.0, 0.5, (1, n)),
        })
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
            dm_mu=10.0, dm_sigma=0.3,
            dm_range=(9.0, 11.0),
            M_met=4, M_loga=4,
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
            dm_mu=base_fitter.dm_mu, dm_sigma=base_fitter.dm_sigma,
            dm_range=base_fitter.dm_range,
            M_met=base_fitter.M_met, M_loga=base_fitter.M_loga,
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
        I = base_fitter._I_tensor.eval()
        J = base_fitter._J_tensor.eval()
        assert I.shape == (Nb_m, 1)
        assert J.shape == (1, Nb_c)

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
        H_t  = pt.as_tensor_variable(np.zeros((Nb_m, Nb_c)))
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
            dm_mu=base_fitter.dm_mu, dm_sigma=base_fitter.dm_sigma,
            dm_range=base_fitter.dm_range,
            M_met=base_fitter.M_met, M_loga=base_fitter.M_loga,
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
        k_ir  = _ccm89(20000.0)
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
    @pytest.mark.parametrize("mass, gamma", [
        (0.05,  4.2),   # m ≤ 0.1
        (0.3,   0.4),   # 0.1 < m ≤ 0.6
        (1.0,   0.3),   # 0.6 < m ≤ 1.4
        (3.0,  -0.5),   # 1.4 < m ≤ 6.5
        (10.0,  0.0),   # m > 6.5
    ])
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
        e_m[::3] = np.nan   # every 3rd point is NaN
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
            dm_mu=10.0, dm_sigma=0.3,
            dm_range=(9.0, 11.0),
            M_met=4, M_loga=4,
        )
        return fitter

    def test_prob_threshold_filters_members(self, setup_fitter):
        rng = np.random.default_rng(0)
        n = 100
        mag = rng.uniform(12.0, 18.0, n)
        bp  = mag + 0.5
        rp  = mag - 0.2
        probs = np.concatenate([np.ones(50), np.zeros(50)])
        data = QTable({
            "Gmag": mag, "G_BPmag": bp, "G_RPmag": rp,
            "e_Gmag": np.full(n, 0.01),
            "e_G_BPmag": np.full(n, 0.012),
            "e_G_RPmag": np.full(n, 0.012),
            "probability_hdbscan": probs,
        })
        setup_fitter.setup(data, prob_threshold=0.5)
        assert setup_fitter._N_obs == 50
        assert len(setup_fitter._obs_mag) == 50

    def test_e_BP_RP_column_used_when_present(self, setup_fitter):
        """e_BP_RP column should be used directly instead of hypot(e_BP, e_RP)."""
        n = 40
        rng = np.random.default_rng(7)
        mag = rng.uniform(12.0, 18.0, n)
        data_separate = QTable({
            "Gmag": mag, "G_BPmag": mag + 0.5, "G_RPmag": mag - 0.2,
            "e_Gmag":    np.full(n, 0.01),
            "e_G_BPmag": np.full(n, 0.02),
            "e_G_RPmag": np.full(n, 0.02),
            "probability_hdbscan": np.ones(n),
        })
        data_combined = QTable({
            "Gmag": mag, "G_BPmag": mag + 0.5, "G_RPmag": mag - 0.2,
            "e_Gmag":  np.full(n, 0.01),
            "e_BP_RP": np.full(n, 0.05),   # explicit combined column
            "probability_hdbscan": np.ones(n),
        })
        f1 = IsochroneFitter(
            isochs_path=setup_fitter.isochs_path,
            loga_range=setup_fitter.loga_range, Av_range=setup_fitter.Av_range,
            dm_mu=setup_fitter.dm_mu, dm_sigma=setup_fitter.dm_sigma,
            dm_range=setup_fitter.dm_range, M_met=4, M_loga=4,
        )
        f2 = IsochroneFitter(
            isochs_path=setup_fitter.isochs_path,
            loga_range=setup_fitter.loga_range, Av_range=setup_fitter.Av_range,
            dm_mu=setup_fitter.dm_mu, dm_sigma=setup_fitter.dm_sigma,
            dm_range=setup_fitter.dm_range, M_met=4, M_loga=4,
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
        assert H.shape == (Nb_m, Nb_c)
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
        idata = az.from_dict({
            "met":  rng.uniform(0.014, 0.016, (1, 3)),
            "loga": rng.uniform(6.5, 6.6, (1, 3)),
            "dm":   rng.uniform(9.5, 10.5, (1, 3)),
            "Av":   rng.uniform(0.0, 0.5, (1, 3)),
        })
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
        idata = az.from_dict({
            "met":  rng.uniform(0.014, 0.016, (1, 2)),
            "loga": rng.uniform(6.5, 6.6, (1, 2)),
            "dm":   rng.uniform(9.8, 10.2, (1, 2)),
            "Av":   rng.uniform(0.0, 0.5, (1, 2)),
        })
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
            "100 0.3 6.5 nan 4.5 3.8\n"   # NaN G → excluded
            "101 0.5 6.5 5.0 5.5 4.8\n"   # good
            "102 1.0 6.5 4.0 4.5 3.8\n"   # good
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
