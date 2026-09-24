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
        )
        fitter.setup(data, prob_threshold=0.5)
        return fitter, data

    def test_setup_sets_obs_arrays(self, fitter_and_data):
        fitter, _ = fitter_and_data
        assert fitter._obs_mag is not None
        assert fitter._obs_col is not None
        assert len(fitter._obs_mag) > 0
        assert fitter._e_obs_mag.shape == fitter._obs_mag.shape

    def test_setup_sets_ext_coefs(self, fitter_and_data):
        fitter, _ = fitter_and_data
        assert fitter._kG is not None
        assert fitter._kBP is not None
        assert fitter._kRP is not None
        # BP band has more extinction than G and RP in optical
        assert fitter._kBP > fitter._kG > fitter._kRP

    def test_node_table_shape(self, fitter_and_data):
        """(n_Z, n_age, 3 + 2 x q-nodes, n_EEP): one Z file, three ages, EEP 100..105."""
        from erotica.analysis._isochrone import _Q_NODES

        fitter, _ = fitter_and_data
        assert fitter._nodes.shape == (1, 3, 3 + 2 * len(_Q_NODES), 6)
        assert np.all(np.isfinite(fitter._nodes))

    def test_save_load_roundtrip(self, fitter_and_data, tmp_path):
        fitter, data = fitter_and_data
        cache = tmp_path / "nodes.npz"
        fitter.save_grid(cache)

        fitter2 = IsochroneFitter(
            isochs_path=fitter.isochs_path,
            loga_range=fitter.loga_range,
            Av_range=fitter.Av_range,
            dm_mu=fitter.dm_mu,
            dm_sigma=fitter.dm_sigma,
            dm_range=fitter.dm_range,
        )
        fitter2.setup(data, prob_threshold=0.5, precompute_grid=False)
        assert fitter2._nodes is None
        fitter2.load_grid(cache)

        np.testing.assert_array_equal(fitter._nodes, fitter2._nodes)
        args = (0.0152, 6.6, 10.0, 0.3, 0.05, 0.02)
        assert fitter2.loglike(*args) == fitter.loglike(*args)

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
    """The sampler's PyTensor graph and the NumPy evaluation are the same function."""

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
        )

    def test_nodes_tensor_matches_nodes(self, base_fitter):
        base_fitter.setup(_make_cluster_data(n=40), prob_threshold=0.0)
        np.testing.assert_array_equal(base_fitter._nodes_tensor.get_value(), base_fitter._nodes)
        # a static shape, or slicing its rows gives JAX a traced length (measured 2026-09-22)
        assert base_fitter._nodes_tensor.type.shape == base_fitter._nodes.shape

    def test_pytensor_graph_equals_numpy(self, base_fitter):
        import pytensor
        import pytensor.tensor as pt

        base_fitter.setup(_make_cluster_data(n=40), prob_threshold=0.0)
        v = [pt.dscalar(n) for n in ("met", "loga", "dm", "Av", "s", "f")]
        fn = pytensor.function(v, pt.sum(base_fitter._star_loglike(*v, pt)))
        x = (0.0152, 6.55, 10.1, 0.4, 0.03, 0.02)
        assert float(fn(*x)) == pytest.approx(base_fitter.loglike(*x), rel=1e-10, abs=1e-8)

    def test_jax_graph_equals_numpy(self, base_fitter, tmp_path):
        """The sampler numpyro compiles is the JAX graph. Run in a subprocess: initialising
        JAX here would make later forking tests raise JAX's fork warning (an error under this
        suite's warning policy -- measured 2026-09-23, two test_structure tests)."""
        import subprocess
        import sys

        pytest.importorskip("jax")
        code = textwrap.dedent(f"""
            import numpy as np, pytensor, pytensor.tensor as pt
            from tests.test_isochrone import _make_cluster_data
            from erotica.analysis._isochrone import IsochroneFitter
            f = IsochroneFitter(isochs_path={str(base_fitter.isochs_path)!r}, loga_range=(6.5, 6.6),
                                Av_range=(0.0, 1.0), dm_mu=10.0, dm_sigma=0.3, dm_range=(9.0, 11.0))
            f.setup(_make_cluster_data(n=40), prob_threshold=0.0)
            v = [pt.dscalar(n) for n in ("met", "loga", "dm", "Av", "s", "f")]
            fn = pytensor.function(v, pt.sum(f._star_loglike(*v, pt)), mode="JAX")
            x = (0.0152, 6.55, 10.1, 0.4, 0.03, 0.02)
            print(repr(float(fn(*x))), repr(f.loglike(*x)))
        """)
        root = Path(__file__).resolve().parent.parent
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=root, check=True
        ).stdout.split()
        assert float(out[-2]) == pytest.approx(float(out[-1]), rel=1e-10, abs=1e-8)

    def test_build_model_after_load_grid_does_not_crash(self, base_fitter, tmp_path):
        data = _make_cluster_data(n=40)
        base_fitter.setup(data, prob_threshold=0.0)
        cache = tmp_path / "nodes_bm.npz"
        base_fitter.save_grid(cache)

        fitter2 = IsochroneFitter(
            isochs_path=base_fitter.isochs_path,
            loga_range=base_fitter.loga_range,
            Av_range=base_fitter.Av_range,
            dm_mu=base_fitter.dm_mu,
            dm_sigma=base_fitter.dm_sigma,
            dm_range=base_fitter.dm_range,
        )
        fitter2.setup(data, prob_threshold=0.0, precompute_grid=False)
        fitter2.load_grid(cache)
        model = fitter2.build_model()
        free_names = {v.name for v in model.free_RVs}
        for param in ("loga", "dm", "Av", "sigma_int", "f_bg"):
            assert any(param in n for n in free_names), f"{param} missing from model"
        # one metallicity file: Z is fixed, not a free parameter with a zero-width prior
        assert "met" in model.named_vars and "met" not in free_names
        # the model's log-density is finite at its initial point
        assert np.isfinite(model.compile_logp()(model.initial_point()))


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

    def test_precompute_false_leaves_nodes_none(self, setup_fitter):
        data = _make_cluster_data(n=40)
        setup_fitter.setup(data, prob_threshold=0.0, precompute_grid=False)
        assert setup_fitter._nodes is None
        assert setup_fitter._nodes_tensor is None

    def test_star_loglike_finite(self, setup_fitter):
        data = _make_cluster_data(n=40)
        setup_fitter.setup(data, prob_threshold=0.0)
        v = setup_fitter._star_loglike(0.0152, 6.55, 10.0, 0.3, 0.02, 0.05)
        assert v.shape == (40,)
        assert np.all(np.isfinite(v))

    def test_save_grid_raises_when_no_nodes(self, setup_fitter, tmp_path):
        with pytest.raises(RuntimeError, match="No node table"):
            setup_fitter.save_grid(tmp_path / "never.npz")

    def test_build_model_raises_before_setup(self, setup_fitter):
        with pytest.raises(RuntimeError):
            setup_fitter.build_model()

    def test_build_model_has_correct_free_vars(self, setup_fitter):
        data = _make_cluster_data(n=40)
        setup_fitter.setup(data, prob_threshold=0.0)
        model = setup_fitter.build_model()
        free_names = {v.name for v in model.free_RVs}
        for param in ("loga", "dm", "Av", "sigma_int", "f_bg"):
            assert any(param in n for n in free_names), f"'{param}' not in free_RVs"
        assert "met" in model.named_vars  # fixed: the fixture has one metallicity file

    def test_posterior_cmd_mag_within_range(self, setup_fitter):
        """Every returned star passes the completeness cut (the faintest member)."""
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
        for mag_s, col_s in cmds:
            assert np.all(mag_s <= setup_fitter._mag_lim)
            assert np.all(np.isfinite(col_s))

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
# The likelihood against closed-form oracles (2026-09-22)
# ---------------------------------------------------------------------------
#
# Until 2026-09-22 the likelihood shifted a precomputed Hess grid. Its frame bug and staircase
# were fixed first (commits e79c42f, 507f779), and what remained was measured to be unfixable
# inside that design: log L depended on the grid's internal reference, and posteriors locked
# onto (dm_mu, mean(Av_range)) and onto metallicity nodes (hub finding
# isochrone-nuts-convergence-2026-09.md). The likelihood is now unbinned per star over
# EEP-interpolated isochrones, and these tests hold it to oracles that do not go through it:
#
# * an analytic toy isochrone family, photometry closed-form in (mass, age, Z), drawn star by
#   star (with its own binaries) -- the Monte Carlo oracle for the deposit;
# * invariance: the likelihood cannot depend on the prior's centre (the old defect);
# * the node itself: at a node the interpolated isochrone is the file;
# * injection-recovery through NUTS with the Vehtari gate.

_TOY_MASSES = np.geomspace(0.1, 8.0, 120)


def _toy_photometry(mass, loga, Z):
    """Closed-form (G_abs, BP-RP): younger / more metal-rich -> brighter / redder.

    The metallicity term depends on mass, as it does in real isochrones. Until 2026-09-23 it
    was a pure additive shift, (dG, dcol) = (40, 20) dZ at every mass -- exactly what some
    (dm, A_V) reproduces -- so Z was not identifiable in this family at all: log L along
    that line was 326.100 to the third decimal for met from 0.0105 to 0.0145 (measured), and
    the NUTS recovery test could not pass for a correct likelihood (tests/AGENTS.md failure
    mode 4). The grid likelihood only looked like it pinned met because it locked onto a node.
    """
    lm = np.log10(mass)
    dz = Z - 0.015
    G = 4.6 - 6.5 * lm - 2.0 * (7.0 - loga) / (1.0 + mass**2) + 40.0 * dz * (1.0 - lm)
    col = 0.8 - 1.4 * lm + 20.0 * dz * (1.0 + lm) + 0.1 * (7.0 - loga)
    return G, col


def _write_toy_family(tmp_path: Path, Zs, ages, trim_per_age: int = 0) -> Path:
    """One file per Z. ``trim_per_age = k`` drops the k·j lowest and highest EEPs of the j-th
    age, so the EEP range changes from node to node as it does in MIST."""
    for k, Z in enumerate(Zs):
        header = textwrap.dedent(f"""\
            # toy isochrone family
            # Yinit  Zinit  FeH
            #  0.270  {Z:.6f}  0.00
            # EEP initial_mass log10_isochrone_age_yr Gaia_G_EDR3 Gaia_BP_EDR3 Gaia_RP_EDR3
        """)
        rows = []
        for j, loga in enumerate(ages):
            G, col = _toy_photometry(_TOY_MASSES, loga, Z)
            for i, (m, g, c) in enumerate(zip(_TOY_MASSES, G, col, strict=True)):
                if i < trim_per_age * j or i >= len(_TOY_MASSES) - trim_per_age * j:
                    continue
                rows.append(f"{i} {m:.6f} {loga:.4f} {g:.6f} {g + 0.6 * c:.6f} {g - 0.4 * c:.6f}")
        (tmp_path / f"toy_{k}.iso.cmd").write_text(header + "\n".join(rows) + "\n")
    return tmp_path


def _toy_stars(n, loga, Z, dm, Av, rng, e=0.01, binaries=False, alpha=0.09, beta=0.94):
    """Star-level draw from the toy family: Chabrier-2014 inverse CDF on a fine mass grid,
    photometry from the closed form (NOT from the isochrone file or the fitter). With
    ``binaries``: Offner fraction, D&K power-law q drawn per star with the exact step in
    primary mass, companion mass floored at 0.1 Msun, fluxes summed in closed form."""
    m = np.geomspace(0.1, 8.0, 20000)
    ln = np.exp(-0.5 * ((np.log10(m) - np.log10(0.2)) / 0.55) ** 2) / m
    pdf = np.where(m < 1.0, ln, np.exp(-0.5 * (np.log10(0.2) / 0.55) ** 2) * m**-2.35)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(m))])
    mass = np.interp(rng.uniform(size=n), cdf / cdf[-1], m)
    G, col = _toy_photometry(mass, loga, Z)
    BP0, RP0 = G + 0.6 * col, G - 0.4 * col
    if binaries:
        isb = rng.uniform(size=n) < np.clip(alpha + beta / (1 + 1.4 / mass), 0, 1)
        gam = np.where(
            mass <= 0.1,
            4.2,
            np.where(
                mass <= 0.6, 0.4, np.where(mass <= 1.4, 0.3, np.where(mass <= 6.5, -0.5, 0.0))
            ),
        )
        m2 = np.maximum(rng.uniform(size=n) ** (1 / (gam + 1)) * mass, 0.1)
        G2, col2 = _toy_photometry(m2, loga, Z)
        flux = lambda x: 10 ** (-0.4 * x)  # noqa: E731
        comb = lambda a, b: -2.5 * np.log10(flux(a) + flux(b))  # noqa: E731
        BP0 = np.where(isb, comb(BP0, G2 + 0.6 * col2), BP0)
        RP0 = np.where(isb, comb(RP0, G2 - 0.4 * col2), RP0)
        G = np.where(isb, comb(G, G2), G)
    kG, kBP, kRP = (_ccm89(lam) for lam in (6390.7, 5182.6, 7825.1))
    Gapp = G + dm + kG * Av + rng.normal(0, e, n)
    BP = BP0 + dm + kBP * Av + rng.normal(0, e, n)
    RP = RP0 + dm + kRP * Av + rng.normal(0, e, n)
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


def _toy_fitter(tmp_path, *, Zs=(0.010, 0.015, 0.020), trim_per_age=0, **kw):
    ages = kw.pop("ages", (6.3, 6.4, 6.5, 6.6, 6.7, 6.8))
    _write_toy_family(tmp_path, Zs, ages, trim_per_age=trim_per_age)
    args = dict(
        loga_range=(6.3, 6.8),
        Av_range=(0.2, 1.0),
        dm_mu=10.0,
        dm_sigma=0.3,
        dm_range=(9.6, 10.4),
        alpha=0.0,  # single stars unless a test asks for the toy binaries
        beta=0.0,
    )
    args.update(kw)
    return IsochroneFitter(tmp_path, **args)


def _set_probes(f, g, c, e):
    """Replace the fitter's stars by probe points with photometric error ``e`` (colour error
    ``sqrt(2) e``, as ``hypot(e_BP, e_RP)`` gives for the toy tables)."""
    f._obs_mag, f._obs_col = np.asarray(g, float), np.asarray(c, float)
    f._e_obs_mag = np.full(f._obs_mag.shape, e)
    f._e_obs_col = np.full(f._obs_mag.shape, np.sqrt(2) * e)
    f._star_weights = np.ones(f._obs_mag.shape)


@requires_bayes_extra
class TestUnbinnedLikelihood:
    """The per-star likelihood against oracles that do not go through it."""

    def test_interpolated_isochrone_at_a_node_is_the_file(self, tmp_path):
        """Oracle: the node's own file rows. At a node the interpolation must return them
        exactly; halfway between two Z nodes in log Z, the mean of the two (linearity at
        fixed EEP). Mutation: interpolating in linear Z moves the midpoint off the mean."""
        f = _toy_fitter(tmp_path)
        f.setup(_toy_stars(200, 6.5, 0.015, 10.0, 0.6, np.random.default_rng(1)), prob_threshold=0)
        X = f._interp_isochrone(0.015, 6.5)
        G_file, col_file = _toy_photometry(_TOY_MASSES, 6.5, 0.015)
        np.testing.assert_allclose(X[0], _TOY_MASSES, rtol=0, atol=1e-6)
        np.testing.assert_allclose(X[1], G_file, atol=2e-6)  # the file keeps 6 decimals
        np.testing.assert_allclose(X[2], col_file, atol=2e-6)
        z_mid = float(np.sqrt(0.010 * 0.015))
        np.testing.assert_allclose(
            f._interp_isochrone(z_mid, 6.5),
            0.5 * (f._interp_isochrone(0.010, 6.5) + f._interp_isochrone(0.015, 6.5)),
            atol=1e-12,
        )

    def test_likelihood_does_not_depend_on_the_prior_centre(self, tmp_path):
        """Oracle: invariance -- the defect that killed the precomputed-grid likelihood, where
        ``dm_mu`` and ``mean(Av_range)`` fixed the grid's frame, the Hess window and the error
        kernel. Same data, two fitters differing only in those prior settings: the likelihood
        at fixed parameters must be identical (the priors legitimately differ; they are not
        part of it). Mutations seen red 2026-09-22: error model evaluated at ``G - dm + dm_mu``
        in the completeness term; completeness cut taken from a reference isochrone at
        ``dm_mu``."""
        data = _toy_stars(300, 6.55, 0.0125, 10.25, 0.7, np.random.default_rng(21))
        lls = []
        for k, (dm_mu, av) in enumerate([(10.0, (0.2, 1.0)), (10.3, (0.5, 1.4))]):
            (tmp_path / str(k)).mkdir()
            f = _toy_fitter(tmp_path / str(k), dm_mu=dm_mu, Av_range=av)
            f.setup(data, prob_threshold=0)
            lls.append([f.loglike(0.0125, 6.55, dm, 0.7, 0.01, 0.01) for dm in (10.15, 10.25)])
        assert lls[0][0] != pytest.approx(lls[0][1], abs=1.0)  # the probe does move log L
        assert lls[0] == pytest.approx(lls[1], rel=0, abs=1e-9), lls

    @pytest.mark.parametrize("binaries", [False, True])
    @pytest.mark.parametrize("dm,Av", [(10.0, 0.6), (10.35, 0.95), (9.65, 0.25)])
    def test_density_matches_monte_carlo_stars(self, tmp_path, dm, Av, binaries):
        """Oracle: 400 000 toy stars drawn star by star in closed form (``_toy_stars``, own
        IMF inverse CDF, own binaries), observed with Gaussian errors and the completeness cut.
        The model's per-star likelihood with ``f_bg = 0`` is the density of *detected* stars,
        so on probe points it must match the Monte Carlo cell fractions. Checked at a node,
        at the prior centre and at both prior corners, singles and binaries.

        What it catches (mutations seen red 2026-09-22): the 1/L of the segment integral,
        IMF weights without the mass step, the completeness term with the wrong sign,
        mass-ratio probabilities not normalised, binaries deposited at q = 1 only.
        """
        e = 0.08
        kw = {"alpha": 0.09, "beta": 0.94} if binaries else {}
        f = _toy_fitter(tmp_path, **kw)
        f.setup(
            _toy_stars(
                400, 6.5, 0.015, dm, Av, np.random.default_rng(5), e=e, binaries=binaries, **kw
            ),
            prob_threshold=0,
        )
        mc = _toy_stars(
            400_000, 6.5, 0.015, dm, Av, np.random.default_rng(6), e=e, binaries=binaries, **kw
        )
        g = np.asarray(mc["Gmag"])
        c = np.asarray(mc["G_BPmag"]) - np.asarray(mc["G_RPmag"])
        det = g <= f._mag_lim
        g, c = g[det], c[det]
        # cells of 0.25 mag x 0.1 mag; model integrated with a 5 x 5 midpoint rule per cell
        ge = np.arange(np.floor(g.min()), f._mag_lim + 0.25, 0.25)
        ce = np.arange(np.floor(10 * c.min()) / 10, c.max() + 0.1, 0.1)
        H, _, _ = np.histogram2d(g, c, bins=[ge, ce])
        busy = np.argwhere(H >= 2000)
        sub = (np.arange(5) + 0.5) / 5
        pg = (ge[busy[:, 0], None, None] + 0.25 * sub[None, :, None]) * np.ones((1, 1, 5))
        pc = (ce[busy[:, 1], None, None] + 0.1 * sub[None, None, :]) * np.ones((1, 5, 1))
        _set_probes(f, pg.ravel(), pc.ravel(), e)
        dens = np.exp(f._star_loglike(0.015, 6.5, dm, Av, 0.0, 0.0)).reshape(len(busy), 25)
        model = dens.mean(axis=1) * 0.25 * 0.1 * g.size
        obs = H[busy[:, 0], busy[:, 1]]
        assert len(busy) >= 15
        # measured 2026-09-22 on the correct code: chi2/dof 0.87-1.17, max |pull| 2.2-3.6 over
        # 74-79 cells in the six cases -- Poisson noise. chi2/dof sd is ~0.16 at 77 dof.
        pull = (obs - model) / np.sqrt(model)
        assert np.mean(pull**2) < 1.5 and np.max(np.abs(pull)) < 5, np.round(pull, 1)

    def test_eeps_a_node_lacks_carry_no_weight_there(self, tmp_path):
        """MIST isochrones change EEP range with age. The node table puts every node on one
        EEP axis and clamps the EEPs a node lacks to its first/last point; at that node their
        mass step, hence IMF weight, must be exactly zero -- otherwise phantom stars appear.
        Mutation: extrapolating the missing EEPs linearly gives them weight."""
        f = _toy_fitter(tmp_path, trim_per_age=4)  # 6.3 keeps all EEPs, 6.8 loses 20 per end
        f.setup(_toy_stars(200, 6.5, 0.015, 10.0, 0.6, np.random.default_rng(2)), prob_threshold=0)
        d = f._deposit(0.015, 6.8, 10.0, 0.6)
        lacks = np.r_[np.arange(20), np.arange(len(_TOY_MASSES) - 21, len(_TOY_MASSES) - 1)]
        assert np.all(d["w_single"][lacks] == 0.0)
        assert d["w_single"][25:95].min() > 0.0

    def test_loglike_is_continuous_across_nodes(self, tmp_path):
        """At an age or Z node the two brackets on either side must give the same log L,
        including where the EEP range changes between nodes."""
        f = _toy_fitter(tmp_path, trim_per_age=4)
        f.setup(_toy_stars(300, 6.5, 0.015, 10.0, 0.6, np.random.default_rng(8)), prob_threshold=0)
        for met, loga in ((0.013, 6.5), (0.015, 6.55)):
            eps = (0.0, 1e-12) if met == 0.013 else (1e-14, 0.0)
            lo = f.loglike(met - eps[0], loga - eps[1], 10.0, 0.6, 0.01, 0.01)
            hi = f.loglike(met + eps[0], loga + eps[1], 10.0, 0.6, 0.01, 0.01)
            assert hi == pytest.approx(lo, abs=1e-6)

    def test_loglike_gradient_is_nonzero_in_met_and_loga(self, tmp_path):
        """NUTS needs d loglike / d(met, loga) != 0. Measured on NGC 6383 with the
        nearest-node Hess grid: exactly zero in met at 283 / 300 prior points, in loga at
        133 / 300."""
        import pytensor
        import pytensor.tensor as pt

        f = _toy_fitter(tmp_path)
        f.setup(_toy_stars(400, 6.5, 0.015, 10.0, 0.6, np.random.default_rng(5)), prob_threshold=0)
        met, loga = pt.dscalar("met"), pt.dscalar("loga")
        ll = pt.sum(f._star_loglike(met, loga, 10.0, 0.6, 0.01, 0.01, pt))
        grad = pytensor.function([met, loga], pytensor.grad(ll, [met, loga]))
        rng = np.random.default_rng(0)
        zero = np.zeros(2, int)
        for _ in range(50):
            g = grad(rng.uniform(0.0101, 0.0199), rng.uniform(6.31, 6.79))
            zero += np.array([float(x) == 0.0 for x in g])
        assert zero.tolist() == [0, 0]

    def test_stale_hess_grid_cache_is_refused(self, tmp_path):
        """A cache from the precomputed-Hess likelihood must not load as a node table."""
        f = _toy_fitter(tmp_path)
        f.setup(_toy_stars(200, 6.5, 0.015, 10.0, 0.6, np.random.default_rng(6)), prob_threshold=0)
        stale = tmp_path / "hgrid.npz"
        np.savez(stale, H_grid=np.zeros((2, 2, 3, 3)), frame_ref=np.zeros(2), frame_pad=np.ones(2))
        with pytest.raises(ValueError, match="rebuild"):
            f.load_grid(stale)

    def test_age_prior_beyond_the_node_table_is_refused(self, tmp_path):
        f = _toy_fitter(tmp_path, ages=(6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9))
        f.setup(_toy_stars(200, 6.5, 0.015, 10.0, 0.6, np.random.default_rng(7)), prob_threshold=0)
        f.set_priors({"loga_range": (6.3, 6.9)})
        with pytest.raises(ValueError, match="rebuild"):
            f.build_model()
        f.set_priors({"dm_range": (7.0, 13.0), "loga_range": (6.3, 6.8)})  # dm: no padding now
        assert f.build_model() is not None


# The strict xfail ``test_likelihood_does_not_depend_on_the_internal_reference_frame`` (2026-09-22)
# moved the precomputed grid's reference via ``_set_reference_frame``. That method no longer
# exists, so the test would pass by construction; it is replaced by
# ``TestUnbinnedLikelihood.test_likelihood_does_not_depend_on_the_prior_centre``, which checks
# the three routes by which the prior centre used to reach the likelihood.


@requires_bayes_extra
@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    # only the assertion counts as "defect still present": a crash (import, API change) must fail
    raises=AssertionError,
    reason=(
        "OPEN (2026-09-24): sampling efficiency, not bias. The medians are inside the absolute "
        "tolerances (measured: met 0.01251, loga 6.535, dm 10.300, A_V 0.723 against 0.0125, "
        "6.55, 10.25, 0.7), R-hat < 1.01 and 0 divergences at 2 x 3000 draws -- but the dm-A_V "
        "ridge mixes with ESS/draw ~0.05, so ESS_bulk is ~115 at the 2 x 1000 used here and "
        "304-335 at 2 x 3000, below the 400 gate. See the 2026-09-23 entry of decisions.md."
    ),
)
def test_nuts_recovers_an_injected_toy_cluster(tmp_path):
    """Injection-recovery through the real sampler, with the Vehtari gate.

    Oracle: stars drawn star by star from the closed-form toy family (``_toy_stars``), never
    through the likelihood. Truth is off the isochrone nodes and off the prior centre.
    Tolerances are **absolute and tied to the truth**, not to the posterior width, and the
    posterior is checked to be informative first (tests/AGENTS.md, failure modes 2 and 4).

    History: a strict xfail from 2026-09-22 until the unbinned likelihood replaced the
    precomputed Hess grid. With the grid, NUTS converged here to dm 10.0009, A_V 0.6003,
    met 0.0151 -- the grid's reference (10.0, 0.6) and a node (0.015) -- against this truth.
    With the unbinned likelihood and the original toy family it failed too, for a reason in
    the oracle: met was not identifiable there (see ``_toy_photometry``); the posterior sat on
    the flat ridge at met 0.0157, A_V 0.582, ESS(met) 252. Tolerances unchanged throughout.
    """
    f = _toy_fitter(tmp_path)
    # Off every node (met between 0.010 and 0.015, loga between 6.5 and 6.6) and off the
    # prior centre (dm_mu = 10.0, mean(Av_range) = 0.6), so a posterior stuck on either
    # cannot pass.
    truth = {"met": 0.0125, "loga": 6.55, "dm": 10.25, "Av": 0.7}
    f.setup(
        _toy_stars(
            500, truth["loga"], truth["met"], truth["dm"], truth["Av"], np.random.default_rng(11)
        ),
        prob_threshold=0,
    )
    # numpyro (in the bayes extra): PyMC's own NUTS took 3 h 25 min here on a loaded machine
    idata = f.fit(
        draws=1000,
        tune=1000,
        chains=2,
        cores=1,
        random_seed=11,
        progressbar=False,
        nuts_sampler="numpyro",
    )
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
