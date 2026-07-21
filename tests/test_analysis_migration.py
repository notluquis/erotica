from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable


class _FakeVar:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape


class _FakePosterior(dict):
    @property
    def data_vars(self):
        return self.keys()


class _FakeTrace:
    def __init__(self, **posterior):
        self.posterior = _FakePosterior({key: _FakeVar(value) for key, value in posterior.items()})


def test_units_and_projected_velocity_values():
    from pumps.analysis.kinematics import projected_velocity_values
    from pumps.analysis.units import angular_size, linear_size

    angle = angular_size(1 * u.pc, 1 * u.kpc).to(u.arcmin)
    size = linear_size(angle, 1 * u.kpc)
    assert np.isclose(size.to_value(u.pc), 1.0)

    values, unit_label, axis_label = projected_velocity_values([1.0, 2.0] * u.mas / u.yr, 1 * u.kpc)
    np.testing.assert_allclose(values, [4.74047, 9.48094])
    assert "km" in unit_label
    assert axis_label == "Projected velocity"


def test_half_mass_and_half_light_radius():
    from pumps.analysis.structure import calculate_half_light_radius, half_mass_radius

    table = QTable(
        {
            "ra": [0.0, 0.0, 0.01, 0.02] * u.deg,
            "dec": [0.0, 0.01, 0.0, 0.0] * u.deg,
            "mass": [1.0, 1.0, 2.0, 4.0] * u.Msun,
            "mass_std": [0.1, 0.1, 0.2, 0.4] * u.Msun,
            "Gmag": [15.0, 14.5, 14.0, 13.0] * u.mag,
        }
    )
    radius, radius_err = half_mass_radius(table, (0.0, 0.0))
    assert radius.unit == u.arcmin
    assert radius_err.unit == u.arcmin
    assert radius.value >= 0

    half_light = calculate_half_light_radius(table, (0.0, 0.0))
    assert half_light.unit == u.arcmin


def test_distance_comparison_and_trace_extraction():
    from pumps.analysis.debugging import diagnose_distance_comparison, extract_distance_samples

    rng = np.random.default_rng(123)
    trace = _FakeTrace(mu_r=rng.normal(1.1, 0.01, size=(2, 100)))
    samples = extract_distance_samples({"traces": [trace]})
    assert samples.shape == (200,)

    comparison = diagnose_distance_comparison(samples, rng.normal(1.15, 0.07, size=200))
    assert comparison.delta.unit == u.kpc
    assert comparison.sigma_combined.value > 0
    assert np.isfinite(comparison.delta_in_sigma_combined)


def test_distance_pathway_plot_smoke(tmp_path):
    from pumps.analysis.figures import plot_distance_pathway_overlap

    rng = np.random.default_rng(456)
    trace = _FakeTrace(mu_r=rng.normal(1.1, 0.01, size=(2, 100)))
    asteca = _FakeTrace(dm=rng.normal(10.25, 0.12, size=(2, 100)))
    out = tmp_path / "distance_pathway_overlap.pdf"
    comparison = plot_distance_pathway_overlap(
        parallax_results={"traces": [trace]},
        asteca_trace=asteca,
        savefig=out,
        show=False,
    )
    assert out.exists()
    assert comparison.sigma_combined.value > 0


def test_optional_external_asteca_error_when_missing(monkeypatch):
    import pumps.analysis.external.asteca as asteca_adapter

    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "asteca":
            raise ImportError("blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    with pytest.raises(ImportError, match="ASteCA"):
        asteca_adapter.instantiate_isochrones()


def _analysis_table(n=24):
    rng = np.random.default_rng(1234)
    return QTable(
        {
            "ra": rng.normal(263.7, 0.01, n) * u.deg,
            "dec": rng.normal(-32.58, 0.01, n) * u.deg,
            "ra_error": np.full(n, 0.0001) * u.deg,
            "dec_error": np.full(n, 0.0001) * u.deg,
            "pmra": rng.normal(2.5, 0.1, n) * u.mas / u.yr,
            "pmdec": rng.normal(-1.0, 0.1, n) * u.mas / u.yr,
            "projected_velocity": rng.normal(1.0, 0.1, n) * u.mas / u.yr,
            "radial_velocity": rng.normal(0.0, 1.0, n) * u.km / u.s,
            "probability": np.linspace(0.45, 0.95, n),
            "mass": np.linspace(0.5, 2.0, n) * u.Msun,
            "mass_std": np.full(n, 0.1) * u.Msun,
            "Gmag": np.linspace(10.0, 17.0, n) * u.mag,
            "e_Gmag": np.full(n, 0.01) * u.mag,
        }
    )


def test_structure_analyzer_class_facade():
    from pumps.analysis import ClusterStructureAnalyzer, RadialDensityProfile

    table = _analysis_table()
    analyzer = ClusterStructureAnalyzer(table)
    selected = analyzer.select(0.6)
    assert len(selected) < len(table)

    center = analyzer.center(0.6, bandwidths=np.linspace(0.001, 0.01, 4))
    assert center.ra.unit == u.deg

    profile = analyzer.radial_density_profile((center.ra, center.dec), probability_threshold=0.6)
    assert isinstance(profile, RadialDensityProfile)
    assert profile.radius.unit == u.arcmin
    assert profile.counts.sum() <= len(selected)


def test_dynamics_analyzer_class_facade():
    from pumps.analysis import ClusterDynamicsAnalyzer

    table = _analysis_table()
    analyzer = ClusterDynamicsAnalyzer(table, distance=1.1 * u.kpc, center=(263.7 * u.deg, -32.58 * u.deg))
    mass = analyzer.cluster_mass()
    assert mass.unit == u.Msun
    galactocentric_distance, galactocentric_error = analyzer.galactocentric_distance()
    assert galactocentric_distance.unit == u.kpc
    assert galactocentric_error.unit == u.kpc
    hill = analyzer.hill_radius(cluster_mass=mass, cluster_mass_err=0.1 * mass, return_linear_size=True)
    assert hill["angular_size"].unit == u.arcmin
    assert hill["linear_size"].unit == u.pc


def test_photometric_mass_estimator_legacy_samples():
    from pumps.analysis import PhotometricMassEstimator, assign_masses

    iso = [
        (
            np.array([10.0, 11.0, 12.0]),
            np.array([0.0, 0.5, 1.0]),
            np.array([np.nan, np.nan, np.nan]),
            np.array([2.0, 1.0, 0.5]),
        )
    ]
    estimator = PhotometricMassEstimator(iso, k=2)
    assigned = estimator.assign_from_samples(
        np.array([10.1, 11.9]),
        np.array([0.1, 0.9]),
        np.array([1, 2]),
    )
    assert assigned["mass"].unit == u.Msun
    direct = assign_masses(iso, np.array([10.1]), np.array([0.1]), np.array([1]), k=1)
    assert direct["mass"][0].unit == u.Msun


def test_figure_builder_cumulative_smoke(tmp_path):
    from pumps.analysis import ClusterFigureBuilder

    table = _analysis_table()
    builder = ClusterFigureBuilder(table)
    out = tmp_path / "cumulative.pdf"
    fig = builder.cumulative([(263.7, -32.58)], prob_number=[60], savefig=out, show=False)
    assert out.exists()
    assert fig is not None


def test_top_level_analysis_exports():
    import pumps
    from pumps import ClusterDynamicsAnalyzer, ClusterInferenceAnalyzer, ClusterStructureAnalyzer
    from pumps.analysis import (
        FitProperMotion2DGaussian,
        parallax_determination,
        pm_determination,
        rv_determination,
        velocity_determination,
    )

    assert pumps.ClusterStructureAnalyzer is ClusterStructureAnalyzer
    assert ClusterDynamicsAnalyzer.__name__ == "ClusterDynamicsAnalyzer"
    assert ClusterInferenceAnalyzer.__name__ == "ClusterInferenceAnalyzer"
    assert callable(FitProperMotion2DGaussian)
    assert callable(parallax_determination)
    assert callable(pm_determination)
    assert callable(velocity_determination)
    assert callable(rv_determination)


def test_public_cosmic_aux_api_is_exported_from_analysis():
    import pumps.analysis as analysis

    legacy_public_names = {
        "ensure_units",
        "quantity_values",
        "angular_size",
        "linear_size",
        "histogram_mode",
        "half_mass_radius",
        "read_isochrones_with_metadata",
        "set_column_types",
        "plot_isochrone",
        "plot_color_color",
        "plot_isochrone_label",
        "plot_errors_bar",
        "plot_iso_mass_curves_across_isochrones",
        "calculate_mode",
        "store_trace_results",
        "load_results",
        "plot_distance_pathway_overlap",
        "assign_masses",
        "plot_hist2d",
        "assign_mass_nearest_isochrone_point_kdtree",
        "distance_model",
        "fit_parallax_model",
        "parallax_determination",
        "pm_determination",
        "FitProperMotion2DGaussian",
        "projected_velocity_values",
        "velocity_model",
        "velocity_determination",
        "center_determination",
        "graph_center_determination",
        "density_annulus_calculator_width",
        "density_annulus_calculator_equip",
        "calculate_galactic_mass",
        "tidal_radius_prior",
        "calculate_absolute_magnitude",
        "estimate_luminosity",
        "estimate_mass_from_luminosity",
        "estimate_cluster_mass",
        "calculate_galactocentric_distance",
        "calculate_hill_radius",
        "grav_bound_radius",
        "calculate_half_light_radius",
        "RDP_bayesian",
        "RDP_bayesian_log_space",
        "king_profile",
        "graph_king",
        "plot_cumulative",
        "plot_cumulative_by_brightness",
        "plot_cumulative_by_mass_and_type",
        "plot_cumulative_by_mass",
        "radial_velocity_model",
        "rv_determination",
        "graph_real",
    }

    exported = set(analysis.__all__)
    assert legacy_public_names <= exported
    for name in legacy_public_names:
        assert hasattr(analysis, name), name
