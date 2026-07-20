"""Bayesian inference helpers with lazy optional dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy import units as u

from .kinematics import projected_velocity_values, radial_velocity_values
from .units import quantity_values


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling knobs shared by PyMC models."""

    draws: int = 2_000
    tune: int = 1_000
    target_accept: float = 0.9
    chains: int | None = None
    random_seed: int | None = None
    nuts_sampler: str = "blackjax"
    progressbar: bool = False
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DistanceFitResult:
    mu_r_mean: float
    std_r_mean: float
    mu_r_std: float
    std_r_std: float
    trace: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParallaxFitResult:
    mu_parallax_mean: float
    sigma_parallax_mean: float
    mu_parallax_std: float
    sigma_parallax_std: float
    trace: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProperMotionFitResult:
    results: dict[str, float]
    trace: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VelocityFitResult:
    mu_v_mean: float
    std_v_mean: float
    mu_v_std: float
    std_v_std: float
    trace: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _require_pymc():
    try:
        import pymc as pm
    except ImportError as exc:
        raise ImportError(
            "PyMC is required for cosmic.analysis.inference. Install the 'bayes' extra."
        ) from exc
    return pm


def _sample(pm, model, config: SamplingConfig):
    kwargs = dict(
        draws=config.draws,
        tune=config.tune,
        target_accept=config.target_accept,
        progressbar=config.progressbar,
        random_seed=config.random_seed,
        nuts_sampler=config.nuts_sampler,
        **config.extra_kwargs,
    )
    if config.chains is not None:
        kwargs["chains"] = config.chains
    with model:
        return pm.sample(**kwargs)


def distance_model(
    data,
    *,
    distance_column: str = "r_med_geo",
    parallax_column: str = "parallax",
    prior_type: str = "uniform",
    return_trace: bool = False,
    sampling: SamplingConfig | None = None,
) -> DistanceFitResult:
    """Fit the hierarchical distance model used by the legacy workflow."""
    pm = _require_pymc()
    sampling = sampling or SamplingConfig()
    distances = quantity_values(data[distance_column], u.kpc)
    parallax_values = quantity_values(data[parallax_column], u.mas)
    prior_mu_r = float(np.nanmean([np.nanmean(1 / parallax_values), np.nanmean(distances)]))
    with pm.Model() as model:
        if prior_type == "uniform":
            mu_r = pm.Uniform("mu_r", lower=0.5 * prior_mu_r, upper=1.5 * prior_mu_r)
        elif prior_type == "normal":
            mu_r = pm.TruncatedNormal(
                "mu_r", lower=0.5 * prior_mu_r, upper=1.5 * prior_mu_r, mu=prior_mu_r, sigma=1
            )
        else:
            raise ValueError("prior_type must be 'uniform' or 'normal'.")
        std_r = pm.HalfNormal("std_r", sigma=float(np.nanstd(distances)))
        pm.Gamma("r", mu=mu_r, sigma=std_r, observed=distances)
    trace = _sample(pm, model, sampling)
    return DistanceFitResult(
        mu_r_mean=float(trace.posterior["mu_r"].mean()),
        std_r_mean=float(trace.posterior["std_r"].mean()),
        mu_r_std=float(trace.posterior["mu_r"].std()),
        std_r_std=float(trace.posterior["std_r"].std()),
        trace=trace if return_trace else None,
        metadata={"backend": sampling.nuts_sampler, "variables": ["mu_r", "std_r"]},
    )


def fit_parallax_model(
    data,
    *,
    parallax_column: str = "parallax",
    prior_distance=None,
    prior_type: str = "uniform",
    return_trace: bool = False,
    sampling: SamplingConfig | None = None,
) -> ParallaxFitResult:
    """Fit a Gaussian cluster parallax model."""
    pm = _require_pymc()
    sampling = sampling or SamplingConfig()
    parallax_values = quantity_values(data[parallax_column], u.mas)
    if prior_distance is not None:
        prior_parallax = prior_distance.to(u.mas, equivalencies=u.parallax()).value if hasattr(prior_distance, "to") else 1 / float(prior_distance)
    else:
        prior_parallax = float(np.nanmean(parallax_values))
    with pm.Model() as model:
        if prior_type == "uniform":
            mu_parallax = pm.Uniform(
                "mu_parallax", lower=0.5 * prior_parallax, upper=1.5 * prior_parallax
            )
        elif prior_type == "normal":
            mu_parallax = pm.TruncatedNormal(
                "mu_parallax",
                lower=0.5 * prior_parallax,
                upper=1.5 * prior_parallax,
                sigma=1,
                mu=prior_parallax,
            )
        else:
            raise ValueError("prior_type must be 'uniform' or 'normal'.")
        sigma_parallax = pm.HalfNormal("sigma_parallax", sigma=float(np.nanstd(parallax_values)))
        pm.Normal("observed_parallax", mu=mu_parallax, sigma=sigma_parallax, observed=parallax_values)
    trace = _sample(pm, model, sampling)
    return ParallaxFitResult(
        mu_parallax_mean=float(trace.posterior["mu_parallax"].mean()),
        sigma_parallax_mean=float(trace.posterior["sigma_parallax"].mean()),
        mu_parallax_std=float(trace.posterior["mu_parallax"].std()),
        sigma_parallax_std=float(trace.posterior["sigma_parallax"].std()),
        trace=trace if return_trace else None,
        metadata={"backend": sampling.nuts_sampler, "variables": ["mu_parallax", "sigma_parallax"]},
    )


def proper_motion_2d_gaussian(
    pm_ra,
    pm_dec,
    *,
    return_trace: bool = False,
    sampling: SamplingConfig | None = None,
) -> ProperMotionFitResult:
    """Fit a correlated 2D Gaussian proper-motion model."""
    pm = _require_pymc()
    sampling = sampling or SamplingConfig()
    pm_ra_values = quantity_values(pm_ra, u.mas / u.yr)
    pm_dec_values = quantity_values(pm_dec, u.mas / u.yr)
    with pm.Model() as model:
        mu_ra = pm.Normal("mu_RA", mu=float(np.nanmedian(pm_ra_values)), sigma=float(np.nanstd(pm_ra_values)))
        mu_dec = pm.Normal("mu_Dec", mu=float(np.nanmedian(pm_dec_values)), sigma=float(np.nanstd(pm_dec_values)))
        sigma_ra = pm.HalfNormal("sigma_RA", sigma=float(np.nanstd(pm_ra_values)))
        sigma_dec = pm.HalfNormal("sigma_Dec", sigma=float(np.nanstd(pm_dec_values)))
        corr = pm.Uniform("corr", lower=-1, upper=1)
        cov = pm.math.stack(
            [[sigma_ra**2, corr * sigma_ra * sigma_dec], [corr * sigma_ra * sigma_dec, sigma_dec**2]]
        )
        pm.MvNormal("obs", mu=pm.math.stack([mu_ra, mu_dec]), cov=cov, observed=np.stack([pm_ra_values, pm_dec_values], axis=1))
    trace = _sample(pm, model, sampling)
    results = {
        "mu_RA_mean": float(trace.posterior["mu_RA"].mean()),
        "mu_Dec_mean": float(trace.posterior["mu_Dec"].mean()),
        "sigma_RA_mean": float(trace.posterior["sigma_RA"].mean()),
        "sigma_Dec_mean": float(trace.posterior["sigma_Dec"].mean()),
        "corr_mean": float(trace.posterior["corr"].mean()),
        "mu_RA_std": float(trace.posterior["mu_RA"].std()),
        "mu_Dec_std": float(trace.posterior["mu_Dec"].std()),
        "sigma_RA_std": float(trace.posterior["sigma_RA"].std()),
        "sigma_Dec_std": float(trace.posterior["sigma_Dec"].std()),
        "corr_std": float(trace.posterior["corr"].std()),
    }
    return ProperMotionFitResult(
        results=results,
        trace=trace if return_trace else None,
        metadata={"backend": sampling.nuts_sampler, "variables": list(results)},
    )


def velocity_model(
    values,
    *,
    distance=None,
    return_trace: bool = False,
    sampling: SamplingConfig | None = None,
) -> VelocityFitResult:
    """Fit a Gaussian velocity model."""
    pm = _require_pymc()
    sampling = sampling or SamplingConfig()
    if hasattr(values, "colnames"):
        if "projected_velocity" not in values.colnames:
            raise ValueError("Table input must contain a 'projected_velocity' column.")
        values = values["projected_velocity"]
    velocity_values, _, _ = projected_velocity_values(values, distance=distance)
    with pm.Model() as model:
        mu_v = pm.Normal("mu_v", mu=float(np.nanmean(velocity_values)), sigma=10)
        std_v = pm.Uniform("std_v", lower=0, upper=40)
        pm.Normal("observed_velocity", mu=mu_v, sigma=std_v, observed=velocity_values)
    trace = _sample(pm, model, sampling)
    return VelocityFitResult(
        mu_v_mean=float(trace.posterior["mu_v"].mean()),
        std_v_mean=float(trace.posterior["std_v"].mean()),
        mu_v_std=float(trace.posterior["mu_v"].std()),
        std_v_std=float(trace.posterior["std_v"].std()),
        trace=trace if return_trace else None,
        metadata={"backend": sampling.nuts_sampler, "variables": ["mu_v", "std_v"]},
    )


def radial_velocity_model(
    radial_velocity,
    *,
    return_trace: bool = False,
    sampling: SamplingConfig | None = None,
) -> VelocityFitResult:
    """Fit a Gaussian radial-velocity model."""
    if hasattr(radial_velocity, "colnames"):
        if "radial_velocity" not in radial_velocity.colnames:
            raise ValueError("Table input must contain a 'radial_velocity' column.")
        radial_velocity = radial_velocity["radial_velocity"]
    return velocity_model(
        radial_velocity_values(radial_velocity) * u.km / u.s,
        return_trace=return_trace,
        sampling=sampling,
    )


def _dataclass_result_to_dict(result) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in result.__dict__.items()
        if key not in {"trace", "metadata"} and value is not None
    }
    if result.trace is not None:
        payload["trace"] = result.trace
    return payload


class ClusterInferenceAnalyzer:
    """Probability-thresholded Bayesian summaries for a cluster source table."""

    def __init__(
        self,
        data,
        *,
        probability_column: str = "probability",
        sampling: SamplingConfig | None = None,
    ) -> None:
        self.data = data
        self.probability_column = probability_column
        self.sampling = sampling or SamplingConfig()

    def select(self, probability_threshold: float | None = None):
        """Return data above an optional probability threshold."""
        if probability_threshold is None:
            return self.data
        return self.data[self.data[self.probability_column] >= probability_threshold]

    @staticmethod
    def _normalise_thresholds(values) -> np.ndarray:
        thresholds = np.asarray(values, dtype=float)
        return np.where(thresholds > 1, thresholds / 100, thresholds)

    def distance_and_parallax_by_probability(
        self,
        probability_thresholds=(0.5, 0.6, 0.7, 0.8),
        *,
        return_trace: bool = False,
        progressbar: bool | None = None,
        distance_prior=None,
        fractional_parallax_error_max: float = 0.1,
        distance_column: str = "r_med_geo",
        parallax_column: str = "parallax",
        parallax_error_column: str = "parallax_error",
    ) -> dict[str, list[Any]]:
        """Fit distance and parallax models for each probability threshold."""
        thresholds = self._normalise_thresholds(probability_thresholds)
        sampling = self.sampling
        if progressbar is not None:
            sampling = SamplingConfig(**{**sampling.__dict__, "progressbar": progressbar})
        results: dict[str, list[Any]] = {
            "mu_r_mean": [],
            "std_r_mean": [],
            "mu_r_std": [],
            "std_r_std": [],
            "mu_parallax_mean": [],
            "sigma_parallax_mean": [],
            "mu_parallax_std": [],
            "sigma_parallax_std": [],
        }
        traces = []
        for threshold in thresholds:
            subset = self.select(float(threshold))
            parallax = quantity_values(subset[parallax_column], u.mas)
            parallax_error = quantity_values(subset[parallax_error_column], u.mas)
            # Only positive parallaxes are physical distance estimators. Requiring
            # parallax > 0 (not just abs(error/parallax)) excludes negative parallaxes,
            # which would otherwise pass the ratio cut, and guards against div-by-zero.
            positive_parallax = parallax > 0
            fractional_error = np.divide(
                parallax_error,
                parallax,
                out=np.full_like(parallax, np.inf),
                where=positive_parallax,
            )
            useful = subset[positive_parallax & (fractional_error <= fractional_parallax_error_max)]
            distance_result = distance_model(
                useful,
                distance_column=distance_column,
                parallax_column=parallax_column,
                return_trace=return_trace,
                sampling=sampling,
            )
            parallax_result = fit_parallax_model(
                useful,
                parallax_column=parallax_column,
                prior_distance=distance_prior,
                return_trace=return_trace,
                sampling=sampling,
            )
            distance_payload = _dataclass_result_to_dict(distance_result)
            parallax_payload = _dataclass_result_to_dict(parallax_result)
            for key in ("mu_r_mean", "std_r_mean", "mu_r_std", "std_r_std"):
                results[key].append(distance_payload[key])
            for key in (
                "mu_parallax_mean",
                "sigma_parallax_mean",
                "mu_parallax_std",
                "sigma_parallax_std",
            ):
                results[key].append(parallax_payload[key])
            if return_trace:
                traces.extend([distance_result.trace, parallax_result.trace])
        if return_trace:
            results["traces"] = traces
        return results

    def proper_motion_by_probability(
        self,
        probability_thresholds=(0.5, 0.6, 0.7, 0.8),
        *,
        pmra_column: str = "pmra",
        pmdec_column: str = "pmdec",
        return_trace: bool = False,
        return_pmdist: bool = False,
        return_pmprob: bool = False,
        progressbar: bool | None = None,
    ):
        """Fit proper-motion Gaussian models for each probability threshold."""
        thresholds = self._normalise_thresholds(probability_thresholds)
        sampling = self.sampling
        if progressbar is not None:
            sampling = SamplingConfig(**{**sampling.__dict__, "progressbar": progressbar})
        stats = []
        distances = []
        pm_probabilities = []
        for threshold in thresholds:
            subset = self.select(float(threshold))
            fit = proper_motion_2d_gaussian(
                subset[pmra_column],
                subset[pmdec_column],
                return_trace=return_trace,
                sampling=sampling,
            )
            row = {"probability": float(threshold), **fit.results}
            if return_trace:
                row["trace"] = fit.trace
            stats.append(row)
            if return_pmdist:
                center_distance = np.sqrt(
                    (subset[pmra_column] - fit.results["mu_RA_mean"] * u.mas / u.yr) ** 2
                    + (subset[pmdec_column] - fit.results["mu_Dec_mean"] * u.mas / u.yr) ** 2
                )
                payload = {"probability": float(threshold), "distancepm": 1 / center_distance}
                distances.append(payload)
                if return_pmprob:
                    pm_probabilities.append((1 / center_distance) * subset[self.probability_column])
        if return_pmdist and return_pmprob:
            return stats, distances, pm_probabilities
        if return_pmdist:
            return stats, distances
        return stats

    def projected_velocity_by_probability(
        self,
        probability_thresholds=(0.5, 0.6, 0.7, 0.8),
        *,
        distance=None,
        return_trace: bool = False,
        progressbar: bool | None = None,
    ):
        """Fit projected-velocity models for each probability threshold."""
        thresholds = self._normalise_thresholds(probability_thresholds)
        sampling = self.sampling
        if progressbar is not None:
            sampling = SamplingConfig(**{**sampling.__dict__, "progressbar": progressbar})
        rows = []
        for threshold in thresholds:
            subset = self.select(float(threshold))
            result = velocity_model(
                subset,
                distance=distance,
                return_trace=return_trace,
                sampling=sampling,
            )
            rows.append({"probability": float(threshold), "model_results": _dataclass_result_to_dict(result)})
        return rows

    def radial_velocity_by_probability(
        self,
        probability_thresholds=(0.5, 0.6, 0.7, 0.8),
        *,
        return_trace: bool = False,
        progressbar: bool | None = None,
    ):
        """Fit radial-velocity models for each probability threshold."""
        thresholds = self._normalise_thresholds(probability_thresholds)
        sampling = self.sampling
        if progressbar is not None:
            sampling = SamplingConfig(**{**sampling.__dict__, "progressbar": progressbar})
        finite = self.data[np.isfinite(quantity_values(self.data["radial_velocity"], u.km / u.s))]
        analyzer = ClusterInferenceAnalyzer(
            finite,
            probability_column=self.probability_column,
            sampling=sampling,
        )
        rows = []
        for threshold in thresholds:
            subset = analyzer.select(float(threshold))
            result = radial_velocity_model(subset, return_trace=return_trace, sampling=sampling)
            rows.append(
                {
                    "probability": float(threshold),
                    "model_results": _dataclass_result_to_dict(result),
                    "len_data": len(subset),
                }
            )
        return rows


def FitProperMotion2DGaussian(pm_RA, pm_Dec, return_trace=False, progressbar=False):
    """Legacy-compatible alias for :func:`proper_motion_2d_gaussian`."""
    fit = proper_motion_2d_gaussian(
        pm_RA,
        pm_Dec,
        return_trace=return_trace,
        sampling=SamplingConfig(progressbar=progressbar),
    )
    payload = {"results": fit.results}
    if return_trace:
        payload["trace"] = fit.trace
    return payload


def parallax_determination(data, prob_thresholds=(50, 60, 70, 80), return_trace=False, progressbar=False, **kwargs):
    """Legacy-compatible non-plotting parallax/distance threshold summary."""
    kwargs.pop("savefig", None)
    kwargs.pop("paper_single", None)
    kwargs.pop("parallax_hist", None)
    distance_prior = kwargs.pop("distance_prior", None)
    analyzer = ClusterInferenceAnalyzer(data, sampling=SamplingConfig(progressbar=progressbar))
    return analyzer.distance_and_parallax_by_probability(
        prob_thresholds,
        return_trace=return_trace,
        progressbar=progressbar,
        distance_prior=distance_prior,
        **kwargs,
    )


def pm_determination(data, savefig=None, prob_number=(50, 60, 70, 80), progressbar=False, **kwargs):
    """Legacy-compatible non-plotting proper-motion threshold summary."""
    del savefig
    kwargs.pop("paper_single", None)
    analyzer = ClusterInferenceAnalyzer(data, sampling=SamplingConfig(progressbar=progressbar))
    return analyzer.proper_motion_by_probability(prob_number, progressbar=progressbar, **kwargs)


def velocity_determination(
    data,
    prob_thresholds=(50, 60, 70, 80),
    return_trace=False,
    progressbar=False,
    savefig=None,
    paper_single=False,
    distance=None,
):
    """Legacy-compatible non-plotting projected-velocity threshold summary."""
    del savefig, paper_single
    analyzer = ClusterInferenceAnalyzer(data, sampling=SamplingConfig(progressbar=progressbar))
    return analyzer.projected_velocity_by_probability(
        prob_thresholds,
        distance=distance,
        return_trace=return_trace,
        progressbar=progressbar,
    )


def rv_determination(data, prob_thresholds=(50, 60, 70, 80), return_trace=False, progressbar=False):
    """Legacy-compatible radial-velocity threshold summary."""
    analyzer = ClusterInferenceAnalyzer(data, sampling=SamplingConfig(progressbar=progressbar))
    return analyzer.radial_velocity_by_probability(
        prob_thresholds,
        return_trace=return_trace,
        progressbar=progressbar,
    )


__all__ = [
    "ClusterInferenceAnalyzer",
    "DistanceFitResult",
    "FitProperMotion2DGaussian",
    "ParallaxFitResult",
    "ProperMotionFitResult",
    "SamplingConfig",
    "VelocityFitResult",
    "distance_model",
    "fit_parallax_model",
    "parallax_determination",
    "pm_determination",
    "proper_motion_2d_gaussian",
    "radial_velocity_model",
    "rv_determination",
    "velocity_determination",
    "velocity_model",
]
