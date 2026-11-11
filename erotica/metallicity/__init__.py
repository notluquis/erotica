"""Metallicity corrections for spectroscopic pipelines.

:mod:`erotica.metallicity.pms_bias` corrects the [Fe/H] deficit of ASPCAP (APOGEE DR17/DR19) in
young late-G/K stars. Read its module docstring before use: the model is empirical and its
regressor, ASPCAP's log g, is itself part of the symptom. The default calibration is the
hierarchical posterior shipped with the package (:func:`load_posterior`); the OLS fit stays as
:data:`CALIBRATED_MODEL`.
"""

from .pms_bias import (
    CALIBRATED_MODEL,
    BiasModel,
    BiasPrediction,
    HierarchicalPosterior,
    PMSBiasDomainWarning,
    fit_bias_model,
    fit_hierarchical_model,
    leave_one_cluster_out,
    load_posterior,
    predict_bias,
    predictive_draws,
    select_calibration_sample,
)

__all__ = [
    "CALIBRATED_MODEL",
    "BiasModel",
    "BiasPrediction",
    "HierarchicalPosterior",
    "PMSBiasDomainWarning",
    "fit_bias_model",
    "fit_hierarchical_model",
    "leave_one_cluster_out",
    "load_posterior",
    "predict_bias",
    "predictive_draws",
    "select_calibration_sample",
]
