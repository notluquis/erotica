"""Metallicity corrections for spectroscopic pipelines.

:mod:`erotica.metallicity.pms_bias` corrects the [Fe/H] deficit of ASPCAP (APOGEE DR17/DR19) in
young late-G/K stars. Read its module docstring before use: the model is empirical and its
regressor, ASPCAP's log g, is itself part of the symptom.
"""

from .pms_bias import (
    CALIBRATED_MODEL,
    BiasModel,
    BiasPrediction,
    PMSBiasDomainWarning,
    fit_bias_model,
    leave_one_cluster_out,
    predict_bias,
    select_calibration_sample,
)

__all__ = [
    "CALIBRATED_MODEL",
    "BiasModel",
    "BiasPrediction",
    "PMSBiasDomainWarning",
    "fit_bias_model",
    "leave_one_cluster_out",
    "predict_bias",
    "select_calibration_sample",
]
