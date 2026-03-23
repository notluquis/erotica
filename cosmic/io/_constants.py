"""Shared constants for the COSMIC data loading utilities."""

from __future__ import annotations

import astropy.units as u

PHOTOMETRIC_SYSTEMS = {
    "Gaia": [
        "source_id",
        "ra",
        "dec",
        "l",
        "b",
        "pm",
        "pmra",
        "pmdec",
        "parallax",
        "pmra_error",
        "pmdec_error",
        "parallax_error",
        "Gmag",
        "G_BPmag",
        "G_RPmag",
        "e_Gmag",
        "e_G_BPmag",
        "e_G_RPmag",
        "e_BP_RP",
    ],
    "TMASS": [
        "tmass_designation",
        "j_m",
        "h_m",
        "ks_m",
        "j_msigcom",
        "h_msigcom",
        "ks_msigcom",
    ],
    "WISE": [
        "allwise_oid",
        "w1mpro",
        "w2mpro",
        "w3mpro",
        "w4mpro",
        "w1mpro_error",
        "w2mpro_error",
        "w3mpro_error",
        "w4mpro_error",
    ],
}

GAIA_DISTANCE_COLUMNS = {
    "geometric": ["r_med_geo", "r_lo_geo", "r_hi_geo"],
    "photogeometric": ["r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo"],
}

ALIASES = {
    "Gmag": {"Gmag", "phot_g_mean_mag"},
    "G_BPmag": {"G_BPmag", "phot_bp_mean_mag"},
    "G_RPmag": {"G_RPmag", "phot_rp_mean_mag"},
    "e_Gmag": {"e_Gmag", "phot_g_mean_mag_error", "ephot_g_mean_mag"},
    "e_G_BPmag": {"e_G_BPmag", "phot_bp_mean_mag_error", "ephot_bp_mean_mag"},
    "e_G_RPmag": {"e_G_RPmag", "phot_rp_mean_mag_error", "ephot_rp_mean_mag"},
    "e_BP_RP": {"e_BP_RP", "phot_bp_rp_excess_error"},
    "parallax": {"parallax"},
    "parallax_error": {"parallax_error"},
    "pm": {"pm"},
    "pmra": {"pmra"},
    "pmra_error": {"pmra_error"},
    "pmdec": {"pmdec"},
    "pmdec_error": {"pmdec_error"},
    "source_id": {"source_id"},
    "ra": {"ra"},
    "dec": {"dec"},
    "l": {"l"},
    "b": {"b"},
    "tmass_designation": {"tmass_designation"},
    "j_m": {"j_m"},
    "h_m": {"h_m"},
    "ks_m": {"ks_m"},
    "j_msigcom": {"j_msigcom"},
    "h_msigcom": {"h_msigcom"},
    "ks_msigcom": {"ks_msigcom"},
    "nu_eff_used_in_astrometry": {"nu_eff_used_in_astrometry"},
    "pseudocolour": {"pseudocolour"},
    "ecl_lat": {"ecl_lat"},
    "astrometric_params_solved": {"astrometric_params_solved"},
    "r_med_geo": {"r_med_geo"},
    "r_lo_geo": {"r_lo_geo"},
    "r_hi_geo": {"r_hi_geo"},
    "r_med_photogeo": {"r_med_photogeo"},
    "r_lo_photogeo": {"r_lo_photogeo"},
    "r_hi_photogeo": {"r_hi_photogeo"},
    "cluster": {"cluster"},
    "probability_hdbscan": {"probability_hdbscan"},
    "av_sagitta": {"av_sagitta"},
    "pms_sagitta": {"pms_sagitta"},
    "age": {"age"},
    "phot_g_mean_flux": {"phot_g_mean_flux"},
    "phot_g_mean_flux_error": {"phot_g_mean_flux_error"},
    "phot_bp_mean_flux": {"phot_bp_mean_flux"},
    "phot_bp_mean_flux_error": {"phot_bp_mean_flux_error"},
    "phot_rp_mean_flux": {"phot_rp_mean_flux"},
    "phot_rp_mean_flux_error": {"phot_rp_mean_flux_error"},
}

ZP_COLUMNS = [
    "nu_eff_used_in_astrometry",
    "pseudocolour",
    "ecl_lat",
    "astrometric_params_solved",
]

FLUX_ERROR_COLUMNS = [
    "phot_g_mean_flux",
    "phot_g_mean_flux_error",
    "phot_bp_mean_flux",
    "phot_bp_mean_flux_error",
    "phot_rp_mean_flux",
    "phot_rp_mean_flux_error",
]

GAIA_PHOTOMETRY_COLUMNS = ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]
PROPER_MOTION_COLUMNS = ["pm", "pmra", "pmdec"]
TMASS_PHOTOMETRY_COLUMNS = ["j_m", "h_m", "ks_m"]
WISE_PHOTOMETRY_COLUMNS = ["w1mpro", "w2mpro", "w3mpro", "w4mpro"]

UNIT_CORRECTIONS = {
    "'electron'.s**-1": u.electron / u.s,
    "log(cm.s**-2)": u.dex(u.cm / u.s**2),
    "'dex'": u.dex,
}

__all__ = [
    "PHOTOMETRIC_SYSTEMS",
    "GAIA_DISTANCE_COLUMNS",
    "ALIASES",
    "ZP_COLUMNS",
    "FLUX_ERROR_COLUMNS",
    "GAIA_PHOTOMETRY_COLUMNS",
    "PROPER_MOTION_COLUMNS",
    "TMASS_PHOTOMETRY_COLUMNS",
    "WISE_PHOTOMETRY_COLUMNS",
    "UNIT_CORRECTIONS",
]
