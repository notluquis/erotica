"""ASteCA 0.6+ adapter helpers.

This module intentionally imports ASteCA lazily inside functions so importing
``erotica.analysis`` remains cheap and does not require optional dependencies.
"""

from __future__ import annotations

import numpy as np


def _require_asteca():
    try:
        import asteca
    except ImportError as exc:
        raise ImportError(
            "ASteCA is required for this adapter. Install the 'asteca' extra."
        ) from exc
    return asteca


def instantiate_isochrones(*args, **kwargs):
    """Instantiate ``asteca.Isochrones`` with an explicit optional-dependency error."""
    asteca = _require_asteca()
    return asteca.Isochrones(*args, **kwargs)


def instantiate_synthetic(*args, **kwargs):
    """Instantiate ``asteca.Synthetic`` with an explicit optional-dependency error."""
    asteca = _require_asteca()
    return asteca.Synthetic(*args, **kwargs)


def get_isochrone(synthcl, params, *, color_idx=0):
    """Return an official ASteCA 0.6+ isochrone tuple."""
    return synthcl.get_isochrone(params, color_idx=color_idx)


def generate_isochrone_for_plot(synthcl, params, *, return_masses=False):
    """Generate an isochrone for plotting without using legacy ``plot_flag`` APIs."""
    if not return_masses:
        return synthcl.get_isochrone(params)
    try:
        from asteca.modules.synth_cluster_priv import generate_synth_arr
    except ImportError as exc:
        raise ImportError("ASteCA private synth generator is unavailable in this version.") from exc
    return generate_synth_arr(
        synthcl.model,
        synthcl.isochs,
        synthcl.ext_law,
        synthcl.max_mass,
        synthcl.gamma,
        np.random.default_rng(),
        synthcl.met_age_dict,
        params,
        return_flag="isoch",
    )


__all__ = [
    "generate_isochrone_for_plot",
    "get_isochrone",
    "instantiate_isochrones",
    "instantiate_synthetic",
]
