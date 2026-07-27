"""Matplotlib styling defaults for EROTICA clustering plots."""

from __future__ import annotations

import matplotlib as mpl

DEFAULT_RC_PARAMS = {
    "figure.figsize": (8, 6),
    "figure.dpi": 100,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.color": "0.8",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
    "legend.fontsize": 10,
}


def apply_default_style(rc_params: dict | None = None) -> None:
    """Update :mod:`matplotlib` rcParams with EROTICA defaults."""
    params = dict(DEFAULT_RC_PARAMS)
    if rc_params:
        params.update(rc_params)
    mpl.rcParams.update(params)


__all__ = ["DEFAULT_RC_PARAMS", "apply_default_style"]
