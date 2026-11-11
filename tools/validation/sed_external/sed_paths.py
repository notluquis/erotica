"""Inputs and per-star outputs of this directory that are not versioned here.

The member table and the spectral types live with the analysis that produced them, outside this
repository; their paths come from environment variables, and a missing one exits naming what it
expects. Per-star results are written next to the scripts, where `.gitignore` keeps them out of git.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Environment variable -> the table it must point to.
INPUTS = {
    "NGC6383_VIRAC2_MEMBERS": "virac2_ngc6383_members_clean.ecsv: NGC 6383 members with VIRAC2 and "
    "2MASS photometry (GaiaDR3, Gmag, e_Gmag, ...)",
    "NGC6383_SPECTRA_JSON": "isochrone_external_spectra.json: published spectral types of the members "
    '({"stars": [...]})',
}

#: Per-star SED fits, written by ``run_real.py``.
SED_PERSTAR = Path(os.environ.get("SED_REAL_PERSTAR", HERE / "sed_real_perstar.ecsv")).expanduser()
#: A_V per member and map, written by ``../isochrone_external/av_maps.py``.
AV_PERSTAR = Path(
    os.environ.get("AV_MAPS_PERSTAR", HERE.parent / "isochrone_external" / "av_maps_perstar.ecsv")
).expanduser()
#: Output of ``figure.py``.
FIGURE = Path(os.environ.get("SED_FIGURE", HERE / "isochrone_sed_external.png")).expanduser()


def input_path(env: str) -> Path:
    """Path of an input in ``INPUTS``, read from the environment variable ``env``."""
    v = os.environ.get(env)
    if not v:
        raise SystemExit(f"set {env} to the path of {INPUTS[env]}")
    return Path(v).expanduser()
