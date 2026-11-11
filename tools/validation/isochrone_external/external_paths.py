"""Inputs and per-star outputs of this directory that are not versioned here.

Per-star tables of the cluster members are not published from this repository (they go with the
paper that uses them), so the scripts write them next to themselves, where `.gitignore` keeps them
out of git, and read them from there unless an environment variable points elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
#: Working-checkout data, not versioned (same variable as ``isochrone_colour/common.py``).
NGC = Path(os.environ.get("EROTICA_NGC6383_DATA", REPO / "data/test/NGC6383")).expanduser()
#: The 254-member C1 sample.
SAMPLE = NGC / "comments_paper/radius_robustness/generated/40/paperfaithful_reference_p06.ecsv"
#: Per-star residuals of the model screen, written by ``isochrone_model_screen`` (``*.npz``, ignored).
AUDIT = REPO / "tools/validation/isochrone_model_screen/audit_perstar.npz"
#: Downloaded dust cubes.
DUST = Path(os.environ.get("EROTICA_DUST_CACHE", Path.home() / ".cache/erotica-dust")).expanduser()
#: A_V per member and map, written by ``av_maps.py``.
AV_PERSTAR = Path(os.environ.get("AV_MAPS_PERSTAR", HERE / "av_maps_perstar.ecsv")).expanduser()
