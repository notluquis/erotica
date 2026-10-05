#!/usr/bin/env python3
"""Member table of NGC 6383 for the multiband isochrone fit: P01's 254 members (Gaia photometry and
errors as the 2D fits use them) joined with the near-infrared magnitudes of the VIRAC2 + 2MASS
member table (hub finding ``virac2-gaia-2mass-calibration.md``; its ``*_best`` columns are in the
2MASS system: VIRAC2 brought to 2MASS with Gonzalez-Fernandez+2018 eqs. 7-9 inverted where VIRAC2 is
above its measured saturation limit, else 2MASS).

WHY A COPY HERE: the source table lives in a private repository; this keeps the fit reproducible
from this repository alone. The input's SHA-256 is recorded in the output's metadata.

MISSING-BAND RULE (declared before any fit, methodology §A.1): a band is missing -- marginalised by
the likelihood -- when ``src_*`` is ``none`` (no usable measurement) or ``sat`` (VIRAC2 saturated and
no 2MASS), and also when ``src_*`` is ``2MASS`` with ``f_blend_2as > 0.1`` (VIRAC2 neighbours inside
the 2MASS beam add more than 10 % of the star's Ks flux: the 2MASS magnitude is biased bright by more
than 0.1 mag). None of these causes depends on the star being fainter than a limit, so marginalising
is the right treatment and no censoring factor is needed.

WHAT WOULD FALSIFY THE JOIN: fewer than 254 matches by source_id, or |G(P01) - G(table)| > 1e-4 mag.

Output: ``ngc6383_multiband_members.ecsv`` + ``build_members.json`` (counts).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.table import Table

HERE = Path(__file__).resolve().parent
NGC = Path("/Users/notluquis/erotica/data/test/NGC6383")
SAMPLE = NGC / "comments_paper/radius_robustness/generated/40/paperfaithful_reference_p06.ecsv"
NIR = Path("/Users/notluquis/phd/agent-findings/scripts/virac2_ngc6383_members_clean.ecsv")
GAIA_COLS = [
    "source_id",
    "ra",
    "dec",
    "Gmag",
    "G_BPmag",
    "G_RPmag",
    "e_Gmag",
    "e_G_BPmag",
    "e_G_RPmag",
    "e_BP_RP",
    "probability_hdbscan",
    "parallax",
    "parallax_error",
]
BANDS = ("J", "H", "Ks")


def col(t, name):
    c = t[name]
    return np.asarray(c.filled(np.nan) if hasattr(c, "filled") else c, dtype=float)


def main() -> None:
    s = Table.read(SAMPLE)
    t = Table.read(NIR)
    sha = hashlib.sha256(NIR.read_bytes()).hexdigest()
    idx = {int(v): i for i, v in enumerate(np.asarray(t["GaiaDR3"]))}
    rows = [idx[int(v)] for v in np.asarray(s["source_id"]) if int(v) in idx]
    if len(rows) != len(s) or len(s) != 254:
        raise SystemExit(f"join: {len(rows)} of {len(s)}")
    n = t[rows]
    dG = np.abs(col(s, "Gmag") - col(n, "G"))
    if np.nanmax(dG) > 1e-4:
        raise SystemExit(f"G disagrees: max {np.nanmax(dG)}")
    out = Table({c: s[c] for c in GAIA_COLS})
    counts = {}
    fb = col(n, "f_blend_2as")
    for b in BANDS:
        m, e = col(n, f"{b}_best"), col(n, f"e{b}_best")
        src = np.asarray(n[f"src_{b}"]).astype(str)
        blended = (src == "2MASS") & (fb > 0.1)
        use = np.isfinite(m) & np.isfinite(e) & (e > 0) & ~blended
        out[f"{b}_fit"] = np.where(use, m, np.nan)
        out[f"e{b}_fit"] = np.where(use, e, np.nan)
        out[f"src_{b}"] = src
        counts[b] = {
            "used": int(use.sum()),
            "virac2": int((use & (src == "VIRAC2->2MASS")).sum()),
            "twomass": int((use & (src == "2MASS")).sum()),
            "missing_none": int((src == "none").sum()),
            "missing_sat": int((src == "sat").sum()),
            "missing_2mass_blend": int(blended.sum()),
        }
    for c in ("f_blend_2as", "var_flag", "sat_J", "sat_H", "sat_Ks", "yso_alpha"):
        out[c] = n[c]
    out.meta["comments"] = [
        f"P01 members ({SAMPLE.name}) + *_best of {NIR.name} (sha256 {sha}); "
        "*_fit = *_best with the missing-band rule of build_members.py applied (NaN = marginalised)."
    ]
    out.write(HERE / "ngc6383_multiband_members.ecsv", overwrite=True)
    res = {"n": len(out), "max_dG": float(np.nanmax(dG)), "source_sha256": sha, "bands": counts}
    (HERE / "build_members.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
