#!/usr/bin/env python3
"""GSP-Phot T_eff against Cao+22 spectroscopic T_eff for the lambda Ori stars: the numbers quoted in
§0.1.3 of the hub finding ``isochrone-colour-mechanism.md`` (bias by T_Cao bin, [M/H], log g,
library), plus the T_gsp-binned calibration that ``hr_fit.py --run R4`` applies.

Why it matters for the HR fit: GSP-Phot's forward model draws T_eff from PARSEC 1.2S isochrones with
log t >= 6.6 (Andrae+2023, 2023A&A...674A..27A, §2.3) -- it is not isochrone-free, and a 1-3 Myr star is
forced onto a >= 4 Myr isochrone. Output ``gspphot_bias_lambda_ori.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CACHE  # noqa: E402
from hr_fit import _f, lori_calibration  # noqa: E402


def main() -> None:
    from astropy.table import Table, join

    lo = Table.read(CACHE / "oracles/lambda_ori/lambda_ori_2mass_dr3.ecsv")
    g = Table.read(HERE / "gspphot_lambda_ori.ecsv")
    g.remove_column("source_id")
    g.rename_column("sid", "Source")
    j = join(lo, g, keys="Source")
    tg, tc = _f(j["teff_gspphot"]), 10 ** _f(j["logTeff"])
    ok = np.isfinite(tg) & np.isfinite(tc)
    d = np.log10(tg) - np.log10(tc)
    out = {
        "n_with_gspphot": int(ok.sum()),
        "n_total": int(len(j)),
        "dlogT_all": [float(x) for x in np.percentile(d[ok], [16, 50, 84])],
        "by_T_cao": [],
        "mh_gspphot_p16_50_84": [
            float(x) for x in np.nanpercentile(_f(j["mh_gspphot"])[ok], [16, 50, 84])
        ],
        "logg_gspphot_p16_50_84": [
            float(x) for x in np.nanpercentile(_f(j["logg_gspphot"])[ok], [16, 50, 84])
        ],
        "calibration_by_T_gsp": lori_calibration(),
    }
    for a, b in ((3000, 3600), (3600, 4200), (4200, 5000), (5000, 7000)):
        s = ok & (tc >= a) & (tc < b)
        out["by_T_cao"].append(
            {"T_cao": [a, b], "n": int(s.sum()), "median_dlogT": float(np.median(d[s]))}
        )
    print(json.dumps(out, indent=1))
    (HERE / "gspphot_bias_lambda_ori.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
