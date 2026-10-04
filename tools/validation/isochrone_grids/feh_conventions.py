#!/usr/bin/env python3
"""What does each grid's metallicity label mean? Measured on the files' own header values.

WHY THIS EXISTS
---------------
The grid layer fits in each grid's [Fe/H] label instead of linear Z, because Z is not comparable
across grids. That claim needs the conventions measured, not recalled: the hub landscape finding
said MIST v2.5's solar Z_init was 0.0185 [V, its Table 1], and the v2.5 [Fe/H]=0 file header says
0.0163577. This script reads every header and reports, per grid, which candidate definition
reproduces all labels:

* ``log10(Z/Zsun)`` (ASteCA's ``z_to_FeH``)
* ``log10(Z/X) - log10(Z/X)_sun`` with X = 1 - Y - Z from the file (MIST: Yinit in the header;
  PARSEC: Y = 0.2485 + 1.78 Z, Bressan+12, and (Z/X)_sun = 0.0207 tested directly)

WHAT WOULD FALSIFY IT
---------------------
A version for which neither definition fits to < 1e-4 dex: then the label is something else and
[Fe/H] cannot be compared across grids without reading the paper's definition.

Output: ``feh_conventions.json`` next to this file. Inputs are local files; a missing one is
reported, not skipped silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))
NGC = Path.home() / "erotica/data/test/NGC6383"
CACHE = Path.home() / ".cache/erotica-grids"


def main() -> None:
    from erotica.analysis.grids import MISTGrid

    out: dict = {}
    for label, path in (
        ("MIST v1.2", NGC / "MIST/UBVRIplus"),
        ("MIST v2.5", CACHE / "mist_v2.5/UBVRIplus_afe0_vvcrit0.4"),
    ):
        if not path.exists():
            out[label] = {"missing": str(path)}
            continue
        g = MISTGrid(path, loga_range=(6.5, 6.5))
        y, z, feh = g.header_compositions.T
        fit = np.polyfit(z, y, 1)
        out[label] = {
            "n_files": int(len(feh)),
            "max_residual": g.feh_convention_residuals(),
            "Y_of_Z": {
                "Yp": float(fit[1]),
                "dY_dZ": float(fit[0]),
                "max_resid": float(np.max(np.abs(np.polyval(fit, z) - y))),
            },
            "compositions": [[float(a), float(b), float(c)] for a, b, c in g.header_compositions],
        }
    pf = NGC / "PARSEC/gaiaedr3/mets_ages.dat"
    if pf.exists():
        import pandas as pd

        names = next(ln for ln in open(pf) if ln.startswith("# Zini")).lstrip("#").split()
        d = pd.read_csv(
            pf, sep=r"\s+", comment="#", header=None, names=names, usecols=["Zini", "MH"]
        ).drop_duplicates()
        z, mh = d["Zini"].to_numpy(float), d["MH"].to_numpy(float)
        x = 1 - (0.2485 + 1.78 * z) - z
        out["PARSEC v1.2S (local CMD 3.7)"] = {
            "n_Z": int(len(z)),
            "max_residual": {
                "log10(Z/0.0152)": float(np.max(np.abs(np.log10(z / 0.0152) - mh))),
                "log10(Z/X)-log10(0.0207), Y=0.2485+1.78Z": float(
                    np.max(np.abs(np.log10(z / x) - np.log10(0.0207) - mh))
                ),
            },
            "pairs": [[float(a), float(b)] for a, b in zip(z, mh, strict=True)],
        }
    else:
        out["PARSEC"] = {"missing": str(pf)}
    (HERE / "feh_conventions.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v.get("max_residual", v) for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
