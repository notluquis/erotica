#!/usr/bin/env python3
"""Sensitivity of arm K x W (isochrone-model-fix.md §2): the per-star DR3 errors replaced by the
fitter's own error model e(G) (``_fit_error_model``), so k scales a smooth function of G only. If the
fit is as good, what k buys is a width growing with G, not the per-star error. -> fits/Kpoly_W.json"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aniso as A  # noqa: E402

real = A.C.real_table()
f = A.make_fitter(real, "W")
f._e_obs_mag = np.asarray(f._e_mag_fn(f._obs_mag), float)
f._e_obs_col = np.asarray(f._e_col_fn(f._obs_mag), float)
res = A.search8(f, "K")
res.update(window="W", variant="per-star errors -> e(G) error model", finished=A.C.now())
A.C.dump(HERE / "fits" / "Kpoly_W.json", res)
print(res["mode"], res["logpost"], res["loglike"])
