#!/usr/bin/env python3
"""Fit one arm (width x window) of ``isochrone-model-fix.md`` §0.2 on the 254 real members:
``python fit_arm.py <I|V|VR> <F|W>`` -> ``fits/<arm>_<win>.json``."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aniso as A  # noqa: E402

arm, win = sys.argv[1], sys.argv[2]
real = A.C.real_table()
f = A.make_fitter(real, win)
res = A.search8(f, arm)
res.update(window=win, started=None, finished=A.C.now())
A.C.dump(HERE / "fits" / f"{arm}_{win}.json", res)
print(res["mode"], res["logpost"], res["at_bound"])
