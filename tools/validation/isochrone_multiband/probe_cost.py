"""Cost probe: wall time and memory of one compiled log L + gradient, Gaia-only vs Gaia + 2MASS
(254 members, MIST v1.2, binaries on). Decides the experiment's budget; not a result."""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_mb import NIR, fitter, members, mist_mb

g = mist_mb()
out = {}
for name, nir, kw in (
    ("gaia", (), {}),
    ("nir", NIR, {}),
    ("nir_fitz_sav", NIR, {"ext": "fitz19", "sigma_av": True}),
):
    t0 = time.time()
    f = fitter(g, nir, **kw)
    f.setup(members(), prob_threshold=0.0)
    t1 = time.time()
    fl = f._compiled_loglike_nd(None)
    th = np.r_[
        -0.4, 6.1, 10.22, 0.7, 0.1, 0.1, [0.1] if f.fit_sigma_av else [], np.zeros(len(f.zp_bands))
    ]
    v, gr = fl(th)
    t2 = time.time()
    for _ in range(5):
        fl(th)
    t3 = time.time()
    out[name] = {
        "setup_s": t1 - t0,
        "compile_first_s": t2 - t1,
        "eval_s": (t3 - t2) / 5,
        "loglike": v,
        "params": f.param_names,
    }
    print(name, out[name], flush=True)
(Path(__file__).resolve().parent / "probe_cost.json").write_text(json.dumps(out, indent=1) + "\n")
