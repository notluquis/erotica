#!/usr/bin/env python3
"""Cheap arm proposed by another finding (hub ``isochrone-sed-external.md``, commits
13582bb..4da0937): a DISTRIBUTED A_V, A_V,i ~ N(mu, tau^2) in the layer's scale, mu ~ N(0.85, 0.2),
tau = 0.30. Its falsifier (theirs): if it does not lower sigma_int (0.153 in X1), it dies.

What is run here (MAP only, not NUTS -- the cores are held by the coverage cell A):
* tau enters as a per-star variance (k_G tau)^2 in G and (k_col tau)^2 in colour, added to the
  catalogue errors. **The covariance k_G k_col tau^2 between G and colour is dropped** (the
  likelihood's segment density takes independent G and colour widths); with k_G 0.84 and k_col 0.45
  the true correlation is 1, so this is an approximation, declared, not the proposed model.
* mu is the fitter's A_V, with the Gaussian prior N(0.85, 0.2) added to the optimised objective
  (``av_synth.with_av_prior``), the dm prior as always.
Arms: tau = 0 (X1 with the A_V prior only) and tau = 0.30, and tau = 0.15 as a half step.
Writes ``avdist_map.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import iv_common as C  # noqa: E402
from av_synth import with_av_prior  # noqa: E402


def main() -> None:
    out = {"source": "isochrone-sed-external.md (another agent), arm proposed there", "arms": {}}
    for tau in (0.0, 0.15, 0.30):
        data = C.real_table()
        f = C.fitter(data)
        kG, kc = f._kG, f._k_col1
        f._e_obs_mag = np.sqrt(f._e_obs_mag**2 + (kG * tau) ** 2)
        f._e_obs_col = np.sqrt(f._e_obs_col**2 + (kc * tau) ** 2)
        with_av_prior(f, 0.85, 0.2)
        res = C.search(f, verbose=False)
        out["arms"][f"tau_{tau:.2f}"] = {"mode": res["mode"], "logpost": res["logpost"],
                                         "at_bound": res["at_bound"], "kG": kG, "k_col": kc}
        print(tau, {k: round(v, 4) for k, v in res["mode"].items()}, res["at_bound"], flush=True)
    C.dump(HERE / "avdist_map.json", out)


if __name__ == "__main__":
    main()
