#!/usr/bin/env python3
"""Prepare one fit of the interval experiment: the data table, the control estimator (MAP +
Laplace, exactly ``av_synth.run_fit``) and the NUTS chain starts.

    prep.py --cell real            the 254 members (X1's configuration)
    prep.py --cell A --rep k       synthetic, cell A (truth drawn per replicate around X1)
    prep.py --cell B --rep k       synthetic, cell B (old solar, ctl-off's truth, real noise)
    prep.py --cell C --rep k       synthetic, cell C (ctl-off exactly: clean noise, no field)

Cells, seeds and truths are the pre-registration of hub finding ``isochrone-intervals-nuts.md`` §0.
Writes ``runs/<cell>_<rep>/{data.ecsv, prep.json, starts.json}``. The truth's log L and log
posterior are evaluated next to the mode's (a construction bug shows as a truth far below the
mode; methodology §A.1.3).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import iv_common as C  # noqa: E402

RUNS = HERE / "runs"
N_STARS = 254


def truth_for(cell: str, rep: int, rng) -> dict:
    dm = float(rng.normal(10.223, 0.026))
    if cell == "A":  # around X1, off every MIST node and every search-lattice point by construction
        return {
            "feh": float(rng.uniform(-0.60, -0.30)),
            "loga": float(rng.uniform(5.95, 6.15)),
            "dm": dm, "Av": 0.66, "sigma_int": 0.153, "f_bg": 0.17,
        }
    if cell == "B":  # ctl-off's truth with the real data's noise
        return {"feh": 0.15, "loga": 6.43, "dm": dm, "Av": 1.0, "sigma_int": 0.153, "f_bg": 0.17}
    if cell == "C":  # ctl-off as run in isochrone-external-constraints.md §5 (no field)
        return {"feh": 0.15, "loga": 6.43, "dm": dm, "Av": 1.0, "sigma_int": 0.04, "f_bg": 0.0}
    raise ValueError(cell)


def logpost(f, x):
    ll = f.loglike(*x)
    return ll, ll - 0.5 * ((x[2] - f.dm_mu) / f.dm_sigma) ** 2


def main() -> None:
    import erotica

    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=["real", "A", "B", "C"])
    ap.add_argument("--rep", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    out = RUNS / f"{a.cell}_{a.rep}"
    out.mkdir(parents=True, exist_ok=True)
    real = C.real_table()
    info = {"cell": a.cell, "rep": a.rep, "erotica_file": erotica.__file__, "started": C.now()}
    if a.cell == "real":
        data = real
        truth = None
    else:
        seed = {"A": 3000, "B": 4000, "C": 5000}[a.cell] + a.rep
        rng = np.random.default_rng(seed)
        truth = truth_for(a.cell, a.rep, rng)
        gen = C.fitter(real)
        data, gi = C.draw_synthetic(gen, truth, N_STARS, rng, real)
        info.update({"seed": seed, "truth": truth, **gi})
    data.write(out / "data.ecsv", overwrite=True)
    f = C.fitter(data)
    ctrl = C.control_fit(f)
    info["control"] = ctrl
    if truth is not None:
        xt = [truth[k] for k in C.PARAMS]
        xm = [ctrl["mode"][k] for k in C.PARAMS]
        lt, pt_ = logpost(f, xt)
        lm, pm_ = logpost(f, xm)
        info["truth_vs_mode"] = {"loglike_truth": lt, "loglike_mode": lm,
                                 "logpost_truth": pt_, "logpost_mode": pm_}
    st = C.chain_starts(f, ctrl, np.random.default_rng(11 + a.rep))
    C.dump(out / "starts.json", st)
    info["seconds"] = round(time.time() - t0, 1)
    C.dump(out / "prep.json", info)
    m = ctrl["mode"]
    print(a.cell, a.rep, {k: round(m[k], 4) for k in C.PARAMS},
          {k: round(ctrl["laplace_sd"][k], 4) for k in ("feh", "loga")},
          info.get("truth_vs_mode"), info["seconds"], flush=True)


if __name__ == "__main__":
    main()
