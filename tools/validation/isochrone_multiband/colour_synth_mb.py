#!/usr/bin/env python3
"""The colour synthetic of ``isochrone_colour/colour_synth.py``, fitted with Gaia only and with
Gaia + 2MASS J, H, Ks: does the near infrared remove the [Fe/H] bias that a lambda Ori-like
colour-T_eff error in BP produces? (Hub finding ``isochrone-multiband-implementation.md``; the
prediction was written in ``isochrone-multiband-nir.md`` §4.4(2) before any of this code existed.)

Generator: ``MultibandIsochroneFitter.draw_stars_nd`` -- the forward model of the likelihood
(Chabrier IMF, Offner binaries, the real members' Gaia and per-band NIR error models, the
completeness cut in G), N = 254, each synthetic star taking the missing-band and zero-point-group
pattern of the real member at the same rank in G (§A.1.3). Truth grid: MIST v1.2 with BP moved by
Delta(T_eff) (``common.shifted``; G, RP and the 2MASS bands untouched), CCM89 extinction, true zero
points 0. Truth: [Fe/H] 0, log t 6.40, A_V 1.0, sigma_int 0.03, dm ~ N(10.223, 0.026) per replicate.
Fits: unmodified MIST v1.2, ``common_mb.search_mb``, priors ``PRI_PLX``; ``gaia`` = no NIR band
(the 2D likelihood, by the control test), ``nir`` = J, H, Ks with one zero point per band on the
VIRAC2-sourced stars, prior N(0, 0.02). Same seeds in every arm: paired replicates.

Arms: ``control`` (Delta = 0), ``lori`` (``colour_synth.dcol_lori``), and ``nirJ`` (no BP shift; the
truth's 2MASS J moved by +0.060 mag, the J-Ks offset measured on lambda Ori, design §4.4(3); only the
``nir`` fit, since the Gaia data are identical to ``control``'s by construction).

PRE-REGISTERED RULE (hub finding §2, written before the first replicate ran):
* primary: b = median over replicates of [Delta[Fe/H](lori) - Delta[Fe/H](control)], per band set;
  ratio rho = |b_gaia| / |b_nir|. rho >= 3: prediction confirmed; rho < 2: prediction dies (and the
  design says abandon); 2 <= rho < 3: neither.
* secondary: the same for log t, read only if the 95 % bootstrap interval of b_gaia(log t) excludes 0.
* interpretability: control median |Delta| <= 0.10 in [Fe/H] and log t in both band sets; and the
  Gaia-only b within 0.05 of the colour experiment's paired -0.173 (else the generators differ).
* ``nirJ``: the NIR fit "breaks" if the paired |Delta[Fe/H]| or |Delta log t| median >= 0.05.

Output: ``colour_synth_mb_<arm>.json``, rewritten after every fit (resumable: done fits are skipped).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "isochrone_colour"))
from colour_synth import TRUTH, dcol_lori  # noqa: E402
from common import shifted  # noqa: E402
from common_mb import NIR, fitter, members, mist_mb, nir_pattern, search_mb  # noqa: E402

N_STARS = 254
J_SHIFT = 0.060


def shifted_j(grid, dj: float):
    import copy

    from erotica.analysis.grids.base import GridNode

    g = copy.copy(grid)
    g._nodes = {}
    for k, n in grid._nodes.items():
        mags = dict(n.mags)
        mags["2MASS_J"] = n.mags["2MASS_J"] + dj
        g._nodes[k] = GridNode(n.eep, n.mass, mags, n.extra)
    g.name = grid.name + f" [J{dj:+.3f}]"
    return g


def main() -> None:
    import erotica
    from erotica.analysis._isochrone_multiband import synthetic_table

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["control", "lori", "nirJ"])
    ap.add_argument("--reps", type=int, default=8)
    a = ap.parse_args()

    base = mist_mb()
    if a.arm == "control":
        truth_grid, info = base, {}
    elif a.arm == "lori":
        dcol, info = dcol_lori()
        truth_grid = shifted(base, dcol, a.arm)
    else:
        truth_grid, info = shifted_j(base, J_SHIFT), {"dJ": J_SHIFT}
    real = members()
    gen = fitter(truth_grid, NIR)
    gen.setup(real, prob_threshold=0.0)
    pat = nir_pattern(gen)
    path = HERE / f"colour_synth_mb_{a.arm}.json"
    out = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "erotica_file": erotica.__file__,
            "arm": a.arm,
            "truth": TRUTH,
            "delta": info,
            "reps": {},
        }
    )
    sets = ("nir",) if a.arm == "nirJ" else ("gaia", "nir")
    for r in range(a.reps):
        rng = np.random.default_rng(1000 + r)
        dm = float(rng.normal(TRUTH["dm_mu"], TRUTH["dm_sd"]))
        theta = [TRUTH["feh"], TRUTH["loga"], dm, TRUTH["Av"], TRUTH["sigma_int"], 0.0, 0, 0, 0]
        cols = gen.draw_stars_nd(theta, N_STARS, rng, nir_pattern=pat)
        syn = synthetic_table(cols)
        rec = out["reps"].setdefault(str(r), {"truth_dm": dm})
        for s in sets:
            if s in rec:
                continue
            f = fitter(base, NIR if s == "nir" else ())
            f.setup(syn, prob_threshold=0.0)
            res = search_mb(f, verbose=False)
            m = res["mode"]
            res["d_feh"] = m["feh"] - TRUTH["feh"]
            res["d_loga"] = m["loga"] - TRUTH["loga"]
            res["d_Av"] = m["Av"] - TRUTH["Av"]
            res["n_nir"] = [int(h.sum()) for h in f._nir_has] if s == "nir" else []
            rec[s] = res
            print(
                a.arm,
                r,
                s,
                json.dumps(
                    {k: res[k] for k in ("d_feh", "d_loga", "d_Av", "at_bound", "seconds")},
                    default=float,
                ),
                flush=True,
            )
            path.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
