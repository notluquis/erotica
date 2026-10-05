#!/usr/bin/env python3
"""NGC 6383 with the multiband likelihood (hub finding ``isochrone-multiband-implementation.md``
§2.3, run only because the synthetic's primary did not die).

Arms (MAP with ``common_mb.search_mb``, priors ``PRI_PLX``, MIST v1.2, 254 members):
``gaia`` (ccm, no NIR: anchor, must reproduce colour R1: [Fe/H] -0.47, log t 6.02, sigma_int 0.15),
``nir`` (ccm), ``nir_fitz`` (EDR3 law per EEP point), ``nir_fitz_sav`` (+ sigma_AV),
``nir_fitz_floor03`` (NIR floor 0.03 instead of 0.01), ``gaia_fitz`` (the law's effect without NIR).
``--profile ARM``: log t profile (MAP over everything else at each log t), with and without the dm
prior (``--noprior`` sets dm_sigma to 10, i.e. dm free inside dm_range; methodology §A.1.1).

WHAT WOULD FALSIFY a NIR age: an age that moves by more than its profile width between ``nir`` and
``nir_fitz`` (the extinction law decides it), or between floors 0.01 and 0.03 (the model floor
decides it), or one set by a prior bound (``at_bound``). Each is reported, not chosen between.

Output: ``real_fit.json`` (rewritten after each arm; resumable).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_mb import NIR, fitter, members, mist_mb, search_mb  # noqa: E402

ARMS = {
    "gaia": dict(nir=()),
    "nir": dict(nir=NIR),
    "nir_fitz": dict(nir=NIR, ext="fitz19"),
    "nir_fitz_sav": dict(nir=NIR, ext="fitz19", sigma_av=True),
    "nir_fitz_floor03": dict(nir=NIR, ext="fitz19", nir_floor=0.03),
    "gaia_fitz": dict(nir=(), ext="fitz19"),
}
PROFILE_LOGA = [5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.8]


def make(grid, arm: str, **over):
    kw = dict(ARMS[arm])
    nir = kw.pop("nir")
    f = fitter(grid, nir, **{**kw, **over})
    f.setup(members(), prob_threshold=0.0)
    return f


def main() -> None:
    import erotica

    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--profile", default=None)
    ap.add_argument("--noprior", action="store_true")
    a = ap.parse_args()
    path = HERE / "real_fit.json"
    out = json.loads(path.read_text()) if path.exists() else {"erotica_file": erotica.__file__}
    g = mist_mb()
    if a.profile:
        key = f"profile_{a.profile}" + ("_noprior" if a.noprior else "")
        prof = out.setdefault(key, {})
        f = make(g, a.profile, **({"dm_sigma": 10.0} if a.noprior else {}))
        for la in PROFILE_LOGA:
            if f"{la:.2f}" in prof:
                continue
            r = search_mb(f, verbose=False, fixed={"loga": la}, polish=3)
            prof[f"{la:.2f}"] = {
                k: r[k] for k in ("mode", "logpost", "loglike", "at_bound", "seconds")
            }
            print(key, la, round(r["logpost"], 2), json.dumps(r["mode"], default=float), flush=True)
            path.write_text(json.dumps(out, indent=1) + "\n")
        return
    for arm in a.arms:
        if arm in out:
            continue
        f = make(g, arm)
        r = search_mb(f, verbose=False)
        r["n_nir"] = [int(h.sum()) for h in getattr(f, "_nir_has", [])]
        r["param_names"] = f.param_names
        out[arm] = r
        print(
            arm,
            json.dumps(
                {k: r[k] for k in ("mode", "logpost", "at_bound", "seconds")}, default=float
            ),
            flush=True,
        )
        path.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
