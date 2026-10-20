#!/usr/bin/env python3
r"""NUTS of one screen configuration with the resumable chunked driver of C1.

``isochrone_c1_chunked.py`` runs the same numpyro kernel as ``sample_jax_nuts`` in resumable
chunks; this wrapper only swaps its ``_fitter`` for a ``ScreenFitter`` of configuration
``--cfg`` (JSON) and builds the search file (mode = the real-data MAP of ``real.json``, starts at the
mode plus a uniform offset of up to 2 Laplace sd, seed 42). Hub finding ``isochrone-model-screen.md``
§0.6: 2 chains x (1000 warmup + 1000), chain_method vectorized, one process.

    python nuts.py prep --name N --cfg '{"zprior":1,...}'
    python nuts.py run  --name N        (resumable)
    python nuts.py finalize --name N
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import isochrone_c1_chunked as ck  # noqa: E402
from model import build_screen_model  # noqa: E402
from screen import ZPRIOR, make_fitter, real_table  # noqa: E402

CACHE = Path(os.path.expanduser("~/.cache/erotica-screen/nuts"))


def _patch(cfg):
    def _fitter():
        f = make_fitter(cfg, real_table())
        f.build_model = lambda: build_screen_model(f, cfg, f._plx, ZPRIOR)
        return f

    ck._fitter = _fitter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "run", "finalize"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--cfg", default="")
    a = ap.parse_args()
    d = CACHE / a.name
    d.mkdir(parents=True, exist_ok=True)
    if a.stage == "prep":
        cfg = json.loads(a.cfg)
        real = json.loads((HERE / "real.json").read_text())
        r = next((c for c in real["configs"] if c["cfg"] == cfg), None)
        if r is None:  # a configuration outside the 16 (the main-effects one): its MAP first
            from screen import fit

            r = fit(make_fitter(cfg, real_table()), cfg)
            real.setdefault("extra", []).append(dict(r, cfg=cfg))
            (HERE / "real.json").write_text(json.dumps(real, indent=1, default=float) + "\n")
        p, sd = r["params"], r["laplace"]["sd"]
        rng = np.random.default_rng(42)
        mode = {k: p[k] for k in ("met", "loga", "dm", "Av", "sigma_int", "f_bg")}
        if cfg["diffred"] > 0:
            mode["sigma_Av"] = p["sigma_Av"]
        f = make_fitter(cfg, real_table())
        lo = {"loga": f.loga_range[0], "dm": f.dm_range[0], "Av": f.Av_range[0]}
        hi = {"loga": f.loga_range[1], "dm": f.dm_range[1], "Av": f.Av_range[1]}
        starts = []
        for _ in range(2):
            st = dict(mode)
            st["met"] = float(10 ** (np.log10(p["met"]) + rng.uniform(-2, 2) * sd[0]))
            for k, i in (("loga", 1), ("dm", 2), ("Av", 3)):
                eps = 1e-4 * (hi[k] - lo[k])
                st[k] = float(np.clip(p[k] + rng.uniform(-2, 2) * sd[i], lo[k] + eps, hi[k] - eps))
            starts.append(st)
        local_sd = {
            "met": float(p["met"] * np.log(10) * sd[0]),
            "loga": sd[1],
            "dm": sd[2],
            "Av": sd[3],
        }
        (d / "search.json").write_text(
            json.dumps(
                dict(
                    cfg=cfg,
                    mode=mode,
                    local_sd=local_sd,
                    loglike=r["logpost"],
                    runner_up=[],
                    chain_starts=starts,
                ),
                indent=1,
            )
        )
        print((d / "search.json").read_text())
        return
    s = json.loads((d / "search.json").read_text())
    _patch(s["cfg"])
    ns = argparse.Namespace(
        search=str(d / "search.json"),
        outdir=str(d / "chains"),
        warmup=1000,
        draws=1000,
        chunk=25,
        chains=2,
        seed=42,
        target_accept=0.9,
        mass="seeded",
        chain_method="vectorized",
        fresh=False,
        stop_at=0,
        chain_index=-1,
        json=str(HERE / f"nuts_{a.name}.json"),
    )
    if a.stage == "run":
        ck.stage_run(ns)
    else:
        ck.stage_finalize(ns)


if __name__ == "__main__":
    main()
