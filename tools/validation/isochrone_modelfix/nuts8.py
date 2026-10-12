#!/usr/bin/env python3
"""``isochrone_intervals/nuts_chain.py`` on an arm of ``isochrone-model-fix.md`` instead of X1's model:
``ARM=K WINDOW=W python nuts8.py run|finalize <nuts_chain arguments>``. Same kernel, chunking,
resume and summaries; only the model (``aniso.build_model8``) and the summarised parameter list
(+ the arm's extra variables) change. ``--data`` is ignored for the model's sample (the 254 members,
cut to the arm's window by ``setup``) but kept for nuts_chain's config check."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "isochrone_intervals"))
import aniso as A  # noqa: E402
import nuts_chain as N  # noqa: E402

ARM, WIN = os.environ.get("ARM", "K"), os.environ.get("WINDOW", "W")
EXTRA = {"k": "k", "beta": "beta", "sC": "sigma_c", "tau": "tau"}


def _model(data_path):
    f = A.make_fitter(N._data(data_path), WIN)
    return f, A.build_model8(f, ARM)


N._model = _model
N.C.PARAMS = tuple(N.C.PARAMS) + tuple(EXTRA[p] for p in A.ARM_FREE[ARM] if p in EXTRA)

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "finalize"])
    for a, kw in [("--data", dict(required=True)), ("--starts", {}), ("--chain", dict(type=int, default=0)),
                  ("--outdir", {}), ("--outdirs", {}), ("--warmup", dict(type=int, default=1000)),
                  ("--draws", dict(type=int, default=1000)), ("--chunk", dict(type=int, default=25)),
                  ("--seed", dict(type=int, default=7)), ("--target-accept", dict(type=float, default=0.9)),
                  ("--json", {}), ("--fig", {}), ("--truth", dict(default="")), ("--title", dict(default="")),
                  ("--save-draws", dict(default=""))]:
        ap.add_argument(a, **kw)
    a = ap.parse_args()
    N.run(a) if a.stage == "run" else N.finalize(a)
