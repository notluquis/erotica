"""Turn a benchmark checkpoint (``*.cells.jsonl``) into the committed JSON sidecar.

WHY THIS EXISTS
---------------
``benchmark_erotica_vs_asteca.py`` checkpoints every cell to ``<out>.cells.jsonl`` and only
assembles ``<out>.json`` at the very end. Two things make that end-of-run assembly an
unreliable place to keep the result:

* it runs the out-of-sample recalibration, which is where an API mismatch crashed a
  completed 97-cell grid (``erotica.calibration.expected_calibration_error`` returns
  ``(ECE, MCE)``, not a scalar) — losing the sidecar for a run whose cells were all fine;
* a long grid can be interrupted, and a partial checkpoint is still a usable result.

So the checkpoint is the source of truth and this script is the assembler. It also strips
the per-star ``_probs`` / ``_truth`` arrays, which are what make the checkpoint several MB:
``tools/validation/CLAUDE.md`` requires the committed JSON to carry everything a paper
quotes, with bulk arrays kept out of git.

Usage:
    python tools/validation/benchmark_finalise.py <out>.json [--controls <controls>.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib.metadata import version as _pkg_version
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="target .json (its .cells.jsonl is read)")
    ap.add_argument(
        "--controls",
        default=None,
        help="a benchmark_controls.json to fold in the negative control "
        "and sweep-sensitivity blocks",
    )
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).parent))
    from benchmark_erotica_vs_asteca import out_of_sample_recalibration

    out = Path(args.out)
    ckpt = out.with_suffix(".cells.jsonl")
    cells = [json.loads(line) for line in ckpt.read_text().splitlines() if line.strip()]
    if not cells:
        raise SystemExit(f"no cells in {ckpt}")

    methods = sorted(
        {
            r["method"]
            for c in cells
            for r in c["calibration"]
            if r["method"] != "_truth_positive_control"
        }
    )
    recal = {m: out_of_sample_recalibration(cells, method=m) for m in methods}

    control = sensitivity = None
    if args.controls:
        cd = json.loads(Path(args.controls).read_text())
        control, sensitivity = cd.get("negative_control"), cd.get("sweep_sensitivity")

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "assembled_from": str(ckpt),
        "n_cells": len(cells),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
            "packages": {
                p: _pkg_version(p)
                for p in (
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "astropy",
                    "pandas",
                    "hdbscan",
                    "asteca",
                    "erotica",
                )
            },
        },
        "cells": [{k: v for k, v in c.items() if not k.startswith("_")} for c in cells],
        "negative_control": control,
        "sweep_sensitivity": sensitivity,
        "out_of_sample_recalibration": recal,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} from {len(cells)} checkpointed cells", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
