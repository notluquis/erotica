#!/usr/bin/env python3
"""One command that has to pass before the EROTICA methods paper (P02) is submitted or
committed.

Shim over the reusable engine (`tools/manuscript_gate.py`), configured for this paper by
`gate.toml` in this same directory. The 22 checks that are not specific to a single manuscript
live in the engine; this paper declares no `local` checks of its own (unlike NGC 6383's Table 1 /
literature-table cross-checks) and marks 14 of the 22 not applicable for a first submission --
see `gate.toml`'s `[not_applicable]` table for which, and why.

Usage
-----
    python3 gate.py            # everything, ~1 min (rebuilds the one document)
    python3 gate.py --quick    # everything that does not need a LaTeX run, ~5 s

Exit code is 0 only if every applicable check passes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

for parent in HERE.parents:
    candidate = parent / "tools" / "manuscript_gate.py"
    if candidate.exists():
        ENGINE_PATH = candidate
        break
else:
    raise SystemExit(f"no encuentro tools/manuscript_gate.py subiendo desde {HERE}")

_spec = importlib.util.spec_from_file_location("manuscript_gate", ENGINE_PATH)
mg = importlib.util.module_from_spec(_spec)
sys.modules["manuscript_gate"] = mg
_spec.loader.exec_module(mg)

mg.configure(HERE / "gate.toml")

# Re-exportado por si algun script de este directorio hace `from gate import ...`.
TEX = mg.TEX
MARKED = mg.MARKED
LETTERS = mg.LETTERS
KB_ROOT = mg.KB_ROOT
KB_NOTES = mg.KB_NOTES
BUILD_DIR = mg.BUILD_DIR
build_paths = mg.build_paths
pages_in = mg.pages_in


if __name__ == "__main__":
    raise SystemExit(mg.main())
