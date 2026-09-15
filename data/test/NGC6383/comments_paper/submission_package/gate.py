#!/usr/bin/env python3
"""One command that has to pass before aa52082-24 is uploaded or committed.

Why this file exists
--------------------
Every defect this package accumulated was found by a human looking in a place nobody had
looked before, and each was of the same shape: an artefact that *describes* the manuscript
drifting away from the manuscript, silently, because nothing compared them.

    - the two letters promised the referee things the paper no longer said
    - nine sentences called the isochrone ensemble a posterior after Sect. 4.4 denied it
    - the marked PDF showed ten figures as deleted when they had only moved
    - the CDS ReadMe on NESTOR was the superseded one, and a third copy had diverged
    - the knowledge-graph note still carried the round-1 R_t of 40.4 arcmin
    - line numbers printed on top of the appendix text on every page

None of those is exotic. Each is cheap to detect and impossible to notice by reading.

The other half of the reason is structural. This paper lives inside the EROTICA package
repository, under `data/`, and the package's `.pre-commit-config.yaml` excludes `^(data/...)`
because `data/` is data. So the manuscript receives no hooks at all -- not by a decision
about the manuscript but as collateral of where it sits. `ARCHITECTURE.md` in the hub calls
this "today's broken state" and defers the fix until after acceptance, because 42 scripts
hardcode paths into this directory. Until then, this file is the substitute owner.

What this file is now (2026-09-15, refactor TOOL-gate-generalize)
-------------------------------------------------------------------
The 26 checks used to live here as Python. A second paper (P02) hit the same failure classes --
a stale copy, a marked diff that lies about what moved -- so 22 of them moved to the reusable
engine, `tools/manuscript_gate.py`, configured per paper by `gate.toml`. The 4 that only make
sense for THIS manuscript (Table 1's internal consistency, the two literature-table
cross-checks, the catalogue-derived numbers) stay in `gate_local.py`, loaded via `gate.toml`'s
`local` key. This file is now a shim: it finds the engine, configures it for NGC 6383, and
re-exports what other scripts in this directory import from `gate` directly --
`check_posterior_claims.py` does `from gate import LETTERS, TEX`, and
`test_gate_mutations.py`/`test_gate_behaviour.py` load this file by spec and read
`gate.KB_NOTES`, `gate.TEX`, `gate.build_paths`, `gate.pages_in`.

Usage
-----
    python3 gate.py            # everything, ~4 min (rebuilds both PDFs)
    python3 gate.py --quick    # everything that does not need a LaTeX run, ~5 s

Exit code is 0 only if every check passes. Each check prints what it compared, not just a
verdict, so a failure says what to look at.
"""

from __future__ import annotations

import importlib.util
import os
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

# GATE_TOML: una copia mutada del toml, para test_gate_behaviour.py. Hasta 2026-09-15 ese test
# escribia y restauraba el gate.toml REAL, y `src/check_findings.py` del hub corre sus guardias con
# 8 hilos: doce de ellas lanzan ese test, asi que dos corridas se pisaban (fallos espurios medidos)
# y una podia "restaurar" la mutacion de la otra y dejar el config trackeado corrupto. Las rutas
# siguen resolviendose contra este directorio. Se imprime, para que nunca actue en silencio.
_toml = os.environ.get("GATE_TOML")
if _toml:
    print(f"config alternativa (GATE_TOML): {_toml}")
mg.configure(Path(_toml) if _toml else HERE / "gate.toml", base=HERE)

# Re-exportado para quien haga `from gate import ...` o cargue este fichero por spec y lea
# `gate.<nombre>` -- ver "What this file is now" arriba para quien depende de cada uno.
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
