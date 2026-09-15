#!/usr/bin/env python3
"""Shim: carga el motor generico (`tools/manuscript_gate.py`) y lo configura para NGC 6383
(aa52082-24) con `gate.toml`, en este mismo directorio.

Los 26 checks y su docstring viven en el motor (los 22 genericos) y en `gate_local.py` (los 4
propios de este paper). Este fichero no implementa ningun check; solo encuentra el motor, lo
configura, y re-exporta lo que otros scripts de este directorio importan de `gate` directamente
(`check_posterior_claims.py` hace `from gate import LETTERS, TEX`; `test_gate_mutations.py` y
`test_gate_behaviour.py` cargan este fichero por spec y leen `gate.KB_NOTES`, `gate.TEX`,
`gate.build_paths`, `gate.pages_in`, etc.)

Preparado en el paso 2 del refactor `TOOL-gate-generalize` (hub, `open-threads.md`): mientras el
`gate.py` de verdad siga siendo el fichero de siempre, este vive aparte como `_gate_shim.py` para
que la paridad se pueda medir sin tocar lo que ya funciona.
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

# Re-exportado para quien haga `from gate import ...` o cargue este fichero por spec y lea
# `gate.<nombre>` -- ver el docstring de arriba para quien depende de cada uno.
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
