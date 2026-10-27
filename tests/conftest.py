"""Pytest environment defaults for optional Bayesian backends."""

from __future__ import annotations

import os
import tempfile
from importlib import metadata
from pathlib import Path


_cache_root = Path(tempfile.gettempdir()) / "cosmic-test-cache"


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "none"


# PyTensor keeps its numba cache in ``<base_compiledir>/numba`` and does not version it (only the
# sibling ``compiledir_*`` carries the python version). Two pytest runs from environments with
# different pytensor/numba, started together on one directory, wrote into each other's entries and
# one of them died with a numba ``TypingError`` (``mul_N`` resolved to the wrong function;
# agent-findings/env-refresh-2026-10-26.md of the PhD hub, section 13: 6 of 60 runs
# when shared, 0 of 60 when separate). One directory per (pytensor, numba) pair removes the race
# and keeps the cache warm between runs of the same environment.
_stack = f"pytensor-{_version('pytensor')}_numba-{_version('numba')}"
_cache_root.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))

_pytensor_flags = os.environ.get("PYTENSOR_FLAGS", "")
if "base_compiledir=" not in _pytensor_flags:
    compiledir_flag = f"base_compiledir={_cache_root / _stack / 'pytensor'}"
    os.environ["PYTENSOR_FLAGS"] = (
        f"{_pytensor_flags},{compiledir_flag}" if _pytensor_flags else compiledir_flag
    )
