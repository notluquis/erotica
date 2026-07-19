"""Pytest environment defaults for optional Bayesian backends."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


_cache_root = Path(tempfile.gettempdir()) / "cosmic-test-cache"
_cache_root.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))

_pytensor_flags = os.environ.get("PYTENSOR_FLAGS", "")
if "base_compiledir=" not in _pytensor_flags:
    compiledir_flag = f"base_compiledir={_cache_root / 'pytensor'}"
    os.environ["PYTENSOR_FLAGS"] = (
        f"{_pytensor_flags},{compiledir_flag}" if _pytensor_flags else compiledir_flag
    )
