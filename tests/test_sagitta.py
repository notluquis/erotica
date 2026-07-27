"""Regression: _ensure_sagitta must NOT auto-pip-install.

Silently shelling out to `pip install git+https://.../Sagitta.git` (an unpinned
moving ref) modified the user's environment without consent and was not
reproducible. It must instead raise ImportError with manual-install instructions.
"""

import importlib.util
import subprocess

import pytest

from erotica.analysis._sagitta import _ensure_sagitta


@pytest.mark.skipif(
    importlib.util.find_spec("sagitta") is not None,
    reason="Sagitta is installed; the missing-package path cannot be exercised here",
)
def test_ensure_sagitta_raises_and_never_shells_out(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("_ensure_sagitta must not shell out to pip install")

    monkeypatch.setattr(subprocess, "run", _fail)
    with pytest.raises(ImportError, match="pip install"):
        _ensure_sagitta()
