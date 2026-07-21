"""Deprecated compatibility shim for old NGC 6383 notebooks.

The paper-era helper implementation was migrated into ``pumps.analysis``.
Keep this module temporarily so archived notebooks that still execute
``from COSMIC_aux import *`` fail softly instead of losing their import path.
New notebooks and scripts should import directly from ``pumps.analysis``.
"""

from __future__ import annotations

import warnings

from pumps.analysis import *  # noqa: F401,F403
from pumps.analysis import __all__ as _analysis_all

warnings.warn(
    "COSMIC_aux is deprecated; import from pumps.analysis instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = list(_analysis_all)
