"""Deprecated compatibility shim for old NGC 6383 notebooks.

The paper-era helper implementation was migrated into ``cosmic.analysis``.
Keep this module temporarily so archived notebooks that still execute
``from COSMIC_aux import *`` fail softly instead of losing their import path.
New notebooks and scripts should import directly from ``cosmic.analysis``.
"""

from __future__ import annotations

import warnings

from cosmic.analysis import *  # noqa: F401,F403
from cosmic.analysis import __all__ as _analysis_all

warnings.warn(
    "COSMIC_aux is deprecated; import from cosmic.analysis instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = list(_analysis_all)
