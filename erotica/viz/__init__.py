"""Figure helpers for journal-ready plots: an A&A style, its widths, a colour-blind-safe palette and
a layout check.

This subpackage does not draw any figure type by itself. The figures EROTICA's analyses need
already live next to the analyses (:mod:`erotica.analysis.figures`, ``erotica.core._plots``), and
generic posterior plots come from established packages — ``corner`` for corner plots and
``arviz-plots`` for traces, posterior-predictive checks and PIT/ECDF calibration. What was missing
is the part every paper figure shares:

* :func:`aanda_style` / :data:`AANDA_RC` — rcParams sized for *Astronomy & Astrophysics* at
  print size (9 pt text, nothing under 7 pt, STIX fonts that ship with matplotlib, TrueType in PDF);
* :func:`figsize` with :data:`AANDA_COLUMN_IN` and :data:`AANDA_TEXT_IN`, taken from ``aa.cls``;
* :data:`OKABE_ITO` — a categorical palette that stays distinguishable under colour-vision
  deficiency;
* :func:`check_layout` — flags text cut off at the canvas edge, legends or labels drawn on top of
  data markers, and text below a minimum size. A figure can reproduce every number it is meant to
  show and still hide a data point under its legend; no numerical test sees that.

Example
-------
>>> import matplotlib.pyplot as plt
>>> from erotica import viz
>>> with viz.aanda_style():
...     fig, ax = plt.subplots(figsize=viz.figsize("column"))
...     _ = ax.plot([0, 1], [0, 1], color=viz.OKABE_ITO[0], label="model")
...     _ = ax.legend(loc="upper left")
...     problems = viz.check_layout(fig)
>>> problems
[]
"""

from ._aanda import AANDA_COLUMN_IN, AANDA_RC, AANDA_TEXT_IN, OKABE_ITO, aanda_style, figsize
from ._layout import check_layout

__all__ = [
    "AANDA_COLUMN_IN",
    "AANDA_RC",
    "AANDA_TEXT_IN",
    "OKABE_ITO",
    "aanda_style",
    "check_layout",
    "figsize",
]
