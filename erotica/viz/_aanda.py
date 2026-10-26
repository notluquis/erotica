"""A&A figure style: widths from ``aa.cls`` and print-size rcParams."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import matplotlib as mpl

_MM_PER_IN = 25.4

#: Full text width of an A&A page, in inches. ``aa.cls`` sets ``\aa@fullwidth`` to 184 mm (both
#: the A4 and the letter branches) and passes it to ``geometry`` as ``textwidth``.
AANDA_TEXT_IN: float = 184.0 / _MM_PER_IN
#: One column of the two-column layout, in inches: ``aa.cls`` sets ``columnsep=4mm``, so a column
#: is (184 mm − 4 mm) / 2 = 90 mm.
AANDA_COLUMN_IN: float = (184.0 - 4.0) / 2 / _MM_PER_IN

#: Four Okabe–Ito colours (blue, bluish green, vermilion, reddish purple), to be assigned in this
#: fixed order. Checked with a colour-vision-deficiency palette validator against a white surface
#: (2026-10-26): every hue inside the lightness band and above the chroma floor, worst adjacent-pair
#: ΔE 11.0 under deuteranopia and 16.4 for normal vision. Reddish purple has only 2.98:1 contrast
#: against white, so give that series its own line style or marker too. The full eight-colour
#: Okabe–Ito set fails the same validator (yellow too light, black has no chroma; orange, sky blue
#: and yellow under 2.3:1 contrast), which is why it stops at four: a fifth series belongs in small
#: multiples or a grey "other", not in a weaker hue.
OKABE_ITO: tuple[str, ...] = ("#0072B2", "#009E73", "#D55E00", "#CC79A7")

#: rcParams for A&A figures included at their natural width (``figsize`` below). Text is 9 pt and
#: tick labels 8 pt; the smallest default text, the legend, is 7.5 pt so that nothing falls under
#: the 7 pt that :func:`erotica.viz.check_layout` enforces. STIX matches the journal's Times-like
#: text and ships with matplotlib. ``pdf.fonttype`` 42 embeds TrueType, which the typesetter can
#: edit.
AANDA_RC: dict[str, object] = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "legend.frameon": False,
    "axes.prop_cycle": mpl.cycler(color=list(OKABE_ITO)),
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
}


def figsize(
    width: Literal["column", "text"] | float = "column", aspect: float = 0.75
) -> tuple[float, float]:
    """Return ``(width, height)`` in inches for an A&A figure.

    Parameters
    ----------
    width
        ``"column"`` (one column, 90 mm), ``"text"`` (full page width, 184 mm), or a width in
        inches.
    aspect
        Height divided by width.
    """
    if isinstance(width, str):
        try:
            w = {"column": AANDA_COLUMN_IN, "text": AANDA_TEXT_IN}[width]
        except KeyError:
            raise ValueError(f"width must be 'column', 'text' or a number, not {width!r}") from None
    else:
        w = float(width)
    if not (w > 0 and aspect > 0):
        raise ValueError("width and aspect must be positive")
    return (w, w * aspect)


@contextmanager
def aanda_style(rc: dict | None = None) -> Iterator[None]:
    """Apply :data:`AANDA_RC` (plus any overrides in ``rc``) inside a ``with`` block.

    The previous rcParams are restored on exit, so this does not change the global style that
    ``erotica.core._style.apply_default_style()`` sets for interactive plots.
    """
    params = dict(AANDA_RC)
    if rc:
        params.update(rc)
    with mpl.rc_context(params):
        yield
