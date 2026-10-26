"""Layout check for a drawn figure: clipped text, text over data markers, text too small.

Generalised from the guard written for the HD 159176 manuscript figures (2026-10-22), where a
legend covered a data point and a y-axis label was cut in half. Both figures reproduced their
numbers exactly; neither defect is visible to a numerical test.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.text import Text

_TOL_PX = 1.0


def _data_markers(ax, max_points: int) -> np.ndarray:
    """Display coordinates of the individual markers inside ``ax``.

    Scatter collections and lines drawn with a marker count; series with more than ``max_points``
    points are treated as a density cloud (posterior samples, a field population) and skipped,
    because text over a cloud does not hide any single datum.
    """
    pts = []
    for c in ax.collections:
        if isinstance(c, PathCollection) and c.get_visible():
            off = c.get_offsets()
            if 0 < len(off) <= max_points:
                pts.append(c.get_offset_transform().transform(off))
    for ln in ax.lines:
        if not ln.get_visible() or ln.get_marker() in (None, "None", "", " ", ","):
            continue
        xy = np.asarray(ln.get_xydata(), float)
        if 0 < len(xy) <= max_points:
            xy = xy[np.all(np.isfinite(xy), axis=1)]
            if len(xy):
                pts.append(ln.get_transform().transform(xy))
    if not pts:
        return np.zeros((0, 2))
    p = np.concatenate(pts)
    bb = ax.get_window_extent()
    inside = (p[:, 0] >= bb.x0) & (p[:, 0] <= bb.x1) & (p[:, 1] >= bb.y0) & (p[:, 1] <= bb.y1)
    return p[inside]


def _hidden_tick_labels(fig: Figure) -> set[int]:
    """Tick labels outside the visible axis interval; matplotlib does not draw them."""
    hidden: set[int] = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tk.get_loc()
                pad_lo, pad_hi = 1e-9 * max(1, abs(lo)), 1e-9 * max(1, abs(hi))
                if loc is None or not (lo - pad_lo <= loc <= hi + pad_hi):
                    hidden.update({id(tk.label1), id(tk.label2)})
    return hidden


def check_layout(
    fig: Figure,
    *,
    allow: Iterable[str] = (),
    min_fontsize: float = 7.0,
    max_points: int = 400,
) -> list[str]:
    """Return the layout problems of ``fig`` at its current size; an empty list means none found.

    Call it on the figure at the size it will be printed (see :func:`erotica.viz.figsize`), after
    everything is drawn and before saving. Three checks:

    ``CLIPPED``
        visible text (labels, tick labels, titles, legends, annotations) partly outside the canvas.
        Text entirely outside is skipped: matplotlib does not draw it.
    ``OVERLAP``
        a legend, or a text inside an axes, covers a data marker of a series with at most
        ``max_points`` points.
    ``FONTSIZE``
        text smaller than ``min_fontsize`` points. The 7 pt default is a legibility floor at print
        size, not a journal rule.

    Parameters
    ----------
    allow
        Substrings of text that are exempt on purpose (e.g. a label that deliberately sits on a
        dense region). Say why at the call site.
    """
    allow = tuple(allow)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fb = fig.bbox
    hidden = _hidden_tick_labels(fig)
    problems: list[str] = []

    def allowed(t: Text) -> bool:
        return any(a in t.get_text() for a in allow)

    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip() or allowed(t) or id(t) in hidden:
            continue
        e = t.get_window_extent(renderer)
        if e.width == 0:
            continue
        if e.x1 < fb.x0 or e.x0 > fb.x1 or e.y1 < fb.y0 or e.y0 > fb.y1:
            continue
        if (
            e.x0 < fb.x0 - _TOL_PX
            or e.y0 < fb.y0 - _TOL_PX
            or e.x1 > fb.x1 + _TOL_PX
            or e.y1 > fb.y1 + _TOL_PX
        ):
            problems.append(f"CLIPPED: {t.get_text()[:40]!r} extends past the canvas edge")
        if t.get_fontsize() < min_fontsize - 1e-6:
            problems.append(
                f"FONTSIZE: {t.get_text()[:40]!r} is {t.get_fontsize():.1f} pt < {min_fontsize} pt"
            )

    for ax in fig.axes:
        pts = _data_markers(ax, max_points)
        if not len(pts):
            continue
        boxes = []
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            boxes.append(("legend", leg.get_window_extent(renderer)))
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip() and not allowed(t):
                # Text's own extent, not Annotation's: an annotation's extent includes its arrow,
                # which reaches the point it marks by construction.
                boxes.append((f"text {t.get_text()[:30]!r}", Text.get_window_extent(t, renderer)))
        for lg in fig.legends:
            boxes.append(("figure legend", lg.get_window_extent(renderer)))
        for what, b in boxes:
            hit = (pts[:, 0] > b.x0) & (pts[:, 0] < b.x1) & (pts[:, 1] > b.y0) & (pts[:, 1] < b.y1)
            if hit.any():
                problems.append(f"OVERLAP: {what} covers {int(hit.sum())} data marker(s)")
    return problems
