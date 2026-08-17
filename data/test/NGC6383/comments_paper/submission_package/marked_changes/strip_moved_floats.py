#!/usr/bin/env python3
"""Remove latexdiff's orphan struck-through captions for floats that only MOVED.

latexdiff has no move detection (upstream issue #162). When the round-2 restructure
relocated a figure, latexdiff comments the float scaffolding out
(``%DIFDELCMD < \\begin{figure}``, ``\\includegraphics``, ``\\end{figure}``) but emits the
caption text live as ``\\DIFdelFL{...}``. The result is a red struck-through caption
paragraph, with no image and no float, at the figure's old position -- which reads to a
referee as "they cut this figure" when the figure is alive three pages later.

This script removes such a span ONLY when every ``\\includegraphics`` path and every
``\\label`` inside it is still present elsewhere in the document. A genuine deletion has
no live twin, so it survives and stays visible.

The unit removed is the **float span** (``%DIFDELCMD < \\begin{figure}`` ...
``%DIFDELCMD < \\end{figure}``), not the enclosing ``\\DIFdelbegin...\\DIFdelend`` block.
Those blocks routinely carry real deletions alongside the moved float -- one carries a
deleted ``\\subsubsection`` heading, another 14.6 kB of restructured prose -- and removing
a whole block would hide them.

Written 2026-08-16, reimplementing the script described in CHANGES.md Sect. 47 that was
lost between then and now; the defect it fixed had silently returned to the marked PDF.

Usage (see MANIFEST.md, "Rebuild the marked-changes PDF"):
    latexdiff old_submitted.tex new_revised.tex > aanda_marked.tex
    python3 strip_moved_floats.py aanda_marked.tex
    latexmk -pdf aanda_marked.tex
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# A float opened/closed inside a latexdiff deletion comment. Both figure and table, both
# starred and unstarred.
_OPEN = re.compile(r"%DIFDELCMD < \\begin\{(figure|table)\*?\}")
_CLOSE = re.compile(r"%DIFDELCMD < \\end\{(figure|table)\*?\}[^\n]*\n?")
_GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_LABEL = re.compile(r"\\label\{([^}]+)\}")


def _uncommented(text: str) -> str:
    """Drop latexdiff's commented-out copies of deleted source."""
    return re.sub(r"%DIFDELCMD <[^\n]*", "", text)


def _float_spans(body: str) -> list[tuple[int, int]]:
    """Byte ranges of every commented-out float, outermost only, non-overlapping."""
    spans: list[tuple[int, int]] = []
    pos = 0
    while (opener := _OPEN.search(body, pos)) is not None:
        closer = _CLOSE.search(body, opener.end())
        if closer is None:  # unbalanced: leave the rest of the file alone
            break
        spans.append((opener.start(), closer.end()))
        pos = closer.end()
    return spans


def strip(text: str) -> tuple[str, list[str]]:
    """Return (stripped text, human-readable log of what was removed and why not)."""
    # latexdiff's own preamble defines \DIFdelbegin/\let\includegraphics; never touch it.
    doc = text.index(r"\begin{document}")
    head, body = text[:doc], text[doc:]

    removed: list[str] = []
    log: list[str] = []
    keep: list[tuple[int, int]] = []

    for start, end in _float_spans(body):
        span = body[start:end]
        graphics = _GRAPHIC.findall(span)
        labels = _LABEL.findall(span)
        # "Alive" means alive in *typeset* text. A genuinely deleted float survives in the
        # diff only as %DIFDELCMD comments, so searching the raw text would find its own
        # corpse and call it alive.
        outside = _uncommented(body[:start] + body[end:])

        dead_g = [g for g in graphics if not _GRAPHIC_alive(g, outside)]
        dead_l = [l for l in labels if ("\\label{%s}" % l) not in outside]

        what = f"{graphics or '(sin grafico)'} {labels or '(sin label)'}"
        if dead_g or dead_l:
            log.append(f"  CONSERVADO {what} -- borrado real: {dead_g + dead_l}")
            continue
        keep.append((start, end))
        removed.append(what)
        log.append(f"  eliminado  {what}")

    for start, end in reversed(keep):
        body = body[:start] + body[end:]

    return head + body, log


def _GRAPHIC_alive(path: str, rest: str) -> bool:
    return re.search(_GRAPHIC.pattern.replace("([^}]+)", re.escape(path)), rest) is not None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    target = Path(argv[1])
    text = target.read_text()
    out, log = strip(text)
    print("\n".join(log) or "  (ningun float movido encontrado)")
    n_removed = sum(1 for line in log if line.strip().startswith("eliminado"))
    n_kept = len(log) - n_removed
    print(f"floats movidos eliminados: {n_removed} | borrados reales conservados: {n_kept}")
    target.write_text(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
