#!/usr/bin/env python3
"""Shrink only the tables that latexdiff made too wide, in the marked file only.

Marking a table where every cell changed puts the old value and the new value in each
cell, so the row is roughly twice as wide as the one in the clean manuscript. Table D.2 of
aa52082-24 overflows its column by 93.9pt that way -- about 3.3 cm of text hanging outside
the page, in a two-column layout where it lands on whatever is beside it.

The clean manuscript is untouched: those tables fit at natural width and shrinking them
there would be a defect, not a fix. This runs on `aanda_marked.tex` after latexdiff.

It wraps only what actually overflows, which is why it compiles first and reads the log
rather than wrapping every tabular it can find. Wrapping a table that already fits makes it
needlessly small, and a post-processor that degrades correct output is worse than none.

Usage, after latexdiff and strip_moved_floats.py (see MANIFEST.md):

    python3 fit_marked_tables.py aanda_marked.tex

Exit codes: 0 if nothing needed wrapping or the wrap succeeded, 1 on failure to parse.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

WRAP_OPEN = "\\resizebox{\\textwidth}{!}{%\n"
WRAP_CLOSE = "}%\n"
MARK = "% fit_marked_tables"


def overfull_line_ranges(log: str) -> list[tuple[int, int]]:
    """Line ranges LaTeX reported as overfull, widest first."""
    hits = []
    for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\) in paragraph at lines (\d+)--(\d+)", log):
        hits.append((float(m.group(1)), int(m.group(2)), int(m.group(3))))
    hits.sort(reverse=True)
    return [(a, b) for _, a, b in hits]


def enclosing_tabular(lines: list[str], lo: int, hi: int) -> tuple[int, int] | None:
    """Index range of the \\begin{tabular}...\\end{tabular} covering the given 1-based lines."""
    start = None
    for i in range(min(hi, len(lines)) - 1, -1, -1):
        if "\\begin{tabular}" in lines[i]:
            start = i
            break
    if start is None:
        return None
    for j in range(start, len(lines)):
        if "\\end{tabular}" in lines[j]:
            return start, j
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    target = Path(argv[1])
    if not target.exists():
        print(f"no existe: {target}")
        return 1

    stem = target.with_suffix("").name
    subprocess.run(["pdflatex", "-interaction=nonstopmode", target.name],
                   cwd=target.parent, capture_output=True, text=True)
    log_path = target.parent / f"{stem}.log"
    if not log_path.exists():
        print("no hay log; no se puede saber que se desborda")
        return 1

    ranges = overfull_line_ranges(log_path.read_text(errors="replace"))
    if not ranges:
        print("nada se desborda; ninguna tabla tocada")
        return 0

    lines = target.read_text().split("\n")
    wrapped = 0
    for lo, hi in ranges:
        span = enclosing_tabular(lines, lo, hi)
        if span is None:
            print(f"  desborde en lineas {lo}--{hi} fuera de una tabular; no se toca")
            continue
        a, b = span
        if any(MARK in lines[k] for k in (max(0, a - 1), a)):
            continue  # already wrapped on an earlier pass
        lines[a] = WRAP_OPEN.rstrip("\n") + MARK + "\n" + lines[a]
        lines[b] = lines[b] + "\n" + WRAP_CLOSE.rstrip("\n") + MARK
        wrapped += 1
        print(f"  tabla en lineas {a + 1}--{b + 1} envuelta en resizebox (desborde en {lo}--{hi})")

    if wrapped:
        target.write_text("\n".join(lines))
    print(f"tablas ajustadas: {wrapped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
