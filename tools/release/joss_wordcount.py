#!/usr/bin/env python3
"""Count the words of ``paper/paper.md`` the way JOSS' bot does, and the way a reader would.

The bot runs ``cat paper.md | wc -w`` over the WHOLE file (``openjournals/gh-action-repo-checks``,
``checks.rb``): the YAML header, HTML comments, the image syntax and the figure caption all count.
For submissions of 2026 it warns above 1800 words and alerts above 1999. The second figure printed
here, with the header, comments and fenced code stripped, is NOT what the bot counts; it was the
only number this repository had until 2026-10-27, when it read 1749 for a file the bot counted as
1997. Gate on the first.

    python tools/release/joss_wordcount.py [paper/paper.md] [--check]

``--check`` exits 1 above the warning threshold (1800), 2 above the alert threshold (1999).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

WARN = 1800
ALERT = 1999


def bot_count(text: str) -> int:
    """Whitespace-separated tokens of the whole file: what ``cat paper.md | wc -w`` prints."""
    return len(text.split())


def stripped_count(text: str) -> int:
    """Body only: no YAML header, HTML comments, fenced code or ``# References`` heading."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            text = text[end + 5 :]
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"^#\s*References\s*$", "", text, flags=re.MULTILINE)
    return len(text.split())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paper", nargs="?", default="paper/paper.md", type=Path)
    ap.add_argument("--check", action="store_true", help="exit 1 above 1800, 2 above 1999")
    args = ap.parse_args(argv)
    text = args.paper.read_text(encoding="utf-8")
    bot, body = bot_count(text), stripped_count(text)
    print(f"{bot}\tbot method: whole file, `wc -w` (warns > {WARN}, alerts > {ALERT})")
    print(f"{body}\tstripped: body only, no header or comments (NOT what the bot counts)")
    if args.check and bot > ALERT:
        return 2
    if args.check and bot > WARN:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
