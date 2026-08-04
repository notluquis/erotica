#!/usr/bin/env python3
"""Recompute the numbers in ``paper/paper.md``'s AI-usage disclosure from git history.

Why this exists
---------------
The disclosure paragraph used to quote counts measured at ``HEAD`` of the default branch.
``HEAD`` moves on every commit, so the paragraph was wrong the moment it was written and got
wronger: it stated 158 commits / 151 trailers / Opus 5 = 104 while the branch had already
moved past that, and a later hand-check produced yet another triple (184 / 177 / 130) that
no longer reproduces either -- the branch is at 186 as of 2026-08-04.

Two things fix that, and both matter:

1. **Count at a tag, not at HEAD.** ``v0.1.0`` is frozen, citable, and the version the JOSS
   submission actually describes, so the numbers stop being a moving target and a referee can
   reproduce them.
2. **Compute them with a script instead of by hand**, so "is the paper still true?" is a
   command rather than an act of memory. ``--check`` answers exactly that question.

Usage
-----
    python tools/release/ai_disclosure_counts.py                 # counts at v0.1.0
    python tools/release/ai_disclosure_counts.py --tag v0.2.0    # counts at another tag
    python tools/release/ai_disclosure_counts.py --paragraph     # the prose, ready to paste
    python tools/release/ai_disclosure_counts.py --check paper/paper.md   # exit 1 if stale

``--check`` normalises whitespace before matching, so it is insensitive to how the paragraph
is line-wrapped but not to a changed integer.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

TRAILER = "Co-Authored-By:"
DEFAULT_TAG = "v0.1.0"

# Trailers are written as ``Claude Opus 5 (1M context) <noreply@anthropic.com>``; the paper
# names the model as ``Claude Opus 5``. Strip the address and any parenthetical so the script's
# output maps one-to-one onto the prose instead of needing a mental translation step.
_ADDRESS = re.compile(r"\s*<[^>]*>\s*$")
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*$")


def _git(*args: str) -> str:
    """Run a read-only git command from the repository root."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return result.stdout


def clean_model(raw: str) -> str:
    """``Claude Opus 5 (1M context) <noreply@anthropic.com>`` -> ``Claude Opus 5``."""
    return _PARENTHETICAL.sub("", _ADDRESS.sub("", raw.strip())).strip()


def collect(tag: str) -> dict:
    """Count commits, trailers and per-model attributions reachable from ``tag``."""
    # %x00 as the record separator: commit messages contain blank lines, so anything
    # line-oriented miscounts multi-paragraph bodies.
    raw = _git("log", tag, "--pretty=format:%H%x00%B%x00%x00")
    records = [r for r in raw.split("\x00\x00") if r.strip()]

    models: Counter[str] = Counter()
    with_trailer = 0
    for record in records:
        _, _, body = record.partition("\x00")
        found = [
            clean_model(line[len(TRAILER) :])
            for line in body.splitlines()
            if line.startswith(TRAILER)
        ]
        if found:
            with_trailer += 1
            models.update(found)

    total = len(records)
    # Cross-check against git's own count. If these disagree the parsing is wrong, and a
    # silently wrong number here is exactly the failure this script exists to prevent.
    expected_total = int(_git("rev-list", "--count", tag).strip())
    if total != expected_total:
        raise SystemExit(
            f"parse error: counted {total} commits, `git rev-list --count {tag}` says "
            f"{expected_total}"
        )

    return {
        "tag": tag,
        "total": total,
        "with_trailer": with_trailer,
        "without_trailer": total - with_trailer,
        "models": models,
        "authors": Counter(
            line for line in _git("log", tag, "--pretty=format:%an").splitlines() if line
        ),
    }


def model_phrase(models: Counter[str]) -> str:
    """``Claude Opus 5 (105), Claude Opus 4.8 (38) and Claude Sonnet 4.6 (9)``."""
    parts = [f"{name} ({count})" for name, count in models.most_common()]
    if len(parts) < 2:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def paragraph(stats: dict) -> str:
    return (
        f"Of the {stats['total']} commits in the {stats['tag']} release, "
        f"{stats['with_trailer']} carry a `Co-Authored-By` trailer naming the model: "
        f"{model_phrase(stats['models'])}."
    )


def report(stats: dict) -> str:
    lines = [
        f"tag                     {stats['tag']}",
        f"commits                 {stats['total']}",
        f"with Co-Authored-By     {stats['with_trailer']}",
        f"without                 {stats['without_trailer']}",
        "models:",
    ]
    lines += [f"  {name:<24} {count}" for name, count in stats["models"].most_common()]
    lines.append("authors:")
    lines += [f"  {name:<24} {count}" for name, count in stats["authors"].most_common()]
    attributed = sum(stats["models"].values())
    lines.append(
        f"consistency             {attributed} attributions vs {stats['with_trailer']} "
        f"commits with a trailer"
        + ("" if attributed == stats["with_trailer"] else "  <-- MISMATCH")
    )
    return "\n".join(lines)


def check(stats: dict, path: Path) -> int:
    """Verify the paper still states these numbers. Returns a process exit code."""
    if not path.exists():
        print(f"FAIL: {path} does not exist", file=sys.stderr)
        return 1
    # Collapse whitespace so line wrapping in the Markdown source cannot cause a false failure.
    haystack = " ".join(path.read_text(encoding="utf-8").split())

    needles = {
        "commit total": f"Of the {stats['total']} commits in the {stats['tag']} release",
        "trailer count": f"{stats['with_trailer']} carry a `Co-Authored-By` trailer",
        "model breakdown": model_phrase(stats["models"]),
    }
    failures = [label for label, needle in needles.items() if needle not in haystack]
    for label in failures:
        print(f"FAIL: {path} does not state the current {label}", file=sys.stderr)
        print(f"      expected: {needles[label]}", file=sys.stderr)
    if failures:
        print(
            "\nRegenerate with: python tools/release/ai_disclosure_counts.py --paragraph",
            file=sys.stderr,
        )
        return 1
    print(f"OK: {path} matches git history at {stats['tag']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default=DEFAULT_TAG, help=f"git tag (default {DEFAULT_TAG})")
    parser.add_argument("--paragraph", action="store_true", help="print the prose sentence")
    parser.add_argument("--check", metavar="FILE", help="verify FILE states these numbers")
    args = parser.parse_args()

    try:
        stats = collect(args.tag)
    except subprocess.CalledProcessError as exc:
        print(f"git failed: {exc.stderr.strip()}", file=sys.stderr)
        return 1

    if args.check:
        return check(stats, Path(args.check))
    print(paragraph(stats) if args.paragraph else report(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
