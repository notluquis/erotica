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

1. **Count at a fixed point, not at HEAD.** A tag if one exists; otherwise a specific commit
   sha (``--tag`` accepts either -- ``git log <ref>`` does not care). Frozen, citable, and the
   point the disclosure actually describes, so the numbers stop being a moving target and a
   referee can reproduce them. Between 2026-09-22 (when the paper was updated on ``dev``) and
   the ``v0.2.0`` tag landing, the paper cited a commit sha instead -- ``is_commit_sha()``
   picks the "as of commit <sha>" sentence template over "in the <tag> release" automatically,
   and the same template still applies to any future work-in-progress state ahead of a tag.
2. **Compute them with a script instead of by hand**, so "is the paper still true?" is a
   command rather than an act of memory. ``--check`` answers exactly that question.

Usage
-----
    python tools/release/ai_disclosure_counts.py                       # counts at v0.3.0
    python tools/release/ai_disclosure_counts.py --tag v0.1.0          # counts at another tag
    python tools/release/ai_disclosure_counts.py --tag ee4bbe3aeaf0d   # counts at a commit sha
    python tools/release/ai_disclosure_counts.py --paragraph           # the prose, ready to paste
    python tools/release/ai_disclosure_counts.py --check paper/paper.md   # exit 1 if stale

``--check`` normalizes whitespace before matching, so it is insensitive to how the paragraph
is line-wrapped but not to a changed integer.

Categories (added 2026-10-26)
-----------------------------
Every commit reachable from the ref falls in exactly one, in this order:

* ``with_trailer``: carries a ``Co-Authored-By`` trailer; the only commits with a model.
* ``declared``: no trailer, listed in ``tools/release/ai_assisted_without_trailer.txt`` -- AI-assisted
  by the author's statement (2026-10-26), model not recorded. The list is read from the counted ref,
  not the working tree, so a count at a tag is reproducible; a ref without the file has none.
* ``merge_pr`` / ``merge_local``: merge commits without a trailer -- GitHub's "Merge pull request #"
  commits, and branch syncs made locally. Until 2026-10-26 these were lumped with "without trailer",
  where 38 merges inflated a count the paper described as housekeeping.
* ``other``: none of the above. The paper calls them early housekeeping, which is true of the seven
  dated up to ``EARLY_CUTOFF``; ``--check`` fails if a later one appears, because the sentence would
  then be false for it.

Pinning to a release
--------------------
A paper commit cannot state counts that include itself. The release procedure
(``RELEASE_INSTRUCTIONS.md``) counts at the parent of the commit that writes the paragraph, with
the "As of commit <sha>" template (``--tag <sha>``), and the release tag then goes on a descendant.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

TRAILER = "Co-Authored-By:"
DEFAULT_TAG = "v0.3.0"
DECLARED_PATH = "tools/release/ai_assisted_without_trailer.txt"
# The last of the seven early commits without a trailer (e372590, b4a0770 are 2026-07-26).
EARLY_CUTOFF = "2026-07-26"
_SHA40 = re.compile(r"^([0-9a-f]{40})\b")

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


def parse_declared(text: str) -> set[str]:
    """Full shas from the declared-assisted list; ``#`` lines and text after the sha are comments."""
    shas = set()
    for line in text.splitlines():
        m = _SHA40.match(line.strip())
        if m:
            shas.add(m.group(1))
    return shas


def read_declared(ref: str) -> set[str]:
    """The declared list as it is AT ``ref``; empty when the file did not exist there."""
    try:
        return parse_declared(_git("show", f"{ref}:{DECLARED_PATH}"))
    except subprocess.CalledProcessError:
        return set()


def trailer_models(body: str) -> list[str]:
    return [
        clean_model(line[len(TRAILER) :]) for line in body.splitlines() if line.startswith(TRAILER)
    ]


def classify(sha: str, parents: list[str], body: str, declared: set[str]) -> str:
    """One category per commit; see the module docstring for the order and why."""
    if trailer_models(body):
        return "with_trailer"
    if sha in declared:
        return "declared"
    if len(parents) > 1:
        subject = body.strip().splitlines()[0] if body.strip() else ""
        return "merge_pr" if subject.startswith("Merge pull request #") else "merge_local"
    return "other"


def validate_declared(declared: set[str]) -> list[str]:
    """A declared entry must exist, be a single-parent commit and carry no trailer."""
    problems = []
    for sha in sorted(declared):
        try:
            out = _git("log", "-1", "--pretty=format:%P%x00%B", sha)
        except subprocess.CalledProcessError:
            problems.append(f"{sha}: not a commit in this repository")
            continue
        parents, _, body = out.partition("\x00")
        if len(parents.split()) > 1:
            problems.append(f"{sha}: is a merge commit")
        if trailer_models(body):
            problems.append(f"{sha}: already carries a trailer")
    return problems


def collect(tag: str) -> dict:
    """Count commits, trailers, declared-assisted commits and merges reachable from ``tag``."""
    # ASCII unit (0x1f) and record (0x1e) separators: commit messages contain blank lines, so
    # anything line-oriented miscounts multi-paragraph bodies; and a NUL pair cannot be the record
    # separator once %P is a field, because a root commit's empty %P writes exactly that pair.
    raw = _git("log", tag, "--pretty=format:%H%x1f%P%x1f%as%x1f%B%x1e")
    records = [r for r in raw.split("\x1e") if r.strip()]

    declared = read_declared(tag)
    problems = validate_declared(declared)
    if problems:
        raise SystemExit(f"{DECLARED_PATH} at {tag}:\n  " + "\n  ".join(problems))

    models: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    declared_dates: list[str] = []
    late_other: list[str] = []
    for record in records:
        sha, parents, date, body = record.lstrip("\n").split("\x1f", 3)
        category = classify(sha, parents.split(), body, declared)
        categories[category] += 1
        if category == "with_trailer":
            models.update(trailer_models(body))
        elif category == "declared":
            declared_dates.append(date)
        elif category == "other" and date > EARLY_CUTOFF:
            late_other.append(f"{sha[:7]} {date}")

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
        "with_trailer": categories["with_trailer"],
        "without_trailer": total - categories["with_trailer"],
        "declared": categories["declared"],
        "declared_dates": (min(declared_dates), max(declared_dates)) if declared_dates else None,
        "merge_pr": categories["merge_pr"],
        "merge_local": categories["merge_local"],
        "other": categories["other"],
        "late_other": late_other,
        "models": models,
        # %aN applies .mailmap, so the tally shows one name per person.
        "authors": Counter(
            line for line in _git("log", tag, "--pretty=format:%aN").splitlines() if line
        ),
    }


def model_phrase(models: Counter[str]) -> str:
    """``Claude Opus 5 (105), Claude Opus 4.8 (38) and Claude Sonnet 4.6 (9)``."""
    parts = [f"{name} ({count})" for name, count in models.most_common()]
    if len(parts) < 2:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


_COMMIT_SHA = re.compile(r"^[0-9a-f]{7,40}$")


def is_commit_sha(tag: str) -> bool:
    """A branch that never gets a release tag (JOSS paper prep on ``dev`` before ``v0.2.0``
    exists, e.g.) still needs a citable, non-moving reference point. ``git log <ref>`` already
    works for either kind of ref -- this only decides which sentence template to render.
    """
    return bool(_COMMIT_SHA.match(tag))


def _date_range(dates: tuple[str, str] | None) -> str:
    """``("2026-10-19", "2026-10-20")`` -> ``19-20 October 2026``."""
    import datetime as _dt

    lo, hi = (_dt.date.fromisoformat(d) for d in dates)
    if lo == hi:
        return f"{lo.day} {lo:%B %Y}"
    if (lo.year, lo.month) == (hi.year, hi.month):
        return f"{lo.day}\u2013{hi.day} {hi:%B %Y}"
    return f"{lo.day} {lo:%B %Y} \u2013 {hi.day} {hi:%B %Y}"


def _lead(stats: dict) -> str:
    if is_commit_sha(stats["tag"]):
        return f"As of commit `{stats['tag'][:7]}`, {stats['with_trailer']} of the {stats['total']} commits"
    return f"Of the {stats['total']} commits in the {stats['tag']} release, {stats['with_trailer']}"


def paragraph(stats: dict) -> str:
    out = [
        f"{_lead(stats)} carry a `Co-Authored-By` trailer naming the model: "
        f"{model_phrase(stats['models'])}."
    ]
    if stats["declared"]:
        out.append(
            f"A further {stats['declared']}, from {_date_range(stats['declared_dates'])}, carry none but "
            f"were AI-assisted, with the model unrecorded; they are listed in `{DECLARED_PATH}`."
        )
    merges = stats["merge_pr"] + stats["merge_local"]
    early = "early notebook-removal and housekeeping work, not reconstructed as assisted."
    if merges:
        out.append(
            f"Of the rest, {merges} are merge commits ({stats['merge_pr']} of pull requests) and "
            f"{stats['other']} are {early}"
        )
    elif stats["other"]:
        out.append(f"The other {stats['other']} are {early}")
    return " ".join(out)


def report(stats: dict) -> str:
    lines = [
        f"tag                     {stats['tag']}",
        f"commits                 {stats['total']}",
        f"with Co-Authored-By     {stats['with_trailer']}",
        f"without                 {stats['without_trailer']}",
        f"  declared assisted     {stats['declared']}",
        f"  merges (PR / local)   {stats['merge_pr']} / {stats['merge_local']}",
        f"  other                 {stats['other']}"
        + (
            f"  <-- after {EARLY_CUTOFF}: {', '.join(stats['late_other'])}"
            if stats["late_other"]
            else ""
        ),
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


def _sentences(stats: dict) -> list[str]:
    """The paragraph cut at its sentence ends (none of its sentences contains '. ')."""
    return [part if part.endswith(".") else part + "." for part in paragraph(stats).split(". ")]


def check(stats: dict, path: Path) -> int:
    """Verify the paper still states these numbers. Returns a process exit code."""
    if not path.exists():
        print(f"FAIL: {path} does not exist", file=sys.stderr)
        return 1
    # Collapse whitespace so line wrapping in the Markdown source cannot cause a false failure.
    haystack = " ".join(path.read_text(encoding="utf-8").split())

    # Every sentence of the paragraph, so a stale category count fails as surely as a stale total.
    needles = {f"sentence {i + 1}": sentence for i, sentence in enumerate(_sentences(stats))}
    if stats["late_other"]:
        print(
            f"FAIL: commits without a trailer after {EARLY_CUTOFF} are neither declared nor merges: "
            f"{', '.join(stats['late_other'])}. Declare them in {DECLARED_PATH} or change the text.",
            file=sys.stderr,
        )
        return 1
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
