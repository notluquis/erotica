#!/usr/bin/env python3
"""Count tools/validation/ scripts that name a committed JSON result file.

Copied into the repository on 2026-10-26 so that paper/paper.md's count ("49 scripts ... 27 of
which commit their result as a JSON file") comes from a script a reader can rerun; it was written
for the author's research notes (joss-prep-2026-09-27, S3), where two cheaper proxies were tried
and rejected first:
  - `grep -l "json.dump("` gave 0 (no script calls the bare function that way in this
    codebase's style -- most write via a `.write_text(json.dumps(...))` pattern).
  - `grep -l "json.dump"` (no anchor) gave 36, but that also matches `json.dumps`, which
    a script can call without ever committing a result file (e.g. printing to stdout).
  - Matching each script's stem against a same-named `.json` file gave 25, an
    undercount: several scripts write to a JSON file with a different stem
    (b7_topk_reproduction.py -> b7_topk_results.json).

This script instead: for each top-level *.py under tools/validation/, extracts every
literal "...something.json" string in its source, and checks whether a tracked file
with that *basename* exists anywhere under tools/validation/ (recursively, so
subdirectory sidecars like isochrone_unbinned/*.json count). A script counts if at
least one such match is a real, committed file.

What it measures is "the source names a tracked .json", not that the JSON holds a falsification
criterion or a negative control; the paper claims only the former.

Usage: python3 tools/release/validation_json_count.py [<path-to-erotica-checkout>]
"""

import pathlib
import re
import subprocess
import sys


def git(cwd: str, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


def main() -> None:
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    tracked = set(git(repo, "ls-files", "tools/validation").splitlines())
    tracked_json_basenames = {pathlib.Path(p).name for p in tracked if p.endswith(".json")}

    scripts = sorted(
        p
        for p in tracked
        if p.startswith("tools/validation/")
        and p.endswith(".py")
        and "/" not in p[len("tools/validation/") :]
    )

    hits = []
    for s in scripts:
        src = (pathlib.Path(repo) / s).read_text(encoding="utf-8", errors="replace")
        names = set(re.findall(r'["\']([A-Za-z0-9_./-]+\.json)["\']', src))
        matched_basenames = {pathlib.Path(n).name for n in names} & tracked_json_basenames
        if matched_basenames:
            hits.append(s)

    print(f"top-level scripts: {len(scripts)}")
    print(f"scripts naming >=1 committed .json result: {len(hits)}")
    for s in hits:
        print(f"  {s}")


if __name__ == "__main__":
    main()
