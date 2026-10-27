"""The JOSS word counter must count what the bot counts: the whole file, comments included."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "joss_wordcount", ROOT / "tools" / "release" / "joss_wordcount.py"
)
jw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jw)

PAPER = "---\ntitle: a b\n---\n\n# Summary\n\none two three <!-- four five -->\n\n# References\n"


def test_the_bot_method_counts_header_and_comments():
    assert jw.bot_count(PAPER) == len(PAPER.split())
    assert jw.bot_count(PAPER) == 16


def test_the_stripped_count_drops_header_comments_and_references_heading():
    assert jw.stripped_count(PAPER) == 5  # "#", "Summary", "one", "two", "three"


def test_check_exit_codes_follow_the_bot_thresholds(tmp_path):
    f = tmp_path / "p.md"
    f.write_text("w " * jw.WARN, encoding="utf-8")
    assert jw.main([str(f), "--check"]) == 0
    f.write_text("w " * (jw.WARN + 1), encoding="utf-8")
    assert jw.main([str(f), "--check"]) == 1
    f.write_text("w " * (jw.ALERT + 1), encoding="utf-8")
    assert jw.main([str(f), "--check"]) == 2


def test_the_cli_runs_under_python_oo_where_the_docstring_is_gone():
    # ``python -OO`` strips docstrings, so ``__doc__`` is None and ``__doc__.split`` raised.
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "-OO", str(ROOT / "tools" / "release" / "joss_wordcount.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


def test_a_utf8_bom_does_not_hide_the_yaml_header_from_the_stripped_count(tmp_path, capsys):
    f = tmp_path / "p.md"
    f.write_bytes(b"\xef\xbb\xbf" + PAPER.encode("utf-8"))
    assert jw.main([str(f)]) == 0
    assert capsys.readouterr().out.splitlines()[1].startswith("5\t")
