"""The AI-usage disclosure counter's classification, on synthetic records.

Live ``git log`` counts are not tested here: CI checks out a shallow clone, where the count is
wrong by construction. What is tested is what decides the numbers: which category a commit falls
in, how the declared list is parsed, and that ``--check`` fails on a stale category count.
"""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ai_disclosure_counts", ROOT / "tools" / "release" / "ai_disclosure_counts.py"
)
adc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adc)

SHA_A = "a" * 40
SHA_B = "b" * 40
TRAILER = "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"


class TestClassify:
    def test_a_trailer_wins_even_on_a_merge(self):
        body = f"Merge main into dev\n\n{TRAILER}\n"
        assert adc.classify(SHA_A, ["p1", "p2"], body, set()) == "with_trailer"

    def test_a_declared_commit_without_trailer(self):
        assert adc.classify(SHA_A, ["p1"], "fix: something\n", {SHA_A}) == "declared"

    def test_a_github_pull_request_merge(self):
        body = "Merge pull request #42 from notluquis/branch\n\ntext\n"
        assert adc.classify(SHA_A, ["p1", "p2"], body, set()) == "merge_pr"

    def test_a_local_merge(self):
        body = "Merge origin/dev into feat/mode-search\n"
        assert adc.classify(SHA_A, ["p1", "p2"], body, set()) == "merge_local"

    def test_an_undeclared_single_parent_commit_without_trailer(self):
        assert adc.classify(SHA_B, ["p1"], "deleted some trash\n", {SHA_A}) == "other"


def test_parse_declared_reads_shas_and_skips_comments():
    text = f"# header {SHA_B}\n\n{SHA_A}  2026-10-19 subject with words\n  {SHA_B}\nnot-a-sha\n"
    assert adc.parse_declared(text) == {SHA_A, SHA_B}


def test_the_committed_declared_list_has_the_authors_thirteen_commits():
    text = (ROOT / adc.DECLARED_PATH).read_text(encoding="utf-8")
    shas = adc.parse_declared(text)
    assert len(shas) == 13
    assert all(len(s) == 40 for s in shas)


def _stats(**over):
    stats = {
        "tag": "v0.3.0",
        "total": 560,
        "with_trailer": 500,
        "without_trailer": 60,
        "declared": 13,
        "declared_dates": ("2026-10-19", "2026-10-20"),
        "merge_pr": 33,
        "merge_local": 7,
        "other": 7,
        "late_other": [],
        "models": Counter({"Claude Opus 5": 301, "Claude Sonnet 5.5": 14}),
        "authors": Counter(),
    }
    stats.update(over)
    return stats


def test_the_paragraph_states_every_category():
    text = adc.paragraph(_stats())
    assert "Of the 560 commits in the v0.3.0 release, 500 carry" in text
    assert "Claude Sonnet 5.5 (14)" in text
    assert "A further 13, from 19–20 October 2026, carry none" in text
    assert "40 are merge commits (33 of pull requests) and 7 are early" in text


def test_check_fails_when_a_category_count_is_stale(tmp_path):
    paper = tmp_path / "paper.md"
    paper.write_text(adc.paragraph(_stats()), encoding="utf-8")
    assert adc.check(_stats(), paper) == 0
    # Same total and trailers; one declared commit fewer, one more "other".
    assert adc.check(_stats(declared=12, other=8), paper) == 1


def test_check_fails_on_an_unexplained_late_commit(tmp_path):
    paper = tmp_path / "paper.md"
    paper.write_text(adc.paragraph(_stats()), encoding="utf-8")
    assert adc.check(_stats(late_other=["1234567 2026-11-01"]), paper) == 1


@pytest.mark.parametrize(
    ("dates", "expected"),
    [
        (("2026-10-19", "2026-10-20"), "19–20 October 2026"),
        (("2026-10-19", "2026-10-19"), "19 October 2026"),
    ],
)
def test_date_range(dates, expected):
    assert adc._date_range(dates) == expected


def test_pinned_sha_reads_the_commit_the_paper_pins_across_line_wraps():
    text = "Tools. As of\ncommit `9e61814`, 486 of the 544 commits carry"
    assert adc.pinned_sha(text) == "9e61814"


def test_pinned_sha_is_none_when_the_paper_pins_a_tag():
    assert adc.pinned_sha("Of the 366 commits in the v0.2.0 release, 359 carry") is None
