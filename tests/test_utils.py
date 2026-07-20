"""Tests for ``cosmic.utils.utils``.

The module's sole public function is :func:`compare_datasets`, a print-only
diagnostic that compares ``source_id`` overlap (and per-column value
discrepancies) across datasets.  Its documented contract, confirmed against the
source:

* Accepts exactly 1, 2, or 4 QTable datasets; any other arity (0, 3, 5, ...)
  raises ``ValueError`` with message ``"Only 1, 2, or 4 datasets supported,
  got N."``.
* Always returns ``None`` -- it communicates entirely through ``print`` (stdout).
* 1 dataset: prints a "no comparison" notice and does nothing else.
* 2 datasets: prints ``Overlap between datasets: N`` plus the two "Missing in
  dataX" counts, then per-shared-id per-column discrepancy lines.
* 4 datasets: interpreted positionally as
  ``(good, bad, good_test, bad_test)``; reports good/bad overlap and the
  program-vs-test missing/common counts for each label.

Progress bars are emitted by ``tqdm`` on stderr, so the assertions below read
``capsys.readouterr().out`` (stdout only).
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import QTable

from cosmic.utils.utils import compare_datasets


def _table(source_ids, values):
    """Build a tiny QTable with a ``source_id`` column and one value column."""
    return QTable({"source_id": list(source_ids), "val": list(values)})


# --------------------------------------------------------------------------- #
# Valid arities
# --------------------------------------------------------------------------- #
def test_single_dataset_returns_none_and_reports_no_comparison(capsys):
    data = _table([1, 2, 3], [10.0, 20.0, 30.0])

    result = compare_datasets(data)

    assert result is None
    out = capsys.readouterr().out
    assert "Only one dataset provided; no comparison can be performed." in out


def test_two_datasets_report_overlap_counts(capsys):
    # Share source_ids {2, 3}; data1 uniquely has 1, data2 uniquely has 4.
    data1 = _table([1, 2, 3], [10.0, 20.0, 30.0])
    data2 = _table([2, 3, 4], [20.0, 30.0, 40.0])

    result = compare_datasets(data1, data2)

    assert result is None
    out = capsys.readouterr().out
    assert "Overlap between datasets: 2" in out
    assert "Missing in data2: 1" in out  # {1} is in data1 but not data2
    assert "Missing in data1: 1" in out  # {4} is in data2 but not data1
    # Values agree on every shared id -> no discrepancy reported.
    assert "Discrepancy" not in out


def test_two_datasets_report_value_discrepancy(capsys):
    # Shared ids {2, 3}; the value for id 3 differs between the datasets.
    data1 = _table([1, 2, 3], [10.0, 20.0, 30.0])
    data2 = _table([2, 3, 4], [20.0, 99.0, 40.0])

    result = compare_datasets(data1, data2)

    assert result is None
    out = capsys.readouterr().out
    assert "Overlap between datasets: 2" in out
    assert "Discrepancy for source_id 3 in column val" in out


def test_four_datasets_no_good_bad_overlap(capsys):
    # (good, bad, good_test, bad_test) -- disjoint good/bad, test == program.
    good = _table([1, 2], [1.0, 2.0])
    bad = _table([3, 4], [3.0, 4.0])
    good_test = _table([1, 2], [1.0, 2.0])
    bad_test = _table([3, 4], [3.0, 4.0])

    result = compare_datasets(good, bad, good_test, bad_test)

    assert result is None
    out = capsys.readouterr().out
    assert "No overlap between good_data and bad_data." in out
    assert "Common source_ids in good_data: 2" in out
    assert "Common source_ids in bad_data: 2" in out


def test_four_datasets_flags_good_bad_overlap(capsys):
    # good and bad share source_id 2 -> the "Error: Overlap detected" branch.
    good = _table([1, 2], [1.0, 2.0])
    bad = _table([2, 3], [2.0, 3.0])
    good_test = _table([1, 2], [1.0, 2.0])
    bad_test = _table([2, 3], [2.0, 3.0])

    result = compare_datasets(good, bad, good_test, bad_test)

    assert result is None
    out = capsys.readouterr().out
    assert "Error: Overlap detected between good_data and bad_data: 1" in out


# --------------------------------------------------------------------------- #
# Invalid arities
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("count", [0, 3])
def test_invalid_arity_raises_value_error(count):
    data = _table([1, 2, 3], [10.0, 20.0, 30.0])
    datasets = [data] * count

    with pytest.raises(ValueError, match=r"Only 1, 2, or 4 datasets supported"):
        compare_datasets(*datasets)
