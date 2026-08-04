"""Tests for ``erotica.utils.utils``.

The module's sole public function is :func:`compare_datasets`, which compares
``source_id`` overlap (and per-column value discrepancies) across datasets. Its
contract, confirmed against the source:

* Accepts exactly 1, 2, or 4 QTable datasets; any other arity (0, 3, 5, ...)
  raises ``ValueError`` with message ``"Only 1, 2, or 4 datasets supported,
  got N."``.
* **Returns a dict** describing what it compared, and *also* prints the report
  unless ``verbose=False``.
* 1 dataset: prints a "no comparison" notice and returns an empty
  ``"comparisons"``, because nothing was compared.
* 2 datasets: prints ``Overlap between datasets: N`` plus the two "Missing in
  dataX" counts, then per-shared-id per-column discrepancy lines; returns them
  under ``comparisons["data1_vs_data2"]``.
* 4 datasets: interpreted positionally as
  ``(good, bad, good_test, bad_test)``; reports good/bad overlap and the
  program-vs-test missing/common counts for each label, and returns them under
  ``"good_bad_overlap"`` and ``comparisons["good"]`` / ``comparisons["bad"]``.

Progress bars are emitted by ``tqdm`` on stderr, so the stdout assertions below
read ``capsys.readouterr().out``.

.. warning::
   **The ``assert result is not None`` lines and the structure assertions that
   follow them are not redundant boilerplate -- do not delete them.**

   Until 2026-08-04 :func:`compare_datasets` returned ``None`` and communicated
   only through ``print``, so nothing downstream could act on a comparison
   without scraping stdout. Five tests in this module *hard-asserted*
   ``result is None``: they were the only assertions in the whole suite that
   pinned a defect in place rather than a behaviour, and they would have failed
   the moment anyone fixed it. Rewriting them to assert the returned structure
   IS the mutation test for that fix -- restore the ``return`` statements to
   bare ``return``/``return None`` and every one of them goes red.
"""

from __future__ import annotations

import pytest
from astropy.table import QTable

from erotica.utils.utils import compare_datasets


def _table(source_ids, values):
    """Build a tiny QTable with a ``source_id`` column and one value column."""
    return QTable({"source_id": list(source_ids), "val": list(values)})


# --------------------------------------------------------------------------- #
# Valid arities
# --------------------------------------------------------------------------- #
def test_single_dataset_reports_no_comparison_and_returns_an_empty_record(capsys):
    """One dataset compares against nothing, and the record must say so rather than
    being absent -- an empty ``comparisons`` is a result, ``None`` was not."""
    data = _table([1, 2, 3], [10.0, 20.0, 30.0])

    result = compare_datasets(data)

    assert result is not None, "compare_datasets must report through its return value"
    assert result["n_datasets"] == 1
    assert result["comparisons"] == {}
    out = capsys.readouterr().out
    assert "Only one dataset provided; no comparison can be performed." in out


def test_two_datasets_report_overlap_counts(capsys):
    # Share source_ids {2, 3}; data1 uniquely has 1, data2 uniquely has 4.
    data1 = _table([1, 2, 3], [10.0, 20.0, 30.0])
    data2 = _table([2, 3, 4], [20.0, 30.0, 40.0])

    result = compare_datasets(data1, data2)

    assert result is not None
    pair = result["comparisons"]["data1_vs_data2"]
    # The ids themselves, not just their counts: a count can be right while the
    # membership is wrong, and set algebra is the whole job of this branch.
    assert pair["overlap"] == [2, 3]
    assert pair["missing_in_right"] == [1]  # {1} is in data1 but not data2
    assert pair["missing_in_left"] == [4]  # {4} is in data2 but not data1
    assert pair["discrepancies"] == []

    out = capsys.readouterr().out
    assert "Overlap between datasets: 2" in out
    assert "Missing in data2: 1" in out
    assert "Missing in data1: 1" in out
    # Values agree on every shared id -> no discrepancy reported.
    assert "Discrepancy" not in out


def test_two_datasets_report_value_discrepancy(capsys):
    # Shared ids {2, 3}; the value for id 3 differs between the datasets.
    data1 = _table([1, 2, 3], [10.0, 20.0, 30.0])
    data2 = _table([2, 3, 4], [20.0, 99.0, 40.0])

    result = compare_datasets(data1, data2)

    assert result is not None
    (found,) = result["comparisons"]["data1_vs_data2"]["discrepancies"]
    assert found["source_id"] == 3
    assert found["column"] == "val"
    assert float(found["left"][0]) == 30.0
    assert float(found["right"][0]) == 99.0

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

    assert result is not None
    assert result["good_bad_overlap"] == []  # disjoint fidelity splits: the healthy case
    assert result["comparisons"]["good"]["overlap"] == [1, 2]
    assert result["comparisons"]["bad"]["overlap"] == [3, 4]
    for label in ("good", "bad"):
        assert result["comparisons"][label]["discrepancies"] == []
        assert result["comparisons"][label]["missing_in_left"] == []
        assert result["comparisons"][label]["missing_in_right"] == []

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

    assert result is not None
    assert result["good_bad_overlap"] == [2]

    out = capsys.readouterr().out
    assert "Error: Overlap detected between good_data and bad_data: 1" in out


def test_verbose_false_silences_stdout_without_losing_the_result(capsys):
    """The printing is now optional; the answer is not.

    ``verbose`` exists only so the pre-2026-08-04 interactive behaviour keeps
    working -- the report was print-only, and anything that scraped stdout must not
    break. Turning it off must leave the returned structure untouched, which is the
    property that makes this function usable inside a pipeline at all.
    """
    data1 = _table([1, 2, 3], [10.0, 20.0, 30.0])
    data2 = _table([2, 3, 4], [20.0, 99.0, 40.0])

    loud = compare_datasets(data1, data2, verbose=True)
    capsys.readouterr()
    quiet = compare_datasets(data1, data2, verbose=False)

    assert capsys.readouterr().out == ""
    quiet_pair = quiet["comparisons"]["data1_vs_data2"]
    loud_pair = loud["comparisons"]["data1_vs_data2"]
    for key in ("overlap", "missing_in_left", "missing_in_right"):
        assert quiet_pair[key] == loud_pair[key]
    assert len(quiet_pair["discrepancies"]) == len(loud_pair["discrepancies"]) == 1


# --------------------------------------------------------------------------- #
# Invalid arities
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("count", [0, 3])
def test_invalid_arity_raises_value_error(count):
    data = _table([1, 2, 3], [10.0, 20.0, 30.0])
    datasets = [data] * count

    with pytest.raises(ValueError, match=r"Only 1, 2, or 4 datasets supported"):
        compare_datasets(*datasets)
