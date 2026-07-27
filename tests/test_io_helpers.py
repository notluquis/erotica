"""Regression tests for masked-column handling in the EROTICA helpers.

Backlog #5 (scientifically critical): masked integer identifier columns such as
Gaia ``source_id`` were promoted to ``float64`` when unmasked/filled.  Gaia
``source_id`` values exceed ``2**53`` and therefore cannot be represented
exactly as ``float64``, so the cast silently corrupted every identifier and
broke all downstream crossmatches (CDS table, Sagitta join).

These tests exercise both duplicated fill helpers (io + preprocess) and assert
that surviving 64-bit identifiers keep exact integer precision while ordinary
float measurement columns are still filled with NaN as before.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.table import MaskedColumn, QTable

import erotica
from erotica.io._helpers import compute_valid_source_counts, handle_masked_columns


def _load_preprocess_helpers():
    """Load ``erotica.preprocess._helpers`` without executing the package
    ``__init__``.

    ``erotica.preprocess.__init__`` eagerly imports ``preprocessor``, which needs
    the optional external ``zero_point`` package that is not installed in the
    test environment.  The masked-fill logic under test lives in ``_helpers`` and
    only depends on ``_constants``, so we import it directly via a bare package
    stub (relative ``from ._constants import`` still resolves through ``__path__``).
    """
    pkg_name = "erotica.preprocess"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(Path(erotica.__file__).parent / "preprocess")]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg
    mod_name = "erotica.preprocess._helpers"
    spec = importlib.util.spec_from_file_location(
        mod_name, Path(erotica.__file__).parent / "preprocess" / "_helpers.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


fill_missing_values = _load_preprocess_helpers().fill_missing_values

# Non-power-of-two 64-bit identifiers.  Both exceed 2**53 and are *not* exactly
# representable as float64, so a float promotion demonstrably corrupts them.
# (2**62 == 4611686018427387904 would survive a float roundtrip because it is a
# power of two -- deliberately avoided so the test actually catches the bug.)
SOURCE_ID_A = 4611686018427387905  # 2**62 + 1
SOURCE_ID_B = 5977390935251699843  # realistic Gaia DR3 source_id


def _sanity_precision_loss() -> None:
    """The chosen IDs must lose precision under a float64 cast, else the test
    would pass even on the buggy code."""
    for value in (SOURCE_ID_A, SOURCE_ID_B):
        assert int(np.float64(value)) != value, (
            f"{value} is exactly representable as float64; pick a value that is "
            "not, otherwise the regression test cannot catch the corruption bug."
        )


def _make_table() -> QTable:
    """A masked table with a big-int ``source_id`` and a masked float column."""
    table = QTable()
    table["source_id"] = MaskedColumn(
        data=[SOURCE_ID_A, SOURCE_ID_B, 123],
        mask=[False, False, True],
        dtype=np.int64,
    )
    table["parallax"] = MaskedColumn(
        data=[1.5, 2.5, 3.5],
        mask=[False, True, False],
        dtype=float,
    )
    return table


@pytest.mark.parametrize(
    "fill_func",
    [handle_masked_columns, fill_missing_values],
    ids=["io.handle_masked_columns", "preprocess.fill_missing_values"],
)
def test_masked_integer_ids_preserve_exact_precision(fill_func):
    _sanity_precision_loss()
    table = _make_table()

    fill_func(table)

    source_id = table["source_id"]

    # (a) The identifier column must NOT have been promoted to float64.
    assert not np.issubdtype(np.asarray(source_id).dtype, np.floating), (
        f"source_id was promoted to {np.asarray(source_id).dtype}; 64-bit "
        "identifiers lose integer precision as float64."
    )

    # (a) Surviving identifiers are preserved EXACTLY -- integer equality on the
    # big values.  On the old (buggy) code SOURCE_ID_A -> 4611686018427387904,
    # so this int comparison fails.
    assert int(source_id[0]) == SOURCE_ID_A
    assert int(source_id[1]) == SOURCE_ID_B
    assert int(source_id[0]) - SOURCE_ID_A == 0  # no off-by-precision drift

    # (b) The ordinary float measurement column is still filled with NaN.
    parallax = np.asarray(table["parallax"], dtype=float)
    assert parallax[0] == 1.5
    assert np.isnan(parallax[1])  # was masked
    assert parallax[2] == 3.5


def test_compute_valid_source_counts_survives_non_float_ids():
    """``compute_valid_source_counts`` must not assume ``source_id`` is float:
    after the fix it can be integer (all present) or object (masked -> NaN).

    Guards the object/NaN path created by the fill fix, where ``np.isnan`` would
    otherwise raise a TypeError on object dtype."""
    table = _make_table()
    handle_masked_columns(table)  # source_id -> object (one masked entry)

    counts = compute_valid_source_counts(table)  # must not raise

    # Two of the three source_ids are present; the masked one counts as missing.
    assert counts["Gaia IDs"] == 2
