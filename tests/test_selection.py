"""Tests for :mod:`erotica.selection`.

These tests never touch the real optional dependencies. They exercise:

* the numerical/wiring logic via **injected fake** selection functions
  (``selection_function=``), so no import is needed;
* the lazy-import *error* path by forcing ``import X`` to fail
  (``sys.modules[name] = None``);
* the lazy-import *happy* path by injecting fake modules into ``sys.modules``,
  so the default-construction wiring (``DR3SelectionFunctionTCG()`` /
  ``HR24SelectionFunction()``) is covered without the real packages.

They do NOT verify the real selection-function numbers -- that requires the real
env (see integration notes).
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

from erotica import selection


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeGaiaSF:
    """Stand-in for gaiaunlimited DR3SelectionFunctionTCG: query(coords, gmag)."""

    def __init__(self, value=0.8):
        self.value = value
        self.last_gmag = None

    def query(self, coords, gmag):
        self.last_gmag = np.asarray(gmag, dtype=float)
        return np.full(len(self.last_gmag), self.value, dtype=float)


class _RecordingCensusModel:
    """Stand-in for hr_selection_function HR24SelectionFunction (callable)."""

    def __init__(self, prob=0.6):
        self.prob = prob
        self.call = None

    def __call__(self, density_or_coordinates, n_stars, median_parallax_error, threshold, mode="median"):
        self.call = dict(
            density_or_coordinates=density_or_coordinates,
            n_stars=np.asarray(n_stars),
            median_parallax_error=np.asarray(median_parallax_error),
            threshold=np.asarray(threshold),
            mode=mode,
        )
        return np.full(np.shape(np.atleast_1d(n_stars)), self.prob, dtype=float)


def _members():
    return QTable(
        {
            "ra": [10.0, 10.1, np.nan] * u.deg,
            "dec": [-20.0, -20.1, -20.2] * u.deg,
            "Gmag": [15.0, 18.0, 19.0] * u.mag,
            "parallax_error": [0.03, 0.05, 0.10] * u.mas,
            "probability": [0.9, 0.7, 0.4],
        }
    )


# ---------------------------------------------------------------------------
# attach_completeness_weights -- logic via injected fake
# ---------------------------------------------------------------------------
def test_attach_completeness_weights_columns_and_inverse():
    sf = _FakeGaiaSF(value=0.8)
    out = selection.attach_completeness_weights(_members(), selection_function=sf)

    assert "completeness" in out.colnames
    assert "completeness_weight" in out.colnames
    # Only the two finite rows were queried.
    assert sf.last_gmag.tolist() == [15.0, 18.0]
    # Finite rows: P = 0.8, weight = 1 / 0.8.
    assert out["completeness"][0] == pytest.approx(0.8)
    assert out["completeness_weight"][0] == pytest.approx(1.0 / 0.8)


def test_attach_completeness_weights_nan_rows_are_nan():
    out = selection.attach_completeness_weights(_members(), selection_function=_FakeGaiaSF())
    # The third row has NaN ra -> NaN completeness and NaN weight (never queried).
    assert np.isnan(out["completeness"][2])
    assert np.isnan(out["completeness_weight"][2])


def test_attach_completeness_weights_floor_keeps_weight_finite():
    sf = _FakeGaiaSF(value=0.0)  # zero completeness would divide by zero without the floor
    out = selection.attach_completeness_weights(
        _members(), selection_function=sf, min_completeness=1e-3
    )
    assert out["completeness_weight"][0] == pytest.approx(1.0 / 1e-3)
    assert np.isfinite(out["completeness_weight"][0])


def test_attach_completeness_weights_does_not_mutate_by_default():
    members = _members()
    selection.attach_completeness_weights(members, selection_function=_FakeGaiaSF())
    assert "completeness" not in members.colnames  # copy=True default


def test_attach_completeness_weights_missing_column_raises():
    members = _members()
    members.remove_column("Gmag")
    with pytest.raises(ValueError, match="Gmag"):
        selection.attach_completeness_weights(members, selection_function=_FakeGaiaSF())


# ---------------------------------------------------------------------------
# cluster_census_detectability -- logic via injected fake
# ---------------------------------------------------------------------------
def test_cluster_census_detectability_passes_expected_args():
    model = _RecordingCensusModel(prob=0.6)
    prob = selection.cluster_census_detectability(
        150, 0.05, 3.0, data_density=1.2e5, selection_function=model
    )
    assert prob == pytest.approx(np.array([0.6]))
    # Scalars are promoted to 1-element arrays; keyword wiring is correct.
    assert model.call["n_stars"].tolist() == [150]
    assert model.call["median_parallax_error"].tolist() == [0.05]
    assert model.call["threshold"].tolist() == [3.0]
    assert model.call["density_or_coordinates"].tolist() == [1.2e5]
    assert model.call["mode"] == "median"


def test_cluster_census_detectability_coordinates_passthrough():
    model = _RecordingCensusModel()
    sentinel = object()  # stands in for a SkyCoord; must be passed through untouched
    selection.cluster_census_detectability(
        [10, 20], [0.1, 0.2], [2.0, 2.5], coordinates=sentinel, selection_function=model
    )
    assert model.call["density_or_coordinates"] is sentinel
    assert model.call["n_stars"].tolist() == [10, 20]


def test_cluster_census_detectability_requires_exactly_one_source():
    model = _RecordingCensusModel()
    with pytest.raises(ValueError, match="exactly one"):
        selection.cluster_census_detectability(10, 0.1, 3.0, selection_function=model)
    with pytest.raises(ValueError, match="exactly one"):
        selection.cluster_census_detectability(
            10, 0.1, 3.0, data_density=1.0, coordinates=object(), selection_function=model
        )


def test_census_detectability_from_members_derives_inputs():
    model = _RecordingCensusModel()
    selection.census_detectability_from_members(
        _members(),
        3.0,
        data_density=1e5,
        probability_threshold=0.6,
        selection_function=model,
    )
    # probability >= 0.6 keeps two rows; median parallax error of {0.03, 0.05} = 0.04.
    assert model.call["n_stars"].tolist() == [2]
    assert model.call["median_parallax_error"].tolist() == pytest.approx([0.04])


# ---------------------------------------------------------------------------
# Lazy-import error paths (force ImportError)
# ---------------------------------------------------------------------------
def test_require_gaiaunlimited_error_message(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "gaiaunlimited", None)
    with pytest.raises(ImportError, match="pip install gaiaunlimited"):
        selection.default_gaia_selection_function()


def test_require_hr_selection_function_error_message(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "hr_selection_function", None)
    with pytest.raises(ImportError, match="pip install hr-selection-function"):
        selection.cluster_census_detectability(10, 0.1, 3.0, data_density=1.0)


# ---------------------------------------------------------------------------
# Lazy-import happy paths (inject fake modules into sys.modules)
# ---------------------------------------------------------------------------
def test_default_gaia_selection_function_wiring(monkeypatch):
    import sys

    fake_subpkg = types.ModuleType("gaiaunlimited.selectionfunctions")
    fake_subpkg.DR3SelectionFunctionTCG = lambda *a, **k: _FakeGaiaSF(0.75)
    fake_pkg = types.ModuleType("gaiaunlimited")
    fake_pkg.selectionfunctions = fake_subpkg
    monkeypatch.setitem(sys.modules, "gaiaunlimited", fake_pkg)
    monkeypatch.setitem(sys.modules, "gaiaunlimited.selectionfunctions", fake_subpkg)

    # No selection_function passed -> exercises _require_gaiaunlimited + default ctor.
    out = selection.attach_completeness_weights(_members())
    assert out["completeness"][0] == pytest.approx(0.75)


def test_hr_selection_function_wiring(monkeypatch):
    import sys

    fake_hr = types.ModuleType("hr_selection_function")
    fake_hr.HR24SelectionFunction = lambda *a, **k: _RecordingCensusModel(0.42)
    monkeypatch.setitem(sys.modules, "hr_selection_function", fake_hr)

    prob = selection.cluster_census_detectability(100, 0.05, 3.0, data_density=1e5)
    assert prob == pytest.approx(np.array([0.42]))

