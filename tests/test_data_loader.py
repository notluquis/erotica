"""Tests for the COSMIC data loader and its column-alias / collection helpers.

Covers:
  * ``resolve_alias``            -- canonical-name / alias resolution + case-sensitivity
  * ``map_requested_columns``    -- canonical -> concrete-name mapping (+ missing set)
  * ``collect_requested_columns``-- expansion of systems/distances/zp/flux/fidelity/prob
  * ``DataLoader``               -- end-to-end ECSV load, column subsetting via aliases,
                                    and the reporting helpers.

``handle_masked_columns`` is deliberately NOT tested here -- it already has dedicated
regression coverage in ``tests/test_io_helpers.py``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from astropy.table import QTable

from cosmic.io._constants import (
    FLUX_ERROR_COLUMNS,
    GAIA_DISTANCE_COLUMNS,
    PHOTOMETRIC_SYSTEMS,
    ZP_COLUMNS,
)
from cosmic.io._helpers import (
    collect_requested_columns,
    map_requested_columns,
    resolve_alias,
)
from cosmic.io.loader import DataLoader

# Non-power-of-two 64-bit identifier that is NOT exactly representable as
# float64: it survives an ECSV round-trip only because the loader keeps
# ``source_id`` as int64 (never promotes it to float).
SOURCE_ID_A = 4611686018427387905  # 2**62 + 1
SOURCE_ID_B = 5977390935251699843  # realistic Gaia DR3 source_id


# ---------------------------------------------------------------------------
# resolve_alias
# ---------------------------------------------------------------------------
class TestResolveAlias:
    def test_canonical_name_present_returns_it(self):
        # Only the canonical column is available -> it must be returned.
        assert resolve_alias({"Gmag"}, "Gmag") == "Gmag"

    def test_only_alias_present_returns_alias(self):
        # The canonical "Gmag" is absent but its Gaia archive alias is present.
        assert resolve_alias({"phot_g_mean_mag"}, "Gmag") == "phot_g_mean_mag"

    def test_nothing_present_returns_none(self):
        # Documented behaviour (loader.py relies on this): returns None, no raise.
        assert resolve_alias({"unrelated_col"}, "Gmag") is None

    def test_is_case_sensitive(self):
        # Backlog note: resolve_alias is case-sensitive. A lowercase "gmag" must
        # NOT satisfy a request for canonical "Gmag" -> None, not "gmag".
        assert resolve_alias({"gmag"}, "Gmag") is None

    def test_default_candidate_when_canonical_absent_from_alias_map(self):
        # With a custom alias map lacking the key, candidates default to the
        # canonical name itself.
        assert resolve_alias({"foo"}, "foo", aliases={}) == "foo"
        assert resolve_alias({"bar"}, "foo", aliases={}) is None

    def test_custom_alias_map_overrides_default(self):
        custom = {"mag": {"mag", "MYMAG"}}
        assert resolve_alias({"MYMAG"}, "mag", aliases=custom) == "MYMAG"


# ---------------------------------------------------------------------------
# map_requested_columns
# ---------------------------------------------------------------------------
class TestMapRequestedColumns:
    def test_resolves_aliases_and_reports_missing(self):
        available = ["phot_g_mean_mag", "parallax", "ra"]
        requested = ["Gmag", "parallax", "e_Gmag"]

        present, missing = map_requested_columns(requested, available)

        # "Gmag" resolves through its alias; "parallax" matches directly.
        assert present == {"Gmag": "phot_g_mean_mag", "parallax": "parallax"}
        # "e_Gmag" has no matching concrete column -> reported missing.
        assert missing == ["e_Gmag"]

    def test_normalize_names_false_disables_alias_resolution(self):
        available = ["phot_g_mean_mag", "parallax"]
        requested = ["Gmag", "parallax"]

        present, missing = map_requested_columns(
            requested, available, normalize_names=False
        )

        # Without normalization "Gmag" is matched literally (absent) -> missing;
        # only the exact "parallax" survives.
        assert present == {"parallax": "parallax"}
        assert missing == ["Gmag"]

    def test_all_missing(self):
        present, missing = map_requested_columns(["Gmag", "parallax"], ["ra", "dec"])
        assert present == {}
        assert sorted(missing) == ["Gmag", "parallax"]


# ---------------------------------------------------------------------------
# collect_requested_columns
# ---------------------------------------------------------------------------
class TestCollectRequestedColumns:
    def test_system_expands_to_its_canonical_columns(self):
        available = ["source_id", "ra"]
        result = collect_requested_columns(
            ["Gaia"], None, False, False, None, None, available
        )
        assert result == set(PHOTOMETRIC_SYSTEMS["Gaia"])

    def test_multiple_systems_union_their_columns(self):
        # The ``for system in systems`` accumulation must union both systems.
        result = collect_requested_columns(
            ["Gaia", "TMASS"], None, False, False, None, None, ["source_id"]
        )
        assert result == set(PHOTOMETRIC_SYSTEMS["Gaia"]) | set(
            PHOTOMETRIC_SYSTEMS["TMASS"]
        )

    def test_unknown_system_raises(self):
        with pytest.raises(ValueError, match="Unknown photometric system"):
            collect_requested_columns(
                ["Nope"], None, False, False, None, None, ["source_id"]
            )

    def test_unknown_distance_raises(self):
        with pytest.raises(ValueError, match="Unknown distance type"):
            collect_requested_columns(
                None, ["bogus"], False, False, None, None, ["source_id"]
            )

    def test_distances_zp_and_flux_flags(self):
        result = collect_requested_columns(
            None,
            ["geometric"],
            include_zp_cols=True,
            include_flux_errors=True,
            fidelity=None,
            probability=None,
            available=["source_id"],
        )
        assert set(GAIA_DISTANCE_COLUMNS["geometric"]).issubset(result)
        assert set(ZP_COLUMNS).issubset(result)
        assert set(FLUX_ERROR_COLUMNS).issubset(result)

    def test_fidelity_and_probability_only_added_when_available(self):
        # "fidelity_v2" is present in the catalogue, "prob_col" is not.
        result = collect_requested_columns(
            None,
            None,
            False,
            False,
            fidelity="fidelity_v2",
            probability="prob_col",
            available=["source_id", "fidelity_v2"],
        )
        assert "fidelity_v2" in result
        assert "prob_col" not in result


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
def _write_catalogue(path) -> QTable:
    """Write a tiny Gaia-flavoured ECSV catalogue and return the source table."""
    table = QTable()
    table["source_id"] = np.array([SOURCE_ID_A, SOURCE_ID_B, 42], dtype=np.int64)
    table["ra"] = [10.0, 20.0, 30.0]
    table["dec"] = [-5.0, -6.0, -7.0]
    table["parallax"] = [1.0, 2.0, 3.0]
    # Gaia-archive-named photometry column -> alias for canonical "Gmag".
    table["phot_g_mean_mag"] = [15.0, 16.0, 17.0]
    # A second photometric system (2MASS) so multi-system requests can be tested.
    table["j_m"] = [12.0, 13.0, 14.0]
    table["h_m"] = [11.0, 12.0, 13.0]
    table["ks_m"] = [10.0, 11.0, 12.0]
    # A column that belongs to no requested system: must be dropped on subset.
    table["junk"] = [0.1, 0.2, 0.3]
    table.write(path, format="ascii.ecsv")
    return table


@pytest.fixture
def catalogue(tmp_path):
    path = tmp_path / "catalogue.ecsv"
    _write_catalogue(path)
    return str(path)


class TestDataLoader:
    def test_load_full_table_when_no_selection(self, catalogue):
        loader = DataLoader(catalogue, verbose=logging.ERROR)
        data = loader.load_data()

        # With no system/flag requested the whole table is returned untouched.
        assert set(data.colnames) == {
            "source_id",
            "ra",
            "dec",
            "parallax",
            "phot_g_mean_mag",
            "j_m",
            "h_m",
            "ks_m",
            "junk",
        }
        assert len(data) == 3
        assert loader.data is data  # cached on the instance

    def test_load_gaia_subset_resolves_alias_and_drops_junk(self, catalogue):
        loader = DataLoader(catalogue, verbose=logging.ERROR)
        data = loader.load_data(systems=["Gaia"])

        # Only the Gaia columns physically present survive; "phot_g_mean_mag" is
        # kept (matched via the "Gmag" alias); "junk" is dropped.
        assert set(data.colnames) == {
            "source_id",
            "ra",
            "dec",
            "parallax",
            "phot_g_mean_mag",
        }
        assert len(data) == 3

        # 64-bit identifier must survive the round-trip EXACTLY (int64, not float).
        assert not np.issubdtype(np.asarray(data["source_id"]).dtype, np.floating)
        assert int(data["source_id"][0]) == SOURCE_ID_A
        assert int(data["source_id"][1]) == SOURCE_ID_B

        # Value spot-checks on ordinary float columns.
        assert list(np.asarray(data["ra"], dtype=float)) == [10.0, 20.0, 30.0]
        assert list(np.asarray(data["phot_g_mean_mag"], dtype=float)) == [
            15.0,
            16.0,
            17.0,
        ]

    def test_load_two_systems_unions_both_and_drops_junk(self, catalogue):
        loader = DataLoader(catalogue, verbose=logging.ERROR)
        data = loader.load_data(systems=["Gaia", "TMASS"])

        # Both the Gaia subset AND the 2MASS photometry survive; "junk" is dropped.
        assert set(data.colnames) == {
            "source_id",
            "ra",
            "dec",
            "parallax",
            "phot_g_mean_mag",
            "j_m",
            "h_m",
            "ks_m",
        }
        assert len(data) == 3
        assert list(np.asarray(data["j_m"], dtype=float)) == [12.0, 13.0, 14.0]

    def test_load_unknown_system_raises(self, catalogue):
        loader = DataLoader(catalogue, verbose=logging.ERROR)
        with pytest.raises(ValueError, match="Unknown photometric system"):
            loader.load_data(systems=["NotASystem"])

    def test_include_distances_must_be_a_list(self, catalogue):
        loader = DataLoader(catalogue, verbose=logging.ERROR)
        with pytest.raises(TypeError, match="must be a list or None"):
            loader.load_data(include_distances="geometric")

    def test_count_valid_sources_before_load_raises(self, catalogue):
        loader = DataLoader(catalogue, verbose=logging.ERROR)
        with pytest.raises(ValueError, match="Data has not been loaded"):
            loader.count_valid_sources()

    def test_count_valid_sources_after_load(self, catalogue):
        loader = DataLoader(catalogue, verbose=logging.ERROR)
        loader.load_data(systems=["Gaia"])
        counts = loader.count_valid_sources()
        # All three source_id values are present/finite.
        assert counts["Gaia IDs"] == 3

    def test_check_available_photometric_systems(self, catalogue):
        loader = DataLoader(catalogue, verbose=logging.ERROR)
        report = loader.check_available_photometric_systems()

        # The catalogue holds only a subset of the Gaia columns, so Gaia is not
        # "fully available"; TMASS/WISE columns are entirely absent.
        assert report["Gaia"]["available"] is False
        # build_available_systems matches canonical names literally (no aliases),
        # so "phot_g_mean_mag" does NOT count towards Gaia's "Gmag" here.
        assert set(report["Gaia"]["columns"]) == {"source_id", "ra", "dec", "parallax"}
        # TMASS has its three magnitudes but not the full column set (no
        # designation / sigcom columns) -> present-but-not-fully-available.
        assert report["TMASS"]["available"] is False
        assert set(report["TMASS"]["columns"]) == {"j_m", "h_m", "ks_m"}
        # WISE is entirely absent from the catalogue.
        assert report["WISE"]["available"] is False
        assert report["WISE"]["columns"] == []
