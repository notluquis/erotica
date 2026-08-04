"""Tests for the reproducibility record.

WHAT THESE ARE CHECKED AGAINST
------------------------------
Every assertion here has an oracle that is independent of the code under test:

* ``git_provenance``  -> a ``git`` subprocess run directly by the test, and a
  throwaway repository the test builds and dirties itself.
* ``file_checksum``   -> the published BLAKE2b vector for the empty string, plus
  a digest the test computes with ``hashlib`` on its own.
* ``dependency_versions`` -> the ``__version__`` attribute of the imported
  module, which is set independently of distribution metadata.
* ``build_metadata``  -> ``json.dumps``/``json.loads`` round-trip equality.

The invariant the whole module rests on is that a provenance record can always
be written to disk. A record that raises at write time destroys the result it
was meant to describe, so ``build_metadata`` is fed hostile values on purpose.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from erotica.analysis import provenance
from erotica.analysis.provenance import (
    build_metadata,
    calculate_mode,
    dependency_versions,
    file_checksum,
    git_provenance,
    load_results,
    posterior_mode,
    store_trace_results,
    summarize_trace,
    write_metadata,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def has_git() -> bool:
    try:
        return subprocess.run(["git", "--version"], capture_output=True).returncode == 0
    except OSError:
        return False


# ---------------------------------------------------------------------------
# git provenance
# ---------------------------------------------------------------------------


def test_git_commit_matches_an_independent_git_call(has_git):
    """Oracle: the SHA git itself reports for this working tree."""
    if not has_git:
        pytest.skip("git unavailable")
    expected = _git("rev-parse", "HEAD", cwd=REPO_ROOT)
    if expected.returncode != 0:
        pytest.skip("tests are not running inside a git repository")

    got = git_provenance()
    assert got is not None
    assert got["commit"] == expected.stdout.strip()
    assert len(got["commit"]) == 40 and all(c in "0123456789abcdef" for c in got["commit"])


def test_dirty_flag_tracks_the_working_tree(tmp_path, has_git):
    """Oracle: a repository the test builds, commits, then deliberately dirties.

    ``dirty`` is the field that makes a commit hash trustworthy. A test that only
    asserted the key exists would pass even if it were hardcoded ``False``.
    """
    if not has_git:
        pytest.skip("git unavailable")
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git("init", "-q", cwd=repo).returncode == 0
    _git("config", "user.email", "t@t.t", cwd=repo)
    _git("config", "user.name", "t", cwd=repo)
    (repo / "a.txt").write_text("one")
    _git("add", "-A", cwd=repo)
    assert _git("commit", "-q", "-m", "init", cwd=repo).returncode == 0

    clean = git_provenance(repo)
    assert clean is not None and clean["dirty"] is False

    (repo / "a.txt").write_text("two")  # same file, changed content
    assert git_provenance(repo)["dirty"] is True

    (repo / "a.txt").write_text("one")  # restored -> clean again
    assert git_provenance(repo)["dirty"] is False

    (repo / "untracked.txt").write_text("x")  # untracked also counts as dirty
    assert git_provenance(repo)["dirty"] is True


def test_returns_none_outside_a_repository(tmp_path, has_git):
    """A wheel installed from PyPI has no git tree; that must not raise."""
    if not has_git:
        pytest.skip("git unavailable")
    outside = tmp_path / "not_a_repo"
    outside.mkdir()
    if _git("rev-parse", "HEAD", cwd=outside).returncode == 0:
        pytest.skip("tmp_path happens to live inside a git repository")
    assert git_provenance(outside) is None


# ---------------------------------------------------------------------------
# input checksums
# ---------------------------------------------------------------------------


def test_checksum_matches_published_blake2b_vector(tmp_path):
    """Oracle: the BLAKE2b-512 digest of the empty string, a published constant."""
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    expected = (
        "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419"
        "d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce"
    )
    rec = file_checksum(empty)
    assert rec["blake2b"] == expected
    assert rec["bytes"] == 0


def test_checksum_tracks_content_not_filename(tmp_path):
    """Same bytes at a different path is the same input; one flipped byte is not."""
    payload = b"parallax,pmra,pmdec\n0.90,1.2,-3.4\n" * 500
    a, b, c = tmp_path / "a.csv", tmp_path / "sub_b.csv", tmp_path / "c.csv"
    (tmp_path / "sub_b.csv").parent.mkdir(exist_ok=True)
    a.write_bytes(payload)
    b.write_bytes(payload)
    c.write_bytes(payload[:-2] + b"5\n")  # one byte differs

    assert file_checksum(a)["blake2b"] == file_checksum(b)["blake2b"]
    assert file_checksum(a)["blake2b"] != file_checksum(c)["blake2b"]
    # Oracle: hashlib, computed here, on the whole payload at once.
    assert file_checksum(a)["blake2b"] == hashlib.blake2b(payload).hexdigest()
    assert file_checksum(a)["bytes"] == len(payload)


def test_checksum_streams_rather_than_reading_whole_file(tmp_path):
    """A 40 MB file must hash correctly; catalogues are far larger than RAM budget."""
    big = tmp_path / "big.bin"
    chunk = np.arange(2**20, dtype=np.int32).tobytes()  # 4 MB
    with big.open("wb") as fh:
        for _ in range(10):
            fh.write(chunk)
    assert file_checksum(big)["blake2b"] == hashlib.blake2b(chunk * 10).hexdigest()
    assert file_checksum(big)["bytes"] == 10 * len(chunk)


# ---------------------------------------------------------------------------
# dependency versions
# ---------------------------------------------------------------------------


def test_dependency_versions_agree_with_the_imported_modules():
    """Oracle: ``module.__version__``, set independently of distribution metadata."""
    import astropy
    import pandas

    deps = dependency_versions()
    assert deps["numpy"] == np.__version__
    assert deps["pandas"] == pandas.__version__
    assert deps["astropy"] == astropy.__version__


def test_uninstalled_optional_extras_are_omitted_not_errors():
    """An absent extra is provenance too: that run produced no Bayesian numbers."""
    deps = dependency_versions()
    assert all(isinstance(v, str) and v for v in deps.values())
    try:
        import pymc  # noqa: F401
    except ImportError:
        assert "pymc" not in deps
    else:
        assert deps["pymc"] == pymc.__version__


# ---------------------------------------------------------------------------
# build_metadata -- the JSON-serialisability invariant
# ---------------------------------------------------------------------------


@dataclass
class _FakeSamplingConfig:
    draws: int = 1000
    chains: int = 4
    nuts_sampler: str = "pymc"


def test_record_survives_a_json_round_trip_with_hostile_values():
    """The invariant: provenance must always be writable.

    NumPy scalars, arrays, ``Path`` and dataclasses all raise in ``json.dumps``
    unhandled, and all of them arrive naturally from callers (a seed from a
    NumPy generator, a ``SamplingConfig``, a catalogue path).
    """
    meta = build_metadata(
        seeds={"sampler": np.int64(11), "bootstrap": 7},
        sampling=_FakeSamplingConfig(),
        cut=np.float32(2.0),
        columns=np.array(["ra", "dec"]),
        catalogue=Path("/data/ngc6383.ecsv"),
        labels={3, 1, 2},
    )
    assert json.loads(json.dumps(meta)) == meta  # round-trip is lossless
    assert meta["seeds"]["sampler"] == 11
    assert meta["sampling"]["nuts_sampler"] == "pymc"
    assert meta["cut"] == pytest.approx(2.0)
    assert meta["columns"] == ["ra", "dec"]
    assert meta["catalogue"] == "/data/ngc6383.ecsv"
    assert sorted(meta["labels"]) == [1, 2, 3]


def test_unserializable_object_is_downgraded_not_dropped():
    """An exotic type loses its type, never its presence -- silence is the failure."""

    class Exotic:
        def __repr__(self) -> str:
            return "Exotic(kind=king)"

    meta = build_metadata(model=Exotic())
    assert "king" in meta["model"]
    json.dumps(meta)


def test_required_fields_are_present():
    meta = build_metadata(seeds=42, cluster=3)
    for key in (
        "created_at",
        "erotica_version",
        "git",
        "python",
        "platform",
        "dependencies",
        "seeds",
        "inputs",
    ):
        assert key in meta, f"missing provenance field: {key}"
    assert meta["seeds"] == 42
    assert meta["cluster"] == 3
    assert sorted(meta)[:3] == ["cluster", "created_at", "dependencies"]  # pins the docstring


def test_absent_seed_is_recorded_as_none_not_omitted():
    """A run without a fixed seed is not reproducible; the record must say so."""
    meta = build_metadata()
    assert "seeds" in meta
    assert meta["seeds"] is None


def test_inputs_are_checksummed_and_a_single_path_is_accepted(tmp_path):
    f = tmp_path / "members.ecsv"
    f.write_text("plx\n0.9\n")
    one = build_metadata(inputs=f)
    many = build_metadata(inputs=[f, f])
    assert len(one["inputs"]) == 1
    assert len(many["inputs"]) == 2
    assert one["inputs"][0]["blake2b"] == hashlib.blake2b(f.read_bytes()).hexdigest()


def test_missing_input_is_recorded_by_default_and_raises_under_strict(tmp_path):
    """Provenance must never destroy the result it describes -- unless asked to."""
    ghost = tmp_path / "not_there.ecsv"
    meta = build_metadata(inputs=ghost)
    assert "error" in meta["inputs"][0]
    assert str(ghost) == meta["inputs"][0]["path"]
    json.dumps(meta)

    with pytest.raises(OSError):
        build_metadata(inputs=ghost, strict=True)


def test_write_metadata_produces_readable_json(tmp_path):
    out = write_metadata(tmp_path / "prov.json", seeds=1, note="hello")
    loaded = json.loads(out.read_text())
    assert loaded["seeds"] == 1
    assert loaded["note"] == "hello"


# ---------------------------------------------------------------------------
# the wiring -- provenance is written next to every saved trace
# ---------------------------------------------------------------------------


def test_saving_a_trace_writes_a_provenance_sidecar(tmp_path):
    """Regression guard: ``build_metadata`` had zero callers before 2026-07-27.

    Dead provenance code is worse than none, because the reproducibility claim
    is made anyway. This test fails if the wiring is removed.
    """
    az = pytest.importorskip("arviz")
    from erotica.analysis.provenance import store_trace_results

    rng = np.random.default_rng(0)
    trace = az.from_dict({"posterior": {"mu": rng.normal(0.9, 0.01, (2, 50))}})
    csv = tmp_path / "fit.csv"
    catalogue = tmp_path / "in.ecsv"
    catalogue.write_text("plx\n0.9\n")

    store_trace_results(trace, csv, metadata={"inputs": catalogue, "seeds": 0})

    sidecars = list(tmp_path.glob("fit_provenance_*.json"))
    assert len(sidecars) == 1
    meta = json.loads(sidecars[0].read_text())
    assert meta["variables"] == ["mu"]
    assert meta["seeds"] == 0
    assert meta["inputs"][0]["blake2b"] == hashlib.blake2b(catalogue.read_bytes()).hexdigest()
    assert meta["dependencies"]["numpy"] == np.__version__
    # the .nc it describes must actually be there, keyed by the same index
    assert (tmp_path / f"fit_trace_{meta['trace_index']}.nc").exists()


def test_no_sidecar_when_the_trace_is_not_saved(tmp_path):
    az = pytest.importorskip("arviz")
    from erotica.analysis.provenance import store_trace_results

    trace = az.from_dict({"posterior": {"mu": np.zeros((2, 10))}})
    store_trace_results(trace, tmp_path / "fit.csv", save_trace=False)
    assert not list(tmp_path.glob("*provenance*.json"))


# ---------------------------------------------------------------------------
# posterior summaries
# ---------------------------------------------------------------------------


def test_posterior_mode_finds_the_mode_not_the_mean():
    """Oracle: a lognormal, whose mode, median and mean are analytically distinct.

    For ``LogNormal(mu=0, sigma=1)``: mode ``exp(-1) = 0.368``, median ``1.0``,
    mean ``exp(0.5) = 1.649``. A stub returning the mean or the median -- the
    easy way to make a mode estimator "pass" -- fails this by a factor of ~3.
    """
    rng = np.random.default_rng(4)
    x = rng.lognormal(0.0, 1.0, 300_000)
    mode = posterior_mode(x)
    assert mode == pytest.approx(np.exp(-1.0), rel=0.10)
    assert mode < np.median(x) < np.mean(x)


def test_posterior_mode_ignores_non_finite_values():
    rng = np.random.default_rng(5)
    clean = rng.normal(3.0, 0.2, 50_000)
    dirty = np.concatenate([clean, [np.nan, np.inf, -np.inf]])
    assert posterior_mode(dirty) == pytest.approx(posterior_mode(clean), rel=0.05)


def test_posterior_mode_rejects_an_empty_posterior():
    with pytest.raises(ValueError, match="empty"):
        posterior_mode(np.array([np.nan, np.inf]))


def test_calculate_mode_ignores_its_legacy_arguments():
    """The notebook-era signature is accepted, but ``bw``/``circular`` are inert."""
    rng = np.random.default_rng(6)
    x = rng.normal(1.0, 0.1, 20_000)
    assert calculate_mode(x) == calculate_mode(x, bw="silverman", circular=True)


def test_summarize_trace_statistics_match_numpy():
    """Oracle: numpy, applied by the test to the same flattened array."""
    az = pytest.importorskip("arviz")
    rng = np.random.default_rng(7)
    mu = rng.normal(0.9, 0.05, (2, 500))
    trace = az.from_dict({"posterior": {"mu": mu, "likelihood": rng.normal(0, 1, (2, 500))}})

    summaries = summarize_trace(trace)
    assert [s.variable for s in summaries] == ["mu"]  # "likelihood" excluded by default
    s = summaries[0]
    assert s.mean == pytest.approx(float(np.mean(mu)))
    assert s.median == pytest.approx(float(np.median(mu)))
    assert s.std == pytest.approx(float(np.std(mu)))


def test_summarize_trace_exclusions_are_configurable():
    az = pytest.importorskip("arviz")
    trace = az.from_dict({"posterior": {"a": np.zeros((2, 10)), "b": np.ones((2, 10))}})
    assert [s.variable for s in summarize_trace(trace, excluded_parameters=("b",))] == ["a"]
    assert len(summarize_trace(trace, excluded_parameters=())) == 2


def test_summarize_trace_rejects_a_non_inferencedata():
    with pytest.raises(ValueError, match="InferenceData"):
        summarize_trace({"mu": [1, 2, 3]})


# ---------------------------------------------------------------------------
# store -> load round trip: the actual reproducibility contract
# ---------------------------------------------------------------------------


def test_stored_results_round_trip_through_load(tmp_path):
    """Oracle: round-trip identity. What was written must come back unchanged."""
    az = pytest.importorskip("arviz")
    rng = np.random.default_rng(8)
    mu = rng.normal(0.9, 0.02, (2, 400))
    trace = az.from_dict({"posterior": {"mu": mu}})
    csv = tmp_path / "fit.csv"

    written = store_trace_results(trace, csv)
    results, back = load_results(csv, load_trace=True)

    assert list(results["Parameter"]) == ["mu"]
    assert results.iloc[0]["Mean"] == pytest.approx(float(np.mean(mu)))
    assert results.iloc[0]["Mean"] == pytest.approx(written.iloc[0]["Mean"])
    # the recovered NetCDF holds the same draws, not merely the same summary
    np.testing.assert_allclose(np.asarray(back.posterior["mu"].values), mu)


def test_load_returns_only_the_latest_run_by_default(tmp_path):
    """Successive fits append; ``only_last`` must not mix two runs' numbers."""
    az = pytest.importorskip("arviz")
    csv = tmp_path / "fit.csv"
    first = az.from_dict({"posterior": {"mu": np.full((2, 50), 1.0)}})
    second = az.from_dict({"posterior": {"mu": np.full((2, 50), 2.0)}})

    store_trace_results(first, csv, save_trace=False)
    all_rows = store_trace_results(second, csv, save_trace=False)
    assert len(all_rows) == 2  # both are on disk

    latest, _ = load_results(csv, load_trace=False)
    assert len(latest) == 1
    assert latest.iloc[0]["Mean"] == pytest.approx(2.0)

    everything, _ = load_results(csv, load_trace=False, only_last=False)
    assert len(everything) == 2


def test_load_reports_a_missing_file_rather_than_returning_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "never_written.csv")


def test_load_reports_a_missing_trace_for_summaries_saved_without_one(tmp_path):
    az = pytest.importorskip("arviz")
    csv = tmp_path / "fit.csv"
    store_trace_results(
        az.from_dict({"posterior": {"mu": np.zeros((2, 10))}}), csv, save_trace=False
    )
    with pytest.raises(FileNotFoundError, match="trace"):
        load_results(csv, load_trace=True)


# ---------------------------------------------------------------------------
# Trace identity. Trace_Index was a UTC timestamp truncated to whole seconds and used as an
# IDENTITY -- it named the NetCDF, named the sidecar, and linked the CSV rows to both. A clock
# reading is the wrong thing to identify data with, and the code hit both failure modes: two
# calls in one second overwrote a trace while leaving both summary rows, and storing the same
# trace twice produced two indistinguishable copies.
# ---------------------------------------------------------------------------


def test_two_fits_in_the_same_second_do_not_destroy_a_trace(tmp_path):
    """The data-loss case. Both traces must survive, and both must be recoverable."""
    az = pytest.importorskip("arviz")
    csv = tmp_path / "fit.csv"
    first = az.from_dict({"posterior": {"mu": np.full((2, 50), 1.0)}})
    second = az.from_dict({"posterior": {"mu": np.full((2, 50), 2.0)}})

    store_trace_results(first, csv)
    store_trace_results(second, csv)  # same wall-clock second in any realistic run

    traces = sorted(tmp_path.glob("fit_trace_*.nc"))
    assert len(traces) == 2, f"a trace was overwritten: {[t.name for t in traces]}"

    # and they hold different draws -- not two copies of one fit
    values = {float(np.mean(az.from_netcdf(t).posterior["mu"].values)) for t in traces}
    assert values == {1.0, 2.0}, f"traces do not hold the two distinct fits: {values}"

    sidecars = sorted(tmp_path.glob("fit_provenance_*.json"))
    assert len(sidecars) == 2, "a provenance sidecar was overwritten"


def test_storing_the_same_trace_twice_is_idempotent(tmp_path):
    """Identity is content-derived, so re-storing one fit must not fabricate a second archive.

    The two stores are FORCED ONTO DIFFERENT CLOCK SECONDS, and that is the whole point of the
    test rather than an incidental detail. The first version of this test called
    ``store_trace_results`` twice back to back, which normally lands both inside one second --
    and the implementation it was written against checked only the slot the clock named, so it
    was idempotent by collision rather than by content. The test passed for months of runs and
    then failed the first time the two calls straddled a second boundary, archiving
    ``fit_trace_1785866397.nc`` and ``fit_trace_1785866398.nc``.

    A test that passes because two events happened to share a timestamp is not testing identity.
    Pinning the clock to two different values makes the property unconditional: the second store
    must find the first by its DRAWS, not by landing on the same integer.
    """
    az = pytest.importorskip("arviz")
    csv = tmp_path / "fit.csv"
    trace = az.from_dict({"posterior": {"mu": np.full((2, 50), 7.0)}})

    real_datetime = provenance.datetime
    # Unbounded: `store_trace_results` also stamps `Date_Time` once per summary row, so a fixed
    # list of ticks exhausts and raises StopIteration instead of testing anything.
    tick = itertools.count(1785866397.0)

    class _Clock:
        """`datetime` stand-in whose `now` advances a full second on every call."""

        @staticmethod
        def now(tz=None):
            return real_datetime.fromtimestamp(next(tick), tz=tz)

    with mock.patch.object(provenance, "datetime", _Clock):
        store_trace_results(trace, csv)
        store_trace_results(trace, csv)

    traces = sorted(tmp_path.glob("fit_trace_*.nc"))
    assert len(traces) == 1, (
        f"the same trace was archived twice: {[t.name for t in traces]}. Identity is supposed to "
        "come from content, so an identical trace must reuse its slot even when the two stores "
        "fall in different clock seconds."
    )


def test_load_trace_works_without_only_last_when_one_fit_is_stored(tmp_path):
    """The documented-but-unreachable case: load_trace=True with only_last=False ALWAYS raised
    FileNotFoundError('... : None'), however many valid traces existed, because trace_path was
    assigned only inside the only_last branch."""
    az = pytest.importorskip("arviz")
    csv = tmp_path / "fit.csv"
    mu = np.random.default_rng(0).normal(size=(2, 100))
    store_trace_results(az.from_dict({"posterior": {"mu": mu}}), csv)

    results, trace = load_results(csv, load_trace=True, only_last=False)
    assert trace is not None, "load_trace=True with only_last=False still returns no trace"
    np.testing.assert_allclose(np.asarray(trace.posterior["mu"].values), mu)
    assert len(results) >= 1


def test_ambiguous_trace_says_what_is_ambiguous(tmp_path):
    """Two fits and only_last=False cannot name 'the' trace. The error must say so, and say how
    many -- the old message was 'No trace file found ... : None', which described neither."""
    az = pytest.importorskip("arviz")
    csv = tmp_path / "fit.csv"
    store_trace_results(az.from_dict({"posterior": {"mu": np.full((2, 50), 1.0)}}), csv)
    store_trace_results(az.from_dict({"posterior": {"mu": np.full((2, 50), 2.0)}}), csv)

    with pytest.raises(FileNotFoundError, match="distinct traces"):
        load_results(csv, load_trace=True, only_last=False)
