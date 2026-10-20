"""Tests for the isochrone-grid layer (erotica.analysis.grids) and its use by IsochroneFitter.

Oracles, by test group (tests/AGENTS.md: an oracle exists independently of the code under test):

* **Legacy reader / legacy fitter.** ``MISTIsochrones`` and the fitter at commit ``5d8faee``
  (loaded from git, not imported) are the reference: the MIST backend must reproduce their arrays
  bit for bit and the C1 log-likelihood to rounding.
* **Closed-form toy family** (as in ``test_isochrone.py``): photometry written from a formula, so
  "the node isochrone is the file" and interpolation errors are checkable without the code.
* **Published file headers.** The [Fe/H]-label conventions are checked against MIST's own header
  values, not against a constant typed here.
* **Exact integrals.** The two-sided window probability against ``scipy.integrate.quad``.

Data-dependent tests SKIP, saying which file is missing; CI has none of the grids.
"""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import numpy as np
import pytest
from astropy.table import QTable

from erotica.analysis._isochrone import MISTIsochrones
from erotica.analysis.grids import MISTGrid, common_eep_table

REPO = Path(__file__).resolve().parent.parent
NGC = Path.home() / "erotica" / "data" / "test" / "NGC6383"
MIST12 = NGC / "MIST" / "UBVRIplus"
C1_SAMPLE = NGC / "comments_paper/radius_robustness/generated/40/paperfaithful_reference_p06.ecsv"
C1_SEARCH = Path.home() / ".cache" / "erotica-c1" / "search_seed42.json"
MIST25 = Path.home() / ".cache" / "erotica-grids" / "mist_v2.5" / "UBVRIplus_afe0_vvcrit0.4"
C1_PRIORS = dict(
    loga_range=(6.0, 7.0), Av_range=(0.5, 2.0), dm_mu=10.3, dm_sigma=0.2, dm_range=(9.5, 10.7)
)

requires_bayes = pytest.mark.skipif(
    importlib.util.find_spec("pytensor") is None, reason="requires the bayes extra (pytensor)"
)


def _skip_unless(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            pytest.skip(f"data not present on this machine: {p}")


# ---------------------------------------------------------------------------------------------
# A MIST-format toy family, with a real header (Yinit Zinit [Fe/H] [a/Fe] v/vcrit)
# ---------------------------------------------------------------------------------------------

_M = np.geomspace(0.1, 8.0, 90)
# v1.2 convention: Z = 0.0142857 * 10**feh, Y = 0.249 + 1.49 Z
_FEH = (-0.5, -0.25, 0.0, 0.25)
_AGES = (6.3, 6.4, 6.5, 6.6, 6.7)


def _phot(mass, loga, feh):
    lm = np.log10(mass)
    G = 4.6 - 6.5 * lm - 2.0 * (7.0 - loga) / (1.0 + mass**2) + 0.6 * feh * (1.0 - lm)
    col = 0.8 - 1.4 * lm + 0.3 * feh * (1.0 + lm) + 0.1 * (7.0 - loga)
    return G, col


def _write_family(d: Path, fehs=_FEH, ages=_AGES, trim: int = 0) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for feh in fehs:
        Z = 0.0142857 * 10**feh
        Y = 0.249 + 1.49 * Z
        head = textwrap.dedent(f"""\
            # MIST version number  = 1.2
            # --------------------------------------------------------------------------------
            #  Yinit        Zinit   [Fe/H]   [a/Fe]  v/vcrit
            # {Y:.4f}  {Z:.5E}  {feh:7.2f}     0.00     0.40
            # --------------------------------------------------------------------------------
            # EEP log10_isochrone_age_yr initial_mass [Fe/H]_init Gaia_G_EDR3 Gaia_BP_EDR3 Gaia_RP_EDR3
        """)
        rows = []
        for j, a in enumerate(ages):
            G, c = _phot(_M, a, feh)
            for i in range(trim * j, len(_M) - trim * j):
                rows.append(
                    f"{i + 1} {a:.4f} {_M[i]:.8f} {feh:.2f} {G[i]:.8f} "
                    f"{G[i] + 0.6 * c[i]:.8f} {G[i] - 0.4 * c[i]:.8f}"
                )
        tag = f"{'p' if feh >= 0 else 'm'}{abs(feh):.2f}"
        (d / f"MIST_v1.2_feh_{tag}_afe_p0.0_vvcrit0.4_toy.iso.cmd").write_text(
            head + "\n".join(rows) + "\n"
        )
    return d


def _stars(n=150, loga=6.52, feh=-0.1, dm=10.0, seed=3):
    rng = np.random.default_rng(seed)
    m = np.geomspace(0.15, 6.0, n)
    G, c = _phot(m, loga, feh)
    e = 0.01
    return QTable(
        {
            "Gmag": G + dm + rng.normal(0, e, n),
            "G_BPmag": G + 0.6 * c + dm + rng.normal(0, e, n),
            "G_RPmag": G - 0.4 * c + dm + rng.normal(0, e, n),
            "e_Gmag": np.full(n, e),
            "e_G_BPmag": np.full(n, e),
            "e_G_RPmag": np.full(n, e),
            "probability_hdbscan": np.ones(n),
        }
    )


_TOY_PRIORS = dict(
    loga_range=(6.3, 6.7),
    Av_range=(0.0, 1.0),
    dm_mu=10.0,
    dm_sigma=0.3,
    dm_range=(9.5, 10.5),
    alpha=0.0,
    beta=0.0,
)


# ---------------------------------------------------------------------------------------------
# The grid contract
# ---------------------------------------------------------------------------------------------


def test_mist_backend_reproduces_the_legacy_reader(tmp_path):
    """Oracle: ``MISTIsochrones`` (the reader the published fits went through). Same rows, same
    EEP order, same values -- bit for bit, so the grid path cannot drift from the legacy one."""
    d = _write_family(tmp_path / "fam", trim=3)
    g = MISTGrid(d)
    legacy = MISTIsochrones(d)
    assert g.version == "1.2" and g.eep_kind == "native"
    for feh in _FEH:
        z_legacy = legacy._met_values[np.argmin(np.abs(legacy._met_values - 0.0142857 * 10**feh))]
        for a in _AGES:
            n = g.node(feh, a)
            eep, mass, G, BP, RP = legacy.get_isochrone_eep(float(z_legacy), a)
            np.testing.assert_array_equal(n.eep, eep)
            np.testing.assert_array_equal(n.mass, mass)
            np.testing.assert_array_equal(n.mags["Gaia_G_EDR3"], G)
            np.testing.assert_array_equal(n.mags["Gaia_BP_EDR3"], BP)
            np.testing.assert_array_equal(n.mags["Gaia_RP_EDR3"], RP)


def test_node_map_is_exact_at_nodes_and_log_linear_between(tmp_path):
    """Oracle: the header Z values. At a node the map returns the header pair; halfway in
    log Z between two nodes it returns the midpoint [Fe/H] (the definition of the node map)."""
    g = MISTGrid(_write_family(tmp_path / "fam"))
    for feh in _FEH:
        z = 0.0142857 * 10**feh
        assert abs(float(g.feh_to_z(feh)) / float(f"{z:.5E}") - 1) < 1e-12
        assert abs(float(g.z_to_feh(float(f"{z:.5E}"))) - feh) < 1e-12
    z0, z1 = (float(g.feh_to_z(f)) for f in (-0.5, -0.25))
    assert abs(float(g.z_to_feh(np.sqrt(z0 * z1))) - (-0.375)) < 1e-12
    with pytest.raises(ValueError):
        g.z_to_feh(1.0)


def test_feh_header_and_data_column_must_agree(tmp_path):
    d = _write_family(tmp_path / "fam", fehs=(0.0,))
    f = next(d.iterdir())
    f.write_text(
        f.read_text().replace("     0.00     0.00     0.40", "     0.10     0.00     0.40", 1)
    )
    with pytest.raises(ValueError, match=r"\[Fe/H\]_init"):
        MISTGrid(d)


def test_without_removes_the_node_and_refuses_a_no_op(tmp_path):
    """The hold-out primitive. A hold-out that removes nothing has zero error by construction --
    the mutation that would make every hold-out test pass -- so it must raise instead."""
    g = MISTGrid(_write_family(tmp_path / "fam"))
    h = g.without(loga=6.5)
    assert not any(h.has(f, 6.5) for f in _FEH) and all(h.has(f, 6.4) for f in _FEH)
    with pytest.raises(ValueError, match="nothing removed"):
        g.without(loga=6.55)
    s = g.select(feh=0.0)
    assert s.feh_nodes.tolist() == [0.0] and len(s.loga_nodes) == len(_AGES)


def test_common_table_clamps_like_the_fitter(tmp_path):
    """EEPs a node lacks are clamped to its end point, so the mass step there is exactly 0."""
    g = MISTGrid(_write_family(tmp_path / "fam", trim=4))
    axis, T = common_eep_table(g, g.feh_nodes, g.loga_nodes, ("Gaia_G_EDR3",))
    last = T[0, -1, 0]  # oldest age has the most trimmed range
    n = g.node(g.feh_nodes[0], g.loga_nodes[-1])
    outside = (axis < n.eep.min()) | (axis > n.eep.max())
    assert outside.any() and np.all(np.diff(last)[outside[1:] & outside[:-1]] == 0)
