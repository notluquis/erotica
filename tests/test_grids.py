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
import json
import subprocess
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.table import QTable

from erotica.analysis._isochrone import IsochroneFitter, MISTIsochrones
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


# ---------------------------------------------------------------------------------------------
# The fitter on the grid path
# ---------------------------------------------------------------------------------------------


@requires_bayes
def test_grid_path_reproduces_the_legacy_fit_off_node(tmp_path):
    """Oracle: the legacy fitter (Z files, interpolation in log Z). The grid path interpolates in
    [Fe/H]; with the node map the weights are identical, so the node table must be equal and
    log L at an off-node point (Z between nodes) equal to rounding. Mutation: interpolating the
    grid path in log10 of [Fe/H] instead of [Fe/H] breaks it."""
    d = _write_family(tmp_path / "fam")
    data = _stars()
    fl = IsochroneFitter(d, **_TOY_PRIORS)
    fl.setup(data, prob_threshold=0.0)
    g = MISTGrid(d)
    fg = IsochroneFitter(grid=g, **_TOY_PRIORS)
    fg.setup(data, prob_threshold=0.0)
    np.testing.assert_array_equal(fl._nodes, fg._nodes)
    z = 0.0142857 * 10**-0.13  # between the -0.25 and 0.0 nodes
    feh = float(np.interp(np.log10(z), fl._node_logz, fg._node_logz))
    for loga in (6.45, 6.52):
        a = fl.loglike(z, loga, 10.02, 0.1, 0.02, 0.01)
        b = fg.loglike(feh, loga, 10.02, 0.1, 0.02, 0.01)
        assert abs(a - b) < 1e-9 * abs(a), (a, b)


@requires_bayes
def test_grid_path_samples_feh_and_legacy_samples_met(tmp_path):
    d = _write_family(tmp_path / "fam")
    fl = IsochroneFitter(d, **_TOY_PRIORS)
    fl.setup(_stars(), prob_threshold=0.0)
    fg = IsochroneFitter(grid=MISTGrid(d), **_TOY_PRIORS)
    fg.setup(_stars(), prob_threshold=0.0)
    ml, mg = fl.build_model(), fg.build_model()
    assert "met" in ml.named_vars and "feh" not in ml.named_vars
    assert "feh" in mg.named_vars and "met" not in mg.named_vars
    lo, hi = fg._met_bounds()
    assert (lo, hi) == (-0.5, 0.25)
    one = IsochroneFitter(grid=MISTGrid(d).select(feh=0.0), **_TOY_PRIORS)
    one.setup(_stars(), prob_threshold=0.0)
    assert "feh" in one.build_model().named_vars  # fixed, as a Deterministic
    assert one.find_start(2, np.random.default_rng(0))["mode"]["feh"] == 0.0


@requires_bayes
def test_node_table_cache_refuses_the_other_metallicity_coordinate(tmp_path):
    d = _write_family(tmp_path / "fam")
    fl = IsochroneFitter(d, **_TOY_PRIORS)
    fl.setup(_stars(), prob_threshold=0.0)
    fl.save_grid(tmp_path / "n.npz")
    fg = IsochroneFitter(grid=MISTGrid(d), **_TOY_PRIORS)
    fg.setup(_stars(), prob_threshold=0.0, precompute_grid=False)
    with pytest.raises(ValueError, match="metallicity coordinate"):
        fg.load_grid(tmp_path / "n.npz")


def test_exactly_one_of_files_or_grid():
    with pytest.raises(ValueError, match="exactly one"):
        IsochroneFitter(**_TOY_PRIORS)


@requires_bayes
def test_two_sided_window_is_the_exact_segment_average(tmp_path):
    """Oracle: ``scipy.integrate.quad`` of Phi((faint - G)/s) - Phi((bright - G)/s) along a
    segment, uniform and linear-weighted. Mutation: dropping the bright-edge subtraction."""
    from scipy.integrate import quad
    from scipy.stats import norm

    d = _write_family(tmp_path / "fam")
    f = IsochroneFitter(grid=MISTGrid(d), **_TOY_PRIORS)
    f.setup(_stars(), prob_threshold=0.0, mag_window=(12.0, 17.0))
    assert f._mag_bright == 12.0 and f._mag_lim == 17.0
    assert f._obs_mag.min() >= 12.0 and f._obs_mag.max() <= 17.0
    A = np.array([11.0, 11.5, 16.0, 16.9, 13.0])
    B = np.array([12.4, 11.6, 17.5, 16.9 + 1e-8, 14.0])
    s2 = 0.03**2
    P, T = f._p_observed(A, B, s2, np, first_moment=True)
    c = f._e_mag_coef
    for k in range(len(A)):
        mid = np.clip(0.5 * (A[k] + B[k]), f._obs_mag.min(), f._obs_mag.max())
        s = np.sqrt((10 ** (c[0] + c[1] * mid + c[2] * mid**2)) ** 2 + s2)

        def win(t, k=k, s=s):
            G = A[k] + t * (B[k] - A[k])
            return norm.cdf((17.0 - G) / s) - norm.cdf((12.0 - G) / s)

        p = quad(win, 0, 1, epsabs=1e-12)[0]
        tt = quad(lambda t, w=win: t * w(t), 0, 1, epsabs=1e-12)[0]
        assert abs(P[k] - p) < 1e-7, (k, P[k], p)
        assert abs(T[k] - tt) < 1e-6, (k, T[k], tt)


# ---------------------------------------------------------------------------------------------
# Real data: the C1 regression and the [Fe/H] conventions (skip when the files are absent)
# ---------------------------------------------------------------------------------------------


def _module_at(commit: str):
    src = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{commit}:erotica/analysis/_isochrone.py"],
        capture_output=True,
        text=True,
    )
    if src.returncode != 0:
        pytest.skip(f"commit {commit} not in this clone's history")
    mod = types.ModuleType("erotica.analysis._isochrone_at_" + commit)
    mod.__package__ = "erotica.analysis"
    exec(compile(src.stdout, f"_isochrone@{commit}", "exec"), mod.__dict__)
    return mod


@requires_bayes
def test_c1_regression_mist_v12_backend_against_the_code_that_ran_c1():
    """THE regression of the grid layer. Oracle: ``_isochrone.py`` at ``5d8faee`` (origin/dev
    when the layer was started, the code C1 ran on), executed from git. On the NGC 6383 C1
    sample and priors:

    * the legacy path (``isochs_path=``) gives the same node table and the same log L at the
      C1 search mode, bit for bit;
    * the MIST v1.2 grid backend (``grid=``) gives the same node table bit for bit, and the same
      log L at the mode once Z is mapped to [Fe/H] with the legacy fitter's own node
      coordinates. (With the header Z instead, log L moves by -2.8e-7: the legacy reader rounds
      Z to 6 decimals, which moves the -0.50 node from 4.51753e-3 to 4.518e-3 -- and the
      [Fe/H] < -3 nodes by up to 30 %. Measured 2026-10-04; the grid path keeps the header Z.)
    * and the mode's stored log L (computed through JAX in the C1 search) to 1e-6.
    """
    from astropy.table import Table

    _skip_unless(MIST12, C1_SAMPLE, C1_SEARCH)
    mode = json.loads(C1_SEARCH.read_text())
    data = QTable(Table.read(C1_SAMPLE))
    old = _module_at("5d8faee")
    fo = old.IsochroneFitter(isochs_path=MIST12, **C1_PRIORS)
    fo.setup(data, prob_threshold=0.0)
    fl = IsochroneFitter(isochs_path=MIST12, **C1_PRIORS)
    fl.setup(data, prob_threshold=0.0)
    fg = IsochroneFitter(grid=MISTGrid(MIST12, loga_range=(5.95, 7.05)), **C1_PRIORS)
    fg.setup(data, prob_threshold=0.0)
    np.testing.assert_array_equal(fo._nodes, fl._nodes)
    np.testing.assert_array_equal(fo._nodes, fg._nodes)
    m = mode["mode"]
    args = [m[k] for k in ("met", "loga", "dm", "Av", "sigma_int", "f_bg")]
    ref = fo.loglike(*args)
    assert fl.loglike(*args) == ref
    feh = float(np.interp(np.log10(m["met"]), fl._node_logz, fg._node_logz))
    assert abs(fg.loglike(feh, *args[1:]) - ref) < 1e-9
    assert abs(ref - mode["loglike"]) < 1e-6


@pytest.mark.parametrize(
    "path,fits,fails",
    [(MIST12, "log10(Z/Zsun)", "log10(Z/X)-sun"), (MIST25, "log10(Z/X)-sun", "log10(Z/Zsun)")],
    ids=["v1.2", "v2.5"],
)
def test_mist_feh_label_conventions_from_the_headers(path, fits, fails):
    """Oracle: MIST's own header triples (Yinit, Zinit, [Fe/H]). v1.2 labels are log10(Z/Zsun);
    v2.5 labels are log(Z/X) - log(Z/X)_sun. Each convention fails on the other version by
    > 0.04 dex at +0.5 -- that is why the fit coordinate is the label, not Z."""
    _skip_unless(path)
    g = MISTGrid(path, loga_range=(6.5, 6.5))
    r = g.feh_convention_residuals()
    assert r[fits] < 1e-4 and r[fails] > 0.04, r


# ---------------------------------------------------------------------------------------------
# Hold-out validation
# ---------------------------------------------------------------------------------------------

from erotica.analysis.grids import (  # noqa: E402
    BHAC15Grid,
    PARSECGrid,
    SPOTSGrid,
    holdout_grid,
    regrid,
    safe_window,
)
from erotica.analysis.grids.base import GridNode, IsochroneGrid  # noqa: E402
from erotica.analysis.grids.holdout import interpolate  # noqa: E402
from erotica.analysis.grids.pseudo_eep import effective_phase, resample  # noqa: E402


class _QuadGrid(IsochroneGrid):
    """Closed-form grid: at fixed mass index, G = 5 - 2 log m + K (log t - 6.5)**2, colour fixed.
    Linear interpolation between log t +- h has error exactly K h**2 at the midpoint."""

    eep_kind = "mass"
    default_bands = ("G", "B", "R")

    def __init__(self, K=1.0, step=0.1):
        super().__init__()
        self.bands = ("G", "B", "R")
        m = np.geomspace(0.2, 2.0, 40)
        for a in np.round(np.arange(6.0, 7.0 + 1e-9, step), 4):
            G = 5 - 2 * np.log10(m) + K * (a - 6.5) ** 2
            self._add(
                0.0,
                float(a),
                GridNode(np.arange(m.size, dtype=float), m.copy(), {"G": G, "B": G + 0.8, "R": G}),
                None,
            )
        self.name = "quad"


def test_holdout_error_is_the_closed_form_second_difference():
    """Oracle: linear interpolation of K (x-6.5)^2 between x-h and x+h errs by exactly K h^2.
    Mutations: not removing the node (error 0, which the guard refuses) or interpolating from
    the wrong neighbours (error 4 K h^2)."""
    K, h = 0.8, 0.1
    r = holdout_grid(
        _QuadGrid(K, h), ("G", "B", "R"), feh=0.0, loga_range=(6.1, 6.9), mass_window=(0.2, 2.0)
    )
    for n in r["nodes"]:
        assert n["neighbours"] == pytest.approx([n["loga"] - h, n["loga"] + h])
        assert n["dG"]["p50"] == pytest.approx(K * h**2, rel=1e-9)
        assert n["dcol"]["p95"] == pytest.approx(0.0, abs=1e-12)


@requires_bayes
def test_holdout_interpolation_is_the_fitters(tmp_path):
    """The hold-out must measure the interpolation the fitter does, not a reimplementation of
    it that could drift: rows (mass, G, colour) of ``_interp_isochrone`` at an off-node point."""
    d = _write_family(tmp_path / "fam", trim=2)
    g = MISTGrid(d)
    f = IsochroneFitter(grid=g, **_TOY_PRIORS)
    f.setup(_stars(), prob_threshold=0.0)
    bands = ("Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3")
    X = f._interp_isochrone(-0.13, 6.47, np)
    m, G, c = interpolate(g, -0.13, 6.47, bands)
    # the fitter's EEP axis spans every node of its table; the hold-out's spans the four
    # bracketing nodes. Compare on the hold-out's axis (same EEP numbers, same clamping).
    fit_axis = np.arange(int(min(g.node(*k).eep.min() for k in g.nodes())), X.shape[1] + 1)
    four = [g.node(fe, a) for fe in (-0.25, 0.0) for a in (6.4, 6.5)]
    ho_axis = np.arange(
        int(min(n.eep.min() for n in four)), int(max(n.eep.max() for n in four)) + 1
    )
    idx = np.searchsorted(fit_axis, ho_axis)
    assert len(ho_axis) == len(m) and len(ho_axis) < X.shape[1]  # the axes really differ
    np.testing.assert_allclose(X[0][idx], m, rtol=0, atol=1e-12)
    np.testing.assert_allclose(X[1][idx], G, rtol=0, atol=1e-12)
    np.testing.assert_allclose(X[2][idx], c, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------------------------
# Pseudo-EEP
# ---------------------------------------------------------------------------------------------


def test_effective_phase_is_the_running_maximum():
    """PARSEC v1.2S flickers 0/1 at the PMS/MS boundary and labels the >= 20 Msun branch 0
    after the MS (measured on the CMD 3.7 file); every row belongs to the latest phase reached."""
    lab = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 2, 3])
    assert effective_phase(lab).tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 3]


def test_arclength_points_are_equally_spaced_in_the_metric():
    """Oracle: a straight HRD segment sampled unevenly must come out evenly spaced in D."""
    t = np.r_[0.0, 0.01, 0.02, 0.5, 0.9, 1.0]
    cols = {"mass": 0.1 + t, "logte": 3.5 + 0.2 * t, "logl": -1 + 2 * t, "x": t}
    eep, r = resample(
        cols, scheme="arclength", phase=np.zeros(6), phase_blocks={0: (10, 11)}, weights=(1.0, 1.0)
    )
    assert eep.tolist() == list(range(10, 21))
    np.testing.assert_allclose(np.diff(r["x"]), 0.1, atol=1e-12)


def test_massquantile_scheme_is_asteca_interp_isochrones():
    """Oracle: ASteCA 0.7.0's own ``interp_isochrones`` (MIT), on the same isochrone. The
    reimplementation is what the hold-out scored; this pins it to the code it claims to be."""
    ip = pytest.importorskip("asteca.modules.isochrones_priv", reason="asteca extra not installed")
    m = np.geomspace(0.1, 40.0, 300)
    G = 10 - 3 * np.log10(m)
    arr = np.zeros(m.size, dtype=[("Mini", float), ("Gmag", float)])
    arr["Mini"], arr["Gmag"] = m, G
    ref = ip.interp_isochrones(2000, "Mini", None, [["0.0152", "6.5"]], [arr])["0.0152"]["6.5"][0]
    _, ours = resample({"mass": m, "G": G}, scheme="massquantile", n_total=2000)
    np.testing.assert_allclose(ours["mass"], ref["Mini"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(ours["G"], ref["Gmag"], rtol=0, atol=1e-12)


def _write_cmd(d: Path) -> Path:
    """A CMD-format table with a flicker at the PMS/MS boundary and a 99.999 padding row."""
    d.mkdir(parents=True, exist_ok=True)
    head = (
        "# isochrones based on PARSEC release v1.2S\n"
        "# Zini     MH   logAge Mini        int_IMF         Mass   logL    logTe  logg  label "
        "mbolmag  Gmag    G_BPmag  G_RPmag\n"
    )
    rows = []
    for z, mh in ((0.01, -0.17553), (0.02, 0.14251)):
        for a in (6.4, 6.5, 6.6):
            m = np.geomspace(0.1, 5.0, 60)
            lab = np.where(m < 2.0, 0, 1)
            lab[30] = 1  # flicker
            for i, mm in enumerate(m):
                G = 6 - 2.5 * np.log10(mm) - 2 * (6.6 - a) + mh
                pad = 99.999 if i == 59 and a == 6.5 else G
                rows.append(
                    f"{z} {mh} {a:.5f} {mm:.6f} 1.0 {mm:.3f} {np.log10(mm) * 2:.3f} "
                    f"{3.6 + 0.2 * np.log10(mm):.4f} 4.0 {lab[i]} 0 {pad:.3f} "
                    f"{G + 0.7:.3f} {G - 0.5:.3f}"
                )
    (d / "cmd.dat").write_text(head + "\n".join(rows) + "\n")
    return d


def test_parsec_reader_uses_mh_drops_padding_and_monotone_phases(tmp_path):
    g = PARSECGrid(_write_cmd(tmp_path / "p"))
    assert g.feh_nodes.tolist() == [-0.17553, 0.14251]
    assert float(g.z_to_feh(0.01)) == pytest.approx(-0.17553)
    n = g.node(-0.17553, 6.5)
    assert n.mass.max() < 5.0  # the padded 99.999 row is gone
    assert np.all(np.diff(n.mass) >= 0) and np.all(np.diff(n.eep) > 0)
    assert g.eep_kind == "pseudo" and g.version == "v1.2S"


def test_regrid_needs_and_uses_the_native_phase(tmp_path):
    d = _write_family(tmp_path / "fam")
    with pytest.raises(KeyError):  # the toy family has no log_Teff/log_L/phase columns
        regrid(MISTGrid(d), "arclength")


# ---------------------------------------------------------------------------------------------
# PMS-only grids and the window
# ---------------------------------------------------------------------------------------------

BHAC = Path.home() / ".cache/erotica-grids/bhac15"
SPOTS = Path.home() / ".cache/erotica-grids/spots"


def test_spots_gaia_magnitudes_exist_only_above_half_a_solar_mass_when_spotted():
    """Oracle: the Zenodo files (-99 outside the colour tables' calibrated range). This is the
    limit a SPOTS f=0.34 control in Gaia inherits; 2MASS J reaches 0.10 Msun."""
    _skip_unless(SPOTS / "f034.isoc", SPOTS / "f000.isoc")
    s34 = SPOTSGrid(SPOTS, 0.34, loga_range=(6.0, 7.0))
    s00 = SPOTSGrid(SPOTS, 0.0, loga_range=(6.0, 7.0))
    assert s34.fspot == pytest.approx(0.339, abs=1e-3)
    assert s34.node(0.0, 6.5).mass.min() == pytest.approx(0.55)
    assert s00.node(0.0, 6.5).mass.min() == pytest.approx(0.15)
    j = SPOTSGrid(SPOTS, 0.34, bands=("J_mag", "J_mag", "K_mag"), loga_range=(6.0, 7.0))
    assert j.node(0.0, 6.5).mass.min() == pytest.approx(0.10)
    assert s34.provenance.license.startswith("CC BY 4.0")


def test_bhac15_reads_every_age_block():
    _skip_unless(BHAC / "BHAC15_iso.GAIA")
    b = BHAC15Grid(BHAC)
    assert len(b.loga_nodes) == 30 and b.loga_nodes[0] == pytest.approx(np.log10(5e5), abs=1e-4)
    n = b.node(0.0, b.loga_nodes[0])
    assert n.mass.min() == pytest.approx(0.01) and n.eep[0] == 0.0
    assert b.feh_nodes.tolist() == [0.0]


def test_safe_window_is_the_intersection_over_grids_and_priors():
    """Oracle: the closed-form grid. Brightest model G at the oldest bracketing node (largest
    K term) plus dm_max + k Av_max; faintest at the node with the smallest, plus dm_min."""
    g = _QuadGrid(K=1.0, step=0.1)
    lo, hi = safe_window(
        [g], "G", loga_range=(6.2, 6.6), dm_range=(9.0, 10.0), Av_range=(0.0, 1.0), k_mag=0.8
    )
    Gb = 5 - 2 * np.log10(2.0) + 1.0 * (6.2 - 6.5) ** 2  # brightest point, at the youngest node
    Gf = 5 - 2 * np.log10(0.2) + 0.0  # faintest point, at log t 6.5
    assert lo == pytest.approx(Gb + 10.0 + 0.8)
    assert hi == pytest.approx(Gf + 9.0)
    assert (
        safe_window(
            [g],
            "G",
            loga_range=(6.2, 6.6),
            dm_range=(9.0, 10.0),
            Av_range=(0.0, 1.0),
            k_mag=0.8,
            data_faint=15.35,
        )[1]
        == 15.35
    )


# ---------------------------------------------------------------------------------------------
# Fetching: the network tests SKIP without network and say so
# ---------------------------------------------------------------------------------------------

from erotica.analysis.grids import fetch as gfetch  # noqa: E402


def _network(host: str) -> None:
    import socket

    try:
        socket.create_connection((host, 443), timeout=5).close()
    except OSError as exc:
        pytest.skip(f"no network to {host} ({exc}); the download test did not run")


def test_a_changed_file_is_refused_not_used(tmp_path):
    """A grid file whose hash differs from the pinned one is a different grid version."""
    f = tmp_path / "x.dat"
    f.write_bytes(b"models")
    good = hashlib_sha256(b"models")
    assert gfetch._checked(f, good, "u")["sha256"] == good
    with pytest.raises(gfetch.ChangedUpstream):
        gfetch._checked(f, "0" * 64, "u")


def hashlib_sha256(b: bytes) -> str:
    import hashlib

    return hashlib.sha256(b).hexdigest()


@pytest.mark.network
def test_fetch_spots_checks_the_zenodo_md5(tmp_path, monkeypatch):
    _network("zenodo.org")
    monkeypatch.setenv("EROTICA_GRIDS_CACHE", str(tmp_path))
    d = gfetch.fetch_spots((0.34,))
    man = json.loads((d / "manifest.json").read_text())
    assert man[0]["file"] == "f034.isoc" and man[0]["checked_against"] == "zenodo md5"
    assert SPOTSGrid(d, 0.34, loga_range=(6.5, 6.5)).node(0.0, 6.5).mass.min() == pytest.approx(
        0.55
    )


@pytest.mark.network
def test_fetch_bhac15_matches_the_pinned_hash(tmp_path, monkeypatch):
    _network("perso.ens-lyon.fr")
    monkeypatch.setenv("EROTICA_GRIDS_CACHE", str(tmp_path))
    d = gfetch.fetch_bhac15(("BHAC15_iso.GAIA",))
    assert len(BHAC15Grid(d).loga_nodes) == 30


def test_cached_mist25_tarball_is_the_validated_one():
    tar = Path.home() / ".cache/erotica-grids/mist_v2.5/raw/UBVRIplus.txz"
    _skip_unless(tar)
    assert gfetch._hash(tar) == gfetch.MIST25_SHA256["UBVRIplus"]
