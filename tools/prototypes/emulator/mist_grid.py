"""Build a dense *regular* MIST grid in (log age, [Fe/H], log initial mass).

Prototype for the differentiable stellar-track emulator (see
``~/phd/agent-findings/differentiable-emulator-design.md``).  Pure NumPy: this
module only *builds and caches* the tensor.  The differentiable evaluation lives
in :mod:`emulator`.

Why *initial mass* and not EEP
------------------------------
EEP is the homologous coordinate and is what MIST recommends for interpolation.
It is nevertheless the wrong primary axis **for a rectangular grid covering young
ages**, and this is measured, not assumed (``_inspect_eep.py``):

======  ==========  ==========  ===================================
logage  EEP range   mass @ 148  what a common EEP floor of 148 costs
======  ==========  ==========  ===================================
6.00    [53, 343]   3.654       everything below 3.7 Msun
6.60    [82, 808]   0.931       everything below 0.93 Msun
8.00    [148, 1710] 0.100       nothing
======  ==========  ==========  ===================================

The common EEP window over log age in [6, 8] is [148, 343]; at log age 6.0 that
window contains only masses 3.65-288 Msun, i.e. it deletes the entire low-mass
pre-main-sequence -- which is exactly where a young cluster's age information
lives.  Initial mass has the opposite failure: MIST isochrones all start at
0.1 Msun, so the *low* end is fully covered at every age, and only the *top* is
truncated (to ``min_age max(M)``).  For the young-cluster regime this trades an
unusable restriction for a benign one.

The cost of the mass basis is turn-off smearing: at fixed mass, adjacent age
nodes can straddle the main-sequence hook, and linear interpolation in age cuts
the corner.  That cost is *measured* by ``emulator_accuracy.py`` (leave-one-age-out
and leave-one-feh-out), not waved away.  It is the reason this grid stops at
log age 7.6: past there the turn-off descends into the covered mass range.

Mass is monotone along every MIST isochrone (verified: ``mass monotone=True``),
so ``G(M)`` and ``(BP-RP)(M)`` are single-valued and ``np.interp`` onto a common
mass axis is well posed with no sorting ambiguity.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

MIST_DIR = Path("/Users/notluquis/erotica/data/test/NGC6383/MIST/UBVRIplus")
CACHE = Path(__file__).parent / "_cache"

# 0-indexed columns of MIST_v1.2 *_UBVRIplus.iso.cmd, read off the file header:
#   1 EEP  2 log10_isochrone_age_yr  3 initial_mass  ...  9 [Fe/H]
#   31 Gaia_G_EDR3  32 Gaia_BP_EDR3  33 Gaia_RP_EDR3  34 phase
_COLS = [0, 1, 2, 8, 30, 31, 32, 33]
_NAMES = ["eep", "logage", "mini", "feh", "G", "BP", "RP", "phase"]

# Default box.  See module docstring for why it ends at 7.6.
AGE_LO, AGE_HI = 6.00, 7.60          # MIST native nodes, 0.05 dex
FEH_LO, FEH_HI = -1.00, 0.50         # MIST files, 0.25 dex
MASS_LO, MASS_HI = 0.10, 8.00        # Msun; 8.0 is min-over-ages max(M) in the box
N_MASS = 160                         # log-spaced nodes


def _feh_of_filename(fn: str) -> float:
    m = re.search(r"feh_([mp])(\d\.\d\d)", fn)
    sign = -1.0 if m.group(1) == "m" else 1.0
    return sign * float(m.group(2))


def build(
    age_lo: float = AGE_LO,
    age_hi: float = AGE_HI,
    feh_lo: float = FEH_LO,
    feh_hi: float = FEH_HI,
    mass_lo: float = MASS_LO,
    mass_hi: float = MASS_HI,
    n_mass: int = N_MASS,
    *,
    stride_age: int = 1,
    stride_feh: int = 1,
    cache: bool = True,
) -> dict[str, np.ndarray]:
    """Build (or load) the regular (age, feh, log-mass) tensor.

    Parameters
    ----------
    stride_age, stride_feh
        Keep every ``stride``-th node.  This is the hook for held-out
        validation, and it **must** be a stride rather than deleting individual
        nodes: the emulator's index map is affine because the axes are regular,
        so an axis with a hole in it silently addresses the wrong node.  (That
        bug was written, run, and caught here -- deleting the 6.60 node made the
        emulator evaluate at 6.65 while reporting 6.60, and C0 and C1 returned
        byte-identical "accuracy" because both reproduce a node exactly.)
        ``stride=2`` halves the resolution regularly, so the nodes at odd
        indices become genuine held-out truth at twice the native spacing.
    cache
        Reuse/write ``_cache/mist_grid_*.npz``.  Withheld-node builds are never
        cached under the same key as the full build.

    Returns
    -------
    dict with keys
        ``logage`` (n_a,), ``feh`` (n_f,), ``logmass`` (n_m,),
        ``G`` (n_a, n_f, n_m) absolute Gaia EDR3 G,
        ``BPRP`` (n_a, n_f, n_m) intrinsic BP-RP,
        ``valid`` (n_a, n_f, n_m) bool -- node inside the source isochrone's own
        mass support (never extrapolated).
    """
    tag = (f"a{age_lo}-{age_hi}_f{feh_lo}-{feh_hi}_m{mass_lo}-{mass_hi}_{n_mass}"
           f"_sa{stride_age}_sf{stride_feh}")
    path = CACHE / f"mist_grid_{tag}.npz"
    if cache and path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}

    files = sorted(glob.glob(str(MIST_DIR / "*.iso.cmd")))
    keep = [(f, _feh_of_filename(f)) for f in files]
    keep = [(f, v) for f, v in keep if feh_lo - 1e-9 <= v <= feh_hi + 1e-9]
    keep.sort(key=lambda t: t[1])
    keep = keep[::stride_feh]
    if not keep:
        raise ValueError("no MIST files in the requested [Fe/H] range")

    logmass = np.linspace(np.log10(mass_lo), np.log10(mass_hi), n_mass)
    mass = 10.0**logmass

    age_axis: np.ndarray | None = None
    G_list, C_list, V_list, feh_list = [], [], [], []

    for fn, fehv in keep:
        df = pd.read_csv(fn, sep=r"\s+", comment="#", header=None,
                         usecols=_COLS, names=_NAMES)
        sel = df[(df.logage >= age_lo - 1e-9) & (df.logage <= age_hi + 1e-9)]
        ages = np.unique(np.round(sel.logage.values, 4))[::stride_age]
        if age_axis is None:
            age_axis = ages
        elif not np.allclose(age_axis, ages):
            raise RuntimeError(f"age nodes differ between files: {fn}")

        Gf = np.empty((len(ages), n_mass))
        Cf = np.empty((len(ages), n_mass))
        Vf = np.empty((len(ages), n_mass), dtype=bool)
        for i, a in enumerate(ages):
            b = sel[np.isclose(sel.logage.values, a)]
            m_i = b.mini.values
            order = np.argsort(m_i)
            m_i = m_i[order]
            g_i = b.G.values[order]
            c_i = (b.BP.values - b.RP.values)[order]
            # np.interp clamps outside the support; the `valid` mask records where
            # that happened so extrapolated corners are never silently trusted.
            Gf[i] = np.interp(mass, m_i, g_i)
            Cf[i] = np.interp(mass, m_i, c_i)
            Vf[i] = (mass >= m_i[0]) & (mass <= m_i[-1])
        G_list.append(Gf)
        C_list.append(Cf)
        V_list.append(Vf)
        feh_list.append(fehv)

    out = {
        "logage": np.asarray(age_axis, float),
        "feh": np.asarray(feh_list, float),
        "logmass": logmass,
        # stack -> (n_feh, n_age, n_mass), then move age first
        "G": np.transpose(np.stack(G_list), (1, 0, 2)),
        "BPRP": np.transpose(np.stack(C_list), (1, 0, 2)),
        "valid": np.transpose(np.stack(V_list), (1, 0, 2)),
    }
    if cache:
        CACHE.mkdir(exist_ok=True)
        np.savez_compressed(path, **out)
    return out


def raw_isochrone(logage: float, feh: float) -> dict[str, np.ndarray]:
    """Read one MIST isochrone straight off disk -- the *ground truth* oracle.

    Used by the emulator-accuracy and injection-recovery checks so that the
    truth never comes from the emulator itself.  ``logage`` must be a native
    MIST node (0.05 dex); ``feh`` must be a file value.
    """
    files = sorted(glob.glob(str(MIST_DIR / "*.iso.cmd")))
    fn = min(files, key=lambda f: abs(_feh_of_filename(f) - feh))
    if abs(_feh_of_filename(fn) - feh) > 1e-6:
        raise ValueError(f"[Fe/H]={feh} is not a MIST file value")
    df = pd.read_csv(fn, sep=r"\s+", comment="#", header=None,
                     usecols=_COLS, names=_NAMES)
    b = df[np.isclose(df.logage.values, logage)]
    if len(b) == 0:
        raise ValueError(f"log age {logage} is not a MIST node")
    o = np.argsort(b.mini.values)
    return {
        "mass": b.mini.values[o],
        "G": b.G.values[o],
        "BPRP": (b.BP.values - b.RP.values)[o],
        "eep": b.eep.values[o],
        "phase": b.phase.values[o],
    }


if __name__ == "__main__":
    import time

    t0 = time.time()
    g = build()
    print(f"built in {time.time() - t0:.1f}s")
    print(f"  logage  {g['logage'].shape} {g['logage'][0]:.2f}..{g['logage'][-1]:.2f} "
          f"step {np.diff(g['logage'])[0]:.3f}")
    print(f"  feh     {g['feh'].shape} {g['feh']}")
    print(f"  logmass {g['logmass'].shape} "
          f"M {10**g['logmass'][0]:.3f}..{10**g['logmass'][-1]:.2f}")
    print(f"  G       {g['G'].shape}  range [{g['G'].min():.2f}, {g['G'].max():.2f}]")
    print(f"  BPRP    {g['BPRP'].shape} range [{g['BPRP'].min():.2f}, {g['BPRP'].max():.2f}]")
    print(f"  valid   {100 * g['valid'].mean():.2f}% of nodes inside native mass support")
    bad = ~g["valid"]
    if bad.any():
        ia, jf, km = np.where(bad)
        print(f"  invalid nodes: ages {np.unique(g['logage'][ia])}, "
              f"masses > {10**g['logmass'][km].min():.2f} Msun")
