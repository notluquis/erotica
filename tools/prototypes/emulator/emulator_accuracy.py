"""How wrong is the emulator, in magnitudes, against tracks it never saw?

Accuracy against held-out tracks is **necessary and not sufficient** -- what
finally matters is posterior bias, measured in ``injection_recovery.py``.  But it
is the quantity that can be measured densely and cheaply, and it localises the
error: an emulator that is 0.005 mag wrong on the main sequence and 0.2 mag wrong
at the turn-off is a different design problem from one that is uniformly 0.05.

Protocol -- held out by *stride*, with no interpolation in the truth
-------------------------------------------------------------------
The truth is always a **raw MIST isochrone read off disk**.  To hold it out, the
grid is rebuilt keeping every second node, so the withheld nodes sit exactly at
the midpoints of the coarsened grid:

* ``stride_age=2``  -> 0.10 dex age spacing; held-out nodes are the odd ones.
* ``stride_feh=2``  -> 0.50 dex [Fe/H] spacing; held-out are -0.75/-0.25/+0.25.

A stride, not a deletion.  Deleting one node leaves an axis with a hole, and the
emulator's index map is affine *because the axes are regular* -- so a holed axis
addresses the wrong node while reporting the right one.  That bug was written
and caught here; see ``mist_grid.build``.

Because the withheld nodes are at 2x the native spacing, these numbers are a
**pessimistic** proxy for production accuracy.  Multilinear error scales as
``h^2``, so the native-spacing error is expected near a quarter of the tabulated
value; ``refinement_check`` measures that directly instead of assuming it.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import mist_grid
from emulator import load

OUT = Path(__file__).parent / "_results"


def _err_at(em, grid, age, feh):
    """Emulator minus raw MIST, on the grid's own mass axis."""
    raw = mist_grid.raw_isochrone(age, feh)
    lm = np.asarray(grid["logmass"])
    m = 10.0**lm
    ok = (m >= raw["mass"][0]) & (m <= raw["mass"][-1])
    lm, m = lm[ok], m[ok]
    Gt = np.interp(m, raw["mass"], raw["G"])
    Ct = np.interp(m, raw["mass"], raw["BPRP"])
    Ge, Ce = em.absolute(age, feh, jnp.asarray(lm))
    return m, np.asarray(Ge) - Gt, np.asarray(Ce) - Ct


def _stats(m, dG, dC, sel=None):
    if sel is None:
        sel = np.ones_like(m, bool)
    if sel.sum() == 0:
        return None
    return {"n": int(sel.sum()),
            "rms_G": float(np.sqrt(np.mean(dG[sel] ** 2))),
            "max_G": float(np.max(np.abs(dG[sel]))),
            "rms_BPRP": float(np.sqrt(np.mean(dC[sel] ** 2))),
            "max_BPRP": float(np.max(np.abs(dC[sel])))}


def held_out(kind, orders):
    """Rows of held-out accuracy for one axis and one interpolation order."""
    kw = {"stride_age": 2} if kind == "age" else {"stride_feh": 2}
    grid = mist_grid.build(**kw)
    em = load(grid, order_age=orders[0], order_feh=orders[1])
    full = mist_grid.build()
    if kind == "age":
        nodes = [(a, 0.0) for a in full["logage"][1::2]]
    else:
        nodes = [(6.60, f) for f in full["feh"][1::2]]

    rows = []
    for age, feh in nodes:
        m, dG, dC = _err_at(em, grid, float(age), float(feh))
        rows.append({"age": float(age), "feh": float(feh), "orders": list(orders),
                     "all": _stats(m, dG, dC),
                     "M<1": _stats(m, dG, dC, m < 1.0),
                     "1-3": _stats(m, dG, dC, (m >= 1.0) & (m < 3.0)),
                     "M>=3": _stats(m, dG, dC, m >= 3.0),
                     "worst_mass": float(m[int(np.argmax(np.abs(dG)))])})
    return rows


def refinement_check(orders, n=4000, seed=0):
    """Native-spacing error, bounded without a finer grid than MIST provides.

    Compare the production emulator (stride 1) against the coarsened one
    (stride 2) at random interior points.  For a convergent scheme the
    production error is smaller than this difference by the refinement ratio,
    so ``|e_1 - e_2|`` is a **bound**, not an estimate, and it needs no truth.
    """
    g1, g2a, g2f = (mist_grid.build(), mist_grid.build(stride_age=2),
                    mist_grid.build(stride_feh=2))
    e1 = load(g1, order_age=orders[0], order_feh=orders[1])
    rng = np.random.default_rng(seed)
    a = rng.uniform(e1.age_lo, e1.age_hi, n)
    f = rng.uniform(e1.feh_lo, e1.feh_hi, n)
    lm = rng.uniform(e1.logmass_lo, e1.logmass_hi, n)
    A, F, L = jnp.asarray(a), jnp.asarray(f), jnp.asarray(lm)
    G1 = np.asarray(e1.absolute(A, F, L)[0])
    out = {}
    for tag, g2 in (("age", g2a), ("feh", g2f)):
        e2 = load(g2, order_age=orders[0], order_feh=orders[1])
        d = G1 - np.asarray(e2.absolute(A, F, L)[0])
        out[tag] = {"rms": float(np.sqrt(np.mean(d**2))),
                    "p95": float(np.percentile(np.abs(d), 95)),
                    "max": float(np.max(np.abs(d)))}
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    res: dict[str, object] = {}
    print("=" * 84)
    print("EMULATOR ACCURACY -- held out by stride, vs raw MIST tracks")
    print("held-out nodes sit at 2x the native spacing: PESSIMISTIC by ~4x for a C0 scheme")
    print("=" * 84)

    for orders, tag in (((1, 1), "C0 trilinear"), ((3, 3), "C1 tricubic")):
        print(f"\n-- {tag} --")
        rows_a = held_out("age", orders)
        rows_f = held_out("feh", orders)
        res[f"{tag}|age"] = rows_a
        res[f"{tag}|feh"] = rows_f
        print(f"{'held out':>18} {'RMS G':>9} {'max G':>9} {'RMS col':>9} "
              f"{'RMS G M<1':>10} {'RMS G M>=3':>11}")

        def _row(label, r):
            print(f"{label:>18} {r['all']['rms_G']:9.4f} {r['all']['max_G']:9.4f} "
                  f"{r['all']['rms_BPRP']:9.4f} "
                  f"{(r['M<1'] or {}).get('rms_G', np.nan):10.4f} "
                  f"{(r['M>=3'] or {}).get('rms_G', np.nan):11.4f}")

        for r in rows_a[::4]:
            _row(f"log age {r['age']:.2f}", r)
        for r in rows_f:
            _row(f"[Fe/H] {r['feh']:+.2f}", r)
        ma = np.median([r["all"]["rms_G"] for r in rows_a])
        mf = np.median([r["all"]["rms_G"] for r in rows_f])
        wa = np.max([r["all"]["max_G"] for r in rows_a])
        print(f"   median RMS(G): age axis {ma:.4f} mag | feh axis {mf:.4f} mag "
              f"| worst |dG| on age axis {wa:.4f}")

    print("\n-- grid-refinement bound on the PRODUCTION (stride 1) error --")
    for orders, tag in (((1, 1), "C0 trilinear"), ((3, 3), "C1 tricubic")):
        rc = refinement_check(orders)
        res[f"{tag}|refine"] = rc
        print(f"   {tag:14s} " + "  ".join(
            f"{k}: RMS {v['rms']:.4f} p95 {v['p95']:.4f} max {v['max']:.4f}"
            for k, v in rc.items()))

    (OUT / "emulator_accuracy.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {OUT / 'emulator_accuracy.json'}")


if __name__ == "__main__":
    main()
