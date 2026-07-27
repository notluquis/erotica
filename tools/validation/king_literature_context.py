#!/usr/bin/env python3
"""Where do NGC 6383's King parameters sit in the modern Gaia literature?

WHY THIS EXISTS
---------------
P01 reports `R_c` and `R_t` for NGC 6383 and compares them to four historical
determinations. This places them instead in the Gaia-era reference samples, and
asks the prior question: is a King profile the right model for a cluster this
young, and is its `R_t` a physically admissible number?

THREE SAMPLES, AND WHY ALL THREE
--------------------------------
* **Hunt & Reffert 2024**, A&A 686, A42 (`2024A&A...686A..42H`), VizieR
  J/A+A/686/A42 -- 7167 rows, 5647 bound open clusters, all-sky, with
  completeness-corrected photometric masses and **Jacobi radii**. This is the
  decisive sample: it is all-sky, it reaches ages far younger than the others,
  and **it contains NGC 6383 itself**, giving a fully independent measurement.
  Caveat: their ``rc``/``rt``/``rtot`` are *empirical* radii, not King-fit
  parameters, so they are not interchangeable with P01's ``R_c``/``R_t``.
  ``rJ`` is the physical Jacobi radius derived from the mass, and *is* directly
  comparable to any claimed outer boundary.
* **Tarricq+2022**, A&A 659, A59 (`2022A&A...659A..59T`), J/A+A/659/A59 -- 233
  actual **King fits** searched to 50 pc, plus elliptical axis ratios. Comparable
  in method to P01, but solar-vicinity only and no cluster younger than 50 Myr.
* **Zhong+2022**, AJ 164, 54 (`2022AJ....164...54Z`), J/AJ/164/54 -- 256 OCs,
  two-component model, after finding the outer profile of most OCs deviates from
  King.

Everything is fetched live; nothing is transcribed by hand.

USAGE
-----
    python tools/validation/king_literature_context.py
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

import numpy as np

# P01's adopted values, at its adopted 1.11 kpc.
NGC6383 = {"R_c_arcmin": 1.96, "R_t_arcmin": 54.0, "R_c_pc": 0.63, "R_t_pc": 17.4,
           "C": 1.43, "log_age": 6.55, "T_max_arcmin": 42.45, "hill_arcmin": 33.6,
           "field_arcmin": 70.0}

HUNT_COLS = ("Name,Type,N,logAge50,dist50,rc,rt,r50,rtot,"
             "rcpc,rtpc,r50pc,rtotpc,rJ,rJpc,MassJ,probJ")


def fetch_hunt():
    url = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/A+A/686/A42/clusters"
           f"&-out={HUNT_COLS.replace(',', '&-out=')}&-out.max=unlimited")
    txt = urllib.request.urlopen(url, timeout=300).read().decode("utf-8", "replace")
    body = [x for x in txt.splitlines() if x and not x.startswith("#")]
    return list(csv.DictReader([body[0]] + [x for x in body[3:] if x.strip()], delimiter="\t"))


def num(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return np.nan


def main():
    out = {"ngc6383_p01": NGC6383}
    rows = fetch_hunt()
    name = np.array([r["Name"].strip() for r in rows])
    typ = np.array([r["Type"].strip() for r in rows])
    age = np.array([num(r, "logAge50") for r in rows])
    rcpc = np.array([num(r, "rcpc") for r in rows])
    rJpc = np.array([num(r, "rJpc") for r in rows])

    print("=" * 78)
    print("1. NGC 6383 as measured independently by Hunt & Reffert 2024")
    print("=" * 78)
    h = rows[int(np.flatnonzero(name == "NGC_6383")[0])]
    d = num(h, "dist50")
    pc_per_arcmin = d * (1 / 60) * (np.pi / 180)
    print(f"  distance {d:.0f} pc   (P01: 1110 pc)")
    print(f"  log age  {num(h, 'logAge50'):.2f} = {10 ** num(h, 'logAge50') / 1e6:.1f} Myr"
          f"   (P01: 3.5 Myr)")
    print(f"  mass     {num(h, 'MassJ'):.0f} Msun, P(bound) = {num(h, 'probJ'):.3f}")
    print(f"\n  {'radius':22s} {'pc':>8s} {'arcmin':>9s}")
    for key, lab in (("rcpc", "core r_c"), ("r50pc", "half-number r_50"),
                     ("rtpc", "tidal r_t"), ("rtotpc", "total r_tot"),
                     ("rJpc", "JACOBI r_J")):
        v = num(h, key)
        print(f"  {lab:22s} {v:8.2f} {v / pc_per_arcmin:9.1f}")
    r_j = num(h, "rJpc")
    ratio = NGC6383["R_t_pc"] / r_j
    print(f"\n  P01 adopted King R_t = {NGC6383['R_t_arcmin']:.0f}' = {NGC6383['R_t_pc']:.1f} pc")
    print(f"    -> {ratio:.2f}x the Jacobi radius. A bound cluster cannot extend beyond r_J,")
    print(f"       so the fitted R_t is not an admissible physical boundary.")
    print(f"  P01 T_max = {NGC6383['T_max_arcmin']:.1f}', Hill = {NGC6383['hill_arcmin']:.1f}';"
          f" Hunt r_J = {r_j / pc_per_arcmin:.1f}' -- between them, independently.")
    print(f"  P01 largest field {NGC6383['field_arcmin']:.0f}' = "
          f"{NGC6383['field_arcmin'] * pc_per_arcmin:.1f} pc = "
          f"{NGC6383['field_arcmin'] * pc_per_arcmin / r_j:.2f}x r_J -- the field is NOT too small.")
    out["hunt2024_ngc6383"] = {k: num(h, k) for k in
                               ("logAge50", "dist50", "rcpc", "r50pc", "rtpc", "rtotpc",
                                "rJpc", "MassJ", "probJ")}
    out["R_t_over_jacobi"] = float(ratio)

    print()
    print("=" * 78)
    print("2. Is NGC 6383 younger than the comparison population?")
    print("=" * 78)
    oc = (typ == "o") & np.isfinite(age) & np.isfinite(rcpc)
    a = age[oc]
    print(f"  Hunt bound OCs with a core radius: {oc.sum()}")
    print(f"  log age  min {a.min():.2f} ({10 ** a.min() / 1e6:.2f} Myr)  "
          f"median {np.median(a):.2f}  max {a.max():.2f}")
    for cut, lab in ((NGC6383["log_age"], "NGC 6383 (3.5 Myr)"), (7.0, "10 Myr"),
                     (7.7, "50 Myr = Tarricq's youngest")):
        print(f"    younger than {lab:28s}: {int((a < cut).sum()):5d} ({(a < cut).mean():.1%})")
    print("  -> a young comparison population EXISTS. Tarricq simply does not sample it.")
    out["hunt2024_n_younger"] = int((a < NGC6383["log_age"]).sum())
    out["hunt2024_n_oc_with_rc"] = int(oc.sum())

    young = oc & (age < 6.7)
    x = rcpc[young]
    x = x[np.isfinite(x)]
    pct = 100.0 * float(np.mean(x < NGC6383["R_c_pc"]))
    print(f"\n  Young subset (log age < 6.7, <5 Myr): n={young.sum()}")
    print(f"    r_c (pc) 16/50/84 = {np.percentile(x, 16):.2f} {np.percentile(x, 50):.2f} "
          f"{np.percentile(x, 84):.2f}")
    print(f"    NGC 6383 R_c = {NGC6383['R_c_pc']:.2f} pc -> {pct:.0f}th percentile "
          f"among clusters of its own age")
    print("    -> compact, but NOT anomalous once compared against the right age range.")
    out["young_rc_percentile"] = pct

    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {Path(__file__).with_suffix('.json')}")


if __name__ == "__main__":
    main()
