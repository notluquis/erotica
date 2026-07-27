#!/usr/bin/env python3
"""Where do NGC 6383's King parameters sit in the modern Gaia literature?

WHY THIS EXISTS
---------------
P01 reports `R_c` and `R_t` for NGC 6383 from a King fit and compares them to
four historical determinations. This places them instead in the two large
homogeneous Gaia-era reference samples, and asks the prior question: **is the
King model the right model for a cluster this young?**

The answer turns out to be measurable rather than rhetorical. NGC 6383 is
~3.5 Myr (log age ~ 6.55). The reference samples do not contain clusters that
young, so the comparison is an extrapolation, not an interpolation.

SOURCES (fetched live from VizieR, so nothing here is transcribed by hand)
-------------------------------------------------------------------------
* Tarricq et al. 2022, A&A 659, A59 (`2022A&A...659A..59T`), VizieR J/A+A/659/A59
  -- 467 OCs searched to 50 pc from centre, King fits for 233, plus elliptical
  Gaussian-mixture axis ratios for core / tail / halo.
* Zhong et al. 2022, AJ 164, 54 (`2022AJ....164...54Z`), VizieR J/AJ/164/54
  -- 256 OCs, two-component model (King core + log-Gaussian outer halo) with
  four radii, after finding that "the radial density profile in the outer region
  for most open clusters deviates from the King profile".

USAGE
-----
    python tools/validation/king_literature_context.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# P01's adopted values, converted at the paper's 1.11 kpc distance.
NGC6383 = {"R_c_pc": 0.63, "R_t_pc": 17.4, "C": 1.43, "log_age": 6.55}


def pct(sample, value):
    s = np.asarray(sample, float)
    s = s[np.isfinite(s)]
    return 100.0 * float(np.mean(s < value))


def summary(sample, label, value=None):
    s = np.asarray(sample, float)
    s = s[np.isfinite(s)]
    line = (f"  {label:30s} n={len(s):4d}  16/50/84 = "
            f"{np.percentile(s, 16):7.2f} {np.percentile(s, 50):7.2f} {np.percentile(s, 84):7.2f}")
    if value is not None:
        line += f"   NGC 6383 at {pct(s, value):5.1f} pct"
    print(line)
    return {"n": len(s), "p16": float(np.percentile(s, 16)),
            "p50": float(np.percentile(s, 50)), "p84": float(np.percentile(s, 84)),
            "ngc6383_percentile": None if value is None else pct(s, value)}


def main():
    from astroquery.vizier import Vizier

    v = Vizier(columns=["*"], row_limit=-1)
    out = {"ngc6383": NGC6383, "tarricq2022": {}, "zhong2022": {}}

    # ---- Tarricq+2022 -----------------------------------------------------
    t = v.get_catalogs("J/A+A/659/A59")[0]
    rc, rt = np.asarray(t["Rc"], float), np.asarray(t["Rt"], float)
    age, ba = np.asarray(t["logAgeNN"], float), np.asarray(t["b/aCore"], float)
    ok = np.isfinite(rc) & np.isfinite(rt) & (rc > 0) & (rt > 0)
    conc = np.log10(rt / rc)

    print("=" * 78)
    print("Tarricq+2022 -- 233 King fits among 467 OCs (pc)")
    print("=" * 78)
    out["tarricq2022"]["R_c"] = summary(rc[ok], "R_c", NGC6383["R_c_pc"])
    out["tarricq2022"]["R_t"] = summary(rt[ok], "R_t", NGC6383["R_t_pc"])
    out["tarricq2022"]["C"] = summary(conc[ok], "C = log10(R_t/R_c)", NGC6383["C"])

    a = age[ok]
    print(f"\n  log age of the King-fitted sample: min {a.min():.2f}  "
          f"median {np.median(a):.2f}  max {a.max():.2f}")
    print(f"  NGC 6383 log age {NGC6383['log_age']:.2f} -> "
          f"{'INSIDE' if a.min() <= NGC6383['log_age'] else 'BELOW THE ENTIRE SAMPLE'}")
    print(f"  clusters younger than NGC 6383 in this sample: {int((a < NGC6383['log_age']).sum())}")
    print(f"  youngest is {10**a.min() / 1e6:.0f} Myr; NGC 6383 is "
          f"{10**NGC6383['log_age'] / 1e6:.1f} Myr "
          f"({10**(a.min() - NGC6383['log_age']):.0f}x younger)")
    out["tarricq2022"]["log_age_min"] = float(a.min())
    out["tarricq2022"]["n_younger_than_ngc6383"] = int((a < NGC6383["log_age"]).sum())

    b = ba[np.isfinite(ba)]
    print(f"\n  Core axis ratio b/a: 16/50/84 = {np.percentile(b, 16):.2f} "
          f"{np.percentile(b, 50):.2f} {np.percentile(b, 84):.2f}")
    print(f"    fraction with b/a < 0.9: {np.mean(b < 0.9):.1%}    "
          f"< 0.8: {np.mean(b < 0.8):.1%}")
    print("    -> circular symmetry, which the King fit assumes, is the exception.")
    out["tarricq2022"]["b_over_a_core"] = {
        "p16": float(np.percentile(b, 16)), "p50": float(np.percentile(b, 50)),
        "p84": float(np.percentile(b, 84)),
        "frac_below_0.9": float(np.mean(b < 0.9)), "frac_below_0.8": float(np.mean(b < 0.8)),
    }

    # ---- Zhong+2022 -------------------------------------------------------
    z = v.get_catalogs("J/AJ/164/54")[0]
    zage = np.asarray(z["Age"], float)
    zrc, zrt = np.asarray(z["Radc"], float), np.asarray(z["Radt"], float)
    zro, zre = np.asarray(z["Rado"], float), np.asarray(z["Rade"], float)

    print()
    print("=" * 78)
    print("Zhong+2022 -- 256 OCs, King core + log-Gaussian outer halo (pc)")
    print("=" * 78)
    out["zhong2022"]["R_c"] = summary(zrc, "r_c (King core)", NGC6383["R_c_pc"])
    out["zhong2022"]["R_t"] = summary(zrt, "r_t (core boundary)", NGC6383["R_t_pc"])
    out["zhong2022"]["R_o"] = summary(zro, "r_o (halo scale)")
    out["zhong2022"]["R_e"] = summary(zre, "r_e (halo boundary)")
    za = zage[np.isfinite(zage)]
    print(f"\n  Age column: min {za.min():.3g}  median {np.median(za):.3g}  max {za.max():.3g}")
    out["zhong2022"]["age_min"] = float(za.min())
    out["zhong2022"]["age_median"] = float(np.median(za))

    print("\n  Zhong's headline: the outer profile of MOST OCs deviates from King,")
    print("  which is why they need four radii instead of two. A single King R_t is")
    print("  therefore not the field's current description of an OC's outer structure.")

    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {Path(__file__).with_suffix('.json')}")


if __name__ == "__main__":
    main()
