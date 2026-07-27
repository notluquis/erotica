#!/usr/bin/env python3
"""First-pass audit of COUP (Chandra Orion Ultradeep Project) membership against Gaia DR3.

COUP = Getman et al. 2005, ApJS 160, 319 (source list; VizieR J/ApJS/160/319)
       Getman et al. 2005, ApJS 160, 353 (membership;  VizieR J/ApJS/160/353)

1616 X-ray sources in an 838 ks ACIS-I exposure of the Orion Nebula Cluster.
Published membership split: ~1315 confirmed members, 16 probable foreground
field stars, 159 likely background AGN, 33 weak/possibly-spurious, plus
embedded / HH / ambiguous classes.

What this script does
---------------------
1. Pulls the COUP source table and membership tables from VizieR.
2. Reconstructs a per-source membership label (VizieR does not ship one flag).
3. Cone-searches Gaia DR3 over the COUP field.
4. Propagates Gaia DR3 positions (epoch J2016.0) back to the COUP observation
   epoch (2003.04) using each source's own proper motion, so the crossmatch is
   done at a common epoch.
5. Nearest-neighbour crossmatch, with a chance-coincidence estimate from
   position-shifted control matches.
6. Reports match rate, G-magnitude distribution, and Gaia astrometric quality
   (RUWE, 5-parameter solution fraction, parallax S/N) split by COUP class.

Optionally runs step 3/6 for a comparison field (IC 348) to gauge whether the
ONC is a sensible first target or the hardest possible case.

Usage
-----
    python coup_gaia_audit.py                 # COUP x Gaia DR3
    python coup_gaia_audit.py --compare       # + IC 348 / NGC 1333 Gaia quality
    python coup_gaia_audit.py --radius 1.5    # match radius in arcsec

Caching: VizieR and Gaia results are cached as FITS next to this script under
./_cache so repeat runs are cheap.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

warnings.filterwarnings("ignore")

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")

# COUP observation: 2003 Jan 08-21, 838 ks. Mid-exposure epoch.
COUP_EPOCH = 2003.04
GAIA_EPOCH = 2016.0

# Comparison fields: (name, ra_deg, dec_deg, radius_arcmin, approx_distance_pc)
COMPARE_FIELDS = [
    ("ONC / COUP", 83.8065, -5.3962, 17.0, 400.0),
    ("IC 348", 56.1417, 32.1583, 10.0, 320.0),
    ("NGC 1333", 52.2958, 31.3083, 10.0, 293.0),
]


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def _cached(name: str, fn):
    """Run fn() and cache the resulting Table as FITS."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".fits")
    if os.path.exists(path):
        return Table.read(path)
    tab = fn()
    tab.write(path, overwrite=True)
    return tab


def fetch_coup() -> Table:
    """COUP main source table (1616 rows) with a reconstructed membership label."""
    from astroquery.vizier import Vizier

    def _pull():
        v = Vizier(row_limit=-1, columns=["**"])
        return v.get_catalogs("J/ApJS/160/319")[0]

    src = _cached("coup_main", _pull)

    v = Vizier(row_limit=-1, columns=["**"])
    memb_cats = {
        t.meta.get("name", "").split("/")[-1]: t
        for t in v.get_catalogs("J/ApJS/160/353")
    }
    t12 = memb_cats["table12"]  # 285 sources without optical/NIR counterparts
    t4 = memb_cats["table4"]  # 33 lightly-obscured uncertain sources

    # Default: the 1331 sources NOT in table12 are counterpart-confirmed members.
    label = np.array(["MEMBER"] * len(src), dtype="U12")
    coup_id = np.asarray(src["COUP"], dtype=int)
    index = {int(c): i for i, c in enumerate(coup_id)}

    # Getman+05 membership classes for the 285:
    #   EG          = likely background AGN
    #   OMC         = new embedded cloud member
    #   'OMC or EG?'= ambiguous embedded / AGN
    #   ONC         = new lightly-obscured ONC member
    #   HH          = Herbig-Haro shock
    #   Unc         = very weak, possibly spurious
    remap = {
        "EG": "AGN",
        "OMC": "EMBEDDED",
        "OMC or EG?": "AMBIG",
        "ONC": "NEW_ONC",
        "HH": "HH_SHOCK",
        "Unc": "SPURIOUS",
    }
    for cid, m in zip(np.asarray(t12["COUP"], dtype=int), t12["Memb"]):
        i = index.get(int(cid))
        if i is not None:
            label[i] = remap.get(str(m).strip(), "OTHER")

    # The 16 probable foreground field stars (table 4, Field == 'F').
    for cid, f in zip(np.asarray(t4["COUP"], dtype=int), t4["Field"]):
        if str(f).strip() == "F":
            i = index.get(int(cid))
            if i is not None:
                label[i] = "FOREGROUND"

    src["MembClass"] = label
    return src


def fetch_gaia(ra: float, dec: float, radius_arcmin: float, tag: str) -> Table:
    """Cone-search Gaia DR3 and return the astrometric/photometric columns we need."""
    from astroquery.gaia import Gaia

    def _pull():
        adql = f"""
        SELECT source_id, ra, dec, ra_error, dec_error,
               parallax, parallax_error, pmra, pmdec, pmra_error, pmdec_error,
               phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp,
               ruwe, astrometric_params_solved, astrometric_excess_noise,
               ipd_frac_multi_peak, ipd_gof_harmonic_amplitude,
               visibility_periods_used, phot_bp_rp_excess_factor
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                           CIRCLE('ICRS', {ra}, {dec}, {radius_arcmin / 60.0}))
        """
        job = Gaia.launch_job_async(adql)
        return job.get_results()

    return _cached(f"gaia_{tag}", _pull)


# --------------------------------------------------------------------------- #
# Crossmatch
# --------------------------------------------------------------------------- #
def gaia_coords_at_epoch(g: Table, epoch: float) -> SkyCoord:
    """Gaia DR3 positions propagated to `epoch` using each source's own PM.

    Sources without a 5-parameter solution (astrometric_params_solved == 3) keep
    their J2016.0 position; the resulting error is <~20 mas for ONC members,
    which is far below the Chandra positional uncertainty.
    """
    dt = epoch - GAIA_EPOCH
    pmra = np.nan_to_num(np.asarray(g["pmra"], dtype=float))  # mas/yr, *cos(dec)
    pmdec = np.nan_to_num(np.asarray(g["pmdec"], dtype=float))
    ra = np.asarray(g["ra"], dtype=float)
    dec = np.asarray(g["dec"], dtype=float)
    dec_new = dec + (pmdec * dt) / 3.6e6
    ra_new = ra + (pmra * dt) / 3.6e6 / np.cos(np.radians(dec))
    return SkyCoord(ra_new * u.deg, dec_new * u.deg)


def crossmatch(coup: Table, gaia: Table, radius_arcsec: float):
    """Nearest-neighbour match of COUP -> Gaia at the COUP epoch."""
    c_coup = SkyCoord(coup["RAJ2000"], coup["DEJ2000"], unit=(u.deg, u.deg))
    c_gaia = gaia_coords_at_epoch(gaia, COUP_EPOCH)
    idx, sep, _ = c_coup.match_to_catalog_sky(c_gaia)
    ok = sep < radius_arcsec * u.arcsec
    return idx, sep, ok, c_coup, c_gaia


def chance_rate(c_coup: SkyCoord, c_gaia: SkyCoord, radius_arcsec: float) -> float:
    """Chance-coincidence match fraction from position-shifted control matches."""
    rates = []
    for dra, ddec in [(30, 0), (-30, 0), (0, 30), (0, -30), (45, 45), (-45, -45)]:
        shifted = SkyCoord(
            c_coup.ra + (dra / 3600.0) * u.deg / np.cos(c_coup.dec.radian),
            c_coup.dec + (ddec / 3600.0) * u.deg,
        )
        _, s, _ = shifted.match_to_catalog_sky(c_gaia)
        rates.append(np.mean(s < radius_arcsec * u.arcsec))
    return float(np.mean(rates))


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def report_match(coup: Table, gaia: Table, idx, sep, ok, chance: float, radius: float):
    n = len(coup)
    print(f"\n{'=' * 74}\nCOUP x Gaia DR3 crossmatch  (r = {radius:.2f}\", epoch-propagated to {COUP_EPOCH})\n{'=' * 74}")
    print(f"COUP sources                : {n}")
    print(f"Gaia DR3 sources in field   : {len(gaia)}")
    print(f"Matched within {radius:.2f}\"        : {ok.sum()}  ({pct(ok.sum(), n):.1f}%)")
    print(f"Chance-coincidence rate     : {100 * chance:.1f}%  (position-shifted controls)")
    print(f"Excess over chance          : {pct(ok.sum(), n) - 100 * chance:.1f} percentage points")

    print(f"\nMatch rate vs radius:")
    for r in [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]:
        m = (sep < r * u.arcsec).sum()
        print(f"   r < {r:4.2f}\"  ->  {m:5d}  ({pct(m, n):5.1f}%)")

    print(f"\n{'-' * 74}\nBy COUP membership class\n{'-' * 74}")
    print(f"{'class':<12}{'N':>6}{'matched':>9}{'rate':>8}   {'median G':>9}{'med RUWE':>10}{'5-par %':>9}")
    for cls in ["MEMBER", "FOREGROUND", "NEW_ONC", "EMBEDDED", "AMBIG", "AGN", "SPURIOUS", "HH_SHOCK"]:
        m = coup["MembClass"] == cls
        if not m.any():
            continue
        mm = m & ok
        nm = mm.sum()
        if nm:
            sub = gaia[idx[mm]]
            gmag = np.nanmedian(np.asarray(sub["phot_g_mean_mag"], dtype=float))
            ruwe = np.nanmedian(np.asarray(sub["ruwe"], dtype=float))
            f5 = 100.0 * np.mean(np.asarray(sub["astrometric_params_solved"]) == 31)
            extra = f"   {gmag:9.2f}{ruwe:10.2f}{f5:8.0f}%"
        else:
            extra = "   " + " " * 28
        print(f"{cls:<12}{m.sum():>6}{nm:>9}{pct(nm, m.sum()):>7.1f}%{extra}")

    # Gaia quality of the matched members
    mem = (coup["MembClass"] == "MEMBER") & ok
    sub = gaia[idx[mem]]
    print(f"\n{'-' * 74}\nGaia astrometric quality of matched COUP MEMBERS (N={mem.sum()})\n{'-' * 74}")
    aps = np.asarray(sub["astrometric_params_solved"])
    ruwe = np.asarray(sub["ruwe"], dtype=float)
    plx = np.asarray(sub["parallax"], dtype=float)
    eplx = np.asarray(sub["parallax_error"], dtype=float)
    g = np.asarray(sub["phot_g_mean_mag"], dtype=float)
    n5 = (aps == 31).sum()
    print(f"5-parameter astrometric solutions : {n5} ({pct(n5, len(sub)):.1f}%)")
    print(f"2-parameter (position only)       : {(aps == 3).sum()} ({pct((aps == 3).sum(), len(sub)):.1f}%)")
    print(f"6-parameter                       : {(aps == 95).sum()} ({pct((aps == 95).sum(), len(sub)):.1f}%)")
    with np.errstate(invalid="ignore"):
        good_ruwe = np.nansum(ruwe < 1.4)
        print(f"RUWE < 1.4                        : {int(good_ruwe)} ({pct(good_ruwe, np.isfinite(ruwe).sum()):.1f}% of those with RUWE)")
        print(f"median RUWE                       : {np.nanmedian(ruwe):.2f}")
        snr = plx / eplx
        for t in [3, 5, 10, 20]:
            k = np.nansum(snr > t)
            print(f"parallax S/N > {t:<3d}                : {int(k)} ({pct(k, len(sub)):.1f}% of matched members)")
        print(f"median parallax                   : {np.nanmedian(plx):.4f} mas  -> {1000 / np.nanmedian(plx):.0f} pc")
    print(f"\nG-magnitude distribution of matched members:")
    for lo, hi in [(0, 12), (12, 14), (14, 16), (16, 18), (18, 19), (19, 20), (20, 30)]:
        k = np.sum((g >= lo) & (g < hi))
        print(f"   {lo:2d} <= G < {hi:<2d} : {k:5d} ({pct(k, len(sub)):5.1f}%)")

    # Unmatched members: why?
    unmem = (coup["MembClass"] == "MEMBER") & ~ok
    print(f"\n{'-' * 74}\nUNMATCHED COUP members (N={unmem.sum()}) - diagnostic\n{'-' * 74}")
    for col, lab in [("Vmag", "has optical V"), ("Ksmag", "has 2MASS Ks"), ("AV", "has A_V")]:
        if col in coup.colnames:
            arr = np.asarray(coup[col][unmem], dtype=float)
            print(f"   {lab:<16}: {np.isfinite(arr).sum():4d} / {unmem.sum()}   median = {np.nanmedian(arr):.2f}")
    if "AV" in coup.colnames:
        av_m = np.asarray(coup["AV"][mem], dtype=float)
        av_u = np.asarray(coup["AV"][unmem], dtype=float)
        print(f"   median A_V matched = {np.nanmedian(av_m):.2f}   unmatched = {np.nanmedian(av_u):.2f}")


def report_audit(coup: Table, gaia: Table, idx, ok):
    """The actual audit: do Gaia astrometry and COUP X-ray labels agree?

    Two directional tests, kept separate because they measure different things:

    (A) PURITY of the COUP member list - of COUP sources labelled MEMBER that
        have usable Gaia astrometry, how many are astrometrically inconsistent
        with the ONC?  These are member-list contamination *candidates*.

    (B) The FOREGROUND label - COUP flagged 16 sources as probable foreground
        field stars using Jones & Walker (1988) photographic proper motions.
        Gaia parallaxes test that call directly and independently.  This is the
        cleanest falsifiable check in the whole catalog.

    NOTE this cannot measure COUP's completeness against Gaia, because a Gaia
    source in the field with ONC-like astrometry but no COUP detection is not
    necessarily a COUP failure - it may simply be X-ray faint.  Completeness
    needs the reverse test and a stated L_X limit; see the write-up.
    """
    plx = np.asarray(gaia["parallax"], dtype=float)
    eplx = np.asarray(gaia["parallax_error"], dtype=float)
    pmra = np.asarray(gaia["pmra"], dtype=float)
    pmdec = np.asarray(gaia["pmdec"], dtype=float)
    aps = np.asarray(gaia["astrometric_params_solved"])
    ruwe = np.asarray(gaia["ruwe"], dtype=float)

    # Proper-motion reference is calibrated from the matched members themselves
    # (see below).  A fixed literature tolerance of ~4 mas/yr corresponds to
    # ~7.6 km/s at 400 pc, i.e. several times the ONC dispersion -- that is a
    # runaway filter, not a membership discriminant, so we do not use one.

    print(f"\n{'=' * 74}\nAUDIT (A): astrometric consistency of the COUP MEMBER list\n{'=' * 74}")
    mem = (coup["MembClass"] == "MEMBER") & ok
    j = idx[mem]
    usable = (aps[j] != 3) & np.isfinite(plx[j]) & (plx[j] / eplx[j] > 5) & (ruwe[j] < 1.4)
    print(f"COUP MEMBER with Gaia match          : {mem.sum()}")
    print(f"  ... with usable astrometry          : {usable.sum()}"
          f"  (5/6-par, parallax S/N>5, RUWE<1.4)")
    if not usable.sum():
        return

    p, pr, pd = plx[j][usable], pmra[j][usable], pmdec[j][usable]
    ep = eplx[j][usable]

    # Calibrate the ONC parallax distribution FROM the data, robustly.  A hard
    # cut against a single literature parallax would count the cluster's real
    # line-of-sight depth as "contamination".
    PLX_ONC = float(np.median(p))
    mad = float(np.median(np.abs(p - PLX_ONC)))
    sig_int = 1.4826 * mad  # robust sigma of the member parallax distribution
    print(f"  cluster parallax (median)            : {PLX_ONC:.4f} mas -> {1000 / PLX_ONC:.0f} pc")
    print(f"  robust intrinsic spread (1.48*MAD)   : {sig_int:.4f} mas"
          f"  (= real cloud depth + astrometric error)")

    # per-source significance folds in BOTH the cluster spread and the measurement error
    sig_tot = np.hypot(sig_int, ep)
    nsig = (p - PLX_ONC) / sig_tot
    for t in [3, 4, 5]:
        out = np.abs(nsig) > t
        print(f"  parallax outliers >{t}sigma            : {out.sum():4d}  ({pct(out.sum(), usable.sum()):5.1f}%)"
              f"   [front {int((nsig > t).sum())} / behind {int((nsig < -t).sum())}]")

    # Proper motion, calibrated the same way as parallax: robust centre + robust
    # dispersion measured FROM the matched members, so the test has comparable
    # sensitivity to the parallax test instead of being a runaway-only filter.
    pm0 = (float(np.median(pr)), float(np.median(pd)))
    s_ra = 1.4826 * float(np.median(np.abs(pr - pm0[0])))
    s_de = 1.4826 * float(np.median(np.abs(pd - pm0[1])))
    print(f"  cluster PM (median)                  : ({pm0[0]:+.3f}, {pm0[1]:+.3f}) mas/yr")
    print(f"  robust PM dispersion (1.48*MAD)      : ({s_ra:.3f}, {s_de:.3f}) mas/yr"
          f"  = ({s_ra * 4.74 / PLX_ONC:.1f}, {s_de * 4.74 / PLX_ONC:.1f}) km/s")
    epr = np.asarray(gaia["pmra_error"], dtype=float)[j][usable]
    epd = np.asarray(gaia["pmdec_error"], dtype=float)[j][usable]
    n_pm = np.hypot((pr - pm0[0]) / np.hypot(s_ra, epr), (pd - pm0[1]) / np.hypot(s_de, epd))
    for t in [3, 4, 5]:
        print(f"  PM outliers >{t}sigma (2D)             : {int((n_pm > t).sum()):4d}"
              f"  ({pct((n_pm > t).sum(), usable.sum()):5.1f}%)")
    bad_pm = n_pm > 3
    disc = (np.abs(nsig) > 3) | bad_pm
    print(f"  EITHER (>3sigma in parallax OR PM)   : {disc.sum():4d}  ({pct(disc.sum(), usable.sum()):5.1f}%)")
    print(f"  BOTH  (>3sigma in parallax AND PM)   : {int(((np.abs(nsig) > 3) & bad_pm).sum()):4d}"
          f"  <- strongest contamination candidates")
    print(f"  -> these are contamination CANDIDATES, not verdicts.")
    print(f"     A Gaussian core would give ~0.3% beyond 3sigma; the excess over that")
    print(f"     is the quantity of interest, and it is asymmetric (front vs behind),")
    print(f"     which is the signature of genuine field-star contamination.")

    print(f"\n{'=' * 74}\nAUDIT (B): the 16 COUP 'probable foreground field star' calls\n{'=' * 74}")
    print("COUP called these foreground from Jones & Walker (1988) photographic proper")
    print("motions. Gaia DR3 parallaxes test each call independently. Each source is")
    print(f"compared against the cluster distribution N({PLX_ONC:.3f}, {sig_int:.3f}) mas fitted above.\n")
    fgm = coup["MembClass"] == "FOREGROUND"
    print(f"{'COUP':>6} {'G':>7} {'plx':>8} {'e_plx':>7} {'d(pc)':>7} {'RUWE':>6} {'n_sig':>7}  verdict")
    conf = amb = contra = 0
    for i in np.where(fgm)[0]:
        cid = int(coup["COUP"][i])
        if not ok[i]:
            print(f"{cid:>6} {'':>7} {'':>8} {'':>7} {'':>7} {'':>6} {'':>7}  no Gaia counterpart")
            amb += 1
            continue
        k = idx[i]
        pv, e, r = plx[k], eplx[k], ruwe[k]
        g = float(gaia["phot_g_mean_mag"][k])
        if not np.isfinite(pv) or e <= 0 or pv / e < 3:
            print(f"{cid:>6} {g:>7.2f} {'':>8} {'':>7} {'':>7} {r:>6.2f} {'':>7}  inconclusive (no/weak parallax)")
            amb += 1
            continue
        ns = (pv - PLX_ONC) / np.hypot(sig_int, e)
        d = 1000.0 / pv
        if not np.isfinite(r) or r > 1.4:
            v = "inconclusive (RUWE unreliable)"
            amb += 1
        elif ns > 3:
            v = "CONFIRMED foreground"
            conf += 1
        elif abs(ns) <= 2:
            v = "*** ONC-CONSISTENT -- label contradicted ***"
            contra += 1
        elif abs(ns) <= 3:
            v = "*** contradicted, but MARGINAL (2-3 sigma) ***"
            contra += 1
        else:
            v = "behind the ONC -- not foreground either"
            contra += 1
        print(f"{cid:>6} {g:>7.2f} {pv:>8.3f} {e:>7.3f} {d:>7.0f} {r:>6.2f} {ns:>+7.1f}  {v}")
    print(f"\n  COUP foreground call CONFIRMED by Gaia   : {conf} / 16")
    print(f"  COUP foreground call CONTRADICTED        : {contra} / 16   <-- these are ONC members")
    print(f"  inconclusive / no counterpart            : {amb} / 16")

    print(f"\n{'=' * 74}\nAUDIT (C): the 159 'background AGN' and 42 'embedded' calls\n{'=' * 74}")
    for cls, expect in [("AGN", "expect ~0% Gaia -- extragalactic, behind the cloud"),
                        ("EMBEDDED", "expect ~0% Gaia -- A_V >> 10"),
                        ("SPURIOUS", "expect ~0% Gaia if genuinely spurious"),
                        ("AMBIG", "'OMC or EG?' -- Gaia detection would break the tie")]:
        m = coup["MembClass"] == cls
        print(f"  {cls:<10} N={m.sum():>4}  Gaia-matched={int((m & ok).sum()):>3}"
              f"  ({pct((m & ok).sum(), m.sum()):.1f}%)   {expect}")


def report_field_quality(name, ra, dec, rad, dist_pc, tag):
    """Gaia DR3 data-quality snapshot for a field - for target selection."""
    g = fetch_gaia(ra, dec, rad, tag)
    aps = np.asarray(g["astrometric_params_solved"])
    ruwe = np.asarray(g["ruwe"], dtype=float)
    gm = np.asarray(g["phot_g_mean_mag"], dtype=float)
    plx = np.asarray(g["parallax"], dtype=float)
    eplx = np.asarray(g["parallax_error"], dtype=float)
    area = np.pi * (rad / 60.0) ** 2  # deg^2
    n5 = (aps == 31).sum() + (aps == 95).sum()
    exp_plx = 1000.0 / dist_pc
    # sources plausibly at the cluster distance with decent parallax
    near = np.abs(plx - exp_plx) < 3 * np.maximum(eplx, 0.05)
    print(f"\n{name}  (r={rad:.0f}', d~{dist_pc:.0f} pc)")
    print(f"   Gaia DR3 sources        : {len(g)}   ({len(g) / area:.0f} / deg^2)")
    print(f"   with parallax+PM (5/6p) : {n5} ({pct(n5, len(g)):.1f}%)")
    print(f"   median G                : {np.nanmedian(gm):.2f}")
    print(f"   median RUWE             : {np.nanmedian(ruwe):.2f}")
    print(f"   RUWE < 1.4              : {pct(np.nansum(ruwe < 1.4), np.isfinite(ruwe).sum()):.1f}%")
    print(f"   parallax S/N > 10       : {pct(np.nansum(plx / eplx > 10), len(g)):.1f}%")
    print(f"   within 3sig of {exp_plx:.2f} mas : {near.sum()}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=1.0, help="match radius (arcsec)")
    ap.add_argument("--compare", action="store_true", help="also profile IC 348 / NGC 1333")
    args = ap.parse_args()

    print("Fetching COUP catalog from VizieR ...", file=sys.stderr)
    coup = fetch_coup()
    import collections

    print("COUP membership label counts:", dict(collections.Counter(coup["MembClass"])), file=sys.stderr)

    print("Querying Gaia DR3 over the COUP field ...", file=sys.stderr)
    ra0, dec0, rad0, _ = COMPARE_FIELDS[0][1], COMPARE_FIELDS[0][2], COMPARE_FIELDS[0][3], None
    gaia = fetch_gaia(ra0, dec0, rad0, "onc")

    idx, sep, ok, c_coup, c_gaia = crossmatch(coup, gaia, args.radius)
    ch = chance_rate(c_coup, c_gaia, args.radius)
    report_match(coup, gaia, idx, sep, ok, ch, args.radius)
    report_audit(coup, gaia, idx, ok)

    if args.compare:
        print(f"\n{'=' * 74}\nGaia DR3 field-quality comparison (target selection)\n{'=' * 74}")
        for nm, ra, dec, rad, d in COMPARE_FIELDS:
            report_field_quality(nm, ra, dec, rad, d, nm.split()[0].lower().replace("/", ""))


if __name__ == "__main__":
    main()
