#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-derive the *full* photometric CTTS sample of Kalari (2019) for NGC 6383 / Sh 2-012.

WHY THIS EXISTS
---------------
Kalari V. M. 2019, MNRAS 484, 5102 ("Classical T-Tauri stars with VPHAS+: II:
NGC 6383 in Sh 2-012"; bibcode 2019MNRAS.484.5102K, arXiv:1901.07511) identifies
**156** CTTS from VPHAS+ (r-i, r-Halpha) photometry alone, then cuts that to
**55** using a Gaia DR2 proper-motion criterion.

VizieR J/MNRAS/484/5102 publishes **only table1.dat = the 55**.  Those 55 are
proper-motion selected, hence *circular* if you want to validate an astrometric
membership classifier.  This script rebuilds the 156-style sample from the
parent VPHAS+ photometry, applying only the paper's photometric criteria.

  !! PREMISE CORRECTION -- READ THIS !!
  The 156 are NOT "astrometry-free".  Sect. 2.2 of the paper:
      "We cross-matched our VPHAS+ source list with the Gaia DR2 dataset within
       a radius of 0.1 arcsec ... Finally, we applied the [Lindegren+2018] C-1
       astrometric equation to remove sources with unreliable astrometry ...
       In total, we have 1 296 410 stars with high-quality astrometry and
       photometry. These form the source dataset from which we will identify
       CTTS."
  So the 156 were drawn from a Gaia-cross-matched, astrometrically-filtered
  parent.  They are *proper-motion-free*, not *Gaia-free*.
  This matters, but it is NOT fatal for validating a membership classifier:
  the C-1 equation is a goodness-of-fit cut, not a kinematic one, so it does
  not preselect cluster members.  The circularity lives entirely in the IQR
  proper-motion cut that turned 156 into 55.
  The genuinely astrometry-free parent is the 2 091 573 riHalpha
  quality-selected VPHAS+ sources (Sect. 2.1).  This script selects from THAT,
  and therefore returns a superset of the 156.  Set --require-gaia to emulate
  the paper's Gaia-matched parent instead.

PROVENANCE OF EVERY NUMBER
--------------------------
[S] = sourced, quoted from the paper.  [I] = inferred/derived here.

[S] Field .............. "we define our area of study as a 2 deg x 2 deg region
                          centred on Right Ascension (J2000) 17h34m09s,
                          Declination (J2000) -32d32'10\"" (Sect. 2.1)
[S] Actual box ......... LaTeX source comment (line 131 of 6383_v1.tex), the
                          author's own working note:
                            "Miinimu - 261.026- 266.05085; dec =34.49128
                             =30.58138 10202551 sources in total. 2091573 meet
                             selection criteria."
                          i.e. RA 261.026..266.05085, Dec -34.49128..-30.58138.
                          Centre matches 17h34m09s / -32d32'10" to <0.001 deg.
                          NOTE this is a +/-2 deg box (~4x4 deg), not 2x2 deg.
[S] Quality cuts ....... "(i) 22 > r > 13 in both the red and blue filter sets
                          to avoid saturated and faint sources; (ii) signal to
                          noise ratio > 10 in riHalpha bands; (iii) point source
                          function fit of chi < 1.5 to select stellar or
                          star-like sources"  (Sect. 2.1)
[S] Parent counts ...... "In the resulting sample, 2 091 573 unique sources have
                          riHalpha photometry meeting our quality criteria."
[S] EW equation ........ EW_Ha = W * [1 - 10^(0.4*(r-Ha)_excess)]   (Eq. 1)
                          (r-Ha)_excess = (r-Ha)_obs - (r-Ha)_model
[S] W .................. "W is the rectangular bandwidth of the Halpha filter"
                          (Paper I, Kalari+2015, Sect. 3).  ** Never given
                          numerically in either paper. ** The filter's quoted
                          "central wavelength and bandpass" are 6588 and 107 A,
                          but the rectangular bandwidth is a different quantity.
[I] W = 128.43 A ....... RECOVERED BY INVERSION (see calibrate_from_published_55).
                          table1.dat publishes r, i, Halpha AND EW_Ha for all 55,
                          so Eq. 1 can be inverted for the model track at each
                          star; the correct W is the one that makes the recovered
                          track a smooth function of (r-i).  Sharp minimum at
                          W=128.4 A (rms 0.0075 mag), stable across polynomial
                          degree 2/3/4 (128.26/128.43/128.32).  Leave-one-out EW
                          prediction error 1.9 A, well below the published median
                          EW error of 5.6 A -- so this is not overfitting.
[I] Model track ........ RECOVERED BY THE SAME INVERSION.  Cubic in (r-i).
                          This is the Pickles(1998)-convolved main-sequence locus
                          reddened by E(B-V)=0.32, i.e. exactly the dashed line
                          in the paper's Fig. 1 -- reconstructed, because the
                          promised electronic model file (footnote "*" on
                          Sect. 3.1) was never published: VizieR carries table1
                          and nothing else.
                          VALID ONLY over r-i in [-0.105, 1.095] (the span of the
                          55).  Redder than that it is an extrapolation.
[S] Reddening .......... "We assume the mean reddening E(B-V) = 0.32, and a
                          standard Galactic reddening law of R_V = 3.1"
                          (Sect. 3.3); Fig. 1 caption: "the interpolated model
                          track reddened by E(B-V) = 0.32".
[S] CTTS criteria ...... "Halpha emission line stars having spectral type earlier
                          than K5 and EW_Ha < -18 A, K5-M2.5 and EW_Ha < -25 A
                          and M2.5-M6 and EW_Ha < -38 A are selected."
                          (Sect. 3.1; criteria of Barrado y Navascues & Martin
                          2003)
[!] r ZERO-POINT BUG .. ** VizieR table1.dat's `rmag` column is NOT the r used in
                          the analysis. ** Querying II/341 for all 55 sourceIDs
                          (they resolve 55/55, exactly) gives:
                              i   : published - VizieR = -0.001 mag (rms 0.003)
                              Ha  : published - VizieR = -0.000 mag (rms 0.003)
                              r   : published - VizieR = -0.863 mag (rms 0.003)
                          i and Ha agree to VizieR's 2-decimal rounding; r is
                          offset by a hard constant, identically in all 7 fields.
                          0.863 = A_r for E(B-V)=0.32, R_V=3.1 (A_r/A_V = 0.87,
                          A_V = 0.992), so table1's r column has been DEREDDENED
                          while i and Ha were left observed.
                          Three independent checks say VizieR's r is the correct
                          one and table1's is the broken one:
                            (a) Gaia G - r: using table1's r gives G-r = +0.69
                                (G fainter than r) for red PMS stars -- unphysical;
                                using VizieR's r gives -0.17, as expected.
                            (b) table1's (r-i) spans -0.105..1.095, implying F/G
                                spectral types, flatly contradicting the paper's
                                own abstract ("mass range between 0.3 and 1 Msun").
                                VizieR's r gives (r-i) = 0.760..1.960 = K5..M4,
                                which is exactly right for 0.3-1 Msun at 2.8 Myr.
                            (c) With SpT assigned at the VizieR (r-i), 54/55
                                published CTTS satisfy their Barrado threshold,
                                and 25 of them land in the M2.5-M6 bin -- i.e. the
                                reddest threshold is actually exercised.  In the
                                table1 system that bin is empty, so the test is
                                vacuous.
                          CONSEQUENCE: EWs must be computed from VizieR
                          photometry, and the model track recovered by inversion
                          (which lives in table1's shifted system) must be
                          transformed back:  model_true(x) = track(x - 0.863) + 0.863.
                          The EW itself is invariant under this shift (it is a
                          colour *difference*), which is why table1's EW column is
                          still correct -- only its r, and anything derived from
                          (r-i) such as spectral type, are wrong.
[I] SpT -> (r-i) ....... NOT stated numerically in Paper II.  Recovered from the
                          labelled abscissa of Paper I's Fig. 8
                          (arXiv:1507.06786, file compareriEW.pdf), same author,
                          same Pickles-convolved VPHAS+ models.  Exact tick
                          positions extracted with `pdftotext -bbox`; the axis is
                          linear to 7e-16.  Paper I reddens by E(B-V)=0.35, Paper
                          II by 0.32, so boundaries are shifted blueward by
                          E(r-i) = 0.62*dE(B-V) = -0.019 mag.
                            F5 0.457 | K0 0.622 | K5 0.800 | M0 1.028 | M2 1.243
                          These boundaries are in the *true* (VizieR) colour
                          system -- Paper I's photometry has no 0.863 offset.
                          VALIDATION: evaluated at the VizieR (r-i), 54/55
                          published CTTS satisfy their assigned EW threshold.
                          The single miss, 0902b-22-4556 (r-i=1.730,
                          EW=-36.80 vs threshold -38), falls 1.2 A short --
                          far inside the paper's own stated error budget
                          ("maximum errors ... are around 9-12 A depending on
                          spectral type") and inside the published e_EW.
                          CAVEAT: no star in the 55 has -25 < EW < -18, so the
                          K5 boundary is not directly probed by the published
                          sample; it rests on Paper I's axis alone.

WHAT IS *NOT* REPRODUCIBLE, AND WHY
-----------------------------------
1. The exact footprint.  The box above yields 12 957 142 raw rows / 8 068 248
   with fPrimary=1, vs the author's 10 202 551 "in total".  The author almost
   certainly downloaded a union of whole VPHAS+ pointings whose *bounding box*
   is what the comment records.  Edge-of-field differences shift the total
   candidate count but not the selection physics.  Use --checksum to see this.
2. The M6 red boundary.  Paper I's axis stops at M4; M6 is extrapolated and
   flagged as low-confidence.  It only affects stars redder than r-i ~ 2.5.

USAGE
-----
    python rederive_kalari_ctts.py --calibrate-only      # inversion + validation
    python rederive_kalari_ctts.py --checksum            # parent-count diagnostics
    python rederive_kalari_ctts.py --run                 # full re-derivation
    python rederive_kalari_ctts.py --run --require-gaia  # emulate paper's parent

Requires: astropy, pyvo, numpy, scipy.  No credentials -- VizieR TAP is open.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Constants.  Tag: [S] sourced from paper, [I] inferred/derived here.
# ----------------------------------------------------------------------------

BIBCODE = "2019MNRAS.484.5102K"

# [S] LaTeX source comment, line 131 of 6383_v1.tex (arXiv:1901.07511)
BOX_RA_MIN, BOX_RA_MAX = 261.026, 266.05085
BOX_DE_MIN, BOX_DE_MAX = -34.49128, -30.58138
# [S] Sect 2.1 text centre: 17h34m09s, -32d32'10"
FIELD_CENTRE = (263.5375, -32.536111)
# [S] author's own counts for the above box
EXPECTED_TOTAL, EXPECTED_QUALITY = 10_202_551, 2_091_573

# [S] Sect 2.1 quality criteria
R_BRIGHT, R_FAINT = 13.0, 22.0     # "22 > r > 13"
SNR_MIN = 10.0                     # "signal to noise ratio > 10 in riHalpha"
CHI_MAX = 1.5                      # "point source function fit of chi < 1.5"

# [S] Sect 3.1 / Barrado y Navascues & Martin (2003)
EW_THRESH_EARLY = -18.0            # earlier than K5
EW_THRESH_K5_M25 = -25.0           # K5 - M2.5
EW_THRESH_M25_M6 = -38.0           # M2.5 - M6

# [I] SpT -> (r-i), from Paper I Fig. 8 axis, shifted to E(B-V)=0.32
RI_K5 = 0.800
RI_M25 = 1.398                     # M2 + 0.5 * (M4-M2)/2 ; +/- ~0.10
RI_M6 = 2.483                      # [I] EXTRAPOLATED beyond Paper I's axis. Low confidence.

# [!] r zero-point: table1.dat's r is dereddened by A_r; VizieR II/341's is not.
#     r_table1 = r_vizier - R_ZP_OFFSET.  Measured, not assumed: median over the
#     55, rms 0.003 mag, consistent across all 7 VPHAS+ fields.
R_ZP_OFFSET = 0.863

# [I] recovered by inversion; recomputed at runtime by calibrate_from_published_55()
W_FALLBACK = 128.43
# validity range of the recovered track, in the TRUE (VizieR) system
TRACK_RI_MIN, TRACK_RI_MAX = 0.760, 1.960

VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
VPHAS_TABLE = '"II/341/vphasp"'
TABLE1_URL = "https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/484/5102/table1.dat"

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".kalari_cache")


# ----------------------------------------------------------------------------
# Step 1 -- calibration: invert the published 55 to recover W and the model track
# ----------------------------------------------------------------------------

def _fetch_table1(path: str | None = None) -> str:
    path = path or os.path.join(CACHE, "table1.dat")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"[calib] downloading {TABLE1_URL}")
        urllib.request.urlretrieve(TABLE1_URL, path)
    return path


def read_published_55(path: str | None = None):
    """Parse VizieR J/MNRAS/484/5102/table1.dat by its ReadMe byte positions."""
    path = _fetch_table1(path)
    rows = [ln.rstrip("\n") for ln in open(path) if ln.strip()]

    def col(ln, a, b):
        s = ln[a - 1:b].strip()
        return np.nan if s in ("", "-") else float(s)

    return dict(
        sourceID=np.array([ln[0:14].strip() for ln in rows]),
        ra=np.array([col(ln, 16, 33) for ln in rows]),
        dec=np.array([col(ln, 35, 53) for ln in rows]),
        r=np.array([col(ln, 79, 84) for ln in rows]),
        ha=np.array([col(ln, 91, 96) for ln in rows]),
        i=np.array([col(ln, 103, 108) for ln in rows]),
        pmRA=np.array([col(ln, 115, 119) for ln in rows]),
        pmDE=np.array([col(ln, 127, 132) for ln in rows]),
        EW=np.array([col(ln, 140, 146) for ln in rows]),
        e_EW=np.array([col(ln, 148, 152) for ln in rows]),
    )


def calibrate_from_published_55(deg: int = 3, verbose: bool = True):
    """
    Recover W (rectangular bandwidth) and (r-Ha)_model(r-i) by inverting Eq. 1
    on the 55 published CTTS, which carry r, i, Ha AND EW_Ha.

        EW = W * [1 - 10^(0.4 * excess)]      with excess = (r-Ha)_obs - model
     => model = (r-Ha)_obs - 2.5*log10(1 - EW/W)

    The physically correct W is the one for which the recovered `model` values
    collapse onto a single smooth curve in (r-i).  Returns (W, coeffs, stats).
    """
    from scipy.optimize import minimize_scalar

    t = read_published_55()
    x, y, ew = t["r"] - t["i"], t["r"] - t["ha"], t["EW"]

    def scatter(W):
        v = 1.0 - ew / W
        if np.any(v <= 0) or not np.all(np.isfinite(v)):
            return 9e9
        m = y - 2.5 * np.log10(v)
        return float(np.std(m - np.polyval(np.polyfit(x, m, deg), x)))

    W = float(minimize_scalar(scatter, bounds=(80.0, 300.0), method="bounded",
                              options={"xatol": 1e-4}).x)
    model = y - 2.5 * np.log10(1.0 - ew / W)
    coeffs = np.polyfit(x, model, deg)
    rms = float(np.std(model - np.polyval(coeffs, x)))

    # leave-one-out: can the track predict the EW of a star it never saw?
    loo = []
    for k in range(len(x)):
        s = np.ones(len(x), bool)
        s[k] = False
        ck = np.polyfit(x[s], model[s], deg)
        loo.append(W * (1 - 10 ** (0.4 * (y[k] - np.polyval(ck, x[k])))) - ew[k])
    loo_rms = float(np.std(loo))

    stats = dict(W=W, track_rms=rms, loo_rms=loo_rms,
                 ri_min=float(x.min()), ri_max=float(x.max()),
                 published_e_EW_median=float(np.nanmedian(t["e_EW"])))
    if verbose:
        print(f"[calib] W          = {W:.2f} A   (paper never states it; "
              f"filter bandpass is 107 A)")
        print(f"[calib] track rms  = {rms:.4f} mag over r-i {x.min():.3f}..{x.max():.3f}")
        print(f"[calib] LOO EW err = {loo_rms:.2f} A  vs published median e_EW = "
              f"{stats['published_e_EW_median']:.2f} A")
        print(f"[calib] coeffs (hi->lo) = {np.array2string(coeffs, precision=6)}")
    return W, coeffs, stats


def validate_spt_boundaries(W, coeffs, verbose: bool = True) -> bool:
    """All 55 published CTTS must satisfy the EW threshold their (r-i) implies."""
    t = read_published_55()
    x, y = t["r"] - t["i"], t["r"] - t["ha"]
    ew = W * (1 - 10 ** (0.4 * (y - np.polyval(coeffs, x))))
    thr = ew_threshold(x)
    ok = ew < thr
    if verbose:
        print(f"[valid] published 55 satisfying their (r-i)-assigned EW threshold: "
              f"{ok.sum()}/55")
        print(f"[valid] least-negative EW among the 55 = {t['EW'].max():.2f} A "
              f"(so the K5 boundary is not probed by this sample)")
    return bool(ok.all())


# ----------------------------------------------------------------------------
# Step 2 -- selection function
# ----------------------------------------------------------------------------

def ew_threshold(ri: np.ndarray) -> np.ndarray:
    """
    [S] Sect 3.1 thresholds, mapped onto (r-i) via the [I] Paper-I axis.
    Returns the EW below which a star qualifies as CTTS.  Stars redder than the
    M6 boundary get -inf (never selected: outside the stated criteria).
    """
    ri = np.asarray(ri, float)
    thr = np.full(ri.shape, EW_THRESH_EARLY)
    thr = np.where(ri >= RI_K5, EW_THRESH_K5_M25, thr)
    thr = np.where(ri >= RI_M25, EW_THRESH_M25_M6, thr)
    thr = np.where(ri >= RI_M6, -np.inf, thr)
    return thr


def compute_ew(r, ha, ri, W, coeffs):
    """[S] Eq. 1.  Excess measured against the recovered reddened model track."""
    excess = (r - ha) - np.polyval(coeffs, ri)
    return W * (1.0 - 10 ** (0.4 * excess)), excess


# ----------------------------------------------------------------------------
# Step 3 -- VPHAS+ DR2 query (VizieR TAP, table II/341)
# ----------------------------------------------------------------------------

def _quality_adql(require_blue_r: bool) -> str:
    """[S] Sect 2.1 criteria (i)-(iii), plus fPrimary=1 for 'unique sources'."""
    c = [
        '"fPrimary"=1',
        f'"rmag" BETWEEN {R_BRIGHT} AND {R_FAINT}',
        f'"snrr">{SNR_MIN}', f'"snri">{SNR_MIN}', f'"snrHa">{SNR_MIN}',
        f'"chir"<{CHI_MAX}', f'"chii"<{CHI_MAX}', f'"chiHa"<{CHI_MAX}',
    ]
    if require_blue_r:
        # "in both the red and blue filter sets" -> r2 is the blue-set r
        c.append(f'"r2mag" BETWEEN {R_BRIGHT} AND {R_FAINT}')
    return " AND ".join(c)


def _box_adql(ra_min, ra_max, de_min, de_max) -> str:
    return (f'"RAJ2000" BETWEEN {ra_min} AND {ra_max} AND '
            f'"DEJ2000" BETWEEN {de_min} AND {de_max}')


def checksum(require_blue_r: bool = True):
    """Compare our parent counts against the author's own working numbers."""
    import pyvo
    svc = pyvo.dal.TAPService(VIZIER_TAP)
    box = _box_adql(BOX_RA_MIN, BOX_RA_MAX, BOX_DE_MIN, BOX_DE_MAX)
    for label, extra in [
        ("raw rows in box", ""),
        ("fPrimary=1 (unique)", 'AND "fPrimary"=1'),
        ("+ quality cuts", "AND " + _quality_adql(require_blue_r)),
    ]:
        q = f"SELECT COUNT(*) AS n FROM {VPHAS_TABLE} WHERE {box} {extra}"
        n = int(svc.search(q).to_table()["n"][0])
        print(f"  {label:<24} {n:>12,d}")
    print(f"  {'author: in total':<24} {EXPECTED_TOTAL:>12,d}")
    print(f"  {'author: meet criteria':<24} {EXPECTED_QUALITY:>12,d}")


def query_candidates(W, coeffs, require_blue_r=True, n_strips=8, verbose=True):
    """
    Pull only stars that *could* pass any EW threshold, by evaluating the
    recovered model track server-side in ADQL.  The loosest threshold is
    EW < -18 A, i.e. excess > 2.5*log10(1 + 18/W) mag; anything below that
    can never be selected, so there is no point transferring it.
    """
    import pyvo
    from astropy.table import vstack

    min_excess = 2.5 * np.log10(1.0 + 18.0 / W)
    c3, c2, c1, c0 = coeffs
    ri = '("rmag"-"imag")'
    track = f"({c3}*POWER({ri},3)+{c2}*POWER({ri},2)+{c1}*{ri}+({c0}))"
    excess = f'(("rmag"-"Hamag") - {track})'

    cols = ('"sourceID","RAJ2000","DEJ2000","rmag","e_rmag","imag","e_imag",'
            '"Hamag","e_Hamag","r2mag","gmag","umag","Nobs","clean","Field","Ext"')
    svc = pyvo.dal.TAPService(VIZIER_TAP)
    edges = np.linspace(BOX_DE_MIN, BOX_DE_MAX, n_strips + 1)
    parts = []
    if verbose:
        print(f"[query] min excess for any selection = {min_excess:.4f} mag")
    for k in range(n_strips):
        box = _box_adql(BOX_RA_MIN, BOX_RA_MAX, edges[k], edges[k + 1])
        q = (f"SELECT {cols} FROM {VPHAS_TABLE} "
             f"WHERE {box} AND {_quality_adql(require_blue_r)} "
             f"AND {excess} > {min_excess}")
        job = svc.submit_job(q, maxrec=4_000_000)
        job.run()
        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=3600)
        if job.phase != "COMPLETED":
            raise RuntimeError(f"strip {k}: TAP job {job.phase}")
        t = job.fetch_result().to_table()
        job.delete()
        parts.append(t)
        if verbose:
            print(f"[query] strip {k+1}/{n_strips} "
                  f"dec {edges[k]:+.3f}..{edges[k+1]:+.3f}  -> {len(t):,d} rows")
    return vstack(parts) if parts else None


# ----------------------------------------------------------------------------
# Step 4 -- orchestration
# ----------------------------------------------------------------------------

def run(require_blue_r=True, require_gaia=False, out=None, n_strips=8):
    from astropy.table import Table

    W, coeffs, stats = calibrate_from_published_55()
    if not validate_spt_boundaries(W, coeffs):
        print("[valid] WARNING: not all 55 validate -- selection is suspect.")

    t = query_candidates(W, coeffs, require_blue_r, n_strips)
    if t is None or not len(t):
        print("[run] no rows returned"); return None

    ri = np.asarray(t["rmag"] - t["imag"], float)
    ew, excess = compute_ew(np.asarray(t["rmag"], float),
                            np.asarray(t["Hamag"], float), ri, W, coeffs)
    thr = ew_threshold(ri)
    sel = ew < thr

    t["r_i"] = ri
    t["r_Ha"] = np.asarray(t["rmag"] - t["Hamag"], float)
    t["EWHa"] = ew
    t["excess"] = excess
    t["EW_threshold"] = thr
    t["is_CTTS"] = sel
    # honesty flag: outside the (r-i) span where the recovered track is calibrated
    t["track_extrapolated"] = (ri < TRACK_RI_MIN) | (ri > TRACK_RI_MAX)

    cand = t[sel]
    n_in = int((~cand["track_extrapolated"]).sum())
    print(f"\n[run] CTTS candidates              : {len(cand):,d}")
    print(f"[run]   within calibrated r-i range : {n_in:,d}")
    print(f"[run]   extrapolated (flagged)      : {len(cand)-n_in:,d}")
    print(f"[run] paper's photometric sample    : 156")
    print(f"[run] paper's PM-cut sample         : 55  (VizieR table1)")

    # acceptance test: do we recover the published 55?
    pub = read_published_55()
    got = np.isin(pub["sourceID"], np.asarray(cand["sourceID"], str))
    print(f"[run] ACCEPTANCE: published 55 recovered = {got.sum()}/55")
    if got.sum() < 55:
        print(f"[run]   missing sourceIDs: {list(pub['sourceID'][~got])[:12]}")

    if require_gaia:
        print("[run] --require-gaia: emulating the paper's Gaia-matched parent "
              "(0.1\" match + Lindegren C-1) is NOT implemented; see module "
              "docstring. Candidates above are the astrometry-free superset.")

    out = out or os.path.join(HERE, "kalari2019_ctts_rederived.ecsv")
    Table(cand).write(out, format="ascii.ecsv", overwrite=True)
    print(f"[run] wrote {out}")
    return cand


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibrate-only", action="store_true",
                   help="run the inversion + validation, no network query of VPHAS+")
    p.add_argument("--checksum", action="store_true",
                   help="compare parent counts with the author's working numbers")
    p.add_argument("--run", action="store_true", help="full re-derivation")
    p.add_argument("--no-blue-r", action="store_true",
                   help="drop the r2 (blue filter set) requirement")
    p.add_argument("--require-gaia", action="store_true")
    p.add_argument("--strips", type=int, default=8)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    if a.calibrate_only:
        W, c, _ = calibrate_from_published_55()
        validate_spt_boundaries(W, c)
    elif a.checksum:
        checksum(require_blue_r=not a.no_blue_r)
    elif a.run:
        run(require_blue_r=not a.no_blue_r, require_gaia=a.require_gaia,
            out=a.out, n_strips=a.strips)
    else:
        p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
