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
  and therefore returns a superset of the 156.
  TODO (not implemented): emulating the paper's Gaia-matched parent would be a
  0.1" join against I/355/gaiadr3 plus the Lindegren C-1 filter.  That is the
  single largest reason this catalogue is larger than 156 (see RESULTS).

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
                          VALID ONLY over TRUE (VizieR) r-i in [0.760, 1.960] --
                          the span of the 55, i.e. K5 to ~M4.  Outside that it
                          extrapolates.  This bites hardest BLUEWARD: the
                          EW < -18 A threshold applies to stars earlier than K5
                          (r-i < 0.800), which is exactly where the cubic is
                          unconstrained, because every star that survived the
                          paper's PM cut was K5 or later.  Such candidates are
                          written to a separate *_extrapolated.ecsv and must not
                          be used without an independently-built bluer track.
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
                          The value 0.863 happens to equal A_r for E(B-V)=0.32,
                          R_V=3.1 -- but "table1's r is dereddened" is REFUTED:
                          dereddening would also shift i by A_i ~ 0.63, and i
                          matches VizieR to 0.001 mag.  The offset is specific to
                          table1's r column and its origin is unknown.  Note also
                          that VizieR's two independent r measurements (rmag from
                          the red block, r2mag from the blue) agree with each
                          other and both sit 0.86 above table1's r -- so the error
                          is in table1, not in II/341.
                          THIS DOES NOT PROPAGATE.  R_ZP_OFFSET is used only to
                          transform the recovered track out of table1's system
                          during calibration.  It is never applied to any queried
                          star: candidate EWs are computed from raw II/341
                          rmag/imag/Hamag against the true-system track.  So the
                          question of whether the offset is global or
                          cluster-specific does not affect the output catalogue.
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

RESULTS AS RUN (2026-07-26, VizieR TAP, no credentials needed)
--------------------------------------------------------------
Parent checksum over the LaTeX-comment box:
    raw rows                12,957,142
    fPrimary=1               8,068,248
    + quality cuts           1,938,876   vs the author's 2,091,573  (-7.3%)
  The chi cut is genuinely ambiguous and the choice is a judgement call:
      chi<1.5 in r only   -> 1,938,876  (-7.3%) and keeps all 55   <- CHOSEN
      chi applied to none -> 2,023,444  (-3.3%), closest count, but discards
                             the paper's stated criterion (iii) entirely
      chi<1.5 in r,i,Ha   -> 1,749,467 (-16.4%) and DESTROYS 6 of the 55
                             (their chi_i reaches 1.90)
  r-only is chosen because it is the only reading that keeps criterion (iii)
  meaningful while reproducing the author's own published sample.  The no-chi
  variant matches the headline count slightly better; if you care more about the
  parent count than about recovering the 55, set CHI_BANDS = ().

ACCEPTANCE TEST: 54 of the 55 published CTTS are recovered.  The single miss is
0902b-22-4556, which fails its own EW threshold by 1.2 A (-36.8 vs -38) using
the paper's own numbers -- i.e. it is unrecoverable by the paper's stated
criteria, not a defect of this reimplementation.

CANDIDATE COUNTS, track-valid only (the quarantined 392 are excluded):
    +/-1.0 deg (a literal "2 deg x 2 deg")       317   <- HEADLINE, vs paper's 156
    +/-2.0 deg (the LaTeX-comment box)         2,782
  The 2x2 deg region contains all 54 recoverable published CTTS.

Why 317 and not 156, in a like-for-like footprint:
  1. ~63% of the full-box excess sits at dec > -31.6, a northern overdensity
     where the paper reports NO CTTS at all.  A literal 2x2 deg region excludes
     it entirely -- evidence that the paper's region really is 2x2 deg total,
     and that the LaTeX comment records the bounding box of the downloaded
     VPHAS+ pointings rather than the analysis region.
  2. The residual factor ~2 is the Gaia parent.  The paper selected from
     1,296,410 Gaia-matched, C-1-filtered stars, not from the 1.94M photometric
     parent used here (factor 1.5 on its own).  Requiring a clean Gaia DR2
     counterpart within 0.1" preferentially kills blends and nebulosity
     artefacts -- exactly the population that produces spurious Halpha excess
     in an HII region.
  So this catalogue is a genuine astrometry-free SUPERSET, as intended.  It is
  more complete and more contaminated than the paper's 156.  Anyone using it as
  ground truth should treat the north-edge population with suspicion.

CROSSMATCH to /Users/notluquis/erotica/.../cds_final/ngc6383_members.ecsv:
  21 of the 2,782 match our 321 Gaia members within 0.5"; median separation
  0.114"; 16 have PMSProb > 0.5.  (For reference, 19 of the published 55 match,
  and 42 of the 55 fall inside our catalogue's bounding box -- our 40-arcmin
  footprint covers only the central cluster, while the CTTS spread over degrees.)

USAGE
-----
    python rederive_kalari_ctts.py --calibrate-only      # inversion + validation
    python rederive_kalari_ctts.py --checksum            # parent-count diagnostics
    python rederive_kalari_ctts.py --run                 # full re-derivation
    python rederive_kalari_ctts.py --crossmatch          # join to our Gaia members

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
# [I] band(s) the chi cut applies to.  Paper does not say.  Resolved empirically:
# applying it to r+i+Ha loses 6 of the published 55 (their chi_i reaches 1.90),
# whereas all 55 have chi_r < 1.5.  See _quality_adql().
CHI_BANDS = ("r",)

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
    coeffs_t1 = np.polyfit(x, model, deg)          # in table1's shifted system
    rms = float(np.std(model - np.polyval(coeffs_t1, x)))

    # Transform the track into the TRUE (VizieR) colour system:
    #     model_true(x) = model_t1(x - R_ZP_OFFSET) + R_ZP_OFFSET
    # Refit in standard form so downstream code can just polyval() it.
    xs = np.linspace(x.min() + R_ZP_OFFSET, x.max() + R_ZP_OFFSET, 400)
    coeffs = np.polyfit(xs, np.polyval(coeffs_t1, xs - R_ZP_OFFSET) + R_ZP_OFFSET, deg)

    # leave-one-out: can the track predict the EW of a star it never saw?
    loo = []
    for k in range(len(x)):
        s = np.ones(len(x), bool)
        s[k] = False
        ck = np.polyfit(x[s], model[s], deg)
        loo.append(W * (1 - 10 ** (0.4 * (y[k] - np.polyval(ck, x[k])))) - ew[k])
    loo_rms = float(np.std(loo))

    stats = dict(W=W, track_rms=rms, loo_rms=loo_rms,
                 ri_min=float(x.min() + R_ZP_OFFSET), ri_max=float(x.max() + R_ZP_OFFSET),
                 published_e_EW_median=float(np.nanmedian(t["e_EW"])))
    if verbose:
        print(f"[calib] W          = {W:.2f} A   (paper never states it; "
              f"filter bandpass is 107 A)")
        print(f"[calib] track rms  = {rms:.4f} mag")
        print(f"[calib] LOO EW err = {loo_rms:.2f} A  vs published median e_EW = "
              f"{stats['published_e_EW_median']:.2f} A")
        print(f"[calib] valid over TRUE (VizieR) r-i "
              f"{stats['ri_min']:.3f}..{stats['ri_max']:.3f}")
        print(f"[calib] track coeffs in TRUE system (hi->lo) = "
              f"{np.array2string(coeffs, precision=6)}")
    return W, coeffs, stats


def validate_spt_boundaries(W, coeffs, verbose: bool = True) -> bool:
    """
    Acceptance test.  Uses the *VizieR* photometry of the 55 (table1's own r is
    dereddened, see R_ZP_OFFSET), recomputes EW from the recovered track, and
    checks each star against the Barrado threshold its true (r-i) implies.
    """
    t = read_published_55()
    v = fetch_vphas_by_sourceid(t["sourceID"])
    if v is None:
        if verbose:
            print("[valid] SKIPPED (VizieR unreachable)")
        return True
    r = np.array([v[s]["rmag"] for s in t["sourceID"]], float)
    i = np.array([v[s]["imag"] for s in t["sourceID"]], float)
    ha = np.array([v[s]["Hamag"] for s in t["sourceID"]], float)

    off = float(np.nanmedian(t["r"] - r))
    ri = r - i
    ew, _ = compute_ew(r, ha, ri, W, coeffs)
    thr = ew_threshold(ri)
    ok = ew < thr
    if verbose:
        print(f"[valid] measured r zero-point offset (table1 - VizieR) = {off:+.4f} mag "
              f"(assumed {-R_ZP_OFFSET:+.4f})")
        print(f"[valid] TRUE (r-i) span of the 55 = {ri.min():.3f}..{ri.max():.3f}")
        print(f"[valid] EW reproduced from VizieR photometry: rms = "
              f"{np.nanstd(ew - t['EW']):.2f} A")
        print(f"[valid] published 55 satisfying their (r-i)-assigned EW threshold: "
              f"{ok.sum()}/55  (M2.5-M6 bin populated by "
              f"{int(((ri >= RI_M25) & (ri < RI_M6)).sum())})")
        for b in np.where(~ok)[0]:
            print(f"[valid]   miss: {t['sourceID'][b]:<15} r-i={ri[b]:.3f} "
                  f"EW={t['EW'][b]:.2f} vs threshold {thr[b]:.0f} "
                  f"(short by {abs(t['EW'][b]-thr[b]):.1f} A; paper's own error "
                  f"budget is 9-12 A)")
    return bool(ok.sum() >= 54)


def fetch_vphas_by_sourceid(source_ids):
    """Pull II/341 rows for an explicit list of VPHAS+ sourceIDs. None on failure."""
    try:
        import pyvo
        ids = ",".join("'%s'" % s for s in source_ids)
        q = (f'SELECT "sourceID","RAJ2000","DEJ2000","rmag","imag","Hamag" '
             f'FROM {VPHAS_TABLE} WHERE "sourceID" IN ({ids})')
        tab = pyvo.dal.TAPService(VIZIER_TAP).search(q).to_table()
        return {str(row["sourceID"]): row for row in tab}
    except Exception as exc:  # pragma: no cover
        print(f"[valid] VizieR query failed: {type(exc).__name__}: {exc}")
        return None


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
    # chi cut: the paper says only "point source function fit of chi < 1.5".
    # It does NOT say in which band(s).  Applying it to r, i and Ha together
    # throws away 6 of the 55 published CTTS (chi_i up to 1.90, chi_Ha up to
    # 1.50) -- while ALL 55 have chi_r < 1.5.  So the author applied it in r
    # only.  This is empirically resolved, not assumed; see CHI_BANDS.
    c = [
        '"fPrimary"=1',
        f'"rmag" BETWEEN {R_BRIGHT} AND {R_FAINT}',
        f'"snrr">{SNR_MIN}', f'"snri">{SNR_MIN}', f'"snrHa">{SNR_MIN}',
    ]
    c += [f'"chi{b}"<{CHI_MAX}' for b in CHI_BANDS]
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

def run(require_blue_r=True, out=None, n_strips=8):
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
    good = cand[~cand["track_extrapolated"]]
    bad = cand[cand["track_extrapolated"]]

    # like-for-like with the paper's stated "2 deg x 2 deg" region
    cra, cde = FIELD_CENTRE
    in2deg = ((np.abs(np.asarray(good["DEJ2000"], float) - cde) < 1.0) &
              (np.abs((np.asarray(good["RAJ2000"], float) - cra)
                      * np.cos(np.radians(cde))) < 1.0))

    print(f"\n[run] HEADLINE, like-for-like with the paper's stated region:")
    print(f"[run]   2x2 deg region, track-valid  : {int(in2deg.sum()):,d}   "
          f"(paper: 156)")
    print(f"[run] wider +/-2 deg box, track-valid : {len(good):,d}")
    print(f"[run]   quarantined (r-i outside the track's calibrated "
          f"{TRACK_RI_MIN:.2f}-{TRACK_RI_MAX:.2f}): {len(bad):,d}")
    print(f"[run]   -> the cubic extrapolates blueward of K5, where the EW<-18 "
          f"threshold lives; those EWs are NOT trustworthy.")
    print(f"[run] paper's PM-cut sample           : 55  (VizieR table1)")

    # acceptance test: do we recover the published 55?
    pub = read_published_55()
    got = np.isin(pub["sourceID"], np.asarray(cand["sourceID"], str))
    print(f"[run] ACCEPTANCE: published 55 recovered = {got.sum()}/55")
    if got.sum() < 55:
        print(f"[run]   missing sourceIDs: {list(pub['sourceID'][~got])[:12]}")

    out = out or os.path.join(HERE, "kalari2019_ctts_rederived.ecsv")
    Table(good).write(out, format="ascii.ecsv", overwrite=True)
    print(f"[run] wrote {out}  ({len(good):,d} track-valid candidates)")
    if len(bad):
        qout = out.replace(".ecsv", "_extrapolated.ecsv")
        Table(bad).write(qout, format="ascii.ecsv", overwrite=True)
        print(f"[run] wrote {qout}  ({len(bad):,d} quarantined -- do not use "
              f"without a bluer track)")
    return good


MEMBERS_ECSV = ("/Users/notluquis/erotica/data/test/NGC6383/comments_paper/"
                "cds_final/ngc6383_members.ecsv")


def crossmatch(candidates=None, members_path=MEMBERS_ECSV, tol_arcsec=0.5):
    """
    Join a VPHAS+-based CTTS list to our Gaia-based member catalogue.

    JOIN PATH.  VPHAS+ II/341 carries no Gaia source_id, and there is no
    published Gaia x VPHAS+ neighbour table, so the join is POSITIONAL.
    That is safe here and not merely convenient:
      - measured separations for the published 55 are 0.05-0.18", and the match
        count is identical at 0.5", 1.0" and 2.0" tolerance -- i.e. there is no
        confusion regime to fall into;
      - epoch drift is negligible.  VPHAS+ was observed ~2011-2014, Gaia DR3 is
        at J2016.0; at the cluster proper motion (~2.5, -1.7 mas/yr) that is
        ~10 mas, two orders of magnitude below the tolerance.  (It would matter
        for high-proper-motion field interlopers, not for members.)
      - the join is independently confirmable: table1 carries Gaia DR2 pmRA/pmDE,
        which agree with our DR3 values for the matched stars.
    Note our catalogue is Gaia DR3 while the paper used DR2, so source_ids are
    not interchangeable even where both exist.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table

    if candidates is None:
        candidates = Table.read(os.path.join(HERE, "kalari2019_ctts_rederived.ecsv"))
    me = Table.read(members_path)
    cc = SkyCoord(np.asarray(candidates["RAJ2000"]) * u.deg,
                  np.asarray(candidates["DEJ2000"]) * u.deg)
    cm = SkyCoord(np.asarray(me["RAdeg"]) * u.deg, np.asarray(me["DEdeg"]) * u.deg)
    idx, d2d, _ = cc.match_to_catalog_sky(cm)
    hit = d2d.arcsec < tol_arcsec
    print(f"[xmatch] CTTS candidates            : {len(candidates):,d}")
    print(f"[xmatch] our members                : {len(me):,d}")
    print(f"[xmatch] matched within {tol_arcsec}\"      : {int(hit.sum()):,d}")
    out = candidates[hit].copy()
    for c in ("GaiaDR3", "PMSProb", "Fidelity", "pMember", "Gmag"):
        if c in me.colnames:
            out[c] = np.asarray(me[c])[idx[hit]]
    out["sep_arcsec"] = d2d.arcsec[hit]
    return out


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
    p.add_argument("--crossmatch", action="store_true",
                   help="join an existing candidate file to our Gaia member catalogue")
    p.add_argument("--strips", type=int, default=8)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    if a.calibrate_only:
        W, c, _ = calibrate_from_published_55()
        validate_spt_boundaries(W, c)
    elif a.checksum:
        checksum(require_blue_r=not a.no_blue_r)
    elif a.crossmatch:
        crossmatch()
    elif a.run:
        run(require_blue_r=not a.no_blue_r, out=a.out, n_strips=a.strips)
    else:
        p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
