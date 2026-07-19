# Independent full-paper review (beyond Pierluigi's annotations)

Reviewer: Claude. Source: paper_source.tex (513 lines). Date 2026-06-02.
Focus: internal numerical consistency, statistical claims, terminology, things a hard external A&A referee would flag.

## 🔴 CRITICAL — fix before resubmission

### C1. Segregation-time contradiction (Table vs text)
- Table 2 (line 211): "Minimum segregation time = 1.80 +/- 0.42 Myr".
- Text (line 421): "a segregation time of 1.470 +/- 0.418 Myr for the most massive star of 14.7 M_sun".
- t_seg = <m>/m * t_rh = 1.59/14.7 * 13.6 = 1.47 Myr (uses <m>=1.59 from line 403). The 1.80 value is unexplained (would need <m>=1.95). Same uncertainty (0.42 ~ 0.418) => 1.80 is a STALE central value.
- ACTION: reconcile; the 1.47 Myr appears correct. Update Table.

### C2. p=0.07 called "significant" (overclaim) — ties to Pierluigi #67
- Line 405: binary vs single "a significant difference ... with a p-value of 0.07 and a K-S test statistic of 0.2079. This suggests binary mass segregation".
- p=0.07 is NOT significant at alpha=0.05. Calling it "significant" then resting the headline "primordial mass segregation" / abstract claim on it is the single most exploitable weakness for a referee.
- ACTION: either soften to "marginal/suggestive (p=0.07)", or strengthen the evidence (Allison Lambda_MSR, larger sample, justify the mass cut). Do not call p=0.07 significant.

### C3. Physically meaningless / inconsistent error propagation
- Half-light radius (line 360): 6.02000 +/- 0.00060 arcmin (1.96000 +/- 0.00020 pc) -> 0.01% error. Absurd next to half-mass radius 6.240 +/- 0.251 arcmin (4%).
- Core radius pc error (line 360): 0.63200 +/- 0.00621 pc = 1%, but the SAME R_c in arcmin is 1.95 +/- 0.19 = 10%. The pc error should also be ~10% (~0.063 pc) once distance error folds in. Mismatch.
- M_gc (line 242): 1.43e11 +/- 9.71e7 (0.07%) — formal propagation of R_gc error only; physically meaningless precision for an enclosed Galactic mass.
- ACTION: redo error propagation for R_c(pc), R_hl, M_gc; the suspiciously tiny errors look like a bug (forgot distance-error term / wrong sigma). A referee WILL notice 6.02+/-0.0006.

## 🟡 IMPORTANT — should fix

### I1. R_t terminology inconsistent
- Abstract (line 37) + Sect 3.1: "King outer radius" (careful, good — weakly constrained).
- Table 2 (line 198): "Tidal radius (R_t)". Same symbol, two names.
- ACTION: unify. Suggest "King outer (tidal) radius R_t" once, then R_t, and keep the weakly-constrained caveat everywhere.

### I2. KS p-value ranges straddling 0.05 described as confirmed/significant
- Line 401 (LF): "confirmed using a K-S test, which yielded p-values from 0.009 to 0.052" — upper end >0.05.
- Line 405/423: p-value ranges 0.04-0.95 and 0.06-0.45 cited as evidence.
- Reporting a RANGE of p-values across bins and calling the result confirmed is the forking-paths problem (Pierluigi #67). 
- ACTION: report the specific test that matters with D + p + N; don't lean on ranges.

### I3. Distance-modulus uncertainty inconsistent
- Table (line 188): "Distance modulus 10.3 +/- 0.3".
- Text (line 366): "distance modulus of 10.300 +/- 0.262 mag".
- ACTION: pick one (0.262 vs 0.3).

### I4. Star-formation range rounding inconsistent
- Table (line 190): 1.58-6.31 Myr. Text (line 368): 1.6-6.3 Myr. Abstract (line 37) & Conclusions (line 459): "1 to 6 Myr".
- ACTION: use one convention (e.g. 1.6-6.3 Myr) throughout.

### I5. Sagitta trained on DR2, applied to DR3
- Line 271: "trained with data from Gaia DR2". Paper data = DR3. Cross-release application is fine but unstated as a caveat.
- ACTION: one sentence acknowledging DR2-trained NN applied to DR3 photometry.

### I6. Projected velocity quoted in mas/yr
- Line 345: "projected velocity is 3.070 +/- 0.010 mas/yr". A "velocity" in angular units (mas/yr) without distance conversion to km/s is confusing; also the 0.010 error is very small.
- ACTION: either convert to km/s (using distance) or rename "projected proper motion".

### I7. Radial velocity headline value
- Table: RV = -15.1 +/- 31.1 km/s (mean, std>|mean|, 16 stars). Heavily caveated in text (good).
- ACTION: consider reporting median (-6.11) as the headline, or flag clearly that RV is indicative only. A referee may question quoting a mean with std twice the value.

## 🟢 MINOR / cosmetic
- Table 1 age column mixes formats ("~20", "Up to 5.0", "Less than 4.0", ranges) — standardize.
- Metallicity Z=0.024 is super-solar (MIST Zsun~0.0152); fine for the fit but worth a half-sentence noting it's above solar.
- IMF params (line 254) alpha=0.090, beta=0.940 "following Chabrier 2014" — non-standard-looking; confirm these are the intended ASteCA inputs.
- Footnote (line 51) "NGC 6383 also known as NGC 6374" — double-check the NGC 6374 identity (cited to Reipurth 2008).
- Eq for M_gc uses (R_gc/30 pc)^1.2 — numerically self-consistent (gives 1.43e11) but the "30 pc" scale reads oddly; confirm constant/units against the source.

## Cross-check vs Pierluigi
- His instinct correct on the substance (all 9 lit checks support him), but:
  - #53 his concern is about wording/sample-membership, NOT a category error — comparison is valid (both YSO fractions).
  - #70 his rewrite inverts Kalari's method — fix wording but not verbatim.
  - #75 he struck a correct point — keep a compressed version.
- He did NOT catch: C1 (t_seg contradiction), C2 (p=0.07 mislabeled significant), C3 (error-propagation bug). These are the highest-value fixes and are independent of his pass.

## Suggested new citations
- Da Rio, Gouliermis & Gennaro (2010) ApJ 723,166 — PMS CMD broadening (for the ASteCA-PMS caveat).
- Allison et al. (2009) MNRAS 395,1449 — MST Lambda_MSR mass-segregation (to harden the segregation claim).
