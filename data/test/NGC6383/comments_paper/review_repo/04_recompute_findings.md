# Recompute findings (real 40-arcmin production data)

Data: `data/40/` + `ASteCA/output/NGC_6383_dr3_all/`. Script: `recompute_C3_allison.py` (reproducible).
Reference sample = 254 (ASteCA output) joined to `masses_asteca.csv` (m1,m2,binar_prob) by row.
Center (263.6826, -32.5838); distance 1.110 +/- 0.060 kpc.

## C3 error propagation (RESOLVED)
Central values reproduce the paper; the errors were the bug.

| radius | paper | corrected (bootstrap + full pc propagation) |
|--------|-------|----------------------------------------------|
| R_c | 1.95+-0.19 arcmin / 0.632+-**0.00621** pc | 0.632 +- **0.070** pc  ✅ applied |
| R_t | 40.4+-14.3 arcmin / 13.10+-4.71 pc | 13.05 +- 4.67 pc (already correct, no change) |
| R_hl| 6.020+-**0.00060** arcmin / 1.960+-**0.00020** pc | **6.02+-1.07 arcmin / 1.94+-0.36 pc** ✅ applied |
| R_hm| 6.240+-0.251 arcmin / **1.650**+-0.066 pc | **6.26+-1.23 arcmin / 2.02+-0.41 pc** ⚠️ NOT applied (cascade) |

- R_c/R_hl errors were photometric-only / distance-omitting -> absurdly small. Fixed.
- **R_hm internal inconsistency**: 6.24 arcmin at 1.11 kpc = 2.02 pc, NOT 1.65 pc. The pc value 1.650 is wrong.

## ⚠️ t_rh / t_seg cascade (NOT auto-applied — needs author rerun + figure regen)
The paper's dynamical chain is internally self-consistent but built on the wrong r_hm = 1.65 pc:
- paper r_hm=1.65 -> t_rh=13.6 Myr -> seg-mass cutoff 6.08 Msun (all consistent with each other).
- Corrected r_hm=2.02 pc gives:
  - **t_rh = 18.5 +- 5.7 Myr** (was 13.6; uses N=254, M_c=902 Msun from Hunt)
  - **min t_seg (14.7 Msun) = 1.77 +- 0.55 Myr**
  - **seg-mass cutoff (t_seg = age 3.53 Myr) = 7.4 Msun** (was 6.08)
- The cutoff change (6.08 -> 7.4) requires regenerating `cumulative_by_mass_and_type_mseg.pdf` (Fig. mass_tseg), which I cannot reproduce faithfully. So left for the authors.
- Conclusion is UNCHANGED / strengthened: longer t_rh = even less relaxed = more support for primordial segregation.
- NOTE: Table "Minimum segregation time" was set to 1.47 (consistent with the CURRENT uncorrected t_rh=13.6 and <m>=1.59). After the r_hm fix it should become ~1.8-2.0.

## ✅ Allison et al. (2009) Lambda_MSR (NEW, applied)
Bin-free MST mass-segregation check on the 254 sample (total mass = m1 + m2 if binar_prob>0.7):

| N_MST | 5 | 8 | 10 | 15 | 20 | 30 | 40 | 50 |
|-------|---|---|----|----|----|----|----|----|
| Lambda| 2.35 | 2.68 | 3.07 | 2.69 | 1.93 | 1.44 | 1.16 | 1.12 |

Strong segregation (Lambda ~2.5-3) for the most massive 5-15 members, declining to ~1. Robustly confirms mass segregation and directly answers Pierluigi #67 (bin-dependence). Figure: `Figures/mass_segregation_mst.pdf` -> added as Fig. \ref{fig:mst}, cite Allison 2009 added to cites.bib.

## ✅ T-Tauri binary (#72, applied)
Gaia DR3 4054567805963940352: binar_prob = **0.82** (matches paper claim), and it IS in the 254 reference sample. Text now states it belongs to both our reference sample and the Kalari (2019) CTTS list.

## ✅ #6 crossmatch radius (resolved, no change needed)
2MASS match uses Gaia's precomputed `tmass_psc_xsc_best_neighbour`; the 0.3 arcsec is the separation cut on that table, not a COSMIC parameter. Paper's "0.3 arcsec" is correct; Pierluigi's "0.03''" was a misremembering. Kept 0.3.

## Caveat
`comments_paper/members.csv` is from a DIFFERENT run (p>=0.6 -> 177, not 254) and stores source_id as float (precision lost). Do NOT use it for the paper catalog; use `data/40/` + ASteCA output.
