# NGC 6383 paper review repository

Local reference store for the A&A resubmission (aa52082-24, "Characterizing NGC 6383...").
Created 2026-06-02.

## Contents
- `paper_source.tex` — full LaTeX source (copy of submission_package/clean_source/aanda.tex, 513 lines)
- `paper_fulltext.txt` — plain-text extraction of the compiled PDF
- `paper_annotated_pierluigi.pdf` — Pierluigi Cerulo's annotated PDF (76 annotations)
- `cites.bib`, `aanda.bbl` — bibliography
- `01_pierluigi_annotations.md` — all 76 annotations transcribed (page, type, target, note)
- `02_literature_verification.md` — ADS/SciX checks of his 9 substantive comments
- `03_independent_full_review.md` — my own full-paper review (numerical consistency, stats, terminology)

## Top fixes (priority order)
1. C1 — segregation time: Table 1.80+/-0.42 vs text 1.47+/-0.418 Myr (contradiction)
2. C2 — p=0.07 mislabeled "significant"; mass-segregation headline rests on it
3. C3 — error-propagation bug (R_hl 6.02+/-0.0006 arcmin; R_c pc error; M_gc error)
4. #67 — bin-dependent KS p-values (forking paths) -> add Allison 2009 Lambda_MSR
5. #70 — Kalari method wording (Halpha-EW primary, PM-Gaussian secondary)
6. Pierluigi's concision/caption pass (~30 strikeouts) + wording fixes
7. Add citations: Da Rio et al. 2010, Allison et al. 2009

## Note
Earlier-session claim that the Buckner & Froebrich Y_frac comparison was a "binary vs YSO category error" was WRONG. The paper's Y_frac is a YSO fraction (same Q-method as B&F); comparison is valid. See 02_literature_verification.md #2.
