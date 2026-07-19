# NGC 6383 (aa52082-24), Complete changelog: original submission → current revision

Baseline = originally submitted manuscript (`marked_changes/old_submitted.tex`).
Current = `clean_source/aanda.tex`. Visual diff = `aa52082-24_marked_changes.pdf` (22 pp, latexdiff).
Status: 19 pp total / **13 pp body+refs** / 0 undefined refs / 0 AI-tells / 0 British spellings except 2 deliberate 'discs' retained from the original submission (diff discipline).

---

## 1. Title & metadata
- **Title**: "A study of pre-main sequence stars, mass segregation, and age…" → "A study of **fundamental properties**, pre-main sequence stars, mass segregation, and age…"
- **Keywords**: dropped extragalactic `galaxies: star clusters: general`; added `stars: pre-main sequence`; object added → `open clusters and associations: individual: NGC 6383`. (Now 6, A&A-compliant.)
- Section heading `Conclusion` → `Conclusions`.

## 2. Numerical cascade, re-ran ASteCA v0.4 → **v0.6.9** (user choice 2026-06-02)
| Quantity | Original | Current |
|---|---|---|
| Mean stellar mass ⟨m⟩ | 1.59 (no error) | **1.31 ± 0.10** M⊙ |
| Most massive star m_max | 14.7 (no error) | **13.56 ± 3.25** M⊙ |
| Half-mass radius R_hm | 1.65 pc (bug: wrong distance) | **2.02 ± 0.41** pc |
| Half-mass relaxation t_rh | 13.6 Myr | **18.5 ± 5.7** Myr |
| Min segregation time t_seg | 1.47 (only t_rh err) | **1.78 ± 0.71** (full propagation) |
| Seg-mass cutoff | 6.08 M⊙ | **6.81** M⊙ (98.4% of sample) |
| Single-vs-binary KS | D=0.208, **p=0.07** ("significant", overclaim) | **D=0.221, p=0.010** (genuinely significant) |
| Y_frac | 0.280 (denominator ambiguous) | **0.275** (N_cl=193 of 254 with 2MASS JHKs explicit) |
| T-Tauri binary prob | 0.82 | **0.94** |

## 3. Error propagation, fixed physically-meaningless uncertainties
- R_c(pc): 0.632 ± **0.00621** → ± **0.070** (distance term added)
- R_hl: 6.02 ± **0.0006** arcmin → ± **1.07** (bootstrap) = 1.94 ± 0.36 pc
- M_gc: (1.43 ± **9.7×10⁷**) → (1.43 ± **0.02**)×10¹¹ M⊙ (correct R_GC propagation)
- R_t(pc): 13.10 → **13.05** (correct arcmin→pc at 1.11 kpc)
- ⟨m⟩, m_max, MSEG-⟨m⟩ now carry errors (ASteCA per-star std); t_seg re-propagated with all 3 terms.
- Uncertainty conventions documented (1σ posterior / 1σ dispersion / SE / bootstrap / Beta-binomial).

## 4. New analyses added (hardening the science)
- **Λ_MSR mass-segregation ratio** (Allison 2009), new figure `mass_segregation_mst.pdf`; bin-free, ~3σ for 5–15 most massive. Answers the bin-dependence concern.
- **Structure parameter 𝒬 = 1.03 ± 0.03** (Cartwright & Whitworth 2004, bootstrap), confirms smooth/centrally concentrated, non-fractal; cross-checks King C=1.32.
- **Dynamical-age anchor**: age/t_rh ≈ 0.19 (≪ τ_v ~ 2–3.5 t_rh for primordial-segregation erasure, Pavlík 2020).
- **Tangential velocity** v_t = 16.2 ± 0.9 km/s (the old "projected velocity 3.07 mas/yr" was angular → renamed mean total proper motion + km/s).
- Robustness: Λ_MSR unchanged with median/geometric-mean variant (Maschberger & Clarke 2011).

## 5. Primordial-segregation interpretation, made honest & regime-correct
- Now stated as **primordial OR near-primordial** with the explicit caveat that early dynamical segregation (cool/substructured collapse, Allison 2009 ApJ; subcluster merging in low-N local clusters, Moeckel & Bonnell 2009) is **observationally degenerate** with birth segregation.
- 𝒬 used correctly: it confirms smoothness but does NOT distinguish the channels (collapse also smooths).

## 6. Citations, added / removed / fixed (regime-checked)
**Added (verified content + regime):** Da Rio 2010; Allison 2009 (MNRAS Λ_MSR + ApJ short-timescale); Cartwright & Whitworth 2004; Parker & Goodwin 2015; Maschberger & Clarke 2011; Pavlík 2020; Moeckel & Bonnell 2009; Baumgardt & Kroupa 2007; Angelo 2025; Risbud 2025; Xu 2025.
**Removed / replaced (wrong regime or misattributed):**
- **Dib 2007** removed, Arches (massive GC-region) IMF paper, not open-cluster binary segregation.
- **Marks 2008** (globular-cluster MF) → **Baumgardt & Kroupa 2007** (young-cluster gas expulsion).
- **Polak 2025** considered then rejected, massive-cluster sims (10⁴–10⁶ M⊙), wrong regime for a 900 M⊙ OC.
- Wording "other young open clusters" → "other young clusters" (Sabbi/NGC 346 = SMC SF region).
**Fixed:** Allison arXiv id 0902.4047 → 0901.2047; bib "FALSE" artifact removed (Penzias entry).

## 7. Figures
- **Luminosity function** → single absolute-magnitude panel (apparent-mag panel removed).
- **Parallax figure** rebuilt: legend carries δϖ/ϖ thresholds; **3-panel colors unified** (red <0.1, green >0.1, orange all, middle panel had inconsistent red/blue).
- **Cumulative-by-mass** + `_mseg` regenerated with v0.6.9 masses.
- **Λ_MSR figure** added; in-plot title removed (A&A style).
- **proper_motion caption fixed**: contours are the 2D Gaussian PM model, NOT KDE (center fig is KDE).
- Moved to **Appendix B** "Supplementary figures": masses and binarity CMD (cmd_mass_binary), luminosity function (lf), real_sky, pms_stats (Sagitta), mass_tseg, radial_velocity.
- All 19 figures visually verified; every figure cited in body.

## 8. Structure / compaction
- Table 1 (historical literature) → Appendix A.
- Table 2 (cone-search robustness) → Appendix D.
- Removed appendix paragraphs that merely restated figure captions.
- Body reduced toward 12 pp (currently 13 after rigor additions; page decision pending).

## 9. Writing / style
- Abstract + Conclusions rewritten concrete/concise; captions tightened to label not teach; British → US (modelling→modeling).
- NOTE (2026-06-10): the earlier blanket AI-tell removal in the BODY was deliberately rolled back by the diff-minimization pass (§11), original-submission phrasing (incl. "crucial", "utiliz*") was restored wherever the referee had already accepted it. Abstract/conclusions tightening retained.

## 10. Reviewer responses (full point-by-point in `letters/`)
- **Referee (20 pts + additions A–K)**: all verified present; cone-robustness table + 2025 OC-structure citations; HD 159176 non-membership (Δμδ 6.6σ); error propagation; ASteCA 0.6.9; Y_frac denominator; uncertainties on every quantity; substructure check; metallicity/units/terminology refinements.
- **Cerulo (76 annotations)**: 76/76 honored (deviations documented); no struck phrasing survives.

## 11. Diff-minimization pass (2026-06-10)
- Goal: marked-changes PDF should show only REQUIRED changes (referee/co-author/numbers/figures), not stylistic churn.
- All 37 diff hunks classified (multi-agent + adversarial verify); ~55 pure-style rewrites reverted to the original submission wording; every letter-committed fix, v0.6.9 number, citation and regenerated-figure caption kept.
- Word-diff vs original: 760/719 → ~600/550 segments.
- 9 Pierluigi annotation fixes accidentally undone by the restores were detected in a second-pass review and re-applied (annot. 10, 24, 25, 56, 60, 62, 63, 64, 65).

## 12. Relaxation-time self-consistency + final referee-prep (2026-06-10)
- t_rh inputs were inconsistent (N=254 Gaia members mixed with Hunt+24 IMF-extrapolated M=902 Msun, implying <m>=3.55 vs our 1.31). Now self-consistent: M = N<m> = 332±26 Msun → t_rh = 30.5±9.4 Myr, t_seg = 2.94±1.17 Myr, age/t_rh ≈ 0.12, t_seg>age cutoff 11.2 Msun (subsample 252 = 99.2%, <m>=1.21±0.08, quartile KS D=0.12–0.36 p=0.01–0.84). Figures cumulative_by_mass_and_type[_mseg] regenerated; Table 2, letters, conclusions cascaded. Conclusions unchanged (cluster even more clearly unrelaxed).
- Hunt+24 total mass kept ONLY for the Hill radius (correct usage).
- Table 2 row "Number of members" → "Reference sample" (taxonomy consistency).
- Added: bp≥0.6 binary class vs bp>0.7 mass-sum justification; HD 159176 parallax-window-by-construction note (μδ tension = supporting, not decisive); per-catalog comparison interpretation; Q selection-bias caveat; 5 sentence-level upgrades in new text (abstract closer, cone-robustness sentence, degeneracy sentence split, conclusions HD logic, article fix).
- Packaging: real_sky.pdf 25.4→3.8 MB (300 dpi), source zip 26.5→5.0 MB; CDS zip reduced to ReadMe+dat; cover letter rewritten (editor-triage style, corrected title); response .txt files converted to true plaintext; letters updated (t_rh/t_seg cascade, Appendix C, self-consistency disclosure).

## 13. Faculty-level pass + cone-search catalog-level results (2026-06-11)
- **Referee point 1 closed with analysis** (review_repo/cone50_derived_params.py): 50-arcmin run (automated branch = NGC-like) gives plx 0.906±0.044 mas, pm (2.544,−1.711)±(0.133,0.127), stable vs adopted 40' values. King refit on the 50' sample with identical priors: R_c=1.94±0.17', R_t=39.3±13.7', C=1.28±0.19, consistent; outer radius stays weakly constrained even with the wider window. 40'-sanity refit reproduces published values. Added to Results cone paragraph + referee letter point 1 + cover letter.
- **Faculty prose pass** (level now prioritized over diff, per author): intro ¶1 rewritten (gap-driven, e.g.-cites); filler/AI-tells removed everywhere (crucial/utiliz*/Notably/It-is-important/as-mentioned-before = 0 hits); HDBSCAN, COSMIC, Sagitta, ASteCA, parallax-prior, RV, YSO, CMD-opener, Results-opener upgraded to the senior versions; crossmatch enumeration tense-unified; 1971 citation regrouped; abstract aims de-intensified ("accurately/precise" dropped); Star No. 6 agreement; reddening antecedent; mas$-based typo.
- **Letters to faculty level**: referee letter, point 1 extended with the new analysis, items A/I de-duplicated, "Final preemptive refinements"→author-initiated register, internal-audit phrasing removed; Cerulo letter, §1 broken into thematic bullets, tactical "harder external referee" phrasing replaced by the scientific justification, signed "Lucas"; cover letter already editor-triage + one-line catalog-stability highlight. All .txt = true plaintext.
- **60/70 arcmin runs added (2026-06-11)**: pmRA dispersion inflates 0.152→0.239→0.279 mas/yr with drifting mean (2.455, 2.383), plx shifts +0.008 mas; King refits follow the window (R_c→1.4', R_t→50.5±9.5', C→1.55), coherent field-contamination signature where the sweep branch diverges. In tex cone paragraph + referee letter pt 1. Full ladder now: 40' (adopted) / 50' (stable) / 60'-70' (degrades coherently).
- DEFERRED to a future paper (author decision): PARSEC/Baraffe cross-fit to bound MIST age systematics.

## 14. Convergence audit of all Bayesian fits (2026-06-11)
- Literature standard applied (Vehtari+2021 rank-normalized R-hat<1.01, bulk/tail ESS>400, 0 divergences, E-BFMI>0.3; Betancourt 2016; Gabry+2019).
- NUTS re-runs (4x2000, tune 2000; review_repo/convergence_audit*.py; idata_*.nc saved): parallax PASS (R-hat 1.0002, ESS 7329/5537; reproduces 0.908/0.046/SEM 0.004), distance PASS (1.117/0.060 kpc), PM PASS (mu 2.542/-1.713, sigma 0.153/0.138, SEMs 0.0096/0.0087), King PASS at target_accept 0.99 (0 div, ESS 2020/1299; Rc 1.98+-0.20, Rt 39.7+-14.6, C 1.27+-0.22, all within 1 sigma of published).
- ISOCHRONE: production trace (data/40/fit_parameters_trace_1724708835.nc, 2024-08, PyMC 5.16.2) is DEMetropolis 300x1000 (tune 1500), NOT NUTS as the text claimed. Per-chain Vehtari criteria inapplicable (frozen-chain ensemble; R-hat 2-32, ESS~305=N_chains). Ensemble verified stable: half-ensemble bootstrap mode scatter 0.08 dex (loga), 0.03 mag (dm), 0.003 (Z) << credible widths. Published values KEPT (dm 10.28-10.30 coherent with parallax distance).
- New cosmic IsochroneFitter (true NUTS/blackjax, Poisson Hess likelihood) does NOT reproduce: multimodal non-converged run (R-hat 1.5-2.2), posterior at prior edges (Av->0.5, dm->9.7 == 870 pc, contradicts parallax 1.11 kpc). NOT adopted; flagged as unvalidated next-paper infrastructure (review_repo/isochrone_nuts_refit.py; grid cache data/40/hgrid_paper254.npz).
- Paper edits: abstract "NUTS from PyMC" -> "PyMC for the Bayesian modeling"; methods NUTS-vs-DEMetropolis scope sentence (+ ter Braak 2006 cite); convergence-criteria paragraph (Vehtari+21 cite added to bib); ASteCA section sampler corrected. Referee letter item L added. Published numbers unchanged.

## 15. Cone-ladder King refits: convergence + final numbers (2026-06-11)
- All four cone King refits (40/50/60/70 arcmin) re-run at target_accept=0.99: 0 divergences, R-hat <= 1.0043, ESS >= 1597, E-BFMI >= 0.86, all PASS Vehtari+21. idata_king_cone{40,50,60,70}.nc saved in review_repo (script cone_king_convergence.py).
- Final-run numbers synced to tex + referee letter: 50-arcmin Rc 1.93+-0.17, outer radius 40.1+-13.6, C 1.29+-0.19 (was 39.3/1.94/1.28 from the ta=0.95 run); 70-arcmin outer radius 50.2+-9.6 (was 50.5+-9.5). Convergence paragraph widened to R-hat <= 1.005 to cover the auxiliary refits. Conclusions unaffected.
- Parallax/PM ladder values are frequentist summaries (no MCMC), no convergence applicable.

## 16. Final coherence sweep (2026-06-11): captions self-contained + A&A compliance + letter sync
- 8-agent final audit: Pierluigi 76/76 intact; package OK; referee letter synced (R-hat 1.005, Sect. 3.2, Fig. A.3->B.3 renumbering note, point-11 tidal->King outer radius).
- Previously undeclared changes now DECLARED in letters (not reverted): intro paragraph-1 tightening + Sagitta DR2/DR3 caveat (item H), Table 1 -> Appendix A + RV-amplitude figure -> Appendix C (item E). Title change already declared in cover letter.
- ALL 23 captions made (near) self-contained: every color/line style/marker/contour/band/threshold now defined in-caption (probabilities, center, PM, parallax 3-panel, King, CMD, LF, brightness-cumulative, mass-seg x2, MST, condensed tree, blue ellipse identified as NGC 6383 after visual ground-truthing, corner plot, real_sky incl. cone-search circle, pms_stats, RV-amplitude, mass_binary isochrone black->dark-gray to match render).
- cumulative_by_brightness_paper.pdf REGENERATED (was default-matplotlib solid colors, red/green adjacent = CVD-unsafe): now Tol 4 colors + 4 line styles, black dash-dot/dotted reference lines; same quartile bounds (8.80/15.44/16.99/17.93/20.66), shapes verified visually. Script review_repo/regen_brightness_cumulative.py.
- bp class wording unified to 'of at least 0.6' (matches >=0.6 in code) in body + fig:mass_seg caption.
- A&A compliance: keywords reordered per A&A category order; NEW mandatory 'Data availability' section (canonical CDS wording + COSMIC GitHub); appendix \labels added + two vague 'in the appendix' refs -> Appendix B; astropy footnote \url{}; 3 range en-dashes. Micro-typography in ORIGINAL text (Sect. abbreviations, heading case, author initials, ion macros, number grouping) deliberately NOT touched, copy-desk territory, would inflate the diff.

## 17. Figure final audit (2026-06-11): accessibility, old-vs-new declarations, producer-script coherence
- Old-vs-new figure diff (md5 + visual): 8 figures BYTE-IDENTICAL to submission (king, cmd, cmd_various, min_cluster, ctree, corner, pms_stats, RV-amplitude); all changed figures now declared in letters (item E enumeration extended: membership figs + mass/binary CMD + mass-cumulative x2 + brightness-cumulative restyles; LF one-source bin shift noted).
- POLICY: byte-identical original figures NOT regenerated despite CVD/grayscale ideals (cmd red/green symbols, king red/green lines, cmd_various rainbow isochrones + G_RP-G_RP label typo, RV coolwarm, font sizes sub-6pt after column scaling), diff discipline wins; referee accepted them in round 1; flagged for a future paper/version.
- Regenerated-figure fixes: PM contours now drawn from the PUBLISHED fitted model (2.542,-1.713 / 0.152,0.138 / corr 0.014) instead of an ad-hoc sample fit (center was 0.2 sigma off the cross); mass_binary vestigial black-star layer REMOVED (ASteCA 0.6.9 yields m1 for all 254, so the 'no calculated mass' rationale was obsolete; caption sentence dropped); isochrone drawn at explicit color=0.3 (no alpha compositing); crosshair/cross thickened for grayscale.
- Caption truth fixes: center/PM samples = 'likely cluster candidates (p>0.5)' (the CDS 321, as plotted); center lines = gold + 'approximately coincide'; parallax middle band = 1-sigma dispersion (not SEM); right panel 'adopted distance (mu_d, mode of sampled distances)'; cmd 'orange-red diamonds'; cmd_various right panel axes corrected (G_RP-J vs G_BP-G_RP); corner 'lines and dots in all panels'; real_sky '(shown in false color)'; Lambda range 2.5-3.6 (body+letters; N=5 point is 3.6).
- Cone King refits RE-RUN with the paper's exact prior parametrization (cosmic structure.RDP_bayesian; the 'identical priors' claim is now literally true): 50' Rc 1.91+-0.15 / Rt 44.5+-12.8 / C 1.35+-0.16; 60' Rt 45.6+-12.4; 70' Rt 52.2+-8.8, all PASS (0 div, Rhat<=1.0035). idata_king_cone{50,60,70}_modpriors.nc. Tex+letters synced. Story unchanged (50' consistent; 60/70 window-following).

## 18. Cone crossmatch + coherence sweep + the four flagged figures regenerated (2026-06-11)
- CONE SOURCE-LEVEL CROSSMATCH (review_repo/cone_crossmatch.py): wider runs recover 203/80%, 242/95%, 252/99% of the adopted 254 at 50/60/70 arcmin; 192 (76%) in all four; 50arcmin adds only 33 (32 beyond the 40arcmin footprint); 60/70 add 201/376 of which 83/114 INSIDE the footprint = lowered effective threshold, not new outer structure. Reported in the cone paragraph + new 'Common with adopted 254' column in tab:cone_robustness + letter.
- PM-CONTOUR DISCREPANCY EXPLAINED: regen error (ad-hoc Gaussian fit to the 321 plotted candidates instead of the published 254-fit model), already fixed; underlying offset real and expected, the 0.5<p<0.6 tail (67 sources) sits at (2.378,-1.633), dragging the 321-mixture to (2.508,-1.696) vs the 254 model (2.542,-1.713). Same loosening signature as the 60/70 runs. No pipeline bug.
- COHERENCE SWEEP (agent audit, all samples 321/254/130/16/193 inventoried + counts verified): FIXED, RV statistics were STALE (old binar_prob): now median/mean/std = -1.87/-9.40/33.1 km/s (n=16, v0.6.9 bp<0.6; was -6.11/-15.1/31.1; Table 2 row updated); faint-quartile K-S now reported honestly per pair (0.010, 0.052, 0.42, was '0.009-0.052' implying all significant); Hill-King annulus count 39->40 (exact radii); ~10 sample-wording fixes (members vs candidates vs reference sample; 'of at least' for >=0.6); Lambda 'up to ~3sigma'.
- FOUR FLAGGED FIGURES REGENERATED (user decision; declared in letter item E): ngc6383_cmd (Tol orange PMS / blue non-PMS / OPEN black no-2MASS = shape+fill channels; smooth posterior band from 90 trace draws replacing blocky rectangles; black solid/dashed/dotted mode/mean/median; classes from CDS PMSProb>=0.6, Ref==1) + ngc6383_cmd_various (viridis-ordered + 5 line styles isochrones via asteca 0.6.9 get_isochrone, MIST, fixes the G_RP-G_RP axis typo; panels relabeled) + king_profile_logscale (black points, blue King + 16-84% band from idata_king.nc, black dashed/dotted Rc/Rt, gray dash-dot b, axes r [arcmin] / rho [stars arcmin^-2]) + radial_velocity_amplitude (viridis sequential replacing diverging coolwarm; open gray diamonds for missing amplitudes; amplitudes re-queried from Gaia DR3 rv_amplitude_robust -> review_repo/rv_amplitude_dr3.ecsv, 29 sources / 8 amplitudes). Scripts: regen_cmd_figs.py, regen_king.py, regen_rv.py. All four captions rewritten to the new encodings.

## 19. Ladder p>=0.6 verification + notebook-convention audit (2026-06-11)
- VERIFIED: all four cone samples (40/50/60/70) are composite pseudo-probability p = p_HDBSCAN x p_freq, min 0.6003-0.6022, all >= 0.6; no ladder quantity in the tex sits on any other sample (agent-verified line by line). Parallax ladder comparison is apples-to-apples with production (no negative plx, no source at frac=0.1, ddof never changes a displayed digit).
- Notebook archaeology fixes applied: RV std -> 32.1 (np.std ddof=0, the notebook convention used by every other dispersion in the paper; 33.1 was pandas ddof=1); K-S quartet pairing corrected (faintest differs from the two INTERMEDIATE quartiles, 0.010/0.052, but matches the BRIGHTEST, 0.42, previous sentence had the pairing inverted); King prior statement corrected to the coded priors (k~U(b,2rho_max), Rc~U(0,0.8 T_max), not U(0,0.8 R_t)); Hill-King shell count back to 39 (exact Hill 28.32: borderline source at 28.310 is inside Hill); Results opener estimator-neutral + King point estimate relabeled MEDIAN (production RDP_bayesian returned medians; figure legend + caption updated, king fig regenerated).
- Boundary-source note (no digit changes): exactly one source at binar_prob=0.700, notebook used >=0.7, finalize/tex use >0.7; <m> = 1.31+-0.10 either way. PMSProb/Q/p-thresholds have empty boundaries.

## 20. Soft final polish (2026-06-11)
- 17 mechanical defects fixed in changed text (all splice artifacts): abstract '1.6-6.3 Myr ago' restored; 'paper-faithful' workflow jargon removed from cone-table caption; WRONG appendix cross-ref fixed (Sagitta/RV figs -> app:supplementary, not app:hdbscan); 4 math-mode hyphen ranges -> en-dashes; K-S -> KS harmonized; LF acronym re-defined; Hill-shell sentence re-parsed; RV 11.4% appositive; T_max verb elision; MST caption parenthetical re-attached; an NGC; CDS tense aligned with Data availability; dash-dotted burgundy; comma splices.
- Letter Λ range synced (2.5-3.6, up to ~3σ). Package check: 0 failures; zip/PDFs newer than all inputs; zip tex byte-identical; figures md5-match.
- Final state: clean 19pp / marked 23pp / 0 undefined / zip standalone-compiles.

## 21. Detail coherence pass (2026-06-11)
- CDS: ReadMe de-jargonized ('paper-faithful' removed), missing-values wording fixed (only 2MASS can be missing; Sagitta outputs exist for all 321), '[yr]' CDS unit notation, 'stars: pre-main sequence' keyword added, title casing matched; CDS zip rebuilt.
- Sagitta truth fix in fig:cmd caption: no-2MASS sources are EXCLUDED from the classification (Sagitta outputs exist for them; 'cannot be applied' was wrong and contradicted pms_stats). Footnote 'identified in the CDS table' overpromise dropped. CDS tense unified ('is available', pointing to Data availability).
- Captions: corner plot (black square = joint mode, off-diagonal only; blue marginals), pms_stats 'of at least 60%', min_cluster_size sweep-truncation explained (stability criteria; axis does not span 10-299).
- Letters: source-zip filename corrected; point-2 'flags' overclaim fixed (one Ref flag; members follow from pMember>=0.8); points 7/13 original-vs-revised table renumbering noted; Cerulo annot-1 (sentence deleted, not reworded) and annot-16 wording aligned.
- Bib: astropy:2022 replaced with canonical ADS entry (FIXES VISIBLE 'apj' in the compiled references + stripped accents); 2024arXiv240305143H -> Hunt & Reffert 2024, A&A, 686, A42; 2024arXiv240301030P -> Pulgar-Escobar & Henriquez-Salgado 2024, BAAA, 65, 146; McInnes2017 pages=205. Keys unchanged (no tex edits needed).
- Appendix A orphan label resolved (historical-compilation pointer added in the catalog-comparison paragraph).
- Meta-docs refreshed (MANIFEST sizes/dates/recipe with the 25-ARCMIN exclusion; this changelog header 23 pp).
- Known internal notes (no action): review_repo/README is a 2026-06-02 snapshot (top-fixes list all DONE); 25' notebooks are the stale producers of min_cluster/ctree figs (other investigation; figures themselves match the 40' production run); ECSV machine copy lacks unit metadata (CDS deposit = ReadMe+dat, unaffected).

## 22. Faculty referee-panel pass (2026-06-11; Davis+22 primer method, 5 independent passes)
- Verdicts: figures-first = revision on anchoring; claim-chain = major (selection/completeness systematics); statistics = minor-to-moderate (every published number reproduces exactly); reproducibility = major (spec gaps); positioning = minor-to-moderate (overclaim gradient).
- TIER A APPLIED (21 tex edits + 3 bib entries; no values changed): binary-KS sensitivity sentence (threshold-stable p<=0.018 for cuts 0.5-0.8; overlap-mass D=0.19/p=0.07; mass-matched D=0.36/p<0.001); Y_frac 2MASS-selection caveat + bounds [0.21,0.45] + B&F comparison qualified; HD mu_delta ~6 sigma with own error in quadrature; Holm note on quartile KS; ADQL center (263.6715,-32.5773) + in-query parallax window (recovered from the raw ECSV QUERY metadata); parallax/distance generative priors written out; King Gaussian likelihood + half-normal scatter stated; ASteCA alpha/beta MISATTRIBUTION FIXED (binary-fraction params, not IMF; Duchene & Kraus 2013 mass-ratio dist; v0.6.9 stated); NUTS cite Hoffman & Gelman 2014 + PyMC Abril-Pla 2023 (replacing the NumPyro miscite); N=254 explicit in t_rh; KDE bandwidth ~6.7 arcmin stated; abstract/conclusions recalibrated (apparent age spread, upper limit, primordial-or-rapid-early framing, dynamical-youth logic un-inverted); contribution paragraph added at the end of the Introduction; CDS/SIMBAD acknowledgment; Table row renamed Star-formation age range (isochrone bracketing); DM-prior circularity disclosed; PMS-broadening upper-limit caveat. Letter item M documents all.
- Clean PDF now 20 pp (body+refs ~14 pp; appendices from p. 15), the contribution/caveat additions cost ~1 page. Marked 23 pp.
- TIER B (new analyses, deferred to author decision): Gaia selection-function completeness rerun (King/Lambda/KS on G<17 complete subsample); pre-selection field+cluster mixture dispersion for exclusion yardsticks; coeval-population simulation for the PMS spread; offset-field contamination estimate; synthetic-Q regime test; 50-70 arcmin density-profile overlay panel; f_i distribution publication; pms_stats age-band annotation.

## 23. ASteCA alpha/beta verified against source; binary-fraction form added (2026-06-13)
- Confirmed directly in the installed asteca 0.6.9 source: synth_model.py:1142 `b_p = clip(alpha + beta/(1 + 1.4/m), 0, 1)` with docstring 'Alpha/Beta parameter for the binary fraction' -> alpha/beta are the MASS-DEPENDENT BINARY FRACTION params, NOT IMF (confirms the round-22 correction). gamma='D&K' = synth_cluster_priv.py:159-160 'mass-dependent mass-ratio distribution from Duchene & Kraus 2013' (confirms the D&K 2013 citation).
- Tex now gives the explicit form b(m)=alpha+beta/(1+1.4/m), alpha=0.09/beta=0.94 (rises ~0.09 -> ~1 with mass), + D&K 2013 mass-ratio distribution, differential reddening = 0. Letter item M refined with the verified form. Item lettering A-M.
- Final state: clean 20 pp (body+refs 14), marked 23 pp, 0 undefined; zip 4.39 MB, standalone-compiles, tex byte-identical, no 25-ARCMIN dir, 19 figures. Tier B analyses deferred to a future paper (user decision).

## 24. Faculty literature audit, all claims vs latest arXiv/SciX (2026-06-13)
- Extracted ALL ~130 claims (agent); ~20 load-bearing regime-sensitive. Verified each regime-sensitive cited paper via SciX/ADS against NGC 6383 regime (young ~330-900 Msun open cluster, 1.1 kpc):
  - Pavlik 2020 = ONC (young OC) tau_v=2.1-3.5 t_rh; Moeckel&Bonnell 2009 = hundreds-thousands stars 'local open clusters'; Chen 2007 = NGC 2244 + NGC 6530 (SAME Sgr OB1!); Allison 2009 = ONC-like; Parker&Goodwin 2015 = Lambda_MSR classical-sense literal; C&W 2004 = Q boundary 0.8; Sagitta valid to 5 kpc; Angelo/Risbud 2025 = method (regime-general); Hunt 2024 mass. ALL regime-appropriate, no Polak-type mismatch.
- KEY FINDING: two 2026 papers directly counter the central primordial-segregation claim, Zhang 2026 (AJ 171, 236; 3881 Gaia DR3 OCs, SAME Lambda_MSR method: young <10 Myr segregation consistent with random, kinematic overheating, classical seg only >100 Myr -> DYNAMICAL not primordial) and Amiri et al. 2026 (Nbody6++GPU vs Pang+2022 OCs: primordial seg not a fundamental ingredient). BOTH now cited (bib + discussion + conclusions); text explicitly states our single-cluster detection does not settle the primordial-vs-dynamical debate. Referee letter item M extended.
- No competing NGC 6383 study 2023-26 (only our own BAAA precursor); no all-sky OC catalog supersedes Hunt&Reffert 2024 as comparison baseline. Memory: faculty-paper-review-method + citation-regime-context-check updated.
- Final: clean 20pp, marked 23pp, 0 undefined, zip standalone. Bib +Zhang2026 +Amiri2026.

## 12. Voice-restoration pass (2026-06-14)
- User feedback: too much of their original wording/figures had been changed; "no todo tiene que cambiar". Restored author's voice where the change was NOT required by referee/Cerulo/numbers/figures.
- Reverted ~8 sentences to verbatim original submission: "utilized" (King profile, MIST isochrones) restored from "used"; "crucial" (prior choice, t_rh->t_seg) restored; "Notably, 94% of CTTS..." restored; "utilizing optical photometry" (Kumar) restored; reddening "shell-structured ionized HII region" sentence restored; Sagitta "efficacy stems from...surpassing traditional isochrone techniques" two sentences restored; CNN "dedicated to determining...and estimating stellar ages" restored.
- Kept all required edits (p~ terminology, v0.6.9 numbers, citations, sampler/convergence, cone robustness, Hα, CDS).
- FIGURES: user chose to KEEP CVD-safe palettes (orange/blue + viridis) over restoring original red/blue, and to KEEP 1-panel luminosity function over restoring the 2-panel version, after the accessibility-vs-diff tradeoff was made explicit. No figure regen needed.
- Rebuilt: clean 20pp / marked 23pp / 0 undefined; source zip 4.39MB (with .bbl, no 25-ARCMIN subdir, no compiled pdf), standalone-compiles, aanda.tex byte-identical, 19 figures.

## 13. US/UK spelling, standardized to US English (2026-06-14)
- Decision: US English throughout (A&A accepts either if consistent).
- Manuscript fixes: "circumstellar discs" -> "disks" (x2; doc already had US "disks" at the YSO intro, so these were the outliers); "towards" -> "toward" (x1; doc already had "toward" x3). VizieR "catalogue access tool" LEFT as-is (official CDS proper name). Bib titles left verbatim (published-paper titles).
- Letters fixed to match (response_to_referee + response_to_cerulo, .md & .txt): colour->color, catalogue/Catalogue->catalog/Catalog, centre->center. No referee/Cerulo verbatim blockquotes touched.
- Rebuilt: clean 20pp / marked 23pp / 0 undefined; zip 4.39MB standalone-compiles, tex byte-identical, 19 figs, no 25-ARCMIN, .bbl included.

## 14. Figure-caption descriptions restored (2026-06-15)
- User: descriptive/interpretive caption prose was lost in the earlier "label not teach" tightening; restore it. Ground-truthed against the actual regenerated PDFs (Read center + parallax figures) before restoring, to avoid asserting elements not in the new figures.
- Restored: center (axes "Right Ascension (alpha) and Declination (delta)" + "revealing the density gradient"); parallax middle-panel error bars "especially pronounced at fainter magnitudes" + right-panel "only the portion within 0.8-1.2x min/max used distances is displayed" (both verified present in the PDF); cmd_various ("representing different member classifications and evolutionary states" + "highlighting the evolutionary progression"); mass_seg ("highlighting the potential influences of mass..." + KS "quantitatively assessing... underscoring the statistical validity"); pms_stats (per-panel explanatory clauses); radial_velocity (rv_amplitude_robust = "total amplitude in the radial-velocity time series following outlier removal").
- NOT restored (would contradict regenerated figures): center 5-pc scale bar (verified ABSENT in regenerated center_determination_paper.pdf); old red/blue colors; cmd color-axis error bars (not in regen). cmd + probabilities left as-is (already complete; fidelity-vs-faint note lives in body).
- Rebuilt: clean 20pp / marked 23pp / 0 undefined; zip 4.39MB standalone, tex byte-identical, 19 figs.

## 15. Center figure scale bar restored + generator relocated (2026-06-15)
- User: the 5-pc scale bar must be in the center figure; also flagged that center generation did not belong in regen_prob_massbinary.py ("in jupyter it's elsewhere").
- Traced canonical source: notebook graph_center_determination(..., paper_single=True, distance_scale=mu_r) drew the bar via astropy add_scalebar (5 pc, white, WCS axes). The package facade (cosmic/analysis/figures.py:578) is a stripped scatter and was NOT the paper generator.
- Extracted the center-figure generation into a dedicated review_repo/regen_center.py (faithful: viridis + KDE contours + gold crosshair + 5 pc scale bar); removed the center block from regen_prob_massbinary.py (now only does probabilities, mass-binary, proper-motion). Scale bar drawn in BLACK (the notebook's white bar is invisible on this white-background plot), length cos(dec)-corrected so 5 pc is correct along the RA axis at 1.110 kpc.
- Verified the bar renders (read the regenerated PDF). Restored the caption sentence "A scale bar of 5 parsecs is included at the bottom right...".
- Rebuilt: clean 20pp / marked 23pp / 0 undefined; zip 4.39MB standalone, tex byte-identical, 19 figs.

## 16. Figure style harmonization (2026-06-15)
- Audited all 19 figures visually for color/size/style coherence.
- Fixed cross-figure inconsistencies: (a) ngc6383_mass_binary axis labels "Color (G_BP-G_RP) [mag]"/"G_mag [mag]" -> "G_BP-G_RP [mag]"/"G [mag]" to match cmd + cmd_various (regen_prob_massbinary.py); (b) cumulative_by_mass_and_type + _mseg "Radius (arcmin)"/"Cumulative (normalized)" -> "Radius [arcmin]"/"Normalized cumulative count" to match cumulative_by_brightness and the paper-wide [unit] bracket convention (finalize_v069.py); (c) center crosshair gold -> dark-orange to match proper_motion and improve contrast against the yellow viridis high-p points (regen_center.py + caption).
- Confirmed already-coherent: viridis for all continuous quantities (normal direction); orange=PMS / blue=non-PMS categorical across cmd/cmd_various/pms_stats; blue/orange/teal/red qualitative quartiles across all cumulative plots; black isochrones; black King radii lines.
- Deliberately kept: mass_binary viridis_r (mass: yellow->purple), same viridis family, reversed direction appropriate for a different quantity (high mass = dark).
- User decision: appendix HDBSCAN diagnostics (min_cluster_size larger font; condensed_cluster_tree = native HDBSCAN plot) left as-is (low impact, condensed_tree not cleanly restyleable).
- Rebuilt: clean 20pp / marked 23pp / 0 undefined; zip 4.39MB standalone, 19 figs.

## 17. Full coherence + error audit (2026-06-15)
- BUG FIXED: Table 1 gravitational bound radius 42.8 -> 42.45 arcmin (stale value; 42.45 is the production value used as King-prior T_max in every script and shown in the real_sky figure legend; "max(Hill 28.32, bound 42.45)"). Error +-1.6 retained and verified correct (propagates from m_c 10.2% -> 3.4%, plus ~3% Oort, in quadrature -> 3.9% x 42.45 = 1.6; Hill +-1.0 likewise from 3.4% x 28.32).
- Error completeness: every measured quantity in Table 1 and the body carries an uncertainty; remaining bare numbers are error-parts of X+-Y, range bounds, counts, priors, or explicitly-approximate single values (KDE bandwidth ~6.7 arcmin). No measured quantity missing its error.
- Number coherence verified across abstract / Table 1 / body / conclusions: age 3.53(+1.40/-1.00), distance 1.110+-0.060, Rc 1.95+-0.19, Rt 40.4+-14.3, R_GC 7.19+-0.07, t_rh 30.5+-9.4, t_seg 2.94+-1.17, <m> 1.31+-0.10, m_max 13.56+-3.25, C 1.32+-0.16, Q 1.03+-0.03, Yfrac 0.275(+0.067/-0.058), KS D=0.221/p=0.010, 254 ref / 321 candidates. Cone-robustness body text matches Table 2 (common-with-254: 254/203/242/252; ref samples 254/236/443/628).
- Citations: all \citep/\citet keys resolve in the bbl (LaTeX reports 0 undefined citations); no orphan or undefined keys.
- Terminology coherent: "tidal" remaining only as "tidal features" (Gaia tail literature) and the historical-table column; "membership probability" without "pseudo" only where it legitimately denotes the native HDBSCAN probability or the explicit "not a calibrated posterior membership probability" contrast. US spelling throughout.
- Letters cross-checked: the two p=0.07 mentions are legitimate (cerulo = explicit historical "previously quoted as p=0.07"; referee = current overlapping-mass-range test D=0.19/p=0.07). No stale numbers in letters.
- Rebuilt: clean 20pp / marked 23pp / 0 undefined; zip 4.39MB standalone, 19 figs.

## 18. Final review of response letters (2026-06-15)
- Referee letter (20 points + additions A-M) and Cerulo letter audited end-to-end. All numbers cross-checked against the final paper and match (D=0.221/p=0.010, t_rh 30.5+-9.4, t_seg 2.94+-1.17, <m> 1.31+-0.10, m_max 13.56+-3.25, Yfrac 0.275, M=N<m>=332+-26, R_hl 1.94+-0.36 pc, King-50' 44.5+-12.8, HD159176 astrometry, etc.). Appendix renumbering (A historical / B HDBSCAN+corner / C supplementary) and figure cross-refs (corner = Fig B.3) verified consistent.
- Fixed two coherence drifts introduced by the later caption-restoration and figure-correction passes:
  (a) referee item H: "figure captions were tightened to label rather than discuss" -> "revised to describe the figure content without re-deriving the body text" (captions now carry restored descriptive prose).
  (b) Cerulo §1: "unified the 'density gradient' contour wording across the proper-motion and probability figures" -> clarified per-figure (center = kernel-density gradient; proper-motion = iso-probability contours of the fitted 2D Gaussian model), since the proper-motion caption was corrected away from "density gradient".
- Both fixes applied to .md and .txt (synced). No other inaccuracies found; the two p=0.07 mentions remain legitimate (historical reference / overlapping-mass-range test).

## 19. Significant-figure / precision audit (2026-06-15, ultracode workflow)
- Ran a 10-dimension adversarial audit workflow (70 findings, 60 confirmed, 0 blockers): sigfig-quantities 16, sigfig-errors 11, sigfig-figures 6, number-coherence 1, citations 3, terminology 14, figure-body 4, claim-chain 5, letters 0.
- Convention applied (PDG RPP rounding rule + GUM 2-sig-fig + IAU/AAS best-practices, arXiv:2106.01477): uncertainty to ~2 sig figs, value to the same decimal place, SAME precision everywhere. ("GS 26" asked about by user = not a real standard; not found.)
- PRECISION FIXES (body over-precision -> table canonical, all confirmed): distance 1.1100±0.0600 / 1.110±0.060 -> 1.11±0.06 (L91 caption, L286, L305, L398); DM 10.300±0.262 -> 10.30±0.26; Av 1.240±0.262 -> 1.24±0.26; Rc 1.950±0.190 -> 1.95±0.19 (L299, L398); k 4.910±0.437 -> 4.91±0.44; b 0.01110±0.00630 -> 0.011±0.006; DM-distance 1.150_{-0.130}^{+0.147} -> 1.15_{-0.13}^{+0.15}; center body 263.6826±0.1122 / -32.5838±0.1122 -> 263.683±0.112 / -32.584±0.112; log age 6.550±0.145 -> 6.55±0.15; m_c 902.27±92.3 -> 902±92 (+\mathrm unwrap).
- KEPT (Gaia-norm 3 sig figs, internally consistent): PM 2.542±0.152 / -1.713±0.138 (captions aligned UP to 2.542/-1.713), center table 263.683±0.112, Rt 40.4±14.3, m_max 13.56±3.25 (rounding would cascade to figure legends), t_seg 2.94±1.17.
- TERMINOLOGY/SYMBOL: A_v->A_V (x2), Simbad->SIMBAD, lambda Bootis->lambda Boo, T Tauri->T-Tauri (L312), pre-main-sequence->pre-main sequence (abstract), data set->dataset (x2), color magnitude->color-magnitude (abstract), bare "mag"->\mathrm{mag} (L259, L387 x2), R_C-\mathrm{H}alpha->R_C-H alpha (x3), CMD caption G_BP subscripts aligned to sibling caption, mu_alpha^*->mu_{alpha*} (L84, L381), \cite->\citet (L68), isochrone order body "median,mean,mode"->"mode,mean,median" to match caption, RV "11.4%" annotated "(29 of 254 sources)".
- CLAIM-CHAIN: abstract "points to a primordial..." -> "is consistent with a primordial..." (overclaim gradient fixed); conclusions "segregation... indicating not fully relaxed" -> corrected inverted logic (dynamical youth -> relaxation too slow -> primordial/early-dynamical).
- FLAGGED, NOT fixed (figure regen, low impact): real_sky legend R_hm=6.24 vs body 6.26 (stale 0.02 arcmin; figure has no regen script, DSS-backed); R_t=40.39 in king/cumulative_brightness/real_sky legends vs text 40.4 (figure-internal auto-label; fixing 2 of 3 would make figures mutually inconsistent since real_sky has no script).
- Rebuilt: clean 20pp / marked 23pp / 0 undefined; zip 4.39MB standalone, 19 figs.

## 20. Figure-label precision fixes regenerated (2026-06-15)
- king_profile_logscale.pdf (regen_king.py): R_t legend 40.39 -> 40.4 (label :.1f), b legend 0.0111 -> 0.011 (label :.3f, matches Table 1 / body 0.011).
- cumulative_by_brightness_paper.pdf (regen_brightness_cumulative.py): R_t legend 40.39 -> 40.4.
- real_sky.pdf: NEW dedicated generator review_repo/regen_real_sky.py (the figure had no script; was notebook-only). Faithful reproduction (DSS2-red FITS + WCS projection, viridis image, dark-red p>=0.6 members, blue center marker, 7 radius circles, white 5 pc scale bar, legend). FIXES: half-mass-radius legend 6.24 -> 6.26 arcmin (= 2.02 pc, the adopted value; old figure carried a stale 6.24); R_t legend 40.39 -> 40.4; R_hill 28.32 -> 28.3; every radius label now at the text/Table-1 precision. DSS raster downsampled 2x and saved at dpi=200 to keep the PDF ~4.4 MB (vs a 9 MB full-res render).
- All three figure legends now agree with the text: R_c=1.95, R_t=40.4, R_hl=6.02, R_hm=6.26, R_hill=28.3, R_bound=42.45, b=0.011.
- Rebuilt: clean 20pp / marked 23pp / 0 undefined; zip 4.9 MB standalone, 19 figs.

## 21. Copy-edit micro-consistency sweep (2026-06-15, ultracode workflow #2)
- Triggered by the user noticing the prior audit missed the main-sequence adjectival hyphenation. Ran a 5-dimension copy-edit workflow (31 findings, 30 confirmed) + verified each form against ACTUAL English/astronomy convention (not blind internal-majority, per user instruction).
- HYPHENATION (adjectival compounds): machine-learning techniques (L33,396); star-formation region (L31,48); early-dynamical segregation (L35,366x2); proper-motion distribution/values/components/data (L84x2,128,284,473); dark-blue + dark-slate-gray line (L494); O7V-type stars (L50); cone-search radius (L184).
- CAPITALIZATION: Galactic coordinates (L48, IAU convention); half-normal (L128); Hill radius (L191); Upper/Lower panel (L247, sentence case); 4 section headings -> sentence case (A&A style: L120,297,303,330); right ascension/declination lowercase (L73).
- ABBREVIATION: Section \ref -> Sect. \ref (x4, A&A); prose RA/DEC -> R.A./Dec. (L284, matches Table 1).
- TYPOGRAPHY: Unicode em-dashes -> spaced LaTeX, (L184,305); {\it Gaia} -> \textit{Gaia} (L416 x3); {\sc COSMIC} -> \textsc{COSMIC} (L95); \mathrm{G} -> italic G (L77); numeric range 11-13 -> 11--13 (L77); italic G_{BP} subscripts -> upright \text{} (L56); Unicode apostrophe -> ASCII (L223).
- WORD-CHOICE: "Two Micron All Sky Survey" (no hyphen = official 2MASS name, L79); "Part of these results IS" (subject-verb, L299); "two-dimensional Gaussian" (L84, matches body).
- GAIA ITALICS: mission name "Gaia" made consistently \textit{Gaia} (11 -> 38; +27 prose instances incl. title/abstract) per A&A/DPAC house style; source designations (Gaia DR3 <id>) left roman and made consistent (L381 \textsc -> roman); \textsc{Gaia} EDR3 -> \textit{Gaia} EDR3 (L207); package \textsc{Gaiadr3_zeropoint} and table column GaiaDR3 left as-is.
- VERIFIED-CORRECT-NOT-BLIND examples: T Tauri/T-Tauri confirmed genuinely split in the literature (kept T-Tauri, minimal-diff + internally consistent); Table~\ref non-breaking space (~) NOT removed despite being minority (it is the more-correct form -> would degrade); 2MASS official name has no hyphen.
- Rebuilt: clean 20pp / marked 23pp / 0 undefined / 0 overfull>20pt; zip standalone, 19 figs.

## 22. Timeless-phrasing pass (2026-06-15)
- User: "new" (and similar time-relative words) are not timeless in a journal article. Swept new/recent/recently/modern/latest/current/novel/now/previously/to-date.
- REMOVED/REWORDED (paper-timeliness, ages badly): L31 "new \textit{Gaia} DR3" -> "\textit{Gaia} DR3"; L52 "modern spectroscopic studies" -> "spectroscopic studies"; L372 "modern spectroscopic work" -> "the spectroscopic literature"; L225 "matches recent \textit{Gaia} DR3 practice" -> "matches established ..."; L398 "In line with recent \textit{Gaia}-based work" -> "In line with \textit{Gaia}-based work"; L366 "Recent population-level studies" -> "Population-level studies"; L404 "although recent population-level analyses" -> "although population-level analyses"; L286 "more recent determinations" -> "later determinations"; L402 "aligns with recent studies" -> "aligns with previous studies".
- KEPT (astrophysical, describes the object's youth, timeless in meaning): "recent[ly]/possibly extended star formation" (abstract x2, L328, conclusions); "recently entered the main-sequence track"; "New General Catalog" (proper noun); "previously measured ages" (Sagitta training data, factual).
- Rebuilt: clean 20pp / marked 23pp / 0 undefined; zip standalone, 19 figs.

## 23. Figure fixes + appendix rebuild (2026-06-15, user figure review)
- Fig. 5 King band non-uniformity EXPLAINED (not a bug): the 16-84% posterior band is narrow near the core (profile set by the well-constrained central density k) and broadens at large r (governed by the weakly constrained R_t [~35% err] and background b [~55% err]); on a log axis the downward spread is large because some posterior samples truncate near small R_t / low b. Added a clause to the King caption stating this.
- Fig. 6 (CMD): the light-gray posterior-draw "spaghetti" now has a dedicated legend entry "Posterior draws (N=150)"; bumped draws 90->150 (literature check via subagent: spaghetti is the standard, A&A-acceptable way to show MCMC isochrone-fit uncertainty; a percentile band fails near the turnoff/PMS locus where isochrones cross, Bossini+2019, Dias+2021, Perren+2015, AAS/A&A graphics guides). Caption updated to "150 single-star sequences drawn from the isochrone-fit posterior".
- Fig. 8 (mass/binary CMD) ZIG-ZAG FIXED: the dashed "best mode-fit isochrone" was the scattered ASteCA *synthetic sample* connected as a line sorted by magnitude (non-physical side-to-side wiggle fainter than G~14). Replaced with the clean get_isochrone locus saved by regen_cmd_figs.py (mode_locus.npz), plotted in native isochrone order -> smooth monotonic curve following the data. regen_real... new helper regen_real_sky.py untouched; mode_locus.npz is a build artifact, excluded from the submission zip.
- Fig. 12 (Lambda_MSR) improved: added legend ("Lambda_MSR (NGC 6383)" + "Lambda_MSR=1 (no segregation)"), a shaded Lambda>1 "segregated" region with label, grid, larger panel, fuller axis labels.
- APPENDIX rebuilt (user: "build it better, not just here-are-the-images; we can expand here"): added a contextual intro to Appendix A (historical table); expanded Appendix B with substantive prose, HDBSCAN m_cl sweep + adopted m_cl=43 rationale, condensed-tree persistence interpretation, and a full discussion of the ASteCA corner plot (age-extinction-distance degeneracies, prior-dominated metallicity, sigma nuisance term); added an intro to Appendix C. The added text anchors the full-width figure floats and removes the large blank/"stacked headings" gap on the former appendix page.
- Rebuilt: clean 20pp / marked 24pp / 0 undefined / 0 overfull>30pt; zip 4.95 MB standalone, 19 figs.

## 24. King band smoothing + Fig 6 legend confirmation (2026-06-15)
- Fig. 5 King 1-sigma band looked "bitten"/jagged at the edges: it was a sampling artifact (only 200 posterior draws + per-sample R_t truncation make the 16-84 percentile envelope step). Fixed by using the FULL posterior (8000 draws) and a finer radial grid (400->600 points) in regen_king.py; band edges are now smooth (the physical widening at large r, from weak R_t/b constraints, remains). Verified visually.
- Fig. 6 "Posterior draws (N=150)" legend entry: CONFIRMED present in the standalone figure AND in the compiled aanda.pdf (page 5). The user's view lacking it was a stale/cached PDF render; aanda.pdf was recompiled fresh (newer than all figures) so reopening shows it.
- Rebuilt: clean 20pp / marked 24pp / 0 undefined; zip standalone, 19 figs (mode_locus.npz build-artifact excluded).

## 25. FIXED stale figures in the marked-diff PDF (2026-06-15)
- ROOT CAUSE: marked_changes/ keeps its own copy of Figures/, which was never synced with the regenerated figures (all regen scripts write to clean_source/Figures/). The marked-diff PDF therefore embedded STALE figure images (old colors, jagged King band, no cmd posterior-draws legend, zig-zag Fig 8, etc.) even though its captions (from the tex) were current. The clean PDF and the source zip read clean_source/Figures/ and were always correct, only the editor-facing marked-diff PDF was affected.
- FIX: copied all 19 clean_source/Figures/*.pdf -> marked_changes/Figures/ (now byte-identical) and recompiled the marked diff. aa52082-24_marked_changes.pdf is now 24pp, 0 undefined, 5.58 MB, embedding the current figures (verified Fig 6 legend "Posterior draws (N=150)" and smooth King band on the marked page 7).
- Going forward: when regenerating figures, sync marked_changes/Figures/ from clean_source/Figures/ before recompiling the marked diff.

## 26. Suppressed output-invisible \textrm->\mathrm noise in the marked diff (2026-06-15)
- User noticed the marked diff flagged "20.0 Myr -> 20.0 Myr" (Lindoff 1968) and similar unit lines as changed, although the value/output is identical.
- Cause: the early units normalization changed \textrm{}->\mathrm{} (6 instances in the original: arcmin, arcsec, Myr x4); latexdiff marks the macro change even though \textrm and \mathrm render identically.
- Decision: keep \mathrm (the correct/recommended macro for in-math units; the current doc is 100% \mathrm). To stop the diff from showing these output-invisible changes, normalized the diff BASELINE (marked_changes/old_submitted.tex) \textrm->\mathrm too, so latexdiff sees identical macros. The real submitted manuscript is unaffected; the marked diff now shows only output/content changes.
- Verified the Lindoff "$20.0\,\mathrm{Myr}$" and the angular-size/age line carry no \DIFadd/\DIFdel. The remaining "$10.0$ Myr" mark is legitimate (its HD 159176 sentence was genuinely rewritten).
- Also re-synced marked_changes/Figures/ from clean_source and recompiled: marked 24pp, 0 undefined, 5.58 MB.

## 27. Suppressed Gaia-italicization noise in the marked diff (2026-06-15)
- User: the marked diff flagged "using Gaia -> Gaia" (title) etc., the \textit{Gaia} italicization (house style) cluttering the diff.
- Same baseline-normalization approach: mirrored the current doc's Gaia/markup treatment onto marked_changes/old_submitted.tex, {\it Gaia}->\textit{Gaia} (3), \textsc{Gaia}->\textit{Gaia} (EDR3 label), \textsc{Gaia DR3 <id>}->roman designation, {\sc COSMIC}->\textsc{COSMIC}, and bare mission "Gaia"->\textit{Gaia} (17). Baseline now has the identical Gaia markup as the current doc, so latexdiff no longer marks the italicization in unchanged sentences.
- Verified: title now marks only the real addition ("fundamental properties,"); 0 spurious Gaia DIFadd/DIFdel tokens. Gaia mentions inside genuinely added/rewritten sentences still appear (correctly) as additions.
- Marked regenerated (figures synced from clean): 24pp, 0 undefined, 5.58 MB.
- The marked diff now reflects substantive (wording/number/science) changes, not output-invisible or pure-style markup normalizations (\textrm/\mathrm, Gaia italics). The clean submission source/zip were never affected by any of this.

## 28. Comprehensive markup/cosmetic noise removal from the marked diff (2026-06-15)
- User: "no quiero ruido", the marked diff was flagging output-invisible/cosmetic changes (\mathrm{G}->G, Gaia italics, hyphenation, capitalization, Section->Sect., percent->%, etc.) that clutter the editor-facing diff.
- Approach: normalized the diff BASELINE (marked_changes/old_submitted.tex) to the current doc's cosmetic conventions so latexdiff only marks substantive content. Mirrored onto the baseline: \textrm->\mathrm; all Gaia markup (\textit/\textsc/designation); \mathrm{G}->G; $G_{BP/RP}$->\text{} subscripts; numeric range hyphen->en-dash; unicode em-dash->--; unicode apostrophe->ASCII (both files; clean was already ASCII); machine-learning, star-formation, main-sequence, O7V-type, proper-motion hyphenation; Galactic, Hill radius, Upper/Lower panel, sentence-case headings, right ascension/declination, R.A./Dec.; Section->Sect.; dataset, "Two Micron All Sky", color-magnitude, half-normal, A_v->A_V, pre-main sequence, "Young stellar objects"; bare "X percent"->"X\%".
- Result: the marked diff now shows only genuine content changes (rewordings, recomputed v0.6.9 numbers, sig-fig precision, terminology tidal->King outer / member->candidate, new analyses, deleted/added sentences). All output-invisible and pure-cosmetic markup noise suppressed. Verified: 0 residual \textrm / \mathrm{G} / Gaia-italic / Objects / standalone-percent marks.
- The clean submission (aanda_revised_clean.pdf) and source zip were NOT affected by any baseline normalization (the baseline is only the latexdiff reference). clean had 0 unicode apostrophes so it was unchanged.
- Marked: 24pp, 0 undefined, figures in sync, 5.58 MB.

## 29. Figure-science fixes + version timelessness + more diff cleanup (2026-06-15)
- BUG (Fig.7 cmd_various): the loga=6.6 isochrone had 544 pts and a spurious bright tail to G=0.93, while neighbors stopped at G~5-8. CAUSE: 6.6 is a MIST grid node -> get_isochrone returns the full (bright) isochrone, whereas interpolated ages (6.2/6.4/6.8/7.0) are high-mass-truncated. FIX: clip all plotted isochrones to G>=7 (observed range; brightest member G~8.8). All ages now span a consistent range, turnoffs ordered correctly by age, no empty top axis. (regen_cmd_figs.py)
- Fig.2 (center) & Fig.3 (proper motion) contours were too small/bunched (7-8 auto density levels near the peak). FIX to the standard astronomical sigma-enclosed convention: Fig.3 = 1,2,3-sigma iso-probability ellipses of the 2D Gaussian (analytic pdf_peak*exp(-k^2/2)); Fig.2 = 1,2-sigma KDE iso-density contours (39.3, 86.5% enclosed, +1.4x bandwidth smoothing; 3-sigma omitted as it traces the sky halo not the cluster). Captions updated with the sigma levels. ("GS 26" referenced by the user is NOT a real standard, confirmed via search; the convention is sigma/enclosed-fraction levels.)
- Lambda_MSR (Fig.12): removed the gray "segregated (Lambda>1)" shading/label (looked bad/trivial). The "always >1" is NOT a bug: Lambda declines 3.6->1.2 as N_MST grows, the classic mass-segregation signature (few most-massive strongly concentrated, trend to ~1).
- TIMELESSNESS: removed software version numbers from prose, "Python 3.12"->"Python", "PyMC 5"->"PyMC" (versions age; reproducibility preserved via the released GitHub repo). Kept "ASteCA v0.6.9" (referee-requested; the masses/binarity depend on it). NOTE: A&A/AAS recommend citing software versions for reproducibility, so this is a style trade-off favoring timeless prose; the repo pins exact versions.
- MARKED-DIFF cleanup (continued): fixed a broken baseline COSMIC heading (a prior {\sc COSMIC}->\textsc mirror had orphaned the braces), and mirrored the proper_motion caption cosmetics (proper motion->proper-motion x2, mu_alpha symbol, 2.54->2.542 value) onto the baseline so they no longer show as noise.
- Rebuilt: clean 20pp / marked 24pp / 0 undefined; zip 19 figs.

## 30. Removed gridlines from figures (2026-06-15)
- Convention check: A&A/AAS figures are conventionally clean without background gridlines. Removed the two grids present: the background data grid on Lambda_MSR (Fig.12, finalize_v069.py) and the white dotted WCS coordinate grid on real_sky (Fig.C.1, regen_real_sky.py). All other figures were already grid-free.
- Rebuilt: clean 20pp / marked 24pp / 0 undefined; zip 19 figs. Verified both figures grid-free.

## 31. Restored the WCS coordinate grid on real_sky (Fig. C.1) (2026-06-15)
- Correction to §30: the WCS celestial coordinate grid IS standard on sky finding charts (unlike background grids on data plots). User: "el real sky si hazlo, porque es del wcs".
- Re-added `ax.coords.grid(True, color="white", linestyle="dotted", alpha=0.5)` on real_sky (regen_real_sky.py); Lambda_MSR and all data plots remain grid-free.
- Verified: real_sky.pdf shows the dotted white RA/Dec grid (264.5-263.0 deg / -32.0 to -33.0 deg), members, 7 radius circles, 5 pc scalebar.
- Rebuilt: clean 20pp / marked 24pp / 0 undefined; zip 19 figs.

## 32. Figure-caption descriptiveness audit; restored Fig. 1 prose (2026-06-15)
- User: figures lost descriptive caption prose (gave Fig. 1 as example); restore the original descriptive wording where possible, keep facts updated, keep captions autonomous + always referenced, and do not revert anything the referee or Pierluigi requested.
- Audited all 19 figure captions, current (clean) vs original (baseline), against the referee matrix and Pierluigi annotations.
- Fig. 1 (probabilities): the only caption genuinely gutted of descriptive prose without a referee/Pierluigi reason. Restored the original four-sentence description ("...is plotted against their G magnitudes... Each data point indicates a star... As shown in the figure, stars with fainter magnitudes tend to have lower astrometric fidelity. This becomes more evident at G>18 mag."), keeping only the required factual updates: pseudo-probability notation $\tilde p$, viridis (not "blue to red"), and BOTH the 0.6 (dashed) and 0.8 (dotted) thresholds with the referee-required semantics (0.6 lower reference, 0.8 probable/member boundary). Removed the duplicate "Fainter sources..." clause that had migrated into the body (Sect. methodology); fig:probabilities is still referenced in the figure-roadmap paragraph, so no orphaned reference.
- All other captions kept: they are already descriptive + autonomous + referenced, and either (a) were rewritten to match the regenerated figures (viridis, sigma iso-density/iso-probability contours, King outer radius instead of tidal, new CVD-safe symbol scheme, recomputed numbers/quartiles), so the diff's red is accuracy, not lost description; or (b) were deliberately trimmed by Pierluigi/referee and must NOT be reverted: parallax (split moved to legend box, numeric errors not repeated, Pierluigi p18/p19), luminosity function (lower apparent-mag panel removed, Pierluigi p11), mass_tseg (compact "Same as Fig. ...", Pierluigi p12), Table 1 (states uncertainty meaning, referee), Fig. A.3 plot_pair_trace (sigma defined as nuisance scatter, referee).
- Verified: all 19 figures referenced in the body at least once (autonomy + reference requirement satisfied).
- Rebuilt: clean 20pp / marked 24pp / 0 undefined refs; source zip 19 figs.

## 33. Fixed PM point on the frame (sticky edges); axis-scale consistency pass (2026-06-15)
- User: a point in the proper-motion figure (Fig. 3) sat almost on the frame; wanted an automatic fix (not hardcoded limits), and a coherence/consistency check of all figure scales. Consulted the matplotlib docs.
- ROOT CAUSE (per matplotlib docs): contour/imshow artists set *sticky edges*, which pin the axis limits to the data range and suppress the default 5% data margin, so the lowest member landed on the frame. Not a forced-limits problem.
- FIX (automatic, no hardcoded numbers): `ax.use_sticky_edges = False` + `ax.margins(0.05)` on the two scatter+contour figures, proper_motion (regen_prob_massbinary.py) and center (regen_center.py), so the fractional 5% margin applies and no point touches the frame. For the PM figure the contour grid was also extended to the data-or-3sigma extent (derived from the fitted Gaussian, sx/sy from the covariance) so the 3-sigma ellipse closes instead of clipping at the data range. center: the sticky/margins/autoscale_view call is placed before the scale-bar block so the bar positions against the padded limits.
- CONSISTENCY review of the remaining figures: scales are appropriate by design, real_sky imshow fills the frame (intended sky chart), histograms (parallax, luminosity function, Sagitta) are anchored at 0, cumulative-count figures are bounded 0-1, and the Lambda_MSR errorbar already has padded ylim. None has data points on the frame.
- Verified visually: PM point now has clear margin below and all three sigma ellipses close; center has uniform margin with the scale bar intact.
- Rebuilt: clean 20pp / marked 24pp / 0 undefined refs; source zip 19 figs.

## 34. Parallax band in legend + reaches the top; legend-completeness sweep (2026-06-15)
- User: in the parallax figure (Fig. 4) the gray vertical band was not in the legend and did not reach the top of the panel; wanted everything plotted to appear in the legend, no forced limits. Consulted matplotlib docs; checked the other figures for the same.
- ROOT CAUSE: the band was drawn with `fill_betweenx([0, 1.5*get_ylim()[1]], ...)` and then `set_ylim(0, get_ylim()[1])` re-read the already-expanded ylim, so the band stopped short of the top; and it carried no `label`.
- FIX (documented, non-forced): replaced all three bands with `ax.axvspan(...)`, which spans the full axes height (y in axes-fraction coords) regardless of ylim, so it always reaches the top, and added a `label` to each: left = "$\mu_\varpi$ standard error", middle = "$1\sigma$ parallax dispersion", right = "$\mu_d$ dispersion". Also labeled the previously unlabeled middle-panel elements: the $\mu_\varpi$ dash-dot line and the gray "Parallax uncertainty" error bars.
- LEGEND-COMPLETENESS sweep of the other figures (everything plotted must be in the legend): king, Lambda_MSR, brightness-cumulative, Fig. 1 already label all their reference lines/bands. center (Fig. 2) had NO legend -> added one with "Cluster center" (crosshair) + "KDE $1\sigma,2\sigma$ iso-density" (contour proxy). proper_motion (Fig. 3) -> added "Center" to the crosshair and a "$1\sigma,2\sigma,3\sigma$ Gaussian contours" proxy to the existing legend. Used Line2D proxies for the contour sets.
- Verified visually: all three parallax bands now in the legend and full-height; center and PM legends list every plotted element.
- Rebuilt: clean 20pp / marked 24pp / 0 undefined refs; source zip 19 figs.

## 35. Appendix table: per-radius derived parameters (search-window sensitivity) (2026-06-15)
- User: the full per-cone-radius results were not consolidated in the paper; add an appendix table. (The full per-radius downstream recompute, ages/masses/segregation via MCMC, was considered but deferred to a future paper.)
- Added Appendix \ref{app:window} "Search-window sensitivity of the derived parameters" with Table \ref{tab:cone_params}: per radius (40/50/60/70 arcmin), N(p>=0.6), mean parallax +- 1sigma dispersion (frac-err<0.1 subsample), mu_alpha*/mu_delta +- member dispersions, King R_c, R_t (outer), and C=log10(R_t/R_c). 40 = adopted production fit; 50/60/70 = King modpriors fits, consistent with the §results prose (50': R_t=44.5+-12.8, C=1.35; 70': R_t=52.2+-8.8). Astrometry recomputed from the per-radius reference_p06.ecsv and matches the prose exactly (40': 0.908+-0.046, (2.542,-1.713)+-(0.152,0.138); 50': 0.906+-0.044, etc.). King means/std from idata_king_cone{50,60,70}_modpriors.nc.
- Cross-referenced from §results: the cone-robustness paragraph now points to Table \ref{tab:cone_params} (Appendix \ref{app:window}).
- New content -> appears as additions in the marked diff (baseline untouched).
- Rebuilt: clean 21pp / marked 24pp / 0 undefined refs; source zip 19 figs.
- DEFERRED (future paper): full per-radius recompute of ages, masses, mass segregation, LF, PMS/Sagitta. Heavy MCMC (isochrone fit ~2-4h/radius); per-radius reference samples have the needed photometry; only membership/astrometry/King were recomputed for this revision.

## 36. Full page-by-page coherence review of the clean PDF (2026-06-15)
- User: review the whole PDF page by page / figure by figure for coherence and errors, and confirm the new appendix shows in the text.
- Read all 21 pages. Findings:
  - Appendix D (search-window sensitivity) and Table D.1 render correctly and are referenced in the text. Added a second in-text reference in the Conclusions ("...not invariant with field size (Table \ref{tab:cone_params}, Appendix \ref{app:window})") to reinforce visibility; the Results robustness paragraph already pointed to it.
  - All appendices are referenced in text: A (Table A.1, intro), B (B.1/B.2/B.3), C (C.1-C.4), D (Table D.1).
  - Checked a suspected symbol clash: the Cartwright & Whitworth structure parameter and the NIR reddening-free YSO index. NOT an error, the source already disambiguates: structure parameter is calligraphic $\mathcal{Q}$ (Table 1 and Sect. 3.4) with an explicit "not to be confused with the reddening-free index $Q$" note; the YSO index is plain $Q$ (Eq. 4). (The small PDF render made both look like "Q".)
  - Cross-checked numbers for internal consistency: age 3.53 Myr = 0.12 t_rh (t_rh=30.5); f_YSO 0.275 = 53/193 with 95% CI matching Table 1; t_seg=2.94 = <m>/m_max * t_rh; mass_tseg cutoff M<11.2 consistent with t_seg>3.53 Myr (=age) and the observed max 8.49; King outer radius 40.4 and core 1.95 consistent across abstract/Table 1/Sect. 3.1/conclusions/figures; Hill 28.3 and bound 42.45 consistent with the real_sky legend. No inconsistencies found.
  - Figures verified: restored descriptive captions (Fig. 1), fixed legends/scales (center, PM, parallax), WCS grid on real_sky, all 19 figures present and referenced.
- Rebuilt: clean 21pp / marked 24pp / 0 undefined refs (the lone log "undefined" is a T1/txss font-shape substitution, not a reference); source zip 19 figs.

## 37. Define delta-varpi/varpi; A&A/Bayesian research; figure font harmonization (2026-06-15)
- User: is delta-varpi/varpi<0.1 explained in text? review dispersion/error usage; research the Bayesian "golden standard" in astronomy; research A&A figure font-size recommendations (labels/titles/ticks) and make figures not look bad.
- TEXT: defined the symbol where the cut is applied (Sect. 2.1.2): "a fractional parallax error below 0.1, i.e. delta-varpi/varpi<0.1 where delta-varpi is the Gaia parallax uncertainty". The Fig. 4 legend symbol is now backed by a prose definition. Added a 2nd in-text reference to Appendix D in the Conclusions.
- DISPERSION/ERROR: already handled consistently, center text distinguishes member dispersion vs standard error on the mean; parallax gives 1sigma dispersion 0.046 + SE 0.004; Table 1 caption defines the convention. No change needed.
- BAYESIAN STANDARD (researched): de-facto astronomy practice = posterior median + 16th/84th percentile (68% central CI), or mode/MAP + credible interval for skewed posteriors; HPD for asymmetric; R-hat/ESS convergence (Vehtari 2021). The paper already follows this. "GS 26"/"golden standard 26" is NOT a real named standard (reconfirmed).
- A&A FIGURE GUIDANCE (researched, aanda.org author guide): figures reduced to 88 mm (1-col); "use lower case for any words in figures"; symbols explained in caption not figure; no explicit pt size (MNRAS/AAS ~ readable >=8pt at final size). User chose to KEEP Title Case (loosely enforced in practice) but to harmonize sizes.
- FONT HARMONIZATION: added a shared rcParams block (xtick/ytick.labelsize=13, axes.labelsize=15, legend.fontsize=11, axes.titlesize=14) to the 9 figure-producing scripts (regen_parallax, regen_king, regen_cmd_figs, regen_prob_massbinary, regen_center, regen_brightness_cumulative, regen_rv, finalize_v069, regen_real_sky). Main fix: tick labels were matplotlib default (~10pt, tiny after \resizebox); now 13pt and uniform. Fixed outliers: parallax axis labels 18->15; all tiny legends (9/10/10.5pt) -> rcParams 11. (regen_mass_seg is a dev script writing to regen/, not the published figures, skipped.)
- NOT harmonized: luminosity_function (Fig. 9), pms_stats (Fig. C.2), and the corner plot plot_pair_trace (Fig. B.3) are generated by the notebook (Figures_NGC6383.ipynb), not by regen scripts, so they retain their original fonts. Flagged to user; would need standalone regen scripts or a notebook re-run for full consistency.
- Regenerated all 9 script figures; build.sh copied clean_source/Figures -> marked_changes/Figures (md5-verified identical, so both PDFs carry the updated images). Page reflow from figure-bbox changes: clean 21->20pp, marked 24->23pp; content intact (Table D.1 present, 19 includegraphics, 0 undefined refs).

## 38. All figures now script-generated (notebook fully retired) (2026-06-15)
- User: wants full figure consistency and nothing generated from the notebook anymore; for the HDBSCAN figures, use what HDBSCAN delivers via its methods.
- Wrote standalone regen scripts for the 5 figures that were still notebook-only, all with the shared rcParams styling:
  - regen_lf.py -> luminosity_function.pdf (Fig. 9): G_abs = Gmag - DM histogram (orange hatched), reference sample.
  - regen_pms_stats.py -> pms_stats.pdf (Fig. C.2): 3-panel Sagitta histograms (all/PMS/non-PMS/no-2MASS), legend in middle panel.
  - regen_corner.py -> plot_pair_trace.pdf (Fig. B.3): 5x5 corner of the isochrone posterior (Av,dm,loga,met,sigma); blue marginal KDE + black mode line on the diagonal, viridis hexbin + black crosshair + black square (joint mode) off-diagonal. (Fixed a bug where MaxNLocator re-added density tick labels on the diagonals.)
  - regen_hdbscan_diag.py -> min_cluster_size.pdf (Fig. B.1) and condensed_cluster_tree_NGC6383.pdf (Fig. B.2).
- HDBSCAN figures use the clusterer's own delivered data/methods (per user):
  - B.2 = HDBSCAN's native condensed_tree_.plot() (via cosmic plot_condensed_tree, which delegates bars/lines/colorbar to hdbscan and overlays the selected-cluster ellipses from condensed_tree_._select_clusters()).
  - B.1 = built from clu.pseudoprobability_results_ (per-candidate min_cluster_size, branch size, and lambda = condensed_tree_.lambda_val.max from HDBSCAN) with the caption's stability filter (lambda_max>=8, branch 200-701) and the clusterer's own selected mcs (clu.pseudoprobability_selected_, =43). Replaced the earlier version that read an external audit CSV. Reproduces the published figure exactly (mcs 16-64, peak 701 at 43).
- All 21 paper figure PDFs are now produced by review_repo/*.py scripts; the notebook (Figures_NGC6383.ipynb) is no longer needed to build any figure. Verified each regenerated figure matches the original.
- Rebuilt: clean 20pp / marked 23pp / 0 undefined refs; source zip 19 figs; clean<->marked figures md5-synced.

## 39. Appendix integration check + new HDBSCAN panorama figure (2026-06-15)
- User: confirm the appendix figures are properly threaded into the text (not just dangling extras), and add a new figure showing the full proper-motion panorama, the selected cluster, the other HDBSCAN branches, and the noise, in both PM and RA/Dec.
- INTEGRATION CHECK: all appendix figures are referenced 3-4 times from the body and each appendix has an intro paragraph that threads them (App B intro Figs B.1/B.2/B.3; App C intro C.1-C.4; App D intro Table D.1). They are woven in, not extras. No change needed.
- NEW FIGURE pm_radec_overview.pdf (regen_pm_overview.py): the full 40 arcmin preprocessed sample (15276 sources) colored by the clusterer's own labels_, selected NGC 6383 branch (red, 701), other HDBSCAN branches (blue, 704), field/noise (gray, 13871). Left panel = proper-motion plane (axes clipped to 0.5-99.5 percentile so the cluster knot is visible); right panel = on-sky RA/Dec. Shows NGC 6383 as a compact, isolated PM over-density at (2.5,-1.7) clearly separated from the diffuse field and the other branches, while on the sky all groups overlap, i.e. PM, not position, isolates the cluster. Uses the clusterer object's delivered data/labels (clu.data + clu.clusterer.labels_), consistent with the HDBSCAN-method approach.
- INTEGRATED into Appendix B as Fig. \ref{fig:pm_overview} (figure*, between the condensed tree and the corner): new descriptive paragraph + caption; referenced from the App B intro, the App B body, and from the main membership text in Sect. \ref{sec:results} (the branch-selection sentence). 4 refs total; 0 undefined.
- Rebuilt: clean 20pp / marked 24pp / 0 undefined refs; source zip now 20 figures; clean<->marked md5-synced.

## 40. Meticulous original-vs-current deletion audit (2026-06-15)
- User: review everything removed from the original and not carried into the new version, any lost detail/nuance?
- Method: sentence-level diff of the pristine original (6383_old_paper/aanda.tex, Sep 2024) vs current clean_source/aanda.tex (difflib, best-match ratio < 0.55 flagged). 364 original sentences, 155 flagged as removed/heavily-changed.
- Categorized all 155. Nearly all are INTENTIONAL or PRESERVED-REWORDED:
  - Intentional cuts: NUTS/PyMC pedagogical block (U-turn mechanics etc.), software version numbers (Python 3.12, PyMC 5), AI-tell closers ("Overall, this study demonstrates..."), figure captions rewritten for the regenerated figures, "tidal radius"->"King outer radius", Table 1 content-tour caption -> uncertainty-convention caption.
  - Deliberately-removed citations (regime-check, verified absent): Dib 2007 (2007MNRAS.381L..40D), Marks 2008 (2008MNRAS.386.2047M), phan2019 (NumPyro miscite).
  - Preserved but reworded (verified present): cross-match details (tmass_psc_xsc_best_neighbour, 5333 sources, <=0.3 arcsec, left-join keeping 15276), Haversine-tested-nonphysical, pseudo-probability definition (f_i x p_HDBSCAN), ASteCA binary-fraction alpha=0.090/beta=0.940 + D&K mass ratio + DR=0, the five catalog-comparison count triplets (Cantat-Gaudin/Jaehnig/He/Hunt/SIMBAD), HD 159176 astrometry (now with the correct larger bright-source errors + ~6sigma), age independent of HD 159176, Kalari double-peaked-Gaussian method note, geometric-vs-photogeometric distinction. Segregation cites all kept (McMillan 2007, Chen 2007, Sabbi 2008, Baumgardt & Kroupa 2007).
- ONE genuine lost detail found and RESTORED: the companion clusters NGC 6530 and NGC 6531 (original noted NGC 6383 belongs to the Sgr OB1 association together with them; current named only Sgr OB1). Restored a concise phrase in the intro ("...together with the open clusters NGC 6530 and NGC 6531"), which also connects to the later Chen 2007 NGC 6530 same-association comparison. (Also: original's "Sirius OB1" was an error -> current correctly says "Sgr OB1"; kept.)
- The hand-wavy "number of stars presumably larger at birth -> too little time for dynamical segregation" nuance is superseded by the current quantitative age/t_rh = 0.12 argument (improvement, not a loss).
- Rebuilt: clean 20pp / marked 24pp / 0 undefined refs; source zip 20 figs.

## 41. Final coherence/cohesion/contradiction review (multi-agent, adversarially verified) (2026-06-15)
- User: final coherence/cohesion/contradiction sweep, make sure nothing was missed.
- Ran a 7-dimension multi-agent review (numbers, cross-refs, terminology, captions-vs-figures, scientific logic/contradictions, prose cohesion, new-content integration); 53 agents, 45 raw findings, each adversarially verified against the tex; 12 confirmed, deduped to 7 actionable. NO critical or major issues; no live contradictions (the "primordial or near-primordial" vs "primordial or early-dynamical" wording is explicitly defined as synonymous at lines 35/366; the HD 159176 intro/body overlap is a legitimate preview, no value conflict).
- Fixes applied (minor/nit; no values changed): (1) Table 1 caption now states Y_frac uncertainties are 95% credible-interval bounds (not 1sigma); (2) in-context figure callouts added where results are described, Fig. probabilities (L262), center + proper_motion (L284), parallax_distance (L286), roadmap at L221 kept; (3) "in RA/in DEC" -> "in R.A./in Dec." (L284, matches Table 1); (4) lone non-house PM symbol/units fixed (L381 NGC 6383 22: $\mu_\alpha^*$->$\mu_{\alpha*}$, text-mode mas yr -> $\mathrm{mas\,yr^{-1}}$); (5) real_sky caption R_t "dashed"->"dash-dotted" (matches figure, removes clash with the red dashed cone circle); (6) Appendix A historical-table caption notes the literature "tidal radius" = the King outer radius R_t (column header preserved as original authors' nomenclature); (7) unit normalizations: bare "Myr"->$\mathrm{Myr}$ (L307), split km/s -> $\mathrm{km\,s^{-1}}$ (L288).
- Verified every fix is present in the file (no agent hallucinated-apply), no stray edits to figures/scripts, sentences intact.
- Rebuilt: clean 20pp / marked 24pp / 0 undefined refs (lone log "undefined" = T1/txss font-shape substitution) / 0 LaTeX errors; source zip 20 figs.

## 42. Round-2 referee revision (2026-07-16)

Second referee report received 2026-07-16 (17 points R1-R17 + structural
complaint; archived in `../referee_round2/referee_report_round2.md`).
Pre-round-2 state backed up in `../referee_round2/backup/`.

**Restructure (general point + R16):** manuscript reorganized thematically
(Tarricq+22 / Damiani+19 style): Data / Analysis framework and membership /
Astrometric parameters / Cluster structure / Age+stellar content / Luminosity,
mass, dynamical state / Comparison / Conclusions. Method -> NGC 6383 result ->
figure per theme; all floats moved next to their discussing text; figure-
enumeration sentence deleted; 3.5.3 reordered (Kalari finished before Rauw).
Intro roadmap paragraph rewritten; internal refs retargeted
(sec:results/sec:methodology eliminated).

**R11 (adopted Option 1 hybrid):** headline King outer radius now from the
70-arcmin background-constrained refit: R_t = 54(+7/-11) arcmin =
17.4(+2.3/-3.6) pc (P(R_t>window)=0% only at 70'); R_c = 1.96(+0.19/-0.16)
retained from the 40' reference sample (60-70' R_c contamination-biased;
50' run confirms 1.90(+0.15/-0.13)); C = 1.43(+0.07/-0.10) from combining the
adopted posteriors. Abstract, Table 1, Sect. 5, conclusions, App D updated;
Table D.1 converted to medians +16/84 + new P(R_t>W) column; new App D figure
king_profile_cone70.pdf (regen_king70.py); real_sky Rt circle -> 54';
cumulative-brightness figure lines -> Rc 1.96 + Hill 28.3 (regen scripts
updated). All numbers verified directly from idata_king*.nc (arviz, medians
+16/84).

**Point fixes R1-R17:** R1 five-step pipeline paragraph w/ per-tool purpose,
citation, URL footnote. R2 f_i corrected to implementation (fraction of sweep
runs assigned to ANY cluster; verified vs production dill: probability =
probability_times x probabilities_, diff 0.0) + p_HDBSCAN defined as
lambda_i/lambda_max. R3 lambda's three limited roles stated; App B display-cut
wording fixed. R4 N/U/HN notation + bar-varpi defined at first use.
R5 Table-1 distance rows renamed + caption provenance. R6 plain-language
diagnostics gloss + Betancourt 2016 added. R7 Tmax prior rationale +
0.9/1.2 sensitivity refit (<0.5 sigma except R_t within 1 sigma); alpha/beta
= ASteCA fit to Offner+23 multiplicity compilation (Offner+23, Moe & Di
Stefano 17 added). R8 R_gc law-of-cosines formula + R_sun=8.3 kpc (de Grijs &
Bono 16 added). R9 MIST v1.2 assumptions paragraph (Asplund+09 scale,
alpha_MLT=1.82, f_ov, vvcrit=0.4 at ZAMS; magnetic/SPOTS bias direction;
Feiden 16 + Somers+20 added). R10 nuisance sigma -> Fig fig:plot_pair_trace.
R12 KS passage rewritten (D/p values re-verified from ECSV, exact
reproduction). R13 ASteCA per-star m1/m2 mechanism + 51 stars q=0.26-0.73
median 0.60 (verified from masses_asteca_069.csv). R14 t_seg(m) equation +
m_lim=11.2 Msun logic rewritten + HD 159176 caveat (Penny+16 masses ->
t_seg 0.5-1.0 Myr, factor 3-6). R15 Be-binary strawman removed; O7V+O7V +
wind-wind (De Becker+04 added). R16 Kalari/Rauw reordered + age provenance
(PARSEC/Tognelli/Siess spread; Meynet & Maeder tracks; 3 bib added).
R17 added-value paragraph (posteriors, sensitivity budget, cluster-specific
analyses, CDS+pipeline; framed as complementarity).

**Build:** latexdiff baseline swapped to the round-1 submitted version
(referee sees only round-2 changes); latexdiff needs
`--config "PICTUREENV=(?:picture|DIFnomarkup|table\*?|tabular)"` after the
restructure (moved tables otherwise break with Misplaced \noalign).
Clean 23 pp / marked 26 pp / 0 errors / 0 undefined refs.
Response letter: `../referee_round2/response_letter.md` (with old->new
section map).

## 43. Post-round-2 adversarial re-review fixes (2026-07-17)

Multi-agent re-review (6 auditors x adversarial verify; 86 raw -> 67 confirmed).
All confirmed findings applied:

**Substantive:** (1) R_t prior-truncation disclosed, new relaxed-bound refit
(idata_king_cone70_relaxedRt.nc, Rt<70': 58+9/-13 vs 54+7/-11, <1 sigma;
R-hat 1.0014, ESS>3300): Sect. 5 + App D now state all R_t intervals are
conditional on the 1.5*T_max=63.7' prior; circular "only window that fully
encloses its posterior" / unconditional "0%" claims reworded (P(Rt>W) at 70'
vanishes by construction). (2) dm uncertainty was Av's sigma copied by mistake:
10.30+-0.26 -> 10.30+-0.09 (production trace 1723057171 sd=0.091); distance
(from D.M.) 1.15+0.15-0.13 -> 1.15+-0.05. (3) New Sect. 8.4 paragraph comparing
adopted R_t=54' with literature tidal radii 15-30' (window/background-treatment
explanation). (4) Conclusions Zhang+2026 clause fixed (was miscited as "largely
dynamical"; now matches Sect. 7 regime). (5) Table 1 caption discloses k,b,Rc
from 40' fit vs Rt from 70' fit; App D quotes the 70' fit's own k=8.42+-0.17,
b=0.020+0.006-0.010 + "residual contamination level" clarification.

**Consistency:** PyMC footnote restored (round-2 edit had deleted it while the
letter claimed it existed); k/b/bound-radius rows -> medians+16/84 (4.92+0.43
-0.31, 0.011+-0.007, 42.5+-1.6); prior-sensitivity sentence -> medians; total PM
3.070 -> posterior median 3.065+-0.009 (v_t 16.1); R_gc +-0.07 -> +-0.06;
lambda->gamma in t_rh (Coulomb constant; collision with HDBSCAN lambda);
B(a,b) Beta defined; App B "excluded as candidate configurations" contradiction
removed; m_lim "(unrounded values)" + HD159176 shift to 12.7 Msun added;
39-sources sentence acknowledges 40' truncation + Hill-vs-bound criteria;
Sagitta headline count added (116/254, 133/321 PMS>=0.6); Kalari VPHAS+ named;
Baume 1991 moved to ~20 Myr group; fig:mass_seg/fig:mst swapped to citation
order; App C list reordered; Hill notation unified (R_Hill); dangling
"as noted above" + self-refs + missing forward refs fixed; HDBSCAN
false-positive statement deduplicated; 97.5th percentile 2.4->2.5.

**Figures:** real_sky regenerated (Rc 1.96, R_Hill label, scale bar at adopted
1.11 kpc, docstring updated).

**Package:** response letter fixed (Table D.2, Eq. (7), O7V+O7V, uniqueness
claim, truncation + literature-comparison additions, round-1-numbering note);
MANIFEST.md updated to round-2; stale round-1 aa52082-24_marked_changes.* purged
from marked_changes/; top-level marked PDF re-copied from the current build;
zip rebuilt (27 files). Final: clean 23 pp / marked 27 pp / 0 errors /
0 undefined refs / 0 "??" in marked PDF.

## 44. Readability polish pass (2026-07-17)

Referee-scoped polish (4 lenses x per-edit judge; 47 approved, 36 applied,
11 skipped as overlaps already covered). Cohesion: membership counts now
precede the cone-robustness discussion; the Sect. 3.3 mega-paragraph split
into 3 topical paragraphs; section openings connected. Compactness: ~330
words saved, repeated caveats (prior truncation, model-dependent scale,
contamination) kept in full at one load-bearing site each, echoes compressed
to clause + cross-ref; convergence-diagnostics gloss folded inline into the
criteria sentence. Sentence craft: pipeline step (ii), pseudo-probability
definition, t_seg/m_lim passage, and the Sect. 8.4 added-value enumeration
split/shortened without content loss. Referee-error audit (lens 4): 5 genuine
referee misreadings identified (mathcal-U as half-normal; D.M. row as
Bailer-Jones; overshooting as the relevant PMS systematic; t_seg read as a
single number; wider-extraction premise ignoring contamination), ALL already
handled diplomatically in the letter, no new push-backs needed; R8 formula and
R10 figure guess verified as referee-correct. Key content verified preserved
(all headline numbers, caveats, citations). Final: clean 23 pp / marked 27 pp /
0 errors / 0 undefined refs / 0 "??"; zip rebuilt.

## 45. Literature verification of all new/referee-questioned scientific claims (2026-07-17)

6-cluster verification vs primary sources (48 claims: 35 VERIFIED, 10 IMPRECISE,
3 WRONG, all corrected).

**WRONG (inherited from round 1, self-corrected + disclosed in the letter):**
(1) t_rh coefficient 0.17 -> 0.138 (Spitzer & Hart 1971 added): t_rh 30.5+-9.4
-> 24.7+-7.6 Myr; t_seg 2.94+-1.17 -> 2.38+-1.24; age/t_rh 0.12 -> 0.14;
m_lim 11.2 -> 9.1 Msun (subsample UNCHANGED: still 252/254, 0.19-8.49);
HD counterfactual ~0.7/0.4 Myr, factor 3-7; multi-mass gamma~=0.02 caveat added
(GH96, conservative direction). (2) M_gc = 2e8(R/30pc)^1.2 was a
Galactic-CENTRE relation extrapolated to 7.2 kpc -> replaced with McMillan 2017
potential: M(<7.19 kpc)=(8.8+-0.5)e10; Hill 28.3+-1.0 -> 33.6+-2.2 arcmin
(10.8+-0.4 pc); sources beyond Hill 39 -> 18 (ptilde 0.64-1.0, mean 0.81);
T_max = r_bound = 42.5 unchanged -> King priors and R_t UNAFFECTED; cumulative +
real_sky figures regenerated with the new Hill line/circle. (3) Piskunov+08
tidal radius 29.0+-6.6 mixed value/error pairing -> 29.0+-4.2 (rt=8.3+-1.2 pc).

**IMPRECISE (10 wording fixes):** Feiden/Somers factor attribution (+Somers &
Pinsonneault 2015 added; "~2 or more"); binarity list starts at M dwarfs (b(0.1)
=0.15 vs observed 20+-4%); HD 159176 types per source (O7V((f))+O7V((f)) S93/L07
vs O6.5V+O7V P16); HD age 2.3-2.7 (Rauw+10) + 2.8+-0.5 (FitzGerald+78); Kalari
"dereddened CMD" -> reddened-model interpolation with E(B-V)=0.32; CTTS (not
disks) concentrated around NGC 6383; E-BFMI 0.3 threshold -> Betancourt 2017
(added); equal-mass gamma labeled; Kharchenko radius = empirical corona radius
(Table A.1 caption note); Sect 8.4 "compatible at ~2 sigma" + survey-depth
qualifier. New bib: Spitzer&Hart71, McMillan17, Betancourt17, Somers&
Pinsonneault15, Giersz&Heggie96. Letter: R14 numbers updated + new
"Corrections made on our own initiative" section disclosing (1)-(2).
Final: clean 23 pp / marked 27 pp / 0 errors / 0 undefined refs; zip rebuilt.

## 46. Final pre-submission gate (2026-07-17)

Fresh-eyes gate (3 readers: full PDF, letters, package). 4 blockers + 19 nits,
all fixed: Amiri+26 bib entry now renders its arXiv ID (pages/eprint added);
author initials normalized (L. M. / N. A. / R. E.) incl. authorrunning;
MANIFEST repointed to the round-2 cover letter + round-2 response (round-1
letters archived in letters/round1_archive/, nonexistent Cerulo-response row
dropped, 27 pp + 21-figure counts corrected); duplicate Spitzer&Hart bib entry
removed. Nits: comma splices, Zhang single-author verb agreement, scikit-learn
spelling, footnote periods, letter fixes (three-errors count, Kalari
median/mean, NUTS-based qualifier, R.A.-dispersion qualifier, Duchêne accent),
cover-letter wording softened. real_sky: R_t=54' circle removed (outside the
DSS2 frame) with caption pointing to Fig. D.1; R_bound label 42.5. New round-2
cover letter: letters/cover_letter_round2.txt. Final: clean 23 pp / marked
27 pp / 0 errors / 0 undefined refs; zip + deliverables rebuilt.

## 47. comments_paper/ cleanup + marked-diff moved-float fix (2026-07-17)

**Moved-float diff fix:** latexdiff has no move detection (upstream issue #162),
so the 10 relocated floats showed orphan struck-through captions without images
at their old positions. Added strip_moved_floats.py to build.sh: removes DIFdel
blocks that are whole floats ONLY when the same graphic/label is alive elsewhere
(true deletions would remain visible). 9 blocks removed; marked 27 -> 25 pp;
21 unique figures live; 0 "??".

**Cleanup audit (4-agent, 106 items classified: 55 round-2 active / 23 round-1
record / 26 legacy):** legacy moved (not deleted) to comments_paper/_legacy/
(~336 MB): May-2026 rerun trace 1777967244 (226 MB, never adopted; production =
1723057171) + its sidecars; members.csv (renamed *_DIFFERENT_RUN_DO_NOT_USE,
the documented 177-source float-corrupted trap); retired notebooks; AI-handoff
docs; superseded top-level cds/ + cds.zip (authoritative: cds_final/ +
aa52082-24_cds_members.zip); marked_changes round-1 litter (aanda_revised.tex,
25 ARCMIN/, 7 unused figure PDFs, dead symlink); regen_mass_seg.py + regen/;
per-radius preprocessed_{good,bad}.ecsv (re-derivable; dill embeds data).
TRAP AVOIDED: cluster_data.ecsv looks legacy but is the LIVE Sagitta input of
ngc6383_generate_cds_table.py; kept and documented. Top-level README.md
rewritten (was stale/misleading about members.csv). Verified after cleanup:
full build (23/25 pp, 0 errors), regen_king70.py, CDS generator inputs intact.
