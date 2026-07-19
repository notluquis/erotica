# Response to Referee 1

Article reference: aa52082-24

Title: Characterizing NGC 6383: A study of fundamental properties, pre-main sequence stars, mass segregation, and age using Gaia DR3 and 2MASS

We thank the referee for the detailed report. Below we describe the changes made in the revised manuscript and the additional validation products generated for the revision.

## General response

The revised manuscript now separates the operational HDBSCAN pseudo-probability from a calibrated membership posterior, gives the exact clustering feature space and sweep settings, adds a cone-search robustness test, clarifies the uncertainty convention in Table 2, and treats HD 159176 conservatively. We also prepared a CDS-style likely-member table and cross-matched the candidate list with the UBVRI H-alpha catalog of Rauw et al. (2010).

## Point-by-point response

1. Cone-search radius and fitted boundary/tidal radius

Response: We repeated the full preprocessing, HDBSCAN sweep, pseudo-probability calculation, and parallax clipping using 40, 50, 60, and 70 arcmin input cones. The NGC 6383-like proper-motion branch is recovered in all runs, but the selected outer membership is not invariant with the field size. At 60 and 70 arcmin, the generic sweep-selected branch differs from the NGC-like branch, so we no longer present the fitted tidal radius as an independently sharp physical boundary. We also added recent Gaia-based open-cluster references to place this caveat in context: wide-field structural work estimates the background from external radial bins, while tidal-tail studies require photometric, dynamical, or 3D diagnostics before interpreting extra-tidal candidates as physical members rather than field contamination.

In addition, we now report the derived parameters for the 50 arcmin run, in which the automated sweep selects the same branch as the NGC-like criterion: mean parallax 0.906 +/- 0.044 mas (vs 0.908 +/- 0.046 adopted) and mean proper motions (2.544, -1.711) with dispersions (0.133, 0.127) mas/yr (vs (2.542, -1.713) +/- (0.152, 0.138)). We also refitted the King profile on the 50 arcmin reference sample with identical priors: R_c = 1.91 +/- 0.15 arcmin, outer radius 44.5 +/- 12.8 arcmin, C = 1.35 +/- 0.16 -- consistent with the adopted values, with the outer radius remaining weakly constrained even in the wider window. This shows the headline parameters are stable at the catalog level, not only at the branch level. We extended the same exercise to the 60 and 70 arcmin runs, where the sweep-selected branch diverges from the NGC-like branch and the p>=0.6 sample inflates to 443 and 628 sources: there the solution degrades coherently -- the pmRA dispersion grows from 0.152 to 0.239 and 0.279 mas/yr with a drifting mean (2.455, 2.383), and the refitted King profile follows the window (R_c drops to ~1.4 arcmin, outer radius grows to 52.2 +/- 8.8 arcmin). This is the expected signature of increasing field contamination, and it is now reported in the manuscript alongside the 50 arcmin stability check.

Manuscript change: Added Table `cone_robustness`, the 50 arcmin catalog-level comparison and King refit (Results, cone-robustness paragraph), and a conservative interpretation in the Results and Conclusions. The text now states that the fitted outer radius is weakly constrained by the spatial window and should be interpreted as a model-dependent outer scale. Added citations to Angelo et al. (2025), Risbud et al. (2025), and Xu et al. (2025).

2. Methods insufficiently described; parameters not defined; CDS table requested

Response: We expanded the membership methodology. The revised manuscript states that HDBSCAN was run only in the two-dimensional proper-motion plane, using the Euclidean metric. Parallax, sky position, and photometry are used for filtering and downstream characterization, not as HDBSCAN clustering coordinates. The grid search is now explicitly given as `min_cluster_size=min_samples=m_cl`, with `m_cl=10,...,299`, and the adopted setting is `m_cl=43` with `leaf` selection. We also define the lambda scale in the condensed tree and the composite pseudo-probability.

Manuscript/data-product change: Added the method details, inserted an excerpt of the likely-member catalog in the manuscript, and prepared a CDS-style catalog with 321 likely candidates and a flag (`Ref`) for the 254-source reference sample; the member subset follows from `pMember >= 0.8`. The CDS upload package is `aa52082-24_cds_members.zip`.

3. Sirius OB1

Response: Corrected. The manuscript now refers to the broader Sgr OB1 star-forming complex and does not mention Sirius OB1.

4. Statement that HD 159176 predates the cluster

Response: Removed the contradictory statement. The revised introduction now follows Rauw et al. (2010): the age of the low-mass PMS candidates is in reasonable agreement with the estimated age of HD 159176 if the binary is assumed to be associated with the same star-formation event.

5. Aidelman et al. (2018), X-ray Be binary, and blue-straggler wording

Response: Revised. We no longer adopt the X-ray Be binary classification. We describe the blue-straggler interpretation as a historical hypothesis and state that modern spectroscopic studies classify HD 159176 as an O-type main-sequence binary, including the Linder et al. (2007) study noted by the referee. We also state that the X-ray emission is naturally associated with massive-star winds and wind interaction.

6. Figure 1 threshold semantics

Response: Corrected. The figure now plots the composite membership pseudo-probability `tilde p`, with threshold markers at 0.6 and 0.8. The caption states that 0.6 is the lower limit of the reference sample and 0.8 is the boundary between probable members and members.

7. Empty Rauw et al. (2010) row in Table 1

Response: Corrected. Table 1 of the original version (the historical compilation, now Table A.1 in Appendix A of the revised manuscript) includes the Rauw et al. (2010) PMS-candidate count, reddening, and age information.

8. HDBSCAN criteria and weighting

Response: Clarified. No weighting among proper motion, parallax, angular distance, and photometry is applied inside HDBSCAN because only proper motions are used as clustering coordinates. The other quantities are used in preprocessing, parallax clipping, and later characterization.

9. Maximum lambda value

Response: Defined. The revised method section states that the HDBSCAN condensed tree is parameterized by the density level `lambda = 1/d_mreach`, where `d_mreach` is the mutual-reachability distance. The `lambda_val` quantity inspected in the sweep is described as a diagnostic of branch persistence and separation, not as a fitted physical cluster parameter.

10. Uniform prior notation

Response: Clarified. The revised text spells out the uniform priors and the notation used for the structural and ASteCA models.

11. Parameters in the structural model

Response: Clarified. The priors for the King-profile parameters now define the background density, central density normalization, core radius, King outer radius, surface-density extrema, and outer-radius scale used in the fit.

12. PMS model dependence

Response: Addressed as a limitation. The revised ASteCA section explicitly states that only MIST PMS evolutionary models were used and that the quoted age uncertainty does not include systematic differences from alternative PMS tracks.

13. Figure 10 versus Table 2; meaning of error bars

Response: Clarified. Table 2 of the original version (Table 1 in the revised manuscript) now reports observed member dispersions for parallax and proper motions rather than the smaller standard errors on the mean. The text gives both quantities where useful. The Sagitta figure is now described as individual-star PMS/extinction/age estimates, whereas Table 2 reports the cluster-level ASteCA posterior. This resolves the apparent mismatch in the broader Sagitta distributions.

14. Proper-motion significance

Response: Clarified. The revised text explains that the cluster branch is selected from the stable HDBSCAN condensed-tree branch centered on the known NGC 6383 proper-motion overdensity, with subsequent parallax clipping and comparison to the member dispersions.

15. Cluster center versus HD 159176

Response: Added. The revised Results and HD 159176 sections state that the inferred center is consistent with the projected position of HD 159176, but this positional coincidence is not sufficient to establish membership.

16. Rauw et al. (2010) UBVRI H-alpha crossmatch

Response: Added and qualified. We cross-matched the 321 likely candidates with the CDS UBVRI H-alpha table of Rauw et al. (2010) using a 1 arcsec radius. We found 141 matches, including 124 from the reference sample. Since the CDS table provides photometry but not a published source-by-source H-alpha emitter flag, we do not claim to recover the original Rauw et al. emitter list. Instead, we applied the published Rauw et al. offsets relative to an empirical non-emitter locus as a diagnostic. This gives 51 matched sources with diagnostic H-alpha excess, including 28 of 57 matched Sagitta PMS candidates and 23 of 51 matched Sagitta PMS candidates in the reference sample. The revised manuscript avoids claiming that this "confirms" H-alpha incompleteness; it states only that H-alpha non-excess is non-decisive for PMS status.

Manuscript/data-product change: Added a paragraph in the PMS comparison section and retain the Rauw et al. cross-match summary, table, and diagnostic plot as supplementary validation material, available on request.

17. Gaia DR3 values and interpretation of HD 159176

Response: Corrected. The manuscript now uses the Gaia DR3 values quoted by the referee: parallax `1.167 +/- 0.071 mas`, proper motion in RA `2.621 +/- 0.083 mas/yr`, and proper motion in Dec `-0.798 +/- 0.058 mas/yr`. We explicitly discuss the magnitude contrast relative to our Gaia-selected reference sample (`G ~= 5.7` for HD 159176, compared with `G_min = 8.80` and median `G ~= 17.0` for the reference sample) and the possibility of bright-source astrometric systematics, citing the Gaia EDR3 astrometric solution, the bright-source proper-motion correction, and the Maiz Apellaniz (2022) parallax-bias analysis. We also added a Gaia-quality check: HD 159176 has normal `RUWE=0.937`, `visibility_periods_used=13`, and no duplicated-source flag, but it has very significant excess noise (`astrometric_excess_noise_sig=135.6`) and a large IPD multi-peak fraction (`ipd_frac_multi_peak=39`) compared with the reference-member distribution. With this caveat, the parallax and RA proper motion are not treated as decisive; the main discrepancy is the Dec proper motion relative to the selected cluster distribution.

18. Catalog-comparison section lacks interpretation

Response: Revised. The catalog-comparison section now explains that the numerical differences mainly reflect methodological choices, spatial windows, probability thresholds, and whether PMS or photometric information is included.

19. Conclusion novelty about HD-independent age

Response: Revised. We no longer claim this is the first HD-independent age estimate. The conclusion now says the reported age is derived from the selected PMS and low-mass population, as in previous PMS-based studies, and does not depend on assuming a particular evolutionary state for HD 159176.

20. Sigma in Fig. A.3

Response: Defined. The revised ASteCA method and the corner-plot caption (Fig. A.3 in the original numbering; Fig. B.3 in the revised manuscript) define `sigma` as the nuisance likelihood-scatter term used in the PyMC diagnostic implementation that generated the figure. It is not a physical velocity, spatial, or age dispersion of the cluster.

## Additional revisions in the current version

Beyond the point-by-point items above, the following changes were made in this version. They reinforce, and do not alter, the main conclusions.

A. **Mass segregation made quantitative and bin-independent.** To avoid relying on mass-bin-dependent KS p-values, we added the minimum-spanning-tree mass-segregation ratio Λ_MSR (Allison et al. 2009; new figure). Λ_MSR reaches ≈2.5–3.6 for the 5–15 most massive members (up to ~3σ above unity) and decreases toward unity, an unambiguous detection of mass segregation. The single-vs-binary KS test now gives D = 0.221, p = 0.010, so the segregation is formally significant and confirmed by three independent diagnostics (cumulative distributions, KS, Λ_MSR). This addresses the earlier concern that the significance depended on the chosen mass range (related to point 1). We also justify the choice of method: Λ_MSR needs no cluster center and, for smooth centrally concentrated clusters, agrees with the other standard diagnostics (Parker & Goodwin 2015). We verified that NGC 6383 is of this type — not substructured — via the Cartwright & Whitworth (2004) structure parameter, Q = 1.03 ± 0.03 (bootstrap; Q > 0.8 = centrally concentrated, consistent with the King concentration C = 1.32), so no substructure correction is needed. The Λ_MSR signal is unchanged using the median or geometric-mean variant that is robust against single massive outliers (Maschberger & Clarke 2011).

B. **Corrected half-mass radius and dynamical times.** We found that the half-mass radius had been converted to parsecs with an incorrect distance; the corrected value is r_hm = 2.02 ± 0.41 pc (was 1.65 pc). In the same pass we made the Spitzer relaxation-time inputs self-consistent: N and M now refer to the same stellar system (the 254-source reference sample, M = N⟨m⟩ = 332 ± 26 M_sun), instead of mixing the Gaia-selected N with an IMF-extrapolated literature total mass. This gives t_rh = 30.5 ± 9.4 Myr and a minimum segregation time of 2.94 ± 1.17 Myr. The cluster remains dynamically unrelaxed, consistent with a primordial origin for the central concentration of binaries.

C. **Error propagation.** The parsec-scale uncertainties of the core and half-light radii now include the distance term and, for the half-light radius, a bootstrap estimate (e.g. R_hl = 6.02 ± 1.07 arcmin = 1.94 ± 0.36 pc; R_c = 0.632 ± 0.070 pc), replacing the previously underestimated values.

D. **Masses re-derived with ASteCA v0.6.9.** The stellar masses and binary probabilities were recomputed with the current ASteCA release; results are consistent with the previous run.

E. **Figure organisation (related to point 13).** The per-star Sagitta histograms, the secondary mass-segregation figure, the radial-velocity-amplitude figure, and the spatial-distribution (`real_sky`) figure were moved to a new supplementary-figures appendix (Appendix C in the revised manuscript), and the historical-parameters table (Table 1 of the submitted version) was moved to Appendix A. The luminosity-function figure was reduced to a single absolute-magnitude panel (one source shifts bin after the distance-consistency pass of item H). The parallax figure legend now carries the δϖ/ϖ threshold definitions directly, shortening its caption. In the same pass, the membership figures (pseudo-probability, center determination, proper motion), the CMD and isochrone-grid figures, the King-profile figure, the radial-velocity figure, the mass/binary CMD, the two mass-cumulative figures, and the brightness-cumulative figure were regenerated with a uniform, perceptually ordered, colorblind-safe color scheme (viridis/Tol palettes with distinct line styles per curve) and the pseudo-probability notation of point 2; the center-determination panel now shows the full input field with its density contours and omits the radius overlays, which are shown in the supplementary spatial-distribution figure. No data values changed in these restyled figures beyond the re-derivations described in items B and D, with three documented exceptions: the radial-velocity amplitudes were re-queried from Gaia DR3 (rv_amplitude_robust; 29 sources, 8 with amplitudes), the radial-velocity subsample statistics were recomputed with the v0.6.9 binary probabilities (now median/mean/std = -1.87/-9.40/32.1 km/s for the same 16 sources), and the faint-quartile K-S statement now reports the three pairwise p-values explicitly (0.010, 0.052, 0.42).

F. **New citations.** Da Rio et al. (2010) (broadening of the pre-main-sequence color-magnitude diagram) and Allison et al. (2009) (MST mass-segregation ratio).

G. **Reproducibility of the YSO fraction.** The Y_frac denominator is now stated explicitly: of the 254 reference members, N_cl = 193 have 2MASS JHKs photometry (required to compute the reddening-free Q index). With 53 YSOs this gives Y_frac = 0.275 (95% CI [0.217, 0.342]), recomputed from the production catalog; the value remains well above the Buckner & Froebrich (2013) sample (no cluster with Y_frac > 0.200), so the conclusion is unchanged.

H. **Internal-consistency pass.** Unit formatting unified (\mathrm{} throughout), the most-massive-star mass made consistent (13.56 M⊙) across text and figures, the King outer radius parsec conversion corrected (13.05 pc at 1.11 kpc), and both the parallax- and CMD-derived distances are now named side by side in the CMD section (Sect. 3.2 of the revised manuscript), with the parallax value adopted for all radius conversions. Several figure captions were revised to describe the figure content without re-deriving the body text, the redundant appendix paragraphs that merely restated their figure captions were removed (every figure remains cited in the body), and the keyword list was corrected (the extragalactic "galaxies: star clusters: general" entry was replaced by "stars: pre-main sequence", and the object name added to the "individual:" keyword). A language pass removed residual filler and tightened the abstract, the opening paragraph of the introduction, and the conclusions. We also added one explicit methodological caveat in the Sagitta section, noting that the network was trained on Gaia DR2 photometry and is applied here to the corresponding Gaia DR3 bands.

I. **Primordial interpretation placed on a firmer footing** (the substructure check and method justification supporting this are described in item A). On the supporting side, the cluster age is only ≃0.12 t_rh, far below the ~2–3.5 t_rh over which primordial mass segregation is erased in N-body models with a realistic primordial binary population (Pavlík 2020), and two-body relaxation is too slow to segregate the intermediate-mass binaries within the cluster age. On the cautionary side, we now explicitly acknowledge that early dynamical segregation — via cool/substructured collapse (Allison et al. 2009b) or subcluster merging, demonstrated down to the low-membership regime of local open clusters (Moeckel & Bonnell 2009) — can reach the observed levels and, because it also smooths the cluster, is observationally degenerate with genuine birth segregation. We therefore state the result as "primordial or near-primordial", the latter encompassing rapid early dynamical segregation. We deliberately did **not** cite recent massive-cluster simulations (e.g. Polak et al. 2025, clouds 10⁴–10⁶ M⊙) for the timescale, since their mass regime is far above NGC 6383 (~900 M⊙) and the extrapolation would be unjustified.

J. **Further author-initiated refinements.** (i) The "projected velocity" previously quoted in mas/yr was an angular quantity; it is now reported as the mean total proper motion (3.07 mas/yr) with its tangential velocity (16.2 ± 0.9 km/s). (ii) The Galactic enclosed mass error was corrected to a physical value, M_gc = (1.43 ± 0.02) × 10^11 M⊙ (the previous ±9.7 × 10^7 understated the propagated R_GC term). (iii) The fitted outer radius is now called "King outer radius" consistently (avoiding the misleading "tidal radius" label for a weakly constrained scale). (iv) A note was added that the fitted metallicity (Z = 0.024, mildly super-solar) is only loosely constrained by the age–metallicity–extinction degeneracy. (v) The faint-end luminosity-function K-S result is explicitly attributed to completeness rather than dynamics.

K. **Uncertainties on every quantity.** We checked that each reported quantity carries an appropriately computed uncertainty, following standard conventions for Gaia open-cluster work: posterior credible intervals for the Bayesian (King, isochrone) parameters; 1σ member dispersions and standard errors on the mean (kept distinct) for the astrometry; bootstrap for the half-light and half-mass radii; full propagation of the distance term into all physical (pc, Myr, M⊙) conversions; and the Cameron (2011) Beta-binomial interval for Y_frac. Two quantities that previously lacked errors now carry them, using the ASteCA per-star mass uncertainties: the mean stellar mass ⟨m⟩ = 1.31 ± 0.10 M⊙ (standard error on the mean) and the most massive star, 13.56 ± 3.25 M⊙. Propagating these into the Spitzer segregation time gives a more realistic minimum t_seg = 2.94 ± 1.17 Myr (propagating only the relaxation-time term would give ±0.90); the conclusion is unchanged.

L. **Convergence audit of all Bayesian fits.** We re-ran the parallax, distance, proper-motion, and King-profile models with NUTS (four chains x 2000 draws after 2000 tuning steps): all satisfy the Vehtari et al. (2021) criteria (rank-normalized R-hat <= 1.005 for every parameter, bulk/tail effective sample sizes > 1200, zero divergent transitions, E-BFMI > 0.79) and reproduce the published values to within rounding. The audit also revealed that the isochrone-fit sampler had been misidentified in the text as NUTS; it is the gradient-free DEMetropolis ensemble sampler (300 chains x 1000 draws after 1500 tuning steps), the appropriate choice for the grid-interpolated isochrone likelihood. The text now states this correctly (citing ter Braak 2006) and reports an ensemble-stability check: the posterior-mode scatter under random half-ensemble bootstraps is 0.08 dex in log(age) and 0.03 mag in distance modulus, well below the quoted credible widths. All published values are unchanged. The same audit covers the cone-search King refits (40-70 arcmin): all four pass the criteria at target_accept = 0.99 with zero divergences.

M. **Faculty-style internal review pass (sensitivity and specification additions).** Following an internal referee-panel review of the revised manuscript, we added: (i) a threshold- and mass-sensitivity statement for the binary-segregation KS test (signal stable for binary-probability cuts 0.5-0.8, p <= 0.018; overlapping-mass-range test D=0.19/p=0.07 and nearest-mass-matched test D=0.36/p<0.001); (ii) a 2MASS-selection caveat and limiting bounds [0.21, 0.45] for Y_frac, with the Buckner & Froebrich comparison qualified as indicative (samples not selection-matched); (iii) the HD 159176 mu_delta tension restated as ~6 sigma with the star's own proper-motion uncertainty added in quadrature; (iv) a Holm-correction note for the three faint-quartile KS comparisons; (v) full reproducibility specifications: the exact cone-search center and in-query parallax window, the parallax/distance generative-model priors, the King-fit likelihood, the ASteCA v0.6.9 binary-fraction parameterization (verified against the ASteCA source: alpha and beta parameterize the mass-dependent binary fraction b(m)=alpha+beta/(1+1.4/m), not the IMF - a misattribution we corrected, with the Duchene & Kraus 2013 mass-ratio distribution), N=254 stated explicitly in the relaxation-time inputs, and the KDE bandwidth dominating the center uncertainty; (vi) corrected sampler citations (Hoffman & Gelman 2014 for NUTS; Abril-Pla et al. 2023 for PyMC; Duchene & Kraus 2013 for the mass-ratio distribution); (vii) an explicit contribution statement closing the Introduction; (viii) the CDS/SIMBAD acknowledgment; (ix) abstract/conclusion wording recalibrated to the body's evidence level (apparent age spread from isochrone bracketing, stated as an upper limit given binary/reddening broadening; segregation pattern consistent with a primordial or near-primordial origin given the cluster's dynamical youth). We additionally engaged the most recent literature on the central interpretation: a faculty-style claim-by-claim audit against arXiv/ADS confirmed that every cited result underpinning the mass-segregation argument applies to NGC 6383's regime (young, few-hundred-member open cluster; e.g. Moeckel & Bonnell 2009 and Pavlik 2020 both simulate this regime), and surfaced two 2026 studies arguing that young open-cluster segregation is largely dynamical (Zhang 2026, AJ; Amiri et al. 2026). We now cite both and explicitly state that our single-cluster detection does not settle the primordial-versus-dynamical question. No published values changed.

## Revision products prepared for upload

- Clean revised source package: `aa52082-24_source.zip`
- Clean compiled manuscript PDF: `aanda_revised_clean.pdf`
- Marked-change PDF: `aa52082-24_marked_changes.pdf`
- CDS likely-member table package: `aa52082-24_cds_members.zip`
- Co-author (P. Cerulo) response: `letters/response_to_cerulo.md`
