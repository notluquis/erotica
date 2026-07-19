# Response to Referee 1

Article reference: aa52082-24

Title: Characterizing NGC 6383: A study of pre-main sequence stars, mass segregation, and age using Gaia DR3 and 2MASS

We thank the referee for the detailed report. Below we describe the changes made in the revised manuscript and the additional validation products generated for the revision.

## General response

The revised manuscript now separates the operational HDBSCAN pseudo-probability from a calibrated membership posterior, gives the exact clustering feature space and sweep settings, adds a cone-search robustness test, clarifies the uncertainty convention in Table 2, and treats HD 159176 conservatively. We also prepared a CDS-style likely-member table and cross-matched the candidate list with the UBVRI H-alpha catalogue of Rauw et al. (2010).

## Point-by-point response

1. Cone-search radius and fitted boundary/tidal radius

Response: We repeated the full preprocessing, HDBSCAN sweep, pseudo-probability calculation, and parallax clipping using 40, 50, 60, and 70 arcmin input cones. The NGC 6383-like proper-motion branch is recovered in all runs, but the selected outer membership is not invariant with the field size. At 60 and 70 arcmin, the generic sweep-selected branch differs from the NGC-like branch, so we no longer present the fitted tidal radius as an independently sharp physical boundary. We also added recent Gaia-based open-cluster references to place this caveat in context: wide-field structural work estimates the background from external radial bins, while tidal-tail studies require photometric, dynamical, or 3D diagnostics before interpreting extra-tidal candidates as physical members rather than field contamination.

Manuscript change: Added Table `cone_robustness` and a conservative interpretation in the Results and Conclusions. The text now states that the tidal radius is weakly constrained by the spatial window and should be interpreted as a model-dependent outer scale. Added citations to Angelo et al. (2025), Risbud et al. (2025), and Xu et al. (2025).

2. Methods insufficiently described; parameters not defined; CDS table requested

Response: We expanded the membership methodology. The revised manuscript states that HDBSCAN was run only in the two-dimensional proper-motion plane, using the Euclidean metric. Parallax, sky position, and photometry are used for filtering and downstream characterization, not as HDBSCAN clustering coordinates. The grid search is now explicitly given as `min_cluster_size=min_samples=m_cl`, with `m_cl=10,...,299`, and the adopted setting is `m_cl=43` with `leaf` selection. We also define the lambda scale in the condensed tree and the composite pseudo-probability.

Manuscript/data-product change: Added the method details, inserted an excerpt of the likely-member catalogue in the manuscript, and prepared a CDS-style catalogue with 321 likely candidates and flags for the 254-source reference sample and the member subset. Files are in `comments_paper/cds_final/`, copied into the final `submission_package/cds/` directory, and packaged for upload as `aa52082-24_cds_members.zip`.

3. Sirius OB1

Response: Corrected. The manuscript now refers to the broader Sgr OB1 star-forming complex and does not mention Sirius OB1.

4. Statement that HD 159176 predates the cluster

Response: Removed the contradictory statement. The revised introduction now follows Rauw et al. (2010): the age of the low-mass PMS candidates is in reasonable agreement with the estimated age of HD 159176 if the binary is assumed to be associated with the same star-formation event.

5. Aidelman et al. (2018), X-ray Be binary, and blue-straggler wording

Response: Revised. We no longer adopt the X-ray Be binary classification. We describe the blue-straggler interpretation as a historical hypothesis and state that modern spectroscopic studies classify HD 159176 as an O-type main-sequence binary, including the Linder et al. (2007) study noted by the referee. We also state that the X-ray emission is naturally associated with massive-star winds and wind interaction.

6. Figure 1 threshold semantics

Response: Corrected. The figure now plots the composite membership pseudo-probability `tilde p`, with threshold markers at 0.6 and 0.8. The caption states that 0.6 is the lower limit of the reference sample and 0.8 is the boundary between probable members and members.

7. Empty Rauw et al. (2010) row in Table 1

Response: Corrected. Table 1 now includes the Rauw et al. (2010) PMS-candidate count, reddening, and age information.

8. HDBSCAN criteria and weighting

Response: Clarified. No weighting among proper motion, parallax, angular distance, and photometry is applied inside HDBSCAN because only proper motions are used as clustering coordinates. The other quantities are used in preprocessing, parallax clipping, and later characterization.

9. Maximum lambda value

Response: Defined. The revised method section states that the HDBSCAN condensed tree is parameterized by the density level `lambda = 1/d_mreach`, where `d_mreach` is the mutual-reachability distance. The `lambda_val` quantity inspected in the sweep is described as a diagnostic of branch persistence and separation, not as a fitted physical cluster parameter.

10. Uniform prior notation

Response: Clarified. The revised text spells out the uniform priors and the notation used for the structural and ASteCA models.

11. Parameters in the structural model

Response: Clarified. The priors for the King-profile parameters now define the background density, central density normalization, core radius, tidal radius, surface-density extrema, and outer-radius scale used in the fit.

12. PMS model dependence

Response: Addressed as a limitation. The revised ASteCA section explicitly states that only MIST PMS evolutionary models were used and that the quoted age uncertainty does not include systematic differences from alternative PMS tracks.

13. Figure 10 versus Table 2; meaning of error bars

Response: Clarified. Table 2 now reports observed member dispersions for parallax and proper motions rather than the smaller standard errors on the mean. The text gives both quantities where useful. The Sagitta figure is now described as individual-star PMS/extinction/age estimates, whereas Table 2 reports the cluster-level ASteCA posterior. This resolves the apparent mismatch in the broader Sagitta distributions.

14. Proper-motion significance

Response: Clarified. The revised text explains that the cluster branch is selected from the stable HDBSCAN condensed-tree branch centered on the known NGC 6383 proper-motion overdensity, with subsequent parallax clipping and comparison to the member dispersions.

15. Cluster center versus HD 159176

Response: Added. The revised Results and HD 159176 sections state that the inferred center is consistent with the projected position of HD 159176, but this positional coincidence is not sufficient to establish membership.

16. Rauw et al. (2010) UBVRI H-alpha crossmatch

Response: Added and qualified. We cross-matched the 321 likely candidates with the CDS UBVRI H-alpha table of Rauw et al. (2010) using a 1 arcsec radius. We found 141 matches, including 124 from the reference sample. Since the CDS table provides photometry but not a published source-by-source H-alpha emitter flag, we do not claim to recover the original Rauw et al. emitter list. Instead, we applied the published Rauw et al. offsets relative to an empirical non-emitter locus as a diagnostic. This gives 51 matched sources with diagnostic H-alpha excess, including 28 of 57 matched Sagitta PMS candidates and 23 of 51 matched Sagitta PMS candidates in the reference sample. The revised manuscript avoids claiming that this "confirms" H-alpha incompleteness; it states only that H-alpha non-excess is non-decisive for PMS status.

Manuscript/data-product change: Added a paragraph in the PMS comparison section and generated `comments_paper/rauw_halpha/summary.json`, `.csv`, `.ecsv`, and a diagnostic PDF.

17. Gaia DR3 values and interpretation of HD 159176

Response: Corrected. The manuscript now uses the Gaia DR3 values quoted by the referee: parallax `1.167 +/- 0.071 mas`, proper motion in RA `2.621 +/- 0.083 mas/yr`, and proper motion in Dec `-0.798 +/- 0.058 mas/yr`. We explicitly discuss the magnitude contrast relative to our Gaia-selected reference sample (`G ~= 5.7` for HD 159176, compared with `G_min = 8.80` and median `G ~= 17.0` for the reference sample) and the possibility of bright-source astrometric systematics, citing the Gaia EDR3 astrometric solution, the bright-source proper-motion correction, and the Maiz Apellaniz (2022) parallax-bias analysis. We also added a Gaia-quality check: HD 159176 has normal `RUWE=0.937`, `visibility_periods_used=13`, and no duplicated-source flag, but it has very significant excess noise (`astrometric_excess_noise_sig=135.6`) and a large IPD multi-peak fraction (`ipd_frac_multi_peak=39`) compared with the reference-member distribution. With this caveat, the parallax and RA proper motion are not treated as decisive; the main discrepancy is the Dec proper motion relative to the selected cluster distribution.

18. Catalogue-comparison section lacks interpretation

Response: Revised. The catalogue-comparison section now explains that the numerical differences mainly reflect methodological choices, spatial windows, probability thresholds, and whether PMS or photometric information is included.

19. Conclusion novelty about HD-independent age

Response: Revised. We no longer claim this is the first HD-independent age estimate. The conclusion now says the reported age is derived from the selected PMS and low-mass population, as in previous PMS-based studies, and does not depend on assuming a particular evolutionary state for HD 159176.

20. Sigma in Fig. A.3

Response: Defined. The revised ASteCA method and Appendix Fig. A.3 caption define `sigma` as the nuisance likelihood-scatter term used in the PyMC diagnostic implementation that generated the figure. It is not a physical velocity, spatial, or age dispersion of the cluster.

## Generated revision products

- Revised manuscript: `data/test/NGC6383/Tex_File/aanda.tex`
- Revised Fig. 1: `data/test/NGC6383/Tex_File/Figures/probabilies_post_sigmaclip.pdf`
- CDS table candidate: `data/test/NGC6383/comments_paper/cds_final/`
- CDS upload package: `data/test/NGC6383/comments_paper/submission_package/aa52082-24_cds_members.zip`
- Rauw H-alpha crossmatch: `data/test/NGC6383/comments_paper/rauw_halpha/`
- Referee action matrix: `data/test/NGC6383/comments_paper/REFEREE_ACTION_MATRIX.md`
