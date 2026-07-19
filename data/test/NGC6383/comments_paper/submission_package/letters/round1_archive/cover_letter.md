# Cover letter

Dear Editor,

We submit the revised version of manuscript aa52082-24, "Characterizing NGC 6383: A study of fundamental properties, pre-main sequence stars, mass segregation, and age using Gaia DR3 and 2MASS." Please note that the title was extended ("fundamental properties") to reflect the content more accurately.

We thank the referee for the detailed and constructive report. All twenty points are addressed in the enclosed point-by-point response (`response_to_referee`). The main changes are: the membership methodology is now fully specified (proper-motion-only HDBSCAN feature space, parameter sweep, and composite membership pseudo-probability); a cone-search robustness test over 40--70 arcmin fields was added -- the headline astrometric and structural parameters are stable at the catalog level (50 arcmin refit consistent), while the fitted outer radius is downgraded to a model-dependent, window-limited scale; HD 159176 is treated conservatively, using the Gaia DR3 values and quality indicators discussed by the referee; and a CDS-style table with the 321 likely candidates is provided as a separate package (`aa52082-24_cds_members.zip`).

During the revision we also found and corrected three issues ourselves, none of which alters the conclusions: the half-mass radius had been converted to parsecs with an inconsistent distance (now r_hm = 2.02 ± 0.41 pc), the relaxation-time inputs were made self-consistent (N and M now refer to the same observed stellar system, giving t_rh = 30.5 ± 9.4 Myr), and several parsec-scale uncertainties that previously omitted the distance term are now fully propagated or bootstrapped. In addition, the stellar masses and binary probabilities were re-derived with the current ASteCA release (v0.6.9), and two complementary, bin-independent mass-segregation diagnostics were added (the minimum-spanning-tree ratio Lambda_MSR and the Cartwright & Whitworth structure parameter Q). These author-initiated changes are documented in the "Additional revisions" section of the response.

The upload contains the clean revised source, a marked-change PDF (optional file), the CDS table package, and the point-by-point responses to the referee and to co-author comments.

Sincerely,

L.M. Pulgar-Escobar, on behalf of the authors
