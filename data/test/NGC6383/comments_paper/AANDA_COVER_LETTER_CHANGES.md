# Cover-letter change summary

Dear Editor,

We submit a revised version of manuscript aa52082-24, "Characterizing NGC 6383: A study of pre-main sequence stars, mass segregation, and age using Gaia DR3 and 2MASS." We thank the referee for the detailed report. In the revised manuscript we have made the following changes.

1. We expanded the HDBSCAN membership-method description, specifying the proper-motion-only feature space, Euclidean metric, `leaf` selection, the exact `min_cluster_size=min_samples` sweep, the definition of the condensed-tree lambda parameter, and the composite membership pseudo-probability.

2. We repeated the membership recovery using 40, 50, 60, and 70 arcmin input cones and added a robustness table. The revised text now states that the NGC 6383-like branch is recovered, but the outer membership and tidal-radius interpretation are field-window sensitive. We added recent Gaia-based open-cluster references to support the distinction between core recovery, background-dependent structural radii, and extra-tidal candidate interpretation.

3. We regenerated Fig. 1 using the composite membership pseudo-probability and added threshold markers at 0.6 and 0.8. The caption now distinguishes the reference-sample threshold from the probable/member boundary.

4. We corrected the introduction regarding Sgr OB1, Rauw et al. (2010), HD 159176, and the Aidelman et al. (2018) blue-straggler interpretation, adding the modern HD 159176 spectroscopic reference requested by the referee.

5. We updated Table 1 with Rauw et al. (2010) values and updated Table 2 so that the quoted astrometric uncertainties correspond to member dispersions where appropriate. The text now separately gives the smaller standard errors on the means.

6. We clarified that Fig. 10 shows individual-star Sagitta estimates, while Table 2 reports cluster-level ASteCA parameters, and we added a caveat that only MIST PMS models were used.

7. We corrected the Gaia DR3 astrometric values for HD 159176 and discuss the bright-source caveat, including magnitude-dependent Gaia systematics and Gaia-quality indicators, before interpreting its proper-motion discrepancy.

8. We cross-matched the likely-member catalogue against the Rauw et al. (2010) UBVRI H-alpha catalogue and added the resulting interpretation to the PMS comparison section.

9. We added an excerpt of the likely-member catalogue to the manuscript and prepared the full CDS-style table with 321 candidates and flags for the 254-source reference sample and the higher-confidence member subset. The full table is provided as a separate upload package, `aa52082-24_cds_members.zip`.

10. We defined the `sigma` parameter shown in the Appendix posterior plot as the nuisance likelihood-scatter term used in the PyMC diagnostic implementation that generated the figure.

The source files contain the clean revised version. A separate marked-change PDF is prepared for upload as an optional file.

Sincerely,

L.M. Pulgar-Escobar et al.
