# Fine referee-response audit

Date: 2026-05-18

Scope: critical check of the revised A&A manuscript against Referee 1, the
thesis text, and the reproducibility products in `comments_paper/`.

## Bottom line

The revised manuscript now addresses the referee's scientific points in a
defensible way, provided the interpretation remains conservative. The strongest
current position is:

- the 40 arcmin membership sample is reproduced, but it is not proven to be the
  unique physical boundary of NGC 6383;
- the 50/60/70 arcmin tests demonstrate field-window sensitivity, not
  contamination by themselves;
- HD 159176 should be treated as a Gaia source outside the magnitude range of
  the selected candidates, with the declination proper motion as the main
  discriminant;
- Sagitta and ASteCA should be explicitly separated as individual-star versus
  cluster-level age/extinction estimates;
- the H-alpha crossmatch is diagnostic, not a published emitter-flag recovery.

## Point-by-point status

| Referee point | Thesis evidence | Current manuscript status | Critical audit |
| --- | --- | --- | --- |
| 40 arcmin cone too close to fitted `R_t` / boundary | `ch07/s03-structural-parameters.tex:23-24`; `ch06/s02-membership-pipeline.tex:52-56`; recent context from Angelo et al. 2025, Risbud et al. 2025, and Xu et al. 2025 | `aanda.tex:208-221`, `aanda.tex:291`, `aanda.tex:332` | Addressed correctly. Do not claim that 40 arcmin is "the real cluster" or that 60/70 additions are contaminants. The defensible claim is window sensitivity plus recovery of an NGC-like PM branch; the added literature supports why external background estimation and extra-tidal interpretation require caution. |
| Methods under-described | `ch06/s02-membership-pipeline.tex:9-20`, `29-56` | `aanda.tex:141-147` | Addressed. PM-only, Euclidean, `leaf`, `m_cl=10..299`, `min_samples=m_cl`, lambda, and `tilde p` are now explicit. |
| CDS table requested | Thesis data-product logic in `ch09/s05-data-products.tex`; generated files in `comments_paper/cds_final/` and `comments_paper/submission_package/cds/` | `aanda.tex:319-347`, `aanda.tex:465` | Addressed as a manuscript excerpt table plus generated CDS-style files. The final package contains fixed-width `.dat`, ECSV, `ReadMe`, validation manifest files, and the upload archive `submission_package/aa52082-24_cds_members.zip`. |
| Sirius OB1 error | Revision plan `aa52082_referee_revision_plan.md:115` | `aanda.tex:46` | Addressed. No residual `Sirius` string in current `aanda.tex`. |
| HD 159176 predates cluster contradiction | `ch07/s07-hd159176.tex:4`, `33-37` | `aanda.tex:48`, `aanda.tex:399`, `aanda.tex:429` | Addressed. Current paper follows Rauw et al. (2010) for the historical age agreement under assumed membership and removes the contradictory claim. |
| Aidelman X-ray Be / blue straggler wording | `ch07/s07-hd159176.tex:9` | `aanda.tex:50`, `aanda.tex:401` | Addressed and strengthened. Added Linder et al. (2007); wording now treats the Aidelman claim as conditional/historical rather than adopting it. |
| Fig. 1 threshold semantics | Thesis tier definitions in `ch06/s02-membership-pipeline.tex:45` | `aanda.tex:56-58` | Addressed. Caption distinguishes `tilde p=0.6` lower reference threshold from `tilde p=0.8` probable/member boundary. |
| Rauw row in Table 1 | Revision plan `aa52082_referee_revision_plan.md:119` | `aanda.tex:104` | Addressed. Row now has PMS candidate count, reddening, and age. |
| HDBSCAN criteria and weighting | `ch06/s02-membership-pipeline.tex:9-13` | `aanda.tex:141-145` | Addressed. The key sentence is that parallax, angular position, and photometry are not clustering coordinates. |
| Maximum lambda undefined | `ch04/s04-density-structure-and-clustering.tex:20-34`; `ch06/s02-membership-pipeline.tex:18-20` | `aanda.tex:149`, condensed-tree appendix caption/text | Addressed. Lambda is defined as `1/d_mreach`, the HDBSCAN condensed-tree density coordinate; `lambda_val` is treated as a diagnostic of persistence/separation, not as a fitted physical cluster parameter. |
| Uniform prior notation | Revision plan `aa52082_referee_revision_plan.md:122` | `aanda.tex:166` | Addressed. `U(a,b)` is defined at first use. |
| Structural model parameters undefined | `ch07/s03-structural-parameters.tex:7-24`, `57-68` | `aanda.tex:240-254`, `aanda.tex:332` | Mostly addressed. The current text defines the King priors and outer dynamical radii; keep the `R_t` caveat prominent. |
| PMS model dependence | `ch06/s06-asteca-isochrone-fitting.tex:9-12`; `ch09/s03-limitations.tex:13-20` | `aanda.tex:261-263` | Addressed as a limitation. Do not imply PARSEC/Baraffe/SPOTS were run. The quoted age uncertainty remains MIST-conditioned. |
| Fig. 10 vs Table 2 mismatch | `ch07/s04-stellar-population.tex:58-70` | `aanda.tex:263`, `aanda.tex:414`, Table caption at `aanda.tex:171` | Addressed conceptually. The paper separates Sagitta individual estimates from ASteCA cluster-level posterior values. |
| Error bars / Table 2 semantics | `ch07/s02-distance-kinematics.tex` and revision plan `aa52082_referee_revision_plan.md:125-126` | `aanda.tex:171`, `aanda.tex:317` | Addressed. Table now uses observed member dispersions for astrometric quantities and text keeps SEM separate. |
| Proper-motion significance | `ch06/s02-membership-pipeline.tex:9-13`; `ch07/s07-hd159176.tex:24-28` | `aanda.tex:289`, `aanda.tex:403` | Addressed, with appropriate caution. The HD conclusion rests mainly on `mu_delta`, not parallax alone. |
| Cluster centre consistent with HD 159176 | `ch07/s03-structural-parameters.tex:24`; `ch07/s07-hd159176.tex:4` | `aanda.tex:405`, `aanda.tex:429` | Addressed. Position is acknowledged as noteworthy but not decisive. |
| Rauw UBVRI H-alpha crossmatch | Referee request; generated products in `comments_paper/rauw_halpha/` | `aanda.tex:414`, `aanda.tex:427` | Addressed, but keep the limitation: the CDS table provides photometry, not a published emitter flag, so the excess class is diagnostic. |
| HD 159176 Gaia DR3 values and bright-source bias | `ch07/s07-hd159176.tex:14-28` | `aanda.tex:403`; new refs in `cites.bib` | Addressed. Added Linder et al. (2007) and Maiz Apellaniz (2022), and rewrote the magnitude caveat as a comparison to the Gaia-selected reference sample rather than a subjective brightness claim. |
| Catalogue-comparison section lacks interpretation | Thesis comparison logic in `ch08/s01-comparison-with-literature.tex` | `aanda.tex:418-419` | Addressed. Now interpreted as method/window/threshold dependence rather than raw counts only. |
| Conclusion novelty about HD-independent age | `ch07/s07-hd159176.tex:33-37` but thesis language is stronger | `aanda.tex:429` | Addressed. Current paper avoids claiming novelty and says the age is PMS/low-mass based as in previous work. |
| Sigma in Fig. A.3 | Current paper figure lineage, not the final thesis ASteCA formalism | `aanda.tex:263`, `aanda.tex:473` | Addressed for the referee. Caveat: the thesis formal ASteCA likelihood is written as a synthetic-CMD density/Poisson-like likelihood, so describe `sigma` only as the nuisance scatter term in the PyMC diagnostic implementation, not as a physical cluster dispersion. |

## Edits applied in this fine pass

- Fixed the color-color caption typo: `G_RP - G_RP` is now
  `G_BP - G_RP`.
- Added explicit Linder et al. (2007) citation for the modern O-type binary
  classification of HD 159176.
- Added Maiz Apellaniz (2022) to the Gaia bright-source / parallax-bias caveat
  for HD 159176.
- Added Angelo et al. (2025), Risbud et al. (2025), and Xu et al. (2025) to
  support the cone-search / structural-radius caveat with recent Gaia
  open-cluster literature.
- Removed one duplicate uncited BibTeX entry for `1971ApJ...164..399S`.
- Added the missing publisher for the Rauw & De Becker (2008) handbook chapter.
- Updated the response and cover-letter drafts so they match the stronger
  citation set and do not overstate the Aidelman correction.

## Residual risks before resubmission

1. The robustness table is scientifically useful, but it is not a proof of
   completeness or contamination. The wording in the manuscript is currently
   acceptable; do not strengthen it.
2. The `sigma` parameter in Fig. A.3 is a weak point because it comes from the
   diagnostic posterior implementation, while the thesis presents the final
   ASteCA likelihood differently. The current definition answers the referee,
   but avoid expanding that claim.
3. The H-alpha excess classification is reconstructed from Rauw et al. thresholds
   and an empirical non-emitter locus. It should remain labelled as diagnostic,
   not as the authors' original published emitter flag. Do not claim that this
   confirms H-alpha incompleteness; say instead that H-alpha non-excess is
   non-decisive for PMS status.
4. The statement that HD 159176 is excluded should remain caveated by Gaia
   bright-source systematics. The defensible result is "not a secure member in
   the present analysis", not an absolute astrophysical impossibility.

## Validation notes

- `latexmk -C aanda.tex && latexmk -pdf -interaction=nonstopmode -halt-on-error aanda.tex`
  builds `aanda.pdf` successfully at 17 pages.
- `cites.bib` has 123 unique BibTeX keys after removing one duplicate uncited
  `1971ApJ...164..399S` entry.
- `aanda.bbl` has 76 bibliography items and no duplicate `\bibitem` keys.
- `aanda.aux` has 76 `\bibcite` keys and no duplicate keys on disk.
- The previous BibTeX warning for the missing publisher in
  `2008hsf2.book..497R` is resolved.
- The final A&A v9.4 builds no longer report `natbib` multiply-defined warnings,
  `hyperref` empty-target warnings, unresolved labels, underfull boxes, or
  overfull boxes. The only remaining log matches in the final check are the
  harmless `etex` extended-allocation message and the loaded `rerunfilecheck`
  package line.
- The marked-changes PDF no longer contains visible A&A line numbers over the
  manuscript body; `\nolinenumbers` is applied immediately after `\maketitle`.
