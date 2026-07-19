# Clustering process transparency

This document explains the clustering workflow used for the NGC 6383 A&A
revision, how it relates to the thesis, and where the current implementation
differs from the thesis-level description.

## Source-of-truth hierarchy

1. **Scientific interpretation source:** the thesis chapters in
   `/Users/notluquis/thesis-test/manuscript/chapters/`, especially:
   - `ch05/s04-sample-definition.tex`
   - `ch06/s02-membership-pipeline.tex`
   - `ch07/s01-membership-census.tex`
   - `ch07/s02-distance-kinematics.tex`
   - `ch07/s03-structural-parameters.tex`
   - `ch09/s03-limitations.tex`
2. **A&A manuscript source:** `data/test/NGC6383/Tex_File/aanda.tex`.
3. **Paper-facing archived tables:** `data/test/NGC6383/comments_paper/`.
4. **Reproducible scripts for the referee response:**
   - `tools/validation/ngc6383_radius_robustness.py`
   - `tools/validation/ngc6383_clustering_audit.py`

## According to the thesis

The thesis defines the membership pipeline as a four-stage process:

1. HDBSCAN clustering in two-dimensional proper-motion space
   `pmra, pmdec`.
2. Hyperparameter sweep over the HDBSCAN minimum cluster size.
3. Pseudoprobability construction.
4. Parallax-based refinement through deterministic sigma clipping.

Key thesis points:

- Proper motion is used because it gives the strongest cluster-field contrast
  for NGC 6383.
- Parallax is not part of the HDBSCAN feature space. It enters only after
  HDBSCAN through sigma clipping.
- Photometry and sky position are not used in clustering. They enter later for
  characterization, centre/radial structure, CMD, PMS and YSO analysis.
- The pseudoprobability is not a calibrated Bayesian posterior membership
  probability. It is an operational proxy:

```tex
\tilde{p}_i = f_i \cdot p_{\mathrm{HDBSCAN},i}
```

where `f_i` is the fraction of sweep runs in which the source appears in a
cluster, and `p_HDBSCAN` is the native HDBSCAN membership/persistence-like
score in the selected final model.

Relevant thesis references:

- `ch06/s02-membership-pipeline.tex:1-56`
- `ch07/s01-membership-census.tex:1-44`
- `ch09/s03-limitations.tex:6-11`

## Paper-faithful implementation used for the A&A robustness runs

The archived paper sample is reproduced by the following exact settings:

```python
columns = ["pmra", "pmdec"]
min_cluster_size_samples = range(10, 300)
probability_threshold = 0.5
min_cluster_members = 200
max_cluster_members = 1000
select_cluster = False
hdbscan_kwargs = {
    "cluster_selection_method": "leaf",
    "allow_single_cluster": True,
    "match_reference_implementation": False,
    "core_dist_n_jobs": 1,
}
```

Then the selected branch is identified from the final HDBSCAN labels by matching
the final branch to the NGC 6383 proper-motion overdensity. In the 40 arcmin
paper run, this agrees with the branch identified by the stored `desired_len`.
In larger fields this distinction matters: the generic `max_members` branch can
select a larger field-like branch rather than the NGC 6383 branch. The
robustness summaries therefore report both the algorithm-selected label and the
NGC-like label nearest to the reference NGC 6383 proper motion
`(pmra, pmdec) = (2.54, -1.71) mas/yr`.

The parallax clipping is applied to the NGC-like branch at `probability >= 0.5`,
and the final reference sample is `probability >= 0.6` after clipping.

This reproduces the submitted 40 arcmin counts:

| threshold | members | G < 19 |
| ---: | ---: | ---: |
| 0.5 | 321 | 288 |
| 0.6 | 254 | 236 |
| 0.7 | 202 | 191 |
| 0.8 | 161 | 153 |

## Difference between thesis text and current paper-faithful code

| Item | Thesis description | Current paper-faithful code | Implication |
| --- | --- | --- | --- |
| Feature space | Proper-motion-only HDBSCAN | Same | Consistent. |
| Metric | Euclidean in proper-motion space | Same | Consistent. |
| Cluster extraction | Leaf extraction | `cluster_selection_method="leaf"` | Consistent. |
| Sweep range | Described as multiple HDBSCAN configurations; thesis emphasizes `m_cl=43` and 701-source peak | Exact paper reproduction requires `range(10, 300)` | Manuscript should state the exact sweep range for reproducibility. |
| Probability | `tilde p = f_i * p_HDBSCAN` | `probability = probability_times * probability_hdbscan` | Consistent. |
| Parallax refinement | 2-sigma clip around parallax mode after HDBSCAN | Same for paper-faithful run | Consistent. |
| Stored audit state | Thesis conceptually discusses all sweep configurations | Current object stores final `probability_times`, candidate summaries, and sweep sizes, but not the full labels matrix | Full per-source branch history over 290 runs cannot be reconstructed without rerunning or modifying code. |
| Branch identity in larger fields | The thesis describes the NGC 6383 branch as the proper-motion overdensity near the cluster mean | At 60/70 arcmin, the generic `max_members` selector can choose a larger field-like branch; the NGC-like branch is selected by proximity to the reference cluster PM | The robustness table must explicitly state which branch is being analyzed. |
| Membership interpretation | Catalogue-conditioned and threshold-sensitive | Current robustness results show strong radius sensitivity | The paper should not treat one binary list as the physical cluster boundary. |

## Difference between original submitted paper workflow and current workflow

The current `aanda.tex` has already moved closer to the thesis:

- It states HDBSCAN used only `pmra, pmdec`.
- It defines `lambda`.
- It defines the uniform prior notation.
- It caveats the tidal radius as weakly constrained by the search-field size.
- It uses the correct Gaia DR3 values for HD 159176.
- It separates the HD 159176 positional coincidence from kinematic membership.

The still-important differences are now submission-package issues rather than
missing manuscript science:

- The paper has the Table 2 uncertainty note separating formal/posterior
  uncertainty from member-distribution dispersions.
- The paper includes the 40/50/60/70 robustness table and conservative
  interpretation.
- The paper defines the exact paper-faithful sweep range and HDBSCAN settings.
- The paper defines `tilde p` as a pseudoprobability rather than a true
  posterior probability.
- The Rauw et al. H-alpha crossmatch exists in `comments_paper/rauw_halpha/`
  and is interpreted as diagnostic, not as a complete published-emitter-flag
  recovery.

## What the clustering audit exports

Run:

```bash
/Users/notluquis/miniforge3/envs/cosmic/bin/python tools/validation/ngc6383_radius_robustness.py --write-full-tables
/Users/notluquis/miniforge3/envs/cosmic/bin/python tools/validation/ngc6383_clustering_audit.py
/Users/notluquis/miniforge3/envs/cosmic/bin/python tools/validation/ngc6383_verify_generated_outputs.py
```

Generated output:

```text
data/test/NGC6383/comments_paper/radius_robustness/generated/
data/test/NGC6383/comments_paper/clustering_audit/generated/
```

The robustness script regenerates the preprocessing, clustering, parallax-clip
flags, per-radius summaries, full ECSV tables, and `.dill` objects. The audit
script then reads those `.dill` objects from
`radius_robustness/generated/dill/`, so the audit does not depend on old
temporary files. The verifier checks that the summaries and source-level branch
flags agree.

For each radius (`40`, `50`, `60`, `70`):

- `audit_summary.json`
- `candidate_sweep_results.csv`
- `min_cluster_size_sweep_track.csv`
- `final_label_summary.csv`
- `final_sources_with_labels.ecsv`
- `final_cluster_persistence.csv`
- `final_condensed_tree.csv`

These outputs allow another AI or reviewer to inspect:

- every retained candidate `min_cluster_size` result,
- all tested `min_cluster_size` values and tracked branch sizes,
- all final HDBSCAN labels, including non-reference branches and noise,
- source-level final labels and probabilities for all branches,
- summary astrometry/photometry/fidelity per final branch,
- the final condensed tree from HDBSCAN.

The most important columns in `final_label_summary.csv` are:

- `is_algorithm_selected_branch`: branch selected by the generic sweep rule.
- `is_ngc_like_branch`: branch nearest to the NGC 6383 reference proper motion.
- `pm_distance_to_reference`: distance in proper-motion space from
  `(2.54, -1.71) mas/yr`.
- `n_probability_ge_0p5`, `n_probability_ge_0p6`, etc.: how much of that
  branch survives the pseudoprobability thresholds before applying the
  NGC-specific parallax clipping.

## What is not recoverable from the stored objects

The current `.dill` objects do not contain the full `labels_matrix` with shape:

```text
n_sources x 290 HDBSCAN runs
```

That matrix exists during `search_pseudoprobability`, but is discarded after
`probability_times` is computed. Therefore, the audit cannot say:

- exactly which final label/source belonged to which branch in every sweep run,
- how each individual non-reference branch evolved across all 290 runs,
- per-run branch membership lists for all branches.

To make that level of transparency available, modify
`Clustering.search_pseudoprobability` or write a separate audit runner that
persists:

- `labels_matrix`,
- per-run condensed-tree summaries,
- per-run label summaries,
- source-level branch membership for every run.

This should be separate from the default package behavior because the label
matrix is large and paper-specific.

## Scientific interpretation boundary

The current robustness results show:

- The NGC 6383-like proper-motion branch is recovered at all radii.
- The final membership catalogue changes with input cone radius.
- The fitted `R_t` from the 40 arcmin field is not an independent proof of the
  cluster boundary.

They do **not** show, by themselves:

- that 60/70 arcmin additions are contaminants,
- that 60/70 arcmin additions are real halo/tail members,
- that the 40 arcmin catalogue is physically complete.

The next diagnostic layer, if needed, is to compare the added sources at larger
radii against the 40 arcmin reference sample in:

- CMD/PMS locus,
- parallax distribution,
- proper-motion distribution,
- radial density profile,
- astrometric quality/fidelity,
- Rauw H-alpha / Kalari CTTS / YSO evidence.
