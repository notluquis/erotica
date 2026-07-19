# Clustering audit

This folder documents the transparent clustering state for the A&A
referee-response robustness runs.

Run from the repository root:

```bash
/Users/notluquis/miniforge3/envs/cosmic/bin/python tools/validation/ngc6383_radius_robustness.py --write-full-tables
/Users/notluquis/miniforge3/envs/cosmic/bin/python tools/validation/ngc6383_clustering_audit.py
/Users/notluquis/miniforge3/envs/cosmic/bin/python tools/validation/ngc6383_verify_generated_outputs.py
```

The audit script reads the `.dill` files produced by the paper-faithful
robustness run in `radius_robustness/generated/dill/` and writes audit products
into `generated/`.

The verification script cross-checks the radius summaries, audit summaries,
final label summaries, and source-level ECSV branch flags. It writes
`generated/verification_report.json`.

## Exported files per radius

- `audit_summary.json`: selected configuration, stored-state limitations, and
  paths to the audit products.
- `candidate_sweep_results.csv`: every candidate retained by
  `search_pseudoprobability` after the min/max branch-size filters.
- `min_cluster_size_sweep_track.csv`: every `min_cluster_size` tested and the
  corresponding `desired_len` tracked during the sweep.
- `final_label_summary.csv`: every final HDBSCAN label, including noise, with
  source counts, probability-threshold counts, astrometric summaries, and
  photometric/fidelity summaries. It also marks both the algorithm-selected
  branch and the NGC-like branch nearest to the reference NGC 6383 proper
  motion.
- `final_sources_with_labels.ecsv`: every source in the final clustering table,
  preserving `cluster_hdbscan`, pseudoprobability columns, and boolean flags for
  algorithm-selected and NGC-like branch membership. Use this file to inspect
  any non-reference branch directly.
- `final_cluster_persistence.csv`: final model cluster-persistence values.
- `final_condensed_tree.csv`: full final HDBSCAN condensed-tree table.

## Important limitation

The current `Clustering.search_pseudoprobability` implementation stores
`probability_times`, candidate summaries, and sweep-track summaries, but it does
not persist the full source-by-run label matrix for all 290 HDBSCAN runs. That
means the generated audit can fully inspect the final model and the retained
sweep summaries, but cannot reconstruct every branch assignment in every sweep
iteration unless HDBSCAN is rerun with an additional persistence hook.

## Branch-selection caution

For larger cones, the largest/persistent branch selected by the generic
`max_members` rule is not necessarily the NGC 6383 branch. The audit therefore
exports two labels:

- `algorithm_selected_final_label`: the final label matched to the sweep's
  selected `desired_len`.
- `ngc_like_final_label`: the non-noise final label with `probability >= 0.5`
  nearest to the reference NGC 6383 proper motion
  `(pmra, pmdec) = (2.54, -1.71) mas/yr`.

For the referee-response robustness table, interpret the NGC-like branch rather
than blindly taking the largest final branch.
