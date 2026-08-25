# Maximizing HDBSCAN for OC membership — plan for EROTICA

_2026-07-21. How to push HDBSCAN as far as possible for NGC 6383 membership (NOT replace it),
from a 3-agent sourced review + code audit. Feeds P02.
`[S]`=sourced (docs/paper), `[I]`=inferred._

```{warning}
**Novelty correction (2026-07-26).** An earlier version of this note called *calibrated
membership* the lead novelty. That claim is **too broad — do not make it.** pyUPMASK
(`2021A&A...650A.109P`) already scored membership with proper scoring rules (Brier,
log-loss, H-measure) over 600 synthetic clusters, and Olivares+2018 (`2018A&A...617A..15O`)
already publishes a per-star *standard deviation* of the membership probability — verified in
the VizieR ReadMe for `J/A+A/617/A15`, column `s_pc`. Measurement-error-aware membership is
likewise precedented (Jaehnig+2021 `2021ApJ...923..129J`, XD-GMM with full covariance over 431
clusters). What remains genuinely open is whether such posteriors are **calibrated** —
simulation-based calibration and a reliability diagram against astrometry-independent labels.
The project keeps a ledger of its own falsified novelty claims; this one is
narrowed against it.
```

## Current state (code-audited)
EROTICA clusters in **2D proper-motion only** (`search_pseudoprobability(columns=["pmra","pmdec"])`),
sweeps `min_cluster_size` ∈ range(10,300), stores `outlier_scores_` but never uses it,
`cluster_persistence_` only for mcs selection. Two p̃ definitions exist:
- **`pMember` (search_pseudoprobability):** p̃ = `probabilities_` (within-cluster λ-strength) × `pFreq`.
- **`pFreq` (_build_pseudoprobability):** f_i alone = fraction of the mcs sweep the star was clustered.
(`probabilities_` is within-cluster λ-strength; **GLOSH = `outlier_scores_`**, a separate object.)

## Upgrades, ranked by payoff
1. **Add parallax / consider 5D — but it's morphology-dependent, not a blanket win** `[S]` — Hunt &
   Reffert (`2023A&A...673A.114H`): sky position + PM + parallax, recentred per field, each axis
   rescaled to a common median/IQR. PM-only admits fore/background interlopers sharing PM, so
   **PM + parallax (velocity space) is the safe upgrade** and the field standard (Tarricq+2022 cluster
   on exactly this, no position). **Full 5D-with-position is NOT a universal win:** equal-weighted sky
   position dilutes coronae/tidal tails (≈half an evolved cluster's mass; Meingast+2021, Tarricq+2022)
   and hits faint low-mass stars hardest → best for *compact* clusters, wrong default for dispersed
   ones. For **very young** objects (NGC 6383, 1–4 Myr) there are no tidal tails yet — dispersion is
   substructure/expansion — so 2D-PM is defensible and **PM+parallax the natural upgrade**, not 5D.
   Feature space is offered as the `columns=` argument with these trade-offs documented in the
   [membership user guide](../guides/membership.md).
2. **CST (cluster significance test)** `[S]` — H&R (`2021A&A...646A.104H`): nearest-neighbour distance
   of members vs surrounding field → S/N; keep >3σ (>5 = "real beyond doubt"). EROTICA selects a
   cluster but never tests it against random fields. Bonus: **tidal radius = the radius maximizing
   CST** (a membership-boundary tool).
3. **Soft clustering — `all_points_membership_vectors`** `[S]` — the true soft-membership: probability
   of belonging to EACH cluster (from distance to persisting **exemplars** + an outlier component,
   Bayesian-combined). Strictly richer than `probabilities_` (defined only within the assigned
   cluster). **Best p̃ = f_i × soft_membership** (keep the dense sweep→f_i — its cross-resolution
   stability is an advantage, orthogonal to any single solution's soft membership). **No OC paper
   uses HDBSCAN soft-clustering vectors → a P02 novelty** `[I]`.
4. **GLOSH `outlier_scores_` as a field-star flag** `[S]` — flag members above ~90th percentile as
   suspect. Do NOT fold (1−outlier_score) into p̃ as a second multiplier: `all_points_membership_
   vectors` already contains the GLOSH component → double-penalizes tail stars.
5. **Fix `min_samples`** `[S/I]` — EROTICA leaves it `None` (≡ min_cluster_size, coupling both knobs);
   H&R fix m_Pts=10 and sweep m_clSize. Decouple.
6. **`exemplars_` + `approximate_predict`** (`prediction_data=True`) `[S]` — robust core/centroid + score
   NEW sources against a frozen clustering (extend membership to a wider Gaia query / tidal-tail
   candidates without re-running). Kerr SPYGLASS (`2021ApJ...917...23K`) sub-clusters via the hierarchy.

## Branch detection (FLASC) — optional post-pass
`hdbscan.BranchDetector` (`detect_branches_in_clusters` → `branch_labels/branch_probabilities`) finds
branches WITHIN a cluster via in-cluster eccentricity (Bot+ FLASC, arXiv:2311.15887 / PeerJ CS 11
e2792) `[S]`. **Win for tidal tails / elongation**; **prototype-and-see for NGC 6530's 5 subgroups**
(Jia 2024 `2024AJ....168...79J` found them by agglomerative, not FLASC — recovery not guaranteed);
**not** for separating Sgr OB1's 3 clusters (distinct density peaks HDBSCAN already resolves). Cheap
optional pass, no re-architecting.

## Tools worth adopting
- **`fast_hdbscan`** (Tutte Institute) — Numba multicore; ships FLASC branch detection + semi-
  supervision + sample weights; active (v0.3.2, 2026). Low-dim Euclidean. Covers scale + branches.
- **cuML HDBSCAN** (RAPIDS GPU) — for all-sky scale; mirrors the hdbscan API + soft clustering.

## Real calibration ground-truth (NOT synthetic)
Crossmatch EROTICA sources to the published **H&R DR3 members** (CDS **J/A+A/673/A114**) by Gaia
`source_id`; label = H&R member (prob>0.5). Reliability diagram of EROTICA p̃ vs H&R-membership
frequency, feeding `erotica/calibration.py` (already built + validated on synthetic; needs the full
members+field catalog `data/40/clustering_results.ecsv`, not the members-only CDS table). **Flag:**
H&R is itself HDBSCAN-based → this calibrates *consistency with the field standard*, not absolute
correctness. Complement with P07's X-ray/Hα/IR youth indicators as an independent truth axis.

## Sources
Hunt & Reffert 2021/2023/2024 (`2021A&A...646A.104H`, `2023A&A...673A.114H`, `2024A&A...686A..42H`) ·
Kerr 2021 (`2021ApJ...917...23K`) · Kounkel & Covey 2019 (`2019AJ....158..122K`) · McBride 2021
(`2021AJ....162..282M`) · hdbscan soft-clustering + GLOSH + prediction docs · FLASC (arXiv:2311.15887).
