# Membership

How EROTICA decides which Gaia sources belong to an open cluster, what choices you have,
and what each choice costs. EROTICA's stance: **maximize what HDBSCAN can tell you, offer
the alternatives, and document the trade-offs** — the scientist picks what fits their
cluster. Nothing here throws random noise at the problem; every option is a defensible
method with a known failure mode, spelled out.

Throughout, {bdg-success}`now` marks what the package does today and {bdg-warning}`planned`
marks the documented upgrade path, so you always know what you are actually running.

---

## 1. What the pseudo-probability means

EROTICA does not read HDBSCAN's cluster label as a hard yes/no. It builds a
**pseudo-probability** p̃ per source. Two definitions live in the code, and they answer
different questions:

- **`pFreq` — recovery frequency $f_i$** {bdg-success}`now`. `search_pseudoprobability`
  sweeps `min_cluster_size` over `range(10, 300)` and records, for each source, the
  **fraction of sweep iterations in which it was assigned to a cluster**. A star clustered
  at every resolution is robust; one that appears only at a single tuning is fragile. This
  cross-resolution stability is an advantage — it is orthogonal to any single solution and
  it is where measurement errors will later enter (§4).

- **`pMember` — frequency × strength** {bdg-success}`now`.
  $\tilde p = f_i \times \texttt{probabilities\_}$, where HDBSCAN's `probabilities_` is the
  within-cluster λ-strength (how deep in the cluster core a star sits). This down-weights
  edge stars that *are* recovered but sit near the boundary.

`probabilities_` is defined **only within the assigned cluster**. It is not the same object
as GLOSH (`outlier_scores_`, a global outlier measure — see §6), and it is weaker than true
soft membership (§6). Keep the two p̃ definitions distinct in any table you publish.

---

## 2. Feature space — what you cluster on

This is the single most consequential choice, and in EROTICA it is **one argument**: the
`columns=` passed to `search`/`search_pseudoprobability`. Each option is legitimate; they
differ in what they let through and what they reject.

| Option | `columns=` | Strength | Disclosure — what it costs |
|--------|-----------|----------|----------------------------|
| **2D proper motion** {bdg-success}`now` (default) | `("pmra","pmdec")` | Robust for young / dispersed clusters; immune to sky-position density dilution; no parallax-error leakage | Admits fore/background interlopers that happen to **share** the cluster's PM |
| **PM + parallax** (velocity space) {bdg-success}`now` | `("pmra","pmdec","parallax")` | Adds distance separation → rejects PM-sharing interlopers at other distances. The **field standard** (Tarricq+2022 run HDBSCAN on exactly this) | Parallax error explodes at faint magnitudes (§3); without error handling this **biases against low-mass members** |
| **5D** (position + PM + parallax) {bdg-success}`now` | `("ra","dec","pmra","pmdec","parallax")` | Best field rejection for **compact** clusters (Hunt & Reffert 2023); position tightens a concentrated core | **Dilutes dispersed members.** Equal-weighted sky position penalizes coronae and tidal tails — which can hold **up to half an evolved cluster's stellar mass** — and hits faint low-mass stars hardest (mass segregation evaporates them outward) |

```{admonition} EROTICA clusters on raw column values — it does not rescale for you
:class: important
HDBSCAN uses a Euclidean metric, so the axis with the largest numeric spread dominates.
The moment you mix units you **must standardize the features yourself** before clustering —
rescale each axis to a common median/IQR and recentre per field (Hunt & Reffert 2023).
Raw `("pmra","pmdec","parallax")` silently **under-weights parallax** (mas spread ≪ PM
mas/yr spread); raw 5D with `ra`/`dec` in **degrees is position-dominated garbage**. 2D-PM
is the one case where both axes already share units, which is part of why it is the safe
default. A helper that standardizes+recentres per field is {bdg-warning}`planned`; until
then, scale the columns before passing them.
```

### Why the default is PM-only, and when to change it

The field's own tidal-tail recovery methods **drop or down-weight sky position**:
Tarricq+2022 cluster on PM + parallax only; Meingast+2021 use velocity-space GMM;
Röser/Jerabkova use convergent-point. Jerabkova's summary is blunt — tidal tails are "not
naturally clustered in any coordinate system." So 5D-with-position is the *wrong* default
for a dispersed cluster, and PM-space is the safe one.

```{admonition} Worked case — NGC 6383 (1–4 Myr)
:class: note
NGC 6383 is **far too young for Galactic tidal tails** (compare the Hyades at ~650 Myr).
Its spatial spread — ~40% of members beyond 5 pc — is **primordial substructure and
gas-expulsion expansion**, not tails. So the 5D-dilutes-the-tails failure mode largely does
not apply, and 2D-PM is defensible; **PM + parallax is the natural upgrade** (distance
separation with no position penalty). The real morphological risk for a cluster this young
is expansion/substructure, handled in analysis, not by tightening the membership metric.
```

**Rule of thumb.** Compact, relaxed, field-contaminated cluster → consider 5D. Dispersed,
young, or with a known corona/tails → stay in velocity space (PM, or PM + parallax with
error handling). When unsure, PM + parallax is the robust middle.

---

## 3. Measurement errors are not uniform

Gaia parallax uncertainty is **magnitude-dependent and severe at the faint end**:
σ_ϖ ≈ 0.02–0.03 mas at G ≈ 9–14, degrading to **~0.5 mas at G = 20** — a 20–30× blow-up
(Lindegren+2021), exactly where low-mass pre-main-sequence members sit. Clustering on the
parallax **point value** scatters faint members off the cluster parallax by ~σ, so a
fixed-tolerance density criterion rejects them → a **magnitude-dependent completeness bias
against the lowest-mass stars**. Unresolved binaries (common among PMS stars) compound it:
they inflate the 5-parameter astrometric residuals, giving RUWE > 1.4 and error bars that
are both larger and non-Gaussian (Belokurov+2020, Penoyre+2020).

This is the disclosure attached to the PM+parallax and 5D options above: the moment
parallax enters the metric, faint-star error must be handled, or the completeness is biased.

---

## 4. Making membership error-aware

The fix drops onto EROTICA's existing machinery with no re-architecting, because the sweep
in §1 is already a resampling loop.

**MC-over-errors resampling of the sweep** {bdg-success}`now`. For each of `n_mc` draws,
**perturb every source by its full covariance** (a correlated Gaussian draw in PM and
parallax using the reported errors *and* correlation coefficients), re-run the
`min_cluster_size` sweep, and record membership. The mean clustered fraction over
draws × resolutions is an **error-aware $f_i$**, and its spread across draws is the
*error-induced membership uncertainty*. A faint star with a huge parallax error flickers in
and out across draws and earns a correctly *lower, hedged* frequency instead of a silent cut.

```python
f_mean, f_std = clu.search_pseudoprobability_error_aware(
    columns=("pmra", "pmdec", "parallax"), n_mc=100,
)
# also written to clu.data["pFreqMC"], clu.data["pFreqMC_std"]
```

This extends UPMASK's resampling (Krone-Martins & Moitinho 2014; pyUPMASK, Pera+2021) two
ways: it samples the **full correlated covariance** (not each observable independently), and
it **fuses the error draws with EROTICA's multi-resolution sweep**, so $f_i$ is stable against
*both* measurement error and clustering resolution. Injection-tested — a star given a large
error scores a lower, higher-variance frequency than its tight-error peers.

It is **frequentist** — a stability frequency *with a spread*, not yet a posterior. Do not
sell the MC frequency as a Bayesian membership probability; that is the next layer (§5). Note
that, like `pFreq`, `pFreqMC` counts being clustered into *any* cluster, not a specific
target — in a field with several comoving groups, pair it with the target-cluster selection.
Cost is `n_mc × sweep` HDBSCAN fits (default ~2900), minutes-to-tens-of-minutes on a full
Gaia field; lower `n_mc` or coarsen the sweep to trade precision for speed.

---

## 5. Membership as a posterior, not just a number

A single p̃ is a point estimate. The literature (Ramezani+2026 frames every method as
$P(\text{member}\mid X)$ and explicitly calls for "calibrated uncertainties rather than
deterministic labels") pushes toward a **posterior**: per-star membership probability *with
its own credible interval*, which then propagates into membership-marginalized population
summaries (mass function, mean age, mean PM).

**Bayesian field+cluster mixture layer** {bdg-warning}`planned`. Run a forward mixture model
(cluster component + field component) whose likelihood carries **per-star covariances** —
the Sarro 2014 / DANCe design (Olivares+2019), or the hierarchical Kalkayotl approach
(Olivares+2020) that jointly infers the cluster distance/size and each star's distance while
convolving the parallax error, zero-point, and Gaia's spatial correlations. Applied to the
error-aware candidate set from §4, this yields the **calibrated posterior**.

The two layers are complementary: **§4 MC-resampling = error-aware candidate generator;
§5 mixture = calibrated posterior.** Neither replaces the other.

### Calibration is checkable {bdg-success}`now`

`erotica.calibration` already provides the tooling: `reliability_diagram`,
`hosmer_lemeshow`, `brier_score`, `expected_calibration_error`, plus isotonic/Platt
recalibration. Ground-truth against the published **Hunt & Reffert DR3 members**
(CDS `J/A+A/673/A114`) by Gaia `source_id`: a reliability diagram of EROTICA p̃ vs
H&R-membership frequency tells you whether "p̃ = 0.8" really means "80% are members."

```{admonition} Calibration caveat
:class: warning
H&R is itself HDBSCAN-based, so this calibrates **consistency with the field standard**,
not absolute truth. Complement it with independent youth indicators (X-ray / Hα / IR) where
available.
```

---

## 6. Getting more out of HDBSCAN

HDBSCAN exposes richer objects than the single label EROTICA currently reads. Each is a
concrete upgrade, none require abandoning the sweep. {bdg-warning}`planned` unless noted.

- **Soft clustering — `all_points_membership_vectors`.** The true soft membership:
  probability of belonging to *each* cluster, from distance to persisting exemplars plus an
  outlier component, Bayesian-combined. Strictly richer than `probabilities_`. The strongest
  p̃ is **$f_i \times$ soft-membership** — keep the sweep's cross-resolution stability *and*
  the single-solution soft membership. No OC membership paper we are aware of uses these
  vectors, so this **appears to be novel** (to be confirmed against the literature before
  claiming it in print).
- **Cluster significance test (CST).** Nearest-neighbour distances of members vs surrounding
  field → S/N; keep clusters > 3σ (> 5σ = real beyond doubt; Hunt & Reffert 2021). EROTICA
  selects a cluster but never tests it against random fields. Bonus: the **tidal radius is
  the radius that maximizes CST** — a membership-boundary tool for free.
- **GLOSH `outlier_scores_` as a field-star flag.** Flag members above ~the 90th percentile
  as suspect. **Do not** fold `(1 − outlier_score)` into p̃ as a second multiplier — the soft
  membership vectors already contain the GLOSH component, so that double-penalizes tail stars.
- **Branch detection (FLASC).** `hdbscan.BranchDetector` finds branches *within* a cluster
  via in-cluster eccentricity (Bot+ FLASC) — a cheap optional post-pass, useful for
  elongation / substructure. Prototype-and-see; recovery of known subgroups is not guaranteed.
- **`exemplars_` + `approximate_predict`** (`prediction_data=True`). Freeze the clustering,
  then score *new* sources (a wider Gaia query, tidal-tail candidates) without re-running.
- **Decouple `min_samples`.** EROTICA leaves `min_samples=None`, which ties it to
  `min_cluster_size`. H&R fix `min_samples = 10` and sweep only `min_cluster_size` — decouple
  the two knobs.

---

## 7. Decision guide

| Your cluster | Feature space | Errors | Membership |
|--------------|--------------|--------|------------|
| Compact, older, field-contaminated | 5D (rescaled) | point ok if bright | p̃, then posterior if publishing a mass function |
| Dispersed / has corona or tails | PM + parallax, **not** 5D | MC-resample (faint members matter) | posterior |
| Very young (≲ 5 Myr, e.g. NGC 6383) | 2D-PM or PM + parallax | MC-resample | p̃ + calibration; substructure handled in analysis |
| Faint-limited, low-mass focus | PM + parallax | **MC-resample required** | posterior, calibrated vs H&R |

---

## 8. Status summary

| Capability | State |
|-----------|-------|
| 2D-PM / PM+ϖ / 5D via `columns=` | {bdg-success}`now` |
| Sweep-based `pFreq` / `pMember` p̃ | {bdg-success}`now` |
| Calibration tooling (`erotica.calibration`) | {bdg-success}`now`, validated on synthetic |
| MC-over-errors resampling (`search_pseudoprobability_error_aware`) | {bdg-success}`now`, injection-tested |
| Bayesian field+cluster posterior | {bdg-warning}`planned` |
| Soft clustering / CST / GLOSH flag / branch detection | {bdg-warning}`planned` |

---

## Sources

Hunt & Reffert 2021/2023 (`2021A&A...646A.104H`, `2023A&A...673A.114H`) ·
Tarricq+2022 (`2022A&A...659A..59T`) · Meingast+2021 (`2021A&A...645A..84M`) ·
Lindegren+2021 (`2021A&A...649A...2L`) · Belokurov+2020 (`2020MNRAS.496.1922B`) ·
Penoyre+2020 (`2020MNRAS.495..321P`) · Krone-Martins & Moitinho 2014 (`2014A&A...561A..57K`) ·
pyUPMASK / Pera+2021 (`2021A&A...650A.109P`) · Sarro+2014 (`2014A&A...563A..45S`) ·
DANCe / Olivares+2019 (`2019A&A...625A.115O`) · Kalkayotl / Olivares+2020 (`2020A&A...644A...7O`) ·
Ramezani+2026 (`arXiv:2607.13711`) · hdbscan soft-clustering / GLOSH / prediction docs ·
FLASC (`arXiv:2311.15887`). See also the design note
[maximize_hdbscan_membership](../design-notes/maximize_hdbscan_membership.md).
