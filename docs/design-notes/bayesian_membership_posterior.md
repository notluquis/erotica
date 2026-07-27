# Bayesian membership posterior — adoption plan

_2026-07-26. What to build for a calibrated per-star membership posterior, which existing code to
adopt as the base, and — critically — which novelty claims do **not** survive contact with the
literature. From a multi-agent sourced review with primary-source verification.
`[S]`=sourced/verified, `[I]`=inferred._

## 0. The headline, corrected

The obvious claim — *"we report a credible interval on per-star membership probability"* — **will be
refuted by a referee from this community in one sentence.** It was published in 2018.

**Olivares+2018** (`2018A&A...617A..15O`, DANCe IV) §2.4 `[S]`:

> "each realisation from the joint posterior distribution of the model parameters (i.e. each
> iteration of the MCMC) results in a value for both cluster and equal-mass binaries membership
> probabilities. … We also report the sensitivity of these membership probabilities to the cluster
> parameters by means of the standard deviation of the 1700 samples obtained for each object."

It is a **published catalogue column**: VizieR `J/A+A/617/A15`, 1,424,893 rows, bytes 47–54 `s_pc` =
*"Standard deviation of cluster membership probability"*. The decision rule (§3.2) is
`P_c + σ_Pc > p_t = 0.84`. And the interval-style fallback is occupied too — **Olivares+2021**
("Miec", `2021A&A...649A.159O`) §4.1 accepts a source when *"86% of its membership probability
distribution is above the optimum probability threshold"* — a credible-interval rule in all but name.

**Sarro+2014** (`2014A&A...563A..45S`) §6 is where the gap was *named* — *"we have no estimate of the
uncertainties in the inferred membership probabilities themselves, a direct consequence of using a
maximum-likelihood model"* — and the same group closed it four years later. `[S]`

### What actually happened: the field abandoned it for scale

**Olivares+2019** (`2019A&A...625A.115O`) §4, the authors on their own 2018 method `[S]`:

> "In our opinion, the methodology of Olivares (2018) has clear advantages over the others: takes
> into account the full covariance matrix of the observations, incorporates photometry, and deals
> with partially observed objects. **However it is so computationally expensive that it is
> impractical for our data set made of millions of sources.**"

So the per-star posterior was not overlooked — it was **traded away for scale**, and every descendant
(DANCe → Mecayotl) dropped it.

### The defensible framing

> We restore, in **6D Gaia phase space at survey scale** — with the full per-star covariance *and*
> parallax spatial correlations — a per-star membership posterior that the field abandoned when it
> scaled, and we are the **first to check whether it is calibrated**.

Each clause does work: 2018/2021 ran on **ground-based DANCe proper motions + photometry with no
parallax**, so no Gaia astrometric covariance and no spatial correlations `[S]`. And across **36
screened Kalkayotl-using works, zero coverage checks** — nobody has ever audited interval
calibration; all calibration evidence is author-generated on synthetic data `[S]`.

## 1. σ_p is cheap — the cost is in the fit, not the evaluation

Olivares+2018 §4 `[S]`: computing 1700 samples of the membership probabilities for ~1.5M stars took
**4.11 h**. That is *evaluation* over θ draws that already exist; the 30-day emcee fit produced them.

**Mecayotl already has the right split** `[S]`: Kalkayotl fits θ on ~10³ members (where correlations
are affordable), then classification runs over 45M sources. So restoring σ_p is a change at two named
call sites, not a redesign:

1. `Amasijo.py:120 _read_kalkayotl` reads **one `statistic` column** (mean/mode) → the synthetic
   cluster is generated from a **point estimate**, so zero cluster-parameter uncertainty reaches
   p_i. Carry θ *draws* through instead.
2. `Kalkayotl/inference.py:1804` collapses per-draw responsibilities with `st.mode(...argmax)`.
   Retain them.

**Kalkayotl HEAD already emits the right object when asked** `[S]`: `_classify(save_probabilities=True)`
writes `Sources_probabilities.csv` indexed `(source_id, sample)`. It is **default-off**, Mecayotl
never enables it, and it appeared in commit `7aa8ff5` on **2025-04-30 — after the 2.0 paper**,
documented in no paper and no README.

## 2. What to adopt

| Component | Verdict | Why |
|---|---|---|
| **Kalkayotl 2.1.4** | **adopt as base** | only fully-released piece; already PyMC5 + **NumPyro/JAX default**; has the spatial-correlation machinery + phase-space model; actively maintained (2026-04-22) |
| **Amasijo 0.2.1** | **adopt outright** | packaged, working; it is the forward simulator |
| **Mecayotl** | **adopt the *architecture*, not the code** | see below |

**License is clean** `[S]`: EROTICA is AGPL-3.0, these are GPL-3.0, and **GPLv3 §13 explicitly permits
combining into an AGPL-3 work** (verified verbatim in `Kalkayotl/COPYING:552`). No relicensing needed.

```{warning}
**Mecayotl is released but NOT runnable.** `github.com/olivares-j/Mecayotl` (GPL-3.0, HEAD 2026-04-06)
does `from gmm import GaussianMixture` at `mecayotl.py:394`, but **`gmm.py` has never existed in the
repo**, and it shells out to `Ayome` (a numba-CUDA EM kernel) which is **nowhere on GitHub or PyPI**.
The released artifact is the orchestrator; the inference engine is unreleased. "Adopting" Mecayotl
means reimplementing its core regardless.
```

Take Mecayotl's **architecture**: the fit-θ-on-10³ / classify-on-10⁷ split, the forward-simulation
trick for the cluster component (fit the cluster GMM to *noise-free simulated* astrometry, so
deconvolution is bypassed for that component), and per-magnitude-bin threshold calibration by MCC.

## 3. Concrete improvements — each is a citable contribution

1. **Reimplement the deconvolving mixture** (XD/EM) as a **NumPyro mixture**. Unavoidable core work.
2. **Break the synthetic-calibration circularity** — the strongest adopt-and-fix item, and it is
   structural. Mecayotl draws synthetics from the Kalkayotl point estimate, fits a GMM to them, then
   optimises thresholds against *those same synthetics* — **the calibration cannot detect model
   misspecification because the validation data is generated by the model being validated.** Galli+2020
   says so outright: these rates *"cannot be understood as absolute measures for the true properties of
   the solution"* `[S]`. The empirical damage is large: IC 4665 synthetic contamination ~2–4% vs
   **30–35%** when validated against Gaia parallaxes (Miret-Roig+2019); Upper Sco <4% vs ~8%
   (Miret-Roig+2022, *Nature Astronomy*) `[S]`.
3. **Fix the field-model bias.** Mecayotl's field GMM is fitted to a subsample that still contains
   true members. Olivares+2018 diagnosed exactly this: *"underestimating the photometric field
   likelihood … increases the cluster membership probabilities"* — a misfit field model biases p_i
   **high** `[S]`.
4. **Restore σ_p** via the two call sites in §1.
5. **Parallax zero-point.** This family subtracts a **scalar** (−0.017 mas). Lindegren+2021 Z5/Z6 is
   implemented **nowhere**. Kalkayotl issue **#22, opened by the author 2021-03-15, still open**:
   *"The parallax zero-point has a dependence in colour, magnitude and position. Please implement
   these functions"* `[S]`. **EROTICA already depends on `gaiadr3-zeropoint`** → nearly free
   differentiator.
6. **Spatial correlations in the membership likelihood — currently absent** `[S]`. Mecayotl's GMM uses
   only the per-star 6×6 `C_i`; Gaia angular correlations enter *nowhere* in the 45M-star
   classification, only inside the ≲10³-star parameter fit. Doing this at scale requires breaking the
   dense Cholesky (low-rank/block kernel approximation, or CG/Hutchinson GP methods).
7. **Fix the decision rule.** `P_c + σ_Pc > p_t` **promotes** uncertain stars — anti-conservative. A
   lower-bound or expected-utility rule is a legitimate concrete improvement `[I]`.
8. **Calibration audit** — nobody has done one. Pairs with `erotica/calibration.py`.

Two code-level inconsistencies worth citing `[S]`: `angular_correlations=None` is passed explicitly at
`mecayotl.py:1191`, and synthetic covariances are built **diagonal-only** at `:1260-1266` (`self.RHO`
excluded at `:1205`) while real stars carry the full 6×6.

## 4. Reimplementation-critical details `[S]`

**Membership probability** (`mecayotl.py:1120`):
`p_c = 1/(1 + exp(ln_prior_ratio + llk_field − llk_cluster))`, with
`ln_prior_ratio = log(n_field/n_cluster)` if `use_prior` else **0** — and `use_prior=False` is the
default, so **the published quantity is a likelihood ratio, not a posterior probability.**

**Evaluation-time convolution:** `N(x_i | m_k, Σ_k + C_i)`. **Fitting-time deconvolution** is XD,
applied to the *field* model; for the *cluster* it is bypassed (GMM fitted to noise-free simulated
astrometry, `sg_syn = zeros`, `mecayotl.py:546`). That is the paper's central trick.

**Gaia spatial correlation kernels** (`Kalkayotl/Functions.py:51-110`, θ in degrees):

```text
CovarianceParallax [mas²]   Lindegren+2020 (default): 0.000142*exp(-θ/16.0)
CovariancePM [(mas/yr)²]    Lindegren+2020 (default): 0.000292*exp(-θ/12.0)
```
Added to the parallax×parallax sub-block and to the pmra/pmdec blocks (same matrix for both, no cross
term), then Cholesky.

**Scaling wall:** `load_data` builds a dense **(N·D)×(N·D)** covariance and Cholesky-factorises it
(`inference.py:344-390`). N=10³, D=6 → 6000² ≈ 288 MB, fine. N=10⁵, D=6 → ≈ **2.9 TB**, infeasible.
`indep_measures=True` bypasses it entirely. The "≲1000 stars" abstract figure is softer than it reads —
N=9562 has been run in 1D (parallax-only → N×N).

**Reconstructed `gmm.py` API** (from call sites — this is the build spec):

```python
GaussianMixture(dimension: int, n_components: int, threads_per_block: int = 32)
  .setup(X, uncertainty=U)          # X (N,6); U (N,6,6)
  .fit(tol, max_iter, rho, tol_covariance, init_params, ...)
  .log_likelihoods(weights, means, covariances) -> (N, K)   # per-component, not summed
  # attributes: weights_, means_, covariances_, determinants_, aic, bic
```
Component selection by **AIC**, restricted to models with ≥10 stars per component. XD regularisation
`w` is set to **the square of the minimum uncertainty in the data set** (Olivares+2019).

## 5. Known failure modes reported by USERS (the step that matters)

- **Galindo-Guil+2022** — the best per-star reliability statement: *"the mere fact of using the
  Bayesian approach is not enough to calculate reliable distance measurements… bona fide members have
  accurate distances, but the background objects and outliers with parallaxes higher than 20% are
  likely to be biased."* `[S]`
- **Reyes-Reyes+2024** behaviourally distrusted the intervals — two runs on the same cloud gave 2.109
  and 1.835 kpc; they averaged and hand-assigned ±0.14 kpc. `[S]`
- **Jatmiko+2026**: *"we have to assume that every member of the cluster has a 100% membership
  probability and neglects the contamination effect of field stars… this is the limitation of
  Kalkayotl."* `[S]`
- **Jadhav+2025 declined to use it** for tidal tails: *"the tidal tails are not spherically
  symmetric… thus, simple prior distributions cannot be used."* `[S]`
- Kalkayotl's own **Assumption 5**: *"the input list of cluster members is neither contaminated nor
  biased"* — *"however, in practice, this rarely happens."* `[S]`
- **No external bug reports in seven years** (31 issues, all by the author bar one co-author) — the
  independent-user record is thin, not hostile. `[I]`
- Mecayotl's own reception: Jadhav+2025 grades the Coma Ber list **SILVER, 5/6 flags**, failing only
  sky-torsion — but *all five* Melotte 111 catalogues fail that flag, so it is cluster geometry, not
  the method. `[S]`

## 6. Collaboration note

`phd-targets.md` ranks **Olivares + Sarro (UNED)** as a top application target. The two genuine gaps —
nobody has tested prior sensitivity outside the DANCe circle, and nobody has checked interval
calibration at all — read as an **opening to build on their work**, not a critique of it. Frame any
paper accordingly.

## Sources
Sarro+2014 `2014A&A...563A..45S` · Olivares+2018 `2018A&A...617A..15O` (VizieR `J/A+A/617/A15`) ·
Olivares+2019 `2019A&A...625A.115O` · Kalkayotl `2020A&A...644A...7O` · Miec `2021A&A...649A.159O` ·
Mecayotl `2023A&A...675A..28O` · Kalkayotl 2.0 `2025A&A...693A..12O` · Jaehnig+2021
`2021ApJ...923..129J` · pyUPMASK `2021A&A...650A.109P` · repos `olivares-j/{Kalkayotl,Mecayotl,Amasijo}`.
