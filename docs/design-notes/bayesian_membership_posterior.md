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

## 5b. The calibration claim, adversarially audited

An agent was tasked with **falsifying** "nobody calibration-checks membership probabilities." It
survives — but only in this narrow form, and with a counter-example that must be cited *first*.

**Hard evidence (enumeration, not search):** across VizieR, **697 catalogs publish a stellar
membership probability; exactly 2 publish a per-star uncertainty on one.** `[S]`

```{danger}
**Cite this up front or a referee will find it.** **redMaPPer IV** (`2015MNRAS.453...38R`)
validates **galaxy**-cluster membership probabilities against spectroscopic rates to ~1%. The claim
must therefore be **scoped to stellar/open clusters**, with the reason stated: there is no
spectroscopic ground-truth analogue at the required depth (see `membership_ground_truth.md` §1).
```

**Import the photo-z literature as prior art — do not treat it as competition.** That line is mature
and self-aware: Bordoloi 2010 (`2010MNRAS.406..881B`) → Wittman 2016 (`2016MNRAS.457.4005W`) →
Schmidt 2020 (`2020MNRAS.499.1587S`) → **Dey 2025 Cal-PIT** (`2025MLS&T...6d5058D`), shipped as
production infrastructure in RAIL/`qp`. An adversarial referee *will* find these. `[S]`

**The transfer is exact, not analogical** — state it precisely:
- **CRPS reduces *identically* to the Brier score in the binary limit.** For a Bernoulli(p)
  predictive the CDF is a two-step function and `∫(F(z) − 1{z≥y})² dz` collapses to `(y − p)²`. Same
  functional. Log-score → log-loss likewise.
- **PIT does *not* transfer** — a Bernoulli predictive has no continuum, so its PIT is degenerate.
  The binary counterpart of PIT-uniformity is the **reliability diagram**; both test *probabilistic
  calibration* (`F(Y)~U(0,1)` continuous vs `E[Y|p̂=p]=p` binary). KS/CvM/AD on PIT ≡ ECE/MCE.
- **Decomposition:** Murphy (1973) partitions Brier into reliability − resolution + uncertainty;
  Hersbach (2000) gives the CRPS analogue. Naming both makes "aggregate proper score ≠ calibration
  check" airtight.

**Why the field's own proper-scoring work does not close the gap** `[S]`: pyUPMASK and PLAsTiCC's
metric paper (`2019AJ....158..171M`) compute **aggregate** scores without ever separating reliability
from resolution — PLAsTiCC selects a proper scoring rule while containing *zero* occurrences of
"calibration". The **Murphy decomposition appears once in the entire astronomical corpus**, in a
weather review. The one coarse membership check ever attempted (Jackson+2020: two bins, one cluster,
against lithium) missed by a factor 2.5 and was never repeated.

### The strongest novelty position: **conditional** calibration
PIT-uniformity and a global reliability diagram both test *marginal* calibration — which can hold
globally while failing in **every** subpopulation. Dey+2025 attacks exactly this for photo-z
(instance-wise/local calibration). The binary analogue — **reliability diagrams in bins of magnitude,
crowding, and cluster radius** — has no counterpart in the membership literature, and for us it is
precisely where miscalibration would live (the faint/large-parallax-error regime). `[I]`

**Final claim wording (use verbatim, then trim):**

> Binary membership probabilities in Galactic star-cluster work are not calibration-checked. Across
> VizieR, 697 catalogs publish a stellar membership probability and exactly two publish a per-star
> uncertainty on one. No open-cluster paper isolates the calibration component — the field's own
> proper-scoring-rule work computes aggregate scores without separating reliability from resolution,
> and the Murphy decomposition appears once in the entire astronomical corpus, in a weather review.
> Meanwhile the ensemble spread needed to do better is already computed and discarded by every major
> code in the field. The conditional-calibration machinery developed for photo-z PDFs has never been
> transposed to the binary case — and cannot be transposed naively, since PIT is degenerate for a
> Bernoulli predictive while CRPS collapses exactly onto the Brier score.

```{note}
**Search-methodology caveat.** ADS `full:` is **not** strict phrase matching, and
`database:astronomy` leaks other fields. Null results are only as good as a body-indexing control,
so present the ADS nulls as *suggestive with stated controls*, never as proof. The VizieR 697/2
census enumerates rather than searches and is the hard evidence. Huertas-Company & Lanusse 2023 is
**not** body-indexed in ADS — do not cite it as a verified negative.
```

## 5c. MiMO as a calibration testbed — measured, not assumed

**MiMO** (method `2022ApJ...930...44L`; catalog `2025AJ....170..288L`) was our best candidate for
"calibrate someone else's published posterior." Data products downloaded and measured `[S]`.

**Correction to the abstract's wording.** *"Full likelihood chains and photometric membership
probabilities"* parses as **chains for the 7 cluster-level parameters, and a single scalar per star.**
`member_stars.fits` is **511,735 rows × 5 columns** — `Cluster`, `source_id`, `ra`, `dec`,
`p_member`. No photometry, no errors, no per-component likelihoods. Access: DOI `10.12149/101693`
(China-VO), **CC BY 4.0**, *not* on VizieR.

**A reliability diagram is comfortably feasible** — the distribution is not saturated:
25.7% sit at exactly 0.0, **zero** at exactly 1.0, and **44.75% (229,013 stars) lie in 0.05 < p < 0.95**,
with 739 clusters having ≥100 interior stars.

**Nobody has calibrated it** — all 24 citations of the method paper and all 7 of the catalog walked;
zero reliability diagrams, Brier/log-loss, coverage checks, or independent membership comparisons.
⚠ **Pre-emption risk:** the catalog paper announces *"a dedicated follow-up study… of blue straggler
candidates"* combining kinematic + photometric probabilities. Not out yet, but the authors have
declared intent on the adjacent question.

```{danger}
**`p_member` is NOT usable as ground truth for an astrometric classifier.** Three structural reasons:
1. **Support censoring** — it exists only for stars that already passed a Cantat-Gaudin+2020
   astrometric cut (`Δμ < 6σ_μ`, parallax `6σ_ϖ`, `r < 3r₅₀`, `G<18`). So `p_member` is
   *P(drawn from the cluster CMD | already passed an astrometric cut)*, and the censoring correlates
   with our score.
2. **Systematic, not random, label error** — the authors state high-kinematic/low-photometric stars
   are **blue-straggler candidates**. For a known class of true members MiMO assigns `p→0` *by
   design*. Using it as truth scores our classifier "overconfident" exactly where it is right.
3. **It is a plug-in, not a posterior** — `p_member = (cl_prob * f_cl) / prob` evaluated at
   `Θ_best-fit` (`MiMO.py:547`), not marginalised over the chains.
```

**The empirical test was run.** MiMO × OCCAM DR17, name-matched **78 clusters**, sky-matched at 1.5″
→ **3,925 star pairs**, scored against OCCAM's *own* published criterion (Myers+2022 §4.1: threshold
**0.01** = within 3σ — these are p-values, not posteriors):

| target | base rate | mean `p_member` | Brier (vs base-rate) | AUC (cluster-bootstrap) |
|---|---|---|---|---|
| OCCAM member (RV, FeH, PM all >0.01) | 0.351 | 0.797 | **0.522** (0.228) | **0.449** (0.376–0.525) |
| RV_PROB > 0.01 alone | 0.769 | 0.797 | 0.261 (0.178) | 0.534 (0.504–0.562) |

**In this regime `p_member` is worse than a constant and its AUC is consistent with chance.**
Reliability is near-flat: the p≈0.99 bin (N=2,419) contains 35.2% OCCAM members; the p≈0.002 bin
(N=527) contains 22.8%.

```{warning}
**Do not over-read that table.** It covers **0.77%** of the catalog; the stars are **APOGEE bright
giants** — structurally MiMO's *worst* regime, a short giant branch sitting in dense field-giant CMD
territory; and **six clusters hold 56%** of the pairs. MiMO's discriminating power lives on the main
sequence, and APOGEE cannot test it there. This is a reason to bring in Gaia-ESO/Jackson+2022, which
reaches the MS — not a result to publish as-is.
```

### If we take it, frame it correctly
**Not** *"MiMO is miscalibrated"* — the authors explicitly call `p_memb` photometric, complementary to
kinematics, and tell users to combine. A bare critique is a strawman and a referee will say so. The
defensible question is: **what does a published `p_member` column actually mean, and does the
recommended combination calibrate?**

- **Labels:** OCCAM DR17 at its own 0.01 threshold **plus** Gaia-ESO/Jackson+2022 (essential — it
  reaches the main sequence where APOGEE cannot).
- **Binning:** equal-count bins in `p`, with separate treatment of the 25.7% at exactly 0 — log-loss
  is undefined without clipping, and **the clip value becomes a free parameter of the analysis**; state it.
- **Statistic:** reliability curve + ECE + Brier decomposed into reliability/resolution, AUC reported
  **separately** for discrimination. Errors from a **cluster-level bootstrap**, never per-star (see
  the 56%/6-cluster concentration).
- **The clean controlled experiment:** recompute `p_member` **marginalised over the published chains**
  (they ship `logwt`, so reweighting is legitimate) versus the plug-in value. Expect a *small* effect —
  the cluster-parameter posteriors are tight. The bigger lever is the one the paper admits: unmodelled
  MS broadening / differential reddening is absorbed into an inflated `f_fs`, shifting every
  `p_member` in that cluster.

A plug-in-vs-marginalised check alone is a technical note. It becomes a **methods paper** when it
answers the general question — *do the per-star probabilities that Gaia-era cluster catalogs now ship
by the hundred-thousand mean what users assume?* — with MiMO as the worked example and the
constructive result being a **calibrated `p_astro × p_phot` combination validated against
spectroscopy**. That lands on our home turf without circularity.

**Code status:** `github.com/luly42/mimo` — 4 files, 3 stars, **no LICENSE**, no tests. Two breakages:
the notebook imports `mimo_01`, which does not exist (it is `MiMO.py` renamed — a `cp` fixes it), and
`iso_model.h5` is **4.29 GB, absent from GitHub**, living only inside an **8.83 GB** China-VO archive.
Better than Mecayotl (nothing is *missing*), but not `pip install`-able.

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
