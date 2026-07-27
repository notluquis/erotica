# Cross-domain imports — calibrating without labels

_2026-07-27. Practices from applied fields **outside astronomy** that survive our binding constraint:
**no ground-truth labels.** Sourced review; `[S]`=verified from primary source, `[I]`=inferred._

## 0. The discriminator

Every field has calibration practice — that is not the question. The question is whether the artefact
**survives having no labels**. Each item below is tagged:

- **`TRUTH-FREE`** — runs on the real catalogue as-is
- **`SIM-ARM`** — needs outcomes, so only runs on synthetic mixtures, where our existing
  proper-scoring plan already reaches

This re-ordered the obvious ranking: clinical/TRIPOD is the famous one, but it is a **governance
vehicle, not an importable estimator**.

## 1. ⭐ Proteomics — target-decoy FDR + posterior error probability `TRUTH-FREE`

**The strongest find, and it beats our current plan.** The only mature field with (i) *no* ground
truth at all, (ii) a **per-item calibrated probability** anyway, and (iii) a community mandate to
report it.

| Artefact | Reference | Gives |
|---|---|---|
| Target–decoy competition | Elias & Gygi 2007, *Nat. Methods* 4(3):207–214 | global FDP at any threshold, no labels |
| **PEP / local FDR** | Käll, Storey, MacCoss & Noble 2008, *Bioinformatics* 24(16):i42–i48 | **per-item calibrated posterior** from the decoy null |
| Entrapment queries | *J. Proteome Res.* 2025, PMC11894652 | validates the decoy assumption itself |

**Mapping.** Generate **decoy sources** — objects that cannot be cluster members by construction
(phase-space-permuted stars, offset-field draws) — push them through the **full unmodified**
membership pipeline, and count how many cross p̃ > threshold. That count estimates the catalogue's
FDP at that threshold. Käll's PEP then converts the decoy score distribution into a **per-star
calibrated membership probability** — exactly the object we want, with no labels.

```{important}
**Why this is better than the injection–recovery we planned.** Injection–recovery requires a
**correct generative model of the true positives** — i.e. of the very thing under test. A decoy
design requires only a generator of things that **cannot** be members. That is a **strictly weaker
assumption**, and it is why proteomics can report a defensible error rate for objects nobody has ever
verified.
```

```{warning}
**The condition of import — state it, do not bury it.** Decoy methods require (verbatim) that *"it is
equally likely for an incorrect match to be mapped to either a decoy or a target candidate."* Our
analogue: **a decoy must be exchangeable with a true non-member under the scoring function.** In Gaia
this is *not* free — field density, kinematics and the selection function vary with sightline, so an
offset-field decoy from a different line of sight is **not** exchangeable with the local field.
**Entrapment queries** are the published way to test this.
```

**Novelty, narrowed honestly** `[I]`: astronomy uses offset/control fields routinely. The importable
artefact is **not** "use an offset field" — it is *counting decoys above threshold as a defined FDP
estimator with stated exchangeability assumptions*, rather than using the offset field only to fit a
field model. A bounded ADS search returned 5 hits, none methodological — call that *"no hits in a
bounded search"*, **not** *"nobody does this"*.

## 2. Record linkage — three artefacts `TRUTH-FREE`

Structurally identical to our problem: per-pair match probability, no labels, threshold to build a
file, then compute population statistics from it.

**A. Truth-free FDP, independently derived.** Robach, Hof & van de Wiel 2025 (`arXiv:2503.20627`):
synthesise records from the data's own empirical distribution; synthetic records *cannot* form true
links, so linked synthetics estimate the false-link count. **Two applied fields converging on the same
design is strong evidence it transfers** — and Robach adds what proteomics lacks: it is explicitly
built for the regime *"where links and non-links have similar distributions"*, i.e. **low contrast**,
which is our hard regime. `[S]`

**B. The three-decision rule.** Fellegi & Sunter 1969 (*JASA* 64(328):1183–1210) do not force a binary
call: **link / possible-link / non-link**, with the middle band *sized by the two error rates you
choose*. Modern Bayesian form: Sadinle 2016 (`arXiv:1601.06630`) — *"partial Bayes estimates that
allow uncertain parts of the bipartite matching to be left unresolved."* `[S]`

> **Mapping:** replace a single `p̃ > 0.7` cut with **two thresholds and a published undecided
> class**. This is the principled version of our radius/threshold-ladder protocol — the ladder
> currently reports *sensitivity* to an arbitrary cut; Fellegi–Sunter turns the cut into a declared
> **error-rate budget**.

**C. Post-linkage inference — the piece astronomy does not have at all.** Our users threshold, then
compute population statistics: precisely the "secondary analysis of linked files" problem.
- **Sadinle 2018** (`arXiv:1812.09590`) — **linkage-averaging**: two-stage propagation of linkage
  uncertainty into **population-size estimation**. Maps directly onto propagating membership
  uncertainty into cluster N and the mass function, and composes with any membership model. `[S]`
- Steorts, Tancredi & Liseo 2018 (`arXiv:1810.04808`) — joint model with feedback; shows the
  two-stage approach can carry bias the feedback removes. `[S]`
- Lahiri & Larsen 2005 (*JASA* 100(469):222–230) — classical bias correction for regression on
  linked data. `[S]`

**Prior astronomy uptake, honestly:** the Fellegi–Sunter *likelihood ratio* was already imported for
**source cross-identification** (Sutherland & Saunders 1992; NWAY, Salvato+2018). But it never reached
*membership*, NWAY validates against a **reference sample** (ground truth, not a truth-free
estimator), and the **post-linkage-inference literature has no astronomy analogue at all.** That third
point is the real gap. `[S]`

## 3. Forensic science — and what it says about Mecayotl `TRUTH-FREE` (discipline) / `SIM-ARM` (metrics)

**ENFSI Guideline for Evaluative Reporting**, v3.0 (2015), verbatim `[S]`:

> *"The role of the forensic practitioner is to consider the probability of the findings given the
> propositions that are addressed, **and not the probability of the propositions**."*
>
> *"**Transposing the conditional** — … a fallacious transposed conditional statement is one that
> equates (or confuses) the probability of particular findings given a proposition with the
> probability of that proposition given these findings."*

Enforced by a checklist item: **"Do the conclusions contain a transposed conditional?"** A one-line,
zero-cost governance import.

```{danger}
**This names a defect in Mecayotl, and sharpens our earlier finding.** At `mecayotl.py:1120`:

    p_c = 1 / (1 + exp(ln_prior_ratio + llk_field − llk_cluster))
    ln_prior_ratio = log(n_field / n_cluster) if use_prior else 0    # default: use_prior = False

With the default, `p_c = LR/(1 + LR)`. That is **not "no prior"** — it is a sigmoid of the log-LR with
**prior odds silently fixed at exactly 1**, i.e. an implicit assumption that any source is a priori
equally likely to be a member or a field star. It is then published on a 0–1 scale, called a
probability, and thresholded as if it were a posterior. ENFSI names this: the **transposed
conditional**.

**The direction is the finding.** The true term `log(n_field/n_cluster) > 0`, so restoring it
**lowers** `p_c`. **Omitting it systematically inflates every published membership probability.**
Scale for the Coma Ber run: ~302 members `[S]` against a classification pool of order 10⁷ ⇒
`n_field/n_cluster` of order 10⁵, i.e. **~12 in log-odds** `[I]` — publish it symbolically as
`log(n_field/n_cluster)`, not as a hard number, until the pool size is confirmed from the paper.

**And it stacks.** The decision rule `P_c + σ_Pc > p_t` (see `bayesian_membership_posterior.md` §3.7)
*promotes* uncertain stars — also anti-conservative. **Two independent anti-conservative biases
compounding in the same pipeline** is a far stronger, cleanly checkable result than either alone.

**Framing caution:** present this as *the published quantity is a likelihood ratio and its prior scale
is unstated* — a specification/documentation defect with a named precedent in another field. **Not**
as an accusation of error: `use_prior` is a real, deliberate switch.
```

**The repair, from the same literature:** report the LR explicitly and let users supply the prior
(ENFSI's rule, costs nothing); if a 0–1 number must be published, state the prior used; and
Morrison 2021 (`arXiv:2104.08846`) gives the standard recipe for turning an uncalibrated *score* into
a calibrated LR by logistic regression. `SIM-ARM` metrics for validating that: **C_llr**
(Brümmer & du Preez 2006) and the **ECE plot** (Ramos & González-Rodríguez 2013). `[S]`

## 4. The governance argument — a pincer, not an analogy

**Claim:** *Two applied fields that assign per-item probabilities and let downstream users threshold
them have made quantified error-rate reporting a condition of publication. Galactic star-cluster work,
which does the same thing, has no such standard — and 697 VizieR catalogues publish a membership
probability with essentially no reliability evidence.*

**Half 1 — the reporting precedent.** TRIPOD (Collins+2015, *BMJ* 350:g7594) and **TRIPOD+AI**
(Collins+2024, *BMJ* 385:e078378) make calibration a **numbered checklist item** `[S]`:
- **Item 12e:** *"Specify all measures and plots used … to evaluate model performance (eg,
  discrimination, **calibration**, clinical utility)"*
- **Item 23a:** *"Report model performance estimates with confidence intervals, **including for any
  key subgroups**"*

Adopt the vocabulary too: the **calibration hierarchy** — mean / weak / moderate / strong
(Van Calster+2016) — and *"Calibration: the Achilles heel of predictive analytics"*
(Van Calster+2019, *BMC Medicine*, ~1,800 cites). `[S]`

```{note}
**The convergence worth putting in the abstract:** TRIPOD+AI item 23a independently mandates
**subgroup** performance reporting — which is exactly the *conditional calibration* position already
identified as our strongest novelty (reliability binned by magnitude, crowding, radius). We are not
proposing an idiosyncratic diagnostic; we are proposing **the thing a consensus statement made
mandatory in another field.**
```

**Half 2 — pre-empts the obvious rebuttal.** *"Clinical medicine observes outcomes; we don't."*
Answer with proteomics: the **HPP Mass Spectrometry Data Interpretation Guidelines 3.0**
(Deutsch+2019) mandate FDR reporting with **1% protein-level FDR** as the operative threshold — and
proteomics has **no ground truth for individual identifications either.** It built the decoy machinery
and mandated the number anyway. `[S]`

**Paper-shaped form:** *"A reporting standard for probabilistic membership catalogues"* — a
TRIPOD-style checklist scoped to what is deliverable **without labels**: state the estimand (posterior
vs likelihood ratio), state the prior if used, report the threshold **and its error-rate budget**,
report calibration conditional on magnitude/crowding/radius, report the decoy-estimated FDP.
**Do not** write it as *"astronomy should validate against outcomes like clinicians do"* — that is
undeliverable and a referee will say so.

## 5. One small but load-bearing import `SIM-ARM`

**Bröcker & Smith 2007** (*Weather and Forecasting* 22(3):651–661) — **reliability diagrams with
consistency bars**, answering *"is this deviation from the diagonal significant given finite bin
counts?"* Directly load-bearing for a **321-source** catalogue: a per-decile reliability diagram has
~32 stars per bin and **will look miscalibrated from noise alone**. One figure-level import. `[S]`

## 6. Not worth it — with reasons

| Field / artefact | Verdict | Why |
|---|---|---|
| **Ecology — occupancy models** | **Does not transfer.** Keep one sentence. | MacKenzie+2002 separates occupancy from detection **only via repeated visits to the same site**. We observe each star once; the mcs sweep is **not** a replicate design (nested sweeps over one dataset, not exchangeable observations), so the identifiability that makes occupancy work is absent. **Keep only:** naive occupancy is biased low without a detection model → naive member counts above threshold are biased, and downstream statistics inherit it. |
| Clinical — decision curve analysis | cite the *reframing*, skip the metric | Net benefit needs outcomes. But free and worth a paragraph: **a threshold is a utility statement** — `p̃ > 0.7` encodes an implicit exchange rate between contaminants and lost members. Essentially zero astronomy uptake. |
| Clinical — calibration slope/intercept | `SIM-ARM` | Needs outcomes. Adopt the *vocabulary*, don't promise the measurements. |
| Forensic — C_llr, ECE plot | `SIM-ARM` | Need labelled validation sets; add nothing over Brier/log-loss in the sim arm. |
| Credit scoring, election forecasting | skip | Validated against observed outcomes; no truth-free estimator. |
| Weather beyond consistency bars | skip | Murphy decomposition / CRPS already covered in `bayesian_membership_posterior.md` §5b. |

## 7b. Within astronomy: gravitational-wave `p_astro`

GW is the most mature "calibrated astrophysical probability" in astronomy — LVK publishes one per
candidate and the community thresholds on it at 0.5.

```{important}
**The foundational `p_astro` paper already ran our problem.** Farr, Gair, Mandel & Cutler 2015
(`arXiv:1302.5341`, `2015PhRvD..91b3005F`) — the FGMC framework — contains a worked section
**"Star Cluster Parameters With Background Contamination"**: a Plummer foreground on a linear-gradient
background, **1,000 cluster stars against 10,000 field stars**, membership flags **marginalised
analytically**, sampled with emcee. Verbatim `[S]`:

> *"Because the peak density of the cluster is equal to the background density at the center of the
> domain, **there is no single star in the domain that is more likely to be a cluster member than a
> background star** (i.e. ⟨gᵢ⟩ ≲ 0.5 for all stars); nevertheless, we will see that our method
> provides good constraints on the cluster parameters."*

**Two consequences.** (1) A decade-old, citable precedent for **calibrated inference when no
individual object is confidently classified** — exactly the regime a referee will call speculative.
(2) It is **not a scoop**: a synthetic toy demonstrating *parameter* recovery, with **no calibration
check on the per-star ⟨gᵢ⟩**. It strengthens our framing rather than threatening it.
```

**Calibration status — three tiers, and the top one is empty** `[S]`:

| Tier | What exists | Limit |
|---|---|---|
| 1 | The **aggregate sum rule** — Σ`p_astro` over candidates >0.5 vs their count (GWTC-3 estimates ~10–15% contamination) | **Cannot fail.** Ashton+2024: it *"formally amounts to the posterior-estimated number of foreground events in the Farr et al. framework"* — near-tautological for a fitted mixture |
| 2 | **Ashton+2024 Fig. 17** — purity vs `p_astro` threshold against injected truth. *"all pipelines under-estimating the actual purity"* | **Cumulative above threshold, not binned**, and conditioned on exactly **one** covariate (pipeline) |
| 3 | Disjoint-bin reliability diagram; validation conditioned on SNR, chirp mass, network, source class | **Genuinely absent** |

**The sign of the miscalibration is unresolved** — Ashton+2024 says `p_astro` is too *low*;
Banagiri+2023 calls their own O3a values *"likely an overestimate."* A binned diagram would settle it.

**Failure modes worth importing as predictions to test** `[S]`:
- **Pipeline disagreement is large:** of 28 GWTC-2.1 candidates found by ≥2 pipelines, **≥7 have a
  `p_astro` spread > 0.5** — GW190413_134308 is MBTA 0.99 / GstLAL **0.04** / PyCBC 0.48. Published
  values for GW151216 span **0.03–0.71** across groups.
- **Their own stated bias condition:** *"may be biased if this distribution deviates significantly
  from the (unknown) true signal distribution. The risk of such bias is largest for regions of
  parameter space containing few, or zero, confirmed detections."* → **predict our miscalibration is
  worst where the training population is thinnest: faint G, crowded fields, sparse clusters.**
- **Sub-class priors feed back into the membership probability:** GW190917_114630 was classified BBH
  at p = 0.77, but parameter estimation found NSBH masses — *"Had it been classified as an NSBH to
  begin with… the resulting pastro would not have made the threshold of 0.5."* **The probability of
  being astrophysical at all depended on the assumed sub-class.** That is precisely our
  binaries / tidal-tail / cluster-like-contaminant problem.
- Andres+2022 budgets **±0.15 from parameter-space binning alone** — a ±0.1 systematic on a quantity
  whose decision threshold is 0.5.

**The asymmetry that makes this our opening** `[I]`: **LVK cannot re-inject synthetic populations into
a rerun observing run. We can** — into real offset fields carrying true Gaia error structure and
crowding.

```{warning}
**Do not pitch FGMC as a drop-in port.** It assumes each event is an independent draw in a **scalar**
ranking statistic. Gaia gives per-star **heteroscedastic, correlated 5×5 covariance** plus spatially
correlated systematics on ~0.1–1° scales. Convolving each mixture component with each star's own
covariance — and deciding whether field-correlated systematics break the independent-Poisson
assumption — **is the real methodological contribution.**
```

## 7. Bottom line

Two practices survive having no labels: the **decoy/entrapment FDP estimator** (proteomics, with
Robach 2025 as an independently-derived twin built for low contrast) and **post-linkage uncertainty
propagation** (Sadinle's linkage-averaging) for the downstream population statistics. The **ENFSI
rule** is a free governance import that turns the Mecayotl finding into a named, precedented defect.
**TRIPOD+AI** is the paper-shaped governance argument — scoped to labels-free reporting items and
paired with proteomics to pre-empt the "you have no outcomes" rebuttal.

## Verify before drafting `[I]`
- The Coma Ber classification-pool size (order 10⁷) was reconstructed from our own design note, not
  read from the paper. Confirm before printing any log-odds figure.
- The "no OC paper counts decoys above threshold" claim rests on a **bounded** ADS search (5 hits,
  none methodological); two confirming queries timed out and should be re-run.
- The proteomics governance claim is sourced to **HPP Guidelines 3.0 Guideline 4**; the stronger
  "journals reject papers without FDR reporting" phrasing was **not** verified — use the
  guideline-document framing.
