# Calibration theory — what our diagnostics actually prove

_2026-07-27. The statistical theory under the membership-calibration plan, including **where the
theory says the plan is weaker than we claimed**. `[S]`=verified, `[I]`=inferred._

## 0. The gate on everything

**Every diagnostic here needs labelled outcomes.** OC membership is **latent**. So:

- On **synthetic** field+cluster mixtures the full toolkit applies (truth by construction).
- On **real NGC 6383** the only label source in the current plan is P07's youth-indicator recovery vs
  `pMember` — incomplete, **and not independent of the astrometry that built p̃**.

```{danger}
**A synthetic-only reliability diagram must NOT be presented as a calibration claim about the
published catalogue.** State which arm every number comes from.
```

## 1. What each diagnostic actually tests

| Diagnostic | Tests | Not |
|---|---|---|
| **Reliability diagram** (binary) | **auto-calibration**, `E[Y\|p̂]=p̂` — for binary Y this *is* all of calibration | a PIT histogram |
| **Aggregate proper score** (Brier, log-loss) | a *confounded* mixture of reliability and resolution | calibration |
| **SBC** | that the **sampler** draws from the posterior implied by prior+likelihood | model adequacy |
| **Conditional (binned) reliability** | **grouping loss** `GL(p)=Var[f*(X)\|f(X)=p]` | calibration loss |

**Aggregate scores cannot substitute for a calibration check** — Bröcker 2009 eq. (13): for *any*
strictly proper score the decomposition carries reliability and resolution with **opposite signs**,
so a reliability gain can be offset by a resolution loss. Verbatim: *"it is impossible to say how the
score will rank unreliable forecast schemes… The lack of reliability of one forecast scheme might be
outbalanced by the lack of resolution of the other."* `[S]`

```{important}
**Corollary worth printing: `UNC = e(π̄)` depends only on the contamination fraction, so Brier and
log-loss are NOT comparable across clusters with different field contamination.** Any cross-cluster
table of Brier scores is measuring contamination, not method quality.
```

**The calibrated-but-useless forecaster** `p̂ ≡ π̄` has REL = 0 *and* RES = 0. What excludes it is not
hand-waving: every forecast is sufficient for it, so it is weakly dominated by every calibrated
forecaster under *every* strictly proper score (DeGroot–Fienberg refinement order). `[S]`

**SBC's own authors bound its scope** — Talts+2018: *"limited exclusively to the computational
aspect… no guarantee… that the model will be rich enough to capture the truth at all."* Model checking
is PPC's job. And per Modrák et al., naive SBC *"could never detect large classes of problems
including when the posterior is equal to the prior"* → use **data-dependent** test quantities, and
test `p_i` explicitly since it is a *derived* quantity. `[S]`

## 2. Use these instead of ECE and Hosmer–Lemeshow

1. **CORP reliability diagram** (PAV-based, **binning-free**) with 90% bands — Dimitriadis, Gneiting
   & Jordan 2021, *PNAS* 118(8). State consistency vs confidence bands, and note the
   **exchangeability assumption** (Gaia field stars are spatially correlated). `[S]`
2. **Brier = MCB − DSC + UNC.** **MCB replaces ECE** as the headline scalar (≥0, zero iff calibrated,
   *exact*); DSC separates "miscalibrated" from "non-discriminating".
3. **Van Calster hierarchy** as vocabulary — mean / weak / moderate / strong — plus calibration
   intercept and slope with CIs. **Claim *moderate* calibration**, not more.
4. **Grouping-loss estimator** (Perez-Lebel+2023) for the conditional axis.
5. Formal test only if wanted: adaptive **T-Cal** or kernel calibration error.
6. **Sample size:** ≥200 members *and* ≥200 non-members for a flexible curve.

```{warning}
**Drop Hosmer–Lemeshow.** `erotica/calibration.py` implements it. Van Calster+2019 recommends against
it; Paul, Pennell & Lemeshow 2013 (the co-author's own paper): *"in very large data sets, small
departures … will be considered significant"*; and it carries the same ad-hoc binning instability as
binned ECE. **Keep the function for backward compatibility, deprecate it in the docs, and do not use
it as the headline statistic.**

**And get the ECE bias direction right** — binned ECE is a **lower bound** on the population
calibration error (Kumar+2019 Prop 3.3; their Ex 3.2 has `CE(f_B)=0` while `CE(f)≥0.49`) while
carrying an **upward** finite-sample bias ~B/n. Writing "ECE is biased downward" flat **will get
caught**.
```

## 3. Multicalibration — right idea, wrong technical frame

Our "conditional reliability binned by magnitude/crowding/radius" is **conceptually** the
multicalibration idea (Hébert-Johnson+2018, `arXiv:1711.08513`), and that is the right citation for
*why marginal calibration is insufficient*. But `[I]`:

- Use `C` = **union of three marginal binnings**, *not* the product partition — product cells are
  empty below n≈10⁵, and the union makes groups **overlap**, which is exactly where the definition has
  content (fixing magnitude panels can break radius panels).
- Every theorem targets a `C` too rich to enumerate; with ~20 groups we check 20. `|C|` enters sample
  complexity only **logarithmically** — the binding term is `α⁻⁶`, and those bounds are **vacuous at
  our n**. Do not quote them as certification.
- **Do not claim omniprediction.** "One probability serves all thresholds" is classical decision
  theory (the post-processor is literally the Bayes action); the actual theorem gives competitiveness
  with `min_{c∈C}`, weaker than readers will infer.

```{attention}
**Terminology trap.** In the Gneiting school *"conditional calibration"* / *"auto-calibration"* means
conditioning on the **forecast value**; in CS *"group-conditional"* means conditioning on
**covariates**. Say **"group-conditional"** or **"covariate-conditional"**, or half the referees read
it the other way.
```

**Unaddressed anywhere:** simultaneous inference across the panels. No off-the-shelf group-conditional
calibration bands exist → use Holm–Bonferroni or a **max-statistic bootstrap** across strata, and say
plainly that it is classical multiple-testing, not a new method.

## 4. Identifiability — publication-ready caveat

`p_i` is well-posed only once a **labelling convention** is fixed, and even then it is *identified
without being well-determined*:

- Gaussian mixtures are identifiable (Yakowitz & Spragins 1968: linear independence over ℝ is
  **necessary and sufficient**) but only **up to permutation**. `p_i` is not permutation-symmetric, so
  it is identified only relative to a convention naming the cluster component. `[S]`
- **Stephens 2000 is stronger than "ambiguous":** under a fully exchangeable prior the exact marginal
  classification probability is **identically 1/k for every star** — *"thus useless for clustering the
  observations into groups."* `[S]`
- Break symmetry with **physically informative priors** (not post-hoc ordering constraints — that is
  precisely what Stephens attacks). **The cost: the anchor becomes part of the *definition* of `p_i`
  and must be reported.**
- Identification bounds nothing about precision: as components merge the Fisher information degenerates
  (**exactly zero** in Chen 1995's construction), and **Heinrich & Kahn 2018 correct Chen** — the local
  minimax rate is `n^(−1/6)`, *slower* than Chen's `n^(−1/4)`. Multivariate location-scale Gaussian
  mixtures are only **weakly identifiable** (Ho & Nguyen 2015). `[S]`
- In heavy overlap `p_i → π` for every star: **still calibrated, carrying vanishing information.**

## 5. ⚠️ Where the theory says our plan is weaker than claimed

1. **Ground truth** (§0) — gates everything.
2. **Circularity.** Calibrating against simulations drawn from the same family we fit is close to an
   **SBC self-consistency check** and will look good near-vacuously. → Simulate from a *different*
   process (non-Gaussian field PM tails, spatially varying extinction, unresolved binaries, Gaia
   completeness) and report calibration **under misspecification**.
3. **Calibration is nearly automatic when the model is right and parameters known** — there `p_i` *is*
   the true conditional probability by construction. Guard against this tautology: pair every
   reliability diagram with **DSC and grouping loss**, so a good curve cannot be read as "the method
   works."
4. **A single-threshold member list barely needs calibration.** Perez-Lebel+2025 Prop 3.3: if the
   calibration curve `c` is monotone, thresholding the **raw, miscalibrated** p̂ at `t = c⁻¹(t*)` gives
   **zero calibration regret**. Scope conditions that stop this deflating the whole programme: you must
   estimate `c` either way (same cost), `c` must be monotone, and threshold-shifting **cannot** touch
   grouping loss. **Honest framing: miscalibration is cheap for one fixed threshold, and expensive for
   (i) reporting honest purity and (ii) reuse at many thresholds.**

```{important}
## 6. The programme's real justification — use this, not "calibration is good practice"

The cluster+field mixture **is** Efron's two-groups model: `1 − p_i` is the **local false discovery
rate**, and the expected contamination of a published member list `S` is

> `E[1 − p_i | i ∈ S]`

**That number is only honest if `p_i` is calibrated *conditional on selection*.** Every catalogue that
publishes a member list is implicitly making this claim. That is a far stronger argument than
"calibration is good practice", and it ties directly to the conformal/FDR line and to the
decoy-FDP estimator in `cross_domain_calibration.md`.
```

## Sourcing honesty
Murphy 1973, DeGroot–Fienberg 1983 and Schervish 1989 have their **theorem statements taken from
restatements** (Dimitriadis–Gneiting–Jordan, Bröcker, and Gneiting–Raftery Thm 3 respectively) — the
originals are paywalled or scanned. **Cite as "X, as restated in Y."** Two arXiv/journal title
mismatches to watch: Gopalan et al. omnipredictors (`arXiv:2109.05389`, arXiv source is a **stale
draft** — use LIPIcs/ar5iv) and DGJ 2021 (`arXiv:2008.03033` carries a *different title* from the PNAS
version).
