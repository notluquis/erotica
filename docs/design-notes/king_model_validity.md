# Is a King profile the right model for NGC 6383?

_Written 2026-07-27, **substantially revised the same day** after adding Hunt & Reffert (2024) to the
comparison — three claims in the first version did not survive it, and are retracted explicitly
below. Everything numeric is produced by `tools/validation/king_literature_context.py`, which fetches
the catalogues live from VizieR._

`structure.king_unbinned` makes the King fit statistically sound. It does not answer whether a King
profile is what NGC 6383 wants, nor whether the resulting `R_t` is a physically admissible number.

## The decisive comparison: NGC 6383 is *in* Hunt & Reffert 2024

Hunt & Reffert (2024, A&A 686, A42, `2024A&A...686A..42H`) is all-sky, covers 5647 bound open
clusters with completeness-corrected photometric masses, and includes **NGC 6383 itself** — an
independent measurement rather than a population to extrapolate into.

| | Hunt & Reffert 2024 | P01 |
|---|---|---|
| distance | 1101 pc | 1110 pc |
| age | 4.1 Myr (log 6.61) | 3.5 Myr |
| mass | 902 M⊙, P(bound) = 1.000 | — |
| core radius | 0.80 pc = 2.5′ | 0.63 pc = 1.96′ |
| half-number / half-light | 1.30 pc = 4.0′ | 1.94 pc = 6.02′ |
| **Jacobi radius `r_J`** | **12.14 pc = 37.9′** | Hill 33.6′, `T_max` 42.45′ |

Distance and age agree closely, and the core radii agree to well within the scatter of the methods.

```{admonition} The finding that replaces the first version's field-of-view argument
:class: important
**P01's adopted King outer radius, `R_t` = 54′ = 17.4 pc, is 1.43× the Jacobi radius.**

A gravitationally bound cluster cannot extend beyond its Jacobi (Roche) surface — that is the
definition Hunt & Reffert use to separate bound clusters from moving groups. So the fitted `R_t` is
**not an admissible physical boundary**, independently of how well the fit converged.

This is not a contradiction of P01, which states plainly that `R_t` "should be regarded as a
model-dependent scale rather than as an independently measured physical boundary". It supplies the
independent number that makes that statement quantitative.

It also explains the unbinned fit's behaviour better than the first version's explanation did. `R_t`
is unconstrained from above not because the survey ran out of field, but because **the fitted
truncation lies outside the radius at which the cluster can still be bound**: the membership
selection extends to the field edge with no density turnover inside the physical range, so there is
no truncation for the profile to find.
```

```{warning}
**Retraction 1 — "the extraction is too small by a factor 2.2".** The first version argued that
because Tarricq+2022 search to 50 pc (155′ at this distance) while P01's largest field is 70′, the
field was inadequate. That reasoning was wrong: 50 pc is Tarricq's *search convention*, not a
property of this cluster. Against NGC 6383's actual Jacobi radius, **70′ = 22.4 pc = 1.85 × r_J** —
the field comfortably encloses the bound cluster. Re-extracting at 155′ would add field stars, not
information about `R_t`.
```

## Is NGC 6383 younger than the comparison population?

| sample | clusters with radii | youngest | younger than NGC 6383 |
|---|---|---|---|
| **Hunt & Reffert 2024** | 5647 | **2.9 Myr** | **12** (346 below 10 Myr, 1470 below 50 Myr) |
| Tarricq+2022 (King fits) | 233 | 50 Myr | 0 |
| Zhong+2022 | 256 | 10 Myr | 0 |

```{warning}
**Retraction 2 — "NGC 6383 is younger than every cluster in both reference samples, so any
comparison is extrapolation".** True of Tarricq and Zhong, and false as a general statement. Hunt &
Reffert contain 12 bound open clusters younger than NGC 6383 and 346 below 10 Myr. A young
comparison population exists; Tarricq simply does not sample it, being solar-vicinity limited.
```

```{warning}
**Retraction 3 — the `R_c` "tension".** The first version noted that NGC 6383's core sits at the 3rd
percentile of Tarricq's distribution while Tarricq report that *younger* clusters have *larger*
cores, and flagged this as needing explanation. The tension was an artefact of comparing a 3.5 Myr
cluster against a population whose youngest member is 50 Myr. Restricted to Hunt clusters younger
than 5 Myr (n = 100), `r_c` runs 0.77 / 2.41 / 5.52 pc at 16/50/84, and NGC 6383's 0.63 pc sits at
the **12th percentile** — compact, but unremarkable for its age. Nothing needs explaining.
```

## What survives from the first version

Both of these are independent of the retractions above.

**Circular symmetry, which the King fit assumes, is the exception.** Tarricq fit elliptical Gaussian
mixtures alongside King profiles. For the core component: median `b/a` = **0.71**, with **92.9%**
below 0.9 and **68.9%** below 0.8. This matches Olivares et al. (2018, `2018A&A...612A..70O`), who
found strong evidence against radially symmetric models relative to elliptical extensions.

At NGC 6383's age the expectation is stronger. Pang et al. (2022, ApJ 931, 156,
`2022ApJ...931..156P`) classify substructure outside the tidal radius and find it is **filamentary or
fractal below 100 Myr** — halo and tidal-tail morphologies appear only above 100 Myr. They report
that for systems ≲30 Myr, axis ratio, mass and half-mass radius all *increase* with age, attributed
to filament dissolution and subgroup mergers. A 4 Myr cluster is a hierarchical, still-assembling
structure, not a relaxed tidally-truncated sphere.

**A single King `R_t` is not how the field describes outer structure.** Zhong et al. (2022, AJ 164,
54, `2022AJ....164...54Z`) searched 256 OCs to 50 pc and found "the radial density profile in the
outer region for most open clusters deviates from the King profile", replacing it with a King core
plus a log-normal halo and **four** radii. What P01 calls the King outer radius is, in current
practice, the boundary of the *core component* only.


## The Jacobi prior, applied

`king_unbinned` accepts `tidal_prior=(mu, sigma)`. Using Hunt & Reffert's `r_J` = 37.9′
(`tools/validation/king_jacobi_prior.json`; 4 chains, 2000 draws, R-hat = 1.000, bulk-ESS ≥ 2095,
**zero divergences** in every run):

| fit | `R_c` (′) | `R_t` (′) | `R_t`/`r_J` |
|---|---|---|---|
| published (binned, `Uniform(R_c, 1.5·T_max)`) | 1.384 ± 0.039 | 53.9 ± 8.8 | 1.42 |
| unbinned, scale-free, full 70′ | 1.324 ± 0.209 | 80.2 ± **7114** | 2.11 |
| unbinned, Jacobi ±20% | 1.497 ± 0.214 | 47.40 ± 6.01 | 1.25 |
| unbinned, Jacobi ±50% | 1.428 ± 0.207 | **59.08 ± 11.43** | 1.56 |
| unbinned, scale-free, **only `r ≤ r_J`** | 1.325 ± 0.210 | 85.9 ± **2661** | 2.27 |

**`R_c` is robust.** 1.32–1.50′ across every likelihood and prior, against a published 1.384′.
Nothing in this work moves it.

**`R_t` is prior-determined, and P01's value is defensible.** With a weakly informative Jacobi prior
(±50%) the unbinned fit returns 59.1 ± 11.4′, consistent with the published 53.9 ± 8.8′ within 1σ.
So the published number is *not* an artefact of the Normal-on-binned-densities likelihood. It is,
as P01 states, conditional on its prior — with a scale-free prior the posterior SD is ~7000′.

Note also that the data pull *upward*: with the prior centred on `r_J` = 37.9′ and a 20% width, the
posterior median lands at 47.4′, about 1.25σ above the prior mean.

```{admonition} The result that argues hardest for EFF
:class: important
**Restricting the fit to stars inside the Jacobi radius does not help.** With only the 355 members
at `r ≤ r_J`, `R_t` is *still* unconstrained (85.9 ± 2661′). Within the region where the cluster can
actually be bound, the radial profile shows **no tidal truncation at all** — the defining feature of
a King profile is simply not present in the data.

That is the expected state for a 4 Myr cluster (Pang et al. 2022: filamentary/fractal below
100 Myr), and it is the argument for EFF (`1987ApJ...323...54E`), which has no tidal cutoff, rather
than for a better-constrained King fit.
```

**43% of the p > 0.6 "members" lie beyond the Jacobi radius** — 272 of 627 within 70′. They are
either genuinely unbound halo/tail stars or membership contamination; either way a bound-cluster
King profile cannot describe them, and they are what drags `R_t` past `r_J`. P01 already declines to
classify the wide-field additions as confirmed halo members or contaminants; this quantifies how
much of the fitted sample is at stake.


## Model comparison: King vs EFF vs Plummer

`compare_radial_profiles` fits each family with sequential Monte Carlo and compares log marginal
likelihoods. SMC is used because the models are **not nested** — EFF has no `R_t` at all, so a
likelihood-ratio test does not apply — and because the point-process likelihood is a single
`Potential` over the whole field rather than a sum of exchangeable per-star terms, so there is no
clean pointwise decomposition for LOO/WAIC to leave one out of. This is the approach Olivares et al.
(2018) use for the Pleiades.

NGC 6383, 627 members within 70′, 2000 draws × 4 chains:

| model | log Z | chain sd | ln B vs best | verdict (Kass & Raftery, on 2 ln B) |
|---|---|---|---|---|
| **King** | **806.37** | 0.029 | 0.00 | best |
| EFF | 805.54 | 0.086 | −0.82 | *not worth more than a bare mention* |
| Plummer (`γ = 4`) | 796.45 | 0.039 | **−9.92** | **very strong** evidence against |

```{admonition} Everything in this note is one phenomenon
:class: important
The EFF fit returns **`γ = 2.324 ± 0.213`** — **1.5σ from `γ = 2`**, and 7.9σ from the Plummer value
of 4.

`γ = 2` is exactly the limit in which **King with `R_t → ∞` *is* EFF**: the King edge term
`c = (1+(R_t/R_c)²)^(-1/2)` vanishes and `Σ → k/(1+(r/R_c)²)`. NGC 6383 therefore sits essentially
*inside the overlap* of the two families, and three results that looked separate are the same fact:

* King and EFF are statistically indistinguishable here (ln B = 0.82) — **because at `γ ≈ 2` they are
  the same model**;
* the King `R_t` cannot be constrained from above — **because the data want `R_t → ∞`, which is that
  same limit**;
* no tidal truncation appears even inside the Jacobi radius — **because there is none to find: the
  profile is a cored power law of slope ≈ 2.3 all the way out**.

Plummer is excluded outright, so this is not a case of the data being uninformative about the
profile family. They are informative, and what they say is *"cored power law, slope ~2.3, no
truncation"*.
```

**Consequence for reporting.** Quoting a King `R_t` for NGC 6383 asserts a truncation the data do
not show, and its value is set by the prior. The defensible summary of the outer structure is the
EFF slope, `γ = 2.32 ± 0.21`, with a scale radius `a = 1.65 ± 0.38′`, plus the Jacobi radius
(12.14 pc = 37.9′) as the physical boundary from the mass. That is a statement the data support.


## Selection function: what it can and cannot say here (in progress)

`king_unbinned(completeness=)` folds a radial detection probability into the normalisation, and on
synthetic data ignoring one inflates `R_c` by 50% and halves the central density. Applying it to the
real cluster is `tools/validation/ngc6383_selection_function.py`. Two things are settled; one is
blocked.

**Settled, and resolution-independent: magnitude incompleteness is not the dominant effect here.**
The Gaia DR3 selection function at this position is correctly magnitude-sensitive —

| G | 10 | 14 | 17 | 19 | 20 | 20.7 | 21 | 21.5 | 22 |
|---|---|---|---|---|---|---|---|---|---|
| S | 1.000 | 1.000 | 1.000 | 1.000 | 0.996 | 0.852 | 0.376 | 0.0002 | 0.000 |

— and NGC 6383's members have median `G = 17.3` with a 98th percentile near 20.4, i.e. they sit
where DR3 source completeness is essentially unity.

```{admonition} The uncomfortable corollary
:class: important
If Gaia's own completeness is ~1 for this sample, then **the largest magnitude-dependent selection
acting on it is ours**. The pipeline's 2σ parallax clip retains **99% of the brightest `Gmag`
quartile and 27% of the faintest**. That is an order of magnitude larger than anything the survey
does in this field.

The correction that matters for NGC 6383 is not the survey's. It is the pipeline's.
```

```{warning}
**`mode='hpx7'` cannot answer the radial question at all, and its flat answer is not evidence.**
Its healpix pixels are **27.5′** across; `R_c` is **1.38′**, so the core is **0.792%** of a single pixel
and the whole 70′ field spans 2.55 pixels. Radial structure at cluster scales is below the map's
resolution *by construction*. A flat `S̄(r)` from hpx7 means **"invisible to this map"**, not
"absent" — and crowding-driven incompleteness, which acts precisely in the core, is exactly what it
cannot see.

The first pass here did run hpx7 and did return a flat 0.998. That number is recorded as a
diagnostic only (`ngc6383_selection_function_hpx7.npz`) and must not be used as a correction.
```

```{admonition} Resolved 2026-07-27 by an independent audit — the correction is negligible here
:class: note
`mode='multi'` is **precomputed and archive-free**, and reaches healpix order 10 (**3.44′**), which
does resolve radial structure across the field. Running it on NGC 6383 gives a **real but negligible**
gradient. **The number first recorded here, 0.258%, was a resolution artefact and is superseded:**
order 10 (3.44′) resolves the field but not the core. Running `mode='patch'` (order 6–12, 0.86′) on
the identical sample gives **1.156%** — a **4.5× under-read**. NGC 6383's `M_10` profile has a
genuine centred depression of **0.36 mag** (20.474 at centre → 20.838 at 70′) that order 10 largely
misses, reading 20.724 at the centre.

Against the note's own synthetic benchmark — computed exactly, not estimated: the repo's
`_crowding_completeness` gives a **50.44%** core suppression under this aperture — a 1.2% gradient
moves nothing. Fitting with the order-12 `S̄(r)` shifts `R_c` by **−0.01σ**. **For this cluster the
Gaia DR3 selection-function correction to the radial profile is not needed** — but the *number* to
quote is 1.156%, not 0.258%. `mode='patch'` (order 12, 0.86′) would still resolve inside
the core and remains worth running when the ESA archive is up, but the expected payoff is now small.

This does not weaken the corollary above — it strengthens it. Gaia's completeness is flat *and* ~1
for this sample, so the selection acting on it really is the pipeline's own.
```

**Still to do:** `mode='patch'` (order 12, ~0.86′) needs a live ESA Gaia archive query; the archive
was in a maintenance window on 2026-07-27 (HTTP 500, then a failed SSL handshake). The script is
written and its output is aligned to the 256-node quadrature grid the weighted normalisation uses.


## The selection that actually acts on this sample is the pipeline's, not the survey's

A causal reading of the pipeline — written as a DAG rather than as a likelihood — makes a prediction
the likelihood formalism does not naturally surface, and the prediction is measurable.

**The chain.** The cone query cuts on *parallax*, so nominally it is not a radial cut. But the cut
bites through `σ_ϖ`, which grows with `G`, and `G` correlates with radius. Measured on the 627
members: **Spearman ρ(G, r) = +0.150, p = 1.6×10⁻⁴**, with the faintest `G` quartile sitting at a mean
radius of 39.1′ against ~29′ for the others. **A non-radial cut inherits a radial gradient.**

**The size of it.** Reconstructing the pre-cut radial distribution by Horvitz–Thompson weighting
(`w_i = 1/ret(G_i)`; averaging retention over survivors would be circular, since survivors are the
cut's output):

| `r` bin | N observed | N reconstructed | `S_pipe` |
|---|---|---|---|
| 0.10–3.75′ | 105 | 175.0 | **0.600** |
| 3.75–13.76′ | 104 | 184.8 | 0.563 |
| 13.76–30.63′ | 104 | 205.9 | 0.505 |
| 30.63–47.08′ | 105 | 229.3 | 0.458 |
| 47.08–61.43′ | 104 | 240.3 | **0.433** |
| 61.43–69.99′ | 105 | 221.0 | 0.475 |

**Core → 47–61′: `S_pipe` falls 27.9%. Over the same span `S_Gaia` falls 0.80%. The pipeline's own
selection is ~35× the survey's.** The 40′ sample gives 28.5%, agreeing to ~1 pp.

```{admonition} The misallocation this exposes
:class: danger
`king_unbinned` has a validated `completeness=` hook, and everything written about it in this
programme wired it to **the survey selection function — the 1.16% term.** Nothing was wired to the
pipeline's own cut, the 27.9% one. A correctly specified likelihood, fitted with the wrong `S`.

Nothing in the likelihood formalism flags that a cut on parallax is a cut on radius. The DAG does,
through `C ← ϖ_meas ← σ_ϖ(G) ← G ⇠ R`.
```

**And yet the fit barely moves.** Refitting with `completeness=S_pipe`:

| fit | `R_c` | `R_t` |
|---|---|---|
| naive | 1.326 ± 0.205′ | 81.4 ± 1532′ |
| `S_pipe` corrected | 1.345 ± 0.213′ | 81.2 ± 774′ |
| shift | **+1.4% = +0.09σ** | −0.3% |

The **direction is the one the DAG predicts** (outskirt suppression biases `R_c` low, so correcting
raises it), and the **magnitude is what the analytic result predicts**: to first order
`δθ = ε I⁻¹v`, the `λ₀`-weighted regression of the completeness gradient onto the score functions, so
only the component of `S(r)` *degenerate with a parameter direction* biases anything. A smooth
outward decline is nearly orthogonal to the core curvature that sets `R_c`.

**A 35× larger selection gradient produces a 0.09σ parameter shift. Gradient magnitude is not bias
magnitude** — which is the whole reason the analytic criterion is worth having rather than a
rule of thumb about how big `S` looks.

```{note}
One selection remains **not correctable**. HDBSCAN membership runs in proper-motion space, and for an
expanding cluster the outer stars carry larger outward proper motions, so they sit further from the
PM centroid and are preferentially lost. That path needs a model for the internal velocity field —
which, for a young expanding cluster, **is the quantity one is trying to measure.** The survey
selection function cannot fix it: `gaiaunlimited` models catalogue detection, not pipeline retention.
That is a genuine identifiability statement, and it is the one the likelihood framing does not
produce on its own.
```


## Substructure: a control the profile fits need, and a Q-based justification that does not work

```{danger}
**The reported credible interval on `γ` is probably 2–3.5× too narrow.** Injection–recovery on the
NGC 6383 geometry (N = 628, 70′ field, EFF `γ = 2.32`, `a = 1.65′`), with a fraction of stars moved
into clumps *whose centres are drawn from the same EFF profile* — so the azimuthally-averaged radial
profile is unchanged in expectation and any shift is attributable to substructure alone:

| configuration | Q | `γ` recovered | reported σ | realization scatter | 1σ cov. | 2σ cov. |
|---|---|---|---|---|---|---|
| smooth control | 1.262 | 2.425 | 0.089 | 0.048 | 40% | **93%** |
| 50% in 15 clumps, σ = 1′ | 1.007 | 2.586 | 0.120 | 0.248 | 20% | **33%** |
| 50% in 8 clumps, σ = 2′ | 1.03–1.12 | 3.008 | 0.187 | **0.647** | 20% | **20%** |

Nominal coverage is 68% / 95%. **The likelihood is well calibrated when the data really are a Poisson
point process** (smooth control: 93% at 2σ), so the failure is caused by substructure, not by the
code — clumping is over-dispersion the point-process likelihood does not model.
```

**The bias direction is favourable, and that matters.** It is **upward** in both `γ` and `a`.
Substructure cannot manufacture a spurious `γ → 2`. So if the fitted `γ = 2.32 ± 0.21` is biased at
all, the **true slope is even closer to the King/EFF degeneracy** — this is a control the argument
needs, not a threat to it.

```{warning}
**The Q parameter cannot justify fitting a radial profile, and P01 uses it that way.** Two
independent failures, both measured rather than argued:

1. **Q is degenerate with contamination.** A perfectly smooth King cluster (`R_c` = 1.38′,
   `R_t` = 54′, N = 628 in 70′) with **zero substructure**, diluted by uniform field, gives
   Q = 1.251 → 1.053 → 0.957 → 0.898 → **0.850** at field fractions 0 → 0.3 → 0.4 → 0.5 → 0.6. The
   observed Q = 0.833 is **fully reproduced by ~60% uniform dilution of a perfectly smooth cluster.**
2. **Q is not monotonic in the quantity of interest.** Across clumped simulations, *higher* Q goes
   with *larger* `γ` bias: Q = 0.986 → +0.29, Q = 1.032 → +0.56, Q = 1.148 → **+1.01**. The bias
   tracks clump scale relative to `a`, which Q does not encode.

So "compute Q; if it exceeds 0.8, the cluster is centrally concentrated and a radial fit is
appropriate" is invalid **regardless of where the boundary is placed**. A calibration caveat
compounds it: an independent implementation gives Q = 0.72–0.735 for a uniform random disc against
Cartwright & Whitworth's quoted 0.79–0.80 — a ~0.07 offset — while reproducing P01's published value
exactly, so P01's comparison to the 0.8 boundary rests on a zero-point that implementation does not
reproduce.
```

```{note}
**Two corrections to what this note previously implied about Pang et al. 2022.**
Their f1/f2/h/t classification is explicitly **qualitative** — *"we qualitatively classify them into
four types"*, by eye, with no Q, MST, fractal dimension or correlation function anywhere in the
paper. There is no statistic separating filamentary from fractal. And **19 of their 60 clusters below
100 Myr (32%) host no extended substructure at all**, so "below 100 Myr the morphology is
filamentary/fractal" is not what they found. NGC 6383 is also **not in their sample** — 85 clusters
within ~500 pc, against its 1110 pc.

Also load-bearing for any comparison: Pang §4.4 does fit EFF to all 85 clusters, but as a **3D volume
density**, so `γ_3D = γ_2D + 1` against the surface density fitted here. Their `γ` appears only in
histograms, with no table column.

Their §4.2 carries a warning that bears directly on the 43% of this sample lying beyond `r_J`:
*"An artificial 'halo'-like substructure will appear when the contamination rate rises above 20%."*
```


## An independent published criterion, and what it can and cannot say here

Muñoz, Padmanabhan & Geha (2012, ApJ 745, 127, `2012ApJ...745..127M`) ran a simulation suite asking
when maximum-likelihood structural parameters are recoverable at all, and stated the answer as three
inequalities in observables:

> *"to recover structural parameters within 10% or better of their true values: (1) the ratio of the
> field of view to the half-light radius … must be greater than three, (2) the total number of stars,
> including background objects, should be larger than 1000, and (3) the central to background stellar
> density ratio must be higher than 20."*

Applied to the 628-member, 70′ NGC 6383 sample:

| criterion | value | threshold | |
|---|---|---|---|
| FoV / half-number radius | **2.28** | > 3 | **fails** |
| total N | **628** | > 1000 | **marginal / fails** |
| central-to-background density (`k/b` = 7.433 / 0.0229) | 325 | > 20 | passes |

The honest score is **not "two of three fail"**. It is *one fails, one is ill-posed, one passes* — and
the two that can be evaluated cannot be read independently. Both corrections make the criterion harder
to attack, not softer.

### (1) is ill-posed for this system, not failed

`FoV / r_half > 3` presupposes a system with a boundary, so that `r_half` is a property of the
*object*. For NGC 6383 the 30.7′ is the half-number radius of the *selected sample*, and that sample
runs to the edge of the query. `r_half` is therefore measuring **where the footprint stopped**, not
where the cluster stopped: enlarge the query and it moves. A ratio whose denominator is set by its own
numerator does not have a value to compare against 3.

This is not a technicality about NGC 6383, it is the Gaia-era result. The bounded King sphere that
Muñoz's criterion assumes is not what surveys find when they look outside the classical radius:
Meingast et al. (2021) resolve coronae extending far past the tidal radius, Bouma et al. (2021)
trace a **500 pc halo** around NGC 2516, Yeh et al. (2019) describe Ruprecht 147 as a dissolving
cluster, and Kuhn et al. (2019) find ~75% of young clusters expanding. For an object with a corona
there is no field of view large enough to satisfy criterion (1), because there is no outer edge for
`r_half` to converge to. **Reporting this as "fails" imports a pre-Gaia ontology.** It should be
reported as inapplicable, which is a stronger and more defensible statement.

### (2) and (3) are coupled by the membership selection, in opposite directions

Muñoz et al. calibrate on a *photometric* field where cluster and background are separated by the fit
itself. Here they are separated beforehand, by proper motion and parallax — and that pre-filtering
moves the two remaining criteria in opposite directions **by construction**:

* filtering only ever removes stars, so `N` falls → criterion (2) moves toward *failing*;
* the stars it removes are preferentially background, so `b` falls → `k/b` rises → criterion (3) moves
  toward *passing*.

Both directions are monotone and guaranteed, no measurement needed. So the sample's comfortable
`k/b = 325` and its short `N = 628` are **the same fact seen twice**, not two independent verdicts,
and "2 of 3" double-counts. The pair is only independent on the unfiltered field Muñoz assumes, and
this analysis does not have one: the ASteCA input `NGC_6383_dr3_all.csv` holds just 194 sources inside
70′, fewer than the member list, so it is a pre-cut subset and cannot serve as the unfiltered
comparison.

### What survives

Criterion (2) is the one that survives intact, and it fails: `N = 628 < 1000`. Note this **is** the
"total including background objects" the paper asks for — the point-process fit models the observed
sample as `λ = k·King(r) + b`, so the 352 ± 30 stars implied by `b` over the 70′ field are already
inside the 628, not additional to them.

```{note}
That single surviving failure still points the same way as everything else in this note — the
unconstrained `R_t`, the 43% of members beyond the Jacobi radius, and Tarricq's own statement that
their tidal-radius distribution is truncated by a 50 pc query. The criterion did not discover the
problem; it puts one published number on it. What it does **not** license is the conclusion that
NGC 6383 is a defective cluster. It is a normally-sized open cluster observed over a footprint that
does not contain it — which, post-Gaia, is the usual case rather than the exception.
```

## What is the background term doing on a membership-selected sample?

King (1962) fits a **three-component** law — core, tidal truncation, and an additive constant `b` —
because the data were photographic star counts in which cluster and field could not be separated.
You fitted both at once because you had no other option.

Gaia plus astrometric clustering removes the field *before* the profile is fitted. So the question is
sharp: **why is there still a background component, and what is it?**

### For NGC 6383 it is not contamination, and the margin is not close

| quantity | value |
|---|---|
| fitted flat background `b` | 0.02285 ± 0.00194 arcmin⁻² |
| stars it accounts for over the 70′ field | **352 ± 30**, i.e. **56% of the 628-star sample** |
| measured false-discovery proportion (target–decoy, `p ≥ 0.6`) | **median 2.8%**, mean 10.5%, p90 30.8% (40 realizations) |
| discrepancy vs the median FDP | **20×** |

The membership pipeline's own false-positive rate, measured independently by the target–decoy
construction in `tools/validation/decoy_fdp.py`, is an order of magnitude too small to account for
the flat component. **Whatever `b` is absorbing, it is mostly real members.**

### And the component really is flat

Measured directly on the member sample, in arcmin⁻²:

| `r` (′) | 1.0 | 3.0 | 5.0 | 7.5 | 10.5 | 14.0 | 18.5 | 24.0 | 30.5 | 38.0 | 47.0 | 57.0 | 66.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Σ | 4.138 | 1.645 | 0.382 | 0.248 | 0.101 | 0.091 | 0.041 | 0.045 | 0.035 | 0.022 | 0.027 | 0.020 | 0.029 |

The profile falls by two orders of magnitude out to ~18′ and then **stops falling**, sitting at
0.02–0.035 all the way to the field edge — matching the fitted `b` = 0.02285. So this is not a
declining corona being mis-modelled as flat; within this footprint it genuinely is flat.

### What that means

The field radius is 70′ = **22 pc** at the P01 distance, and the flat component is 0.23 stars pc⁻²
across all of it. Meanwhile **43.5%** of members lie beyond the Jacobi radius and **25.2%** beyond the
fitted `R_t`.

```{important}
On a Gaia-selected sample the King background term stops meaning *"field stars I could not remove"*
and starts meaning *"co-moving stars my profile cannot represent"*. It is doing the same arithmetic
job King gave it in 1962 — soaking up whatever the core-plus-truncation form leaves over — but what
it soaks up has changed identity, and nothing in the fit announces that.

A flat projected surface density over the whole footprint is what you see when the footprint sits
*inside* a structure much larger than itself. The candidates are the cluster's own corona/tidal tails
(Meingast et al. 2021; Bouma et al. 2021's 500 pc halo) and the surrounding OB association. Both are
real, co-moving, and correctly admitted by astrometric membership — and both are exactly what
Hunt et al. (2026) exclude from their mock clusters on the stated assumption that their impact is
negligible.
```

**Practical consequence for any King fit on Gaia members:** `b` is not a nuisance parameter to be
marginalised and forgotten. Reporting `k`, `R_c`, `R_t` while silently absorbing 56% of the sample
into a constant is reporting the profile of a minority of the data. At minimum the fraction attributed
to `b` should be quoted alongside the structural parameters; at best the flat term should be replaced
by an explicit extended component and the two compared by Bayes factor.

```{admonition} Checked against the published paper from several angles — and the severity was overstated
:class: warning
**This note previously implied the published statement was a serious defect. Measured, it is a
wording issue and not a numerical one.** Four angles, 2026-08-02:

**1. The 56% is the 70′ fit, which P01 does not adopt for `R_c` or `k`.** The paper says so outright:
*"The core radius and central density of the 70 arcmin fit are contamination-biased … and are not
adopted."* The adopted structural parameters come from the 40′ window.

**2. In the adopted 40′ fit the background claims 23.5%, not 56%** — 59.5 of 253 stars. Against a
target–decoy false-discovery proportion of ~6% that is a factor of **3.9**, not 9. Still a
discrepancy, and still not "residual contamination", but a much milder one.

**3. `b` grows with footprint exactly as a corona should.** 40′ = 1.06 `r_J` gives 23.5%; 70′ = 1.85
`r_J` gives 59.5%. A residual-contamination term would not track the Jacobi radius that way; a corona
does, because a wider footprint contains more of it.

**4. The decisive test — does it move the adopted numbers?** The paper's `R_t` **is** taken from the
70′ fit, and `b`↔`R_t` are coupled, so this is where it could bite. Refitting under the paper's own
prior bound with the flat background replaced by the corona component:

| | flat `b` | corona | published |
|---|---|---|---|
| `R_t` | 59.1 (+12.6/−10.4)′ | **56.1 (+12.3/−10.5)′** | 54 (+7/−11)′ |
| `R_c` | 1.422′ | **1.364′** | 1.38′ |

`R_t` moves **3.0′** and `R_c` **0.06′**, both far inside the published uncertainties.

**So the published numbers stand.** What does not survive is the interpretive sentence *"The
background of this membership-selected sample measures the residual contamination level of the
selection, not the raw field density."* It is measuring cluster structure, not selection residue —
and EFF's own 1987 abstract anticipated the scale: *"up to 50% of the total masses in unbound
halos."* The correction is one sentence, and it makes the paper stronger rather than weaker, because
the alternative reading it currently forecloses is the interesting one.
```

### Three corrections from reading the primary sources

**1. The `+ b` is not King's.** King (1962) Eq. (14) is the bracket squared and nothing more. His
background was handled *outside* the formula — subtracted from the counts where a wide-field plate
reached past the cluster (M15), and where it did not (NGC 5053 on the 200-inch), *"chosen so as to
make the outermost points satisfy the empirical law of Eq. (14), with the same value of `r_t`"*. He
even warns against the misreading the extra term invites: *"the second term in brackets in Eq. (14)
could be replaced by a single constant; it is written in this more complicated form in order to show
the role of `r_t`"* — the constant already inside the bracket is the **truncation** constant.

The additive form is **folk practice with no primary citation**. Seleznev (2016) introduces it as
*"in order to take into account stellar background, this formula is supplemented by stellar
background density `F_b` as a constant addition"* and cites nobody.

**2. King identified the degeneracy in 1962**, criticising Wallenquist for having *"underestimated
the radii of the clusters and consequently chosen incorrect values for the background densities. In
M37, for instance, Wallenquist chooses a limiting radius of 17′, a distance at which the data …
indicate that the density is 15 or 20% above that of the true background."*

**3. "King 1964" does not exist.** Paper II is King (1965, AJ 70, 376), on steady-state *velocity*
distributions — not a density profile. Paper III is King (1966, AJ 71, 64), the lowered-isothermal
dynamical model with concentration `W_0`, which King says *"supersedes the purely empirical curves"*
of 1962 while agreeing closely with them for `W_0 < 7`. `ocelot`'s `king64.py` contains, in full,
`class King64(BaseClusterDistributionModel): pass` — an unimplemented placeholder with no docstring
and no ADS link, unlike its `king62.py`. Note also that `ocelot`'s King62 implements Eq. (14) **with
no background term at all**, i.e. faithfully to King.

### The mechanism has been published — qualitatively — and nobody followed it up

```{admonition} Seleznev (2016, MNRAS 456, 3757) says it outright
:class: important
> *"The reason is that the King model does not have an extended corona, and the cluster corona … **is
> perceived by the approximation algorithm as part of the stellar background**."*

And gives the arithmetic: *"Stellar background density `F_b^King`, obtained in the limits of the King
model, is usually larger than `F_b^comb` obtained in the limits of the combined model (the latter one
is usually very close to the visual estimate of this value)."* His fix is a King core **plus a uniform
sphere** for the corona, from Danilov & Putkov (2012) — and the projection of a uniform sphere,
`ΔF(r) = 2 R₂ δ_f √(1 − (r/R₂)²)`, is **nearly flat over the inner region**. That near-degeneracy with
a constant is precisely the mechanism.

Caveat: Seleznev's sample is 2MASS star counts, **not** membership-selected, so what he sees absorbed
is corona on top of a real field. The membership-cleaned case measured above is one step further.
```

Corroborating, from three independent directions:

* **Nilakshi et al. (2002, A&A 383, 153)**, 38 rich open clusters: the corona *"contains ~75% of the
  cluster members due to its larger area in comparison to the core region."* **The 56% measured here
  is squarely in that range** — it is what a corona fraction is supposed to look like.
* **Rui, Hosek, Lu et al. (2019, ApJ 877, 37)** fit King + `b` to an HST **membership-selected**
  Quintuplet sample, calling `b` *"a background term to capture remaining contaminants"*, and then
  report the compensation: *"our `r_t` posterior distribution runs up against the upper edge of the
  prior, forcing a larger best-fit `r_c` (and field contamination) value to compensate."* Their `b` is
  **4× larger in the King fit than in the EFF fit on the same data** — the same direction as
  Seleznev's `F_b^King > F_b^comb`. They do not comment on the comparison.
* **Pera, Perren, Navone & Vázquez (2021, BAAA 62, 119)** ran the cleaned-versus-uncleaned experiment
  directly: *"We repeated this process two times: first using the subset of probable members estimated
  with pyUPMASK, and then using all the stars in the frame … **We found the using the subset of most
  probable members impacts negatively on the results.** … The reason behind this appears to be the
  strong dependence of the fit on the field density parameter … **When using the sample cleaned by
  pyUPMASK this value is zero** … This is a surprising result that we will investigate further in a
  future more in depth analysis."* That follow-up does not appear to exist. They **reverted to fitting
  the uncleaned frame.** Pera et al. (2024) then states the design assumption outright: *"in the
  ASteCA method, King's profile models both field stars and cluster members."*

ASteCA's own source documents the degeneracy where no paper does — `packages/structure/king_profile.py`:
*"the value given to the field density has a **very large** influence on the final (rc, rt) values"*,
with the free-background variant abandoned because *"emcee tends to (rc→0, rt→inf)"*.

### What is actually unclaimed

The concept is Seleznev's. What is **not** in the literature:

1. **A quantitative test on a membership-selected sample against an independently measured
   contamination rate.** The comparison above needs a false-discovery estimate that does not come
   from the profile fit itself, and the target–decoy construction supplies one — but at 40
   realizations it supplies a **distribution, not a rate**: median 2.8%, mean 10.5%, p90 30.8% at
   `p ≥ 0.6`, with the decoy finding *nothing* in 11 of 40 realizations and as many as 516 stars in
   the worst. The mean moved 72% between 20 and 40 realizations while the median did not, so the mean
   is tail-dominated and not converged. **Quote the median as the central estimate and the p90 as the
   risk; a single mean is not a contamination bound.**
2. **A reformulated likelihood for the `b → 0` regime that decontamination creates** — which is
   exactly why Pera's cleaned fit degraded, and which they deferred and never returned to.
3. **The consequence for the radius estimator.** A large fraction of single-cluster papers use King's
   2-parameter Eq. (13), which never truncates, and define the radius *outside* the fit from the
   background's **uncertainty** via Bukowiecki et al. (2011), `r_lim = r_c √(f₀/(3σ_bg) − 1)`. For
   those papers, cleaning the sample first silently **destroys the radius estimator**, because there
   is no `σ_bg` left. Nobody appears to have said this.

Worth noting how the two most careful Gaia-era groups sidestep the problem entirely rather than solve
it: Hunt & Reffert abandon profile fitting and define `r_t` as the radius of maximum field contrast;
Olivares et al. make the field a **normalised mixture component with per-star weights** instead of an
additive pedestal, and warn in their Appendix A that *"the inference of the parameters in the King's
profile can be biased even after truncation has been accounted for … this effect can be generalised to
any maximum-likelihood estimator."*

### Seleznev's fix, implemented and fitted — and the answer is that the footprint is too small

`compare_radial_profiles(models=(..., "king_corona"))` now fits a **King core plus a uniform-sphere
corona**, the Danilov & Putkov (2012) / Seleznev (2016) two-component model, replacing the flat `b`.
The corona's projected density and its closed-form field integral are

```
ΔΣ(r) = 2 δ_f √(R₂² − r²)      Λ = (4πδ_f/3) [ R₂³ − (R₂² − min(R_f, R₂)²)^{3/2} ]
```

verified against quadrature to better than 1e-8 in all three regimes (corona inside the field, corona
larger than the field, exact boundary).

**Model comparison on NGC 6383** (SMC, 4 chains, 2000 draws; `2 ln B` on Kass & Raftery):

| model | log ML | chain sd | `2 ln B` vs best |
|---|---|---|---|
| King + flat `b` | **806.36** | 0.071 | 0 |
| EFF + flat `b` | 805.60 | 0.059 | −1.53 |
| **King + corona** | 805.23 | 0.040 | **−2.27** |
| Plummer | 796.40 | 0.039 | −19.93 |

**The data cannot distinguish a flat background from a uniform-sphere corona.** `2 ln B = −2.27` is
barely past "not worth more than a bare mention".

That is not a failure of the corona model — it is the *prediction*. As `R₂ → ∞` the corona projection
tends to the constant `2 δ_f R₂`, so a corona wider than the footprint **is** a flat background, exactly.
The fit says which case obtains:

```{important}
Fitting King + corona directly (NUTS, 4 chains, R-hat = 1.00, **zero divergences**):

| parameter | posterior |
|---|---|
| `R_c` | 1.30 ± 0.21′ — unchanged from every other fit |
| **`R₂`** | median **176.6′**, 95% interval **[96.8, 895]′** |
| `P(R₂ > field radius = 70′)` | **1.000** |
| `P(R₂ > 2 × field radius)` | 0.704 |
| corona stars inside the field | median **382 = 61%** of 627 |

**The corona is larger than the footprint with probability 1, and its outer radius is unbounded
above.** The 56% attributed to a flat background and the 61% attributed to a corona are the same
stars, relabelled — and the model cannot separate them *because the structure does not fit inside the
field*. `R₂`'s median of 176.6′ is **55 pc** at the P01 distance, against a 70′ = 22 pc footprint.
```

This closes the question the way the Gaia-era literature suggests it should: 55 pc is the scale of
Meingast et al.'s coronae and an order below Bouma et al.'s 500 pc halo around NGC 2516, and it is
consistent with 43.5% of members lying beyond `r_J` and 25.2% beyond the fitted `R_t`. **Nothing here
is a defect of the cluster or of the fit; the observation is smaller than the object.**

The practical consequence sharpens: a flat `b` is not merely mis-labelled, it is the best a
single-footprint fit can do. Distinguishing corona from background requires a field that contains the
corona — which is a statement about survey design, not about likelihoods.

### One more model this points at

**Wilson (1975, AJ 80, 175)** is the published alternative that keeps a finite truncation but is
spatially more extended than King. McLaughlin & van der Marel (2005, ApJS 161, 304) fit all three
families to a large cluster sample and find *"in the majority of cases that the Wilson models — which
are spatially more extended than King models but still include a finite, 'tidal' cutoff in density —
fit clusters of any age, in any galaxy, as well as or better than King models"*, adding that *"the
extended halos known to characterize many Magellanic Cloud clusters may be examples of the **generic
envelope structure** of self-gravitating star clusters, not just transient features associated
strictly with young age."* That belongs in the `compare_radial_profiles` Bayes-factor comparison
alongside King, EFF and Plummer. See `~/phd/open-threads.md` G6, G7.

## What else is not separately identified — a systematic audit

This programme kept finding degeneracies **one at a time and by accident**: `King(R_t→∞) ≡ EFF(γ=2)`,
corona-wider-than-field ≡ flat background, `R_t` ↔ `b` (King's own 1962 warning), `Q` ↔ contamination.
`tools/validation/degeneracy_audit.py` now looks for them on purpose, by reading the posterior
correlation matrix before a number is quoted.

On NGC 6383 (N = 627, 70′, 4 chains, zero divergences in all three fits):

| model | condition number | pairs above \|r\| = 0.7 |
|---|---|---|
| King + `b` | 17.8 | `R_c`↔`k` = −0.873 |
| **EFF + `b`** | **87.8** | `a`↔`γ` = **+0.871**, `a`↔`k` = −0.815, (`γ`↔`b` = +0.656) |
| King + corona | **15.1** | `R_c`↔`k` = −0.870 |

Three things follow, and the second is the one that matters for the census sweep.

**1. `R_c`↔`k` is universal and benign.** Present in every model at ≈ −0.87: a compact bright core and
a broader fainter one produce similar counts. It is the amplitude–scale trade every profile fit has,
and it does not touch the shape parameters.

**2. EFF is by far the worst-conditioned model, and `γ` is coupled to the background.** `γ`↔`b` at
+0.656 means **the slope is not measured independently of whatever the background term absorbs** — and
§ above establishes that on a membership-selected sample what it absorbs is the corona. So `γ` inherits
the footprint: a cluster whose field contains more of its corona has a larger `b` and, through the
correlation, a different `γ`.

Displayed on the real data by refitting NGC 6383 at nested footprints:

| field radius | N | `γ` | `a` (′) | background fraction |
|---|---|---|---|---|
| 20′ | 245 | 2.471 ± 0.552 | 1.803 ± 0.617 | 0.15 |
| 30′ | 310 | 2.542 ± 0.405 | 1.863 ± 0.521 | 0.29 |
| 40′ | 365 | 2.194 ± 0.264 | 1.487 ± 0.409 | 0.27 |
| 50′ | 445 | 2.300 ± 0.249 | 1.614 ± 0.408 | 0.41 |
| 60′ | 508 | 2.119 ± 0.194 | 1.388 ± 0.347 | 0.41 |
| 70′ | 627 | 2.322 ± 0.213 | 1.649 ± 0.380 | 0.56 |

`γ` and `a` move together in **perfect rank order** across all six — the +0.871 correlation made
visible — and the background fraction climbs from 0.15 to 0.56 as the footprint grows. The `γ` drift
itself (2.12–2.54) is not resolved at this `N`: the samples are nested and the errors are 0.19–0.55.

```{important}
**The consequence for the census sweep (A5) is a design requirement, not a caveat.** `γ` is coupled
to a background that depends on how much of each cluster's corona falls inside its footprint. Across
5647 clusters with heterogeneous search radii that becomes a **population-level trend** even where it
is unresolvable per cluster — and it correlates with distance, because nearby clusters have larger
angular coronae.

So the sweep must fix the footprint **physically** — in units of `r_J`, say — rather than accept
whatever angular or fixed-parsec radius each catalogue entry came with. Otherwise any structure found
in the `γ` distribution is partly a map of the survey geometry.
```

**3. The corona model is the best-conditioned of the three** (15.1), despite being the one with the
most parameters, and its `R_2`↔`delta_f` coupling is only −0.427 — much milder than the SMC evidence
noise suggested. The noise there is a property of the marginal likelihood estimate, not of the
posterior geometry.

## What follows

1. **Use the Jacobi radius as the `R_t` prior.** `king_unbinned` already accepts `tidal_prior=(mu, sigma)`
   and `dynamics.tidal_radius_prior()` computes it. Hunt & Reffert's `r_J` = 12.14 pc = 37.9′ is an
   independent value to use or to check against. This is the change PART J recommends, and the one
   with a physical justification rather than a numerical one.
2. **Fit EFF alongside King with a Bayes factor.** EFF (Elson, Fall & Freeman 1987,
   `1987ApJ...323...54E`) has no tidal cutoff, which is the honest model for a cluster whose
   truncation is not locatable. `king_unbinned`'s point-process likelihood extends to it by swapping
   `Σ(r)` and its closed-form integral.
3. **Test ellipticity before adding radii** — with reference `b/a` medians of 0.71, circular symmetry
   may be the larger modelling error.
4. **Do not compare `R_c` across catalogues** without checking the definition. Hunt's `rc`, `rt` and
   `rtot` are *empirical* radii, not King-fit parameters; only `rJ` is a physical quantity directly
   comparable to a claimed boundary. Zhong's `r_c` medians (5.80 pc) and Tarricq's (1.78 pc) differ
   by 3× for overlapping populations, which is definitional, not physical.

## References

* Hunt & Reffert 2024, A&A 686, A42 — `2024A&A...686A..42H` — VizieR `J/A+A/686/A42`
* Tarricq et al. 2022, A&A 659, A59 — `2022A&A...659A..59T` — VizieR `J/A+A/659/A59`
* Zhong et al. 2022, AJ 164, 54 — `2022AJ....164...54Z` — VizieR `J/AJ/164/54`
* Pang et al. 2022, ApJ 931, 156 — `2022ApJ...931..156P`
* Olivares et al. 2018, A&A 612, A70 — `2018A&A...612A..70O`
* Elson, Fall & Freeman 1987, ApJ 323, 54 — `1987ApJ...323...54E`
* Küpper et al. 2010, MNRAS 407, 2241 — `2010MNRAS.407.2241K` — a fitted `r_t` is the
  *time-averaged* tidal radius, not the perigalactic one
