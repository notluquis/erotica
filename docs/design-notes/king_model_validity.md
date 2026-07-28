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
