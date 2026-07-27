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
