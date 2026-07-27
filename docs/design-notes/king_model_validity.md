# Is a King profile the right model for NGC 6383?

_Written 2026-07-27. Everything numeric here is produced by
`tools/validation/king_literature_context.py`, which fetches both reference catalogues live
from VizieR — nothing is transcribed by hand. Companion to the delta report in
[decisions](decisions.md)._

The package has an unbinned King fit whose statistics are now sound
(`structure.king_unbinned`). That answers *how* to fit a King profile. It does not answer
whether a King profile is what NGC 6383 wants, and the delta report raised the question
sharply: with a scale-free prior, `R_t` has a posterior SD of thousands of arcmin.

The honest reading is that this is **not a fitting problem**. It is a field-of-view problem
compounded by a model-choice problem, and both are measurable.

## 1. The extraction is too small, by a factor of 2.2

At the paper's adopted 1.11 kpc, 1 arcmin = **0.3229 pc**. So:

| | angular | physical |
|---|---|---|
| P01 reference extraction | 40′ | 12.9 pc |
| P01 largest extraction (Appendix D) | 70′ | 22.6 pc |
| P01 adopted `R_t` | 54′ | 17.4 pc |
| **Tarricq+2022 search radius** | **155′** | **50 pc** |

Tarricq et al. (2022, A&A 659, A59, `2022A&A...659A..59T`) search to **50 pc from the centre**
precisely because they found "vast coronae around almost all the clusters" and concluded that
"OCs are more extended than previously expected, **regardless of their age**". At NGC 6383's
distance that search radius is **155 arcmin**. P01's largest field is 70′.

The consequence is arithmetic rather than arguable:

> **Tarricq's *median* King `R_t` is 26.1 pc, which is 81 arcmin at 1.11 kpc — beyond the edge of
> the largest field P01 fits.**

A typical open cluster placed at NGC 6383's distance would have its King outer radius *outside the
extraction*. The published `R_t` = 54′ is 77% of the extraction radius. That the posterior runs into
its prior ceiling is then the expected outcome, not an anomaly, and no likelihood can repair it —
the information is not in the data.

It is worth noting that the unbinned fit's `R_t` median (**80.2′**) lands essentially on the
literature median for open clusters at this distance (**81′**). That is consistency, not
confirmation: both are what you get when the data say only "at least as large as the field".

**The fix for `R_t` is a bigger cone, not a better model.** ~155′ to match the reference method.

## 2. NGC 6383 is younger than every cluster in both reference samples

| sample | N with King fits | youngest | NGC 6383 |
|---|---|---|---|
| Tarricq+2022 | 233 | log age **7.70** (50 Myr) | log age 6.55 (3.5 Myr) |
| Zhong+2022 | 256 | log age **7.0** (10 Myr) | log age 6.55 |

**Zero** clusters in Tarricq's King-fitted sample are younger than NGC 6383; its youngest is
**14× older**. So comparing NGC 6383's structural parameters to the modern homogeneous samples is
an **extrapolation**, and any statement of the form "typical for an open cluster" is unsupported at
this age.

Where it does land, for what that is worth:

| quantity | NGC 6383 | Tarricq 16/50/84 (pc) | percentile |
|---|---|---|---|
| `R_c` | 0.63 pc | 1.09 / 1.78 / 2.84 | **3.0%** |
| `R_t` | 17.4 pc | 12.56 / 26.13 / 38.56 | 27.9% |
| `C = log₁₀(R_t/R_c)` | 1.43 | 0.79 / 1.13 / 1.44 | 82.8% |

```{note}
The `R_c` result carries a tension worth stating. Tarricq's own conclusion is that **"the size of
the cores is smaller for old clusters than for young ones on average"** — yet NGC 6383, far younger
than any of them, has a core smaller than 97% of that older sample. Either the core is genuinely
exceptionally compact, or `R_c` fitted to a membership-selected 40′ field is not measuring the same
quantity Tarricq measures over 50 pc. The field-of-view result above makes the second reading
plausible and it should be excluded before the first is claimed.

Zhong's `r_c` is **not** comparable to Tarricq's (medians 5.80 pc vs 1.78 pc for overlapping cluster
populations), so the "0.0 percentile" the script prints against Zhong reflects a definitional
difference, not a physical one. Cross-catalogue `R_c` comparison is unsafe.
```

## 3. Circular symmetry — which the King fit assumes — is the exception

Tarricq fit an elliptical Gaussian mixture alongside the King profile, so their catalogue carries
axis ratios. For the **core** component:

* median `b/a` = **0.71**
* **92.9%** of clusters have `b/a < 0.9`
* **68.9%** have `b/a < 0.8`

A radially symmetric King profile is fitted to a population that is, empirically, almost never
round. This matches Olivares et al. (2018, `2018A&A...612A..70O`), who found **strong evidence
against radially symmetric models** relative to elliptical extensions.

For a cluster of NGC 6383's age the expectation is stronger still. Pang et al. (2022, ApJ 931, 156,
`2022ApJ...931..156P`) classify substructure outside the tidal radius and find that below 100 Myr it
is **filamentary or fractal** — halo and tidal-tail morphologies only appear above 100 Myr. They
report that for systems ≲30 Myr the axis ratio, mass and half-mass radius all *increase* with age,
which they attribute to filament dissolution and subgroup mergers. A 3.5 Myr cluster is squarely in
the filamentary/fractal regime: a hierarchical, still-assembling structure, not a relaxed
tidally-truncated sphere.

## 4. A single King `R_t` is no longer how the field describes outer structure

Zhong et al. (2022, AJ 164, 54, `2022AJ....164...54Z`) searched 256 OCs to 50 pc and found that
**"the radial density profile in the outer region for most open clusters deviates from the King
profile"**. They replace it with a two-component model — a King core plus a log-normal outer halo —
and argue for **four** radii (`r_c`, `r_t`, `r_o`, `r_e`) rather than two.

So the quantity P01 calls the King outer radius is, in the current literature, only the boundary of
the *core component*; the halo needs its own description. This is independent of the field-of-view
problem and would remain true with a 155′ cone.

## What follows

Ordered by value, and none of it is a correction to P01 — the manuscript already reads `R_t` as
"a model-dependent scale rather than an independently measured physical boundary", which the above
supports.

1. **Re-extract at ~155′** if `R_t` is ever to be measured rather than bounded. This is the only
   change that adds information. Everything else redistributes it.
2. **Fit EFF alongside King with a Bayes factor.** EFF (Elson, Fall & Freeman 1987,
   `1987ApJ...323...54E`) has no tidal cutoff and is the standard choice for young clusters; if the
   data cannot locate `R_t`, a model that does not require one may be preferred outright. PART J of
   `~/phd/methodology.md` already recommends this. `king_unbinned`'s point-process likelihood
   extends to EFF by swapping `Σ(r)` and its closed-form integral — the machinery is in place.
3. **Test ellipticity before adding radii.** With `b/a` medians of 0.71 in the reference sample,
   circular symmetry may be a larger modelling error than the profile family, and it is cheap to
   check on a sample that already exists.
4. **Do not compare `R_c` across catalogues** without checking the definition.

## References

* Tarricq et al. 2022, A&A 659, A59 — `2022A&A...659A..59T` — VizieR `J/A+A/659/A59`
* Zhong et al. 2022, AJ 164, 54 — `2022AJ....164...54Z` — VizieR `J/AJ/164/54`
* Pang et al. 2022, ApJ 931, 156 — `2022ApJ...931..156P`
* Olivares et al. 2018, A&A 612, A70 — `2018A&A...612A..70O`
* Elson, Fall & Freeman 1987, ApJ 323, 54 — `1987ApJ...323...54E`
* Küpper et al. 2010, MNRAS 407, 2241 — `2010MNRAS.407.2241K` — fitted `r_t` is the
  *time-averaged* tidal radius, not the perigalactic one
