# What the synthetic data is, what the field's baseline is, and which claims depend on the difference

Every quantitative claim in this repository that is not measured on NGC 6383 itself is measured on
**synthetic data**, and the generative model behind that data was chosen without first checking what
the field's own standard is. This note fixes that: it records the baseline, and it classifies each
synthetic-backed claim by whether a different generative model would change it.

The trigger was noticing that both references `erotica/analysis/synthetic.py` cites — Goodwin &
Whitworth (2004) and McLuster (Küpper et al. 2011) — are **pre-Gaia**.

```{admonition} The premise turned out to be wrong, and that is the useful finding
:class: important
"Pre-Gaia" is not the same as "superseded". A full-text sweep of the successor literature shows the
**box-fractal prescription is still the live standard in 2026**, and there are in fact *two* golden
standards which do not overlap — see §0. What is genuinely missing is not a better generator but the
**combination** nobody ships.
```

---

## 0. There is no single golden standard. There are two, and the gap between them is the opening

### Standard A — spatial substructure: still Goodwin & Whitworth (2004)

Not superseded, and the proof is contemporaneous. Amiri et al. 2026 (`arXiv:2606.04509`, 3 June 2026),
from the McLuster lineage group itself:

> *"Implementation of the initial condition is done using the most updated version of the code
> McLuster."* … *"Initial fractal substructures were implemented using the method described in
> Goodwin and Whitworth [2004]."* … *"we investigated two extreme values: D = 3.0, which results in
> no substructure, and D = 1.6, which results in a cluster with the maximum degree of fractality."*

That is the same prescription and the same two limiting values this repo implements. Kamlah et al.
(2022, `2022MNRAS.511.4060K`, 84 cit) is often cited as "updated McLuster", but its own abstract
scopes it to *"updated stellar evolution recipes"* — SSE/BSE, winds, kicks, remnants. **The spatial
prescription is untouched since 2011.** AMUSE's `new_fractal_cluster_model` implements the same
Goodwin & Whitworth construction.

The honest qualification comes from Torniamenti et al. (2022, `2022MNRAS.510.2097T`), and it is
measured rather than a demolition:

> *"For more than a decade, fractal initial conditions have been used as a starting point for
> realistic simulations … **but even this approach does not guarantee that all the relevant
> characteristics of the actual primordial conditions of star clusters are correctly captured.**"*

The improvement path is **hydro-derived** initial conditions — Ballone et al. 2020/2021
(`2020MNRAS.496...49B`, `2021MNRAS.501.2920B`), which find `D = 1.0–1.8` from SPH, *more*
substructured than typical fractal ICs; Torniamenti et al. 2022 resamples hydro snapshots to make
many ICs cheaply. That is the upgrade to name in P02's future-work, not a defect in the current
implementation.

### Standard B — the Gaia observational layer: Hunt et al. 2025/2026 and `ocelot`

This is what did not exist when Goodwin & Whitworth was written, and it is where the modern bar sits:

* **Hunt, Cantat-Gaudin, Anders et al. 2025**, `2025A&A...699A.273H` (Paper I) — *"We develop a method
  to generate realistic mock clusters … accounting for Gaia's selection function and astrometric
  errors. We then inject mock clusters into Gaia DR3 data, and attempt to recover them in a blind
  search using HDBSCAN."*
* **Hunt et al. 2026**, `2026A&A...706A.341H` (Paper II) — **80,590 injection/retrievals**, and
  *"we open sourced our cluster-simulation code in the upcoming Python package `ocelot`"*.

Their recipe is the one to match: PARSEC v1.2s isochrones + Kroupa (2001) IMF; Moe & Di Stefano (2017)
binary pairing; **errors by "twin" resampling** — match each synthetic star to the nearest real Gaia
star in `G` and adopt its real uncertainties; and a **two-stage selection function**, Cantat-Gaudin
et al. (2023) for DR3 itself plus Castro-Ginard et al. (2023) for the quality-cut subsample.

Twin-resampling is not a Hunt idiosyncrasy — **Buckner et al. 2024** (`2024MNRAS.527.5448B`) arrives at
it independently: *"Each simulated star is randomly assigned a RUWE, and the uncertainties in parallax
and proper motion of a DR3 star in the same magnitude bin."*

### The gap, verified at source-code level (2026-07-28)

```{important}
`ocelot/src/ocelot/model/distribution/` contains **only** `king62.py`, `king64.py`, `plummer.py`.
There is no fractal, substructured, or N-body-snapshot distribution model. And Hunt et al. say so
themselves, in Paper II:

> *"Recent Gaia-based results have shown that the bound cores of many clusters are surrounded by
> tidal tails and coronae (Röser et al. 2019; Meingast & Alves 2019; Meingast et al. 2021; Tarricq
> et al. 2022; Kos 2024), which are not included by King profiles; however, given the sparsity of
> these structures, we assume that their impact on cluster detectability is negligible, and we do
> not include them in our cluster simulations."*

So: **the Gaia-aware mock-cluster standard is spherically symmetric and smooth, and the substructure
standard has no Gaia error or selection layer. No tool does both.** Buckner+2024 and Pang+2022 pair
substructure with an error layer and injection, but both are bespoke and neither applies a selection
function.
```

Buckner et al.'s conclusions are the ones that bear directly on structural work, and they align with
this repo's own coverage result: *"most quantitative conclusions are likely to be inaccurate"*, driven
by *"the disappearance of cluster substructure as the data become more incomplete"* and, at close
range, *"the misidentification of asterisms as true structure."*

### Where the referee bar actually is

Lower than feared, and worth knowing before over-engineering. Across 13 refereed 2024–2026 papers
that forward-simulate: **4** apply a formal Gaia selection function (all Hunt-lineage plus one),
**3** use ASteCA's internal empirical completeness *without saying so*, and **6** apply a plain
magnitude cut or nothing. ADS full-text: `"gaiaunlimited" AND "open cluster"`, refereed 2024–2026 =
**16 papers, 0.8%**.

The negative control is decisive: **Li et al. 2025** (MiMO, AJ 170, 288) publishes mass functions for
**1232** open clusters with **no selection function at all** — a hard `G < 18` cut and real per-star
uncertainties. Refereed, published.

**So adopting the Hunt/`gaiaunlimited` stack is a differentiator, not an entry requirement.** The
cheapest currently-accepted pattern is Cordoni et al. 2023 (A&A 672, A29): *"We assumed that the Gaia
DR3 catalog is complete for magnitudes brighter than `G_RP = 18.5`"*, citing Boubert & Everall without
running their code.

```{warning}
**Gaia DR4 is 2 December 2026.** Every DR3 selection function above will need re-deriving. Any
submission that leans on one should say which release it is conditioned on.
```

```{admonition} A published critique that lands on this repo's own tests
:class: caution
Daffern-Powell & Parker (2020, `2020MNRAS.493.4925D`, 40 cit): *"The interpretation of the
Q-parameter often relies on comparing observed values of Q, m̄, and s̄ to idealized synthetic
geometries, where there is little or no match between the observed star-forming regions and the
synthetic regions … we caution that the observed Q-parameter should not be directly compared to
idealized geometries."*

This repo independently reached the same place by a different route — the measurement that a
perfectly smooth King diluted by ~60% uniform field reproduces NGC 6383's observed `Q = 0.833`
exactly, recorded in `king_model_validity.md` as "the Q parameter cannot justify fitting a radial
profile". **The convergence is worth citing rather than re-deriving.**

What it permits and forbids, precisely: comparing Q *between synthetic geometries* is fine, and that
is the only way `tests/test_synthetic.py` uses it (Q must rise monotonically with `D`). Comparing the
*observed* Q to a synthetic Q is the flagged misuse, and any place this note does that — including the
`Q = 0.833` bracketing in §2 — is context, never evidence.
```


---

## 1. The field's baseline is ASteCA, and ASteCA generates no positions

ASteCA (Perren, Vázquez & Piatti 2015, `2015A&A...576A...6P`) is the standard open-source tool for
this exact task, and a July-2026 paper on ~7000 clusters still states *"the interpolation method was
tested using synthetic clusters built with ASteCA"* (Neumannová et al., `arXiv:2607.15149`). So it is
current practice, not history.

**Its synthetic clusters are purely photometric.** Verified at three independent levels — the 2015
methods section, the vendored v0.4.3 source used for P01
(`data/test/NGC6383/ASteCA/packages/synth_clust/`), and the current v0.6.9 release. The generated
object is

```
[mag, c1, (c2), mass, mass_b]
```

— magnitudes, colours, masses. **No positions, no radial profile, no astrometry.** The forward model
is isochrone → IMF (Chabrier 2014 by default in v0.6.9; Kroupa 2002 in the version P01 used) →
extinction with differential reddening → binaries with a mass-dependent multiplicity fraction fitted
to Offner et al. (2023) → completeness removal → photometric errors calibrated on the observed
cluster.

In the vendored version the spatial generator is the file literally named
`synth_clust_gen_NOTIMPLEMENTED.py` ("In place for #239"), and where it sketches positions it imports
`structure.king_profile.KingProf` with `rc = uniform(0.25·rt, 0.5·rt)` — **a smooth King, no
substructure.**

### Where ASteCA's spatial validation actually came from

The 2015 paper draws the distinction itself and warns against exactly the confusion above: the
objects with spatial structure are **SOCs from the external MASSCLEAN package** (Popescu & Hanson
2009, `2009AJ....138.1724P`), *"a tool able to create artificial stellar clusters following a King
model spatial distribution … including added field star contamination"*, which *"must not be
confused with the synthetic clusters generated internally by ASteCA"*.

The validation grid is **432 SOCs**: mass 50–1000 M☉, `z ∈ {0.002, 0.008, 0.019, 0.03}`,
`log(age) ∈ {7, 8, 9}`, `d ∈ {0.5, 1, 3, 5}` kpc, `A_V ∈ {0.1, 0.5, 1, 3}`, scored as
`Δparam = true − ASteCA` against the **contamination index**.

```{important}
**So the reference practice for validating a structural method is: recover a smooth King from a
smooth King plus uniform field contamination, over a grid, scored against contamination.**

That design cannot detect a failure mode caused by structure the generator does not produce. It is a
baseline to match on *coverage of the grid* — and to exceed on *realism of the generator*.
```

Two further facts, recorded because they bear on what this repo can claim:

* **ASteCA ships no test suite.** Its validation lives entirely in the 2015 paper.
* **ASteCA treats completeness as `S(magnitude)`**, estimated empirically from the observed luminosity
  function (`completeness_rm.py`), never as `S(r)`. That is independent corroboration, from the
  software side, of the literature search finding that no paper applies a selection-function-corrected
  radial density profile.

---

## 2. Which of this repo's claims survive a change of generative model

A synthetic-backed claim is **SAFE** when the synthetic data is used for a *within-model* comparison —
two estimators on the same data, or an analytic identity — because the conclusion is a statement about
the estimators and holds whatever the data really look like. It is **VULNERABLE** when a *magnitude*
is asserted that is meant to transfer to a real cluster, because a different generative model gives a
different number.

| claim | where | verdict | why |
|---|---|---|---|
| `King(R_t → ∞) ≡ EFF(γ = 2)`, exact for all `r` | `king_model_validity.md` | **SAFE** | Algebraic identity, verified to 80 decimal places. No synthetic data involved. |
| Unbinned point process beats equal-count binning | `king_binning_likelihood.py` | **SAFE** | Both estimators see the *same* realizations. Which one loses information is a property of the estimators. |
| First-order selection bias is `δθ = ε I⁻¹v`; constant `S` biases no shape parameter | `king_model_validity.md` | **SAFE** | Analytic; degree-1 homogeneity. Synthetic runs only confirm the algebra. |
| Likelihood is well calibrated on a genuine Poisson point process (93% at 2σ) | `substructure_coverage.py` | **SAFE** | This *is* the within-model check, and it is correctly scoped as one. |
| **"Ignoring completeness inflates `R_c` by 50% and halves the central density"** | `king_model_validity.md:194`, `test_structure.py` | **VULNERABLE → SURVIVED, with a corrected number** | The extrapolation assumed a linearity nobody had measured. Measured: linear with χ²/dof = 0.18 **below ~20% suppression**, superlinear above, exactly as a first-order expansion should behave. NGC 6383 sits far inside the linear regime, so the step is legitimate — but the slope must come from there (0.663), not from the 50% anchor (1.13). Predicted `R_c` bias **+0.77%**, not +1.12%. See §2. |
| **"The `γ` credible interval is 2–3.5× too narrow"** | `king_model_validity.md:324` | **VULNERABLE → SURVIVED** | Substructure was injected as ad-hoc Gaussian clumps. Re-tested with Goodwin & Whitworth fractal clumps at matched amplitude and matched `Q`: understatement 3.15 → 3.21. **The model does not matter, only the amplitude.** See §2. |
| `R_t` is unidentifiable even inside `r_J` | `CANDIDATE-king-rt-identifiability.md` | **VULNERABLE → SUPERSEDED** | Was a smooth-King *synthetic* recovery study, so the magnitude was model-dependent. It no longer needs to carry the claim: fitting the corona model to the **real data** gives `P(R₂ > field radius) = 1.000` with `R₂` unbounded above (median 55 pc against a 22 pc footprint). `R_t` is unidentifiable because **the footprint does not contain the object** — measured, not simulated. See `king_model_validity.md`. |

### What is being done about the two vulnerable ones

* `tools/validation/completeness_bias_scaling.py` measures the bias as a function of suppression
  amplitude instead of assuming linearity. **Run, and the answer has a twist worth keeping.**

  | core suppression | `R_c` bias | bias / suppression |
  |---|---|---|
  | 0.00% (null) | +0.61% ± 0.83% | — |
  | 4.23% | +2.83% ± 0.85% | 0.670 |
  | 8.46% | +5.51% ± 0.96% | 0.652 |
  | 16.94% | +11.28% ± 1.02% | 0.666 |
  | 29.71% | +22.41% ± 1.07% | 0.754 |
  | 42.53% | +37.86% ± 1.30% | 0.890 |
  | 55.42% | +62.65% ± 1.16% | 1.131 |

  A single line through the origin fitted to **all** levels gives slope 0.971 with
  **χ²/dof = 22.9** — the relation is *not* globally linear. But look at where it breaks: the ratio
  is flat at **0.66** from 4% to 17% and only climbs above ~20%. Fitting the low-suppression levels
  alone gives slope **0.663 with χ²/dof = 0.18**.

  ```{important}
  That is exactly what `δθ = ε I⁻¹v` predicts. It is a **first-order** expansion, so it must hold for
  small `ε` and must fail once the perturbation is large — and the measured knee sits at ~20%
  suppression. **The analytic backbone is validated, and so is the extrapolation, but only below the
  knee.**

  The consequence is a corrected number. The design note's yardstick was anchored at the toy's 50%
  suppression, where the local slope is 1.13; used naively it predicts a **+1.12%** `R_c` bias at
  NGC 6383's measured 1.156% suppression. The correct low-suppression slope gives **+0.77%**.
  NGC 6383 sits far inside the linear regime, so the extrapolation is legitimate — it was simply
  being done with the wrong slope.
  ```

  Both numbers are negligible: +0.77% of `R_c = 1.38′` is 0.011′, about **0.05σ**, which is
  consistent with the −0.01σ measured by actually fitting with the real `S̄(r)`. Analytic prediction,
  scaling experiment and direct measurement now agree. The null at zero suppression
  (+0.61% ± 0.83%) confirms the harness has no bias of its own.
* `tools/validation/substructure_coverage.py` re-runs the coverage experiment with fractal
  substructure alongside the original ad-hoc clumps. It also had **no script behind it** until now;
  the numbers in the design note were produced ad hoc and only the table survived.

### First result, and one row of it had to be thrown away

40 realizations per configuration, NGC 6383 geometry (`N = 628`, 70′, EFF `γ = 2.32`, `a = 1.65′`):

| configuration | Q | `γ` recovered | reported σ | realization scatter | understatement | 1σ cov. | 2σ cov. |
|---|---|---|---|---|---|---|---|
| smooth control | 1.259 | 2.395 | 0.086 | 0.076 | **0.88** | 62% | **95%** |
| 50% in 15 clumps, σ = 1′ | 0.985 | 2.608 | 0.121 | 0.381 | **3.15** | 20% | 45% |
| 50% in 8 clumps, σ = 2′ | 1.036 | 2.808 | 0.159 | 0.555 | **3.49** | 12% | 38% |
| ~~fractal, `D` = 1.6~~ | ~~0.537~~ | ~~2.416~~ | ~~0.087~~ | ~~0.085~~ | ~~0.98~~ | ~~48%~~ | ~~88%~~ |

The clump rows **reproduce the published table**, so those numbers are now backed by a script, and
the smooth control is well calibrated at 2σ — the likelihood is right when its assumptions hold.

```{danger}
**The fractal row is struck through because it measured nothing.** The first implementation remapped
radii by substituting the *order statistics* of a fresh EFF draw, which makes the resulting radii an
**iid EFF sample** — statistically identical to the smooth control. Since `eff_unbinned` is a *radial*
likelihood that never sees angles, that construction rendered the substructure invisible by design.
Its "well calibrated" answer (understatement 0.98 against the control's 0.88) was a tautology.

Had it been reported, the conclusion would have been *"realistic fractal substructure does not bias
the fit, unlike ad-hoc clumps"* — clean, publishable, and **completely false**.
```

The second attempt was a **population-level** quantile map, `F_EFF⁻¹(F_fractal(r))` with both CDFs
fixed in advance. It fails too, and differently: fractals at low `D` have enormous
realization-to-realization variance in extent, so the pooled CDF's upper quantiles are set by rare
wide realizations, and a typical compact realization maps entirely into the low part of that CDF and
comes out too small. A 7% median offset that no tuning removes, because it is structural.

*(That attempt also surfaced a third generator defect worth recording: the reference CDF was first
built from one 200,000-star realization, but the construction stops subdividing once it holds
`8 × n_stars` points, so a 200k draw runs to a different number of levels than a 628-star one and
obeys a different radial law — the mapped median came out at 2.64 against EFF's 6.17. **The fractal's
radial distribution is n-dependent.**)*

### The design that works, and the answer

Fix the profile **by construction** rather than by a transform that has to be verified: keep the
published configuration exactly — clump centres drawn from EFF, same clumped fraction, same clump
count, same amplitude — and change **only** the internal structure of each clump from Gaussian to
Goodwin & Whitworth fractal. Expected radial profile is then exactly EFF by construction, and `Q`
lands on the Gaussian value (1.014 vs 1.039; 1.024 vs 1.051), so it is a genuine one-variable test.

| configuration | Q | `γ` recovered | reported σ | realization scatter | understatement | 1σ cov. | 2σ cov. |
|---|---|---|---|---|---|---|---|
| smooth control | 1.259 | 2.395 | 0.086 | 0.076 | **0.88** | 62% | **95%** |
| 15 clumps, σ = 1′, **Gaussian** | 0.985 | 2.608 | 0.121 | 0.381 | **3.15** | 20% | 45% |
| 15 clumps, σ = 1′, **fractal** | 0.962 | 2.604 | 0.120 | 0.387 | **3.21** | 15% | 42% |
| 8 clumps, σ = 2′, **Gaussian** | 1.036 | 2.808 | 0.159 | 0.555 | **3.49** | 12% | 38% |
| 8 clumps, σ = 2′, **fractal** | 0.990 | 2.811 | 0.158 | 0.506 | **3.20** | 12% | 30% |

```{important}
**The substructure model does not matter; the amplitude does.** Swapping ad-hoc Gaussian clumps for
the citable Goodwin & Whitworth construction moves nothing measurable — `γ` 2.608 → 2.604, reported σ
0.121 → 0.120, understatement 3.15 → 3.21 at matched amplitude, and the same again for the wider
configuration.

**So the VULNERABLE claim survives the audit.** The published "the `γ` credible interval is 2–3.5×
too narrow" is *not* an artifact of the ad-hoc clump model, which is precisely what the audit was
built to find out. It can now be defended by pointing at the standard prescription rather than at a
sprinkling of Gaussians.
```

### A second finding the run produced for free: the estimator is biased at this N

The smooth control recovered `γ = 2.395` against an injected 2.32 — a bias of **+0.075 ± 0.012**, 6σ,
with no substructure and no selection effect anywhere. The EFF radial sampler was checked against the
analytic EFF CDF first and is exact (max deviation 7 × 10⁻⁴ on 400,000 draws), so this is
**finite-sample estimator bias, not a generator bug**. It decays with sample size:

| `N` | recovered `γ` | bias |
|---|---|---|
| 628 | 2.381 | +0.061 |
| 2512 | 2.351 | +0.031 |
| 10048 | 2.331 | +0.011 |

roughly `N^-0.6`. **This bears on P01 directly.** The published `γ = 2.32 ± 0.21` is measured at
`N = 628`, so the debiased value is nearer **2.25**, which sits *closer* to the King/EFF degeneracy at
`γ = 2` — 1.2σ rather than 1.5σ. The correction strengthens the existing argument rather than
threatening it, but it should be applied and stated rather than left implicit.

---

## 2b. Checked against the reference packages, not just against the papers

Reading what the standard tools do is weaker than running them. Both were installed from PyPI and
exercised on 2026-07-28.

### `erotica`'s King agrees with `ocelot`'s `King62` to within Poisson noise

`ocelot.model.distribution.King62` is the Hunt-lineage reference implementation — the one behind
80,590 published injection/retrievals. Drawing 400,000 3D positions from it, projecting isotropically,
and comparing the binned surface density against **area-averaged** `king_profile` over the same
annuli (evaluating the model at bin centres instead is a 16% error on the innermost bin, and the
first comparison did exactly that):

> mean ratio **1.0036**, scatter **1.67%**, max deviation **3.7%** across 0–30 pc, against a Poisson
> floor of **2.9%**.

That is an independent external validation of the surface-density implementation the published
`R_c` / `R_t` rest on.

### Three reproducibility barriers in `ocelot`, verified rather than inferred

Worth stating in P02, and they strengthen the case for contributing upstream rather than depending on
it:

1. **`ocelot.simulate` does not import from a clean `pip install ocelot`** (v0.4.10). It raises
   `ValueError: zero-size array to reduction operation minimum` because `AVAILABLE_METALLICITIES` is
   empty — the isochrone data is a manually downloaded blob behind the `OCELOT_DATA` environment
   variable. The simulation half of the Gaia-aware standard is therefore not installable.
2. **`ocelot.model.distribution` exports exactly one concrete model, `King62`** — not King62, King64
   and Plummer as the file listing suggests. The gap is starker than a directory listing implies.
3. **`King62` raises `NotImplementedError` for `dimensions=2`** ("Other levels of dimensionality are
   coming soon"), so the *projected* surface density — the thing every observational radial-profile
   fit actually needs — is not available, and it requires physical units (`pc`), rejecting `arcmin`.

The extension point for a fractal model is small and stable: `BaseClusterDistributionModel` requires
only `base_unit`, `pdf`, `cdf`, `rvs`.

---

## 3. The honest statement of where this repo stands relative to the baseline

**They are complementary, not competing.** ASteCA is the standard on the *photometric* axis — IMF,
binaries, extinction, differential reddening, error calibration — and models no spatial structure at
all. This repository's work is entirely on the *spatial* axis and models no photometry at all.

The consequence for P02 is specific: a novelty claim may be made on the structural axis, but on the
CMD axis ASteCA is ahead and must be **cited as the standard rather than benchmarked against**. The adjudication rests on the two BAAA proceedings
(Pera+2021, Pera+2024) that are the real prior art for a Bayesian King fit and that must be read
before any "first Bayesian King profile" sentence is written.
