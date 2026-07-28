# What the synthetic data is, what the field's baseline is, and which claims depend on the difference

Every quantitative claim in this repository that is not measured on NGC 6383 itself is measured on
**synthetic data**, and the generative model behind that data was chosen without first checking what
the field's own standard is. This note fixes that: it records the baseline, and it classifies each
synthetic-backed claim by whether a different generative model would change it.

The trigger was noticing that both references `erotica/analysis/synthetic.py` cites — Goodwin &
Whitworth (2004) and McLuster (Küpper et al. 2011) — are **pre-Gaia**.

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
| **"Ignoring completeness inflates `R_c` by 50% and halves the central density"** | `king_model_validity.md:194`, `test_structure.py` | **VULNERABLE** | Drawn from a smooth King with a toy `S(r)` (`floor=0.35`). Load-bearing: it is the yardstick that declares NGC 6383's measured 1.156% suppression negligible, and that step is an **extrapolation over a factor of 44** which assumes the bias is linear in the suppression. |
| **"The `γ` credible interval is 2–3.5× too narrow"** | `king_model_validity.md:324` | **VULNERABLE** | Substructure injected as **ad-hoc Gaussian clumps**, which is not a citable model of cluster structure and gives `Q = 1.019` — barely displaced from the smooth control's 1.276. |
| `R_t` is unidentifiable even inside `r_J` | `CANDIDATE-king-rt-identifiability.md` | **VULNERABLE** | Smooth-King recovery study. The direction is almost certainly robust (substructure can only widen the posterior) but the quoted magnitude is not. |

### What is being done about the two vulnerable ones

* `tools/validation/completeness_bias_scaling.py` measures the bias as a function of suppression
  amplitude instead of assuming linearity. The analytic result **predicts** linearity, so this tests
  the analytic backbone at the same time; a measured departure at small suppression would mean the
  extrapolation from 50% down to 1.2% is unsafe and the dossier's central claim needs re-deriving.
* `tools/validation/substructure_coverage.py` re-runs the coverage experiment with fractal
  substructure alongside the original ad-hoc clumps. The fractal configuration is radially remapped
  onto the same EFF profile, so the angular structure changes while the radial marginal does not and
  the comparison isolates substructure. Measured `Q` after the remap: **0.591** at `D = 1.6` and
  **0.729** at `D = 2.0`, against 1.019 for the ad-hoc clumps and 1.276 for the smooth control —
  i.e. substantially more substructured than the configuration behind the published table, and
  bracketing NGC 6383's observed `Q = 0.833`.

  This experiment also had **no script behind it** until now; the numbers in the design note were
  produced ad hoc and only the table survived.

---

## 3. The honest statement of where this repo stands relative to the baseline

**They are complementary, not competing.** ASteCA is the standard on the *photometric* axis — IMF,
binaries, extinction, differential reddening, error calibration — and models no spatial structure at
all. This repository's work is entirely on the *spatial* axis and models no photometry at all.

The consequence for P02 is specific: a novelty claim may be made on the structural axis, but on the
CMD axis ASteCA is ahead and must be **cited as the standard rather than benchmarked against**. See
`~/phd/software-landscape.md` for the full adjudication, including the two BAAA proceedings
(Pera+2021, Pera+2024) that are the real prior art for a Bayesian King fit and that must be read
before any "first Bayesian King profile" sentence is written.
