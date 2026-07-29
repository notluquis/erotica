# erotica/analysis/ — the modules whose numbers reach papers

Every scientific choice made here has a family of published alternatives. **Read the row for your
module in `~/phd/model-landscape.md` before extending it**, and update that row when you make a
choice — naming what you rejected. An `UNSURVEYED` row is a recorded liability, not a neutral state.

## Standing conventions in this package

- **Priors are fixed constants, independent of the data being fitted.** `KingPriors`, `EFFPriors`,
  `ParallaxPriors` etc. exist because deriving prior bounds from `nanstd`/`nanmin`/`nanmax` of the
  observed data is the data used twice — the P01 referee raised exactly this ("*appear arbitrary*").
  Half-Cauchy, scale-free, following Olivares et al. (2018). Never reintroduce a data-derived bound
  without labelling it empirical Bayes.
- **Likelihoods are unbinned point processes**, `Σ log λ(rᵢ) − Λ`, not Gaussians on binned densities.
  Binning discards the Poisson structure precisely in the sparse outer bins that set `R_t`.
- **Per-star uncertainties enter the likelihood.** Treating precomputed distances as exact
  observations discards the parallax errors entirely.
- **Models return the trace.** Collapsing a posterior to mean±sd on exit is how `dynamics.py` ended
  up with uncertainty-free tidal radii.
- **Fixed-`ν` HalfStudentT, not HalfCauchy** — see the warning in `KingPriors`; a PyTensor numba
  `CauchyRV` bug (issue #2308, PR #2309) makes the two non-identical in sampling.

## Things that are easy to get wrong here, because they already were

- **King's `+ b` is not King's**, and on a Gaia membership-selected sample it is not field
  contamination — it absorbs the corona. For NGC 6383 it claims **56% of the sample** against a
  measured false-discovery rate of 6.1%. Do not report `k`, `R_c`, `R_t` without the fraction
  attributed to `b`. See `docs/design-notes/king_model_validity.md`.
- **`R_t` is prior-determined** for this cluster and unidentifiable even inside the Jacobi radius.
  Any `R_t` quoted must name its prior.
- **The EFF `γ` estimator is not usable at census-typical geometry.** The controlling variable is the
  footprint-to-scale ratio, not `N`: at `r_tot/a = 2` a true `γ = 2` is recovered as **3.6**, and true
  values of 2.00/2.32/3.00 come back as 3.58/3.78/4.21 — shrunk and shifted, not merely biased. At
  NGC 6383's ratio of 42 the bias is +0.075 ± 0.012 at `N = 628`. Verified not to be a sampler
  artefact and not removable by widening the prior. See `tools/validation/eff_gamma_bias.py`.
- **Circular symmetry is the exception**, not the norm — Tarricq+2022 find median axis ratio 0.71 —
  and every profile here assumes it.
- **`synthetic.py` is for validation, not science.** Its `noise` parameter defaults to 0 because the
  Goodwin & Whitworth per-level jitter manufactures a ×1.8 central cusp; read its docstring before
  raising it.

## Documentation standard — paper-level, not code-level

Every model, prior and physical constant here reaches a paper. The docstring is where its
justification lives, and it is held to the standard a referee would apply to a methods section, not
to the standard of "explains what the code does".

**Each model or likelihood must carry:**

1. **The formula, in LaTeX**, with every symbol defined. Not prose describing the formula.
2. **The primary citation, with bibcode**, verified on ADS/SciX — and, when the implemented form
   differs from the cited paper's, an explicit statement of the difference. (Worked example: the
   `+ b` in `king_profile` is *not* in King 1962 Eq. 14; the docstring says so and names the folk
   practice.)
3. **Whether it is empirical or dynamical.** King 1962 is a fitting formula by its author's own
   description; King 1966 is a dynamical model. Conflating them is a physics error, not a naming one.
4. **The validity range and what breaks outside it**, with numbers where they are measured.
5. **The known degeneracies**, named. Run `tools/validation/degeneracy_audit.py` before quoting
   parameters; anything above |r| = 0.9 is not a separate measurement.

**Each prior must additionally carry:**

6. **Where the numbers came from.** "Independent of the data being fitted" is necessary but not
   sufficient — the *values* need a provenance or they are arbitrary constants with a good excuse.
   The pattern to follow is `EFFPriors`: scales set from an **external catalogue** (Hunt & Reffert
   2024, regenerable via `tools/validation/fetch_hr24.py`), with the percentiles tabulated in the
   docstring so a reader can check the choice rather than trust it.
7. **The distributional argument.** Why half-Cauchy and not half-normal? Because the tail is heavy
   enough that an order-of-magnitude error in the scale costs little — cite Gelman (2006), Polson &
   Scott (2012), and the in-field precedent (Olivares et al. 2018).
8. **Where the prior is NOT neutral**, measured. `EFFPriors.gamma_mu` carries the measurement that
   at a footprint-to-scale ratio of 2 the recovered slope is biased +1.6 and true values of
   2.00/2.32/3.00 return as 3.58/3.78/4.21. A prior that dominates somewhere must say where.
9. **Values that are NOT justified must say so.** `k_scale` and `b_scale` are order-unity because
   surface densities are order unity, not because a catalogue was consulted, and the docstring
   states that in those words. An honest gap is documentation; a silent one is a defect.

## Units policy — because they have already gone wrong twice

Units are part of the physics, and both failures here were **silent**: no exception, plausible
numbers, wrong by orders of magnitude.

1. **Hunt & Reffert publish angular radii in degrees.** Reading them as arcmin understates every
   radius 60-fold and would have set a prior scale two orders of magnitude wrong. Caught by checking
   the implied physical radius against the catalogue's own `rcpc` column, which agrees to the digit
   once the units are right.
2. **`quantity_values(x)` with no target unit strips whatever unit `x` carries.** Handing a profile
   radii in degrees where arcmin was meant returned values wrong by up to **480×** — `king_profile`
   gave 0.896 instead of 0.0019 at `r = 20′` — with no error raised.

### The rules

* **Always pass the target unit**: `quantity_values(x, u.arcmin)`, never `quantity_values(x)`, for
  anything dimensional. It converts a `Quantity` correctly, lets a plain array through under the
  documented convention, and **raises `UnitConversionError` on the wrong physical type** (a length
  where an angle is meant). Verified in `tests/test_structure.py::test_profiles_respect_the_unit_of_the_radius`.
* **State the convention for bare arrays in the docstring.** Here a plain array of radii is **arcmin**.
  An undocumented convention is the same defect one layer down.
* **Prefer `.ecsv` over `.csv` for any table with dimensional columns.** ECSV carries units in its
  YAML header, so a round-trip preserves them; CSV silently discards them and the next reader guesses.
* **Cross-check every ingested quantity** against an independent column or a physical expectation
  before it enters a model. The degrees/arcmin error was caught this way and nothing else would have
  caught it.

### On `@u.quantity_input`

Astropy's decorator is the idiomatic validator — `@u.quantity_input(radius='angle')` rejects a
`Quantity` of the wrong physical type. It is **not** used here, for a stated reason: the astropy docs
note it *"checks for unit compatibility but does not perform unit conversions on the input"*, and this
package accepts bare arrays throughout. `quantity_values(x, unit)` validates **and** converts **and**
admits plain arrays, which is the behaviour this API needs. Use the decorator in any new code path
that requires a `Quantity`.

## Adding a profile or model

The point-process machinery takes any `Σ(r)` that has a **closed-form radial integral** — that
integral is evaluated at every leapfrog step, so quadrature inside the PyTensor graph is not viable.
Follow `king_expected_count` / `eff_expected_count`: derive the closed form, verify it against
`scipy.integrate.quad` to ~1e-9 across several decades of parameter space, and add it to
`compare_radial_profiles` so it is scored by Bayes factor rather than asserted.

Design decisions go in `docs/design-notes/decisions.md`, **append-only**, recording the number that
was wrong and why — not only the fix.
