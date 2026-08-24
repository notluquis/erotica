# Decision log — why the code is the way it is

_A running record of **non-obvious choices**: bugs found and how they were fixed, defaults that
encode a scientific judgement, and things deliberately left alone. Written for the person who later
asks "why is this like this?" — including a referee, a co-author, or us in six months._

**Conventions.** Each entry states the **symptom**, the **cause**, the **fix**, the **oracle** the
test checks against, and any **number that moved**. Entries are append-only; if a decision is
reversed, add a new entry rather than editing the old one.

---

## 2026-07-27 — `R_0` was inconsistent inside a single chain

**Symptom.** `analysis/dynamics.py` carried **two different values of the solar galactocentric
distance**: `8.125 kpc` in `calculate_galactic_mass`, and `8.3 kpc` in
`calculate_galactocentric_distance` and `calculate_hill_radius`. All three feed the *same*
Hill-radius computation, so a single call mixed two Galactic geometries.

**Cause.** Independently written defaults, never cross-checked; no test compared them.

**Fix.** One module constant, `dynamics.SOLAR_RADIUS` (`erotica/analysis/dynamics.py:25`), used as
the default in all three. The **value was not changed** — 8.125 kpc, the value already in
`calculate_galactic_mass` — so this commit is a pure de-duplication, not a re-calibration.

```{warning}
**The adopted 8.125 kpc does not correspond to any verified published measurement** `[I]`.
Its provenance in this codebase is unknown — it predates the current history and no source is cited.
The best abstract-verified modern value is **R_0 = 8178 ± 13(stat) ± 22(sys) pc**, i.e.
**8.178 kpc**, from GRAVITY Collaboration 2019, *"A geometric distance measurement to the Galactic
center black hole with 0.3% uncertainty"*, A&A 625, L10, `2019A&A...625L..10G`,
doi:10.1051/0004-6361/201935656 `[S — quoted from the abstract]`.

Adopting 8.178 kpc is a **science decision, not a bug fix**: it moves every galactocentric and
Hill-radius number by a further ~0.6%. **Deferred to an explicit call by the author.**

⚠️ Two citations that appear in earlier drafts of this note were wrong and are recorded here so
they are not repeated: `2018A&A...615L..15G` is the *gravitational-redshift* paper (its abstract
reports f = 0.90 ± 0.09, not R_0), and `2021A&A...647A..59G` is *"Improved GRAVITY astrometric
accuracy from modeling optical aberrations"* — **not** an R_0 determination, though it does state it
resolves earlier systematic discrepancies in R_0.
```

**Oracle.** Not a golden number — the law of cosines has closed forms the test checks exactly:
at `l=0, b=0` it collapses to `R_gc = |R_0 − d|`; at `l=180, b=0` to `R_gc = R_0 + d`.
See `tests/test_dynamics.py::test_galactocentric_distance_matches_closed_form_toward_centre`.

**Numbers that moved.** Any quantity computed through the `8.3 kpc` path shifts by the R_0 change
(≈2%): galactocentric distance, and the Hill radius through it. `calculate_galactic_mass` with the
default `legacy_power_law` model does **not** use `solar_radius` at all, so enclosed-mass numbers
are unaffected unless `model='solar_scaled'` was used. **A re-derivation against
`data/test/NGC6383/` is required before these appear in print.**

---

## 2026-07-27 — `solar_radius` was silently dropped on one branch

**Symptom.** `calculate_hill_radius(center=..., solar_radius=X)` ignored `X`. The sibling branch,
`calculate_hill_radius(distance, l, b, solar_radius=X)`, honoured it. Same call, same keyword, two
behaviours depending on which optional argument the caller used.

**Cause.** The `center` branch called `calculate_galactocentric_distance(...)` without forwarding
`solar_radius`, so it silently fell back to the default.

**Fix.** Forward the keyword on both branches.

**Oracle.** `test_solar_radius_override_is_honoured_on_every_path` passes a deliberately extreme
`R_0 = 4 kpc` and asserts the result *changes* on **both** routes. A test that only checked the
result was finite would have passed throughout the bug's lifetime.

---

## 2026-07-27 — the default NUTS backend was broken, and no test could have caught it

**Symptom.** Every Bayesian entry point failed on a real call:

```
TypeError: build_kernel.<locals>.kernel() got an unexpected keyword argument 'progress_bar'
```

`SamplingConfig.nuts_sampler` defaulted to `"blackjax"`, which is incompatible with the installed
blackjax 1.6.2 / pymc 6.1.0. **So `pip install -e ".[bayes]"` produced a package whose every
Bayesian model raised on first use.** Three entry points carried the default: `inference.py:24`,
`_isochrone.py:950`, `analyzer.py:353`.

**Why it went unnoticed — the deeper finding.** *No test in the suite had ever sampled a PyMC model.*
`tests/test_inference.py` monkeypatched the sampler away; `tests/test_structure.py` fed a `_FakeVar`
trace and never called `RDP_bayesian`; and CI installed `.[dev]`, not `[bayes]`. The Bayesian paths
that produce every published number were entirely unexercised. A JOSS reviewer installs and runs.

**Fix.** Default to `"pymc"` — PyMC's built-in NUTS, always present when the extra is installed.
`blackjax` and `numpyro` remain available opt-in for speed. Verified working: recovers
`mu = 0.8948 ± 0.0027` against an injected truth of 0.90.

**Oracle.** Injected truth, not a golden number — synthetic parallaxes drawn from a known
`(mu, sigma)` must be recovered, plus the convergence floor from `~/phd/methodology.md` PART A
(Vehtari+2021): **R-hat < 1.01, bulk-ESS > 400, zero divergences**. See
`tests/test_inference.py::test_fit_parallax_model_recovers_injected_truth`.

**Also fixed in the same pass:** CI gained a `test-bayes` job that installs `.[dev,bayes]` and
**asserts the extra actually imported**, so these tests execute rather than silently skipping.
It is a separate job because pymc>=6 requires Python >=3.12 while the matrix still covers 3.11.

```{note}
`arviz >= 1` returns a **display-formatted** summary frame — `r_hat` and `ess_bulk` arrive as
strings, so `summary["r_hat"].max() < 1.01` raises `TypeError: '<' not supported between instances
of 'str' and 'float'`. Coerce with `pd.to_numeric(..., errors="coerce")` first.
```

---

## 2026-07-27 — a mutation audit falsified "every test has an independent oracle"

**39 mutations applied to the shipping source, one at a time; 21 caught, 18 survived. A 46%
survival rate.** The claim made repeatedly in this log — that each new test has an oracle
independent of the code under test and can fail for the reason it was written — **does not hold.**

Three failure modes, and the third is the worst.

**1. Ratio-only oracles leave prefactors free.** Every test of `tidal_radius_prior` checks a *ratio*
(`r_J ∝ M^(1/3)`, or `doubled/base`), so the prefactor cancels. Changing `(M / 2M_G)^(1/3)` to
`(M / 3M_G)^(1/3)` — a **14.5% shift in the Jacobi radius**, the physical boundary the entire `R_t`
argument rests on — passed the suite. The only absolute assertions on it were `isfinite` and `> 0`.

**2. Self-widening tolerances.** `assert abs(estimate - truth) < N * posterior_std` **cannot fail for
the reason it was written**: a mutation that destroys the constraint inflates `posterior_std`, and
with it the tolerance meant to catch the mutation. Measured on the parallax test: the unmutated fit
gave `sigma = 0.00933 ± 0.00510`, so the window was **±0.0204 on a 0.010 quantity** — twice the value
being measured, before any mutation. This pattern was used throughout `test_inference.py` and in
`test_structure.py`.

**3. Tests that reimplement rather than exercise.** `test_priors_do_not_depend_on_the_data` and
`test_prior_predictive_covers_the_plausible_range` rebuilt the prior block **inline with their own
`pm.Model()`** instead of calling `_king_model`. They tested a copy. Putting `sigma=np.std(r)` back
into the shipping model — **the exact "data used twice" defect the P01 referee raised** — left the
suite green.

### What was repaired, each verified by re-applying the mutation

| repair | mutation it now catches |
|---|---|
| prior tests call `_king_model` / the real function | `R_c` prior `sigma=np.std(r)` |
| absolute tolerances replace `< N·std`, plus a bound on the posterior width itself | `sqrt(σ² + e_i)` — the dropped square |
| the parallax fixture's per-star errors scaled down 5× | as above; see the note below |
| `zero_point_scale` asserted as a **literal**, not read from the dataclass the model reads | hardcoding 8 µas in the model |
| absolute Jacobi check against the closed form | prefactor `2 → 3` |
| `posterior_summary` tested on a **lognormal**, not `np.arange` | reporting the mean as the median; swapping `minus`/`plus` |
| `cos(b)` tested at `b = 0, ±15, 45, −30°` | deleting `cos(b)` from all three sites |

```{admonition} The fixture change matters more than the tolerance change
:class: important
Three mutations survived even after the tolerances were made absolute, and measuring why was the
real finding: **the test data sat in a regime where the parameter is not identifiable.** With
per-star errors ≫ the intrinsic spread, `sigma_parallax` is barely constrained (posterior std half
its own value), and dropping the square moved the estimate by *less than a tenth of its uncertainty*
— in fact **toward** the truth. No tolerance can separate that.

Scaling the fixture's errors down 5× so they are comparable to the injected spread changed the
recovered value from `0.00933 ± 0.00510` to `0.00981 ± 0.00084` — a **6× tighter posterior**, 2%
accuracy — and the mutation is now caught. *A recovery test is only a test where the parameter is
recoverable.*
```

```{warning}
**Known gap, deliberately left open.** The off-diagonal of the per-star proper-motion covariance is
**not tested**: the fixture draws `e_ra` and `e_dec` from the same scale, so `ρ·e_ra·e_dec` and
`ρ·e_ra·e_ra` are numerically indistinguishable. Setting `e_dec = 3·e_ra` makes it testable, but in
that regime the fit stops recovering the intrinsic correlation — it returns **+0.37 for an injected
0.0**, with `sigma_Dec` 30% low. Whether that is an identifiability limit or a modelling error is
**unresolved**; a longer investigation timed out.

Shipping a test tuned until it passed in a regime that is not understood would repeat exactly the
failure this audit exposed, so the gap is recorded instead. See `~/phd/open-threads.md` C5.
```

**Still outstanding**, listed so they are not mistaken for fixed: `distance_model` still survives
`σ = (hi − lo)` instead of `(hi − lo)/2`; `king_unbinned`/`eff_unbinned` still survive
`nanmedian → nanmean`; `compare_radial_profiles` has **no behavioural coverage under
`-m "not slow"`**, so model selection can invert silently; `photometry.add_photometric_errors` has
none anywhere.

**Suite 390 → 397.** The mutation harness itself was validated with a positive control before use,
and the working tree was verified clean afterwards.

---

## 2026-07-27 — two dispatch bugs in `dynamics.py`, and posteriors now reach it

Found while wiring posterior propagation (plan step 3). Both fail on **scalars** too, so
neither is an array-broadcasting issue — they were simply never called.

**Bug 1 — the keyword Galactic call form raised.** `calculate_galactocentric_distance` supports five
call forms. The guard choosing the equatorial branch fired on `distance is not None` **alone**, so

```python
calculate_galactocentric_distance(distance=d, l=..., b=...)
```

— the Galactic form the docstring advertises — fell into the *equatorial* branch with `ra=None` and
raised inside `SkyCoord`. Only the positional `(distance, l, b)` form ever reached the Galactic
code. The existing test even documented this as intended: *"NB: passing distance=/l=/b= as keywords
routes to the equatorial branch instead"*. **That comment was a bug being recorded as behaviour.**

The fix has to discriminate all five forms, because a naive "equatorial requires ra/dec" guard breaks
`calculate_hill_radius(center=...)`, which passes `(ra, dec)` positionally *with* a keyword
`distance`. Caught by the existing `solar_radius` test — which is exactly what it was written for.

**Bug 2 — `tidal_radius_prior` was broken on its default path.**
`calculate_galactic_mass` returns a bare `Quantity` when no error is supplied and a `(mass, err)`
tuple when one is. `tidal_radius_prior` unpacked two values unconditionally, so with the **default**
`galactocentric_distance_err=None` the call raised `ValueError: too many values to unpack` (arrays)
or `TypeError: 'Quantity' object ... is not iterable` (scalars).

```{note}
**Corrected 2026-07-27 after an audit.** This entry originally said the bug was "in two places",
because `calculate_hill_radius` contains the identical unconditional unpack and both were patched
together. The audit established that **only `tidal_radius_prior`'s was reachable** —
`calculate_hill_radius` always supplies a non-`None` error on the path that reaches it, so its copy
never raised. The fix to both is still correct (the second is defensive), but the claim that two
functions were broken was an overstatement.
```

```{warning}
`tidal_radius_prior` is the function PART J recommends for the King `R_t` prior — the physical
Jacobi-radius prior that the whole `R_t` argument now rests on. It was unusable as documented. The
Jacobi-prior fits reported in `king_model_validity.md` avoided it only because the Hunt & Reffert
value was passed as a literal, and are unaffected.
```

**Oracles.** All five call forms are pinned and the three equatorial spellings must agree exactly;
equatorial and Galactic must agree for the *same* sky position (converted, not a rounded `l`/`b`);
`r_J ∝ M^(1/3)`, so 8× the mass must double the radius, to `rel=1e-6`.

**Posterior propagation (plan step 3).** The dynamics functions turned out to be array-safe already,
so what was missing was a tested path and a way to report the result. `dynamics.posterior_summary`
reduces a sample of a derived quantity to median plus an equal-tailed credible interval, preserving
units, without pretending the distribution is Gaussian. A test pushes 500 posterior draws of mass,
distance, half-mass radius and velocity dispersion through the galactocentric distance, Jacobi
radius, crossing time and relaxation time, and asserts each comes out with the right shape **and a
non-zero spread** — a scalar answer means the uncertainty was silently dropped.

**Photometry coverage.** `photometry.py` produces the cluster mass, the mass sets the Jacobi radius,
and the Jacobi radius is the axis the `R_t` argument turns on — so a wrong mass moves a physical
boundary, not a table entry. New `tests/test_photometry.py` checks the mass–luminosity chain against
closed forms (distance modulus vanishing at 10 pc; a factor 100 in luminosity per 5 mag; the `M^3.5`
inversion), isochrone mass assignment against a synthetic track where stars placed on the track must
get their own mass back, and the mass → Jacobi radius link against the analytic cube-root scaling.

**Coverage:** `dynamics.py` 53% → **68%**, `photometry.py` 41% → **50%**, `units.py` 55% → **81%**;
total 59% → 62%. Suite 363 → **386**.

---

## 2026-07-27 — a residual parallax zero-point, as a nuisance parameter

**Why.** Gaia's published parallax zero-point correction is, in its own authors' words, *"not
perfect"*, and its use is *"at the researcher's discretion"* (Lindegren et al. 2021,
`2021A&A...649A...4L`). The residual is **spatially correlated on the scale of a cluster**, so every
member carries essentially the *same* leftover offset — which no amount of averaging removes.
Maíz Apellániz, Pantaleoni González & Barbá (2021, A&A 649, A13, `2021A&A...649A..13M`) estimate
the angular covariance at zero separation at 106 µas², *"yielding a minimum (systematic) uncertainty
for EDR3 parallaxes of **10.3 µas** for individual stars or compact stellar clusters"* — quoted from
their abstract.

```{warning}
An earlier version of this entry attributed that number to Vasiliev & Baumgardt (2021,
`2021MNRAS.505.5978V`). **That was wrong.** Both bibcodes appear in `~/phd/methodology.md` PART J
and the figure was attached to the wrong one, then propagated into a shipped docstring. Caught by
an independent audit, not by review. The lesson is the one already in this log: a number taken from
a summary must be round-tripped to the source abstract before it enters code.
```
Riess et al. measure a residual `zp = −3 ± 4 µas` in open clusters.

**Fix.** `fit_parallax_model(zero_point=True)` adds one nuisance offset shared by every member,
prior width `ParallaxPriors.zero_point_scale = 0.0103 mas`. It is **exactly degenerate with
`mu_parallax`** for a single cluster, and that is the point: the degeneracy propagates the correlated
systematic into the reported uncertainty instead of leaving it out. PART J notes that a full
covariance matrix buys only ~30% over this, which is why one nuisance parameter is the right call.

**Oracle.** Quadrature: enabling it must widen `mu_parallax` to `hypot(sd, floor)`, must not move the
mean, and must leave the intrinsic depth alone. All three are asserted.

**On the 321 published NGC 6383 members:**

| | `mu_parallax` (mas) | distance |
|---|---|---|
| per-star errors only | 0.9025 ± **0.0035** | 1.1080 ± 0.0043 kpc |
| plus ZP nuisance | 0.9023 ± **0.0109** | 1.1083 ± 0.0135 kpc |

`0.0109 = hypot(0.0035, 0.0103)` exactly. **The uncertainty on the mean parallax triples**, and the
mean itself does not move.

```{admonition} The general point, which outlives this cluster
:class: important
With 321 members the **statistical** error on the mean parallax (3.5 µas) is **smaller than Gaia's
correlated systematic floor** (10.3 µas). Past a few hundred stars, adding members no longer buys
precision on a cluster's mean parallax — the systematic has taken over. Any paper quoting a
statistical-only error on a cluster mean parallax is overstating its precision, and the more members
it has, the worse the overstatement.
```

**No correction to P01.** The paper quotes **1.11 ± 0.06 kpc**, which is more conservative than even
the ZP-inflated interval here (±0.014 kpc). Nothing moves.

---

## 2026-07-27 — the fourth model: `velocity_model` had all three defects too

**I said the `inference.py` sweep was complete after parallax, proper motion and distance. It was
not** — `velocity_model` was missed, and it carried the same three defects, one of them flagged
explicitly in `~/phd/methodology.md` PART J as a ten-minute fix.

```python
mu_v  = pm.Normal("mu_v", mu=float(np.nanmean(velocity_values)), sigma=10)   # data-derived
std_v = pm.Uniform("std_v", lower=0, upper=40)                               # ~80x too wide
pm.Normal("observed_velocity", mu=mu_v, sigma=std_v, observed=velocity_values)  # no errors
```

**`Uniform(0, 40)` km/s is roughly 80× too wide** for a quantity that sits near 0.3–1 km/s in open
clusters. In the low-dispersion regime that leaves the posterior prior-dominated rather than
data-dominated — the parameter reports the prior's shape, not the cluster's.

**Fix.** `VelocityPriors` with fixed constants (`HalfNormal(sigma=2)` on the dispersion, zero-centred
`Normal(sigma=50)` on the mean), plus optional per-star velocity errors entering as
`sqrt(sigma_int² + e_i²)` — the same treatment the other three models now get.

**Oracles.** Injected internal dispersion 0.4 km/s against per-star errors of 0.8–3.0 km/s: the
error-aware fit recovers it, the naive one reports at least twice as much. And a prior-predictive
check that fails for the *reason it was written* — the new prior must put **>60% of its mass below
2 km/s** while still reaching 5 km/s in its tail, where `Uniform(0, 40)` put under 10% below 2 km/s.

**Lesson recorded rather than quietly fixed:** "the sweep is complete" was asserted after auditing
three of four models. The count came from the models I had already opened, not from the module. A
sweep is over the module, not over the parts of it already in view.

---

## 2026-07-27 — Bailer-Jones distance uncertainties enter the distance model

**Symptom.** `distance_model` built `prior_mu_r` from `nanmean([nanmean(1/parallax), nanmean(distances)])`
**of the data being fit**, centred a `Uniform(0.5x, 1.5x)` on it, put
`HalfNormal(sigma=nanstd(distances))` on the spread, and then fitted
`Gamma("r", observed=distances)` — treating Bailer-Jones geometric distances as **exact**. The data
used twice, twice over, plus a likelihood that discards the catalogue's own uncertainties.

**Fix.** `DistancePriors` (fixed constants) and, given `r_lo_geo`/`r_hi_geo`, a genuine hierarchy:

$$r^{\mathrm{true}}_i \sim \mathrm{Gamma}(\mu_r, \sigma_r), \qquad
  r^{\mathrm{obs}}_i \sim \mathcal{N}(r^{\mathrm{true}}_i, \sigma_i), \qquad
  \sigma_i = (r_{\mathrm{hi}} - r_{\mathrm{lo}})/2 .$$

> **Superseded 2026-08-24.** This is the hierarchy as it shipped, and it is what produced the
> `std_r = 0.0234 ± 0.0106` in the table below. Sampling those latents was a *centred*
> parameterisation and it funnelled; the population is now `\mathcal{N}` and the latents are
> integrated out in closed form. The old formula is left standing because the entry below it only
> makes sense against it.


**Numbers** (injected depth 20 pc; Bailer-Jones fractional errors 4–14%, median σ = 104 pc):

| | `mu_r` | `std_r` | implied depth | vs truth |
|---|---|---|---|---|
| truth | 1.110 kpc | 0.020 kpc | 20 pc | — |
| naive | 1.1144 ± 0.0077 | 0.1110 ± 0.0054 | 111 pc | **5.6×** |
| error-aware | 1.1156 ± 0.0068 | **0.0234 ± 0.0106** | **23 pc** | 1.2× |

The naive `std_r` is not merely wrong, it is **the median catalogue uncertainty** (111 vs 104 pc):
the parameter named "cluster depth" was reporting Bailer-Jones' error bars almost exactly. The mean
distance is unaffected in both.

`parallax_column` is retained for API compatibility and is no longer read.

**This closes three of the four models** — parallax, proper motion and distance now carry per-star
measurement uncertainty and scale-free priors, having all conflated measurement scatter with
intrinsic cluster spread in the same way. `velocity_model` was missed at the time; see the entry
above.

---

## 2026-07-27 — per-star proper-motion covariance enters the likelihood

**Symptom.** `proper_motion_2d_gaussian` centred `mu_RA`/`mu_Dec` on `nanmedian` **of the data**,
scaled every width by `nanstd` **of the data**, and ignored the per-star proper-motion covariance
entirely — even though `core/_error_aware.py` already builds the full correlated Gaia covariance.
The fitted `sigma_RA`/`sigma_Dec` were therefore the cluster dispersion *plus* Gaia's measurement
scatter, which for an open cluster is usually the larger term. Any virial mass or crossing time
derived from them is inflated.

**Fix.** Each star gets its own total covariance `Σ_i = Σ_int + C_i`, where `C_i` carries
`pmra_error`, `pmdec_error` and `pmra_pmdec_corr`, broadcast to `(n, 2, 2)` inside a single
`MvNormal`. Plus `ProperMotionPriors`, fixed constants.

**Numbers** (injected intrinsic dispersion 0.05 mas/yr; per-star errors drawn from Gaia-like
magnitude quartiles, median 0.12 mas/yr; **injected intrinsic correlation zero**, with 0.4
correlation present only in the measurement errors):

| | `mu_RA` | `sigma_RA` | `sigma_Dec` | `corr` |
|---|---|---|---|---|
| truth | −1.350 | **0.050** | 0.050 | **0.0** |
| naive | −1.360 | 0.2392 | 0.2201 | **+0.374** |
| covariance-aware | −1.348 | **0.0471** | **0.0581** | **−0.012** |

The naive fit inflates the dispersion ~4.8× **and reports the measurement-error correlation of 0.4
as an intrinsic kinematic correlation of the cluster**. That second failure is the more insidious of
the two: a spurious velocity-ellipse orientation is the kind of thing that gets interpreted
physically. The test asserts both.

```{note}
`corr` is sampled as `tanh(z)` with `z ~ Normal(0, 1)`, not `Uniform(-1, 1)`. At `|corr| = 1` the
covariance is singular and a hard uniform boundary lets NUTS propose arbitrarily close to it; with a
per-star covariance added, the Cholesky then fails. `tanh` keeps `|corr| < 1` strictly and is smooth.

Separately, the batched `(n, 2, 2)` covariance kills PyMC's multiprocess workers with `EOFError` on
this machine, so these fits run with `cores=1`. Sequential is ~70× slower per chain here and is the
correct trade.
```

---

## 2026-07-27 — per-star parallax errors now enter the likelihood

**Symptom.** `fit_parallax_model` fitted `Normal(mu, sigma, observed=parallax)` — every star treated
as measured **exactly**. Gaia supplies `e_parallax` per star and it was ignored, so the fitted
`sigma_parallax` absorbed measurement scatter as well as real cluster depth. For a Gaia sample the
measurement term is usually the larger of the two, which means the parameter named "cluster parallax
spread" was mostly reporting the error bars.

The default priors had the same defect as the King fit: `Uniform(0.5x, 1.5x)` centred on
`nanmean(parallax)` **of the data being fit**, and `HalfNormal(sigma=nanstd(parallax))`.

**Fix, one pass** (likelihood and priors together, for the reason given in the King entry):

$$\varpi_i \sim \mathcal{N}\!\left(\mu_\varpi,\ \sqrt{\sigma_{\mathrm{int}}^2 + \sigma_{\varpi,i}^2}\right)$$

via `parallax_error_column=`, plus `ParallaxPriors` holding fixed constants. `prior_distance=`
remains the informative path and is **not** data reuse — it comes from outside the sample.

**Oracle — an injected intrinsic spread deliberately much smaller than the measurement errors**,
which is the regime every Gaia open cluster is in. Per-star errors are drawn from the *real* median
`e_Plx` per `Gmag` quartile of the published member table (0.027, 0.065, 0.123, 0.293 mas), so the
error distribution is the observed one:

| model | `mu_plx` (truth 0.9000) | `sigma_plx` (truth 0.0100) |
|---|---|---|
| naive | 0.9151 ± 0.0072 | 0.1419 ± 0.0050 — **14.2× truth** |
| error-aware | **0.9036 ± 0.0026** | **0.0093 ± 0.0051** — 0.9× truth |

The naive fit misses the mean by 2σ as well; the bias is not confined to the width. The test
asserts both that the error-aware fit recovers truth **and** that the naive one is at least 5×
high — it cannot pass on a stopped clock.

**On the real published members** (N = 321, median `e_Plx` = 0.098 mas):

| model | `mu_plx` | distance | `sigma_plx` | implied depth |
|---|---|---|---|---|
| naive | 0.8986 ± 0.0032 | 1.113 kpc | 0.0568 mas | 70 pc |
| error-aware | 0.9025 ± 0.0034 | **1.108 kpc** | **0.0238 mas** | **29 pc** |

```{note}
**The distance does not move** — 1.108 vs 1.113 kpc, both well inside the published
1.11 ± 0.06 kpc. **No correction to P01 is implied.**

And P01 does not walk into the naive trap: it quotes the parallax dispersion from the subsample with
fractional parallax error below 0.1. That is a legitimate mitigation. It is also **partial and
expensive**: that subsample is **150 of 321 members**, so 53% of the data is discarded, and its
observed sd (0.0466 mas) still contains measurement scatter — roughly twice the 0.0238 mas the
error-aware model attributes to the cluster itself. Modelling the errors reaches the same goal using
**all** the members.

Even 0.0238 mas is a ~29 pc depth against a fitted `R_t` of 17.4 pc, so residual measurement error
plausibly still dominates. This should be read as an upper bound on the depth, not a measurement of
it.
```

**Both siblings are now fixed too** — see the proper-motion and distance entries above.

---

## 2026-07-27 — delta report: unbinned vs the published King numbers

**Measured against `data/test/NGC6383/`** with `tools/validation/king_unbinned_delta.py`
(4 chains, 2000 draws, seed 20260727; every fit R-hat ≤ 1.000, bulk-ESS ≥ 1910, **zero divergences**).

| quantity | published (binned, equal-count) | unbinned, same sample | shift |
|---|---|---|---|
| `R_c` (adopted, 40′ sample) | 1.96 (+0.19 −0.16)′ | **1.530 ± 0.262′** | −21.9%, **1.3σ combined** |
| `R_c` (70′ sample, *not adopted*) | 1.384 ± 0.039′ | 1.324 ± 0.209′ | −4.4%, well inside 1σ |
| `R_t` (adopted, 70′ sample) | 54 (+7 −11)′ | 80.2′, **SD 7113′** | unidentified from above |
| `R_t` (40′ sample) | 40 (+16 −17)′ | 107.5′, **SD 3190′** | unidentified from above |
| `b` (70′) | 0.020 | 0.024 | +20% |

**On `R_t`, this confirms and sharpens what P01 already says — it does not contradict it.** The
manuscript states plainly that "the upper bound 1.5·T_max = 63.7 arcmin **truncates the R_t
posterior of every fit**", that all quoted intervals "remain conditional on this physically motivated
prior", and that under a relaxed bound "the upper tail [is] still limited by the prior support rather
than turned over by the data" (§ structural, and Appendix D). `T_max` is **not** an arbitrary
coefficient: it is the larger of the Hill radius and the gravitational bound radius.

What the unbinned fit adds is the **magnitude**. With a scale-free half-Cauchy in place of the
truncating uniform, the `R_t` posterior SD is ~3200–7100 arcmin. The data do not merely fail to turn
the tail over — they supply **essentially no upper constraint at all**, and the reported `+7/−11`
is a property of the prior support, exactly as the paper says. The paper's own reading of `R_t` as
"a model-dependent scale rather than an independently measured physical boundary" is, if anything,
**understated**.

**On `R_c`, the shift is real but not significant.** 1.53 ± 0.26 against 1.96 (+0.19 −0.16) differs
by 0.43′; with both uncertainties combined that is **1.3σ**. Reported because it is the kind of
thing a referee re-deriving the fit would notice, not because it changes a conclusion.

```{admonition} The limit on all of the above — read before quoting any of it
:class: warning
**The unbinned likelihood is not validly applicable to the samples the paper fitted.** Its
normalisation `Λ = ∫ λ dA` is taken over the selection footprint, and `king_unbinned` assumes that
footprint is the complete disc of radius `field_radius`. The published fits run on
`paperfaithful_reference_p06.ecsv` — **p > 0.6 member lists**, where the footprint is whatever the
membership selection carved out, which is emphatically not a uniform disc.

So the deltas above are **indicative, not authoritative**. They answer "what happens if you apply
the correct likelihood to the sample as fitted", and part of the answer is "the sample is the wrong
shape for it". The paper independently reaches the same conclusion from the other direction: "the
background of this membership-selected sample measures the **residual contamination level of the
selection, not the raw field density**".

Fitting the raw 70′ cone instead (N = 78 477, correct disc footprint) gives `R_c` = 1.047 ± 0.424′,
`R_t` = 4.19 ± 3.19′, `b` = 5.09 stars arcmin⁻², i.e. the unfiltered cone is so field-dominated that
the cluster is a small perturbation — that fit is not comparable either, because the published
analysis applies quality and CMD cuts before clustering.

**The sound application needs a sample that is both cleaned and complete over a known footprint**,
with the selection function folded into `Λ`. `selection.py` already wraps Cantat-Gaudin+2023 and
Hunt+2026 and is not yet wired to the RDP — PART J flags that as an unclaimed novelty (a search for
selection-function-corrected OC radial density profiles returns **zero** papers). That is the real
next step, and it is a paper, not a patch.
```

**Nothing in P01 is falsified by this.** `R_c` moves 1.3σ; `R_t` behaves exactly as the paper
predicts it would under a relaxed prior. **No correction to the manuscript is implied by these
numbers**, and whether to tell a referee anything mid-review is the author's call, not this log's.

---

## 2026-07-27 — the King fit is now unbinned, and why not "Normal → Poisson"

**The plan was wrong, and measuring it said so.** The approved plan called for replacing
`pm.Normal("obs_density", …)` in `analysis/structure.py` with a per-annulus **Poisson likelihood on
counts**, on the stated grounds that "the Gaussian approximation fails in the sparse outer bins that
set `R_t`". Measuring the real profile refuted the premise: the published fit uses
`method="equip"` — **equal-count** annuli — so every bin holds 24–25 stars. There are no sparse
outer bins.

**And the proposed fix would have introduced a new defect.** Under equal-count binning the count per
annulus is fixed by construction; the annulus *area* is what varies between realizations. A Poisson
likelihood asserts `Var(N_i) = E(N_i)`. If the binner already conditioned on `N_i`, that assertion
is false — the same *data used twice* defect as a data-dependent prior, wearing a different hat.

`tools/validation/king_binning_likelihood.py` measures it against an oracle needing no fitting: the
**Poisson dispersion index** `Var(N_i)/E(N_i)`, which is 1 for a Poisson count *by definition*.
400 realizations of a known King point process:

| binning | mean dispersion | min E[N_i] | area CV | verdict |
|---|---|---|---|---|
| equal-count (`method="equip"`, current) | **0.045** | 49.6 | 16.4% | **not Poisson** |
| fixed-width | **1.006** | 24.0 | 0% | Poisson OK |

This is analytic, not merely empirical. With `n ~ Poisson(Λ)` split into `n_bins` equal-count
annuli, `Var(N_i) = Var(n)/n_bins² = Λ/n_bins²` while `E(N_i) = Λ/n_bins`, so the dispersion index
is **`1/n_bins`**. Measured across `n_bins = 10, 25, 50` it tracks that law (ratios 1.18, 1.19,
1.43; the drift at large `n_bins` is integer quantisation of `n/n_bins`, which bites once `E[N_i]`
falls to single digits). At the 25 bins the published fit uses, `1/25 = 0.04` against the 1.0 a
Poisson likelihood asserts: a **~25× mis-specification**.

**Fix: skip binning entirely.** `structure.king_unbinned` models the stars as an inhomogeneous
Poisson point process with intensity `λ(r) = 2πr Σ(r)`:

$$\log L = \sum_i \log \lambda(r_i) - \Lambda, \qquad \Lambda = \int_0^{R_f} 2\pi r\,\Sigma(r)\,dr$$

the continuous form of the Cash (1979) statistic, `1979ApJ...228..939C`, and the "unbinned, per
star" approach PART J identifies as best practice in the OC literature (Olivares et al. 2018,
A&A 612, A70, `2018A&A...612A..70O`). It is also **less** code than either binned option: no binner,
no bin-count choice for a referee to question, and **no `sigma` at all** — a point process has no
nuisance scatter parameter, so `HalfNormal("sigma", sigma=nanstd(density_values))` is deleted rather
than replaced.

`king_expected_count` evaluates `Λ` in closed form, so no quadrature runs inside the PyTensor graph.
Expanding `(u−c)²` gives three elementary integrals; the cluster term truncates at `min(R_t, R_f)`
and the background covers the whole disc.

**This lands three plan steps in one pass, which is why they could not be split.** The likelihood,
the priors and the returned posterior are entangled: change the likelihood and the density-scaled
prior bounds are in the wrong units, so splitting them would mean two re-derivations of `R_c`/`R_t`
and a delta nobody could attribute.

| Defect | Before | After |
|---|---|---|
| Likelihood | `Normal` on binned densities, shared `sigma` | unbinned point process, no `sigma` |
| Priors | all four bounds from `nanstd`/`nanmin`/`nanmax` **of the data being fit** | `KingPriors`, fixed constants |
| `R_t > R_c` | enforced by a data-derived `Uniform` bound | fits the **increment** `R_t − R_c`, true by construction |
| `R_t` ceiling | `Uniform(R_c, 1.5·T_max)` — physically motivated, but **truncating** (see below) | scale-free half-Cauchy, or a physical Jacobi prior via `tidal_prior=` |
| Posterior | discarded unless `return_trace=True` | `return_trace=True` **by default** |

**Oracles** (`tests/test_structure.py`, 17 tests). No golden numbers: every target is analytic or
injected.
* `Λ` → `scipy.integrate.quad` over five parameter sets spanning four decades, `rel=1e-8`,
  **including a field that stops inside `R_t`** so the truncation branch is exercised; plus
  monotonicity in `R_f` and the `k = 0` case where `Λ` must equal `b·πR_f²` exactly.
* the fit → parameters the test injects into a simulated point process, recovered within 3σ
  (measured: `R_c` 0.6%, `R_t` 0.1%), with the PART A convergence floor — **R-hat < 1.01,
  bulk-ESS > 400, zero divergences**. The `pm.math.switch` kink at every star's radius was a
  live concern; it produces **zero** divergences.
* prior independence → prior-predictive draws must be *identical* for two datasets differing by two
  orders of magnitude in size and concentration. If any prior read the data they would differ.

```{warning}
**PyMC 6.1.0 bug found while writing the prior-predictive test.** `pm.HalfCauchy(beta=…)` has the
correct `logp` — scale = `beta`, matching its own docstring — but its **random draws use `1/beta`
as the scale**. Confirmed at `beta = 0.5, 5, 20`, where the sampled IQR is wrong by exactly `1/beta²`;
`pm.Cauchy` is affected identically, `Normal`/`HalfNormal`/`Gamma`/`Exponential`/`Uniform` are not.

NUTS reads `logp`, so **posteriors are unaffected**. `sample_prior_predictive` reads the draws, so
**every prior-predictive check built on `HalfCauchy` is silently wrong** — which is how this was
found: the check failed with 0.7% of `R_t` draws above 30′ where the prior implies ~37%.

The priors are therefore built as `pm.HalfStudentT(nu=1, sigma=…)`, which *is* a half-Cauchy and is
correct in both paths (verified against `scipy.stats.halfcauchy` for density and IQR).
`test_half_cauchy_prior_is_built_without_the_pymc_halfcauchy_bug` pins both the workaround **and the
bug**, so a PyMC upgrade that fixes it makes the test fail and the workaround can be dropped.
```

**`RDP_bayesian` is unchanged and still exported.** It is what the paper's figure scripts call;
breaking it would break reproduction of the published figures. `king_unbinned` is additive.

```{admonition} Unsound regardless of likelihood: the background on a member list
:class: warning
Both published King fits run on `paperfaithful_reference_p06.ecsv` — a **p > 0.6 member list**
(N = 628 within 70′, `probability_hdbscan` ≥ 0.722), not a cone. A free additive background `b` is
defensible only if the fitted sample genuinely contains field stars over the fitted range. On an
already-selected member list `b` has nothing physical to absorb: it soaks up membership leakage and
is degenerate with `k` and `R_t`. This predates the likelihood question and is **not** fixed by
going unbinned.

It also breaks the unbinned normalisation, which assumes the footprint is the full disc: for a
member list the footprint is whatever the membership selection carved out. `king_unbinned` therefore
**refuses** samples with stars beyond `field_radius` rather than fitting them silently, and the
delta report fits the **full 70′ cone** alongside the member list to separate the two effects.
```

---

## 2026-07-27 — the reproducibility record recorded almost nothing, and nothing called it

**Symptom.** `analysis/provenance.py:146` was:

```python
def build_metadata(**kwargs):
    return {"created_at": ..., "erotica_version": ..., **kwargs}
```

A timestamp and a version string. **No git commit, no seed, no dependency versions, no checksum of
the input data.** Two runs of the same version against different catalogues, on different NumPy
releases, from a dirty working tree, produce byte-identical provenance.

**The worse half.** `grep` found **zero callers**. Nothing in the package ever built a metadata
record. Dead provenance code is worse than no provenance code, because the reproducibility claim
gets made anyway and there is a function name to point at.

**Fix.** `build_metadata` now records, and `store_trace_results` now *calls* it — every saved trace
gets a `*_provenance_<index>.json` sidecar next to its NetCDF:

| Field | Why it is there |
|---|---|
| `git.commit`, `git.dirty`, `git.branch` | A commit hash is misleading without `dirty` — the normal state during analysis is uncommitted edits. `None` outside a repo (a PyPI wheel), never an exception. |
| `inputs[].blake2b`, `inputs[].bytes` | Identifies input **content**. A catalogue that is renamed, re-downloaded or re-sorted into another path but is byte-identical is the same input; one flipped byte is not. Streamed, so a multi-GB file costs kilobytes of memory. |
| `seeds` | Kept even when `None` — a run with no fixed seed is *not* reproducible and the record has to say so rather than omit the field. |
| `dependencies` | Read from installed distribution metadata, **not** by importing each package: importing PyMC costs seconds and is a side effect a provenance call has no business causing. An absent optional extra is itself provenance (that run produced no Bayesian numbers). |
| `python`, `platform` | Cheap, and the first thing asked when a number fails to reproduce. |

**The invariant.** A provenance record that raises at write time destroys the result it was meant to
describe. So `build_metadata` calls `json.dumps` on itself before returning — failing loudly at
build time rather than after a six-hour sampling run — and `_jsonable` coerces the values that
arrive naturally from callers and would otherwise raise: NumPy scalars and arrays, `Path`,
dataclasses such as `SamplingConfig`, and sets. An unrecognised object is downgraded to `repr`,
never dropped: losing a field silently is the failure mode, losing its type is not.

**Oracles** (`tests/test_provenance.py`, 28 tests). Each is independent of the code under test:
* git → a `git` subprocess the test runs itself, plus a throwaway repository the test **builds,
  commits, dirties, restores, and dirties again with an untracked file**. A test asserting only that
  the key exists would pass against a hardcoded `False`.
* checksums → the published BLAKE2b-512 vector for the empty string
  (`786a02f7…e9be2ce`), plus digests the test computes with `hashlib` on its own; content-vs-path
  equality; a 40 MB multi-chunk file.
* dependencies → `module.__version__`, which is set independently of distribution metadata.
* `build_metadata` → `json.loads(json.dumps(m)) == m`, fed hostile values on purpose.
* `store_trace_results` → the sidecar exists, names the same `trace_index` as the `.nc` beside it,
  and carries the caller's checksum. This one is a **regression guard against the dead-code state**.
* `posterior_mode` → a lognormal, whose mode `exp(−1) = 0.368`, median `1.0` and mean `1.649` are
  analytically distinct; the estimator lands at 0.375. The easy way to fake a mode estimator is to
  return the mean or the median, and both fail this by a factor of ~3.
* `store → load` → round-trip identity, including the recovered draws, not merely the summary.

```{warning}
**The wiring reaches the legacy notebooks, not the current analysis path.** `store_trace_results` is
called from three committed notebooks under `data/test/NGC6383/`; **no module or script in the
current pipeline calls it**. So published NGC 6383 numbers still have no provenance record attached.
Closing that is a separate step: the call belongs wherever the paper's fits are driven from
(`review_repo/regen_king*.py`, `convergence_audit*.py`, `isochrone_nuts_refit.py`), passing the
input catalogue path and the sampler seed. Recorded rather than quietly left, so the entry above is
not read as "reproducibility: solved".
```

**Numbers that moved.** None — this is new information recorded alongside results, not a change to
any computation. Coverage of `provenance.py` **27% → 95%**; suite **290 → 318**.

Also renamed the leftover `_PUMPS_VERSION` import (from the COSMIC → PUMPS → EROTICA renames).

---

## 2026-07-27 — `RuntimeWarning` was globally ignored, hiding a real bug

**Symptom.** `pytest.ini` carried a blanket `ignore::RuntimeWarning`. In a numerics package that is
the category that says *divide by zero*, *invalid value in sqrt*, *overflow* — i.e. exactly how a
silent `NaN` reaches a published number.

**What the audit found.** Re-running the suite under `-W error::RuntimeWarning` produced only two
failures. One was third-party and harmless: PyTensor's graph rewriter divides by zero while constant-folding
a graph that is then discarded. The other was **ours**.

**The bug.** `analysis/_isochrone.py::_fit_error_model` guarded its constant-error fallback with:

```python
def _fit(mag_ok, e_ok, fallback):
    if mag_ok.sum() < 3:          # intended: "fewer than 3 valid points"
```

`mag_ok` is not a boolean mask — it is `obs_mag[valid_m]`, the **filtered magnitude array**. So
`.sum()` added up magnitudes. Two stars at G = 14 and 15 sum to 29, comfortably ≥ 3, so the fallback
**never fired for any non-empty input**. Two points were fitted with a degree-2 polynomial, the
Vandermonde matrix was rank-deficient, NumPy raised `RankWarning` — a `RuntimeWarning` subclass in
NumPy 2 — and `pytest.ini` swallowed it. The docstring meanwhile asserted the opposite, claiming
`Polynomial.fit` was chosen precisely to avoid `RankWarning`.

The test that covered this line asserted `np.isfinite(f_m(...)[0])`. Garbage from an
under-determined fit is finite, so the test passed for the whole life of the bug. This is the
house rule in the standing section, violated: *a test must be able to fail for the reason it was
written*.

**Fix.** Count **distinct abscissae**, `np.unique(mag_ok).size < 3` — rank deficiency is about
distinct magnitudes, not the point count, so six stars all at G = 15 must also fall back.

**Numbers that moved.** **None.** The old and new guards were compared across
`n = 0, 1, 2, 3, 10, 60, 321` points: they disagree only at `n = 1` and `n = 2`, and agree
everywhere else including the empty case. A real cluster fit never enters that regime, so no
published NGC 6383 quantity changes. The bug was latent, not active — but it was one degenerate
input away from producing a silently arbitrary error model.

**Fix to the filter.** `error::RuntimeWarning`, with three narrow `ignore` lines scoped to
`pytensor.tensor.rewriting.math` by message. Nothing of ours can be hidden again.

**Oracles added** (`tests/test_isochrone.py::TestFitErrorModel`): the fallback must return the
**exact median** (`0.015` for `e_m = [0.01, 0.02]`) and be constant in magnitude; six repeated
magnitudes must also fall back; a quadratic injected in log₁₀ space must be recovered to `rtol=1e-6`;
and zero/NaN/negative errors must be dropped *before* the branch is chosen.

---

## 2026-07-27 — the 2σ parallax clip is a magnitude-dependent selection function

**Status: measured, not yet changed.** The measurement is the deliverable; the fix is a science
decision that changes a published number and is sequenced with the P01 delta report.

**Symptom.** `analysis/_clipping.sigma_clip_parallax` clips on the **raw** parallax column at a
fixed multiple of the *sample* scatter. Gaia's per-star parallax uncertainty is strongly
magnitude-dependent, so a faint star with a large `e_Plx` is scattered further from the cluster
centroid than a bright one **even when both are members**. The cut therefore removes faint stars
preferentially. That makes it a selection function, not only an outlier test.

**The oracle — by construction, no golden numbers.** `tools/validation/parallax_clip_selection_function.py`
builds synthetic clusters in which **every star is a true member**: one common true parallax, no
contamination, no intrinsic depth. Per-star uncertainties are the **real** `e_Plx` values from the
published member table (`data/test/NGC6383/comments_paper/cds_final/ngc6383_members.ecsv`, 321
stars), so the magnitude-dependence of the errors is empirical rather than modelled. Any rejection
is then, unambiguously, a **false** rejection, and the retention rate per `Gmag` quartile *is* the
induced selection function.

**Numbers** (400 realizations, σ = 2, seed 20260727; `tools/validation/parallax_clip_selection_function.json`):

| | overall | Q1 bright | Q2 | Q3 | Q4 faint |
|---|---|---|---|---|---|
| **raw parallax (current behaviour)** | **67.5%** | 99.1% | 85.6% | 59.1% | **26.7%** |
| clip on normalized residual `(plx − centre)/err_i` | **95.7%** | 95.6% | 95.7% | 95.8% | 95.8% |

Median `e_Plx` per quartile: **0.027, 0.065, 0.123, 0.293 mas** — an 11× spread, which is the whole
mechanism. Bright→faint retention gradient: **raw +72.4 pp, normalized −0.2 pp**. The normalized
clip sits at ~95.7% everywhere, which is simply what a 2σ cut *should* retain.

So the current cut discards **about a third of genuine members**, and **roughly three-quarters of
the faintest quartile**.

```{admonition} CORRECTED 2026-07-27 — the attribution was wrong, the number was nearly right
:class: danger
An adversarial audit found the causal attribution above is **wrong**. The +72.4 pp gradient is not
the 2σ clip's. It is the gradient of **the ADQL query window plus the clip**.

The oracle draws synthetic member parallaxes with **no bound**, but the real cone was queried with a
parallax constraint: the raw file `data/70/NGC_6383_70-result.ecsv` spans **0.750–1.100 mas** and all
78 893 raw sources lie inside it. A faint-quartile member (median `e_Plx` = 0.293 mas) at a true
0.90 mas scatters over 0.02–1.78 mas — mostly **outside** that window, so in reality it never enters
the catalogue and the clip never sees it. The simulation charged those losses to the clip.

Re-running with the query window applied first, 300 realizations:

| stage | overall | Q1 | Q2 | Q3 | Q4 | gradient |
|---|---|---|---|---|---|---|
| clip alone, unbounded (the original, wrong, setup) | 67.6% | 99.1% | 85.8% | 59.3% | 26.7% | +72.4 pp |
| **ADQL query window alone** | 80.6% | 99.8% | 97.4% | 82.8% | **43.0%** | **+56.7 pp** |
| query window **then** clip (what actually happened) | 64.6% | 98.6% | 81.8% | 54.4% | 24.1% | +74.5 pp |

```{danger}
**CORRECTED AGAIN, 2026-07-27. The correction above was itself wrong.** It read "the query window
alone accounts for 78% of the effect", and that 78% is **an artefact of evaluation order**, not a
property of the data.

The two cuts are near-redundant — both act through the single channel of `e_Plx` growing with `G` —
so marginal contributions depend entirely on which is applied first:

| decomposition | window | clip |
|---|---|---|
| window first | 56.7 pp (**76%**) | 17.8 pp (24%) |
| clip first | 2.1 pp (3%) | 72.4 pp (**97%**) |
| **Shapley (order-symmetric, the only order-free split)** | **29.4 pp (39%)** | **45.1 pp (61%)** |

Shapley sums to 74.5 pp exactly, and it puts the **σ-clip ahead of the window** — inverting the
claim made in this entry. The ADQL was independently recovered from the ECSV header of the raw file
(`AND gs.parallax BETWEEN 0.75 AND 1.1`), and the +56.7 pp window figure was reproduced analytically
by a Gaussian-CDF calculation agreeing with the Monte Carlo to <0.2 pp — so the *numbers* are right
throughout. Only the ranking was wrong, twice.

**The defensible statement:** *either mechanism alone reproduces most of the joint +74.5 pp gradient;
they are not additive and cannot be rank-ordered by marginal contribution.* Any claim of the form
"X dominates" requires naming the decomposition, and the order-free one favours the clip.

Which means PART J's `[R3]` item 1 — change the clip — addresses at least as much of the problem as
anything else, contrary to what this entry previously concluded.
```

**What survives unchanged:** a strong magnitude-dependent selection is present, the joint gradient is
**+74.5 pp**, and it is larger than previously stated. That sharpens rather than softens the point
for P01's faint quartile.
```

```{admonition} What this experiment does *not* establish
:class: caution
**The 95.7% is close to tautological.** The synthetic draws each star's parallax with the same
per-star `err_i` that the normalized residual then divides by, so `z` is exactly `N(0,1)` by
construction and a 2σ cut retains ~95.4% whatever the magnitude distribution. The flat row is a
consistency check on the arithmetic, **not evidence that the normalized clip is better**.

**The experiment is one-sided: there is no contamination in it.** With every star a true member it
measures the false-*rejection* rate only. The normalized clip is necessarily more permissive, and
permissiveness has a cost this design cannot see: a field star at 0.6 mas from the centroid with a
large `e_Plx` gets `z ≈ 2` and *survives* normalisation, where the raw clip removes it. Retention
and contamination move together, and only one of them was measured.

**What does stand** is the raw clip's **+72.4 pp bright→faint gradient**, which needs no comparison
to interpret: it is a statement about the current cut alone, against an oracle that is true by
construction.

Before the delta report recommends any switch, this needs a contaminated arm — inject field stars
from the real 70′ cone at a distinct parallax and report retention of members **and** contaminants
for both cuts, i.e. an ROC rather than a single operating point.
```

**Why the alternative is *a priori* attractive.** On the normalized residual, a large uncertainty
buys *tolerance* rather than a rejection: the star is an outlier only if it is discrepant relative
to **its own** error. That is the same principle as the error-aware `f_i` in `core/_error_aware.py`.
It is the motivation for testing it, not a result of this test.

```{admonition} Implication for P01 — unproven, not disproven
:class: warning
P01 attributes a faint-quartile luminosity-function / KS signal to **Gaia incompleteness**. Q4 is
exactly the quartile this clip mutilates, so the pipeline supplies a **competing explanation of the
same signal** that the paper does not currently address. That is a referee-grade objection.

**This does not show the P01 result is wrong.** Both effects act in the same direction and can
coexist; nothing here quantifies their relative size on the *real* (contaminated) field. The
discriminating test is a re-run of the faint-quartile LF/KS on a normalized-residual-clipped sample.

```{admonition} RUN 2026-08-02 — the signal survives, and strengthens
:class: important
`tools/validation/faint_quartile_clip_test.py`. The published numbers are reproduced exactly first —
quartile edges 8.80 / 15.44 / 16.99 / 17.93 / 20.66 mag and `D` = 0.156, 0.230, 0.278 with
`p` = 0.418, 0.052, 0.0097 against the paper's 0.16, 0.23, 0.28 and 0.42, 0.052, 0.010 — so any change
is attributable to the change in the clip.

Replacing the raw 2σ parallax clip (retention gradient across magnitude quartiles **0.724**) with a
normalised-residual clip `|ϖ − ϖ₀|/σ_ϖ < 2` (gradient **−0.002**, i.e. magnitude-blind):

| faintest quartile vs | published `D` (`p`) | normalised-clip `D` (`p`) |
|---|---|---|
| brightest | 0.156 (0.418) | **0.230 (0.040)** |
| second | 0.230 (0.052) | **0.284 (0.0050)** |
| third | 0.278 (0.0097) | **0.335 (0.0003)** |
| | N = 254, **1** significant after Holm | N = 295, **3** significant after Holm |

**The raw clip was diluting the signal, not manufacturing it** — the opposite of the objection. And
the *ordering* is preserved: `D` still rises monotonically toward the adjacent quartile and is
weakest against the brightest, which is the pattern P01 uses to argue against dynamical mass
segregation. So the attribution to Gaia incompleteness survives, and so does its supporting argument.

**The objection is answered and the answer is favourable.** It should still be stated in the paper —
a referee who notices the clip's magnitude dependence will ask, and the answer is now measured.
```
```

**Tests.** `tests/test_clipping.py` pins the behaviour with two fast characterisation tests
(`test_raw_clip_rejects_large_error_stars_preferentially`, asserting the ≥15 pp bias exists, and
`test_normalized_residual_clip_is_precision_blind`, asserting the alternative is within 5 pp). They
are characterisation, not aspiration: they make any future change to the clip **visible** rather
than silent. The full measurement stays in `tools/validation/` because it needs the real catalogue
and 400 realizations.

**Not changed yet, deliberately.** Switching the clip changes the published NGC 6383 membership
list. Per the approved plan the sequence is *fix → re-derive → report deltas*, and the delta report
is the artefact the author needs before deciding what, if anything, to tell a referee mid-review.

---

## 2026-07-27 — `sigma_clip_parallax` has flag-dependent return arity

**Status: documented, not changed.**

`sigma_clip_parallax` returns 2, 3, 4 or 5 values depending on `in_place` and `return_mask`. This is
an API wart — a caller cannot unpack the result without knowing both flags.

**Why not fixed now.** It is public API with existing callers, including the paper's figure scripts.
Changing the signature is a breaking change that belongs in a version bump, not in a test pass.

**What was done instead.** `test_return_arity_is_flag_dependent` pins all four shapes, so a refactor
cannot alter them silently, and the wart is recorded here for the next major version.

---

## 2026-08-02 — three defaults that did not use the capability sitting next to them

Three separate defects, one shape: **the correct code already existed and the default path did not
reach it.** Fixed together because they are the same class of defect, measured apart because they
are not the same mechanism. Ladder, reproduction gates and every number below:
`tools/validation/wiring_fix_deltas.py`.

| where | default was | default is | old behaviour reachable via |
|---|---|---|---|
| `ClusterInferenceAnalyzer.distance_and_parallax_by_probability` | `parallax_error` read only to discard stars; errors, `r_lo_geo`/`r_hi_geo` and `zero_point` never forwarded | all three forwarded | `parallax_error_column=None`, `distance_lo_column=None`, `zero_point=False` |
| `ClusterStructureAnalyzer.fit_king_profile` | `RDP_bayesian` (binned Gaussian, data-derived priors) | `king_unbinned` (point process) | `method="binned"` — warns |
| `_clipping.sigma_clip_parallax` | 2σ on the **raw** parallax, `maxiters` implicit | 2σ on `(ϖ−ϖ₀)/√(σ_exc²+σ_ϖ,i²)`, `maxiters=5` explicit | `method="raw"` |

**Reproduction gates, run before anything changed.** Membership counts 701 / 412 / 321 / 254 at 40′;
KS statistics 0.156 / 0.230 / 0.278 against the published 0.16 / 0.23 / 0.28; the old-defaults
parallax fit returns ϖ = 0.9083 ± 0.0040 mas and d = 1.1166 ± 0.0052 kpc against a published
0.908 ± 0.004 and 1.11.

### The clip: the alternative that was planned is worse than what it replaces

The entry of 2026-07-27 above measures a **−0.002** retention gradient for a normalised-residual
clip, against **+0.724** for the raw one, and its own `{caution}` block says the −0.002 is "close to
tautological". That caution was right, and the number does not survive contact with the catalogue.
Measured on the 412 pre-clip members with the package's own estimators:

| clip basis | Q1 bright | Q4 faint | gradient | post-clip N |
|---|---|---|---|---|
| published (histogram mode + std, raw ϖ) | 94.2% | 72.8% | **+0.214** | 321 |
| raw ϖ, package estimators | 96.1% | 81.6% | +0.146 | 369 |
| `z = d/σ_ϖ`, scale re-fitted from the sample | 50.5% | 100.0% | **−0.495** | 327 |
| `z = d/σ_ϖ`, fixed 2σ | 81.6% | 100.0% | −0.184 | 389 |
| **shipped: `z = d/√(σ_exc²+σ_ϖ²)`, fixed 2σ** | **93.2%** | **100.0%** | **−0.068** | **403** |

The naive normalised clip does not remove the magnitude dependence — it **inverts and enlarges** it.
Two mechanisms, neither present in the simulation that produced the −0.002:

1. **It injects no excess scatter**, so `z` is `N(0,1)` by construction. The real sample has a
   maximum-likelihood excess variance of **σ_exc = 0.044 mas**, larger than the 0.030 mas median
   error of the brightest quartile, so dividing by σ_ϖ alone inflates `|z|` for the best-measured
   stars and clips them preferentially.
2. **It has no query window.** The real cone was queried `parallax BETWEEN 0.75 AND 1.1` — about
   1.2 median errors wide for the faintest quartile — which makes a large `|z|` *structurally
   unreachable* there. This is the same ADQL window whose contribution this log has already
   mis-attributed twice; it bites the normalised clip far harder than the raw one.

Three choices in the shipped form, each measured against its alternative and each pinned by a
mutation-verified test:

* **σ_exc is in the denominator**, matching `fit_parallax_model`'s `√(σ_int² + σ_i²)`.
* **The threshold is asserted at 2σ, not re-fitted.** Normalising already fixes the scale of `z` at
  1. The re-fitted scale on this sample is 0.71, i.e. a nominal 2σ cut acting at 1.4σ, 56 extra
  stars discarded.
* **The dispersion is fitted on the whole selection, never on the survivors.** Re-fitting a variance
  inside its own 2σ cut is the classic shrinkage, and here it does not bias the estimate, it
  destroys it: iterating on survivors drives σ_exc **0.0443 → 0.0241 → 0.0000 mas** in three passes
  and the bright-quartile retention 93.2% → 86.4% → 82.5%. Fitted on the full selection it is stable
  and the selection is identical at `maxiters` 1, 2, 5, 20, `None`.

```{important}
**σ_exc is an excess-variance term, not a cluster depth, and must never be reported as one.** At
ϖ₀ = 0.926 mas, 0.044 mas is 4.8% — an implied ~52 pc against this paper's own `R_t` = 17.4 pc and
against `DistancePriors`, which states in its own docstring that 50 pc is "larger than any bound
cluster". It absorbs real depth **plus** underestimated bright-end Gaia uncertainties (the brightest
quartile has observed robust scatter 0.0469 mas against a 0.0295 mas median quoted error, ratio
**1.59**) **plus** residual contamination. Same discipline as `p̃`: label the proxy.
```

```{caution}
**The contaminated ROC this log asked for still does not exist**, and every number above counts
rejections rather than *false* rejections. The clip default was changed anyway, because the raw
clip's magnitude dependence is measured and real — but the defensible statement is *"+0.214 became
−0.068"*, not *"the new clip is better"*. The normalised basis is necessarily more permissive and
nothing here measures what that costs in contamination.
```

**`maxiters` was not free.** astropy's implicit default of 5 was doing real work on the raw basis:
396 / 385 / 377 / 369 / 358 / 358 stars survive at `maxiters` = 1 / 2 / 3 / 5 / 10 / converged —
**38 stars, 9.2%**, between one pass and convergence. Pinned at 5, which is what astropy applied, so
making it explicit moves nothing.

**P01's faint-quartile conclusion was re-derived, not assumed.** The clip changes the member list, so
the KS test that supports the Gaia-incompleteness attribution had to be re-run on the list the
package now produces (`wiring_fix_deltas.py`, Stage A2):

| member list | N | D vs Q1 | D vs Q2 | D vs Q3 | significant after Holm |
|---|---|---|---|---|---|
| paper, Sect. 5 | 254 | 0.160 | 0.230 | 0.280 | 1 |
| published clip, reproduced | 254 | 0.156 | 0.230 | 0.278 | 1 |
| shipped normalised clip | 313 | **0.222** | **0.295** | **0.321** | **3** |

The signal **strengthens** (p = 0.036, 0.0021, 0.0006) and, decisively, the *ordering* is preserved:
`D` still rises toward the adjacent quartile and is weakest against the brightest, which is the
pattern P01 uses to argue against dynamical mass segregation. The raw clip was diluting the signal,
not manufacturing it.

### The pre-cuts

`ϖ > 0` and `σ_ϖ/ϖ ≤ 0.1` are removed from the default path. They discarded **124 of the 254
published members (49%)** — and 351 of 628 on the 70′ list — and they are the practice Luri et al.
(2018, A&A 616, A9, `2018A&A...616A...9L`) argue against: *"negative parallaxes, or parallaxes with
relatively large uncertainties still contain valuable information"*. Truncating on σ_ϖ/ϖ is a second
magnitude-dependent selection, imposed on the inference sample on top of the one in the member list.
With the errors in the likelihood the imprecise stars are down-weighted instead of deleted, which is
what the cut was approximating.

`zero_point=True` by default. Cruz Reyes & Anderson (2023, A&A 672, A85, `2023A&A...672A..85C`)
measure cluster parallaxes to ~7 µas from non-variable members and report that over 12.5 < G < 17
the Lindegren et al. (2021) corrections "are accurate and require no further offset corrections", so
the residual offset is consistent with zero and the prior is correctly centred there. **It widens the
uncertainty; it does not correct a bias**, and giving the nuisance a non-zero mean would turn it into
a correction that measurement does not support.

### A defect the fix exposed: the error-aware distance model does not converge at N > 250

**Status: measured, then fixed on 2026-08-24 by marginalising the latents. Everything below this line is the state *before* that fix and is kept for the record — see **"The latents are integrated out, and the funnel goes with them"** at the end of this file. The prohibition on quoting `mu_r`/`std_r` above ~250 stars was lifted by that fix.**

Turning on the Bailer-Jones bounds gives `distance_model` one latent `r_true` per star, and the
Gamma hierarchy over those latents does not sample adequately once the member list grows. Measured
in `wiring_fix_deltas.py --stage B`, PART A floor (R-hat < 1.01, bulk-ESS > 400, zero divergences):

| rung | N | R-hat | bulk-ESS | divergences | verdict |
|---|---|---|---|---|---|
| B0 old defaults (no latents) | 130 | 1.0004 | 2726 | 0 | pass |
| B1/B2 error-aware | 130 | 1.0009 | 579 | 0 | pass |
| B3, published clip | 254 | 1.0058 | **272** | 0 | **fails ESS** |
| B3, normalised clip | 313 | **1.0409** | **43** | **9** | **fails everything** |

The **parallax** model is clean at every rung (R-hat ≤ 1.0019, ESS 1275–2874, zero divergences),
including with the zero-point nuisance whose exact degeneracy with the mean was the obvious suspect.
So this is specifically the hierarchical distance model, and it is not the zero-point.

**Consequence for the deltas.** The only floor-passing distance comparison in the ladder is
B0 → B1/B2 at N = 130: `sigma_r` 0.0598 → 0.0330 kpc, −45%, against a published 0.06 that B0
reproduces exactly. That is the defensible D1 result on the distance side. **No `mu_r` or `std_r`
from a fit with more than ~250 stars should be quoted** until this is fixed.

**Remedy, deliberately not attempted here.** A non-centred parameterisation of `r_true` (or a higher
`target_accept`, or more draws) is a *modelling* change, not a wiring fix; putting it in the same
diff would make its own delta unattributable — the mistake this whole entry exists to avoid. It
needs its own diff and its own measurement.

### Also fixed, same defect class

Every model in `inference.py` called `_require_pymc()` *before* validating its input, so a malformed
table raised `ImportError` rather than the `ValueError` naming the actual problem. Validation now
runs first — which is what made eight input-validation tests unrunnable in the CI job that installs
`.[dev]` without the `bayes` extra.

`figures.graph_king` is pinned to `king_method="binned"` while the analyzer default moved. It exists
to reproduce published figures; re-plotting them from a different likelihood would move a published
curve without saying so.

### Mutations re-applied

21 of 21 killed. Two survived first and are worth recording, both instances of failure mode 4,
*fixtures in a non-identifiable regime*:

* **Dropping the square from `√(σ_int² + e_i²)`** survived the new wrapper test at n = 200 and
  n = 400. The mutant returns σ_int = 0.0046 against an injected 0.010 — inside a 0.006 band — while
  its posterior width quadruples, 0.0008 → 0.0035. An informativeness gate,
  `sigma_parallax_std < 0.003`, kills it; a mean-only assertion never can.
* **Re-estimating the clip threshold from the sample scale of `z`** survived every synthetic fixture,
  because without a query window `z` really is `N(0,1)` and the two forms coincide — the same blind
  spot that produced the −0.002. Killed by an invariance test: adding a window-truncated
  sub-population whose `|z|` is bounded near zero by the survey query rather than by membership must
  not change the retention of the well-measured stars.

---

## Standing decisions (not tied to one date)

### The pseudo-probability `p̃` is a score, not a probability
`p̃ = probabilities_ × probability_times` (`core/clustering.py`) has no prior, no likelihood and no
normalisation. It is deliberately labelled an *operational ranking statistic* rather than a
posterior — this was a direct referee lesson from P01 (see `~/phd/methodology.md` PART D:
*"Label operational proxies as proxies"*). `calibration.py` exists to measure, empirically, whether
it behaves like a frequency. **Do not describe `p̃` as a membership probability in any paper.**

### Tests assert on behaviour, not execution
The house style is that a test must be able to **fail for the reason it was written**. Examples
worth copying: the σ-clip monotonicity test (a tighter cut can never keep more sources, and its
selection must be a *subset* of the looser one), and the analytic law-of-cosines oracle above. A
test that only asserts a result is finite is not a test — the `solar_radius` bug survived years of
"it runs" checking.

### `--no-verify` is the standing commit workflow
The `nbstripout` pre-commit hook chokes on the ~3.1 GB committed `data/test/NGC6383/` tree and can
revert unstaged work into a `.cache/pre-commit/` patch. Until the hook is fixed, commits use
`git commit --no-verify`. **Consequence: no hook runs on any commit, including nbstripout itself.**
Recorded so nobody assumes hooks are protecting them.

### Synthetic validation data is fractal, not smooth — and the generator had two bugs worth recording

Validating a radial-profile fit on data drawn from a smooth King or EFF profile begs the question:
it assumes exactly the smoothness the fit assumes. `analysis/synthetic.py` therefore generates
substructure by the Goodwin & Whitworth (2004, `2004A&A...413..929G`) box-fractal construction — the
same algorithm McLuster uses (Küpper et al. 2011, `2011MNRAS.417.2300K`) — so the test data comes
from a citable standard method rather than an ad-hoc sprinkling of Gaussian clumps.

Two defects were found by writing the tests, **after** the generator had already been used to
produce Q values:

1. **It returned a cube, not a sphere.** The construction fills a cube of half-width 0.5, whose
   corner is at 0.87; pruning at radius 1.0 removed nothing. Every radial statistic had a
   corner-shaped outer edge. Fixed by pruning to the inscribed sphere and rescaling.
2. **The prescription's own per-level jitter manufactures a central cusp.** The jitter displaces
   each block *coherently with all its descendants*, and blocks that move inward pile up where they
   meet at the origin. Measured on the `D = 3` case, where the answer must be a uniform ball, the
   innermost shells come out over-dense by 1.35× at `noise = 0.10` and **1.80×** at `0.20`. The
   default is now `noise = 0.0`; the cubic lattice it existed to break is erased exactly, and without
   bias, by spreading each star inside its own final cell.

The oracles are external and parameter-free: `D = 3` must reproduce a uniform ball (checked as
Poisson pulls against the `r³` law, not a flat percentage — the innermost bin holds ~20 stars, where
a 20% tolerance is 0.9σ and would fail on shot noise), and Q must rise monotonically with `D`,
landing at Cartwright & Whitworth's ~0.8 for a uniform sphere. Both bugs are pinned by tests that
were verified to fail when the bug is re-applied.

**The general lesson, which is the same one the fixture-regime episode taught:** the first use of a
new generator produced numbers that were quietly wrong in a way no amount of staring at the output
would reveal. Only an external closed-form oracle caught it.

### Post-Gaia, "the cluster fails criterion X" is usually the wrong sentence

`king_model_validity.md` previously scored NGC 6383 as failing **two of three** Muñoz, Padmanabhan &
Geha (2012) recoverability criteria. That framing imported a pre-Gaia ontology in which a cluster is
a bounded King sphere. The corrected reading is *one fails, one is ill-posed, one passes*:

* `FoV / r_half > 3` is **ill-posed**, not failed. `r_half` here is the half-number radius of the
  *selected sample*, which runs to the edge of the query — so it measures where the footprint
  stopped, not where the cluster stopped, and enlarging the query moves it. For a system with a
  corona (Meingast et al. 2021; Bouma et al. 2021's 500 pc halo around NGC 2516) there is no field of
  view large enough, because there is no outer edge for `r_half` to converge to.
* `N > 1000` and `k/b > 20` are **not independent** on an astrometrically pre-filtered sample.
  Filtering only removes stars, so `N` falls (hurts the first) and `b` falls, so `k/b` rises (helps
  the second). Both directions are monotone and guaranteed. Reporting them as two verdicts
  double-counts one fact.

Only `N = 628 < 1000` survives as a clean failure. It still says the footprint does not contain the
sample — but that is a statement about the *observation*, not a defect of the cluster, and post-Gaia
it is the usual case rather than the exception.

---

## 2026-08-03 — a tensor-product interpolator is exempt from the no-quadrature rule

`erotica/analysis/CLAUDE.md` states the rule that governs every radial profile added to the
point-process machinery:

> the point-process machinery takes any `Σ(r)` that has a **closed-form radial integral** — that
> integral is evaluated at every leapfrog step, so **quadrature inside the PyTensor graph is not
> viable**.

The differentiable stellar-track emulator has no closed-form integral in that sense, and the rule as
written would forbid it. **The exemption is argued here rather than assumed, because building on an
unrecorded exemption is how a rule quietly stops meaning anything.**

**The rule targets *adaptive* work in the graph, and a tensor-product interpolant has none.**
Quadrature is forbidden because its cost and its node placement depend on the parameter values, so
every leapfrog step pays an unpredictable price and the gradient depends on where the quadrature
happened to put its nodes. A tensor-product interpolant is the opposite: evaluation is a **fixed**
number of multiply–adds — **8 taps for trilinear, 32 for the tricubic (4×4×2)** — decided at
grid-build time and **independent of the parameter values**. It is a finite closed-form expression
in the grid values, so it is exactly differentiable, has no training loss, cannot introduce
structure absent from the underlying tracks, and its error is bounded by the stencil.

**What the exemption does not cover.** It licenses *fixed-stencil interpolation only*. Any future
component whose evaluation cost or node placement varies with the sampled parameters falls back
under the prohibition, whatever it is called.

**Measured consequence, which is why this matters.** The incumbent binned-Hess path has **46.7%
exactly-zero gradient** in log age and **95.0%** in metallicity; the emulator has **0.0%** on all
four physical parameters over 2000 prior draws, with 0.0% non-finite. Local minima per dex in log
age fall from **45–55** to **2.5 (C0) / 3.1 (C1)**. The interpolant is also PyTensor-native:
`pt.grad` returns finite nonzero gradients and the model compiles under both
`nuts_sampler="numpyro"` and the default PyTensor NUTS — so the isochrone module does **not** have to
leave PyMC, which was the premise the original design note assumed and the spike refuted.

Full derivation, scripts and numbers: `~/phd/agent-findings/differentiable-emulator-design.md`.

---

## Known-and-accepted, with the reason

| Thing | Why it is still like that |
|---|---|
| PyMC is an optional extra | Keeps the default install light; **but a default `pip install erotica` therefore produces only frequentist and heuristic numbers.** Anything Bayesian requires `[bayes]`. |
| `data/test/NGC6383/` (~3.1 GB) is committed | It is the paper's reproducibility artefact, tagged `ngc6383-aanda-resubmission`. It is also the reason pre-commit is broken. |
| ~30 files hardcode `/Users/notluquis/erotica/...` | Paper figure-regeneration scripts. They were rewritten during the 2026-07-21 directory move; if the directory moves again they must be rewritten in the same pass. |


## The latents are integrated out, and the funnel goes with them

*2026-08-24. Supersedes "A defect the fix exposed: the error-aware distance model does not converge
at N > 250" and the hierarchy quoted at the top of this file.*

**What was wrong.** `distance_model`'s error-aware branch declared `r_true_i ~ Gamma(mu_r, std_r)`
and `r_i ~ Normal(r_true_i, errors_i)` — one latent per star, sampled. That is a centred hierarchical
parameterisation, and when the group scale falls below the measurement errors each latent is weakly
informed by its own observation relative to `std_r`; as `std_r -> 0` they all collapse onto `mu_r`.
Neal's funnel. It is not an edge case for this model: **the intrinsic depth being smaller than the
catalogue errors is the regime the model exists for.**

**The number that was wrong, and what replaced it.** The entry above prohibited quoting `mu_r` or
`std_r` from a fit with more than ~250 stars, and its table recorded R-hat 1.0409 with bulk-ESS 43
and 9 divergences at N = 313. On the 200-star test fixture (median catalogue error 5.2x the injected
depth) the failure was sharper still:

| parameterisation | `std_r` | R-hat | ESS | divergences |
|---|---|---|---|---|
| centred, latents sampled (old) | 0.0185–0.0262 across seeds | **1.377**, and **1.817** on another seed | **5** | 0, then **344** |
| latents marginalised (new) | 0.0233 | ≤ 1.009 | 849–992 | 0 on three seeds |

`std_r` — the parameter the model is *for* — was not sampling at all, and the test that guarded it
asserted only a divergence count, which is arbitrary on a chain that has not converged.

**The fix, and why it is exact rather than approximate.** For a normal population the latent
integrates in closed form: if `r_true ~ N(mu, s)` and `r | r_true ~ N(r_true, e)`, then
`r ~ N(mu, sqrt(s^2 + e^2))`. No latents, no funnel, N fewer parameters, and no approximation. The
price is the population family: this branch is now `Normal` where it was `Gamma`. The no-errors
branch keeps the `Gamma` likelihood over the observations, and `metadata["population"]` records
which of the two was fitted, per threshold, all the way out to
`ClusterInferenceAnalyzer.distance_and_parallax_by_probability`.

**Validity range, which the old entry did not state.** After marginalising, `std_r` is identified
only as the *excess* of observed variance over the known errors, so the honest question is whether
the reported depth is the data or the `HalfNormal(sigma_scale)` prior. Measured by sweeping
`sigma_scale` 8x, from 0.025 to 0.20:

| `sigma_scale` | prior mean | fitted `std_r` |
|---|---|---|
| 0.025 | 0.0199 | 0.0194 |
| 0.05 (default) | 0.0399 | 0.0233 |
| 0.10 | 0.0798 | 0.0243 |
| 0.20 | 0.1596 | 0.0253 |

`std_r` moves 0.0058 kpc while the prior moves 0.1396 — **4%**. It is not prior-dominated. But
0.0058 kpc is **29% of the injected depth**, so that residual prior sensitivity is the range to
quote alongside any depth, and it is pinned by
`tests/test_inference.py::test_distance_depth_is_identified_by_the_data_not_by_its_prior`.

**What the guard does not cover, measured by mutating it.** Reverting to the centred hierarchy
leaves that test **green**: both parameterisations identify `std_r` equally well, and what the old
one broke was the geometry, not the identification. The R-hat and ESS assertions in
`_assert_no_divergences` are what guard the geometry. The two mutations that do turn the prior test
red are the ones that remove `std_r` from the likelihood.
