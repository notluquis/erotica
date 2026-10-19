# `IsochroneFitterKW` — the K x W model (error scale + magnitude window)

_2026-10-19. The model that passes the posterior predictive check on NGC 6383's pre-main-sequence
window, ported from the research code on branch `isochrone-model-fix`. Numbers are from the hub
findings `isochrone-model-fix.md` and `isochrone-intervals-nuts.md` §2; where a number is a sentence
of those findings and not something this repository re-computes, it says so._

## Scope — read this first

**`IsochroneFitterKW` is a model of the pre-main-sequence in a window (G 14-19) with the catalogue errors
scaled by a free factor `k`. It is not the age of the cluster.**

- **Why a window: the MS vs PMS tension.** One isochrone does not describe the main sequence and the
  pre-main-sequence of NGC 6383 together; a single-age fit of the whole CMD is falsified
  (`isochrone-ms-vs-pms.md`), and the scaled-error model on all stars fails the predictive check. The window
  removes the tension from the fit; it does not explain it.
- **The log t is conditional on the model and on the prior on dm.** The likelihood alone has a second hump at
  log t ~ 6.45, which is the bound dm >= 9.50 of the prior range and which the cluster parallax excludes
  (hub finding D15). Over log t 5.80-6.56 the likelihood alone is flat within -2.7: the prior on dm fixes the
  age, and an error in its centre is not in the interval.
- **`k` absorbs the mismatch.** A `k` > 1 is what makes the predictive check pass. It is not a measurement of
  the catalogue's errors.
- **The coverage was calibrated on simulations of the same model, with n = 8** replicates per cell: "no gross
  under-coverage detected", not "calibrated", and silent about a truth that is not this model.

## What it is

`IsochroneFitter` with two changes, everything else inherited (EEP-interpolated MIST isochrones,
Chabrier 2014 IMF, Offner/Duchêne & Kraus binaries, CCM89 extinction, uniform field fraction):

- **K**: star $i$ has width $s_i^2 = (k\,e_i)^2 + \sigma_{\rm floor}^2 + \sigma_{\rm int}^2$ in both
  axes, $k \sim U(1, 6)$. The whole-CMD fit with $k = 1$ fails the predictive check; $k = 2.45$ and
  $\sigma_{\rm int} = 0.094$ mag on the 201 stars in $14 \le G \le 19$. **What $k$ is** (a Gaia
  error underestimate or a spread that scales with the error) is not identified here.
- **W**: `setup(mag_window=(14.0, 19.0))`. A single age for the whole CMD is falsified for this cluster
  (`isochrone-ms-vs-pms.md`), so the answer is **the age of the stars in the window**.

```python
from erotica.analysis import IsochroneFitterKW

f = IsochroneFitterKW(grid=grid, magnitude="Gaia_G_EDR3", color=("Gaia_BP_EDR3", "Gaia_RP_EDR3"),
                      loga_range=(5.7, 7.0), Av_range=(0.0, 3.0),
                      dm_mu=10.223, dm_sigma=0.026, dm_range=(9.5, 10.7))
f.setup(members, prob_threshold=0.0, mag_window=(14.0, 19.0))
idata = f.fit(draws=1000, tune=1000, chains=4, target_accept=0.9)
```

`find_start` is the mode search that was measured; its chain starts are over-dispersed along the age
ridge. **A mode search is not a proof that the mode is unique** — see below.

## What was measured, and what carries to the package

| claim | where measured | status here |
|---|---|---|
| log L at the K x W mode and at an injected truth | research `aniso.WidthFitter`, own segment density, field term and `k` | **re-checked** to 1e-8 on two data sets (`tests/test_isochrone_kw.py`). **Needs the local MIST v1.2 files: skipped in CI**, so this evidence exists on the author's machine only. The segment density and `F(theta)` are inherited from the parent in both implementations, so they are not checked independently |
| mode search finds the same mode, and the same failure on replicate A_0 | `aniso.search8` | **re-checked** (`slow`) |
| NGC 6383: log t = 5.965 (68 %: 5.933-6.000; 90 %: 5.913-6.025), 0.92 (+0.08/-0.07) Myr | NUTS 4 x 2000, R-hat <= 1.003, ESS >= 1920, 0 divergences | **re-run with the package's `fit`** (`slow`, tolerance = Monte Carlo error of the difference). The sampler is not the research one: research used numpyro NUTS in chunks, `target_accept` 0.9, jittered starts; `fit` uses PyMC NUTS with a seeded diagonal mass matrix and `target_accept` 0.8 by default. This test is the only bridge between them |
| cell A coverage of log t: 6/8 at 68 %, 7/8 at 90 % (6 of 8 fits converged) | 8 NUTS fits, research code | **not re-run** (8 fits = hours). It carries because the likelihood is the same function. "No gross under-coverage detected at R = 8", never "calibrated": the rule passes with probability 0.58 even at true coverage 0.50 / 0.75 |
| [Fe/H], A_V and k under-cover in cell A (1/8, 4/8, 3/8 at 68 %) | same | the [Fe/H] of the fit (-0.456) is **not citable** |
| cell C (log t 6.43, [Fe/H] +0.15): NUTS does **not** converge (0/8, R-hat 1.76-2.76) | same | a fit in that regime is not reported without passing the gate |

## What the age is conditional on (do not drop these)

- **The prior on dm.** Log L alone is flat within -2.7 over log t 5.80-6.56 with dm falling from 10.57
  to 9.51: the prior N(10.223, 0.026) fixes the age along an age-distance ridge. Shifting its centre by
  +/-0.026 mag moves the median by -/+0.014 dex.
- **The grid** (MIST v1.2) and the **width model**: the MAP of the other width arms ranges 5.950-5.989.
  No between-grid term is in the interval.
- **The window** and the choice of model among >= 11 tried on the same 201 stars.
- **Binaries.** See the section below.

## Binaries under K x W (hub `isochrone-intervals-nuts.md` §2.17)

C3 of the earlier pre-registration (log t of binary synthetic clusters +0.013 to +0.024 dex off, 0/4
inside the 90 % interval) was measured on the whole-CMD model, not on this one, and stays "not met"
there. Under K x W the cell-A generator shares the likelihood's binary sheet (stride, q nodes,
smoothing), so its coverage cannot see an error in that treatment. `kw_c3.py` (branch
`isochrone-model-fix`) draws binaries with an independent generator: continuous primary mass,
isochrone read at fixed mass, continuous q. Measured with 4 paired replicates of 8 x 201 stars
(mode of the likelihood, not NUTS): the mode sits 0.010 dex younger than the truth for singles and
0.014 dex younger for binaries; the **binary-attributable difference is -0.004 dex, 90 % interval
[-0.015, +0.007]**: the upper end of that interval is +0.007. The old +0.013 to +0.024 was measured on another model and is not tested here. **One regime** (log t about 5.97, G 14-19),
the model's own Offner fraction, four pairs: "no systematic detected at <= 0.015 dex", not "no
systematic". The fraction itself being wrong was not measured.

## Inherited convention

The detected fraction $F(\theta)$ of the window uses the error model of the catalogue **unscaled** at the
segment midpoint, not $k\,e$. At the faint edge it is the smaller term (0.01 mag against $\sigma_{\rm int}$
0.094); its effect on $F$ is not measured.
