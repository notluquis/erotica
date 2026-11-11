# Radial structure

How EROTICA fits a King (1962) profile to the radial distribution of a cluster's members with
{func}`~erotica.analysis.king_unbinned` {bdg-success}`now`, what the output means, and what the fit
cannot tell you.

---

## 1. What the method does

`king_unbinned` treats the members' angular distances from the center as an **inhomogeneous
Poisson point process** with intensity $\lambda(r) = 2\pi r\,\Sigma(r)$, where $\Sigma(r)$ is a King
profile plus a flat term `b`. From its docstring:

> the log-likelihood $\log L = \sum_i \log \lambda(r_i) - \Lambda$, the continuous form of the
> Cash (1979) statistic. There are no bins, so there is no binning choice for a referee to
> question, and no shared scatter parameter: a point process has no nuisance ``sigma``.

The priors (`KingPriors`) are fixed constants, "independent of the data being fit", and the full
posterior is kept by default (`return_trace=True`) so its uncertainty can reach the dynamical
quantities derived from it. Sampling uses NUTS through PyMC (the `bayes` extra).

| Option | Argument | What it costs |
|--------|----------|---------------|
| Scale-free priors (default) | `priors=KingPriors()` | none; the half-Cauchy scales are constants |
| Physical prior on $R_t$ | `tidal_prior=(mu, sigma)` in arcmin, e.g. from {func}`~erotica.analysis.dynamics.tidal_radius_prior` | the result is now partly the Jacobi radius you supplied |
| Completeness correction | `completeness=` a radial $\bar S(r)$ | you build $\bar S(r)$ yourself (§4) |
| Sampler budget | `sampling=SamplingConfig(...)` | the default is 2000 draws and 1000 tuning steps |

## 2. A minimal example

The stars below are one realization of a King profile plus a background, drawn as a point process
with a fixed seed. Replace `radii` with the distances of your members (arcmin, all of them inside
`field_radius`).

```python
import numpy as np
from erotica.analysis import king_profile

# One realization of a King (1962) profile plus a flat background, drawn as an
# inhomogeneous Poisson point process in radius (radii in arcmin).
truth = dict(amplitude=6.0, background=0.05, core_radius=4.0, tidal_radius=30.0)
field_radius = 70.0
rng = np.random.default_rng(3)

grid = np.linspace(0.0, field_radius, 20_001)
intensity = 2.0 * np.pi * grid * king_profile(grid, **truth)
cdf = np.concatenate([[0.0], np.cumsum(0.5 * (intensity[1:] + intensity[:-1]) * np.diff(grid))])
radii = np.interp(rng.uniform(0.0, cdf[-1], rng.poisson(cdf[-1])), cdf, grid)
print(f"{radii.size} stars inside {field_radius:g} arcmin")
```

```text
1181 stars inside 70 arcmin
```

```python
from erotica.analysis import king_unbinned
from erotica.analysis.inference import SamplingConfig

res = king_unbinned(
    radii,
    field_radius=field_radius,
    sampling=SamplingConfig(draws=500, tune=500, chains=2, random_seed=1),
)
for name in ("k", "b", "R_c", "R_t"):
    print(f"{name:>4}: {res[f'{name}_median']:.3f} +/- {res[f'{name}_std']:.3f}")
print(f"b carries {res['b_fraction_median']:.2f} +/- {res['b_fraction_std']:.2f} of the stars")
```

```text
   k: 5.737 +/- 0.544
   b: 0.048 +/- 0.002
 R_c: 3.974 arcmin +/- 0.451 arcmin
 R_t: 29.881 arcmin +/- 3.506 arcmin
b carries 0.63 +/- 0.02 of the stars
```

The injected values were $k = 6$, $b = 0.05$, $R_c = 4'$ and $R_t = 30'$. These numbers come from
one run on one machine; on another machine or with other package versions the draws differ, so
compare within the Monte Carlo error (see the reproducibility note in the {doc}`API reference
</api/index>`). This example takes about 35 s; a
publication run should use the default budget or more.

## 3. Reading the output

The result is a `dict`: posterior medians and standard deviations of `k`, `b`, `R_c` and `R_t`
(the radii as `astropy` quantities in arcmin), `n_stars`, `field_radius`, `completeness_corrected`,
`b_fraction_median`/`b_fraction_std`, the post-sampling `mode_search` record, and the posterior
itself as `king_trace`. Check the sampler before quoting a number:

```python
import arviz as az

trace = res["king_trace"]
print(az.summary(trace, var_names=["k", "b", "R_c", "R_t"])[["mean", "sd", "r_hat", "ess_bulk"]])
print("divergences:", int(trace.sample_stats["diverging"].values.sum()))
print("mode search ok:", res["mode_search"]["ok"], "| modes found:", res["mode_search"]["n_modes"])
```

```text
       mean      sd r_hat ess_bulk
k      5.74    0.54  1.00      406
b    0.0481  0.0019  1.00      729
R_c    3.99    0.45  1.01      300
R_t    30.4     3.5  1.01      306
divergences: 0
mode search ok: True | modes found: 1
```

**Read `b_fraction` next to the structural parameters.** It is the share of the expected count
inside the field that the flat term carries, computed per draw. From the docstring of
{func}`~erotica.analysis.king_background_fraction`:

> on a membership-selected sample the field has already been removed, so ``b`` is **not**
> contamination, and a fit can still hand it most of the sample. On NGC 6383 it carries 56% of the
> 628 members (erotica#5). Reporting ``k``, ``R_c`` and ``R_t`` without this number reports the
> profile of a minority of the data as the profile of the cluster.

In the simulation above the injected share is:

```python
from erotica.analysis import king_background_fraction

# The share b carries in the simulation, from the injected parameters
print(f"{king_background_fraction(6.0, 0.05, 4.0, 30.0, field_radius):.2f}")
```

```text
0.62
```

## 4. Limits, as the docstring states them

- **The sample must be complete inside `field_radius`.** `radii` "Must be the **complete** sample
  inside `field_radius` -- the normalisation integral assumes the footprint is that full disc."
  A star outside it raises `ValueError` ("The normalisation integral assumes the sample is complete
  within that disc, so this would bias the fit"), and fewer than ten stars is an error.
- **Completeness is yours to build.** With `completeness=` the fit describes "the **true** cluster
  rather than the observed one", but "{mod}`erotica.selection` does **not** build it: it provides
  per-star weights … and no function returns a radial $\bar{S}(r)$."
- **$R_t$ is time-averaged.** "The ``R_t`` recovered by any King fit is the cluster's
  *time-averaged* tidal radius, not its perigalactic one (Küpper et al. 2010, MNRAS 407, 2241)."
- **`tidal_prior` changed in 2026-10 (R29-01).** It is now a truncated normal on $R_t$ itself;
  results fitted with the earlier version "sit +0.04 to +0.15 sigma above the ones from this
  version."

What the King fit can and cannot identify, and when an EFF or Wilson profile is the better model
({func}`~erotica.analysis.compare_radial_profiles`), is in the design note
[King model validity](../design-notes/king_model_validity.md).
