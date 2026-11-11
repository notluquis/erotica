# Isochrone fitting

How to fit isochrones to a cluster's color–magnitude diagram with
{class}`~erotica.analysis.IsochroneFitterKW` {bdg-success}`now`, what the output means, and what it
does not mean.

```{admonition} Scope — read this before quoting a number from this class
:class: warning
From the class docstring: "**A model of the PMS in the window G 14-19 with errors scaled by k; not
the age of the cluster.** Its log t is conditional on the model and on the prior on dm; ``k``
absorbs the mismatch; coverage was calibrated on simulations of the same model with n = 8. The
MS vs PMS tension is the reason for the window."
```

---

## 1. What the method does

`IsochroneFitterKW` is {class}`~erotica.analysis.IsochroneFitter` with two changes; everything else
(EEP-interpolated isochrones, Chabrier IMF, unresolved binaries, extinction law, field component) is
inherited. From the module docstring:

- **K**: "the width of star $i$ is $s_i^2 = (k\,e_i)^2 + \sigma_{\rm floor}^2 + \sigma_{\rm int}^2$
  in both axes, with $k \ge 1$ a free scale of the catalogue's errors and $\sigma_{\rm int}$ the
  isotropic intrinsic width." The prior is $k \sim U(1, 6)$; "the upper bound is the search bound,
  not a physical statement."
- **W**: "pass ``mag_window=(14.0, 19.0)`` to `setup`." The age it returns is "**the age of the
  stars in the window**, never 'the age of the cluster'."

The seven parameters are `(met, loga, dm, Av, sigma_int, k, f_bg)`. Sampling is NUTS (nutpie on JAX
by default), started from the mode search {meth}`~erotica.analysis.IsochroneFitterKW.find_start`.

| Choice | Argument | What it costs |
|--------|----------|---------------|
| Magnitude window | `setup(..., mag_window=(bright, faint))` | the age describes only the stars inside it |
| Isochrone grid | `isochs_path=` (MIST `.iso.cmd` files) or `grid=` (MIST v1.2/v2.5, PARSEC, BHAC15, SPOTS) | the fit is conditional on the grid |
| Extinction law | `extinction_law="gaia_edr3"` (default for Gaia bands) or `"ccm89"` | `"ccm89"` reproduces P01 |
| Binaries | `alpha`, `beta` (Offner relation); `0, 0` = single stars | — |
| Distance prior | `dm_mu`, `dm_sigma`, `dm_range` | it fixes the age along the age–distance ridge (§4) |

## 2. The grid

**EROTICA ships no isochrone grid.** The files are not the package's to redistribute; the
fetchers in `erotica.analysis.grids.fetch` (`fetch_mist25`, `fetch_parsec`, `fetch_bhac15`,
`fetch_spots`) download them into `$EROTICA_GRIDS_CACHE` (default `~/.cache/erotica-grids`) and
check their hashes. For an example that runs offline, this guide writes the toy family the test
suite uses (`tests/test_isochrone.py`): MIST's `.iso.cmd` layout, one file per Z, with closed-form
photometry. **It is not a stellar model.**

```python
from pathlib import Path

import numpy as np

# A toy isochrone family in the MIST .iso.cmd layout: one file per Z, every age in it.
# Photometry is closed form in (mass, age, Z) -- younger and more metal-rich is brighter
# and redder -- so the example needs no download. It is NOT a stellar model.
MASSES = np.geomspace(0.1, 8.0, 120)


def toy_photometry(mass, loga, Z):
    lm, dz = np.log10(mass), Z - 0.015
    G = 4.6 - 6.5 * lm - 2.0 * (7.0 - loga) / (1.0 + mass**2) + 40.0 * dz * (1.0 - lm)
    col = 0.8 - 1.4 * lm + 20.0 * dz * (1.0 + lm) + 0.1 * (7.0 - loga)
    return G, col  # absolute G and intrinsic BP-RP


grid_dir = Path("toy_grid")
grid_dir.mkdir(exist_ok=True)
for i, Z in enumerate((0.010, 0.015, 0.020)):
    rows = []
    for loga in (6.3, 6.4, 6.5, 6.6, 6.7, 6.8):
        G, col = toy_photometry(MASSES, loga, Z)
        for eep, (m, g, c) in enumerate(zip(MASSES, G, col)):
            rows.append(f"{eep} {m:.6f} {loga:.4f} {g:.6f} {g + 0.6 * c:.6f} {g - 0.4 * c:.6f}")
    header = (
        "# toy isochrone family\n# Yinit  Zinit  FeH\n#  0.270  {:.6f}  0.00\n"
        "# EEP initial_mass log10_isochrone_age_yr Gaia_G_EDR3 Gaia_BP_EDR3 Gaia_RP_EDR3\n"
    ).format(Z)
    (grid_dir / f"toy_{i}.iso.cmd").write_text(header + "\n".join(rows) + "\n")
print(sorted(p.name for p in grid_dir.glob("*.iso.cmd")))
```

```text
['toy_0.iso.cmd', 'toy_1.iso.cmd', 'toy_2.iso.cmd']
```

## 3. A minimal example

The stars are drawn from the same closed form, not through the fitter: Chabrier masses, single
stars, CCM89 reddening, and a true scatter twice the reported errors ($k = 2$). The table needs
the columns `setup` reads: `Gmag`, `G_BPmag`, `G_RPmag`, their errors, and `probability_hdbscan`.

```python
from astropy.table import QTable

from erotica.analysis._isochrone import _ccm89  # the CCM89 coefficients the fitter inverts

rng = np.random.default_rng(7)
n, truth = 400, dict(loga=6.55, Z=0.0125, dm=10.25, Av=0.7, k=2.0)

# Chabrier (2014) masses by inverse CDF, single stars only
m = np.geomspace(0.1, 8.0, 20_000)
pdf = np.where(m < 1.0, np.exp(-0.5 * (np.log10(m / 0.2) / 0.55) ** 2) / m,
               np.exp(-0.5 * (np.log10(0.2) / 0.55) ** 2) * m**-2.35)
cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(m))])
mass = np.interp(rng.uniform(size=n), cdf / cdf[-1], m)

G0, col0 = toy_photometry(mass, truth["loga"], truth["Z"])
kG, kBP, kRP = (_ccm89(lam) for lam in (6390.7, 5182.6, 7825.1))
mags = {
    "Gmag": G0 + truth["dm"] + kG * truth["Av"],
    "G_BPmag": G0 + 0.6 * col0 + truth["dm"] + kBP * truth["Av"],
    "G_RPmag": G0 - 0.4 * col0 + truth["dm"] + kRP * truth["Av"],
}
# reported errors grow with magnitude; the TRUE scatter is k = 2 times larger
err = 0.004 + 0.0008 * (mags["Gmag"] - mags["Gmag"].min()) ** 2
stars = QTable({c: v + rng.normal(0.0, truth["k"] * err) for c, v in mags.items()})
for c in ("e_Gmag", "e_G_BPmag", "e_G_RPmag"):
    stars[c] = err
stars["probability_hdbscan"] = np.ones(n)
print(f"{n} stars, G from {stars['Gmag'].min():.1f} to {stars['Gmag'].max():.1f}")
```

```text
400 stars, G from 10.2 to 21.0
```

Set up the fitter and run the mode search (about 30 s):

```python
from erotica.analysis import IsochroneFitterKW

priors = dict(loga_range=(6.3, 6.8), Av_range=(0.2, 1.0), dm_range=(9.6, 10.4))
fitter = IsochroneFitterKW(
    grid_dir,
    dm_mu=10.0, dm_sigma=0.3,     # a loose distance prior, centered 0.25 mag off the truth
    alpha=0.0, beta=0.0,          # single stars, as simulated
    extinction_law="ccm89",       # the law the toy stars were reddened with
    **priors,
)
fitter.setup(stars, prob_threshold=0.0, mag_window=(14.0, 19.0))

start = fitter.find_start(2, np.random.default_rng(0))
print({p: round(v, 3) for p, v in start["mode"].items()})
print(f"log L at the mode {start['loglike']:.1f}, at the truth "
      f"{fitter.loglike(0.0125, 6.55, 10.25, 0.7, sigma_int=1e-4, f_bg=1e-4, k=2.0):.1f}")
print("at a bound:", start["at_bound"])
```

```text
  EEP node table: (3, 6, 23, 120) (Z x age x rows x EEP)
{'met': 0.011, 'loga': 6.588, 'dm': 9.798, 'Av': 0.254, 'sigma_int': 0.0, 'k': 1.883, 'f_bg': 0.0}
log L at the mode -181.7, at the truth -186.6
at a bound: ['sigma_int', 'f_bg']
```

`k` comes back near the injected 2. The mode's distance and extinction do not: inside the window the
likelihood alone runs further along the ridge (to `dm` 9.3 in this replicate), and the loose prior on
`dm`, centered 0.25 mag below the truth, pulls the mode back only part of the way. The 4.9 gain in
log L over the truth is what seven free parameters buy on noise (median 2.2 over 20 seeds), not a
preference beyond it. That is the ridge the Scope box warns about, on data where the truth is
known. `sigma_int` and `f_bg` sit on their lower bounds because the toy has neither.

Then sample. This block took about 8 minutes on our machine (two chains of 500 + 500, JAX compilation
included); a publication fit uses more draws, and the NGC 6383 fit in the module docstring ran
4 x 2000.

```python
import arviz as az
from erotica.analysis.inference import fit_provenance_summary, mode_search_summary

idata = fitter.fit(draws=500, tune=500, chains=2, random_seed=11, progressbar=False,
                   target_accept=0.9)
summary = az.summary(idata, var_names=["met", "loga", "dm", "Av", "sigma_int", "k", "f_bg"])
print(summary[["mean", "sd", "r_hat", "ess_bulk"]])
print("divergences:", int(idata.sample_stats["diverging"].values.sum()))
print("mode search ok:", mode_search_summary(idata)["ok"])
print("seeds:", fit_provenance_summary(idata)["seeds"])
```

```text
             mean      sd r_hat ess_bulk
met        0.0119  0.0014  1.01      200
loga         6.58     0.1  1.01      197
dm           9.94    0.17  1.01      194
Av           0.38   0.126  1.00      275
sigma_int  0.0116  0.0081  1.00      451
k           1.855   0.093  1.01      761
f_bg       0.0076  0.0069  1.00      728
divergences: 0
mode search ok: True
seeds: {'random_seed': 11, 'mode_search': 2389632199}
```

Against the injected values (log t 6.55, Z 0.0125, dm 10.25, $A_V$ 0.7, $k$ 2): log t and Z are
within one standard deviation, $k$ within two, `dm` is 1.8 sd short and $A_V$ 2.5 sd short, in the
direction of the mode above. This is one synthetic replicate, not a coverage test. The numbers come
from one run on one machine; elsewhere the draws differ, so compare within the Monte Carlo error.

## 4. Reading the output

`fit` returns an ArviZ `InferenceData`. Before quoting anything, check R-hat, the effective sample
size, the divergences, and the post-fit mode search (`mode_search_summary(idata)`); the seeds, the
engine, the start and the package versions are in `fit_provenance_summary(idata)`, both from
`erotica.analysis.inference`. Then read
the posterior of `loga`, not the mode: from the `find_start` docstring, "**A mode search is not a
proof of uniqueness** … ``mode`` is the search's maximum, and a reported age needs the profile of
the posterior in log t as well." `posterior_cmd(idata)` draws synthetic CMDs with each draw's own
`k`, for a predictive check against the observed stars.

## 5. Limits, as the docstrings state them

- **Not the age of the cluster.** "One isochrone does not describe the main sequence and the
  pre-main-sequence of NGC 6383 at once … The window is what removes the tension; it does not
  resolve it."
- **The window comes from the grid and the priors.** From `setup`: "The window must come from the
  grid and the priors (`erotica.analysis.grids.safe_window`), never from the data." The
  `(14.0, 19.0)` above is the NGC 6383 window, used on a toy.
- **The distance prior fixes the age.** "The likelihood alone has a second hump at log t ~ 6.45,
  which is the bound dm >= 9.50 of the prior range … the prior on dm is what fixes the age. A wrong
  centre of that prior is not in the interval."
- **`k` is not a measurement of the errors.** "Whether it is an error underestimate or a spread
  that scales with the error is not identified, and a k > 1 is what makes the check pass."
- **Coverage.** "The coverage was calibrated on simulations of the same model, with n = 8
  replicates per cell. It is 'no gross under-coverage detected', not 'calibrated', and it says
  nothing about a different truth." In the harder cell (log t 6.43, [Fe/H] +0.15) "NUTS does
  **not** converge … A fit there is not to be reported without passing the gate."

The model, its measurements on NGC 6383 and the history of the window are in the design note
[the K x W model](../design-notes/isochrone_kw_model.md).
