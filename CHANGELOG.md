# Changelog

All notable changes to the EROTICA project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed — BREAKING
- **`dill_cache` is gone from `ClusterAnalyzer`, `load_dataset` and `_load_from_path`.** It was
  never a cache: it wrote `path.with_suffix(".dill")`, while the read fired only when the
  *caller's own* path ended in `.dill`, so the sidecar was written on every load and consulted
  on none. Removed rather than repaired — a working cache needs an mtime check against the
  source and a comparison of the stored `dataloader_kwargs`, neither of which existed, and a
  stale sidecar silently shadowing an edited catalogue is a worse failure than the one being
  fixed. The keyword still accepts a value and raises `DeprecationWarning`; a sentinel default
  means `dill_cache=False` warns too, because that was the value that used to *avoid* the write.
  Loading a path that itself ends in `.dill` is unaffected.

### Changed — BREAKING
- **`PhotometricMassEstimator.assign_nearest` now takes arrays, not a `QTable` plus column
  names.** The two `assign*` methods had no input on which both could be called: one took
  `(mag_column, color_column, source_id)` as arrays, the other a table plus name strings, and
  `color_column` meant an array in the first and a string in the second. Both now use the array
  signature and return `source_id`, `mass`, `mass_std`. `assign_nearest`'s `mass_std` is
  **`NaN`** — a single nearest isochrone point carries no spread, and `0.0` would assert an
  exact mass. `assign_from_samples` is unchanged, including its genuine `0.0` at `k=1`. The
  constructor now classifies the isochrone form it was given, so calling the method that does
  not match raises by name instead of failing inside a delegate.

### Fixed
- **`king_profile` truncates at `r_t`.** Beyond the tidal radius the squared bracket climbed
  back toward `1/(1+(r_t/r_c)²)` — with `r_c=5, r_t=20, b=0` it returned `0` at `r=20` and then
  0.0061, 0.0205, 0.0371, 0.0564 at 30, 50, 100, 1000, i.e. more at large radius than at
  1.35 `r_c`. It now returns the background `b` outside `r_t`, matching the four
  implementations in this package that already did (`_king_model`, `_king_corona_model`,
  `RDP_bayesian`, and the test oracle). **No published result changes**: `king_expected_count`
  already capped its integral at `min(R_t, field_radius)`, and no fitting path calls
  `king_profile`.
- **`eff_surface_density` and `corona_surface_density` accept a Quantity scale radius**
  (`a`, `R_2`), which `king_profile` always did. The previous failure was at `1.0 + (r/a)**2`,
  not at the background term as the docstring claimed, so `b = 0` never avoided it.
- **`compare_datasets` returns its comparison** instead of only printing. Printing is unchanged
  and now behind `verbose=True`.
- **The docs build fails when it is broken.** `fail_on_warning: true` is on for Read the Docs
  and CI enforces zero warnings; previously the CI step piped `sphinx` through `tee` without
  `pipefail`, so a step named "failing on errors" passed on every build that failed.

### Added
- **An external oracle for `king_profile`**, which `tests/CLAUDE.md` had advertised for months
  without one existing: a form-independent check against King (1962) Eq. 18 (always runs), plus
  a cross-check against `ocelot`'s `king62` under a new **`oracles`** extra (skips if absent).

## [0.1.0] - 2026-08-03

First archived release. This is the version deposited to Zenodo and the one the
JOSS submission describes.

### Added — validation and methods
- `erotica.analysis.segregation`: mass segregation (`Λ_MSR`) with three variants, each
  carrying its own citation — the geometric-mean variant is Olczak et al. (2011), not
  Maschberger & Clarke. Significance is an **exact permutation p-value** computed from the
  reference sets the estimator already draws, not `(Λ−1)/σ` against a null of exactly 1
  (the null median is not 1 at small `N_MST`).
- `erotica.analysis.identifiability`: prior-sensitivity, posterior-geometry and
  Muñoz et al. (2012) recoverability diagnostics, plus reconstruction of the per-star
  `log_likelihood` group that a `pm.Potential` model does not emit.
- `coulomb_argument_from_mass_function`: Hénon (1975) closed form for γ, reproducing his
  Tables II and III to four decimals.
- Validation sweeps under `tools/validation/`: ellipticity bias, EFF slope recoverability,
  target–decoy false-discovery proportion, and the faint-quartile clip test.

### Changed — capabilities that existed but were not wired
Three defects of the same shape: the correct path was implemented and the default did not use it.

- **Per-star parallax uncertainties now reach the likelihood.** The analyzer loaded
  `parallax_error` and used it only to discard stars. It now forwards the errors, the
  Bailer-Jones interval and the zero-point flag. The `ϖ > 0` and `σ_ϖ/ϖ ≤ 0.1` pre-cuts are
  removed — on the reference sample they discarded **124 of 254 members**, a documented
  Luri et al. (2018) bias, and they are unnecessary once the errors are modelled.
- **`fit_king_profile` defaults to `king_unbinned`.** It routed to the binned Gaussian
  `RDP_bayesian`, which remains reachable for reproducing older results and now warns.
  `field_radius` is required rather than inferred.
- **The membership clip is no longer magnitude-dependent.** Retention gradient across
  magnitude quartiles: **+0.214 → −0.068**. Note the honest scope — the prescribed
  `|ϖ−ϖ₀|/σ_ϖ` clip measured **−0.184** on the real catalogue, *worse* than the raw clip,
  because the synthetic oracle carried neither excess scatter nor the ADQL parallax window.
  What ships uses a fitted excess dispersion. Without the contaminated ROC that
  `decisions.md` asks for, the defensible claim is the gradient change, **not** that the new
  clip is better.
- Input validation now runs **before** the PyMC import guard, so argument errors raise
  `ValueError` rather than `ImportError` on a core-only install.

### Changed
- Structural fits default to the **unbinned** inhomogeneous Poisson point process
  (continuous Cash 1979 statistic). The binned Gaussian `RDP_bayesian` remains reachable for
  reproducing older results but is no longer the default: binning into equal-count annuli
  fixes the count per bin by construction, and the measured Poisson dispersion index is 0.045
  against the 1.0 a Poisson likelihood asserts.
- `COULOMB_CALIBRATIONS` now carries a verified ADS bibcode per entry, with γ = 0.4 marked
  superseded by Hénon's 0.15 — it is the virial coefficient of ⟨v²⟩ = 0.4 GM/R_h reaching
  `ln(γN)` through a choice of cutoff, not an independent constant.
- Python floor raised to 3.13; dependency floors set to versions that actually resolve there.
- `environment.yml` added: a reproducible conda-forge environment. The package had been
  developed in the mamba `base` environment, so recorded provenance referenced a set nobody
  could recreate.

### Fixed
- `matplotlib.cm.get_cmap`, removed in matplotlib 3.11, in an untested plotting path.
- ArviZ 1.x ships no netCDF engine; `h5netcdf` is now declared, so trace I/O works on a clean
  install rather than by accident.
- Convergence gates read `az.rhat`/`az.ess` rather than `az.summary`, which rounds and
  therefore cannot decide `< 1.01` at the boundary.

### Known limitations, stated rather than deferred
- A free background term **fabricates a background where none exists** and biases the profile
  slope upward (+0.0116 ± 0.0020 on a control whose true `b` is exactly zero). Fit both ways
  and report the difference.
- Fitting a circular profile to an elliptical cluster biases the EFF slope **downward**, toward
  the King limit γ = 2, by up to −0.36 at γ = 4, q = 0.3. The recovered scale radius is the
  geometric mean `a√q`, not the semi-major axis.
- The EFF slope is not recoverable at typical census geometry; the controlling variable is the
  footprint-to-scale ratio, not `N`.
- **The error-aware `distance_model` does not converge above ~250 stars.** ESS 272 at N = 254;
  R-hat 1.041, ESS 43 and 9 divergences at N = 313. The cause is the Gamma hierarchy over one
  latent per star. **No distance or `σ_r` from a fit of more than ~250 stars should be quoted**
  until the non-centred reparameterisation lands, which is deliberately left to its own change
  rather than mixed into the wiring fix.
- **The King tidal radius is prior-determined, not measured**, for a footprint that does not
  contain the object: removing the truncating bound leaves a posterior SD of 698–29 581 arcmin.
  Any `R_t` from this package must be quoted with the prior that produced it.
- `eff_gamma_bias.py`'s bias surface currently confounds finite-sample bias with the free-background
  degeneracy above, and must not be used as a correction until the two are separated
  (`--pin-background` exists for exactly that difference).

### Added
- `erotica.__version__`, sourced from installed package metadata (`importlib.metadata`).
- Real test coverage replacing placeholder stubs: preprocessing corrections
  (Cantat-Gaudin & Brandt proper-motion spin correction, photometric errors,
  fidelity splitting, parallax zero-point), data loading (alias resolution,
  `DataLoader`), plus regression tests for dynamics, inference, King structure,
  I/O helpers, and the Sagitta guard.
- GitHub Actions CI workflow (pytest on Python 3.11–3.13).
- JOSS software-paper scaffolding (`paper/paper.md`, `paper/paper.bib`) and
  `CITATION.cff`.
- King-profile summaries now expose correctly named `*_median` keys.

### Changed
- Declared previously-undeclared runtime dependencies `gaiadr3-zeropoint` and
  `fast-histogram` (fixes `import erotica.preprocess` failing on a clean install).
- Reconciled `LICENSE` to AGPL-3.0 to match `pyproject.toml`.
- Provenance metadata now records the real package version instead of a
  hardcoded `"0.0.1"`.

### Fixed
- `AttributeError` crash in the Galactic branch of
  `calculate_galactocentric_distance` (`Angle.radians` → `.radian`).
- Fractional parallax-error selection admitted unphysical negative parallaxes and
  divided by zero at `parallax == 0`; now requires `parallax > 0`.
- Masked 64-bit identifier columns (e.g. Gaia `source_id`) were promoted to
  `float64`+NaN, silently corrupting IDs above 2^53 and breaking crossmatches.
- King-trace point estimates were medians mislabeled as `*_mean`.

### Security
- Removed a silent runtime `pip install` of an unpinned git ref in
  `erotica.analysis._sagitta`; it now raises `ImportError` with install
  instructions instead of modifying the user's environment.

### Removed
- Fake "build passing" / "docs" status badges from the README.

## [0.0.1] - 2025-10-03

### Added
- Initial alpha release of EROTICA
- Core clustering functionality with HDBSCAN and Optuna optimization
- Data loading utilities for Gaia, 2MASS, and WISE photometric systems
- Comprehensive data preprocessing and cleaning tools
- Statistical analysis and visualization capabilities
- Modular package structure with organized submodules:
  - `erotica.core` - Clustering algorithms and core functionality
  - `erotica.io` - Data loading and I/O operations
  - `erotica.preprocess` - Data preprocessing and quality control
  - `erotica.analysis` - Statistical analysis and characterization
  - `erotica.utils` - General utility functions
- Backward-compatible shims for legacy import patterns
- Professional package configuration with `pyproject.toml`
- Comprehensive README with installation and usage instructions
- Development environment setup with testing framework

### Changed
- Reorganized flat module structure into logical subpackages
- Converted duplicate files to clean re-export shims
- Updated project metadata for first release
- Improved code organization and maintainability
- **Eliminated legacy directory after successful migration**

### Fixed
- Resolved circular import issues in utility modules
- Corrected package export paths and import statements
- Fixed inconsistent function naming across modules

### Removed
- Legacy code files after complete migration to organized structure
- Obsolete backup files and temporary directories

### Technical Details
- Python 3.11+ requirement established
- Dependencies properly specified in `pyproject.toml`
- Clean separation of concerns across submodules
- Maintained API compatibility during reorganization

---

**Note**: This is an alpha release intended for development and testing. The API may change significantly in future versions as we work toward a stable 1.0.0 release.
