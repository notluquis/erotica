# Changelog

All notable changes to the EROTICA project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **King/EFF/corona model builders now use `pm.HalfCauchy` directly, dropping the
  `HalfStudentT(nu=1)` workaround** for pytensor#2308 (numba `CauchyRV` drew with location
  `loc/scale` and scale `1/scale` instead of `loc, scale`; `logp`/NUTS were unaffected but
  `sample_prior_predictive` was silently wrong). Fixed upstream in PR #2309, released
  `rel-3.2.4` (2026-08-01); `pytensor>=3.2.4` and `pymc>=6.2` are now the floor in `[bayes]`
  and `[paper]`, and the builders raise `RuntimeError` below that floor instead of silently
  reproducing the bug. Closes hub thread F1. See `docs/design-notes/decisions.md` (2026-09-24).
- **`IsochroneFitter`: single stars are deposited with a density continuous along the track**
  (linear within each segment between EEP points) instead of each segment's IMF weight spread
  uniformly. Where MIST's EEP points are sparse and slide fast along the track with age, the
  uniform deposit made log L rough in log t, and a MIST synthetic fit failed the R-hat/ESS gate
  (R-hat 1.026, ESS 167). See `docs/design-notes/decisions.md` (2026-09-24, later).
- **`IsochroneFitter`: the completeness term is integrated along each isochrone segment** instead
  of evaluated at its midpoint. The midpoint made `N ln F` step once per EEP segment of shift,
  turning log L into a sawtooth of local modes (teeth 2-4 log-units on the toy family) that NUTS
  hopped between: ESS/draw on A_V 0.03-0.05 → 0.25-0.26 on the toy, and its mode moves from
  dm 10.30 / A_V 0.72 to 10.27 / 0.69 (truth 10.25 / 0.70). This, not a dm-A_V ridge, was the
  slow mixing listed under Known issue.
- **`IsochroneFitter`: the model Hess had no stars brighter than the window's faint end minus the
  distance modulus.** The precomputed grid was binned in absolute magnitudes on the apparent
  window and then shifted by the full `dm + k_G A_V`; on NGC 6383 the model put zero mass
  brighter than G = 17, where 129 of 254 members sit, and NUTS stalled against the prior walls
  (R-hat 1.5-2.2). The grid is now binned in a padded reference apparent frame and shifted by the
  offset. Also: the grid interpolates between isochrone nodes instead of snapping to the nearest
  one (d loglike / d met was exactly zero at 283 of 300 prior points), the two members on the
  histogram's upper edges are no longer dropped, grid caches from the old frame are refused, and
  priors wider than the grid's padding raise. NUTS now passes the convergence gate on NGC 6383
  at the default grid reference (4 chains, R-hat <= 1.0021, ESS_bulk >= 2523, 0 divergences) --
  but that pass is reference-dependent (the same fit with the grid's internal reference moved
  half a bin gives R-hat 1.03, ESS 152) and not robust on synthetic data (6 of 16 runs fail it).

### Changed
- **`IsochroneFitter` likelihood rewritten: unbinned per star over EEP-interpolated MIST
  isochrones**, replacing the shifted precomputed Hess grid. Exact Gaussian integral along each
  isochrone segment, binaries with q marginalised, completeness term, field fraction, free
  intrinsic width (0.01 mag floor). The likelihood no longer depends on the prior centre (ΔlogL =
  0.0 on NGC 6383 where the grid gave −10.5 / −10.6 / +4.5). `fit()` starts chains from a global
  mode search with a seeded mass matrix. `M_met` / `M_loga` are ignored; grid caches are refused.

### Known issue
- ~~**`IsochroneFitter` is experimental and its parameters are biased.**~~ The reference locking
  is gone (see Changed). **Still experimental**: the dm-A_V ridge mixes slowly (ESS_bulk ~300 at
  2 × 3000 draws on a toy cluster whose truth is recovered), and full injection-recovery on MIST
  synthetics is not yet measured. Pinned by a strict-xfail test; see
  `docs/design-notes/decisions.md` (2026-09-23).

## [0.2.0] - 2026-09-22

### Removed
- **The NGC 6383 (aa52082-24) manuscript and its review tooling moved to their own public repo**,
  `github.com/notluquis/paper-ngc6383-aa52082-24` (2026-09-22): `data/test/NGC6383/comments_paper/`
  (181 tracked files), the eight `tools/validation/ngc6383_*` scripts, and the
  `manuscript.yml` CI workflow that checked only that paper. `erotica`'s own git history is
  unchanged; everything that left is preserved up to the `p01-pre-extraction` tag, with a commit
  map in the new repo. `tools/manuscript_gate.py` stays here — the new repo vendors a pinned copy.
  See `docs/design-notes/decisions.md` (2026-09-22 entry) for what stayed on disk unpublished and
  which runtime reads it left broken (none in `tests/` or the package itself). (`06f6556`)

### Removed — BREAKING
- **`dill_cache` is gone from `ClusterAnalyzer`, `load_dataset` and `_load_from_path`.** It was
  never a cache: it wrote `path.with_suffix(".dill")`, while the read fired only when the
  *caller's own* path ended in `.dill`, so the sidecar was written on every load and consulted
  on none. Removed rather than repaired — a working cache needs an mtime check against the
  source and a comparison of the stored `dataloader_kwargs`, neither of which existed, and a
  stale sidecar silently shadowing an edited catalogue is a worse failure than the one being
  fixed. The keyword still accepts a value and raises `DeprecationWarning`; a sentinel default
  means `dill_cache=False` warns too, because that was the value that used to *avoid* the write.
  Loading a path that itself ends in `.dill` is unaffected. The `ClusterAnalyzer.dill_path`
  property was deleted outright alongside it, then given a named `AttributeError` explaining the
  removal instead of a bare one, so both halves of the sidecar fail the same way.
  (`04749f6`, `76a10bf`)
- **Five unnamespaced top-level shim modules are no longer installed.** `pyproject.toml` declared
  `py-modules = ["clustering", "cluster_analysis", "data_loader", "data_preprocessor", "utils"]`,
  so `pip install erotica` dropped those five names *unnamespaced* into site-packages — `utils` in
  particular is one of the most collision-prone names on PyPI, able to shadow a user's own module
  or another package's. This shipped in v0.1.0 and was live on PyPI. The files stay in the working
  tree as re-export shims (`from erotica.utils.utils import compare_datasets`, 81 lines total,
  nothing in this repository imports them) so a local notebook still runs, but `import utils`
  after a fresh `pip install erotica` now fails; use `from erotica.utils.utils import
  compare_datasets` etc. instead. (`71919bd`)

### Changed — BREAKING
- **`PhotometricMassEstimator.assign_nearest` now takes arrays, not a `QTable` plus column
  names.** The two `assign*` methods had no input on which both could be called: one took
  `(mag_column, color_column, source_id)` as arrays, the other a table plus name strings, and
  `color_column` meant an array in the first and a string in the second. Both now use the array
  signature and return `source_id`, `mass`, `mass_std`. `assign_nearest`'s `mass_std` is
  **`NaN`** — a single nearest isochrone point carries no spread, and `0.0` would assert an
  exact mass. `assign_from_samples` is unchanged, including its genuine `0.0` at `k=1`. The
  constructor now classifies the isochrone form it was given, so calling the method that does
  not match raises by name instead of failing inside a delegate. (`04749f6`)
- **`Clustering.search_pseudoprobability`'s `selection` default changed from `"max_members"` to
  `"max_persistence"`.** `"max_members"` was the argmax of condensed-tree *row* count — the same
  unit mismatch behind the selector fix below — so at high contamination it could select a
  `min_cluster_size` rooted in the field rather than the cluster. `"max_persistence"` scores each
  sweep step by HDBSCAN's own `cluster_persistence_` for the cluster actually returned. Measured
  on the existing 54-cell benchmark across all three selection rules on identical frames; a
  coincident-row guard was added after the first measurement overstated the new rule's win (more
  than `min_samples` near-duplicate rows drive every cluster's persistence toward an
  indistinguishable value). An identical call to `search_pseudoprobability()` with no `selection`
  argument now returns a different result than in v0.1.0. Pass `selection="max_members"` for the
  old behavior, or construct `Clustering(..., legacy_cluster_selection=True)` to also restore the
  pre-fix selector below. (`e1e1a4c`, `0d03cb4`)
- **The cluster label selector no longer confuses condensed-tree row counts with flat-cluster
  point counts.** `_desired_tree_branch_size` returned a count of condensed-tree rows —
  immediate children, a mix of falling points and sub-cluster nodes — which
  `_cluster_label_for_size` matched against flat-cluster *point* counts, different units that
  coincided by accident in 23 of 83 benchmark cells. On the other 60, the fallback returned the
  largest non-noise cluster, which at contamination 0.8–0.95 is the field: selected-branch purity
  fell to 0.394 ± 0.052 despite HDBSCAN isolating the cluster at ≥0.8 purity in 96% of cells. The
  label is now resolved from the final model's own condensed tree instead. `Clustering.__init__`
  gained `legacy_cluster_selection: bool = False`, which exists solely to reproduce results
  published before 2026-08-03 by restoring the old (defective) selector; there is no other reason
  to set it. (`14454e5`)
- **`IsochroneFitter.__init__`, `ClusterAnalyzer.fit_isochrone` and
  `ClusterAnalyzer.prepare_isochrone_fitter` require `loga_range`, `dm_mu` and `dm_range`** —
  they no longer default to `(6.0, 7.0)`, `10.2` and `(9.5, 10.7)`. Those were NGC 6383's own
  fitted values, baked into the general-purpose API; a caller who omitted them got that one
  cluster's priors for whatever cluster they were fitting, silently. `fit_isochrone` also built
  its own `IsochroneFitter` and re-injected the same three defaults internally even after the
  first pass removed them from the constructor, so the leak had to be closed twice. Every call
  site that relied on the old defaults now raises `TypeError` for the missing keyword instead of
  fitting silently against NGC 6383's numbers. (`1646656`, `a539363`)

### Fixed — BREAKING
- **A `probability_threshold` against a missing probability column now raises `KeyError` instead
  of silently returning the unfiltered table.** Four call sites —
  `erotica.selection.census_detectability_from_members`, `ClusterAnalyzer.center_determination`,
  `.half_mass_radius` and `.half_light_radius` — tested `column in table.colnames` and skipped
  the filter entirely when it was absent, returning every row with no indication that nothing had
  been filtered. Measured on a 331-row table from `Clustering.search()` (which writes only
  `probability_hdbscan`, not `probability`): requesting `probability_column="probability"` at
  threshold 0.6 silently returned all 331 rows, 67 of them (20%) never checked against that
  threshold at all. The new
  `erotica._membership.select_by_probability` raises `KeyError` naming which method wrote which
  column instead. A call that used to return an unfiltered table because the wrong column was
  requested will now raise. (`a539363`)
- **`distance_model`'s error-aware branch is now marginalized in closed form** instead of sampling
  a per-star latent `r_true ~ Gamma(mu_r, std_r)` with `r ~ Normal(r_true, errors)`. That centered
  hierarchy is Neal's funnel whenever `std_r` is much smaller than the catalog errors —
  precisely the regime the branch exists for — and is the mechanism behind v0.1.0's stated
  limitation that no `distance_model` fit above ~250 stars should be trusted (R-hat 1.041–1.817,
  ESS as low as 5, up to 344 divergences). With a normal population the latent integrates exactly:
  if `r_true ~ N(mu, s)` and `r | r_true ~ N(r_true, e)`, then `r ~ N(mu, sqrt(s² + e²))`, and the
  per-star parameters — and the funnel — are gone. Verified on the same data that showed the
  funnel: R-hat ≤ 1.009, ESS 849–992, zero divergences across three seeds. **This changes the
  posterior an identical error-aware call returns** — it is the fix for the v0.1.0 known
  limitation, not a cosmetic change; the no-errors branch is unaffected and keeps sampling the
  Gamma. `metadata["population"]` now records which family was fitted. (`f190562`, `0743ee9`)
- **`provenance.load_results(..., only_last=True)` returned one parameter of a fit instead of the
  whole fit.** `datetime.now().isoformat()` was evaluated inside a per-row comprehension and
  carries microseconds, so a single call's rows each got a different `Date_Time`; filtering on the
  maximum then kept whichever parameter serialized last. A King fit over `R_c, R_t, k, b` came
  back as one row instead of four. Every parameter of a call now shares one timestamp. Trace-index
  allocation is also no longer a raw clock reading: two `store_trace_results` calls landing in the
  same wall-clock second used to get the same index and collide, and two calls a second apart used
  to get different indices and silently duplicate an identical trace. `_allocate_trace_index` now
  resolves against what is already on disk — free slot: take it; same trace present: reuse its
  slot; different trace: step forward — so storing a trace twice now reuses its slot instead of
  archiving a duplicate, and two calls in one second no longer overwrite each other.
  (`d0fe388`, `25d9289`, `682b94a`)

### Added
- **An external oracle for `king_profile`**, which `tests/CLAUDE.md` had advertised for months
  without one existing: a form-independent check against King (1962) Eq. 18 (always runs), plus
  a cross-check against `ocelot`'s `king62` under a new **`oracles`** extra (skips if absent).
  (`04749f6`, `3db1ddf`)
- **`erotica.core.NoCandidateClusters`** (a `RuntimeError` subclass), raised when a
  `search_pseudoprobability` sweep finds no candidate cluster. Measured over 24 field seeds: 82
  occurrences across 70 cells of the 5D arm and 12 of the 3D arm, and zero once there is
  structure — an unstructured field is the expected response, not a crash, but it previously
  arrived as a bare `RuntimeError` indistinguishable from one. Inherits from `RuntimeError`, so
  existing `except RuntimeError` callers are unaffected. (`329d3a8`)
- **`Clustering.search_pseudoprobability` gains four keyword arguments**, all opt-in with the
  previous hardcoded behavior as their default:
  - `probability_method: str = "hdbscan"` — `"soft"` uses
    `all_points_membership_vectors` instead of HDBSCAN's `probabilities_`, which clamps to
    exactly 1.0 for 83.6% of an EOM-merged cluster's points and cannot rank them. Measured over
    2700 fits, 15 seeds: ROC-AUC 0.9867 vs. 0.7706 for soft vs. hdbscan. Off by default.
  - `recovery_frequency: str = "any"` — `"target"` counts recovery relative to the selected
    cluster rather than any cluster; requesting it with `select_cluster=False` now raises
    `ValueError` instead of being silently dropped (the target term is only defined relative to a
    selected cluster).
  - `approx_min_span_tree: bool = False` — exposes HDBSCAN's approximate-vs-exact
    minimum-spanning-tree switch, which measurably changes labels when `leaf_size` varies
    (synthetic ROC-AUC 0.8887–0.9043 by `leaf_size` under the approximate tree; byte-identical
    labels under the exact one). A no-op under the package's default
    `match_reference_implementation=True`, which already forces the exact tree.
  - `match_reference_implementation: bool = True` — the flag this package has hardcoded since
    before v0.1.0, now exposed and documented rather than buried in `base_kwargs`. It does four
    things, not the three HDBSCAN's own source comment lists: `min_samples -= 1`,
    `min_cluster_size += 1`, `approx_min_span_tree = False`, and an extra label reassignment in
    `do_labelling`. Default unchanged, so this does not itself move results.
  (`ace5283`, `bf194ce`, `a3a7f99`, `db0fafb`, `5ede087`, `962693a`)
- **`probability_column` parameter** on `IsochroneFitter.setup`, `IsochroneFitter.cmd_distances`,
  `ClusterAnalyzer.fit_isochrone` and `ClusterAnalyzer.prepare_isochrone_fitter`, defaulting to
  the previously hardcoded `"probability_hdbscan"` (`erotica._membership.COLUMNA_ISOCRONA`). The
  rest of the package reads `"probability"` (`probabilities_ × probability_times`) by default,
  which is always ≤ `probability_hdbscan`, so the same threshold selects different members
  depending on which path produced the table; the asymmetry is documented rather than unified,
  because the accepted NGC 6383 manuscript's published numbers depend on the isochrone path
  keeping its column. (`a539363`)
- **`DistanceFitResult.mode_r`** — the closed-form mode of the predictive distribution, added
  because the only mode the package had was a KDE estimate inside a diagnostic. Analytic rather
  than KDE deliberately: measured, the KDE disagreed with the closed form by 0.0152/0.0049/0.0045
  kpc at 8k/100k/2M draws, 20–70× the difference between the two priors the methods section
  discusses. **`ParallaxFitResult.distance_mean` / `.distance_std`** — the parallax zero-point
  floor (10.3 µas, Maíz Apellániz et al. 2021) accounts for 86.5% of the reported mean's variance,
  and a naive standard error alone falls short by 2.7×. Both fields default to `NaN` and are
  additive; no previously-returned field changes. (`9cc5b93`)

### Changed
- `asteca` extra floor raised to `>=0.7` from `>=0.6`. Verified that `asteca` 0.6.9 runs to
  completion against the same calls and silently reports a *different* baseline for
  `tools/validation/benchmark_erotica_vs_asteca.py`'s published comparison table — no error, no
  warning, just a different number, so the floor is load-bearing rather than a routine bump.
  (`3980e49`)
- `h5netcdf[h5py]>=1.3` added to the `bayes` and `paper` extras. ArviZ 1.x ships no netCDF engine
  of its own, so every `InferenceData.to_netcdf()` call and every provenance sidecar failed to
  write without it; `[h5py]` is load-bearing because `h5py` is an optional extra of `h5netcdf`
  rather than one of its dependencies, so a bare `h5netcdf` installs an engine with no backend.
  (`aeb60f1`, `2861be3`)

### Fixed
- **`PhotometricMassEstimator.__init__` no longer exhausts a generator or other one-shot
  iterator passed as isochrone points.** `_classify_isochrone_form` consumed its input with
  `list(...)` to inspect it, then `__init__` stored the *original*, now-drained object; any
  iterator reaching `assign_masses` had already been consumed, so the call failed with "No finite
  isochrone points with masses were supplied" — blaming the data for a constructor defect. The
  classifier now returns `(form, items)` and both are kept. (`76a10bf`)
- **`assign_masses` now warns instead of silently discarding an isochrone** that lacks a mass
  column, or a point that is not finite, rather than dropping it without notice. The only prior
  warning fired at zero surviving points, the case that least needed one because it already fails
  on its own. (`737c89b`)
- **`king_profile` truncates at `r_t`.** Beyond the tidal radius the squared bracket climbed
  back toward `1/(1+(r_t/r_c)²)` — with `r_c=5, r_t=20, b=0` it returned `0` at `r=20` and then
  0.0061, 0.0205, 0.0371, 0.0564 at 30, 50, 100, 1000, i.e. more at large radius than at
  1.35 `r_c`. It now returns the background `b` outside `r_t`, matching the four
  implementations in this package that already did (`_king_model`, `_king_corona_model`,
  `RDP_bayesian`, and the test oracle). **No published result changes**: `king_expected_count`
  already capped its integral at `min(R_t, field_radius)`, and no fitting path calls
  `king_profile`. (`04749f6`)
- **`eff_surface_density` and `corona_surface_density` accept a Quantity scale radius**
  (`a`, `R_2`), which `king_profile` always did. The previous failure was at `1.0 + (r/a)**2`,
  not at the background term as the docstring claimed, so `b = 0` never avoided it. (`04749f6`)
- **`compare_datasets` returns its comparison** instead of only printing. Printing is unchanged
  and now behind `verbose=True`. (`04749f6`)
- **The docs build fails when it is broken.** `fail_on_warning: true` is on for Read the Docs
  and CI enforces zero warnings; previously the CI step piped `sphinx` through `tee` without
  `pipefail`, so a step named "failing on errors" passed on every build that failed. (`0e24a07`)

### Known limitations, stated rather than deferred
- **The deprecated `RDP_bayesian`'s `priors` / `priors_parameters` arguments remain a no-op.**
  They read back what was passed in and never reach the PyMC model, which its King priors
  hardcode as `Uniform`s. Not repaired, because `RDP_bayesian` is deprecated since 2026-08-02 in
  favor of `king_unbinned`, where the equivalent capability already works:
  `KingPriors(tidal_prior=(mu, sigma))` does enter `_king_model` as a `TruncatedNormal` on `R_t`.
  The signature is kept only for code reproducing pre-deprecation results. (`f6972db`)
- **`erotica.analysis._sagitta` has no installable extra**, and cannot: PyPI's own upload
  validator rejects any package with a direct git-URL requirement in *any* extra, which is what
  a `sagitta = ["sagitta @ git+..."]` entry would be (measured on the v0.2.0 TestPyPI rehearsal,
  HTTP 400 `Can't have direct dependency`). Install it manually per the `ImportError` message;
  see `pyproject.toml`'s `[project.optional-dependencies]` comment for why a bare `sagitta>=...`
  requirement isn't a safe substitute without first verifying that release's API.

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
