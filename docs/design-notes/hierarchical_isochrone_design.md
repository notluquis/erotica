# Hierarchical isochrone fitter — design, Chi-2026 comparison, credit

_2026-07-21. Strategy + implementation blueprint for EROTICA's JAX/NumPyro hierarchical
isochrone fitter. Companion to `isochrone_sampler_fix.md` (why the old PyMC NUTS fitter
failed). `[S]`=sourced, `[I]`=inferred/standard-method (flagged per the research)._

## 1. EROTICA vs Chi et al. 2026 — and the credit

**Chi et al. 2026, A&A 710, A160** (`2026A&A...710A.160C`; H. Chi, F. Wang, X. Tian, X. Zhu —
Yunnan Open Univ. + Guangzhou Univ.; code public at github.com/chihuanbin/r147) is the SOTA
for this exact problem and **published first** (A&A, June 2026; no arXiv preprint). Its method:
per-star Gaussian mixture (single + unresolved-binary + outlier) in (G, BP−RP)+parallax, params
{age, [M/H], dm, Av, f_bin, mass-ratio γ, rotation α/β}, **NUTS in NumPyro** over a **4D
differentiable PARSEC v2.0 emulator** (`jax.scipy.ndimage.map_coordinates`; steps ω 0.1, age
0.05 dex, [M/H] 0.1 dex). Target: **Ruprecht 147, an intermediate-age (~2.3 Gyr) single
cluster.** `[S]`

**Chi explicitly does NOT do:** young / pre-main-sequence clusters; membership (assumed);
selection function / completeness; spots / magnetic activity; multiple clusters. `[S]`

**EROTICA's honest position** — the JAX-`map_coordinates` emulator + NUTS/NumPyro per-star
mixture *is Chi's design*, built on **public tools** (JAX, NumPyro, PARSEC/MIST — nobody's
invention) and Chi's **public code**. So: **adopt it, cite Chi 2026 as the method reference,
and contribute a genuine extension** — no priority claim (EROTICA's NUTS fitter is still broken;
Chi's works and is published). The differentiation is on the **science**:

| | Chi 2026 | **EROTICA's extension** |
|---|---|---|
| Regime | intermediate-age (2.3 Gyr) MS | **young / PMS (~3.5 Myr)** — harder; spots/magnetic (P05) |
| Membership | assumed | **calibrated p̃ pipeline** (EROTICA's lead novelty) |
| Selection function | none | `erotica.selection` (gaiaunlimited + Hunt 2026) |
| Scope | single cluster | **multi-cluster hierarchical** (Sgr OB1 / 10-cluster programme) |
| Model choice | PARSEC only | MIST/PARSEC/SPOTS + KOH discrepancy budget (P05) |

That wedge (young-PMS + calibrated membership + selection-function-aware + population-level) is
real, unoccupied, and matches EROTICA's actual clusters. Frame P02/P05 as building on Chi 2026.

## 2. Hierarchical blueprint — per-star latents ARE the bottom plate

"Hierarchical" = the multi-level structure; "per-star latents" = one level of it. They compose.
Three plates (data attach only at the bottom; information flows *up* via shared hyperpriors):

- **Upper — population hyperparameters (one per cluster):** binary fraction f_bin, IMF/mass-
  function slope, age spread, rotation-rate distribution, extinction-field spread.
- **Middle — cluster parameters:** age, [Fe/H], distance modulus, mean A_V.
- **Lower — per-star latents (one set per star):** mass (≡ EEP along the isochrone), binary
  flag + mass ratio q, membership, per-star extinction/rotation → mapped through the
  differentiable isochrone → predicted (G, color) vs Gaia + per-star errors.

**Precedents (disaggregated + sourced):** per-star *mass along a physical isochrone* = Chi 2026
`[S]` (EROTICA's target); per-star *distance* = Kalkayotl / Olivares 2020 `2020A&A...644A...7O`
`[S]`; per-star *membership* = Sarro 2014 `2014A&A...563A..45S` + DANCe (Olivares 2018/2019,
empirical isochrone) `[S]`; partial pooling/shrinkage = Gelman BDA3 `[S]`; **multi-cluster
hierarchical (the frontier)** = Si 2018 (arXiv:1806.06733), Wen 2023 (arXiv:2311.03704) `[S]`.

**Is hierarchical "better"?** Yes when parameters are correlated/poolable, each star has few
photons, and you need honest per-object uncertainty — all true for cluster CMDs. Cost: the
**Neal's-funnel** geometry (population scale × its latents pinch); cure = **non-centered
reparametrization** (Betancourt & Girolami 2015, arXiv:1312.0906): sample z~N(0,1), θ=μ+σ·z.
`[S]`

## 3. JAX/NumPyro implementation (doc-grounded)

1. **Regular EEP grid** `[I, MIST/Dotter method]`: parse MIST `.iso.cmd` (one file per [Fe/H]);
   **align on the EEP index, not mass** (mass at fixed EEP shifts with age/Z); clip to the common
   EEP range; interpolate onto uniform `(N_age, N_feh, N_EEP)` axes → **two** tensors `grid_G`,
   `grid_color` (map_coordinates is scalar-valued).
2. **Differentiable lookup** `[DOC]`: `jax.scipy.ndimage.map_coordinates(grid, coords, order=1,
   mode="nearest")` — **`order=1`** (linear, differentiable), **`mode="nearest"`** (the default
   `cval=0` re-creates the gradient cliff you are removing).
3. **NumPyro model** `[DOC API]`: priors via `numpyro.sample` (informative parallax/XP→dm,
   dust→Av); per-star × per-EEP component log-probs (single/binary/outlier), `logsumexp` with
   IMF weights to **marginalize EEP** → per-star logL; `numpyro.factor("loglik", logL.sum())`.
   (If you need LOO/WAIC, use `MixtureGeneral` + `sample(..., obs=)` under `numpyro.plate` so a
   pointwise `log_likelihood` exists — `factor` doesn't leave one.)
4. **NUTS with a BLOCK mass matrix** `[DOC]`: `NUTS(model, dense_mass=[("log_age","feh","dm",
   "Av")])` — dense **only** over the 4 correlated globals (the age–dm–Av–[Fe/H] ridge),
   diagonal elsewhere. **Do NOT use `dense_mass=True`** with per-star latents (O(d²) → OOM).
   **Non-center every per-star latent** via `numpyro.handlers.reparam` + `LocScaleReparam`.
5. **Validation** `[DOC+I]`: `az.from_numpyro(mcmc, ...)`; injection-recovery via
   `Predictive(model)` (simulate CMD from known θ → refit → compare); SBC = manual
   prior→simulate→fit→rank-of-truth, assert uniform rank histogram. Gate: Vehtari 2021
   (R̂<1.01, ESS>400, 0 divergences) + the met-slice regression probe (~N/N unique, not 11/200).
6. **Speed** `[DOC API; numbers I]`: pure JAX → jit/vmap inside NUTS; `numpyro.set_platform("gpu")`;
   ~10–100× over PyMC — but the decisive win is qualitative (nonzero gradients → NUTS mixes).

## 4. Roadmap
- **v1 (now, Chi-2026 tier, published/achievable):** three-plate **single-cluster** model —
  per-star mass (+binary/q, membership) latents under cluster age/[Fe/H]/dm/Av, population
  hyperpriors on f_bin + MF slope. Non-center per-star latents from day one. PoC-first: build
  the differentiable EEP interpolator, prove ~N/N unique met-slices (vs 11/200), then the model.
- **v2 (frontier, the PhD-novelty vector):** multi-cluster **top plate** — shared IMF /
  age–metallicity relation across the Sgr OB1 / 10-cluster set (Si-2018 style). Only after v1 is
  stable + injection-recovery-validated.
- **Young/PMS + SPOTS/magnetic** (P05) and **calibrated-membership integration** (P02) are the
  science differentiators layered on the same machinery.

## 4b. Implementation status (2026-07-21) — build + convergence diagnosis

**Proven / working (prototypes in `tools/prototypes/`):**
- **Differentiability fix** (`isochrone_jax_poc.py`): `map_coordinates(order=1)` over an
  aligned MIST grid → metallicity sweep 11/200 → **200/200 unique**, `jax.grad` **95%→0%**
  zero-gradient. The staircase that broke NUTS is gone. Age axis likewise (46.7%→0%).
- **Mass-based grid** (`isochrone_mass_grid.py`): research-recommended basis (initial mass,
  not EEP) → proper young CMD (G[-0.3,9.4], BP−RP[-0.26,2.90], blue→red PMS).
- **NumPyro fitter** (`isochrone_numpyro_fitter.py`): per-star single/binary/outlier mixture,
  marginalized over mass. **Recovers truth** (age 6.76 vs 6.6, feh 0.06 vs 0, dm 10.1 vs 10.3,
  Av 1.1 vs 1.2, f_bin 0.37 vs 0.3) — the model is correct (logL(truth) beats a wrong point by
  ~4300). **Mixture params fully converge** (f_bin R̂ 1.01/ESS 813, f_out R̂ 1.00/ESS 4600).

**Convergence — where it landed (empirical, ~20 MCMC runs):** the fix direction is
**per-star mass latents** (Chi 2026's actual code: sample mass in `plate('stars',N)`, don't
marginalize) + a tight parallax→dm prior. This took R̂ **2.84 (marginalized) → ~1.74 (latents)**,
recovering truth (dm converges, R̂ 1.18). **But it plateaus at R̂ ~1.7–1.8** — longer chains
(3000 vs 1200) do NOT improve it, and ESS stays ~5 (doesn't grow with samples). So the block
mixes far better but not fully (R̂<1.05). **Root cause of the residual:** a young/PMS cluster's
age is set by the *diffuse PMS locus* where **age and per-star mass trade off strongly**, a
harder geometry than Chi's cluster (Ruprecht 147 is *old*, with a sharp turnoff that pins age).
The C0-vs-C1 interpolation was **ruled out** (cubic tested, no help); the residual is geometry.

**Tried + insufficient (cheap fixes exhausted):** `init_to_median` (got means→truth), realistic
photometric scatter (R̂ 2.84→2.39), the binary/outlier mixture (converged the fractions),
`dense_mass` block, tighter dust/XP priors. None fixed the frozen core.

**Next directions (focused, deeper — the real cure):**
1. ~~C1-smooth interpolation~~ **TESTED + RULED OUT (2026-07-21):** replaced `map_coordinates
   (order=1)` with `interpax` cubic (C1) — the 4 globals **still froze** (R̂ 2.7–2.86). So the
   cause is NOT gradient roughness; it is the **curved degeneracy-ridge geometry itself**.
2. ~~Per-star mass latents~~ **DONE — the fix direction (R̂ 2.84→1.74), but plateaus ~1.7.**
   Remaining residual = the young-cluster per-star **age↔mass** degeneracy (diffuse PMS locus).
3. **Reparametrize the per-star age↔mass degeneracy** — the residual coupling. E.g. sample a
   per-star "position-along-isochrone" that is age-invariant, or reparametrize mass conditional
   on age; non-center the latents. **Leading next step.**
4. **Nested sampling / SMC for the global block** — dynesty on {age,feh,dm,Av} with the mass
   latents/mixture handled separately, if HMC can't close the young-PMS geometry.
5. **Validate on an OLD cluster first** (like Chi's R147) — expect clean convergence there;
   isolates that the residual is the young/PMS regime, not the code.
Empirically falsified: interpolation roughness (cubic), init, scatter, tighter priors, dense
mass, longer chains. Confirmed helpful: per-star latents, tight dm prior, realistic scatter.
Gate unchanged: injection-recovery + Vehtari 2021 before paper use.

## 5. Credit / attribution (integrity)
Cite **Chi et al. 2026** as the method reference for the differentiable-emulator + NUTS/NumPyro
per-star-mixture architecture; acknowledge their public code if EROTICA ports the emulator design.
Cite the public tools (JAX, NumPyro, PARSEC/MIST, corner) via `\software{}`. EROTICA's claim is
the **young-PMS + calibrated-membership + selection-function-aware + population-hierarchical
extension**, not the base method. This is standard science: build on published work + public
packages, contribute a genuine, differently-scoped result.

## Sources
Chi et al. 2026 A&A 710 A160 (`2026A&A...710A.160C`) · Olivares 2020 (`2020A&A...644A...7O`,
Kalkayotl) · Sarro 2014 (`2014A&A...563A..45S`) · Olivares 2018/2019 (DANCe) · Si 2018
(arXiv:1806.06733) · Wen 2023 (arXiv:2311.03704) · Betancourt & Girolami 2015 (arXiv:1312.0906) ·
Gelman BDA3 · jax.scipy.ndimage.map_coordinates docs · NumPyro docs (factor, plate, reparam,
NUTS dense_mass block form) · arviz from_numpyro (v0.20).

_Caveat: Chi 2026's exact latent parameterization (sampled vs marginalized per-star mass) is
inferred from the abstract/paper text, not a line-by-line reading of their code; verify against
github.com/chihuanbin/r147 before porting._
