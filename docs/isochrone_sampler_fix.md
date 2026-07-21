# Isochrone NUTS sampler — diagnosis + fix spec

_2026-07-21. Why `cosmic.analysis._isochrone` NUTS fit fails (R̂ 1.5–2.2, posterior at
prior edges) and the golden-standard fix. Confirmed empirically + grounded in the 2026
literature. This is **next-paper (P02/P05) infrastructure**; the published P01 numbers stand
on the validated gradient-free DEMetropolis ensemble (see descope below)._

## Confirmed root cause (empirical)

The binned Poisson-Hess likelihood is **piecewise-constant** → NUTS gets no gradient and
random-walks. A 10-second probe on the real Hess grid (`data/40/hgrid_paper254.npz`,
shape 200×200×17×12):

- **MET axis: 189/199 adjacent isochrone-slices IDENTICAL → 95.0% of the metallicity domain
  has exactly-zero gradient.** Only **11 unique met-slices out of 200**.
- LOGA axis: 46.7% zero-gradient.

Two code-level culprits in `_isochrone.py`:
1. `_precompute_H_grid` caches each Hess cell by **nearest-neighbor MIST file index** (lines
   ~702–722) — the 200-point met linspace snaps to the 15 `feh_*` files (11 in the age
   window). `_interp_H`'s bilinear met-gradient `(H[i1]−H[i0])·…` is then exactly 0 wherever
   adjacent slices are identical.
2. `np.histogram2d` hard binning (lines ~570/654/665) — zero gradient at bin interiors,
   delta-like jumps at edges.
Plus a scale–shape ridge: a free `log_s = Normal(log N_obs, 1)` multiplies a θ-dependent
total count, coupling global scale to CMD shape (H2); dm→9.7 vs the 1.11 kpc parallax is a
genuine likelihood edge-pull, not just sampler noise.

## Is the posterior multimodal? (empirical, 2026-07-21)

Probed the saved traces directly. **It is a degeneracy _ridge_, not separated modes:**
- **Production DEMetropolis** (300×1000): marginals unimodal-but-lumpy (bimodality
  coefficient ~0.19, far below the 0.555 bimodal threshold). The apparent "10 loga / 4 met
  modes" are **staircase artifacts** — bumps at the 11 discrete metallicity slices.
- **NUTS refit** (4 chains): 2 "modes"/param = **4 chains stuck on different staircase
  plateaus** (the R̂ inflation), plus **strong degeneracy correlations dm–loga −0.94,
  dm–Av +0.70, loga–Av −0.70** — the age–distance–extinction ridge.

This matches Plevne & Akbaba 2026: dynesty on 5056 clusters shows age–Z and dm–reddening as
**elongated/curved ridges**, "none... multi-modal." Informative priors (parallax→distance,
XP→[Fe/H], SFD→reddening) collapse the ridge so posteriors read compact. Genuine separated
modes (young-reddened vs old aliasing, sparse CMDs, binaries) are a real minority case.

## Recommended sampler (supersedes the "make NUTS work" framing)

**Primary: nested sampling (dynesty).** Gradient-free (so it works on the CURRENT likelihood —
no JAX rewrite needed to sample), robust to the ridge AND to any latent multimodality, and it
returns **Bayesian evidence → model comparison (MIST vs PARSEC vs SPOTS = P05)**. Field-proven:
Plevne 2026 used dynesty (`N_live=400`, MultiEllipsoid, ΔlnZ=0.01, 2–5 min/cluster). Cross-check
with UltraNest; nautilus only if the likelihood becomes expensive; PolyChord overkill at ~6D.

**First-line degeneracy fix (do before anything fancy): informative priors** — Gaia parallax→
distance, Gaia XP metallicity→[Fe/H], 3D dust (Edenhofer 2024)→reddening. This collapses the
ridge per Plevne, and is the cheapest highest-impact change.

**Decision rule (per cluster):**
1. Add informative priors → collapse the ridge.
2. Multi-chain NUTS, overdispersed init. R̂<1.01 + unimodal marginals → **keep NUTS** (fast
   path, the ~75% well-behaved majority).
3. Linear ridge → dense mass matrix (no reparam); curved/banana ridge → reparametrize along
   the degeneracy axis.
4. Chains split / bimodal / need evidence → **escalate to dynesty** (default) or **tempered
   SMC with a NUTS mutation kernel** (`blackjax.smc.tempered` + `smc.from_mcmc` — the branch
   that reuses gradients under multimodality); report ln Z.

**Where the differentiable work still earns its keep** (NOT the primary sampler fix):
- **Resolution**: smooth isochrone interpolation removes the metallicity quantization (11
  discrete values → continuous), which nested sampling would otherwise inherit as a coarse met
  posterior. Do it regardless of sampler.
- **Unimodal fast-path** + future hierarchical / multi-cluster models where gradients pay off.

## Likelihood / interpolation fix (resolution — see ranked options below), 2026 golden standard

**Headline:** COSMIC uses a galaxy-SFH tool (binned Hess + Poisson over an age–Z template
grid — Dolphin 2002 `2002MNRAS.332...91D`; Garling 2025 `2025ApJS..277...61G`) on a
**single-population cluster**. The SOTA for COSMIC's exact problem (one cluster, Gaia,
continuous age/Z, NUTS) is **Chi et al. 2026** (A&A 710, A160, `2026A&A...710A.160C`), which
**drops binning entirely** and kills both zero-gradient pathologies.

1. **Differentiable isochrone interpolation (highest payoff — replaces nearest-neighbor snap).**
   Chi: 4D **linear** interpolation over a *regular* precomputed PARSEC grid — inputs
   (mass/EEP, log age, [M/H], ω), outputs (G, BP−RP) — via `jax.scipy.ndimage.map_coordinates`
   on a `(N_ω, N_τ, N_[M/H], N_M, 2)` tensor; JAX makes it fully differentiable. **Not** a
   neural net. For COSMIC: build a **dense regular MIST grid in (age, [Fe/H], EEP)** +
   differentiable multilinear interp → nonzero gradient everywhere. **Interpolate on EEP
   index, not raw mass** (mass ranges shift with age/Z; the age signal is at the turnoff —
   standard MIST practice, Dotter 2016 / Choi 2016).
2. **Drop the histogram — unbinned per-star likelihood (removes the binning pathology).**
   Chi's likelihood is a **per-star Gaussian mixture**:
   `L_i = (1−f_out)[(1−f_bin)·L_single + f_bin·L_binary] + f_out·L_outlier`, each a 2D Gaussian
   in (G, color) with photometric error + jitter, marginalizing per-star mass. No bin edges →
   no piecewise-constant gradient. The **binary + outlier components are load-bearing** — they
   absorb the MS-broadening (unresolved binaries, rotation) that breaks the current fit.
3. **If binning is kept** (speed at large N): differentiable Gaussian deposit (Garling Eq. 2 —
   each star deposits its 2D error-Gaussian integrated over each bin). Gotcha: Gaia bright-star
   errors are mmag; if kernel σ ≪ bin width the deposit is near-delta and gradients collapse —
   set **σ ≳ bin width**. Unbinned (step 2) has no such floor.
4. **Likelihood + degeneracy.** Poisson, never χ² (Dolphin 2002). Poisson factorizes as
   **Multinomial(shape) × Poisson(total N)**; for a single cluster the total is a nuisance, so
   **normalize the model Hess to a probability (≡ condition on N ≡ multinomial), or profile out
   the amplitude analytically (Â = N)** — do NOT sample a free `log_s`. Unbinned removes
   normalization as a parameter entirely, so the ridge vanishes by construction.

**Garling ≠ port target:** its differentiability is only over *linear template weights*
(per-age-Z templates are precomputed, never differentiated) — borrow its Gaussian-deposit
(Eq. 2) and Poisson-ratio (Eq. 6), not its fixed-template architecture.

## Implementation path
A **JAX/NumPyro rewrite following Chi** is the lower-risk, SOTA-aligned path (PyMC/PyTensor
*can* express differentiable multilinear interp, but `jax.scipy.ndimage.map_coordinates` gives
it in one call and Chi is a proven NumPyro/NUTS template for the identical problem). New
fitter alongside the existing one; do NOT rip out the validated DEMetropolis path.

## Validation gate (before it's used in any paper)
Injection–recovery on a synthetic cluster of known (age, Z, dm, Av): must recover truth and
pass Vehtari+2021 (R̂<1.01, ESS>400, 0 divergences). Keep the gradient-flatness probe
(`unique met-slices` count) as a regression test — a smooth interpolator must give ~200/200
unique slices, not 11.

## Descope (current paper)
P01's isochrone numbers stand on the validated gradient-free **DEMetropolis ensemble**
(`data/40/fit_parameters_trace_1724708835.nc`; half-ensemble mode scatter 0.08 dex loga /
0.03 mag dm / 0.003 Z ≪ credible widths; dm 10.28–10.30 coherent with parallax) + the
parallax/PM/King stack that passes Vehtari+2021. `CHANGES.md` §14 already reworded
"NUTS"→"PyMC" and labelled the true-NUTS fitter unvalidated. If a gradient fit is wanted
before the rewrite lands, use PyMC **DEMetropolisZ/SMC** (gradient-free, correct for the
current non-differentiable target) — not NUTS.

## Sources
Chi et al. 2026 A&A 710 A160 (`2026A&A...710A.160C`) · Hon, Li & Ong 2024 ApJ 973 154
(`2024ApJ...973..154H`, flow emulator — heavier alternative) · Garling et al. 2025 ApJS 277 61
(arXiv:2407.19534) · Dolphin 2002 MNRAS 332 91 (`2002MNRAS.332...91D`) · Choi 2016 / Dotter 2016 (MIST/EEP).
