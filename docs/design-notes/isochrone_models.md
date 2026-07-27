# Isochrone / evolutionary-model grids — reference for EROTICA

_2026-07-21. Which stellar-model grids exist, their photometric systems, and how to use/combine
them in the fitter + P05 (model-choice ages). Grounded in the official model sites + the P01
thesis/referee record. `[S]`=site/source-confirmed, `[I]`=founding-paper/inferred (approximate)._

## On disk already (`data/test/NGC6383/`)
- **MIST** v1.2 — `MIST/UBVRIplus/*.iso.cmd`, 15 [Fe/H] files, **EEP-based**, UBVRIplus (incl. Gaia).
- **PARSEC** v1.2S — `PARSEC/gaiaedr3/mets_ages.dat` (Gaia EDR3) + `PARSEC/2mass/` — **mass-based**
  (`Zini MH logAge Mini … Gmag G_BPmag G_RPmag`), 6 metallicities (MH −0.18…+0.70, metal-rich),
  300 ages (logAge 5–8, young), 544k rows.
- **ASteCA** bundled Gaia-EDR3 isochrones (`ASteCA/isochrones/gaiaedr3/`, PARSEC-based).
So a first MIST-vs-PARSEC comparison needs no downloads — but the on-disk PARSEC is young + metal-rich
only, so a *fair* P05 comparison needs matched-coverage grids.

## The grids that matter (crux: basis + photometric systems)

| Grid | Basis | Rotation | Mag/Spots | PMS | Gaia | VISTA/VVV | Notes |
|---|---|---|---|---|---|---|---|
| **MIST** v1.2 | **EEP** | v/vcrit 0,0.4 `[S]` | — | yes | **DR2 only** `[I]⚑` | **yes** `[S]` | [Fe/H] −4…+0.5, [α/Fe]=0 only; 0.1–300 M☉ |
| **PARSEC** v1.2S | mass | — | — | yes (~0.09) | DR2 | **yes** `[S]` | +COLIBRI TP-AGB; Z 1e-4–0.06 |
| **PARSEC** v2.0 | mass | **ω/ω_crit 0–0.99** `[S]` | — | yes | **EDR3** `[S]` | yes | **Chi 2026 used this**; reaches ~14 M☉ (NGC 6383's O7 HD 159176); +Roman/JWST/Euclid |
| **BaSTI-IAC** | mass | **none** `[S]⚑` | — | yes | **DR1/DR2/EDR3/DR3** `[S]` | **yes** `[S]` | **uniquely Gaia DR3 + VISTA in one grid** → best for NIR (P06); [α/Fe] −0.2/0/+0.4 |
| **Dartmouth/DSEP** | mass | — | — | limited | DR2 | **no** | JC+2MASS+SDSS+HST |
| **Feiden** (magnetic) | mass | PMS | **B-field magneto-conv.** | yes | Dartmouth systems | — | ~0.1–1.7 M☉; bespoke Dartmouth runs (not turnkey) |
| **BHAC15** (Baraffe) | mass | — | — | **core PMS** | tables `[I]` | `[I]` | **solar-Z only**, ≤~1.4 M☉; best non-magnetic low-mass PMS interiors |
| **SPOTS** (Somers) | mass | — | **spot f=0–0.85** | yes | some mags `[I]` | — | **solar-Z, ≤1.3 M☉**; turnkey (Zenodo/GitHub) |
| Geneva, YaPSI, Victoria-Regina, PISA | mass | (Geneva rot) | — | mixed | mostly JC/2MASS | — | niche |

Sites: MIST `mist.science`; PARSEC/COLIBRI `stev.oapd.inaf.it/cmd` (pick version × COLIBRI × one
phot. system); BaSTI-IAC `basti-iac.oa-abruzzo.inaf.it`; Dartmouth `rcweb.dartmouth.edu/stellar`;
Feiden/BHAC15/SPOTS/Geneva = author-site/GitHub/Zenodo ASCII.

## Regime → grid
- **Young low-mass PMS** (≲1.3 M☉, K/M): **SPOTS or Feiden magnetic**; BHAC15 as non-magnetic baseline.
- **Young massive** (OB, upper MS — NGC 6383's O7): **PARSEC v2.0** (to ~14 M☉) / MIST.
- **Intermediate-age turnoff / rotation**: PARSEC v2.0, MIST v/vcrit, Geneva.
- **Old MS**: MIST, PARSEC.
- **Gaia colors, all M dwarfs**: apply **Wang 2025** corrections (`2411.12987`), **prefer G−RP over BP−RP**.
- **NIR (VVV/VISTA, P06)**: **BaSTI-IAC** (Gaia DR3 + VISTA) or PARSEC.

## The PMS magnetic/spots systematic (load-bearing for young clusters)
Non-magnetic, theoretical-color PMS models make cool K/M stars appear **~2× younger** than warm
A/F/G stars in the same cluster — **uniformly one-directional** (every fix pushes low-mass ages
*older* into agreement). `[S]` Feiden 2016: ~2.5 kG fields give a consistent 10 Myr across A–M in
Upper Sco vs ~5 Myr non-magnetic = **factor-2 upward shift `[S]`**. Bell 2013: empirical colors/BCs
below 4000 K raise young ages up to a factor 2 `[S]`. **Reconciliation of the P01 vs thesis
numbers:** the paper's "**factor ~2 or more**" is the sourced/ceiling value; the thesis/P05
"**30–50%**" is the moderate-spot regime `[I]`. → **Do not age NGC 6383 from the uncorrected
M-dwarf locus.** Fit low-mass PMS with a spotted/magnetic grid + Wang color corrections.

## Multi-grid fitter recipe (P05)
1. **Basis = INITIAL MASS, not EEP.** EEP doesn't cleanly parametrize the PMS Hayashi/Henyey
   contraction; initial mass works for MIST *and* PARSEC and is correct on the PMS. (The PoC used
   EEP — fine for the differentiability proof; the young-cluster multi-grid fitter should be
   mass-based.) Regrid every model onto one `(initial-mass, log-age, [M/H])` cube **per photometric
   system**, then the differentiable `map_coordinates` interpolator is grid-agnostic.
2. **Stitching is forced:** BHAC15/SPOTS cap ~1.3–1.4 M☉, so the O-star members *require* stitching
   a low-mass PMS grid (BHAC15/SPOTS/Feiden) to PARSEC-v2/MIST for the massive end.
3. **Systematic budget:** KOH (Kennedy & O'Hagan 2001 — stats paper, no ADS bibcode) per-grid
   discrepancy → envelope of inferred ages; **or** Bayesian model averaging / stacking with
   **LOO/WAIC weights** (Vehtari 2017), each grid a model, age posterior propagated across grids.
4. **Fitter template:** Chi 2026 (`2026A&A...710A.160C`) — JAX-differentiable emulator + NUTS
   hierarchical Bayes; add a grid-index/stacking weight to make MIST-vs-PARSEC-vs-BHAC15-vs-SPOTS
   native. Cite as **method precedent, not a grid**.

## P01 integrity guardrail (from `FINE_REFEREE_AUDIT.md:41`)
P01's age is **MIST-conditioned**; the paper must NOT imply PARSEC/Baraffe/SPOTS were run — they
are named as the deferred comparison. **P05 is where they actually get run.** The referee (R9)
explicitly demanded the grid-systematics acknowledgement that is now in `aanda.tex:254`.

## Flags
`[S]` = read off the official site this session (photometric-system menus, MIST/BaSTI [Fe/H]/rotation).
`[I]`/⚑: MIST Gaia-DR2-only + BHAC15 Gaia tables inferred (not on-page); BaSTI has **no rotation**
axis (contra a common assumption); age/mass/[Fe/H] ranges mostly from founding papers, approximate;
the "30–50%" spot shift and "0.1–0.3 dex" inter-grid floor are approximate, not verbatim. WebSearch
was budget-exhausted, so grounding is official sites + the local thesis/referee record.
