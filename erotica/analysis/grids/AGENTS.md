# erotica/analysis/grids/ — isochrone grids behind one contract

> **Contexto raíz:** bajo la convención `AGENTS.md` **gana el fichero más cercano y no se
> concatena** — al contrario que `CLAUDE.md`, que los concatena todos. Lee también
> `erotica/analysis/AGENTS.md` y el `AGENTS.md` de la raíz. `~/phd` es el hub, un repo aparte que
> puede no estar presente: lo que dependa de él va marcado.

A grid here is nodes in **([Fe/H], log t)**, each an isochrone ordered along an EEP coordinate
(`native` / `pseudo` / `mass`, see `base.py`). `IsochroneFitter(grid=...)` interpolates linearly
between nodes at fixed EEP and samples `feh`, the grid's **own [Fe/H] label** — never linear Z,
because the label means different things in different grids (measured in
`tools/validation/isochrone_grids/feh_conventions.py`: MIST v1.2 `log10(Z/0.0142857)`, MIST v2.5
`log(Z/X) − log(Z/X)⊙` with Z⊙ = 0.0163577, PARSEC v1.2S `log(Z/X) − log 0.0207`).

The legacy path `IsochroneFitter(isochs_path=...)` (MIST files, `met` = linear Z) is unchanged and
is held to the code that ran C1 by `tests/test_grids.py::test_c1_regression_mist_v12_backend_against_the_code_that_ran_c1`.

## The rule: no backend enters without its control and its oracle

A new backend (or a new version of one) is not usable until all four exist, each as a test or a
script with a JSON sidecar:

1. **Reader oracle** — an independent reading of the same files (the legacy reader for MIST,
   ASteCA's `interp_isochrones` for the mass-quantile scheme, the published header values for the
   metallicity convention). Not a re-read with the same code.
2. **Hold-out validation** — `holdout.py` at 2× the native step, run in
   `tools/validation/isochrone_grids/holdout_backends.py`: G, colour and **inferred log-age** error.
   This is the grid's error; R̂ says nothing about it (methodology §A.1.2).
3. **Control** — the same cluster, the same likelihood and window, fitted with another grid; the
   difference in log t (and [Fe/H] when both are free) is reported as a model systematic, never
   averaged away. PMS-only grids (BHAC15, SPOTS) fix [Fe/H] = 0, so the main grid is re-fitted with
   `grid.select(feh=0.0)` for the comparison.
4. **External oracle** — a published result reproduced within a tolerance written BEFORE running
   (λ Ori / Cao+22 for the PMS controls: hub `agent-findings/isochrone-grids-implementation.md`).
   A failure is a finding, not a tolerance to widen.

Mutation-test each one (re-apply the defect, see red, restore); the mutations already run are listed
in the hub finding.

## Measured traps (2026-10-04)

| trap | number | consequence |
|---|---|---|
| MIST v2.5 [Fe/H] = +0.5 has **no model below ~0.50 M☉** at log t 6–7 (v1.2: 0.10) | hold-out of the +0.25 node from 0 and +0.5: orth p68 **0.55 mag**, Δ[Fe/H] saturates | v2.5 above [Fe/H] = 0 is unusable for a PMS fit as is; select the nodes or cut the window |
| MIST v2.5 at log t 6.0 starts at 0.25 M☉ (0.13 at 6.3) | — | young fits need the faint edge of `safe_window` |
| PARSEC v1.2S labels are not monotone in mass (1168/1800 isochrones) | flicker 0/1 at PMS/MS; ≥ 20 M☉ labelled 0 after the MS | `effective_phase` = running max; test pins it |
| ASteCA-style mass-quantile resampling interpolated between ages | MIST hold-out orth p68 0.36 mag (N = 400), 0.06 (N = 5000); Δlog t up to 0.05 | not used; arc-length pseudo-EEP instead (orth 0.0036) |
| SPOTS f = 0.34 has Gaia only for M ≥ 0.55–0.65 M☉ (−99 below); 2MASS reaches 0.10 | — | Gaia controls need the faint edge; or fit 2MASS |
| a grid comparison in the CMD mixes physics and **colour tables** | λ Ori (Cao+22, pre-registered): SPOTS 0.34 − 0 = +0.053 dex in (J, J−Ks) against +0.211 published; +0.210 in Cao's HR space with the same likelihood. J−Ks at fixed T_eff: MIST −0.035, SPOTS +0.060 mag from the data | do not quote a CMD grid systematic before equalising colour/extinction (hub E13) |
| legacy reader rounds Z to 6 decimals | −0.50 node 4.51753e-3 → 4.518e-3; [Fe/H] < −3 nodes up to 30 % off | log L moves 2.8e-7 at C1; the grid path keeps header Z |

## Licences — never commit grid files

Only SPOTS declares a data licence (CC BY 4.0). MIST, PARSEC, BHAC15: none declared ("all rights
reserved" footers). Files live in `~/.cache/erotica-grids/` (`fetch.py`), are cited, and their
SHA-256 goes in `GridProvenance`. A test that needs a file skips and says which one.
