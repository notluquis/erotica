<!--
Variant A -- the isochrone module described as a VALIDATED capability, for swapping into
paper/paper.md if the gate ever passes. Written 2026-09-27, INACTIVE.

Status against agent-findings/isochrone-nuts-convergence-2026-09.md S9.5's own criteria,
as of the 2026-09-25 batch on the current unbinned per-star likelihood:

  C1 (NGC 6383, 4x2000, R-hat<1.01, ESS>400, 0 divergences)  -- rerun in progress, not done
  C2 (prior-center invariance, O-inv)                        -- MET (1e-9)
  C3 (injection-recovery, batch A, >=12/16 in 90% CI)        -- NOT MET: loga 5/10 misses
                                                                 (batches A_2-A_7 not run --
                                                                 cannot change the verdict)
  C4 (leave-one-node-out, <5% pooled fraction near a node)   -- MET (0.000%), with three
                                                                 caveats in S10.15 (loo_5
                                                                 failed the gate; three fits
                                                                 missed the search mode; loga
                                                                 misses its 90% CI in all six)

All three must pass together. C3 already failed for the batch that ran, so this variant
cannot be activated by C1 finishing alone -- it needs a NEW batch, on a likelihood where the
measured binary age bias (S10.15: binaries 0/4 on loga, medians +0.013 to +0.024 dex above
truth) has been diagnosed and fixed, then a fresh C1/C3/C4 run that passes.

DO NOT paste the blocks below into paper/paper.md until that fresh batch exists. Replace
every [bracketed placeholder] with that batch's real numbers, cited to its finding section
(finding-fidelity: cite the number, do not restate it from memory), then re-run
scratchpad/wc_joss.py (see agent-findings/joss-prep-2026-09-27.md) to confirm the word count
still clears 1750, and re-check that every @key below still resolves in paper.bib.
-->

## Summary -- replaces "experimental gradient-based isochrone fitting"

gradient-based isochrone fitting

## State of the field -- replaces the isochrone paragraph (the block starting
   "For isochrones, `EROTICA` implements an unbinned per-star likelihood...")

For isochrones, `EROTICA` implements an unbinned per-star likelihood over isochrones
interpolated at fixed evolutionary phase, with unresolved binaries marginalized over the
mass ratio and a completeness term, sampled with a No-U-Turn Sampler -- replacing an earlier
*binned* Poisson Hess-diagram form [@dolphin2002] whose discretization locked the posterior
onto the grid's own reference values. `BASE-9` established unbinned per-star Bayesian
single-cluster CMD fitting with non-gradient MCMC [@vonhippel2006]; @chi2026 apply a
No-U-Turn Sampler to a differentiable, unbinned per-star isochrone likelihood for an open
cluster -- the same family used here -- and @garling2025 sample a *binned* Poisson
Hess-diagram likelihood with Hamiltonian Monte Carlo. Injection-recovery on synthetic
clusters, including unresolved binaries, recovers age, distance modulus, extinction and
metallicity within their 90% credible intervals in [N] of [M] trials, with every parameter
meeting the convergence gate (R-hat < 1.01, ESS_bulk > 400, 0 divergences) on [K] of [M]
fits [cite the passing finding section here, e.g. `isochrone-nuts-convergence-2026-09.md`
SX.X].

## Software design -- no change from the active (B) text

The "Both modules now evaluate an unbinned, per-star likelihood, for different measured
reasons rather than a shared design" paragraph already describes the shipped code under
variant A too (the likelihood does not change between B and A -- only whether its recovery
is validated does). Do not revert it when swapping in variant A.

## Research impact statement -- optional addition, not required to activate variant A

- [Optional: a bullet reporting the passing batch's injection-recovery bias/coverage
  numbers for age, distance modulus, extinction and metallicity, cited directly to its
  finding section rather than restated from memory.]

## Checklist before activating this variant

- [ ] A fresh C1/C3/C4 batch exists and all three are met, per S9.5's criteria (not a
      loosened version of them).
- [ ] The binary age bias (S10.15) has a diagnosed cause and a fix, not just a passing rerun
      that happens not to reproduce it.
- [ ] `python3 scratchpad/wc_joss.py paper/paper.md` (method in
      `agent-findings/joss-prep-2026-09-27.md`) still reports <=1750 after the swap.
- [ ] Every `@key` cited above still resolves in `paper/paper.bib`.
- [ ] `pyproject.toml` / `CITATION.cff` version matches whatever release this batch shipped
      under (a new one, if the code changed further to fix the binary bias).
