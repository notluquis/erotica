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

DO NOT paste the block below into paper/paper.md until that fresh batch exists. Replace
every [bracketed placeholder] with that batch's real numbers, cited to its finding section
(finding-fidelity: cite the number, do not restate it from memory), then re-run
`python3 agent-findings/scripts/joss_wc.py paper/paper.md` (in ~/phd; see
agent-findings/joss-prep-2026-09-27.md) to confirm the word count still clears 1750, and
re-check that every @key below still resolves in paper.bib.

WORD BUDGET, measured against the (B) text this replaces: the block below is written to be
close in length to the sentence it swaps out (the "This module is **experimental**: ..."
sentence, ~24 words), so activating it should be close to word-neutral against whatever
paper.md's word count is at swap time -- re-measure rather than assume. Do NOT also restore
the longer "over isochrones interpolated at fixed evolutionary phase, with unresolved
binaries marginalized over the mass ratio and a completeness term" clause that an earlier
draft of this file carried in the first sentence -- that clause is not gated by C1/C3/C4 (it
is true of the code regardless of whether recovery is validated) and was cut from (B) for
word budget; adding it back to (A) too would cost ~20 words this file does not budget for.
-->

## Summary -- replaces "experimental gradient-based isochrone fitting"

gradient-based isochrone fitting

## State of the field -- replaces ONLY the last sentence of the isochrone paragraph
   (the sentence starting "This module is **experimental**:", not the whole paragraph)

Injection–recovery on synthetic clusters, including unresolved binaries, recovers age,
distance modulus, extinction and metallicity within their 90% credible intervals in [N] of
[M] trials, meeting the convergence gate (R-hat < 1.01, ESS_bulk > 400, 0 divergences) on [K]
of [M] fits [cite the passing finding section here, e.g.
`isochrone-nuts-convergence-2026-09.md` SX.X].

## Software design -- no change from the active (B) text

The "Both modules now evaluate an unbinned, per-star likelihood, for different measured
reasons rather than a shared design" paragraph already describes the shipped code under
variant A too (the likelihood does not change between B and A -- only whether its recovery
is validated does). Do not revert it when swapping in variant A.

## Research impact statement -- optional addition, not required to activate variant A

- [Optional: a bullet reporting the passing batch's injection-recovery bias/coverage
  numbers for age, distance modulus, extinction and metallicity, cited directly to its
  finding section rather than restated from memory. This is new content, not a swap, so
  budget its word cost separately -- see agent-findings/joss-prep-2026-09-27.md S3 for what
  else in the paper was cut to make room the last time words were tight.]

## Checklist before activating this variant

- [ ] A fresh C1/C3/C4 batch exists and all three are met, per S9.5's criteria (not a
      loosened version of them).
- [ ] The binary age bias (S10.15) has a diagnosed cause and a fix, not just a passing rerun
      that happens not to reproduce it.
- [ ] `python3 agent-findings/scripts/joss_wc.py paper/paper.md` (run from `~/phd`) still
      reports <=1750 after the swap.
- [ ] Every `@key` cited above still resolves in `paper/paper.bib`.
- [ ] `pyproject.toml` / `CITATION.cff` version matches whatever release this batch shipped
      under (a new one, if the code changed further to fix the binary bias).
