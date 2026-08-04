# tests/ — what a test has to be here

A test that has never been seen to fail is a hypothesis, not a test. This directory has produced
false confidence before: a mutation audit of 39 deliberate bugs left **18 alive (46%)**.

## The rule

**Assert on behaviour and invariants, against an oracle that exists independently of the code.**
The strongest ones already here, worth copying:

- an **analytic closed form** — `king_expected_count` against `scipy.integrate.quad` to 1e-9;
  the `King(R_t→∞) ≡ EFF(γ=2)` identity to 80 decimal places;
- a **parameter-free special case** — `fractal_cluster(D=3)` must be a uniform ball, checked as
  Poisson pulls against the exact `r³` law;
- an **independent published equation** — `king_profile` integrated by `scipy.integrate.quad` against
  King (1962) **Eq. (18)**, the cumulative count, derived by hand in `_king_eq18_count` rather than
  imported. Different algebra, so it catches a wrong prefactor, a wrong exponent *and* a wrong
  truncation. Always runs;
- an **external implementation** — `king_profile` against `ocelot`'s `king_surface_density`
  (`test_king_profile_matches_ocelot_king62`). Read its strength correctly: ocelot evaluates the
  **same algebraic form**, so it pins transcription and truncation, not the shape of the model —
  which is why the Eq. (18) check above exists alongside it. **`ocelot` is not a declared dependency
  of this project**, so the test is `importorskip`-gated and does *not* run in a clean checkout;
  `pip install --no-deps ocelot` enables it;
- a **monotonicity that cannot be faked** — a tighter σ-clip can never keep more sources, and its
  selection must be a *subset* of the looser one.

> **The ocelot bullet was aspirational until 2026-08-04; the Eq. (18) bullet above it is new.**
> This file claimed the ocelot cross-validation as an existing strength of the suite;
> `grep -rn "ocelot\|King62" tests/` returned only this sentence. `King62` truncates at `R_t` and
> `king_profile` did not, so the advertised oracle would have caught that defect on day one. Instead
> an un-truncated King profile shipped — climbing back to half its central value at large radius —
> and was used as the data generator for the test that makes the published NGC 6383 corona
> interpretation falsifiable, where **69.4% of the injected stars came from the spurious tail**.
> **Do not describe an oracle here before the test exists**; a claimed oracle is worse than an
> admitted gap, because it stops anyone from writing the real one.

## The four ways tests here have failed to bite

1. **Ratio-only oracles.** Asserting a ratio lets a prefactor move — the Jacobi prefactor could shift
   **14.5%** undetected.
2. **Self-widening tolerances.** `assert abs(x - truth) < N * posterior_std` passes automatically
   whenever the fit is uncertain, i.e. it tests nothing exactly when it matters. Use **absolute**
   tolerances tied to the injected truth.
3. **Reimplementing instead of exercising.** A prior test that rebuilds the prior block inline passed
   while the shipped code still contained `sigma = np.std(r)` — the exact defect the P01 referee
   flagged. Import and call the real function.
4. **Fixtures in a non-identifiable regime.** *A recovery test is only a test where the parameter is
   recoverable.* Three mutations survived absolute tolerances because the fixture data could not
   constrain the parameter; scaling the errors down 5× gave a 6× tighter posterior and killed them.
   Check the posterior is informative before trusting a recovery test.

## Before you claim a test works

Re-apply the bug and watch the suite go red. Every bug fixed in this repo has had its mutation
re-applied and the killing test recorded. If you cannot construct a mutation the test catches, the
test is decorative — say so in the docstring rather than leaving it to be discovered.

## Conventions

- Docstrings state **what the oracle is and why it is valid**, not what the test does.
- `-m "not slow"` is the default sweep; anything sampling-heavy is marked `slow`.
- PyTensor cache setup lives in `conftest.py` — do not re-do it per test.
- Sampler quality gates where a model is fitted: **R-hat < 1.01, ESS > 400, zero divergences.**

Full failure taxonomy with cases: `~/phd/methodology.md` PART K.
