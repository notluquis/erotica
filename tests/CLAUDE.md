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
- an **external implementation** — `king_profile` cross-validated against `ocelot`'s `King62`;
- a **monotonicity that cannot be faked** — a tighter σ-clip can never keep more sources, and its
  selection must be a *subset* of the looser one.

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
