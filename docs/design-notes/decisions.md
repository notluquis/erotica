# Decision log — why the code is the way it is

_A running record of **non-obvious choices**: bugs found and how they were fixed, defaults that
encode a scientific judgement, and things deliberately left alone. Written for the person who later
asks "why is this like this?" — including a referee, a co-author, or us in six months._

**Conventions.** Each entry states the **symptom**, the **cause**, the **fix**, the **oracle** the
test checks against, and any **number that moved**. Entries are append-only; if a decision is
reversed, add a new entry rather than editing the old one.

---

## 2026-07-27 — `R_0` was inconsistent inside a single chain

**Symptom.** `analysis/dynamics.py` carried **two different values of the solar galactocentric
distance**: `8.125 kpc` in `calculate_galactic_mass`, and `8.3 kpc` in
`calculate_galactocentric_distance` and `calculate_hill_radius`. All three feed the *same*
Hill-radius computation, so a single call mixed two Galactic geometries.

**Cause.** Independently written defaults, never cross-checked; no test compared them.

**Fix.** One module constant, `dynamics.SOLAR_RADIUS` (`erotica/analysis/dynamics.py:25`), used as
the default in all three. The **value was not changed** — 8.125 kpc, the value already in
`calculate_galactic_mass` — so this commit is a pure de-duplication, not a re-calibration.

```{warning}
**The adopted 8.125 kpc does not correspond to any verified published measurement** `[I]`.
Its provenance in this codebase is unknown — it predates the current history and no source is cited.
The best abstract-verified modern value is **R_0 = 8178 ± 13(stat) ± 22(sys) pc**, i.e.
**8.178 kpc**, from GRAVITY Collaboration 2019, *"A geometric distance measurement to the Galactic
center black hole with 0.3% uncertainty"*, A&A 625, L10, `2019A&A...625L..10G`,
doi:10.1051/0004-6361/201935656 `[S — quoted from the abstract]`.

Adopting 8.178 kpc is a **science decision, not a bug fix**: it moves every galactocentric and
Hill-radius number by a further ~0.6%. **Deferred to an explicit call by the author.**

⚠️ Two citations that appear in earlier drafts of this note were wrong and are recorded here so
they are not repeated: `2018A&A...615L..15G` is the *gravitational-redshift* paper (its abstract
reports f = 0.90 ± 0.09, not R_0), and `2021A&A...647A..59G` is *"Improved GRAVITY astrometric
accuracy from modeling optical aberrations"* — **not** an R_0 determination, though it does state it
resolves earlier systematic discrepancies in R_0.
```

**Oracle.** Not a golden number — the law of cosines has closed forms the test checks exactly:
at `l=0, b=0` it collapses to `R_gc = |R_0 − d|`; at `l=180, b=0` to `R_gc = R_0 + d`.
See `tests/test_dynamics.py::test_galactocentric_distance_matches_closed_form_toward_centre`.

**Numbers that moved.** Any quantity computed through the `8.3 kpc` path shifts by the R_0 change
(≈2%): galactocentric distance, and the Hill radius through it. `calculate_galactic_mass` with the
default `legacy_power_law` model does **not** use `solar_radius` at all, so enclosed-mass numbers
are unaffected unless `model='solar_scaled'` was used. **A re-derivation against
`data/test/NGC6383/` is required before these appear in print.**

---

## 2026-07-27 — `solar_radius` was silently dropped on one branch

**Symptom.** `calculate_hill_radius(center=..., solar_radius=X)` ignored `X`. The sibling branch,
`calculate_hill_radius(distance, l, b, solar_radius=X)`, honoured it. Same call, same keyword, two
behaviours depending on which optional argument the caller used.

**Cause.** The `center` branch called `calculate_galactocentric_distance(...)` without forwarding
`solar_radius`, so it silently fell back to the default.

**Fix.** Forward the keyword on both branches.

**Oracle.** `test_solar_radius_override_is_honoured_on_every_path` passes a deliberately extreme
`R_0 = 4 kpc` and asserts the result *changes* on **both** routes. A test that only checked the
result was finite would have passed throughout the bug's lifetime.

---

## 2026-07-27 — the default NUTS backend was broken, and no test could have caught it

**Symptom.** Every Bayesian entry point failed on a real call:

```
TypeError: build_kernel.<locals>.kernel() got an unexpected keyword argument 'progress_bar'
```

`SamplingConfig.nuts_sampler` defaulted to `"blackjax"`, which is incompatible with the installed
blackjax 1.6.2 / pymc 6.1.0. **So `pip install -e ".[bayes]"` produced a package whose every
Bayesian model raised on first use.** Three entry points carried the default: `inference.py:24`,
`_isochrone.py:950`, `analyzer.py:353`.

**Why it went unnoticed — the deeper finding.** *No test in the suite had ever sampled a PyMC model.*
`tests/test_inference.py` monkeypatched the sampler away; `tests/test_structure.py` fed a `_FakeVar`
trace and never called `RDP_bayesian`; and CI installed `.[dev]`, not `[bayes]`. The Bayesian paths
that produce every published number were entirely unexercised. A JOSS reviewer installs and runs.

**Fix.** Default to `"pymc"` — PyMC's built-in NUTS, always present when the extra is installed.
`blackjax` and `numpyro` remain available opt-in for speed. Verified working: recovers
`mu = 0.8948 ± 0.0027` against an injected truth of 0.90.

**Oracle.** Injected truth, not a golden number — synthetic parallaxes drawn from a known
`(mu, sigma)` must be recovered, plus the convergence floor from `~/phd/methodology.md` PART A
(Vehtari+2021): **R-hat < 1.01, bulk-ESS > 400, zero divergences**. See
`tests/test_inference.py::test_fit_parallax_model_recovers_injected_truth`.

**Also fixed in the same pass:** CI gained a `test-bayes` job that installs `.[dev,bayes]` and
**asserts the extra actually imported**, so these tests execute rather than silently skipping.
It is a separate job because pymc>=6 requires Python >=3.12 while the matrix still covers 3.11.

```{note}
`arviz >= 1` returns a **display-formatted** summary frame — `r_hat` and `ess_bulk` arrive as
strings, so `summary["r_hat"].max() < 1.01` raises `TypeError: '<' not supported between instances
of 'str' and 'float'`. Coerce with `pd.to_numeric(..., errors="coerce")` first.
```

---

## 2026-07-27 — the reproducibility record recorded almost nothing, and nothing called it

**Symptom.** `analysis/provenance.py:146` was:

```python
def build_metadata(**kwargs):
    return {"created_at": ..., "erotica_version": ..., **kwargs}
```

A timestamp and a version string. **No git commit, no seed, no dependency versions, no checksum of
the input data.** Two runs of the same version against different catalogues, on different NumPy
releases, from a dirty working tree, produce byte-identical provenance.

**The worse half.** `grep` found **zero callers**. Nothing in the package ever built a metadata
record. Dead provenance code is worse than no provenance code, because the reproducibility claim
gets made anyway and there is a function name to point at.

**Fix.** `build_metadata` now records, and `store_trace_results` now *calls* it — every saved trace
gets a `*_provenance_<index>.json` sidecar next to its NetCDF:

| Field | Why it is there |
|---|---|
| `git.commit`, `git.dirty`, `git.branch` | A commit hash is misleading without `dirty` — the normal state during analysis is uncommitted edits. `None` outside a repo (a PyPI wheel), never an exception. |
| `inputs[].blake2b`, `inputs[].bytes` | Identifies input **content**. A catalogue that is renamed, re-downloaded or re-sorted into another path but is byte-identical is the same input; one flipped byte is not. Streamed, so a multi-GB file costs kilobytes of memory. |
| `seeds` | Kept even when `None` — a run with no fixed seed is *not* reproducible and the record has to say so rather than omit the field. |
| `dependencies` | Read from installed distribution metadata, **not** by importing each package: importing PyMC costs seconds and is a side effect a provenance call has no business causing. An absent optional extra is itself provenance (that run produced no Bayesian numbers). |
| `python`, `platform` | Cheap, and the first thing asked when a number fails to reproduce. |

**The invariant.** A provenance record that raises at write time destroys the result it was meant to
describe. So `build_metadata` calls `json.dumps` on itself before returning — failing loudly at
build time rather than after a six-hour sampling run — and `_jsonable` coerces the values that
arrive naturally from callers and would otherwise raise: NumPy scalars and arrays, `Path`,
dataclasses such as `SamplingConfig`, and sets. An unrecognised object is downgraded to `repr`,
never dropped: losing a field silently is the failure mode, losing its type is not.

**Oracles** (`tests/test_provenance.py`, 28 tests). Each is independent of the code under test:
* git → a `git` subprocess the test runs itself, plus a throwaway repository the test **builds,
  commits, dirties, restores, and dirties again with an untracked file**. A test asserting only that
  the key exists would pass against a hardcoded `False`.
* checksums → the published BLAKE2b-512 vector for the empty string
  (`786a02f7…e9be2ce`), plus digests the test computes with `hashlib` on its own; content-vs-path
  equality; a 40 MB multi-chunk file.
* dependencies → `module.__version__`, which is set independently of distribution metadata.
* `build_metadata` → `json.loads(json.dumps(m)) == m`, fed hostile values on purpose.
* `store_trace_results` → the sidecar exists, names the same `trace_index` as the `.nc` beside it,
  and carries the caller's checksum. This one is a **regression guard against the dead-code state**.
* `posterior_mode` → a lognormal, whose mode `exp(−1) = 0.368`, median `1.0` and mean `1.649` are
  analytically distinct; the estimator lands at 0.375. The easy way to fake a mode estimator is to
  return the mean or the median, and both fail this by a factor of ~3.
* `store → load` → round-trip identity, including the recovered draws, not merely the summary.

```{warning}
**The wiring reaches the legacy notebooks, not the current analysis path.** `store_trace_results` is
called from three committed notebooks under `data/test/NGC6383/`; **no module or script in the
current pipeline calls it**. So published NGC 6383 numbers still have no provenance record attached.
Closing that is a separate step: the call belongs wherever the paper's fits are driven from
(`review_repo/regen_king*.py`, `convergence_audit*.py`, `isochrone_nuts_refit.py`), passing the
input catalogue path and the sampler seed. Recorded rather than quietly left, so the entry above is
not read as "reproducibility: solved".
```

**Numbers that moved.** None — this is new information recorded alongside results, not a change to
any computation. Coverage of `provenance.py` **27% → 95%**; suite **290 → 318**.

Also renamed the leftover `_PUMPS_VERSION` import (from the COSMIC → PUMPS → EROTICA renames).

---

## 2026-07-27 — `RuntimeWarning` was globally ignored, hiding a real bug

**Symptom.** `pytest.ini` carried a blanket `ignore::RuntimeWarning`. In a numerics package that is
the category that says *divide by zero*, *invalid value in sqrt*, *overflow* — i.e. exactly how a
silent `NaN` reaches a published number.

**What the audit found.** Re-running the suite under `-W error::RuntimeWarning` produced only two
failures. One was third-party and harmless: PyTensor's graph rewriter divides by zero while constant-folding
a graph that is then discarded. The other was **ours**.

**The bug.** `analysis/_isochrone.py::_fit_error_model` guarded its constant-error fallback with:

```python
def _fit(mag_ok, e_ok, fallback):
    if mag_ok.sum() < 3:          # intended: "fewer than 3 valid points"
```

`mag_ok` is not a boolean mask — it is `obs_mag[valid_m]`, the **filtered magnitude array**. So
`.sum()` added up magnitudes. Two stars at G = 14 and 15 sum to 29, comfortably ≥ 3, so the fallback
**never fired for any non-empty input**. Two points were fitted with a degree-2 polynomial, the
Vandermonde matrix was rank-deficient, NumPy raised `RankWarning` — a `RuntimeWarning` subclass in
NumPy 2 — and `pytest.ini` swallowed it. The docstring meanwhile asserted the opposite, claiming
`Polynomial.fit` was chosen precisely to avoid `RankWarning`.

The test that covered this line asserted `np.isfinite(f_m(...)[0])`. Garbage from an
under-determined fit is finite, so the test passed for the whole life of the bug. This is the
house rule in the standing section, violated: *a test must be able to fail for the reason it was
written*.

**Fix.** Count **distinct abscissae**, `np.unique(mag_ok).size < 3` — rank deficiency is about
distinct magnitudes, not the point count, so six stars all at G = 15 must also fall back.

**Numbers that moved.** **None.** The old and new guards were compared across
`n = 0, 1, 2, 3, 10, 60, 321` points: they disagree only at `n = 1` and `n = 2`, and agree
everywhere else including the empty case. A real cluster fit never enters that regime, so no
published NGC 6383 quantity changes. The bug was latent, not active — but it was one degenerate
input away from producing a silently arbitrary error model.

**Fix to the filter.** `error::RuntimeWarning`, with three narrow `ignore` lines scoped to
`pytensor.tensor.rewriting.math` by message. Nothing of ours can be hidden again.

**Oracles added** (`tests/test_isochrone.py::TestFitErrorModel`): the fallback must return the
**exact median** (`0.015` for `e_m = [0.01, 0.02]`) and be constant in magnitude; six repeated
magnitudes must also fall back; a quadratic injected in log₁₀ space must be recovered to `rtol=1e-6`;
and zero/NaN/negative errors must be dropped *before* the branch is chosen.

---

## 2026-07-27 — the 2σ parallax clip is a magnitude-dependent selection function

**Status: measured, not yet changed.** The measurement is the deliverable; the fix is a science
decision that changes a published number and is sequenced with the P01 delta report.

**Symptom.** `analysis/_clipping.sigma_clip_parallax` clips on the **raw** parallax column at a
fixed multiple of the *sample* scatter. Gaia's per-star parallax uncertainty is strongly
magnitude-dependent, so a faint star with a large `e_Plx` is scattered further from the cluster
centroid than a bright one **even when both are members**. The cut therefore removes faint stars
preferentially. That makes it a selection function, not only an outlier test.

**The oracle — by construction, no golden numbers.** `tools/validation/parallax_clip_selection_function.py`
builds synthetic clusters in which **every star is a true member**: one common true parallax, no
contamination, no intrinsic depth. Per-star uncertainties are the **real** `e_Plx` values from the
published member table (`data/test/NGC6383/comments_paper/cds_final/ngc6383_members.ecsv`, 321
stars), so the magnitude-dependence of the errors is empirical rather than modelled. Any rejection
is then, unambiguously, a **false** rejection, and the retention rate per `Gmag` quartile *is* the
induced selection function.

**Numbers** (400 realizations, σ = 2, seed 20260727; `tools/validation/parallax_clip_selection_function.json`):

| | overall | Q1 bright | Q2 | Q3 | Q4 faint |
|---|---|---|---|---|---|
| **raw parallax (current behaviour)** | **67.5%** | 99.1% | 85.6% | 59.1% | **26.7%** |
| clip on normalized residual `(plx − centre)/err_i` | **95.7%** | 95.6% | 95.7% | 95.8% | 95.8% |

Median `e_Plx` per quartile: **0.027, 0.065, 0.123, 0.293 mas** — an 11× spread, which is the whole
mechanism. Bright→faint retention gradient: **raw +72.4 pp, normalized −0.2 pp**. The normalized
clip sits at ~95.7% everywhere, which is simply what a 2σ cut *should* retain.

So the current cut discards **about a third of genuine members**, and **roughly three-quarters of
the faintest quartile**.

```{admonition} What this experiment does *not* establish
:class: caution
**The 95.7% is close to tautological.** The synthetic draws each star's parallax with the same
per-star `err_i` that the normalized residual then divides by, so `z` is exactly `N(0,1)` by
construction and a 2σ cut retains ~95.4% whatever the magnitude distribution. The flat row is a
consistency check on the arithmetic, **not evidence that the normalized clip is better**.

**The experiment is one-sided: there is no contamination in it.** With every star a true member it
measures the false-*rejection* rate only. The normalized clip is necessarily more permissive, and
permissiveness has a cost this design cannot see: a field star at 0.6 mas from the centroid with a
large `e_Plx` gets `z ≈ 2` and *survives* normalisation, where the raw clip removes it. Retention
and contamination move together, and only one of them was measured.

**What does stand** is the raw clip's **+72.4 pp bright→faint gradient**, which needs no comparison
to interpret: it is a statement about the current cut alone, against an oracle that is true by
construction.

Before the delta report recommends any switch, this needs a contaminated arm — inject field stars
from the real 70′ cone at a distinct parallax and report retention of members **and** contaminants
for both cuts, i.e. an ROC rather than a single operating point.
```

**Why the alternative is *a priori* attractive.** On the normalized residual, a large uncertainty
buys *tolerance* rather than a rejection: the star is an outlier only if it is discrepant relative
to **its own** error. That is the same principle as the error-aware `f_i` in `core/_error_aware.py`.
It is the motivation for testing it, not a result of this test.

```{admonition} Implication for P01 — unproven, not disproven
:class: warning
P01 attributes a faint-quartile luminosity-function / KS signal to **Gaia incompleteness**. Q4 is
exactly the quartile this clip mutilates, so the pipeline supplies a **competing explanation of the
same signal** that the paper does not currently address. That is a referee-grade objection.

**This does not show the P01 result is wrong.** Both effects act in the same direction and can
coexist; nothing here quantifies their relative size on the *real* (contaminated) field. The
discriminating test is a re-run of the faint-quartile LF/KS on a normalized-residual-clipped
sample: if the signal survives, the incompleteness attribution is strengthened rather than
weakened. Estimated ~1 h. **Not yet run** — recorded here so the claim is not made in either
direction without it.
```

**Tests.** `tests/test_clipping.py` pins the behaviour with two fast characterisation tests
(`test_raw_clip_rejects_large_error_stars_preferentially`, asserting the ≥15 pp bias exists, and
`test_normalized_residual_clip_is_precision_blind`, asserting the alternative is within 5 pp). They
are characterisation, not aspiration: they make any future change to the clip **visible** rather
than silent. The full measurement stays in `tools/validation/` because it needs the real catalogue
and 400 realizations.

**Not changed yet, deliberately.** Switching the clip changes the published NGC 6383 membership
list. Per the approved plan the sequence is *fix → re-derive → report deltas*, and the delta report
is the artefact the author needs before deciding what, if anything, to tell a referee mid-review.

---

## 2026-07-27 — `sigma_clip_parallax` has flag-dependent return arity

**Status: documented, not changed.**

`sigma_clip_parallax` returns 2, 3, 4 or 5 values depending on `in_place` and `return_mask`. This is
an API wart — a caller cannot unpack the result without knowing both flags.

**Why not fixed now.** It is public API with existing callers, including the paper's figure scripts.
Changing the signature is a breaking change that belongs in a version bump, not in a test pass.

**What was done instead.** `test_return_arity_is_flag_dependent` pins all four shapes, so a refactor
cannot alter them silently, and the wart is recorded here for the next major version.

---

## Standing decisions (not tied to one date)

### The pseudo-probability `p̃` is a score, not a probability
`p̃ = probabilities_ × probability_times` (`core/clustering.py`) has no prior, no likelihood and no
normalisation. It is deliberately labelled an *operational ranking statistic* rather than a
posterior — this was a direct referee lesson from P01 (see `~/phd/methodology.md` PART D:
*"Label operational proxies as proxies"*). `calibration.py` exists to measure, empirically, whether
it behaves like a frequency. **Do not describe `p̃` as a membership probability in any paper.**

### Tests assert on behaviour, not execution
The house style is that a test must be able to **fail for the reason it was written**. Examples
worth copying: the σ-clip monotonicity test (a tighter cut can never keep more sources, and its
selection must be a *subset* of the looser one), and the analytic law-of-cosines oracle above. A
test that only asserts a result is finite is not a test — the `solar_radius` bug survived years of
"it runs" checking.

### `--no-verify` is the standing commit workflow
The `nbstripout` pre-commit hook chokes on the ~3.1 GB committed `data/test/NGC6383/` tree and can
revert unstaged work into a `.cache/pre-commit/` patch. Until the hook is fixed, commits use
`git commit --no-verify`. **Consequence: no hook runs on any commit, including nbstripout itself.**
Recorded so nobody assumes hooks are protecting them.

---

## Known-and-accepted, with the reason

| Thing | Why it is still like that |
|---|---|
| PyMC is an optional extra | Keeps the default install light; **but a default `pip install erotica` therefore produces only frequentist and heuristic numbers.** Anything Bayesian requires `[bayes]`. |
| `data/test/NGC6383/` (~3.1 GB) is committed | It is the paper's reproducibility artefact, tagged `ngc6383-aanda-resubmission`. It is also the reason pre-commit is broken. |
| ~30 files hardcode `/Users/notluquis/erotica/...` | Paper figure-regeneration scripts. They were rewritten during the 2026-07-21 directory move; if the directory moves again they must be rewritten in the same pass. |
