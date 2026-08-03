# `nuts_sampler="blackjax"` raises `TypeError` — a blackjax 1.6 break plus a PyMC ordering bug

**Status:** already reported upstream as
[pymc-devs/pymc#8367](https://github.com/pymc-devs/pymc/issues/8367) (open, 2026-07-16), with a fix
in [PR #8373](https://github.com/pymc-devs/pymc/pull/8373) (open, +108/−4, no reviews yet).
**Affects:** `blackjax` >= 1.6 with any PyMC (verified upstream on 5.27.1 and 6.1.0; reproduced here
on 6.1.0 + blackjax 1.6.2 + jax 0.10.2, macOS arm64). Works with `blackjax` <= 1.5.
**Impact for this project:** none. `nuts_sampler="numpyro"` is the adopted backend.

## Symptom

```python
cfg = SamplingConfig(draws=1000, tune=800, chains=2, progressbar=False, nuts_sampler="blackjax")
eff_unbinned(radii, field_radius=70.0, priors=EFFPriors(), sampling=cfg)
```

```
TypeError: build_kernel.<locals>.kernel() got an unexpected keyword argument 'progress_bar'
```

`progressbar=False` was passed. Disabling the progress bar does not avoid it.

## Whose bug is it — both, and the split matters

**blackjax is the trigger.** 1.6 removed the `progress_bar` parameter from `window_adaptation`
(progress bars became a context manager, `with blackjax.progress_bar(): warmup.run(...)`) and
**dropped the `gen_scan_fn` helper**. Worse for diagnosis, `window_adaptation` now *forwards*
unknown `**extra_parameters` into the NUTS kernel instead of rejecting them — so a stale keyword
does not fail at the call site, it fails deep inside JAX tracing with a message naming a function
nobody called.

**PyMC is the fragility.** `pymc/sampling/jax.py` in 6.1.0:

| line | code | effect |
|---|---|---|
| 359 | `nuts_kwargs["progress_bar"] = progressbar` | injects the key **unconditionally** |
| 366 | `partial(_blackjax_inference_loop, ..., **nuts_kwargs)` | forwards it into `**adaptation_kwargs` |
| 253–259 | `blackjax.window_adaptation(algorithm=..., **adaptation_kwargs)` | **the key is still there → raises** |
| 277 | `adaptation_kwargs.pop("progress_bar", False)` | the pop meant to consume it — **after line 258** |

The `pop` is downstream of the `window_adaptation` call it was supposed to protect. So even had
blackjax kept accepting the kwarg, the ordering is wrong; blackjax 1.6 is what turned a latent
mistake into a hard failure.

**Neither PyTensor nor JAX is involved.** The JAX sampling path touches PyTensor only to compile the
logp; the fault is entirely in the PyMC↔blackjax boundary.

## Verified

```bash
python -c "
import blackjax, inspect
from blackjax.mcmc import nuts
print(inspect.signature(nuts.build_kernel()))
# (rng_key, state, logdensity_fn, step_size, inverse_mass_matrix, max_num_doublings=10)
#  -> no progress_bar parameter
"
```

`pymc/sampling/jax.py` on `main` (fetched 2026-08-02) still has the pop after `window_adaptation`,
so **upgrading to 6.2.0 does not fix it** — only PR #8373 does.

## What PR #8373 does, and where we could contribute

Two changes, both in `pymc/sampling/jax.py`, plus tests:

1. Pop `progress_bar` from `adaptation_kwargs` **before** calling `window_adaptation`.
2. Dispatch on `hasattr(blackjax.progress_bar, "gen_scan_fn")` to support both the pre-1.6
   module API and the 1.6+ context-manager API. This is the part the local trace missed: line 280,
   `blackjax.progress_bar.gen_scan_fn(draws, progress_bar)`, is a **second** break behind the
   first — it no longer exists in 1.6.

### Review of #8373 — two defects found, both verified locally against blackjax 1.6.2

**The `hasattr` dispatch itself is correct.** Confirmed by inspection on the installed 1.6.2:
`blackjax.progress_bar` is a *function*, not a module; `hasattr(pb, "gen_scan_fn")` is `False`; its
signature is `(label='BlackJAX', print_rate=None, output_file=None)`, so `label="NUTS"` is valid.

**Defect 1 — the whole scan block is duplicated after a `return`, so eleven lines are
unreachable.** The diff adds the `keys = ...` / `if hasattr(...)` / `return samples, stats` block
**twice**, the second copy sitting below the first `return`:

```python
    ...
    return samples, stats      # <- first return

    keys = jax.random.split(seed, draws)     # <- dead code from here down
    if hasattr(blackjax.progress_bar, "gen_scan_fn"):
        ...
    return samples, stats      # <- never reached
```

This is a copy-paste slip, almost certainly from a rebase. **It also explains the codecov number**:
patch coverage of 55.6% is what you get when roughly half the added lines cannot execute. Deleting
the second copy fixes the review comment and the coverage figure at once.

**Defect 2 — the `progress_bar=True` path needs an optional extra the PR never mentions.** On
blackjax >= 1.6 the context manager is powered by `jaxtap`, shipped as `blackjax[progress]`.
Without it:

```
ImportError: blackjax.progress_bar requires the 'progress' optional extra.
Install it with:  pip install 'blackjax[progress]'
```

Reproduced here on a plain `blackjax 1.6.2` install. So the PR's own
`test_sample_blackjax_nuts_progressbar_true` will **fail on any CI image that installs blackjax
without the extra**, and a user passing `progressbar=True` trades a `TypeError` for an
`ImportError`. The fix wants either a `try/except ImportError` falling back to a bare `jax.lax.scan`
with a warning, or `blackjax[progress]` added to PyMC's JAX test dependencies — the former is
better, since a progress bar should never be able to abort a sampling run.

### Both defects verified by patching and running, not by reading

The PR's fix was applied to the installed `pymc/sampling/jax.py` **with the duplicate block removed
and an `ImportError` guard added**, then the issue's own reproduction was run:

```
progressbar=False   ->  OK   posterior shape (1, 50), mean 0.1311
progressbar=True    ->  OK   warns "blackjax progress bar needs the 'progress' extra
                             (pip install 'blackjax[progress]'); sampling without it"
```

So the PR's approach is right and its two defects are both real and both fixable in a few lines.
The guard matters because without it `progressbar=True` trades a `TypeError` for an `ImportError`:
a progress bar should never be able to abort a sampling run. Suggested shape:

```python
    elif progress_bar:
        try:
            with blackjax.progress_bar(label="NUTS"):
                _, (samples, stats) = jax.lax.scan(_one_step, last_state, (jnp.arange(draws), keys))
        except ImportError:
            warnings.warn(
                "blackjax progress bar needs the 'progress' extra "
                "(pip install 'blackjax[progress]'); sampling without it.",
                UserWarning, stacklevel=2)
            _, (samples, stats) = jax.lax.scan(_one_step, last_state, (jnp.arange(draws), keys))
```

The installed PyMC was restored to its original state afterwards; this repo carries no patch.

Open contributions, cheap and real:

- **Report the two defects above on the PR.** Both are concrete, both are one-line-ish, and neither
  requires disagreeing with the approach.
- **Confirm the reproduction on a configuration the issue does not list.** #8367 cites blackjax 1.6;
  this project reproduces on **1.6.2, pymc 6.1.0, jax 0.10.2, macOS arm64**.
- **Review it.** No reviews after eleven days.

## The upstream history, because it explains why nobody caught the 1.6 break

The blackjax progress-bar path has a long tail, and the older threads are **not** about this bug:

| | what | outcome |
|---|---|---|
| pymc [#7049](https://github.com/pymc-devs/pymc/issues/7049) (2023-12) | *"Is blackjax progressbar still broken?"* — progress bars under `pmap` | Closed in a week: *"Closing this as it is blackjax related"*. zaxtax then pointed at blackjax #655 as the coming fix. |
| blackjax [#655](https://github.com/blackjax-devs/blackjax/pull/655) (2024-04) | migrate the progress bar from `fastprogress` to `tqdm`, +40/−27 | **Closed unmerged 2026-03-26**, two years stale. junpenglao: *"The fastprogress → tqdm migration motivation has faded (the related bug was resolved elsewhere)"* — and earlier, *"I forgot the reason why we were doing this beside the pmap bug (which is now fixed)"*. |
| blackjax [#712](https://github.com/blackjax-devs/blackjax/pull/712) (2024-08, **merged**) | *"Enable progress bar under pmap"*, by andrewdipper | This is the *"resolved elsewhere"*. It touched `window_adaptation.py`, `progress_bar.py` and `util.py`, and changed the carry of `progress_bar_scan` — which the author flagged as a break for external users. |
| pymc [#8367](https://github.com/pymc-devs/pymc/issues/8367) (2026-07) | **this bug** | Open. Different cause: blackjax 1.6 removing the `window_adaptation` kwarg and `gen_scan_fn` outright. |

So the 2023–2024 thread was about `pmap`, was fixed on the blackjax side, and is unrelated. The
1.6 API removal is a fresh break that landed after everyone had stopped looking at this code
path — and PyMC's latent ordering mistake is what made it fail as a cryptic `TypeError` rather
than a clean "unexpected keyword".

## Correction to an earlier version of this file

An earlier draft cited [#7049](https://github.com/pymc-devs/pymc/issues/7049) as the upstream
issue. **That is wrong** — #7049 is from 2023, was closed in a week, and concerns progress bars
under `pmap`, a different problem. It came from a web search and was not checked. The real issue is
#8367, found by querying the repository directly. Search results are a lead, not a citation.

## Why the swap to numpyro was safe

The three NUTS backends PyMC exposes — `"pymc"`, `"numpyro"`, `"blackjax"` — are **the same
algorithm**: Hamiltonian Monte Carlo with the No-U-Turn Sampler. `numpyro.infer.NUTS` subclasses
`HMC`, and its docstring reads *"Hamiltonian Monte Carlo inference, using the No U-Turn Sampler
(NUTS)"*. What differs is the substrate: PyTensor-compiled C for `"pymc"`, JAX-traced and JIT-
compiled for the other two.

Measured on one cell of the ellipticity grid (γ = 2.5, q = 0.71, N = 60 000):

| backend | wall time | γ | a |
|---|---|---|---|
| pymc | 243.5 s | 2.498320 | 1.405783 |
| numpyro | 89.8 s | 2.498259 | 1.405335 |
| blackjax | — | `TypeError` above | |

**2.71× faster, |Δγ| = 0.00006** — about 70× below the sweep's own SEM of ~0.004. A speedup that
moved the answer would be worthless; agreement at this level is the evidence that it did not.
