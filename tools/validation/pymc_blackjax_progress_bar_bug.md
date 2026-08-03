# PyMC 6.1.0: `nuts_sampler="blackjax"` raises `TypeError` — a kwarg is consumed after it is used

**Status:** reproduced locally, not yet reported upstream.
**Affects:** `pymc` 6.1.0 with `blackjax` 1.6.2, `jax` 0.10.2.
**Impact for this project:** none — `nuts_sampler="numpyro"` is the backend adopted for
`tools/validation/ellipticity_bias.py`. Recorded because "it's incompatible" was the first
explanation reached for, and it is wrong.

## Symptom

```python
cfg = SamplingConfig(draws=1000, tune=800, chains=2, progressbar=False, nuts_sampler="blackjax")
eff_unbinned(radii, field_radius=70.0, priors=EFFPriors(), sampling=cfg)
```

```
TypeError: build_kernel.<locals>.kernel() got an unexpected keyword argument 'progress_bar'
```

Note `progressbar=False` was passed. Disabling the progress bar does not avoid it.

## It is not a version incompatibility — it is an ordering bug in PyMC

`pymc/sampling/jax.py`, PyMC 6.1.0:

| line | code | effect |
|---|---|---|
| 359 | `nuts_kwargs["progress_bar"] = progressbar` | unconditionally injects the key |
| 366 | `partial(_blackjax_inference_loop, ..., **nuts_kwargs)` | forwards it into `**adaptation_kwargs` |
| 253–259 | `blackjax.window_adaptation(algorithm=..., **adaptation_kwargs)` | **passes it through to the algorithm, which builds the kernel → raises here** |
| 277 | `progress_bar = adaptation_kwargs.pop("progress_bar", False)` | the pop that was meant to consume it — **runs after line 258** |

The `pop` on line 277 lives in `_blackjax_sample_raw`, *downstream* of the
`window_adaptation` call on line 253 inside `_blackjax_inference_loop`. So the key is still in
`adaptation_kwargs` when `window_adaptation` forwards its extra keyword arguments to
`blackjax.nuts`, and `blackjax.mcmc.nuts.build_kernel()` returns a kernel whose signature is

```
(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix, max_num_doublings=10)
```

with no `progress_bar`. Verified by `inspect.signature`.

**The fix is to pop before adapting, not after.** BlackJAX has not removed anything PyMC relies
on; `blackjax.progress_bar.gen_scan_fn` on line 280 exists and works. The two lines are simply in
the wrong order relative to each other.

## Reproduce

```bash
python -c "
import pymc, blackjax, inspect
from blackjax.mcmc import nuts
print(pymc.__version__, blackjax.__version__)
print(inspect.signature(nuts.build_kernel()))   # no progress_bar parameter
"
```

## Before reporting upstream

Check whether this is already fixed on `pymc` `main` and whether an issue exists; the surrounding
code changed recently enough that it may be known. If not, the report is small: the two line
numbers above and the kernel signature.

## The wider point, which is why this file exists

The three NUTS backends PyMC exposes — `"pymc"`, `"numpyro"`, `"blackjax"` — are **the same
algorithm**, Hamiltonian Monte Carlo with the No-U-Turn Sampler. `numpyro.infer.NUTS` subclasses
`HMC` and its docstring reads *"Hamiltonian Monte Carlo inference, using the No U-Turn Sampler
(NUTS)"*. What differs is the execution substrate: PyTensor-compiled C for `"pymc"`, JAX-traced and
JIT-compiled for the other two.

That is what licenses swapping them. Measured on one cell of the ellipticity grid
(γ = 2.5, q = 0.71, N = 60 000):

| backend | wall time | γ | a |
|---|---|---|---|
| pymc | 243.5 s | 2.498320 | 1.405783 |
| numpyro | 89.8 s | 2.498259 | 1.405335 |
| blackjax | — | `TypeError` above | |

**2.71× faster, |Δγ| = 0.00006** — about 70× below the sweep's own SEM of ~0.004. A speedup that
moved the answer would be worthless; agreement at this level is the evidence that it did not.
