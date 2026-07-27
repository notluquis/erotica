> **Status:** filed as [pymc-devs/pytensor#2308](https://github.com/pymc-devs/pytensor/issues/2308),
> fix submitted as [PR #2309](https://github.com/pymc-devs/pytensor/pull/2309).
> Maintainer response (ricardoV94): *"yeah I hate tests with scale=1, loc=0 precisely because of that."*
>
> Once released, drop the `HalfStudentT(nu=1)` workaround in `analysis/structure.py` —
> `tests/test_structure.py::test_half_cauchy_prior_is_built_without_the_pymc_halfcauchy_bug`
> fails when the upstream fix lands, which is the signal.

# BUG: numba backend samples `Cauchy`/`HalfCauchy` with the wrong loc and scale

**Repo:** `pymc-devs/pytensor` · **Version:** 3.1.3 (and current `main`) · **Backend:** numba (the default linker)
**Introduced:** Aesara commit `5ea74e973` (2022-03-03); rewritten and preserved in `14da898c6` (2024-05-10) — present for ~4.4 years.

## Description

`pytensor/link/numba/dispatch/random.py` implements `CauchyRV` as

```python
@numba_core_rv_funcify.register(ptr.CauchyRV)
def numba_core_CauchyRV(op, node):
    @numba_basic.numba_njit
    def random(rng, loc, scale):
        return (loc + rng.standard_cauchy()) / scale

    return random
```

For a location–scale family the draw must be `loc + scale * z`. Written as
`(loc + z) / scale`, the sampled variable instead has

* **location** `loc / scale` instead of `loc`
* **scale** `1 / scale` instead of `scale`

The reference implementations are correct — `CauchyRV.rng_fn_scipy` calls
`stats.cauchy.rvs(loc=loc, scale=scale, ...)`, and the JAX backend routes `CauchyRV`
through the generic `jax_sample_fn_loc_scale`. **Only the numba path is wrong, and numba is the
default linker**, so the bug is hit by default settings.

`logp` is unaffected, so this does not bias NUTS posteriors. It does corrupt anything that *draws*:
`pymc.draw`, `pm.sample_prior_predictive`, and `pm.sample_posterior_predictive`. A prior-predictive
check on a `HalfCauchy` prior — the single most common use of this distribution — is therefore
silently wrong, which is how we found it.

`pm.HalfCauchy` is affected transitively: `HalfCauchyRV.rv_op` builds `pt.abs(cauchy(loc=0, scale=beta))`.

## Reproduce

```python
import numpy as np, pytensor, scipy.stats as st
from pytensor.tensor.random.basic import cauchy
from pytensor.compile.mode import Mode

def iqr(x): return float(np.subtract(*np.percentile(np.asarray(x), [75, 25])))

g = cauchy(3.0, 5.0, size=200_000)
ref = st.cauchy(3.0, 5.0)
no_opt = Mode(linker="py", optimizer=None)

print("numba (default):", np.median(g.eval()), iqr(g.eval()))
print("py linker      :", *[f(pytensor.function([], g, mode=no_opt)()) for f in (np.median, iqr)])
print("scipy          :", ref.median(), ref.ppf(.75) - ref.ppf(.25))
```

```
numba (default): 0.5998   0.4002      <-- loc/scale = 3/5,  1/scale = 1/5
py linker      : 2.9871  10.0272
scipy          : 3.0     10.0
```

Measured across parameter values, the implied scale is exactly `1/scale` and the implied location
exactly `loc/scale`:

| call | median | IQR | implied scale | expected |
|---|---|---|---|---|
| `cauchy(0, 0.5)` | −0.001 | 3.9997 | 2.0000 | 0.5 |
| `cauchy(0, 2)` | −0.001 | 0.9970 | 0.4985 | 2.0 |
| `cauchy(0, 5)` | 0.001 | 0.4002 | 0.2001 | 5.0 |
| `cauchy(0, 10)` | 0.000 | 0.2002 | 0.1001 | 10.0 |
| `cauchy(3, 5)` | 0.600 | 0.4008 | 0.2004 | 5.0 |
| `cauchy(−7, 2)` | −3.499 | 1.0022 | 0.5011 | 2.0 |

Among ten continuous distributions checked the same way (`normal`, `gamma`, `exponential`,
`laplace`, `logistic`, `t`, `vonmises`, `pareto`, `weibull`), **only `cauchy` is affected**.

## Why it has survived 4.4 years

The expression has been written **twice**, by two different authors, and survived a deliberate
rewrite of that exact line:

```
5ea74e973  2022-03-03  Brandon T. Willard  Fix the Numba implementation of CauchyRV
             -> introduced in Aesara as a string template:
                f"    return ({loc} + np.random.standard_cauchy()) / {scale}"

14da898c6  2024-05-10  Ricardo Vieira      Add support for RandomVariable with Generators
                                           in Numba backend and drop support for RandomState
             -> rewrote it from a template into a real njit function, carrying the
                expression forward verbatim:
                    def random(rng, loc, scale):
                        return (loc + rng.standard_cauchy()) / scale
```

(Nine further commits touched the surrounding region — the Aesara→PyTensor rename, style passes,
the Generator contract fix — without altering the expression. Note that `git log -S standard_cauchy`
reports only the 2022 commit, because the 2024 port kept the substring count unchanged; `git log -L`
on the function is what shows both.)

**The 2022 commit also set the test parameters, and they are the one case that cannot fail** — which is why neither author had a signal. The
single `cauchy` entry in `tests/link/numba/test_random.py` is

```python
(
    ptr.cauchy,
    [
        (pt.dvector(), np.array([1.0, 2.0], dtype=np.float64)),   # loc
        (pt.dscalar(), np.array(1.0, dtype=np.float64)),          # scale  <-- 1.0
    ],
    (2,), "cauchy", lambda *args: args,
),
```

validated with a Cramér-von Mises goodness-of-fit test requiring `pvalue > 0.1`. That is a good
test. But **at `scale = 1` the buggy expression is algebraically identical to the correct one**:
`(loc + z) / 1 == loc + 1 * z`. The GOF check therefore passes comfortably, and both the scale error
and the location error vanish simultaneously:

| loc | scale | median | IQR | scipy median | scipy IQR | CvM p | |
|---|---|---|---|---|---|---|---|
| 1.0 | **1.0** | 0.992 | 2.005 | 1.000 | 2.000 | 9.2e-01 | passes |
| 2.0 | **1.0** | 2.002 | 1.997 | 2.000 | 2.000 | 9.9e-01 | passes |
| 1.0 | 2.0 | 0.506 | 1.007 | 1.000 | 4.000 | 1.0e-08 | fails |
| 3.0 | 5.0 | 0.599 | 0.399 | 3.000 | 10.000 | 3.4e-08 | fails |
| 0.0 | 0.5 | −0.023 | 4.008 | 0.000 | 1.000 | 6.5e-09 | fails |

Adding any `scale != 1` to the existing parametrisation is enough to catch it.

## Fix

```diff
 @numba_core_rv_funcify.register(ptr.CauchyRV)
 def numba_core_CauchyRV(op, node):
     @numba_basic.numba_njit
     def random(rng, loc, scale):
-        return (loc + rng.standard_cauchy()) / scale
+        return loc + scale * rng.standard_cauchy()

     return random
```

Verified numerically: `loc + scale * z` reproduces `scipy.stats.cauchy(loc, scale)` in median and
IQR at `(0, 5)`, `(3, 5)` and `(−7, 2)`.

## Suggested regression test

The existing GOF machinery is fine; it just needs a `scale != 1` case. If you prefer an explicit
check, note that a scale-sensitive statistic is required — the median alone still passes for
`loc = 0`, since `0/scale == 0`:

```python
@pytest.mark.parametrize("loc, scale", [(0.0, 5.0), (3.0, 5.0), (-7.0, 2.0)])
def test_cauchy_numba_matches_scipy_loc_scale(loc, scale):
    draws = pytensor.function([], cauchy(loc, scale, size=200_000))()
    ref = stats.cauchy(loc, scale)
    assert np.median(draws) == pytest.approx(ref.median(), abs=0.1)
    iqr = np.subtract(*np.percentile(draws, [75, 25]))
    assert iqr == pytest.approx(ref.ppf(0.75) - ref.ppf(0.25), rel=0.05)
```

Since Cauchy has no finite mean or variance, IQR (or another quantile spread) is the right statistic;
a mean/std comparison would be too noisy to be a reliable test.
