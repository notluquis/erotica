# Fitting a circular profile to an elliptical cluster

Every radial profile in this package assumes circular symmetry. **Circular symmetry is the
exception.** Tarricq et al. (2022) measure a median axis ratio `b/a = 0.71` for the core component
across 233 open clusters, with 92.9% below 0.9 and a 10th percentile of 0.42 — so the assumption is
violated for almost every cluster, and the size of the resulting bias was unquantified.

It matters for a specific claim. The A5 recoverability work asks whether the EFF slope `γ` piles up
near 2, the value at which EFF and an untruncated King coincide. **If ellipticity pushed `γ` toward
2, a pile-up there could be manufactured by an unmodelled axis ratio rather than by physics.** That
had to be measured before the claim could be made in either direction.

Regenerate everything here with `python tools/validation/ellipticity_bias.py --realizations 6`.

## The design

Sample an **elliptical** EFF surface density with semi-axes `a` and `b = qa`, then fit the
**circular** model to the resulting radii. Grid: `γ ∈ {2.0, 2.5, 3.0, 4.0}` × `q ∈ {1.0, 0.71, 0.50,
0.30}`, six realisations per cell, `N = 15 000` after a fixed-length truncation, field radius 70′.

**The oracle is the `q = 1` cell**, where the answer must be zero by construction. It is not
decoration, and the section below is what it bought.

## What the null control caught

```{danger}
**The first run of the circular null returned `+0.0116 ± 0.0020` — 5.8σ from the zero it was
required to give.**

Cause: `_eff_model` fits a free flat background `b`; the generator injects none; so the true `b` is
**exactly zero**, on the boundary of a half-Cauchy prior. The fit put wing density into `b` anyway
and raised `γ` to compensate.

| true `γ` | background | bias in `γ` | inferred `b` |
|---|---|---|---|
| 2.0 | free | **+0.0116 ± 0.0020** (5.8σ) | **0.00716** |
| 2.0 | pinned ≈0 | −0.0038 ± 0.0034 (1.1σ) | 0.00000 |
| 2.5 | free | +0.0098 ± 0.0065 | 0.00247 |
| 2.5 | pinned ≈0 | −0.0010 ± 0.0064 | 0.00000 |

**The artefact does not cancel against the control.** The spurious `b` is ~3× larger at `γ` = 2.0
than at 2.5, because a shallower profile leaves more wing density to absorb — so its size depends on
the shape being measured, and on `q` as well. Subtracting the circular cell, which was the obvious
repair, would not have removed it.

The background is therefore pinned near zero by default. `--background` reruns the same grid with it
free, which answers the *applied* question ("how wrong is the fit as actually performed?") rather
than the clean one ("does ellipticity bias `γ`?"). They are two experiments, not one with a flag.
```

The same degeneracy confounds `tools/validation/eff_gamma_bias.py`, which also fits a free
background: its bias surface measures finite-sample bias **plus** background degeneracy, and the
second term grows as `N` falls with the same sign as the first. That surface must not be used as the
A5 correction until the two are separated.

## Result 1 — the slope is biased downward, toward the King limit

`Δγ` relative to the circular control, all cells converged (`r̂` < 1.008, ESS > 540, zero
divergences):

| `γ` true | `q` = 0.71 | `q` = 0.50 | `q` = 0.30 |
|---|---|---|---|
| 2.0 | −0.0010 | −0.0160 | −0.0239 |
| 2.5 | +0.0060 | −0.0246 | −0.0578 |
| 3.0 | +0.0069 | −0.0366 | −0.0919 |
| **4.0** | **−0.0782** | **−0.1484** | **−0.3636** |

Null control: largest `|Δγ|` at `q = 1` is 0.0175 against a largest SEM of 0.0177 — not significant.

```{important}
**The sign is the result.** A circular fit *underestimates* the slope, pushing it toward `γ = 2`.
At the population median axis ratio the effect is consistent with zero, but it grows steeply with
both flattening and slope, reaching −0.36 at `γ = 4`, `q = 0.3` — 9% of `γ`.

So **an apparent pile-up at `γ ≈ 2` in a census of circular fits can be manufactured by unmodelled
ellipticity**, and any A5 statement about clustering near the King limit has to say so.
```

## Result 2 — the scale radius is not biased, it is a different quantity

`a_fit / (a√q) = 1.020 ± 0.021` over all elliptical cells.

The circular fit recovers the **geometric-mean** scale radius, not the semi-major axis. A published
circular `r_c` is therefore not wrong — it is low by `√q`, and trivially correctable once an axis
ratio is known. This is the more useful of the two results for reading existing catalogues.

## What this does and does not claim

```{warning}
**Nobody-fits-an-axis-ratio would be false, and an earlier draft of the script's docstring implied
it.** The counter-examples are in this repository's own bibliography:

* **Pera et al. (2021)**, `2021BAAA...62..119P` — an **elliptical, rotated** King profile with
  rotation angle and eccentricity as free parameters, fitted by Bayesian inference to ten Milky Way
  open clusters.
* **Olivares et al. (2018)** — cluster elongation as a fitted parameter, selected by Bayes factor.

The surviving claim is narrower: **the bias in the recovered slope from fitting a circular model to
an elliptical cluster was unquantified.** That is the error made by the majority of published fits,
which are circular — not a statement that doing better is impossible.

This is the eighth novelty claim in this programme to need narrowing before publication, and the
pattern has been identical every time: a true statement about what *most* papers do, written as a
statement about what *no* paper does. See `~/phd/methodology.md` PART K.4.
```

## Limitations

* One profile family (EFF), one footprint-to-scale ratio, no substructure, no contamination and no
  completeness. The number quantifies *ellipticity alone*.
* The generator produces a smooth elliptical EFF. Real flattened clusters may also be rotating or
  tidally distorted, which are different geometries with possibly different biases.
* `N = 15 000` is far above the census median. The finite-sample bias measured on the circular null
  at this `N` is −0.0038 ± 0.0034; at census `N` it is larger and is the subject of
  `eff_gamma_bias.py`, which is currently confounded as noted above.
* No axis ratio was measured for NGC 6383, so nothing here changes its published `R_c` — it only
  identifies what that number *is* (a geometric-mean radius).
