# Calibration of membership probabilities

How to check whether a membership statistic behaves as a probability with
{mod}`erotica.calibration` {bdg-success}`now`, and how to recalibrate it. The question it answers,
in the module's words: "whether a source assigned p̃ ≈ 0.8 is a true cluster member roughly 80% of
the time."

---

## 1. What the module does

Given per-star probabilities and **ground-truth labels**, it computes four diagnostics and fits two
recalibration maps. Everything is `numpy`/`scipy`/`scikit-learn`; no PyMC.

| Function | What it measures or does | What it costs |
|----------|--------------------------|---------------|
| `reliability_diagram` | binned predicted vs observed member fraction | bins: `"uniform"` (Guo et al. 2017) or `"quantile"` |
| `brier_score` | mean squared error between p̃ and the label; 0 is perfect | mixes calibration with sharpness |
| `expected_calibration_error` | `(ECE, MCE)`: count-weighted mean and max bin gap | depends on the binning |
| `hosmer_lemeshow` | $\hat H$ statistic and its $\chi^2_{g-2}$ p-value | dof convention, §4 |
| `calibration_report` | all of the above in one object | none |
| `corp` | CORP reliability (Dimitriadis, Gneiting & Jordan 2021): PAV fit and `BS = MCB - DSC + UNC` | MCB in-sample, biased up by ~log(n)/n |
| `fit_isotonic` / `fit_platt` / `recalibrate` | a fitted map p̃ → calibrated p̃ | needs labelled data, §4 |

`calibration_report_from_clustering` reads the `probability` column (p̃) straight from a
`Clustering` object after `search_pseudoprobability`; the labels must align row by row.

## 2. A minimal example

The labels need a truth you trust. Here it is synthetic: a latent score sets each star's true
membership probability, and the statistic under test is overconfident by construction.

```python
import numpy as np

from erotica.calibration import calibration_report

rng = np.random.default_rng(7)
n = 4000
# Synthetic truth: a latent score sets each star's true membership probability,
# and the label is drawn from it. The statistic under test is OVERCONFIDENT:
# the same score with a steeper slope, so it is right about the ranking and
# wrong about the probabilities.
score = rng.normal(0.0, 1.5, n)
is_member = rng.random(n) < 1.0 / (1.0 + np.exp(-score))
p_tilde = 1.0 / (1.0 + np.exp(-1.8 * score))

report = calibration_report(p_tilde, is_member)
hl = report.hosmer_lemeshow
print(f"Brier {report.brier_score:.4f}  ECE {report.expected_calibration_error:.4f}  "
      f"MCE {report.maximum_calibration_error:.4f}")
print(f"Hosmer-Lemeshow H = {hl.statistic:.1f}, dof {hl.dof}, p = {hl.pvalue:.2g}")
```

```text
Brier 0.1886  ECE 0.0946  MCE 0.1453
Hosmer-Lemeshow H = 678.7, dof 8, p = 2.7e-141
```

## 3. Reading the output

The reliability diagram says *where* the statistic is off. An overconfident p̃ sits below the
diagonal at high p̃ and above it at low p̃:

```python
rd = report.reliability
for lo, hi, pred, obs, cnt in zip(rd.bin_edges[:-1], rd.bin_edges[1:],
                                  rd.mean_predicted, rd.observed_frequency, rd.counts):
    print(f"[{lo:.1f}, {hi:.1f})  predicted {pred:.3f}  observed {obs:.3f}  n = {cnt}")
```

```text
[0.0, 0.1)  predicted 0.039  observed 0.154  n = 853
[0.1, 0.2)  predicted 0.144  observed 0.289  n = 387
[0.2, 0.3)  predicted 0.248  observed 0.363  n = 292
[0.3, 0.4)  predicted 0.347  observed 0.409  n = 264
[0.4, 0.5)  predicted 0.451  observed 0.383  n = 266
[0.5, 0.6)  predicted 0.550  observed 0.492  n = 250
[0.6, 0.7)  predicted 0.650  observed 0.591  n = 235
[0.7, 0.8)  predicted 0.750  observed 0.675  n = 280
[0.8, 0.9)  predicted 0.854  observed 0.762  n = 361
[0.9, 1.0)  predicted 0.962  observed 0.871  n = 812
```

Empty bins carry `nan`. To draw it:

```python
import matplotlib

matplotlib.use("Agg")
report.reliability.plot(show_counts=True, save_path="reliability.png")
```

**Recalibrate on one half, judge on the other.** A map fitted and scored on the same stars looks
better than it is:

```python
from erotica.calibration import fit_isotonic, fit_platt

# Fit the map on one half, judge it on the other.
train = rng.random(n) < 0.5
for fit in (fit_isotonic, fit_platt):
    recal = fit(p_tilde[train], is_member[train])
    after = calibration_report(recal(p_tilde[~train]), is_member[~train])
    print(f"{recal.method:>8}: Brier {after.brier_score:.4f}  "
          f"ECE {after.expected_calibration_error:.4f}  "
          f"H-L p = {after.hosmer_lemeshow.pvalue:.2g}")
before = calibration_report(p_tilde[~train], is_member[~train])
print(f"    none: Brier {before.brier_score:.4f}  ECE {before.expected_calibration_error:.4f}  "
      f"H-L p = {before.hosmer_lemeshow.pvalue:.2g}")
```

```text
isotonic: Brier 0.1792  ECE 0.0320  H-L p = 0.29
   platt: Brier 0.1785  ECE 0.0310  H-L p = 0.057
    none: Brier 0.1874  ECE 0.0923  H-L p = 3e-66
```

Both maps remove most of the miscalibration on the held-out half. In this toy the true relation is
a sigmoid in logit(p̃), which Platt can represent exactly; on real data neither form is guaranteed.
Both maps are monotone, so they never reverse the order of two stars; the isotonic step function
can tie stars that were distinct.

## 4. Limits, as the docstrings state them

- **Hosmer–Lemeshow degrees of freedom.** "``dof = n_groups - 2`` is the convention Hosmer &
  Lemeshow calibrated for probabilities *fit* to the same sample by logistic regression. EROTICA's
  p̃ is computed externally (not fit to these labels), so ``g-2`` is mildly anti-conservative here;
  ``n_groups - 1`` is a more conservative reference." Groups whose expected variance collapses to
  ~0 are skipped.
- **Isotonic** is "Flexible but data-hungry and prone to overfit on small samples; fit and evaluate
  on independent draws for an honest estimate."
- **Platt** here is the plain maximum-likelihood fit on logit(p̃) (`C=1e10`), without the
  target-smoothing prior of Platt (1999): "for the large synthetic samples used to calibrate EROTICA
  this is negligible, but note it for small-sample use."
- **The labels decide what "calibrated" means.** The package's own checks use synthetic
  field + cluster mixtures, where truth is known. Against a published catalog, such as the Hunt &
  Reffert DR3 members, a reliability diagram measures consistency with that catalog, not absolute
  truth (see {doc}`membership`, §5).

The theory behind these choices is in the design note
[calibration theory](../design-notes/calibration_theory.md).
