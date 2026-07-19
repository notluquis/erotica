# Full pipeline rerun report

Date: 2026-05-14

Purpose: document the end-to-end rerun used for the NGC 6383 A&A referee
response robustness checks.

## Environment

Use:

```bash
/Users/notluquis/miniforge3/envs/cosmic/bin/python
```

The base Python at `/Users/notluquis/miniforge3/bin/python` does not include the
Gaia `zero_point` dependency required by `DataPreprocessor.apply_zero_point_correction()`.

## Commands

```bash
TQDM_DISABLE=1 /Users/notluquis/miniforge3/envs/cosmic/bin/python tools/validation/ngc6383_radius_robustness.py --write-full-tables
/Users/notluquis/miniforge3/envs/cosmic/bin/python tools/validation/ngc6383_clustering_audit.py
/Users/notluquis/miniforge3/envs/cosmic/bin/python tools/validation/ngc6383_verify_generated_outputs.py
```

## Pipeline fixes made before the successful rerun

- `ngc6383_radius_robustness.py` now writes its own `.dill` clustering objects
  to `radius_robustness/generated/dill/`.
- `ngc6383_clustering_audit.py` now reads those generated `.dill` files by
  default instead of relying on old `/private/tmp` objects.
- The parallax sigma-clipping centre function now wraps `histogram_mode` so it
  is compatible with newer Astropy calls that pass `axis`.
- The sigma-clipping centre now preserves the input parallax unit, avoiding
  dimensionless/quantity subtraction errors.
- `ngc6383_verify_generated_outputs.py` checks consistency across the radius
  summaries, audit summaries, final label summaries, and source-level ECSV
  flags.

## Preprocessing and clustering summary

All radii were processed through:

1. `DataLoader.load_data(...)`
2. `DataPreprocessor.rename_columns()`
3. `DataPreprocessor.drop_invalid_sources()`
4. `DataPreprocessor.fill_missing_values()`
5. `DataPreprocessor.apply_zero_point_correction()`
6. `DataPreprocessor.correct_proper_motion()`
7. `DataPreprocessor.add_photometric_errors()`
8. `DataPreprocessor.filter_data(fidelity_threshold=0.5)`
9. paper-faithful `search_pseudoprobability(...)`
10. parallax-mode sigma clipping on the NGC-like branch

| radius | loaded | invalid dropped | good | bad | best mcs | algorithm label | NGC-like label | NGC branch | post-clip p>=0.5 | final p>=0.6 | p>=0.6, G<19 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 23740 | 1525 | 15276 | 6939 | 43 | 0 | 0 | 701 | 321 | 254 | 236 |
| 50 | 38677 | 2785 | 23844 | 12048 | 18 | 28 | 28 | 767 | 355 | 236 | 215 |
| 60 | 56784 | 4767 | 33309 | 18708 | 143 | 1 | 0 | 696 | 476 | 443 | 384 |
| 70 | 78893 | 7318 | 44438 | 27137 | 177 | 1 | 0 | 798 | 650 | 628 | 518 |

## Main interpretation from this rerun

- The submitted 40 arcmin sample is reproduced: 254 reference members at
  `probability >= 0.6` after parallax clipping.
- The NGC-like branch is recovered at all tested radii.
- At 60 and 70 arcmin, the generic sweep-selected branch is not the NGC-like
  branch. This is why the robustness table must report both labels.
- The larger-radius additions cannot be called contamination from this test
  alone. They need CMD, parallax, sky-position, astrometric-quality, and
  H-alpha/YSO diagnostics.

## Generated outputs

Radius robustness:

```text
data/test/NGC6383/comments_paper/radius_robustness/generated/
```

Clustering audit:

```text
data/test/NGC6383/comments_paper/clustering_audit/generated/
```

Verification:

```text
data/test/NGC6383/comments_paper/clustering_audit/generated/verification_report.json
```

Verification status:

```text
40 ok algorithm_label=0  ngc_label=0  reference_p06=254
50 ok algorithm_label=28 ngc_label=28 reference_p06=236
60 ok algorithm_label=1  ngc_label=0  reference_p06=443
70 ok algorithm_label=1  ngc_label=0  reference_p06=628
```

