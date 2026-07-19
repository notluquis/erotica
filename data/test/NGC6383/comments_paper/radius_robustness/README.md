# Radius robustness check

Purpose: answer the A&A referee comment about whether the NGC 6383 conclusions
change if the Gaia cone search is run at 50 or 60 arcmin instead of the
submitted 40 arcmin.

Run from the repository root:

```bash
/Users/notluquis/miniforge3/envs/cosmic/bin/python tools/validation/ngc6383_radius_robustness.py
```

The base Python currently lacks the Gaia `zero_point` dependency; use the
`cosmic` conda environment for full preprocessing.

By default this now writes the intermediate `Clustering` objects to:

```text
data/test/NGC6383/comments_paper/radius_robustness/generated/dill/
```

The clustering audit reads from that folder, so the whole chain can be
regenerated without depending on old `/private/tmp` artifacts.

To also write the large ECSV tables:

```bash
/Users/notluquis/miniforge3/envs/cosmic/bin/python tools/validation/ngc6383_radius_robustness.py --write-full-tables
```

Generated outputs go to `generated/`, which is intentionally ignored by git.

## Last verified paper-faithful run

Settings:

- preprocessing via `DataLoader` and `DataPreprocessor`
- radii: 40, 50, 60, 70 arcmin
- `min_cluster_size_samples=range(10, 300)`
- `cluster_selection_method="leaf"`
- `allow_single_cluster=True`
- `match_reference_implementation=False`
- composite `probability = probability_hdbscan * probability_times`
- parallax sigma clipping from the selected branch at `probability >= 0.5`
- final reference sample at `probability >= 0.6`

| radius | good | bad | best mcs | algorithm label | NGC-like label | NGC branch | post-clip p>=0.5 | final p>=0.6 | p>=0.6, G<19 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 15276 | 6939 | 43 | 0 | 0 | 701 | 321 | 254 | 236 |
| 50 | 23844 | 12048 | 18 | 28 | 28 | 767 | 355 | 236 | 215 |
| 60 | 33309 | 18708 | 143 | 1 | 0 | 696 | 476 | 443 | 384 |
| 70 | 44438 | 27137 | 177 | 1 | 0 | 798 | 650 | 628 | 518 |

The `NGC branch` column refers to the NGC 6383-like branch, selected by
proximity to the reference NGC 6383 proper-motion overdensity. The accompanying
clustering audit shows that at 60 and 70 arcmin the generic sweep-selected
branch differs from this NGC-like branch.

Interpretation for the response: the NGC 6383 proper-motion branch is recovered
at all radii, but the final membership count is radius-sensitive once the cone
search becomes much larger. The 60 and 70 arcmin runs grow strongly; that does
not by itself prove that the added sources are contaminants or real extended
members. Those alternatives require CMD, parallax, spatial, astrometric-quality,
and H-alpha/YSO diagnostics.
