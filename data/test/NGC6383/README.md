# NGC 6383 working area

This directory is a project-specific workspace used while developing the
`erotica` package and revising the A&A NGC 6383 manuscript. It is not the
general package source; reusable code should live under `erotica/`.

> The package was renamed COSMIC → EROTICA on 2026-07-21. This file still said
> `cosmic/` until 2026-08-04, which made it an instruction to put new code in a
> directory that does not exist. The manuscript sources under `Tex_File/` and
> `comments_paper/` legitimately say COSMIC — that was the name at submission
> time, and they are the historical record, not a rename that was missed.

## Layout

- `comments_paper/`: paper-facing revision material. This is the source of
  truth for the A&A response work, including the archived submitted member
  tables and the current referee-response checks.
- `Tex_File/`: A&A manuscript source. Use `aanda.tex` as the active manuscript;
  old MNRAS or presentation material belongs under `Tex_File/otros/`.
- `data/`: cone-search inputs and generated clustering products by radius
  (`40`, `50`, `60`, `70`). Raw `*-result.ecsv`, serialized runs, traces, and
  generated clustering outputs are data artifacts and should stay ignored.
- `25/`: archived 25 arcmin attempt. Keep this as historical context only.
- `notebooks/`: exploratory notebooks that are not the paper-facing source of
  truth.
  - `legacy_experiments/`: notebooks moved from the repo-level `experiments/`
    folder because they are NGC 6383 specific.
  - `workflow/`: loose workflow notebooks for local process/preprocess checks.
- `comparison_database/`: external comparison catalogues used in the paper.
- `MIST/` and `PARSEC/`: isochrone tables. These are large data dependencies,
  ignored by git.
- `ASteCA/`: old ASteCA copy/results used for comparison and historical checks.
- `COSMIC_aux.py`: deprecated compatibility shim for archived notebooks. (The
  filename keeps the old spelling because the archived notebooks import it by
  that name; it is a historical artefact, not a missed rename.) The reusable
  implementation now lives in `erotica/analysis/`; new notebooks should import
  from `erotica.analysis`.

## Paper-faithful clustering settings

The submitted 40 arcmin member sample is reproduced by the legacy composite
pseudo-probability path:

- `min_cluster_size_samples=range(10, 300)`
- `cluster_selection_method="leaf"`
- `allow_single_cluster=True`
- `match_reference_implementation=False`
- `probability = probability_hdbscan * probability_times`
- parallax clipping on the selected branch with `probability >= 0.5`
- final reference table with `probability >= 0.6`

Use `tools/validation/ngc6383_radius_robustness.py` for the 40/50/60/70
robustness check instead of reimplementing this in new notebooks.
