# COSMIC — repo guidance

Python package `cosmic/` (clustering, isochrone, dynamics, kinematics,
photometry, structure, analysis) **plus** the full source of the NGC 6383 A&A
paper (aa52082-24) under `data/test/NGC6383/`. Remote: `notluquis/COSMIC`,
working branch `dev`.

Pipeline/status/roadmap for **all** papers live in the hub repo (attached via
`--add-dir`): `~/phd/PIPELINE.md` (state), `~/phd/ROADMAP.md` (plan),
`~/phd/cosmic-package.md` (package backlog), `~/phd/papers/PXX.md` (dossiers).
Keep those current; do **not** spawn TODO files here.

## Landmines
- **26 scripts hardcode `/Users/notluquis/COSMIC/...`** (paper figure regen). Do
  not move or rename this dir — it breaks them all.
- **Paper source of truth:**
  `data/test/NGC6383/comments_paper/submission_package/clean_source/aanda.tex`.
  An older working copy sits at `data/test/NGC6383/Tex_File/aanda.tex` — don't
  edit that one by mistake. (The legacy `Tex_File/NGC6383_manuscript.tex` was
  retired in the analysis-layer migration; state tagged
  `ngc6383-aanda-resubmission`.)
- `data/test/NGC6383/` is ~3.1 GB and committed to the repo (paper repro
  artifacts, hardcoded paths). Reproducibility tag: `ngc6383-aanda-resubmission`.
- **pre-commit is broken here:** the `nbstripout` hook chokes on the large
  working-tree stash and can revert unstaged changes into a `.cache/pre-commit/`
  patch. Commit with `git commit --no-verify` until the hook is fixed
  (release-blocker: fix the pre-commit config — see `~/phd/cosmic-package.md`).

## Data traps (respect in any catalog work)
- `comments_paper/_legacy/rerun_2026-05/members_DIFFERENT_RUN_DO_NOT_USE.csv` =
  a different run (177 not 254 members, corrupted float source_ids). Never use.
- At 60/70′ the generic HDBSCAN `max_members` branch is NOT the NGC branch — use
  the dual-label exports.
- Notebook 25′ figures are stale producers; `comments_paper/members.csv`-style
  short sweeps ≠ the paper reference run.
