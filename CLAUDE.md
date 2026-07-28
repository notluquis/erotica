# EROTICA — repo guidance

Python package `erotica/` (clustering, isochrone, dynamics, kinematics,
photometry, structure, analysis) **plus** the full source of the NGC 6383 A&A
paper (aa52082-24) under `data/test/NGC6383/`. Remote: `notluquis/erotica`,
working branch `dev`. (Renamed COSMIC→EROTICA 2026-07-21 — package, import, PyPI
dist, GitHub repo, and the working dir `/Users/notluquis/COSMIC` →
`/Users/notluquis/erotica`.)

Pipeline/status/roadmap for **all** papers live in the hub repo (attached via
`--add-dir`): `~/phd/PIPELINE.md` (state), `~/phd/ROADMAP.md` (plan),
`~/phd/erotica-package.md` (package backlog), `~/phd/papers/PXX.md` (dossiers).
Keep those current; do **not** spawn TODO files here.

## Before writing scientific code here, read these two

- **`~/phd/methodology.md` PART K — the execution craft.** The failure modes that have actually
  produced wrong numbers in this repo, each with its case: generators that produce something other
  than their label, experiments degenerate with their own control, tests that cannot fail, and the
  verification order (generator → estimator → interpretation). **Every generator needs a
  parameter-free special case with a known closed form, and that case must be a test.** Re-apply each
  bug you fix and confirm the suite goes red — a test you have not seen fail is a hypothesis.
- **`~/phd/model-landscape.md`** — per-module map of the published model alternatives, marked
  SURVEYED / PARTIAL / UNSURVEYED. Read the row for the module you are touching. An UNSURVEYED row is
  a recorded liability, not a neutral state; when you make a modelling choice, update the row and name
  what you rejected.

Design decisions are logged **append-only** in `docs/design-notes/decisions.md` — record the number
that was wrong and why, not only the fix. A struck-through row in a results table is worth more than
the row that replaced it.

## Landmines
- **~30 files hardcode `/Users/notluquis/erotica/...`** (paper figure regen). They
  were rewritten during the 2026-07-21 dir move; if this dir moves again, rewrite
  them in the same pass or they all break.
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
  (release-blocker: fix the pre-commit config — see `~/phd/erotica-package.md`).

## Data traps (respect in any catalog work)
- `comments_paper/_legacy/rerun_2026-05/members_DIFFERENT_RUN_DO_NOT_USE.csv` =
  a different run (177 not 254 members, corrupted float source_ids). Never use.
- At 60/70′ the generic HDBSCAN `max_members` branch is NOT the NGC branch — use
  the dual-label exports.
- Notebook 25′ figures are stale producers; `comments_paper/members.csv`-style
  short sweeps ≠ the paper reference run.
