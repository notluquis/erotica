# Moved

The NGC 6383 A&A manuscript (aa52082-24) and its review/validation tooling moved to their own
public repository on 2026-09-22:

**https://github.com/notluquis/paper-ngc6383-aa52082-24**

That repo carries `submission_package/`, `cds_final/`, `referee_round2/`, `referee_round3/`,
`review_repo/`, this directory's former top-level files, and `validation/ngc6383_*.py`
(extracted from `erotica/tools/validation/`).

The full commit history of everything that used to live here is preserved in this repo
(`erotica`) up to the tag `p01-pre-extraction`. The mapping from each old commit SHA in this
history to its new SHA in the paper repo is recorded there, at
`provenance/erotica-commit-map.txt`.

The raw input data this manuscript's figures and tables were built from —
`data/test/NGC6383/ASteCA/`, `MIST/`, `PARSEC/`, `Tex_File/`, `25/`, `data/`,
`comparison_database/`, `NGC6383_DSS2-red.fits`, `optuna_study.db` — stays in **this** repo,
because `erotica`'s own figure-regeneration and validation scripts (`tools/validation/*.py`,
`tools/probes/*.py`, `tools/prototypes/*.py`) read it by path. Only the manuscript source,
referee correspondence, and its dedicated review/validation tooling moved.

A handful of ignored/untracked files under this directory (`_legacy/`, `radius_robustness/generated/`,
`clustering_audit/generated/`, `6383_old_paper/`, `submission_package/*.zip`,
`submission_package/**/_gate_build/`) were never tracked here and were left on disk rather than
copied or deleted, since some of `erotica`'s own scripts still read the gitignored
`radius_robustness/generated/` outputs at runtime.
