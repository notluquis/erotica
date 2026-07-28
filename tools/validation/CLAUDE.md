# tools/validation/ — one-off experiments that become paper numbers

Scripts here produce numbers that get quoted in design notes and papers. That makes them permanent,
whatever the intent when they were written.

## The rule that created this directory

**A number in a design note must have a script here.** The γ-coverage table in
`king_model_validity.md` was produced in an ad-hoc session; only the table survived, and reconstructing
it later found a design flaw that would have published a false result. If an experiment is worth
quoting, it is worth a file.

## What each script must carry

- **A module docstring that states why the experiment exists and what would falsify the conclusion.**
  Not what the code does — why the number is needed and what answer would overturn it.
- **The negative controls and the nulls**, run and reported. `completeness_bias_scaling.py` runs the
  zero-suppression level precisely so "consistent with zero" is distinguishable from "too noisy".
- **A JSON sidecar** with the full result, committed. Large arrays go to `.npz`, which is
  **gitignored** — so anything a paper quotes must be in the JSON, and the `.npz` must be
  regenerable by re-running the script.
- **Live queries stay live.** Scripts fetch from VizieR / the Gaia archive / `gaiaunlimited` rather
  than caching numbers into the source, so a stale catalogue cannot silently persist.

## Two failure modes specific to this directory

- **Resolution and regime artefacts.** A selection-function value was under-read **4.5×** by running
  at the wrong healpix order, and a completeness extrapolation was wrong by 46% because the slope came
  from the wrong side of a nonlinearity. State the resolution and the validity range in the docstring,
  and check the target sits inside it.
- **`| tail` masks exit codes and buffers output.** It has hidden a failure twice. Redirect to a file
  and inspect it, or check `$?` explicitly.

## Before trusting a result from here

Verify upstream to downstream: **generator → estimator → interpretation**. The EFF sampler was checked
against the analytic CDF *before* a `γ` offset was attributed to estimator bias — which is what turned
a suspected bug into a result that corrects a published number.

See `~/phd/methodology.md` PART K.
