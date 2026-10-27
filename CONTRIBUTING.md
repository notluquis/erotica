# Contributing to EROTICA

Thanks for your interest. EROTICA is a research package for Gaia open-cluster analysis,
so contributions are judged on scientific correctness as much as on code quality.

## Reporting bugs

Open a [GitHub issue](https://github.com/notluquis/erotica/issues) with:

- what you ran (the call, the columns you clustered on, the data source),
- what you expected and what happened,
- the traceback if there is one, and your Python + EROTICA versions
  (`python -c "import erotica; print(erotica.__version__)"`).

For a scientific discrepancy (membership, ages, radii), say which numbers you expected and
why — a literature value, another tool, a previous run.

## Asking questions

Use [GitHub Discussions](https://github.com/notluquis/erotica/discussions) for usage
questions and methodology questions ("should I cluster in 5D for my cluster?"). The
[membership guide](https://erotica.readthedocs.io/en/latest/guides/membership.html)
covers the main trade-offs.

## Development setup

```bash
git clone https://github.com/notluquis/erotica.git
cd erotica
pip install -e ".[dev]"
pytest -q
```

### Commit identity (and git worktrees)

Set `user.name` and `user.email` before the first commit. Without them git falls back to the
login name and `<user>@<hostname>.local`, and the commit is recorded under an identity nobody
chose: 64 commits on `dev` as of 2026-10-26 carry `Lucas Matías Pulgar
<notluquis@MacBook-Pro-de-Lucas-3.local>` that way. `.mailmap` folds that identity into the
canonical one for `git shortlog` and `git log --format=%aN`; it does not change the history.

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.org"   # or your GitHub noreply address
```

**In a `git worktree`, a plain `git config user.name …` is not local to the worktree**: it writes
the shared `.git/config` of the main checkout, so every worktree and the main checkout change
together. Measured 2026-10-26. To give one worktree its own identity, either pass it per command
(`git -c user.name="…" -c user.email="…" commit …`) or enable per-worktree configuration, which
itself edits the shared config once:

```bash
git config extensions.worktreeConfig true          # shared config, once per repository
git config --worktree user.name  "Your Name"        # this worktree only
git config --worktree user.email "you@example.org"
```

AI-assisted commits carry a `Co-Authored-By:` trailer naming the model. A commit that was
assisted but lacks the trailer is declared in `tools/release/ai_assisted_without_trailer.txt`,
which `tools/release/ai_disclosure_counts.py` reads for the paper's AI-usage disclosure.

## Submitting changes

1. Branch off `dev` (`git checkout -b fix/short-description`).
2. Make the change **and add a test that fails without it**. For anything touching the
   science — membership, isochrones, structure, dynamics — the test should assert on
   behaviour, not just that the code runs.
3. Keep the style consistent with **`ruff`**, which replaced `black` + `isort` + `flake8` and is
   what `.pre-commit-config.yaml` actually runs (line length 100, configured in `pyproject.toml`):

   ```bash
   ruff check --fix erotica/ tests/
   ruff format erotica/ tests/
   ```

   `black` and `isort` are **not** installed by the `dev` extra. This step named them until
   2026-08-04, so following it produced "command not found".
4. Run the suite: `pytest -q`.
5. Open a pull request against `dev` describing what changed and, for science changes,
   what you validated it against.

## Scientific conventions

These are house rules; a PR that breaks them will be asked to change:

- **Don't silently change a published default.** The NGC 6383 results in `data/test/` are a
  reproducibility artifact; if a change moves those numbers, say so in the PR.
- **Options with disclosure.** When adding a method that can fail on some cluster
  morphology, add it as an option and document the failure mode — do not make it the
  default. See the membership guide for the pattern.
- **State what is estimated vs assumed.** Priors, fixed parameters, and the model grid an
  age is conditioned on belong in the docstring.
- **Uncertainties are not decoration.** Prefer returning a distribution or an interval over
  a bare point estimate where the method supports it.

## Code of conduct

Be respectful and constructive. Harassment or personal attacks are not tolerated; report
concerns to [lescobar2019@udec.cl](mailto:lescobar2019@udec.cl).
