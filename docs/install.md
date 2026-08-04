# Installation

EROTICA requires **Python ≥ 3.13**. This is `requires-python` in `pyproject.toml`, and it is what
CI runs (3.13 and 3.14) — an older version cannot install the package at all, so the "≥ 3.11" this
page used to state was not a soft floor but a wrong one.

```bash
pip install erotica
```

Or from source:

```bash
git clone https://github.com/notluquis/erotica
cd erotica
pip install -e .
```

## Optional extras

The core install pulls only what clustering needs. Heavier or task-specific stacks are
opt-in:

| Extra | Adds | For |
|-------|------|-----|
| `bayes` | PyMC, ArviZ, blackjax | Bayesian isochrone / membership fitting |
| `selection` | gaiaunlimited, selection functions | completeness-aware work |
| `examples` | Jupyter, seaborn | running the example notebooks |
| `docs` | Sphinx + pydata theme | building this documentation |
| `dev` | pytest, pytest-cov, **ruff**, mypy, pre-commit, nbstripout | contributing |

```bash
pip install "erotica[bayes]"
```

The `dev` extra ships **`ruff`**, which replaced `black` + `isort` + `flake8`; it is what
`.pre-commit-config.yaml` runs and therefore what gates a commit. This table named `black` until
2026-08-04, so anyone following it installed the `dev` extra and then ran a formatter that was not
there.
