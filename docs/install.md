# Installation

COSMIC targets Python ≥ 3.11.

```bash
pip install cosmic-cluster-analysis
```

Or from source:

```bash
git clone https://github.com/notluquis/COSMIC
cd COSMIC
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
| `dev` | pytest, black, mypy | contributing |

```bash
pip install "cosmic-cluster-analysis[bayes]"
```
