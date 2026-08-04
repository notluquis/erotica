# EROTICA: Estimation, Recovery & Optimization, together with Inference, for Cluster Analysis

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Development Status](https://img.shields.io/badge/status-alpha-red.svg)](https://github.com/notluquis/erotica)
[![CI](https://github.com/notluquis/erotica/actions/workflows/ci.yml/badge.svg)](https://github.com/notluquis/erotica/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/erotica/badge/?version=latest)](https://erotica.readthedocs.io/en/latest/)

EROTICA is a Python package for analyzing star clusters using machine learning and Bayesian
inference. Built for Gaia data, it uses unsupervised clustering to identify open star clusters
and characterize their membership, ages, and structure.

📖 **[Documentation](https://erotica.readthedocs.io/en/latest/)** · [Membership guide](https://erotica.readthedocs.io/en/latest/guides/membership.html) · [API reference](https://erotica.readthedocs.io/en/latest/api/index.html)

## 📦 Installation

### From Source (Current)

```bash
git clone https://github.com/notluquis/erotica.git
cd erotica
pip install -e ".[dev,docs]"
```

### Requirements

- Python 3.13 or higher
- See `pyproject.toml` for complete dependency list

## 🚀 Quick Start

```python
import erotica

# Load and preprocess
loader = erotica.DataLoader("your_gaia_catalog.ecsv")
data = loader.load_data(systems=["Gaia", "TMASS"])

preprocessor = erotica.DataPreprocessor(data)
preprocessor.apply_zero_point_correction()
good_data, bad_data = preprocessor.filter_data(fidelity_threshold=0.5)

# Membership: sweep min_cluster_size, score by recovery frequency x strength.
# `columns` IS the feature-space choice — 2D proper motion is the safe default.
# Mixing units (adding parallax) requires standardizing the columns first;
# see the membership guide for the trade-offs.
clusterer = erotica.Clustering(good_data, bad_data)
clusterer.search_pseudoprobability(columns=("pmra", "pmdec"))

clusterer.clustering_statistics()
summary = clusterer.get_cluster_summary()
clusterer.save_results("members.ecsv", format="ascii.ecsv")
```

Downstream analysis (isochrone fitting, structure, dynamics) goes through
`ClusterAnalyzer`, which is constructed from a saved catalog — see the
[quickstart](https://erotica.readthedocs.io/en/latest/quickstart.html).

## 🏗️ Project Structure

```
erotica/
├── erotica/               # 📦 Main package
│   ├── core/              # 🔧 HDBSCAN clustering + membership
│   ├── io/                # 📊 Data loading and I/O
│   ├── preprocess/        # 🧹 Data preprocessing
│   ├── analysis/          # 📈 Isochrones, structure, dynamics, inference
│   │   ├── structure.py       #   radial profiles: King, EFF, Plummer, King+corona
│   │   ├── identifiability.py #   is this parameter measured, or is the prior answering?
│   │   ├── synthetic.py       #   fractal cluster generation, for validation
│   │   └── provenance.py      #   git SHA, checksums, seeds, dependency versions
│   ├── selection.py       # 🔭 Gaia DR3 selection function
│   ├── calibration.py     # 🎯 Probability calibration
│   └── utils/             # 🛠️ Utility functions
├── docs/                  # 📚 Sphinx documentation
│   ├── guides/            # User guides (membership, …)
│   ├── design-notes/      # Grounded method notes
│   └── api/               # Auto-generated API reference
├── tools/                 # 🔧 Development + release tooling
│   └── validation/        #   every quoted number has a script here
├── data/test/NGC6383/     # 🔬 Paper reproduction artifacts
└── tests/                 # 🧪 Test suite
```

## 🌟 What it does

### Radial structure, without binning
Profiles are fitted as an **unbinned inhomogeneous Poisson point process**, `Σ log λ(rᵢ) − Λ`, so
there is no bin width to choose and no Gaussian approximation to break in the sparse outer bins that
set the truncation radius. Four families share the machinery and are scored against each other by
SMC Bayes factor:

| model | when |
|---|---|
| **King (1962)** | the empirical standard. Note the additive background is *not* King's — see the docstring |
| **EFF** (Elson, Fall & Freeman 1987) | no tidal cutoff; the honest model when truncation is not locatable |
| **Plummer** | `EFF(γ = 4)` |
| **King + uniform-sphere corona** (Danilov & Putkov 2012; Seleznev 2016) | when the flat background is absorbing cluster structure rather than field |

Each carries a closed-form normalisation verified against quadrature, because the integral is
evaluated at every leapfrog step.

### Asking whether a number means anything
`erotica.analysis.identifiability` answers *"is this parameter measured, or is the prior answering for
it?"* — **before** it is quoted. Four angles, because a parameter can pass one and fail another:
power-scaling prior sensitivity (Kallioinen et al. 2024), posterior geometry and condition number,
relative width, and the Muñoz et al. (2012) geometric criteria. It exists because this project
repeatedly reported numbers the data did not determine and found out late.

### Membership and calibration
HDBSCAN with persistence thresholding and a hyperparameter sweep scored by recovery frequency ×
strength. The resulting pseudo-probability is **calibrated against an external benchmark** rather than
assumed to be a probability, and a target–decoy construction measures the false-discovery proportion
independently of the fit.

### Selection functions
Gaia DR3 completeness via `gaiaunlimited`, foldable directly into the point-process normalisation, so
a radial completeness gradient enters the likelihood rather than being corrected afterwards.

### Synthetic clusters for validation, not for science
`fractal_cluster` implements the Goodwin & Whitworth (2004) box-fractal construction — still the live
standard in 2026 — because drawing test data from a smooth profile begs the question when the thing
under test *is* a profile fit.

### Reproducibility
`build_metadata` records git SHA and dirty flag, input-data checksums, RNG seeds, sampler
configuration and dependency versions. A result without those is not reproducible, only repeatable.

## ⚠️ What it will not tell you

Stated up front because the alternative is a referee stating it:

- **A tidal radius is often prior-determined.** For a footprint that does not contain the cluster,
  `R_t` is unconstrained regardless of how well the sampler converged. Any `R_t` must name its prior.
- **The EFF slope is not recoverable at typical Gaia-census geometry.** The controlling variable is
  the footprint-to-scale ratio, not the sample size.
- **Circular symmetry is assumed and is the exception** — median core axis ratio **0.71**, computed
  here from Tarricq et al. (2022)'s published table, not quoted from their text (they report the
  distribution as peaking at 0.8–0.9).
- **On a membership-selected sample the background term is not contamination.** It absorbs the
  corona, and for one cluster it accounts for 56% of the sample against a 6.1% measured
  false-discovery rate.

Each of these is measured, scripted under `tools/validation/`, and written up in
`docs/design-notes/`.

## 🔭 How it relates to other packages

Written once because three reviewers ask for it: pyOpenSci's review template asks for a comparison to
similar packages, JOSS requires a **State of the field** section, and Astropy's affiliated criteria
ask that a package avoid duplicating existing functionality.

**Supported Python: 3.13+** (CI runs 3.13 and 3.14).

| package | what it does that EROTICA does not | overlap |
|---|---|---|
| [**ASteCA**](https://github.com/asteca/ASteCA) | The standard for **photometric** synthetic clusters and CMD fitting: IMF, mass-dependent binary fraction fitted to Offner et al. (2023), extinction with free `R_V`, differential reddening, errors calibrated on the observed cluster. **Cite it; do not compete with it.** | membership; a private, unwired least-squares King fit |
| [**ocelot**](https://github.com/emilyhunt/ocelot) | Injects mock clusters into real Gaia DR3 with a two-stage selection function and error twin-resampling — 80,590 injection/retrievals behind the census completeness function. | radial profiles (it ships King62 only) |
| [**Kalkayotl**](https://github.com/olivares-j/Kalkayotl) | Bayesian cluster distances with **parallax spatial correlations** (Lindegren et al. 2020), which EROTICA does not model. The strongest error model in the Bayesian open-cluster set. | distance inference |
| [**gaiaunlimited**](https://github.com/gaia-unlimited/gaiaunlimited) | The Gaia DR3 selection function itself. EROTICA **uses** it rather than reimplementing it. | — |
| [**pyUPMASK**](https://github.com/msolpera/pyUPMASK) / UPMASK | Membership by iterative clustering plus a random-field null. | membership |
| [**SPISEA**](https://github.com/astropy/SPISEA) | Stellar population synthesis: IMF, multiplicity, extinction, IFMR. No spatial structure. | — |

**What is distinctive here**, stated narrowly enough to survive checking:

- Radial profiles as an **unbinned inhomogeneous Poisson point process**, with King, EFF, Plummer and
  King-plus-corona compared by Bayes factor under one likelihood.
- A **selection function folded into the normalisation** rather than applied as an after-the-fact
  correction.
- An **identifiability toolkit** that asks whether a parameter is measured before it is quoted.

**What is not distinctive, and is documented as such**: Bayesian King fitting is not new — Pera et al.
(2021, 2024) precede it. EFF fits to Galactic open clusters as a population are not new either;
Cordoni et al. (2023) fit 78 and Pang et al. (2022) fit 82. Claims of novelty in this project have
been falsified seven times, so they are checked before they are written.

## 📖 Documentation

Full documentation: **[erotica.readthedocs.io](https://erotica.readthedocs.io/en/latest/)**

> **Resolved 2026-08-03.** This block used to warn that the Read the Docs slug was still
> `cosmic-clusters` and that `erotica.readthedocs.io` would 404. That is no longer true, and the
> paragraph had also been corrupted by the rename pass — it printed the *same* URL on both sides of
> "while", so it read as a contradiction and then instructed future editors not to fix it.
> Measured directly: `erotica.readthedocs.io/en/latest/` → **HTTP 200**,
> `cosmic-clusters.readthedocs.io/en/latest/` → **HTTP 404**. The link above is correct; use it.

### Getting started
- [Installation](https://erotica.readthedocs.io/en/latest/install.html)
- [Quickstart](https://erotica.readthedocs.io/en/latest/quickstart.html)

### Guides
- [Membership](https://erotica.readthedocs.io/en/latest/guides/membership.html) —
  which features to cluster on, error-aware membership, calibration, and the trade-offs of each choice

### Reference
- [API reference](https://erotica.readthedocs.io/en/latest/api/index.html)
- [Design notes](https://erotica.readthedocs.io/en/latest/design-notes/index.html) —
  grounded notes on the isochrone sampler, model grids, and HDBSCAN membership

Build the docs locally with `pip install -e ".[docs]" && sphinx-build -b html docs docs/_build/html`.

## 🔧 Development

### Quick Setup

```bash
# Automated development environment setup
python tools/dev/setup_environment.py
```

### Manual Setup

```bash
# Install in development mode
pip install -e ".[dev,docs]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest

# Run comprehensive test suite
python tools/testing/run_comprehensive_tests.py
```

### Code Quality

**`ruff` replaced `black` + `isort` + `flake8`.** It is what `.pre-commit-config.yaml` actually runs,
so it is what gates a commit; the three tools this section used to name are no longer installed by
the `dev` extra.

```bash
# Lint and autofix, then format — the same two hooks pre-commit runs
ruff check --fix erotica/ tests/
ruff format erotica/ tests/

# Types (deliberately NOT a pre-commit hook — too slow for the commit loop)
mypy erotica/ --ignore-missing-imports

# Or just install the hooks and let them run on every commit
pre-commit install
```

## 📊 Status

**Current Version**: v0.0.1 (Alpha)

**Development Status**: Active development for v0.1.0 stable release

**Test Coverage**: Core functionality covered, expanding test suite

**Documentation**: API documented, tutorials in development

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest -q` (what CI runs, and what `CONTRIBUTING.md` says)
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📣 Citing EROTICA

If EROTICA contributes to work you publish, please cite it. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff), which GitHub renders as a "Cite this repository" button.

```bibtex
@software{erotica,
  author = {Pulgar, Lucas M.},
  title  = {{EROTICA}: Estimation, Recovery \& Optimization, together with Inference,
            for Cluster Analysis},
  url    = {https://github.com/notluquis/erotica},
  license = {AGPL-3.0-or-later}
}
```

> **Pre-release.** The author list, ORCIDs and a Zenodo DOI are not final — see the header of
> `CITATION.cff`. A software paper is in preparation; this entry will be superseded by it.

**Please also cite the methods you actually used**, which are not ours: King (1962) or Elson, Fall &
Freeman (1987) for the profile you fitted, Hunt & Reffert (2024) for census cross-matches,
Cantat-Gaudin et al. (2023) via `gaiaunlimited` for the selection function, Kallioinen et al. (2024)
for the power-scaling diagnostics, and Goodwin & Whitworth (2004) for the synthetic generator. Each
is cited at the point of use in the API documentation.

## 📄 License

EROTICA is licensed under the [GNU Affero General Public License v3.0](LICENSE). This ensures that any modifications or derivative works remain open source.

## 👥 Team

- **Lucas Pulgar-Escobar** - Universidad de Concepción, Chile ([lescobar2019@udec.cl](mailto:lescobar2019@udec.cl))
- **Nicolás Henríquez Salgado** - Universidad de Concepción, Chile

## 🙏 Acknowledgments

EROTICA builds upon excellent open-source libraries:
- [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) for density-based clustering
- [Optuna](https://optuna.org/) for hyperparameter optimization
- [Astropy](https://www.astropy.org/) for astronomical data handling
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities
- [Matplotlib](https://matplotlib.org/) for visualization

## 📞 Support

- **Documentation**: [erotica.readthedocs.io](https://erotica.readthedocs.io/en/latest/)
- **Issues**: [GitHub Issues](https://github.com/notluquis/erotica/issues)
- **Discussions**: [GitHub Discussions](https://github.com/notluquis/erotica/discussions)
- **Email**: [lescobar2019@udec.cl](mailto:lescobar2019@udec.cl)

## 🗺️ Roadmap

- [x] **v0.0.1** - Initial alpha release with core functionality
- [ ] **v0.1.0** - Stable API and comprehensive documentation
- [ ] **v0.2.0** - PyPI distribution and additional clustering algorithms
- [ ] **v1.0.0** - Production-ready release with full validation suite

---

**Citation**: If you use EROTICA in your research, please cite our paper (in preparation) and acknowledge the underlying libraries.

*Made with ❤️ for the astronomical community*
