# EROTICA: Estimation, Recovery & Optimization, together with Inference, for Cluster Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Development Status](https://img.shields.io/badge/status-alpha-red.svg)](https://github.com/notluquis/erotica)
[![CI](https://github.com/notluquis/erotica/actions/workflows/ci.yml/badge.svg)](https://github.com/notluquis/erotica/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/cosmic-clusters/badge/?version=latest)](https://cosmic-clusters.readthedocs.io/en/latest/)

EROTICA is a Python package for analyzing star clusters using machine learning and Bayesian
inference. Built for Gaia data, it uses unsupervised clustering to identify open star clusters
and characterize their membership, ages, and structure.

📖 **[Documentation](https://cosmic-clusters.readthedocs.io/en/latest/)** · [Membership guide](https://cosmic-clusters.readthedocs.io/en/latest/guides/membership.html) · [API reference](https://cosmic-clusters.readthedocs.io/en/latest/api/index.html)

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
[quickstart](https://cosmic-clusters.readthedocs.io/en/latest/quickstart.html).

## 📦 Installation

### From Source (Current)

```bash
git clone https://github.com/notluquis/erotica.git
cd erotica
pip install -e ".[dev,docs]"
```

### Requirements

- Python 3.11 or higher
- See `pyproject.toml` for complete dependency list

## 🏗️ Project Structure

```
erotica/
├── erotica/               # 📦 Main package
│   ├── core/              # 🔧 HDBSCAN clustering + membership
│   ├── io/                # 📊 Data loading and I/O
│   ├── preprocess/        # 🧹 Data preprocessing
│   ├── analysis/          # 📈 Isochrones, structure, dynamics, inference
│   ├── calibration.py     # 🎯 Probability calibration
│   └── utils/             # 🛠️ Utility functions
├── docs/                  # 📚 Sphinx documentation
│   ├── guides/            # User guides (membership, …)
│   ├── design-notes/      # Grounded method notes
│   └── api/               # Auto-generated API reference
├── tools/                 # 🔧 Development + release tooling
├── data/test/NGC6383/     # 🔬 Paper reproduction artifacts
└── tests/                 # 🧪 Test suite
```

## 🌟 Features

### 🔬 Advanced Clustering
- **HDBSCAN** with persistence thresholding
- **Hyperparameter optimization** via Grid Search and Optuna
- **Multiple validation metrics** (relative validity, DBCV, persistence)
- **Robust outlier detection** and noise handling

### 📊 Multi-System Data Support
- **Gaia** (positions, proper motions, parallaxes, photometry)
- **2MASS** (near-infrared photometry)
- **WISE** (mid-infrared photometry)
- **Automatic unit handling** and data validation

### 🧹 Comprehensive Preprocessing
- **Zero-point corrections** for Gaia photometry
- **Proper motion corrections** for systematic effects
- **Quality-based data splitting** with fidelity metrics
- **Missing value handling** and outlier detection

### 📈 Statistical Analysis
- **Cluster characterization** with kinematic properties
- **Membership probability** assessment
- **Sagitta integration** for stellar parameter estimation
- **Comprehensive visualization** suite

## 📖 Documentation

Full documentation: **[cosmic-clusters.readthedocs.io](https://cosmic-clusters.readthedocs.io/en/latest/)**

### Getting started
- [Installation](https://cosmic-clusters.readthedocs.io/en/latest/install.html)
- [Quickstart](https://cosmic-clusters.readthedocs.io/en/latest/quickstart.html)

### Guides
- [Membership](https://cosmic-clusters.readthedocs.io/en/latest/guides/membership.html) —
  which features to cluster on, error-aware membership, calibration, and the trade-offs of each choice

### Reference
- [API reference](https://cosmic-clusters.readthedocs.io/en/latest/api/index.html)
- [Design notes](https://cosmic-clusters.readthedocs.io/en/latest/design-notes/index.html) —
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

```bash
# Format code
black erotica/ tests/
isort erotica/ tests/

# Check code quality
flake8 erotica/ tests/
mypy erotica/ --ignore-missing-imports

# Run all quality checks
python tools/testing/run_comprehensive_tests.py
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
4. Run the test suite: `python tools/testing/run_comprehensive_tests.py`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

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

- **Documentation**: [cosmic-clusters.readthedocs.io](https://cosmic-clusters.readthedocs.io/en/latest/)
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
