# COSMIC: Characterization Of Star clusters using Machine-learning Inference and Clustering

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Development Status](https://img.shields.io/badge/status-alpha-red.svg)](https://github.com/notluquis/COSMIC)
[![Build Status](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/notluquis/COSMIC)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/README.md)

COSMIC is a professional Python package for analyzing star clusters using machine learning techniques and Bayesian inference. Built specifically for processing Gaia satellite data, COSMIC employs unsupervised clustering algorithms and statistical analysis to identify and characterize open star clusters.

## 🚀 Quick Start

```python
import cosmic

# Load and preprocess data
loader = cosmic.DataLoader("your_gaia_catalog.ecsv")
data = loader.load_data(systems=["Gaia", "TMASS"])

preprocessor = cosmic.DataPreprocessor(data)
good_data, bad_data = preprocessor.process()

# Perform clustering
clusterer = cosmic.Clustering(good_data, bad_data)
clusterer.search(['pmra', 'pmdec', 'parallax'])

# Analyze results
analyzer = cosmic.ClusterAnalyzer(clusterer.combined_data)
analyzer.run_analysis()
```

## 📦 Installation

### From Source (Current)

```bash
git clone https://github.com/notluquis/COSMIC.git
cd COSMIC
pip install -e ".[dev,docs,examples]"
```

### Requirements

- Python 3.11 or higher
- See `pyproject.toml` for complete dependency list

## 🏗️ Project Structure

```
COSMIC/
├── cosmic/                 # 📦 Main package
│   ├── core/              # 🔧 Clustering algorithms
│   ├── io/                # 📊 Data loading and I/O
│   ├── preprocess/        # 🧹 Data preprocessing
│   ├── analysis/          # 📈 Statistical analysis
│   └── utils/             # 🛠️ Utility functions
├── examples/              # 💡 Usage examples
│   ├── basic_clustering/  # Simple workflows
│   ├── data_loading/      # Data handling examples
│   ├── visualization/     # Plotting tutorials
│   ├── advanced_clustering/ # Complex scenarios
│   └── ngc6383/          # Real-world case study
├── docs/                  # 📚 Documentation
│   ├── api/              # Auto-generated API docs
│   ├── tutorials/        # Step-by-step guides
│   ├── contributing/     # Development guides
│   └── reference/        # Technical specifications
├── tools/                 # 🔧 Development tools
│   ├── build/            # Build automation
│   ├── testing/          # Test automation
│   └── release/          # Release management
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

### Getting Started
- [Installation Guide](docs/contributing/development_setup.md)
- [Quick Start Tutorial](examples/basic_clustering/)
- [Data Loading Guide](examples/data_loading/)

### Examples
- [Basic Clustering](examples/basic_clustering/) - Simple clustering workflows
- [Data Loading](examples/data_loading/) - Loading and preprocessing
- [NGC 6383 Analysis](examples/ngc6383/) - Complete real-world example
- [Advanced Clustering](examples/advanced_clustering/) - Parameter optimization

### API Reference
- [Core Clustering](docs/api/) - Main clustering algorithms
- [Data I/O](docs/api/) - Data loading and preprocessing
- [Analysis Tools](docs/api/) - Statistical analysis and visualization

## 🔧 Development

### Quick Setup

```bash
# Automated development environment setup
python tools/dev/setup_environment.py
```

### Manual Setup

```bash
# Install in development mode
pip install -e ".[dev,docs,examples]"

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
black cosmic/ tests/ examples/
isort cosmic/ tests/ examples/

# Check code quality
flake8 cosmic/ tests/
mypy cosmic/ --ignore-missing-imports

# Run all quality checks
python tools/testing/run_comprehensive_tests.py
```

## 📊 Status

**Current Version**: v0.0.1 (Alpha)

**Development Status**: Active development for v0.1.0 stable release

**Test Coverage**: Core functionality covered, expanding test suite

**Documentation**: API documented, tutorials in development

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing/development_setup.md) for details.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `python tools/testing/run_comprehensive_tests.py`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📄 License

COSMIC is licensed under the [GNU Affero General Public License v3.0](LICENSE). This ensures that any modifications or derivative works remain open source.

## 👥 Team

- **Lucas Pulgar-Escobar** - Universidad de Concepción, Chile ([lescobar2019@udec.cl](mailto:lescobar2019@udec.cl))
- **Nicolás Henríquez Salgado** - Universidad de Concepción, Chile

## 🙏 Acknowledgments

COSMIC builds upon excellent open-source libraries:
- [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) for density-based clustering
- [Optuna](https://optuna.org/) for hyperparameter optimization
- [Astropy](https://www.astropy.org/) for astronomical data handling
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities
- [Matplotlib](https://matplotlib.org/) for visualization

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/notluquis/COSMIC/issues)
- **Discussions**: [GitHub Discussions](https://github.com/notluquis/COSMIC/discussions)
- **Email**: [lescobar2019@udec.cl](mailto:lescobar2019@udec.cl)

## 🗺️ Roadmap

- [x] **v0.0.1** - Initial alpha release with core functionality
- [ ] **v0.1.0** - Stable API and comprehensive documentation
- [ ] **v0.2.0** - PyPI distribution and additional clustering algorithms
- [ ] **v1.0.0** - Production-ready release with full validation suite

---

**Citation**: If you use COSMIC in your research, please cite our paper (in preparation) and acknowledge the underlying libraries.

*Made with ❤️ for the astronomical community*
