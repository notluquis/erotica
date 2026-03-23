# Development Setup Guide

This guide will help you set up a development environment for contributing to COSMIC.

## Prerequisites

- Python 3.11 or higher
- Git
- A Python virtual environment manager (venv, conda, or similar)

## Quick Setup

Use our automated setup script:

```bash
# Clone the repository
git clone https://github.com/notluquis/COSMIC.git
cd COSMIC

# Run the setup script
python tools/dev/setup_environment.py
```

This script will:
- Install COSMIC in development mode
- Set up pre-commit hooks
- Install all development dependencies
- Configure testing and documentation tools

## Manual Setup

If you prefer manual setup:

### 1. Create Virtual Environment

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n cosmic-dev python=3.11
conda activate cosmic-dev
```

### 2. Install Development Dependencies

```bash
# Install COSMIC in development mode with all extras
pip install -e ".[dev,docs,examples]"
```

### 3. Set Up Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### 4. Verify Installation

```bash
# Run dependency check
python tools/build/check_dependencies.py

# Run tests
pytest

# Check code formatting
black --check cosmic/ tests/
isort --check-only cosmic/ tests/
flake8 cosmic/ tests/
mypy cosmic/ --ignore-missing-imports
```

## Development Workflow

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Format code
black cosmic/ tests/
isort cosmic/ tests/

# Check code quality
flake8 cosmic/ tests/
mypy cosmic/ --ignore-missing-imports
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cosmic --cov-report=html

# Run comprehensive test suite
python tools/testing/run_comprehensive_tests.py
```

### Documentation

```bash
# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

### Building Packages

```bash
# Build distribution packages
python -m build

# Check package
twine check dist/*
```

## Development Tools

Our `tools/` directory contains helpful development utilities:

- `tools/dev/setup_environment.py` - Automated environment setup
- `tools/build/check_dependencies.py` - Dependency verification
- `tools/testing/run_comprehensive_tests.py` - Complete test suite
- `tools/release/` - Release preparation tools

## Project Structure

```
COSMIC/
├── cosmic/                # Main package
│   ├── core/             # Core clustering algorithms
│   ├── io/               # Data loading and I/O
│   ├── preprocess/       # Data preprocessing
│   ├── analysis/         # Analysis and visualization
│   └── utils/            # Utility functions
├── tests/                # Test suite
├── examples/             # Usage examples
├── docs/                 # Documentation
├── tools/                # Development tools
└── data/                 # Sample data
```

## Git Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and commit: `git commit -m "Add your feature"`
3. Run tests: `python tools/testing/run_comprehensive_tests.py`
4. Push branch: `git push origin feature/your-feature`
5. Create pull request

Pre-commit hooks will automatically run formatting and basic checks.

## Getting Help

- Check existing [Issues](https://github.com/notluquis/COSMIC/issues)
- Start a [Discussion](https://github.com/notluquis/COSMIC/discussions)
- Contact maintainers: lescobar2019@udec.cl

## Next Steps

- Review the [Contributing Guidelines](../CONTRIBUTING.md)
- Explore [Examples](../../examples/README.md)
- Check the [API Documentation](../api/)
- Browse [Tutorials](../tutorials/)