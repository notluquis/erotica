# Contributing to COSMIC

Thank you for your interest in contributing to COSMIC! This document provides guidelines for contributing to the project.

## Development Status

COSMIC is currently in **alpha development** (v0.0.1). The API is not yet stable and may change significantly between versions. We welcome contributions but recommend discussing major changes first.

## Getting Started

### Development Environment Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/COSMIC.git
   cd COSMIC
   ```

2. **Create a development environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

3. **Verify the installation**
   ```bash
   python -c "import cosmic; print('COSMIC imported successfully')"
   pytest  # Run the test suite
   ```

## How to Contribute

### Reporting Bugs

Before creating bug reports, please:
- Check if the issue has already been reported
- Include detailed information about your environment
- Provide a minimal example that reproduces the issue

**Bug Report Template:**
- **Python version**: (e.g., 3.11.5)
- **COSMIC version**: (e.g., 0.0.1)
- **Operating system**: (e.g., macOS 14.1)
- **Dependencies**: List relevant package versions
- **Description**: Clear description of the problem
- **Reproduction steps**: Minimal code example
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens

### Suggesting Features

Feature suggestions are welcome! Please:
- Search existing issues first
- Explain the use case and motivation
- Consider if it fits COSMIC's scope (astronomical clustering and analysis)
- Provide a detailed description of the proposed functionality

### Pull Requests

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed
   - Test your changes thoroughly

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

4. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

### Code Style
- Follow [PEP 8](https://pep8.org/) for Python code style
- Use meaningful variable and function names
- Add docstrings for all public functions and classes
- Maximum line length: 88 characters (Black default)

### Documentation
- All public functions must have comprehensive docstrings
- Use NumPy-style docstrings
- Include examples in docstrings when helpful
- Update README.md for user-facing changes

### Testing
- Write tests for all new functionality
- Maintain or improve test coverage
- Use pytest for testing framework
- Place tests in the `tests/` directory

### Type Hints
- Use type hints for function signatures
- Import types from `typing` or built-in equivalents (Python 3.9+)
- Use `from __future__ import annotations` for forward compatibility

### Example Code Structure

```python
\"\"\"Module docstring describing the module's purpose.\"\"\"
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from astropy.table import QTable


def process_data(
    data: QTable,
    columns: Sequence[str],
    *,
    threshold: float = 0.5,
    verbose: bool = False,
) -> QTable:
    \"\"\"Process astronomical data with the given parameters.
    
    Parameters
    ----------
    data : QTable
        Input astronomical data table.
    columns : Sequence[str]
        Column names to process.
    threshold : float, optional
        Processing threshold, by default 0.5.
    verbose : bool, optional
        Enable verbose output, by default False.
        
    Returns
    -------
    QTable
        Processed data table.
        
    Examples
    --------
    >>> import cosmic
    >>> data = cosmic.DataLoader('catalog.ecsv').load_data()
    >>> processed = process_data(data, ['pmra', 'pmdec'])
    \"\"\"
    if verbose:
        logging.info(f"Processing {len(data)} rows with threshold {threshold}")
    
    # Implementation here
    return data
```

## Commit Message Guidelines

Use conventional commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for adding tests
- `refactor:` for code refactoring
- `style:` for formatting changes
- `chore:` for maintenance tasks

Examples:
```
feat: add Optuna optimization for HDBSCAN parameters
fix: resolve circular import in utils module
docs: update installation instructions for v0.0.1
test: add integration tests for clustering workflow
```

## Review Process

1. **Automated checks**: All PRs must pass automated tests
2. **Code review**: Maintainers will review code for quality and style
3. **Testing**: Verify that changes don't break existing functionality
4. **Documentation**: Ensure documentation is updated appropriately

## Release Process

### Versioning
COSMIC follows [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

### Alpha/Beta Releases
During development:
- `0.0.x` - Initial alpha releases
- `0.x.0` - Feature development releases
- `1.0.0` - First stable release

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: Contact Lucas Pulgar-Escobar at lescobar2019@udec.cl

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:
- Be respectful and constructive in discussions
- Focus on what is best for the community
- Show empathy towards other community members

## Recognition

Contributors will be acknowledged in:
- The README.md file
- Release notes and changelogs
- Academic papers using COSMIC (when appropriate)

Thank you for contributing to COSMIC! 🌟