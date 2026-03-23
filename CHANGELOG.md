# Changelog

All notable changes to the COSMIC project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Planned: PyPI distribution setup
- Planned: Comprehensive documentation and tutorials
- Planned: Extended test coverage

## [0.0.1] - 2025-10-03

### Added
- Initial alpha release of COSMIC
- Core clustering functionality with HDBSCAN and Optuna optimization
- Data loading utilities for Gaia, 2MASS, and WISE photometric systems
- Comprehensive data preprocessing and cleaning tools
- Statistical analysis and visualization capabilities
- Modular package structure with organized submodules:
  - `cosmic.core` - Clustering algorithms and core functionality
  - `cosmic.io` - Data loading and I/O operations
  - `cosmic.preprocess` - Data preprocessing and quality control
  - `cosmic.analysis` - Statistical analysis and characterization
  - `cosmic.utils` - General utility functions
- Backward-compatible shims for legacy import patterns
- Professional package configuration with `pyproject.toml`
- Comprehensive README with installation and usage instructions
- Development environment setup with testing framework

### Changed
- Reorganized flat module structure into logical subpackages
- Converted duplicate files to clean re-export shims
- Updated project metadata for first release
- Improved code organization and maintainability
- **Eliminated legacy directory after successful migration**

### Fixed
- Resolved circular import issues in utility modules
- Corrected package export paths and import statements
- Fixed inconsistent function naming across modules

### Removed
- Legacy code files after complete migration to organized structure
- Obsolete backup files and temporary directories

### Technical Details
- Python 3.11+ requirement established
- Dependencies properly specified in `pyproject.toml`
- Clean separation of concerns across submodules
- Maintained API compatibility during reorganization

---

**Note**: This is an alpha release intended for development and testing. The API may change significantly in future versions as we work toward a stable 1.0.0 release.