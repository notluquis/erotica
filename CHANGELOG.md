# Changelog

All notable changes to the PUMPS project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `pumps.__version__`, sourced from installed package metadata (`importlib.metadata`).
- Real test coverage replacing placeholder stubs: preprocessing corrections
  (Cantat-Gaudin & Brandt proper-motion spin correction, photometric errors,
  fidelity splitting, parallax zero-point), data loading (alias resolution,
  `DataLoader`), plus regression tests for dynamics, inference, King structure,
  I/O helpers, and the Sagitta guard.
- GitHub Actions CI workflow (pytest on Python 3.11–3.13).
- JOSS software-paper scaffolding (`paper/paper.md`, `paper/paper.bib`) and
  `CITATION.cff`.
- King-profile summaries now expose correctly named `*_median` keys.

### Changed
- Declared previously-undeclared runtime dependencies `gaiadr3-zeropoint` and
  `fast-histogram` (fixes `import pumps.preprocess` failing on a clean install).
- Reconciled `LICENSE` to AGPL-3.0 to match `pyproject.toml`.
- Provenance metadata now records the real package version instead of a
  hardcoded `"0.0.1"`.

### Fixed
- `AttributeError` crash in the Galactic branch of
  `calculate_galactocentric_distance` (`Angle.radians` → `.radian`).
- Fractional parallax-error selection admitted unphysical negative parallaxes and
  divided by zero at `parallax == 0`; now requires `parallax > 0`.
- Masked 64-bit identifier columns (e.g. Gaia `source_id`) were promoted to
  `float64`+NaN, silently corrupting IDs above 2^53 and breaking crossmatches.
- King-trace point estimates were medians mislabeled as `*_mean`.

### Security
- Removed a silent runtime `pip install` of an unpinned git ref in
  `pumps.analysis._sagitta`; it now raises `ImportError` with install
  instructions instead of modifying the user's environment.

### Removed
- Fake "build passing" / "docs" status badges from the README.

## [0.0.1] - 2025-10-03

### Added
- Initial alpha release of PUMPS
- Core clustering functionality with HDBSCAN and Optuna optimization
- Data loading utilities for Gaia, 2MASS, and WISE photometric systems
- Comprehensive data preprocessing and cleaning tools
- Statistical analysis and visualization capabilities
- Modular package structure with organized submodules:
  - `pumps.core` - Clustering algorithms and core functionality
  - `pumps.io` - Data loading and I/O operations
  - `pumps.preprocess` - Data preprocessing and quality control
  - `pumps.analysis` - Statistical analysis and characterization
  - `pumps.utils` - General utility functions
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