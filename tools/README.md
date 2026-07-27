# EROTICA Development Tools

This directory contains development and maintenance tools for the EROTICA project.

## Structure

### Build and Release
- `build/` - Build automation scripts
- `release/` - Release preparation and deployment tools
- `packaging/` - Package distribution utilities

### Development
- `dev/` - Development environment setup
- `testing/` - Test automation and CI/CD tools
- `profiling/` - Performance monitoring and optimization

### Data Management
- `data/` - Data processing and validation tools
- `migration/` - Database and format migration utilities
- `validation/` - Data quality assurance tools

## Usage

Most tools can be run from the project root:

```bash
# Build tools
python tools/build/check_dependencies.py
python tools/build/generate_docs.py

# Release tools
bash tools/release/prepare_release.sh
python tools/release/update_version.py

# Development tools
python tools/dev/setup_environment.py
python tools/testing/run_comprehensive_tests.py
```