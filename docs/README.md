# COSMIC Documentation

This directory contains comprehensive documentation for the COSMIC project.

## Structure

- `api/` - Auto-generated API documentation
- `tutorials/` - Step-by-step tutorials and examples
- `guides/` - How-to guides for specific tasks
- `reference/` - Reference materials and technical specifications
- `contributing/` - Development and contribution guidelines

## Building Documentation

To build the documentation locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html
```

The built documentation will be available in `docs/_build/html/`.

## Contributing to Documentation

See the [Contributing Guide](../CONTRIBUTING.md) for guidelines on improving documentation.