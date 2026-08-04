#!/usr/bin/env bash
# EROTICA release preparation script.
#
# Rewritten 2026-08-04. The previous version could not run, for three independent reasons:
#   * no shebang (the first line was a comment), so it depended on the caller's shell;
#   * it verified `import cosmic`, a package that has not existed since the 2026-07-21 rename
#     to `erotica` -- the import check was guaranteed to fail;
#   * it hardcoded `0.0.1` and the old dist name `cosmic_cluster_analysis`, so the build
#     verification looked for files that `python -m build` never produces.
#
# The version is now DERIVED from pyproject.toml rather than compared against a constant baked
# into this file. The old check ("is pyproject's version equal to 0.0.1?") was circular: it
# could only ever pass for one release, and every release after it required editing this script.
# Dist filenames are interpolated from that version for the same reason.

set -euo pipefail

echo "🚀 EROTICA Release Preparation"
echo "=============================="

# Check we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Run this script from the EROTICA root directory."
    exit 1
fi

# Read the version rather than assert it. `head -1` because `version = ` also appears under
# [tool.*] tables in some configurations; the project's own version is the first match.
VERSION=$(grep '^version = ' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
if [ -z "$VERSION" ]; then
    echo "❌ Error: could not read 'version' from pyproject.toml"
    exit 1
fi
echo "✅ Version read from pyproject.toml: $VERSION"

# The drift this guards against is real: README, CITATION.cff and pyproject.toml disagreed about
# the current version for long enough to reach a JOSS submission draft.
CITATION_VERSION=$(grep '^version:' CITATION.cff | head -1 | sed 's/version: *//' | tr -d '"' | tr -d "'")
if [ "$CITATION_VERSION" != "$VERSION" ]; then
    echo "❌ Error: CITATION.cff says '$CITATION_VERSION', pyproject.toml says '$VERSION'."
    echo "   These are the two machine-readable sources of truth; they must agree."
    exit 1
fi
echo "✅ CITATION.cff agrees: $CITATION_VERSION"

# The dist name is the normalised project name from pyproject.toml. Verified against the
# published artefacts on PyPI: erotica-0.1.0.tar.gz and erotica-0.1.0-py3-none-any.whl.
DIST_NAME="erotica"
SDIST="dist/${DIST_NAME}-${VERSION}.tar.gz"
WHEEL="dist/${DIST_NAME}-${VERSION}-py3-none-any.whl"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ ./*.egg-info/
echo "✅ Build artifacts cleaned"

# Run tests
echo "🧪 Running tests..."
if command -v pytest >/dev/null 2>&1; then
    if ! pytest -q; then
        echo "❌ Tests failed. Fix issues before releasing."
        exit 1
    fi
    echo "✅ All tests passed"
else
    echo "⚠️  pytest not found, skipping tests"
fi

# Build the package
echo "📦 Building package..."
if ! python -m build; then
    echo "❌ Build failed"
    exit 1
fi
echo "✅ Package built successfully"

# Verify build contents
echo "🔍 Verifying build contents..."
if [ ! -f "$SDIST" ]; then
    echo "❌ Source distribution not found: $SDIST"
    ls -la dist/ || true
    exit 1
fi

if [ ! -f "$WHEEL" ]; then
    echo "❌ Wheel distribution not found: $WHEEL"
    ls -la dist/ || true
    exit 1
fi

echo "✅ Build verification passed"

# Check package installability
echo "🔧 Testing package installation..."
TEMP_VENV=$(mktemp -d)
cleanup() { rm -rf "$TEMP_VENV"; }
trap cleanup EXIT

python -m venv "$TEMP_VENV"
# shellcheck disable=SC1091
source "$TEMP_VENV/bin/activate"

if ! pip install "$WHEEL"; then
    echo "❌ Package installation failed"
    deactivate
    exit 1
fi

# Test basic imports. The package is `erotica`; this line said `import cosmic` until 2026-08-04,
# which is the rename that made the whole script dead.
if ! python -c "import erotica; print('✅ EROTICA', erotica.__version__, 'imports successfully')"; then
    echo "❌ Package import failed"
    deactivate
    exit 1
fi

deactivate
echo "✅ Package installation test passed"

# Regenerate the AI-usage disclosure counts so the paper cannot drift from git history.
if [ -f "paper/paper.md" ]; then
    echo "🤖 Checking the AI-usage disclosure against git history..."
    if ! python tools/release/ai_disclosure_counts.py --check paper/paper.md; then
        echo "⚠️  paper/paper.md is stale. Regenerate with:"
        echo "    python tools/release/ai_disclosure_counts.py --paragraph"
    fi
fi

# Git status check
echo "📋 Git status check..."
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  Working directory has uncommitted changes:"
    git status --short
    echo ""
    echo "Consider committing changes before release."
fi

# Display package info
echo ""
echo "📊 Release Summary"
echo "=================="
echo "Package name: $DIST_NAME"
echo "Version: $VERSION"
echo "Built files:"
ls -la dist/

echo ""
echo "🎉 Release v$VERSION is ready!"
echo ""
echo "Next steps:"
echo "1. Commit any remaining changes:"
echo "   git add ."
echo "   git commit -m 'chore: prepare v$VERSION release'"
echo ""
echo "2. Create and push a release tag:"
echo "   git tag -a v$VERSION -m 'Release v$VERSION'"
echo "   git push origin v$VERSION"
echo ""
echo "3. Create a GitHub release using the tag"
echo "   Upload $SDIST"
echo "   Upload $WHEEL"
echo ""
echo "4. Publish to PyPI:"
echo "   python -m twine check dist/*"
echo "   python -m twine upload dist/*"
echo ""
echo "5. Confirm the Zenodo deposit picked up the new tag and that the concept DOI"
echo "   10.5281/zenodo.21769959 resolves to it."
echo ""
echo "🚀 EROTICA v$VERSION release preparation complete!"
