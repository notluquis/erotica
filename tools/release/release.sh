# COSMIC v0.0.1 Release Script

# This script helps prepare and execute the v0.0.1 release of COSMIC

set -e  # Exit on any error

echo "🚀 COSMIC v0.0.1 Release Preparation"
echo "===================================="

# Check we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Run this script from the COSMIC root directory."
    exit 1
fi

# Verify version in pyproject.toml
VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
if [ "$VERSION" != "0.0.1" ]; then
    echo "❌ Error: Version in pyproject.toml is '$VERSION', expected '0.0.1'"
    exit 1
fi

echo "✅ Version check passed: $VERSION"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/
echo "✅ Build artifacts cleaned"

# Run tests
echo "🧪 Running tests..."
if command -v pytest >/dev/null 2>&1; then
    if ! pytest tests/ -v; then
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
if [ ! -f "dist/cosmic_cluster_analysis-0.0.1.tar.gz" ]; then
    echo "❌ Source distribution not found"
    exit 1
fi

if [ ! -f "dist/cosmic_cluster_analysis-0.0.1-py3-none-any.whl" ]; then
    echo "❌ Wheel distribution not found"
    exit 1
fi

echo "✅ Build verification passed"

# Check package installability
echo "🔧 Testing package installation..."
TEMP_VENV=$(mktemp -d)
python -m venv "$TEMP_VENV"
source "$TEMP_VENV/bin/activate"

if ! pip install dist/cosmic_cluster_analysis-0.0.1-py3-none-any.whl; then
    echo "❌ Package installation failed"
    deactivate
    rm -rf "$TEMP_VENV"
    exit 1
fi

# Test basic imports
if ! python -c "import cosmic; print('✅ COSMIC imports successfully')"; then
    echo "❌ Package import failed"
    deactivate
    rm -rf "$TEMP_VENV"
    exit 1
fi

deactivate
rm -rf "$TEMP_VENV"
echo "✅ Package installation test passed"

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
echo "Package name: cosmic-cluster-analysis"
echo "Version: $VERSION"
echo "Built files:"
ls -la dist/

echo ""
echo "🎉 Release v0.0.1 is ready!"
echo ""
echo "Next steps:"
echo "1. Commit any remaining changes:"
echo "   git add ."
echo "   git commit -m 'chore: prepare v0.0.1 release'"
echo ""
echo "2. Create and push a release tag:"
echo "   git tag -a v0.0.1 -m 'Release v0.0.1'"
echo "   git push origin v0.0.1"
echo ""
echo "3. Create a GitHub release using the tag"
echo "   Upload dist/cosmic_cluster_analysis-0.0.1.tar.gz"
echo "   Upload dist/cosmic_cluster_analysis-0.0.1-py3-none-any.whl"
echo ""
echo "4. (Optional) Publish to PyPI when ready:"
echo "   python -m twine upload dist/*"
echo ""
echo "🚀 COSMIC v0.0.1 release preparation complete!"