#!/usr/bin/env python3
"""Run comprehensive test suite for PUMPS."""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run command and report results."""
    print(f"\n🧪 {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} passed")
        return True
    else:
        print(f"❌ {description} failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False

def main():
    """Run comprehensive test suite."""
    project_root = Path(__file__).parent.parent.parent
    print(f"🚀 Running comprehensive tests for PUMPS in {project_root}")
    
    os.chdir(project_root)
    
    tests = []
    
    # Unit tests
    tests.append((
        "python -m pytest tests/ -v --cov=cosmic --cov-report=term-missing",
        "Unit tests with coverage"
    ))
    
    # Code formatting
    tests.append((
        "black --check pumps/ tests/",
        "Code formatting (Black)"
    ))
    
    # Import sorting
    tests.append((
        "isort --check-only pumps/ tests/",
        "Import sorting (isort)"
    ))
    
    # Linting
    tests.append((
        "flake8 pumps/ tests/",
        "Code linting (flake8)"
    ))
    
    # Type checking
    tests.append((
        "mypy pumps/ --ignore-missing-imports",
        "Type checking (mypy)"
    ))
    
    # Build test
    tests.append((
        "python -m build --sdist --wheel",
        "Package build test"
    ))
    
    # Import test
    tests.append((
        "python -c 'import pumps; print(\"PUMPS imports successfully\")'\n",
        "Import test"
    ))
    
    # Run all tests
    passed = 0
    failed = 0
    
    for cmd, description in tests:
        if run_command(cmd, description):
            passed += 1
        else:
            failed += 1
    
    # Summary
    total = passed + failed
    print(f"\n📊 Test Results: {passed}/{total} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed! Ready for release.")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed. Please fix before release.")
        return False

if __name__ == "__main__":
    import os
    success = main()
    sys.exit(0 if success else 1)