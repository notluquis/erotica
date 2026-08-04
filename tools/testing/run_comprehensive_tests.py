#!/usr/bin/env python3
"""Run the comprehensive test suite for EROTICA.

README's "Manual Setup" section points users here, so its defects were user-facing. Fixed
2026-08-04:

* ``--cov=cosmic`` measured **nothing**. The package has been ``erotica`` since the
  2026-07-21 rename, and ``coverage`` reports 0% for a package that does not exist rather
  than erroring, so the step passed while measuring an empty set.
* ``black``, ``isort`` and ``flake8`` are not installed by the ``dev`` extra -- ``ruff``
  replaced all three, and it is what ``.pre-commit-config.yaml`` runs. Three of the seven
  checks were therefore guaranteed "command not found" failures for anyone following the
  README.
* ``import os`` sat at the bottom of the file, inside ``if __name__ == "__main__":``, while
  ``main()`` used ``os.chdir`` at its top. That worked only because the import happened to
  execute one line before the call, and broke on any other entry point.
* ``python -m build`` was run unconditionally, but ``build`` is in no extra -- so it too
  could only ever fail for anyone who followed the README. It is now skipped with a message
  when absent.

Together those were four of the seven checks failing for reasons that said nothing about the
code, plus a coverage number measured over an empty set.
"""

import importlib.util
import os
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
    print(f"🚀 Running comprehensive tests for EROTICA in {project_root}")

    os.chdir(project_root)

    tests = []

    # Unit tests. `--cov=erotica` is the package that exists; `--cov=cosmic` measured nothing.
    tests.append(
        (
            "python -m pytest tests/ -v --cov=erotica --cov-report=term-missing",
            "Unit tests with coverage",
        )
    )

    # Formatting — ruff, which replaced black + isort + flake8 and is what pre-commit runs.
    tests.append(("ruff format --check erotica/ tests/", "Code formatting (ruff format)"))

    # Linting, including import sorting (ruff's `I` rules subsume isort).
    tests.append(("ruff check erotica/ tests/", "Linting and import order (ruff check)"))

    # Type checking
    tests.append(("mypy erotica/ --ignore-missing-imports", "Type checking (mypy)"))

    # Build test. `build` is NOT in any extra in pyproject.toml -- not `dev`, not `docs` -- so
    # for anyone who set up with `pip install -e ".[dev]"` (which is what README and
    # CONTRIBUTING tell them to do) this step was an unconditional "No module named build".
    # That is the same defect as the black/isort/flake8 steps removed above: a check that can
    # only fail tells you nothing about the code. Skip it honestly instead, and say how to
    # enable it.
    if importlib.util.find_spec("build") is not None:
        tests.append(("python -m build --sdist --wheel", "Package build test"))
    else:
        print(
            "\n⏭️  Skipping the package build test: `build` is not installed and is not part "
            "of any extra.\n   Enable it with `pip install build`."
        )

    # Import test
    tests.append(
        ("python -c 'import erotica; print(\"EROTICA imports successfully\")'\n", "Import test")
    )

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
    success = main()
    sys.exit(0 if success else 1)
