#!/usr/bin/env python3
"""Setup development environment for COSMIC."""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, check=True):
    """Run shell command with error handling."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result

def main():
    """Set up development environment."""
    project_root = Path(__file__).parent.parent.parent
    print(f"Setting up COSMIC development environment in {project_root}")
    
    # Install package in development mode
    print("\n1. Installing COSMIC in development mode...")
    run_command(f"cd {project_root} && pip install -e '.[dev,docs,examples]'")
    
    # Install pre-commit hooks
    print("\n2. Setting up pre-commit hooks...")
    run_command("pip install pre-commit")
    run_command(f"cd {project_root} && pre-commit install")
    
    # Install additional development tools
    print("\n3. Installing development tools...")
    dev_packages = [
        "black",
        "isort",
        "flake8",
        "mypy",
        "pytest-cov",
        "sphinx",
        "sphinx-rtd-theme",
        "jupyter",
        "jupyterlab"
    ]
    run_command(f"pip install {' '.join(dev_packages)}")
    
    print("\n✅ Development environment setup complete!")
    print("\nNext steps:")
    print("- Run tests: pytest")
    print("- Format code: black .")
    print("- Check types: mypy cosmic/")
    print("- Build docs: cd docs && make html")
    print("- Start Jupyter: jupyter lab")

if __name__ == "__main__":
    main()