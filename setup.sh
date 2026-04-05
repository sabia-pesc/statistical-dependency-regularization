#!/bin/bash
# Quick setup script for StatisticalDependencyRegularization

set -e  # Exit on any error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="StatisticalDependencyRegularization"

echo "=== StatisticalDependencyRegularization Quick Setup ==="
echo "Project root: $PROJECT_ROOT"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda."
    exit 1
fi

# Create or update environment
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' exists. Updating..."
    conda env update -f environment.yml
else
    echo "Creating environment '$ENV_NAME'..."
    conda env create -f environment.yml
fi

# Configure PYTHONPATH
echo "Configuring PYTHONPATH..."
conda env config vars set PYTHONPATH="\${PYTHONPATH}:$PROJECT_ROOT" -n $ENV_NAME

echo ""
echo "=== Setup Complete ==="
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To download datasets, run:"
echo "  python setup.py"
echo ""
echo "Or to download datasets manually, activate the environment and run:"
echo "  python -c 'from setup import download_datasets; download_datasets()'"
