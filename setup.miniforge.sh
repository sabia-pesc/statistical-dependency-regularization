#!/bin/bash
# Quick setup script for StatisticalDependencyRegularization (Miniforge / Apple Silicon)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="StatisticalDependencyRegularization"

echo "=== StatisticalDependencyRegularization Quick Setup (Miniforge) ==="
echo "Project root: $PROJECT_ROOT"

if ! command -v mamba &> /dev/null; then
    echo "Error: mamba not found. Miniforge installation is broken."
    exit 1
fi


cleanup() {
    rm -f "$ENV_FILE"
}
trap cleanup EXIT

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' exists. Updating..."
    mamba env update -f "$ENV_TEMP"
else
    echo "Creating environment '$ENV_NAME'..."
    mamba env create -f "$ENV_TEMP"
fi

echo "Configuring PYTHONPATH..."
conda env config vars set PYTHONPATH="\${PYTHONPATH}:$PROJECT_ROOT" -n "$ENV_NAME"

echo ""
echo "=== Setup Complete ==="
echo "Activate with:"
echo "  conda activate $ENV_NAME"
