#!/bin/bash
# Quick setup script for StatisticalDependencyRegularization (Miniforge / Apple Silicon)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="StatisticalDependencyRegularization"
ENV_FILE="$PROJECT_ROOT/environment.yml"
ENV_TEMP="$PROJECT_ROOT/environment.macos.tmp.yml"

echo "=== StatisticalDependencyRegularization Quick Setup (Miniforge) ==="
echo "Project root: $PROJECT_ROOT"

if ! command -v mamba &> /dev/null; then
    echo "Error: mamba not found. Miniforge installation is broken."
    exit 1
fi

echo "Generating macOS-compatible environment file..."
python3 - <<'PYEOF'
import sys, re
from pathlib import Path

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("environment.yml")
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("environment.macos.tmp.yml")

# Conda packages that are Linux-only (build strings reference linux or h*linux hashes)
linux_only_patterns = [
    r'^  - _libgcc_mutex',
    r'^  - _openmp_mutex',
    r'^  - libgcc-ng',
    r'^  - libgomp',
    r'^  - libstdcxx-ng',
    r'^  - glib=',
    r'^  - gstreamer',
    r'^  - gst-plugins-base',
    r'^  - libclang13',
    r'^  - libcups',
    r'^  - libllvm',
    r'^  - libpq',
    r'^  - mysql=',
    r'^  - cyrus-sasl',
    r'^  - dbus=',
    r'^  - krb5=',
    r'^  - pcre=',
    r'^  - qt-main',
    r'^  - pyqt=',
    r'^  - pyqt5-sip',
    r'^  - libxcb',
    r'^  - libxkbcommon',
    r'^  - ld_impl_linux',
    r'^  - libuuid',
]

# pip packages that are Linux/CUDA-only
pip_linux_only = {
    'nvidia-cublas-cu12', 'nvidia-cuda-cupti-cu12', 'nvidia-cuda-nvrtc-cu12',
    'nvidia-cuda-runtime-cu12', 'nvidia-cudnn-cu12', 'nvidia-cufft-cu12',
    'nvidia-curand-cu12', 'nvidia-cusolver-cu12', 'nvidia-cusparse-cu12',
    'nvidia-nccl-cu12', 'nvidia-nvjitlink-cu12', 'nvidia-nvtx-cu12',
    'triton', 'tensorflow-io-gcs-filesystem',
}

lines = Path("environment.yml").read_text().splitlines()
out = []
in_pip = False
for line in lines:
    # Track whether we're in the pip: section
    stripped = line.strip()
    if stripped == '- pip:':
        in_pip = True
        out.append(line)
        continue
    if in_pip and not stripped.startswith('-'):
        in_pip = False

    # Skip Linux-only conda packages
    if not in_pip and any(re.match(p, line) for p in linux_only_patterns):
        continue
    # Skip hardcoded prefix
    if line.startswith('prefix:'):
        continue
    # Skip pip packages that are Linux-only
    if in_pip and stripped.startswith('-') and '==' in stripped:
        pkg_name = stripped.lstrip('- ').split('==')[0].lower()
        if pkg_name in pip_linux_only:
            continue
    # Fix hardcoded PYTHONPATH in variables section
    if 'PYTHONPATH' in line and '/home/' in line:
        import os
        project_root = os.environ.get('PROJECT_ROOT', str(Path('.').absolute()))
        out.append(f'  PYTHONPATH: "${{PYTHONPATH}}:{project_root}"')
        continue
    # Strip Linux build hashes from conda package specs (e.g. pkg=1.0=h5eee18b_0 -> pkg=1.0)
    if not in_pip and re.match(r'^  - \S+=\S+=\S+', line):
        line = re.sub(r'(^  - \S+=\S+)=\S+$', r'\1', line)
    # Strip .postN suffixes from pip package versions (e.g. tensorflow==2.15.0.post1 -> tensorflow==2.15.0)
    if in_pip and re.search(r'==\S+\.post\d+', line):
        line = re.sub(r'(==\S+?)\.post\d+', r'\1', line)
    out.append(line)

Path("environment.macos.tmp.yml").write_text('\n'.join(out) + '\n')
print(f"Written macOS-compatible env file: environment.macos.tmp.yml")
PYEOF

cleanup() {
    rm -f "$ENV_TEMP"
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
