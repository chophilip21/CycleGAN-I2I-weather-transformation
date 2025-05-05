#!/bin/bash
set -e

# Determine script and project root
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT" || exit 1

echo "Project root: $PROJECT_ROOT"

# Verify Python 3 presence and version
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi
PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
if (( $(echo "$PYTHON_VERSION > 3.11" | bc -l) )); then
    echo "Error: Python version must be 3.11 or lower. Current: $PYTHON_VERSION"
    exit 1
fi

# 1) Create or recreate virtual environment
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv env
    echo "Virtual environment created."
else
    read -p "Virtual environment 'env' already exists. Recreate? (yes/no): " response
    if [[ "$response" == "yes" ]]; then
        echo "Removing existing 'env'..."
        rm -rf env
        python3 -m venv env
        echo "Virtual environment recreated."
    else
        echo "Using existing virtual environment."
    fi
fi

# 2) Activate virtual environment
OS_TYPE=$(uname -s)
if [[ "$OS_TYPE" == MINGW* || "$OS_TYPE" == CYGWIN* || "$OS_TYPE" == MSYS_NT* ]]; then
    source env/Scripts/activate
else
    source env/bin/activate
fi

# 3) Upgrade pip, setuptools, wheel
python3 -m pip install --upgrade pip setuptools wheel

# 4) Install numpy (pre-built wheel)
pip install --only-binary :all: "numpy>=1.21.6"

# # 5) Install CPU-only PyTorch and torchvision
# python3 -m pip install torch==2.0.1+cpu torchvision==0.15.2+cpu \
#     -f https://download.pytorch.org/whl/torch_stable.html

# 6) Install NVIDIA DALI
pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda110

# 7) Install Apex
rm -rf /tmp/unique_for_apex
mkdir -p /tmp/unique_for_apex
cd /tmp/unique_for_apex
git clone --depth 1 https://github.com/NVIDIA/apex apex
cd apex
# disable CUDA mismatch check
sed -i '/raise RuntimeError: Cuda extensions are being compiled/ s/^/#/' setup.py
python3 setup.py install --cuda_ext --cpp_ext
cd "$PROJECT_ROOT"
rm -rf /tmp/unique_for_apex

# 8) Install mseg-api from GitHub
echo "Installing mseg-api from GitHub..."
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install git+https://github.com/mseg-dataset/mseg-api.git

# 9) Install project dependencies
pip install --upgrade -e .[devel]

echo "Setup completed. To activate: source env/bin/activate"
