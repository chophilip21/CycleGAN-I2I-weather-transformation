#!/bin/bash 

# helpers/setup.sh

# Check Python version
PYTHON_VERSION=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
if (( $(echo "$PYTHON_VERSION >= 3.11" | bc -l) )); then
    echo "Error: Python version must be less than 3.11."
    echo "Current version: $PYTHON_VERSION"
    echo "Please use Python 3.10 or lower."
    exit 1
fi

# 1) Check if the virtual environment exists
if [ -d "env" ]; then

    # If a virtual environment might be active, deactivate it first
    if [[ -n "$VIRTUAL_ENV" && "$VIRTUAL_ENV" == *"/env" ]]; then
        echo "Deactivating currently active virtual environment..."
        deactivate
    fi

    read -p "Virtual environment 'env' already exists. Do you want to recreate it? (yes/no): " yn
    case $yn in
        [Yy]* ) 
            echo "Removing existing 'env' directory..."
            rm -rf env
            ;;
        [Nn]* ) 
            echo "Using existing virtual environment."
            ;;
        * ) 
            echo "Please answer yes or no."
            exit 1
            ;;
    esac
fi

# 2) Create the virtual environment if it's missing
if [ ! -d "env" ]; then
    python -m venv env
    echo "Virtual environment created."
fi

# 3) Detect OS and activate the virtual environment
OS_TYPE=$(uname -s)
if [[ "$OS_TYPE" == "MINGW"* || "$OS_TYPE" == "CYGWIN"* || "$OS_TYPE" == "MSYS_NT"* ]]; then
    # Windows activation
    source env/Scripts/activate
else
    # Linux or macOS activation
    source env/bin/activate

    # 4) Install NVIDIA DALI
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110==1.7.0

    # 5) Install Apex
    mkdir -p /tmp/unique_for_apex
    cd /tmp/unique_for_apex
    git clone https://github.com/NVIDIA/apex
    cd apex
    python setup.py install --cuda_ext --cpp_ext
    cd -
    rm -rf /tmp/unique_for_apex

    # 6) Install mseg-api
    mkdir -p /tmp/unique_for_mseg_api
    SHA=ToUcHMe git clone https://github.com/mseg-dataset/mseg-api.git /tmp/unique_for_mseg_api/mseg-api
    pip install -e /tmp/unique_for_mseg_api/mseg-api
    rm -rf /tmp/unique_for_mseg_api

    # 7) Upgrade pip
    python -m pip install --upgrade pip

    # 8) Continue with the rest of the packages
    python -m build
    python -m pip install --upgrade pip setuptools wheel
    pip install --upgrade -e .[devel]
