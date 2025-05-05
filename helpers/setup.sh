#!/bin/bash 

# helpers/setup.sh

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
fi

# 4) Confirm the virtual environment is active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment activation failed."
    exit 1
else
    echo "Virtual environment activated successfully."
fi

# 5) Upgrade pip
python -m pip install --upgrade pip

# 7) Continue with the rest of the packages
python -m build
python -m pip install --upgrade pip setuptools wheel
pip install --upgrade -e .[devel]
