#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Creating python3.10 virtual environment in 'venv'..."
    python3.10 -m venv venv
fi

echo "Activating the virtual environment..."
source venv/bin/activate

echo "Installing the project in editable mode..."
pip install -e .

# Save the original directory
ORIG_DIR=$(pwd)

echo "Entering third_party/dust3r to install its dependencies and build CUDA kernels..."
cd third_party/dust3r || { echo "third_party/dust3r directory not found! Exiting."; exit 1; }

echo "Installing dust3r dependencies..."
pip install -r requirements.txt
pip install -r requirements_optional.txt

# Return to the original directory
cd "$ORIG_DIR"
