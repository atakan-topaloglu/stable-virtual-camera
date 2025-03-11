#!/bin/bash

# if [ ! -d "venv" ]; then
#     echo "Creating python3.10 virtual environment in 'venv'..."
#     python3.10 -m venv venv
# fi

# echo "Activating the virtual environment..."
# source venv/bin/activate

# Install seva dependencies.
echo "Installing seva dependencies..."
pip install -e .

git submodule update --init --recursive

# Install pycolmap dependencies
pip install -e third_party/pycolmap

# Install dust3r dependencies for demo (our model is not dependent on it).
echo "Installing dust3r dependencies (only for demo)..."
pushd third_party/dust3r
pip install -r requirements.txt
pip install -r requirements_optional.txt
popd