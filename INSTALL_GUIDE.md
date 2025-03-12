# :wrench: Installation

### Model Dependencies
```python
# Install seva model dependencies.
pip install -e .
```

### Demo Dependencies
To use the cli demo (`demo.py`) or the gradio demo (`demo_gr.py`), do the following.
```python
# Initialize and update submodules for demo.
git submodule update --init --recursive

# Install pycolmap dependencies for cli demo (our model is not dependent on it).
echo "Installing pycolmap (only for cli demo)..."
pip install -e third_party/pycolmap

# Install dust3r dependencies for gradio demo (our model is not dependent on it).
echo "Installing dust3r dependencies (only for gradio demo)..."
pushd third_party/dust3r
pip install -r requirements.txt
pip install -r requirements_optional.txt
popd
```

### Dev and Speeding Up (Optional)
```python
# [OPTIONAL] Install seva dependencies for development.
pip install -e ".[dev]"
pre-commit install

# [OPTIONAL] Install the torch nightly version for faster JIT via. torch.compile (speed up sampling by 2x in our testing).
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
```