```python
# For demo.
pip install -e .

# Or for development.
pip install -e ".[dev]"
pre-commit install

# Install the torch nightly version for faster JIT (speed up sampling by 2x in our testing).
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
```
