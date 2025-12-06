# Colab/Kaggle Testing Guide for Co-DETR PyTorch

This guide provides step-by-step instructions for setting up and testing the Co-DETR PyTorch codebase.

## 1. Clone the Repository

```bash
!git clone https://github.com/QuackPhuc/co-DETR.git
%cd co-DETR
```

## 2. Install Dependencies

Install the package in editable mode (recommended for testing):

```bash
!pip install -e .
```

This will install all dependencies listed in `requirements.txt` and make the `codetr` package importable.

## 3. Verify Installation

Run a simple import check to verify the installation:

```python
import codetr
from codetr.configs.config import Config, load_config, merge_config
print("Import successful!")
```

## 4. Run Tests

Use `pytest` to run the test suite:

```bash
# Run all tests
!python -m pytest tests/ -v

# Run a specific test file
!python -m pytest tests/test_config.py -v

# Run tests with short summary
!python -m pytest tests/ --tb=short
```

## 5. Run Individual Test Scripts

Alternatively, you can run individual test files directly:

```bash
!python tests/test_config.py
```

## Troubleshooting

### `ImportError: cannot import name 'X' from 'Y'`

This usually means the package is not installed correctly. Reinstall in editable mode:

```bash
!pip uninstall codetr -y
!pip install -e .
```

### `ModuleNotFoundError: No module named 'codetr'`

Ensure you are in the repository root directory and have run `pip install -e .`.

### Missing Dependencies

If a specific library is missing, install it manually:

```bash
!pip install <library_name>
```

### CUDA / PyTorch Issues

Colab and Kaggle come with pre-installed PyTorch. If you face version conflicts:

```bash
# Check current versions
!pip show torch torchvision

# Reinstall if needed (match CUDA version)
# For Colab GPU runtime:
!pip install torch torchvision --upgrade
```

## Quick Test Command

A one-liner to quickly verify everything works after setup:

```bash
%cd /content/co-DETR && pip install -e . -q && python -m pytest tests/test_config.py -v
```
