# Napistu PyTorch Library

Python package supporting integration of Napistu network graphs and PyTorch NNs

[![PyPI version](https://badge.fury.io/py/napistu_torch.svg)](https://badge.fury.io/py/napistu_torch)
[![CI](https://github.com/napistu/napistu-torch/actions/workflows/ci.yml/badge.svg)](https://github.com/napistu/napistu-torch/actions/workflows/ci.yml)
[![Release](https://github.com/napistu/napistu-torch/actions/workflows/release.yml/badge.svg)](https://github.com/napistu/napistu-torch/actions/workflows/release.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This Python package builds on the [Napistu Python package](https://github.com/napistu/napistu-py) with PyTorch-specific data, model, and results management. As part of the broader [Napistu project](https://github.com/napistu/napistu) this work is intended to create and interrogate genome-scale networks of cellular physiology.

## Setup

Napistu is available on [PyPI](https://pypi.org/project/napistu-torch) so the recommended way to use it is just to pip install with:

```bash
pip install napistu-torch
```

Alternatively, you can clone this repository and perform a local install. e.g., from this directory:

```bash
pip install .
```

### Mac setup

```bash
uv pip install torch==2.8.0
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
uv install 'napistu-torch[dev]'
```

## Documentation
- 🚸 **Project Documentation**: [napistu/wiki](https://github.com/napistu/napistu/wiki)

## Advanced setup

### Wandb

If this is your first time using wandb, you'll need to authenticate:

1. Go to https://wandb.ai/ and create an account
2. Get your API key from https://wandb.ai/authorize
3. Run: wandb login
4. Or set environment variable: export WANDB_API_KEY=your_api_key_here