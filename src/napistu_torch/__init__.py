"""
Napistu Torch - PyTorch-based toolkit for working with Napistu network graphs
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("napistu_torch")
except PackageNotFoundError:
    # package is not installed
    pass
