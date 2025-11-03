import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "Napistu-Torch Python library"
copyright = "2025, Sean Hackett"
author = "Sean Hackett"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.autosummary",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "private-members": True,
    "undoc-members": False,
    "special-members": "__init__",
    "show-inheritance": True,
}

autodoc_member_order = "groupwise"

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "torch_sparse",
    "torch_scatter",
    "pytorch_lightning",
    "wandb",
]
