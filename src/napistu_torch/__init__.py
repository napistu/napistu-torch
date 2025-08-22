"""
Napistu Torch - PyTorch-based toolkit for working with Napistu network graphs
"""

import configparser
from pathlib import Path


def _get_version():
    """Get version from setup.cfg"""
    setup_cfg_path = Path(__file__).parent.parent.parent / "setup.cfg"
    if setup_cfg_path.exists():
        config = configparser.ConfigParser()
        config.read(setup_cfg_path)
        return config.get("metadata", "version", fallback="0.1.0")
    return "0.1.0"


__version__ = _get_version()
