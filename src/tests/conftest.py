"""Shared fixtures for napistu_torch tests."""

import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from napistu_torch.load.constants import ENCODING_MANAGER


@pytest.fixture
def valid_encoding_config():
    """Valid encoding configuration without conflicts."""
    return {
        "categorical": {
            ENCODING_MANAGER.COLUMNS: ["node_type", "species_type"],
            ENCODING_MANAGER.TRANSFORMER: OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            ),
        },
        "numerical": {
            ENCODING_MANAGER.COLUMNS: ["weight", "score"],
            ENCODING_MANAGER.TRANSFORMER: StandardScaler(),
        },
    }


@pytest.fixture
def invalid_encoding_config():
    """Invalid encoding configuration with column conflicts."""
    return {
        "cat": {
            ENCODING_MANAGER.COLUMNS: ["node_type", "weight"],
            ENCODING_MANAGER.TRANSFORMER: OneHotEncoder(),
        },
        "num": {
            ENCODING_MANAGER.COLUMNS: ["weight", "score"],  # 'weight' conflict
            ENCODING_MANAGER.TRANSFORMER: StandardScaler(),
        },
    }


@pytest.fixture
def override_encoding_config():
    """Override encoding configuration for composition tests."""
    return {
        "categorical": {
            ENCODING_MANAGER.COLUMNS: [
                "node_type",
                "species_type",  # Use existing column instead of "new_col"
            ],  # node_type conflicts with base
            ENCODING_MANAGER.TRANSFORMER: OneHotEncoder(
                sparse_output=False
            ),  # Different transformer than base
        },
        "embeddings": {
            ENCODING_MANAGER.COLUMNS: ["source_col"],
            ENCODING_MANAGER.TRANSFORMER: OneHotEncoder(
                sparse_output=False
            ),  # Use real transformer
        },
    }


@pytest.fixture
def simple_raw_graph_df():
    """Sample raw graph DataFrame for testing transformations."""
    return pd.DataFrame(
        {
            "node_type": ["species", "reaction", "species", "reaction"],
            "species_type": ["gene", None, "metabolite", None],
            "weight": [1.0, 2.0, 3.0, 4.0],
            "score": [0.1, 0.2, 0.3, 0.4],
            "source_col": ["src1", "src2", "src3", "src4"],
        }
    )
