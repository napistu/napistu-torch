"""Shared fixtures for napistu_torch tests."""

import os

import pandas as pd
import pytest
from napistu import consensus, indices
from napistu.network.constants import NAPISTU_WEIGHTING_STRATEGIES
from napistu.network.net_create import process_napistu_graph
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from napistu_torch.evaluation.pathways import get_comprehensive_source_membership
from napistu_torch.load.constants import (
    ENCODING_MANAGER,
    ENCODINGS,
    SPLITTING_STRATEGIES,
)
from napistu_torch.load.napistu_graphs import (
    augment_napistu_graph,
    construct_supervised_pyg_data,
    construct_unsupervised_pyg_data,
    napistu_graph_to_pyg,
)


@pytest.fixture
def test_data_path():
    """Path to test data directory."""
    return os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture
def pw_index(test_data_path):
    """Create a pathway index for metabolism test data."""
    return indices.PWIndex(os.path.join(test_data_path, "pw_index_metabolism.tsv"))


@pytest.fixture
def sbml_dfs(pw_index):
    """Create a consensus SBML_dfs model from metabolism test data."""

    # Create SBML_dfs dictionary
    sbml_dfs_dict = consensus.construct_sbml_dfs_dict(pw_index)

    # Create consensus model
    return consensus.construct_consensus_model(sbml_dfs_dict, pw_index)


@pytest.fixture
def napistu_graph(sbml_dfs):
    """Create a NapistuGraph from sbml_dfs_metabolism with directed=True and topology weighting."""

    napistu_graph = process_napistu_graph(
        sbml_dfs,
        directed=True,
        weighting_strategy=NAPISTU_WEIGHTING_STRATEGIES.TOPOLOGY,
    )

    return napistu_graph


@pytest.fixture
def valid_encoding_config():
    """Valid encoding configuration without conflicts."""
    return {
        ENCODINGS.CATEGORICAL: {
            ENCODING_MANAGER.COLUMNS: ["node_type", "species_type"],
            ENCODING_MANAGER.TRANSFORMER: OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            ),
        },
        ENCODINGS.NUMERIC: {
            ENCODING_MANAGER.COLUMNS: ["weight", "score"],
            ENCODING_MANAGER.TRANSFORMER: StandardScaler(),
        },
    }


@pytest.fixture
def valid_simple_encoding_config():
    """Valid encoding configuration in simple format (equivalent to valid_encoding_config).

    Note: Uses lists instead of sets to ensure consistent column ordering with valid_encoding_config.
    """
    return {
        ENCODINGS.CATEGORICAL: ["node_type", "species_type"],
        ENCODINGS.NUMERIC: ["weight", "score"],
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


@pytest.fixture
def augmented_napistu_graph(napistu_graph, sbml_dfs):
    """Create a NapistuGraph that has been augmented with SBML_dfs information."""
    # Augment the graph with SBML_dfs information
    augment_napistu_graph(sbml_dfs, napistu_graph, inplace=True)

    return napistu_graph


@pytest.fixture
def napistu_data(augmented_napistu_graph):
    """Create a NapistuData object using the no_mask split strategy."""
    # Convert to NapistuData using no_mask strategy
    return napistu_graph_to_pyg(
        augmented_napistu_graph, splitting_strategy=SPLITTING_STRATEGIES.NO_MASK
    )


@pytest.fixture
def supervised_napistu_data(sbml_dfs, napistu_graph):
    """Create a supervised NapistuData object using default settings."""
    return construct_supervised_pyg_data(sbml_dfs, napistu_graph)


@pytest.fixture
def unsupervised_napistu_data(sbml_dfs, napistu_graph):
    """Create an unsupervised NapistuData object using default settings."""
    return construct_unsupervised_pyg_data(sbml_dfs, napistu_graph)


@pytest.fixture
def comprehensive_source_membership(sbml_dfs, napistu_graph):
    """Create a comprehensive source membership VertexTensor."""
    return get_comprehensive_source_membership(napistu_graph, sbml_dfs)
