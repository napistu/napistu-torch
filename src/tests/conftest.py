"""Shared fixtures for napistu_torch tests."""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from napistu import consensus, indices
from napistu.network.constants import NAPISTU_WEIGHTING_STRATEGIES
from napistu.network.net_create import process_napistu_graph
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from napistu_torch.configs import (
    DataConfig,
    ExperimentConfig,
)
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

# Suppress napistu logging during tests to reduce noise
logging.getLogger("napistu").setLevel(logging.ERROR)
logging.getLogger("napistu.consensus").setLevel(logging.ERROR)
logging.getLogger("napistu.network").setLevel(logging.ERROR)
logging.getLogger("napistu.ingestion").setLevel(logging.ERROR)
logging.getLogger("napistu.sbml_dfs_core").setLevel(logging.ERROR)
logging.getLogger("napistu.sbml_dfs_utils").setLevel(logging.ERROR)
logging.getLogger("napistu.utils").setLevel(logging.ERROR)


# Define custom markers for platforms
def pytest_configure(config):
    config.addinivalue_line("markers", "skip_on_windows: mark test to skip on Windows")
    config.addinivalue_line("markers", "skip_on_macos: mark test to skip on macOS")
    config.addinivalue_line(
        "markers", "unix_only: mark test to run only on Unix/Linux systems"
    )


# Define platform conditions
is_windows = sys.platform == "win32"
is_macos = sys.platform == "darwin"
is_unix = not (is_windows or is_macos)


# Apply skipping based on platform
def pytest_runtest_setup(item):
    # Skip tests marked to be skipped on Windows
    if is_windows and any(
        mark.name == "skip_on_windows" for mark in item.iter_markers()
    ):
        pytest.skip("Test skipped on Windows")

    # Skip tests marked to be skipped on macOS
    if is_macos and any(mark.name == "skip_on_macos" for mark in item.iter_markers()):
        pytest.skip("Test skipped on macOS")

    # Skip tests that should run only on Unix
    if not is_unix and any(mark.name == "unix_only" for mark in item.iter_markers()):
        pytest.skip("Test runs only on Unix systems")


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
    return augment_napistu_graph(sbml_dfs, napistu_graph, inplace=False)


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
def edge_masked_napistu_data(sbml_dfs, napistu_graph):
    """Create a NapistuData object with train/val/test masks for testing."""
    return construct_unsupervised_pyg_data(
        sbml_dfs, napistu_graph, splitting_strategy=SPLITTING_STRATEGIES.EDGE_MASK
    )


@pytest.fixture
def experiment_config():
    """Create a basic experiment config for testing."""
    return ExperimentConfig(
        name="test_experiment",
        seed=42,
        deterministic=True,
        fast_dev_run=True,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        data=DataConfig(
            sbml_dfs_path=Path("stub_sbml.pkl"),
            napistu_graph_path=Path("stub_graph.pkl"),
            napistu_data_name="edge_prediction",
        ),
    )


@pytest.fixture
def data_config():
    """Create a basic data config for testing with side-loaded data."""

    return DataConfig(
        name="test_data",
        sbml_dfs_path=Path("test_sbml.pkl"),
        napistu_graph_path=Path("test_graph.pkl"),
        napistu_data_name="unsupervised",
        other_artifacts=[],  # No other artifacts when side-loading
    )


@pytest.fixture
def comprehensive_source_membership(sbml_dfs, napistu_graph):
    """Create a comprehensive source membership VertexTensor."""
    return get_comprehensive_source_membership(napistu_graph, sbml_dfs)


@pytest.fixture
def temp_napistu_data_store_with_edge_data(
    sbml_dfs, napistu_graph, edge_masked_napistu_data
):
    """Create a temporary NapistuDataStore with edge_masked_napistu_data saved to it."""
    import tempfile

    from napistu_torch.napistu_data_store import NapistuDataStore

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create real pickle files
        sbml_dfs_path = Path(temp_dir) / "sbml_dfs.pkl"
        napistu_graph_path = Path(temp_dir) / "napistu_graph.pkl"

        sbml_dfs.to_pickle(sbml_dfs_path)
        napistu_graph.to_pickle(napistu_graph_path)

        # Create the store
        store = NapistuDataStore.create(
            store_dir=temp_dir,
            sbml_dfs_path=sbml_dfs_path,
            napistu_graph_path=napistu_graph_path,
            copy_to_store=False,
        )

        # Save the edge_masked_napistu_data to the store
        store.save_napistu_data(
            edge_masked_napistu_data, name="edge_prediction", overwrite=True
        )

        yield store


@pytest.fixture
def temp_data_config_with_store(temp_napistu_data_store_with_edge_data):
    """Create a DataConfig that points to the temporary store."""
    store = temp_napistu_data_store_with_edge_data

    return DataConfig(
        name="test_data_with_store",
        store_dir=store.store_dir,
        sbml_dfs_path=store.sbml_dfs_path,
        napistu_graph_path=store.napistu_graph_path,
        napistu_data_name="edge_prediction",
        other_artifacts=[
            "unsupervised"
        ],  # This fixture has a real store, so it can have other artifacts
    )
