"""Tests for NapistuDataStore functionality."""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from napistu_torch.constants import (
    ARTIFACT_TYPES,
    NAPISTU_DATA,
    NAPISTU_DATA_STORE,
    VERTEX_TENSOR,
)
from napistu_torch.load.artifacts import ArtifactDefinition
from napistu_torch.load.constants import (
    DEFAULT_ARTIFACTS_NAMES,
    SPLITTING_STRATEGIES,
)
from napistu_torch.load.napistu_graphs import construct_unsupervised_pyg_data
from napistu_torch.napistu_data_store import NapistuDataStore


@pytest.fixture(scope="function")
def temp_napistu_data_store():
    """Create a temporary NapistuDataStore for testing.

    Mocks the sbml_dfs_path and napistu_graph_path since they won't be used
    in the save/load tests.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock files for the required paths
        sbml_dfs_path = Path(temp_dir) / "mock_sbml_dfs.pkl"
        napistu_graph_path = Path(temp_dir) / "mock_napistu_graph.pkl"

        # Create empty files to satisfy the file existence check
        sbml_dfs_path.touch()
        napistu_graph_path.touch()

        # Create the store
        store = NapistuDataStore.create(
            store_dir=temp_dir,
            sbml_dfs_path=sbml_dfs_path,
            napistu_graph_path=napistu_graph_path,
            copy_to_store=False,
        )

        yield store


@pytest.fixture(scope="function")
def temp_napistu_data_store_with_real_data(sbml_dfs, napistu_graph):
    """Create a temporary NapistuDataStore with real SBML_dfs and NapistuGraph data.

    This fixture is used for testing ensure_artifacts which needs actual data.
    """
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

        yield store


@pytest.fixture(scope="function")
def test_dataframe_with_nans():
    """Create a test pandas DataFrame with NaN values for testing."""
    return pd.DataFrame(
        {
            "gene_id": ["GENE1", "GENE2", "GENE3", "GENE4"],
            "expression": [1.5, 2.3, np.nan, 3.1],
            "p_value": [0.01, pd.NA, 0.1, 0.001],
            "log_fold_change": [0.5, 1.2, -0.3, pd.NA],
            "significant": [True, True, False, True],
        }
    )


def _verify_napistu_data_equality(original, loaded):
    """Helper function to verify NapistuData objects are equal after round-trip."""
    # Check basic attributes
    assert original.name == loaded.name
    assert original.splitting_strategy == loaded.splitting_strategy

    # Check tensors
    assert torch.equal(original.x, loaded.x)
    assert torch.equal(original.edge_index, loaded.edge_index)
    assert torch.equal(original.edge_attr, loaded.edge_attr)

    assert torch.equal(original.edge_weight, loaded.edge_weight)
    assert original.ng_vertex_names.equals(loaded.ng_vertex_names)
    assert original.ng_edge_names.equals(loaded.ng_edge_names)
    assert original.vertex_feature_names == loaded.vertex_feature_names
    assert original.edge_feature_names == loaded.edge_feature_names

    # Check optional tensors
    if original.y is not None:
        assert torch.equal(original.y, loaded.y)
        # Check labeling manager if it exists
        if (
            hasattr(original, NAPISTU_DATA.LABELING_MANAGER)
            and original.labeling_manager is not None
        ):
            assert (
                original.labeling_manager.to_dict() == loaded.labeling_manager.to_dict()
            )
    else:
        assert loaded.y is None
        # Check that labeling_manager is None if it exists
        if hasattr(loaded, NAPISTU_DATA.LABELING_MANAGER):
            assert loaded.labeling_manager is None


def test_supervised_napistu_data_roundtrip(
    temp_napistu_data_store, supervised_napistu_data
):
    """Test save/load round-trip for supervised NapistuData."""

    # Save the supervised data
    temp_napistu_data_store.save_napistu_data(supervised_napistu_data, overwrite=True)

    # Load it back
    loaded_data = temp_napistu_data_store.load_napistu_data(
        supervised_napistu_data.name
    )

    # Verify the data integrity
    _verify_napistu_data_equality(supervised_napistu_data, loaded_data)

    # Verify registry entry
    registry_entry = temp_napistu_data_store.registry[NAPISTU_DATA_STORE.NAPISTU_DATA][
        supervised_napistu_data.name
    ]
    assert registry_entry[NAPISTU_DATA.NAME] == supervised_napistu_data.name
    assert (
        registry_entry[NAPISTU_DATA.SPLITTING_STRATEGY]
        == supervised_napistu_data.splitting_strategy
    )
    assert registry_entry[NAPISTU_DATA.LABELING_MANAGER] is not None


def test_unsupervised_napistu_data_roundtrip(
    temp_napistu_data_store, unsupervised_napistu_data
):
    """Test save/load round-trip for unsupervised NapistuData."""
    napistu_data = unsupervised_napistu_data

    # Save the unsupervised data
    temp_napistu_data_store.save_napistu_data(napistu_data, overwrite=True)

    # Load it back
    loaded_data = temp_napistu_data_store.load_napistu_data(napistu_data.name)

    # Verify the data integrity
    _verify_napistu_data_equality(napistu_data, loaded_data)

    # Verify registry entry
    registry_entry = temp_napistu_data_store.registry[NAPISTU_DATA_STORE.NAPISTU_DATA][
        napistu_data.name
    ]
    assert registry_entry[NAPISTU_DATA.NAME] == napistu_data.name
    assert (
        registry_entry[NAPISTU_DATA.SPLITTING_STRATEGY]
        == napistu_data.splitting_strategy
    )
    assert (
        registry_entry[NAPISTU_DATA.LABELING_MANAGER] is None
    )  # Should be None for unsupervised


def test_overwrite_behavior(temp_napistu_data_store, supervised_napistu_data):
    """Test overwrite behavior when saving NapistuData with same name."""

    # Save first time
    temp_napistu_data_store.save_napistu_data(supervised_napistu_data, overwrite=True)

    # Try to save again without overwrite - should raise FileExistsError
    with pytest.raises(FileExistsError, match="already exists in registry"):
        temp_napistu_data_store.save_napistu_data(
            supervised_napistu_data, overwrite=False
        )

    # Save with overwrite=True - should succeed
    temp_napistu_data_store.save_napistu_data(supervised_napistu_data, overwrite=True)

    # Verify it can still be loaded
    loaded_data = temp_napistu_data_store.load_napistu_data(
        supervised_napistu_data.name
    )
    _verify_napistu_data_equality(supervised_napistu_data, loaded_data)


def test_load_nonexistent_napistu_data(temp_napistu_data_store):
    """Test loading a NapistuData that doesn't exist in registry."""
    with pytest.raises(KeyError, match="not found in registry"):
        temp_napistu_data_store.load_napistu_data("nonexistent_name")


def _verify_vertex_tensor_equality(original, loaded):
    """Helper function to verify VertexTensor objects are equal after round-trip."""
    # Check basic attributes
    assert original.name == loaded.name
    assert original.description == loaded.description
    assert original.feature_names == loaded.feature_names

    # Check tensors
    assert torch.equal(original.data, loaded.data)

    # Check vertex names
    assert original.vertex_names.equals(loaded.vertex_names)


def test_vertex_tensor_roundtrip(
    temp_napistu_data_store, comprehensive_source_membership
):
    """Test save/load round-trip for VertexTensor."""
    vertex_tensor = comprehensive_source_membership
    tensor_name = "test_comprehensive_membership"

    # Save the VertexTensor
    temp_napistu_data_store.save_vertex_tensor(
        vertex_tensor, tensor_name, overwrite=True
    )

    # Load it back
    loaded_tensor = temp_napistu_data_store.load_vertex_tensor(tensor_name)

    # Verify the data integrity
    _verify_vertex_tensor_equality(vertex_tensor, loaded_tensor)

    # Verify registry entry
    registry_entry = temp_napistu_data_store.registry[
        NAPISTU_DATA_STORE.VERTEX_TENSORS
    ][tensor_name]
    assert registry_entry[VERTEX_TENSOR.NAME] == vertex_tensor.name
    assert registry_entry[VERTEX_TENSOR.DESCRIPTION] == vertex_tensor.description


def test_vertex_tensor_overwrite_behavior(
    temp_napistu_data_store, comprehensive_source_membership
):
    """Test overwrite behavior when saving VertexTensor with same name."""
    vertex_tensor = comprehensive_source_membership
    tensor_name = "test_overwrite_membership"

    # Save first time
    temp_napistu_data_store.save_vertex_tensor(
        vertex_tensor, tensor_name, overwrite=True
    )

    # Try to save again without overwrite - should raise FileExistsError
    with pytest.raises(FileExistsError, match="already exists in registry"):
        temp_napistu_data_store.save_vertex_tensor(
            vertex_tensor, tensor_name, overwrite=False
        )

    # Save with overwrite=True - should succeed
    temp_napistu_data_store.save_vertex_tensor(
        vertex_tensor, tensor_name, overwrite=True
    )

    # Verify it can still be loaded
    loaded_tensor = temp_napistu_data_store.load_vertex_tensor(tensor_name)
    _verify_vertex_tensor_equality(vertex_tensor, loaded_tensor)


def test_load_nonexistent_vertex_tensor(temp_napistu_data_store):
    """Test loading a VertexTensor that doesn't exist in registry."""
    with pytest.raises(KeyError, match="not found in registry"):
        temp_napistu_data_store.load_vertex_tensor("nonexistent_tensor")


def test_list_napistu_datas(
    temp_napistu_data_store, supervised_napistu_data, unsupervised_napistu_data
):
    """Test listing NapistuData objects in the store."""
    # Initially empty
    assert temp_napistu_data_store.list_napistu_datas() == []

    # Save supervised data
    temp_napistu_data_store.save_napistu_data(supervised_napistu_data, overwrite=True)
    assert supervised_napistu_data.name in temp_napistu_data_store.list_napistu_datas()

    # Save unsupervised data
    temp_napistu_data_store.save_napistu_data(unsupervised_napistu_data, overwrite=True)
    napistu_data_names = temp_napistu_data_store.list_napistu_datas()
    assert supervised_napistu_data.name in napistu_data_names
    assert unsupervised_napistu_data.name in napistu_data_names
    assert len(napistu_data_names) == 2


def test_list_vertex_tensors(temp_napistu_data_store, comprehensive_source_membership):
    """Test listing VertexTensor objects in the store."""
    # Initially empty
    assert temp_napistu_data_store.list_vertex_tensors() == []

    # Save vertex tensor
    tensor_name = "test_membership"
    temp_napistu_data_store.save_vertex_tensor(
        comprehensive_source_membership, name=tensor_name, overwrite=True
    )

    vertex_tensor_names = temp_napistu_data_store.list_vertex_tensors()
    assert tensor_name in vertex_tensor_names
    assert len(vertex_tensor_names) == 1


def test_summary(
    temp_napistu_data_store,
    supervised_napistu_data,
    comprehensive_source_membership,
    test_dataframe_with_nans,
):
    """Test store summary method."""
    # Initially empty
    summary = temp_napistu_data_store.summary()
    assert summary["napistu_data_count"] == 0
    assert summary["vertex_tensors_count"] == 0
    assert summary["pandas_dfs_count"] == 0
    assert summary["napistu_data_names"] == []
    assert summary["vertex_tensor_names"] == []
    assert summary["pandas_df_names"] == []
    assert "store_dir" in summary
    assert "last_modified" in summary

    # Add some data
    temp_napistu_data_store.save_napistu_data(supervised_napistu_data, overwrite=True)
    temp_napistu_data_store.save_vertex_tensor(
        comprehensive_source_membership, name="test_tensor", overwrite=True
    )
    temp_napistu_data_store.save_pandas_df(
        test_dataframe_with_nans, "test_df", overwrite=True
    )

    # Check updated summary
    summary = temp_napistu_data_store.summary()
    assert summary["napistu_data_count"] == 1
    assert summary["vertex_tensors_count"] == 1
    assert summary["pandas_dfs_count"] == 1
    assert supervised_napistu_data.name in summary["napistu_data_names"]
    assert "test_tensor" in summary["vertex_tensor_names"]
    assert "test_df" in summary["pandas_df_names"]


def test_pandas_dataframe_io(temp_napistu_data_store, test_dataframe_with_nans):
    """Test pandas DataFrame save/load round-trip, listing, and error handling."""
    df = test_dataframe_with_nans
    df_name = "test_nan_data"

    # Save DataFrame
    temp_napistu_data_store.save_pandas_df(df, df_name, overwrite=True)

    # Load and verify data integrity
    loaded_df = temp_napistu_data_store.load_pandas_df(df_name)

    # Normalize null values to avoid FutureWarning about mismatched <NA> and nan
    df_normalized = df.replace({pd.NA: np.nan})
    loaded_df_normalized = loaded_df.replace({pd.NA: np.nan})

    pd.testing.assert_frame_equal(
        df_normalized, loaded_df_normalized, check_dtype=False
    )

    # Test listing
    assert df_name in temp_napistu_data_store.list_pandas_dfs()

    # Test loading missing entry
    with pytest.raises(KeyError, match="not found in registry"):
        temp_napistu_data_store.load_pandas_df("missing_df")


# Tests for ensure_artifacts
@pytest.mark.skip_on_windows
def test_ensure_artifacts_comprehensive(
    temp_napistu_data_store_with_real_data,
    napistu_graph,
):
    """Comprehensive test for ensure_artifacts functionality.

    Tests:
    1. Creating missing NapistuData artifact (unsupervised)
    2. Creating multiple missing artifacts in one call (edge_prediction, comprehensive_pathway_memberships, edge_strata_by_node_species_type)
    3. Skipping existing artifacts when overwrite=False
    4. Overwriting existing artifacts when overwrite=True
    """

    store = temp_napistu_data_store_with_real_data

    # Test 1: Create single NapistuData artifact (unsupervised)
    store.ensure_artifacts(["unsupervised"], overwrite=False)
    assert "unsupervised" in store.list_napistu_datas()

    # Load and verify it's usable
    unsupervised_data = store.load_napistu_data("unsupervised")
    assert unsupervised_data.splitting_strategy == SPLITTING_STRATEGIES.NO_MASK
    assert unsupervised_data.x.shape[0] == napistu_graph.vcount()

    # Test 2 & 3 & 4: Create multiple artifacts of different types
    store.ensure_artifacts(
        [
            DEFAULT_ARTIFACTS_NAMES.EDGE_PREDICTION,
            DEFAULT_ARTIFACTS_NAMES.COMPREHENSIVE_PATHWAY_MEMBERSHIPS,
            DEFAULT_ARTIFACTS_NAMES.EDGE_STRATA_BY_NODE_SPECIES_TYPE,
        ],
        overwrite=False,
    )

    # Verify all were created
    assert DEFAULT_ARTIFACTS_NAMES.EDGE_PREDICTION in store.list_napistu_datas()
    assert (
        DEFAULT_ARTIFACTS_NAMES.COMPREHENSIVE_PATHWAY_MEMBERSHIPS
        in store.list_vertex_tensors()
    )
    assert (
        DEFAULT_ARTIFACTS_NAMES.EDGE_STRATA_BY_NODE_SPECIES_TYPE
        in store.list_pandas_dfs()
    )

    # Test 5: Loading and verifying data integrity
    edge_pred = store.load_napistu_data(DEFAULT_ARTIFACTS_NAMES.EDGE_PREDICTION)
    assert edge_pred.splitting_strategy == SPLITTING_STRATEGIES.EDGE_MASK

    pathway_membership = store.load_vertex_tensor(
        DEFAULT_ARTIFACTS_NAMES.COMPREHENSIVE_PATHWAY_MEMBERSHIPS
    )
    assert pathway_membership.data.shape[0] == napistu_graph.vcount()

    edge_strata = store.load_pandas_df(
        DEFAULT_ARTIFACTS_NAMES.EDGE_STRATA_BY_NODE_SPECIES_TYPE
    )
    assert len(edge_strata) == napistu_graph.ecount()

    # Test 6: Skipping existing artifacts (should not recreate)
    initial_napistu_data_count = len(store.list_napistu_datas())
    store.ensure_artifacts([DEFAULT_ARTIFACTS_NAMES.UNSUPERVISED], overwrite=False)
    assert len(store.list_napistu_datas()) == initial_napistu_data_count

    # Test 7: Overwriting existing artifact
    store.ensure_artifacts([DEFAULT_ARTIFACTS_NAMES.UNSUPERVISED], overwrite=True)
    # Should still have same count but artifact was recreated
    assert len(store.list_napistu_datas()) == initial_napistu_data_count


@pytest.mark.skip_on_windows
def test_ensure_artifacts_error_handling(temp_napistu_data_store_with_real_data):
    """Test error handling in ensure_artifacts.

    Tests:
    1. Raises KeyError on unknown artifact name
    2. Raises ValueError on invalid artifact registry
    """
    store = temp_napistu_data_store_with_real_data

    # Test 1: Unknown artifact name
    with pytest.raises(KeyError, match="Cannot create artifacts not in registry"):
        store.ensure_artifacts(["unknown_artifact_name"], overwrite=False)

    # Test 2: Invalid registry (empty dict)

    with pytest.raises(ValueError, match="cannot be empty"):
        store.ensure_artifacts(
            [DEFAULT_ARTIFACTS_NAMES.UNSUPERVISED],
            artifact_registry={},
            overwrite=False,
        )


@pytest.mark.skip_on_windows
def test_ensure_artifacts_with_custom_registry(
    temp_napistu_data_store_with_real_data, sbml_dfs, napistu_graph
):
    """Test ensure_artifacts with a custom artifact registry."""
    store = temp_napistu_data_store_with_real_data

    # Create a minimal custom registry with just one artifact
    def custom_unsupervised(sbml_dfs, napistu_graph):
        return construct_unsupervised_pyg_data(
            sbml_dfs, napistu_graph, splitting_strategy=SPLITTING_STRATEGIES.NO_MASK
        )

    custom_registry = {
        "custom_unsupervised": ArtifactDefinition(
            name="custom_unsupervised",
            artifact_type=ARTIFACT_TYPES.NAPISTU_DATA,
            creation_func=custom_unsupervised,
            description="Custom unsupervised data",
        )
    }

    # Use custom registry
    store.ensure_artifacts(
        ["custom_unsupervised"], artifact_registry=custom_registry, overwrite=False
    )

    # Verify it was created
    assert "custom_unsupervised" in store.list_napistu_datas()

    # Verify it can be loaded
    custom_data = store.load_napistu_data("custom_unsupervised")
    assert custom_data.splitting_strategy == SPLITTING_STRATEGIES.NO_MASK


def test_validate_store(
    temp_napistu_data_store,
    supervised_napistu_data,
    comprehensive_source_membership,
    test_dataframe_with_nans,
    caplog,
):
    """Test store validation method."""
    store = temp_napistu_data_store

    # Empty store should validate fine
    store.validate()

    # Add some artifacts with different names - should still validate
    store.save_napistu_data(supervised_napistu_data)
    store.save_vertex_tensor(comprehensive_source_membership, name="test_tensor")
    store.save_pandas_df(test_dataframe_with_nans, "test_df")

    store.validate()  # Should pass

    # Test name conflict - should just log a warning, not raise an error
    store.save_vertex_tensor(comprehensive_source_membership, name="test_df")

    # Should log a warning but not raise an error - capture logger output
    with caplog.at_level(logging.WARNING):
        store.validate()
        assert "Duplicate artifact names found" in caplog.text


def test_validate_artifact_name(temp_napistu_data_store, supervised_napistu_data):
    """Test artifact name validation method."""
    store = temp_napistu_data_store

    # Test 1: Valid artifact name from registry (not in store)
    store.validate_artifact_name(DEFAULT_ARTIFACTS_NAMES.UNSUPERVISED)  # Should pass

    # Test 2: Valid artifact name with required type
    store.validate_artifact_name(
        DEFAULT_ARTIFACTS_NAMES.UNSUPERVISED, required_type=ARTIFACT_TYPES.NAPISTU_DATA
    )  # Should pass

    # Test 3: Invalid artifact name (not in registry)
    with pytest.raises(ValueError, match="was not found"):
        store.validate_artifact_name("nonexistent_artifact")

    # Test 4: Wrong type requirement
    with pytest.raises(KeyError, match="is a napistu_data, not a vertex_tensor"):
        store.validate_artifact_name(
            DEFAULT_ARTIFACTS_NAMES.UNSUPERVISED,
            required_type=ARTIFACT_TYPES.VERTEX_TENSOR,
        )

    # Test 5: Artifact already in store
    store.save_napistu_data(supervised_napistu_data, overwrite=True)
    store.validate_artifact_name(supervised_napistu_data.name)  # Should pass

    # Test 6: Artifact in store but wrong type requirement
    with pytest.raises(KeyError, match="already exists in store but is not of type"):
        store.validate_artifact_name(
            supervised_napistu_data.name, required_type=ARTIFACT_TYPES.VERTEX_TENSOR
        )


def test_list_artifacts(
    temp_napistu_data_store,
    supervised_napistu_data,
    unsupervised_napistu_data,
    comprehensive_source_membership,
    test_dataframe_with_nans,
):
    """Test list_artifacts method."""
    store = temp_napistu_data_store

    # Initially empty
    all_artifacts = store.list_artifacts()
    assert all_artifacts == set()

    # List specific types when empty
    assert store.list_artifacts(ARTIFACT_TYPES.NAPISTU_DATA) == []
    assert store.list_artifacts(ARTIFACT_TYPES.VERTEX_TENSOR) == []
    assert store.list_artifacts(ARTIFACT_TYPES.PANDAS_DFS) == []

    # Add NapistuData
    store.save_napistu_data(supervised_napistu_data)
    assert supervised_napistu_data.name in store.list_artifacts(
        ARTIFACT_TYPES.NAPISTU_DATA
    )
    assert supervised_napistu_data.name in store.list_artifacts()  # All artifacts

    # Add another NapistuData
    store.save_napistu_data(unsupervised_napistu_data)
    napistu_datas = store.list_artifacts(ARTIFACT_TYPES.NAPISTU_DATA)
    assert supervised_napistu_data.name in napistu_datas
    assert unsupervised_napistu_data.name in napistu_datas
    assert len(napistu_datas) == 2

    # Add VertexTensor
    store.save_vertex_tensor(comprehensive_source_membership, name="test_tensor")
    vertex_tensors = store.list_artifacts(ARTIFACT_TYPES.VERTEX_TENSOR)
    assert "test_tensor" in vertex_tensors
    assert len(vertex_tensors) == 1

    # Add pandas DataFrame
    store.save_pandas_df(test_dataframe_with_nans, "test_df")
    pandas_dfs = store.list_artifacts(ARTIFACT_TYPES.PANDAS_DFS)
    assert "test_df" in pandas_dfs
    assert len(pandas_dfs) == 1

    # Check all artifacts together
    all_artifacts = store.list_artifacts()
    assert (
        len(all_artifacts) == 4
    )  # 2 NapistuData + 1 VertexTensor + 1 pandas DataFrame
    assert supervised_napistu_data.name in all_artifacts
    assert unsupervised_napistu_data.name in all_artifacts
    assert "test_tensor" in all_artifacts
    assert "test_df" in all_artifacts

    # Test invalid artifact type
    with pytest.raises(ValueError, match="Invalid artifact type"):
        store.list_artifacts("invalid_type")
