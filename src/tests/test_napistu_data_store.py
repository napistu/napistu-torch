"""Tests for NapistuDataStore functionality."""

import tempfile
from pathlib import Path

import pytest
import torch

from napistu_torch.constants import NAPISTU_DATA, NAPISTU_DATA_STORE, VERTEX_TENSOR
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
    temp_napistu_data_store, supervised_napistu_data, comprehensive_source_membership
):
    """Test store summary method."""
    # Initially empty
    summary = temp_napistu_data_store.summary()
    assert summary["napistu_data_count"] == 0
    assert summary["vertex_tensors_count"] == 0
    assert summary["napistu_data_names"] == []
    assert summary["vertex_tensor_names"] == []
    assert "store_dir" in summary
    assert "last_modified" in summary

    # Add some data
    temp_napistu_data_store.save_napistu_data(supervised_napistu_data, overwrite=True)
    temp_napistu_data_store.save_vertex_tensor(
        comprehensive_source_membership, name="test_tensor", overwrite=True
    )

    # Check updated summary
    summary = temp_napistu_data_store.summary()
    assert summary["napistu_data_count"] == 1
    assert summary["vertex_tensors_count"] == 1
    assert supervised_napistu_data.name in summary["napistu_data_names"]
    assert "test_tensor" in summary["vertex_tensor_names"]
