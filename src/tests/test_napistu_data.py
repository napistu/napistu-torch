"""
Tests for NapistuData class functionality.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch
from napistu.network.constants import (
    NAPISTU_GRAPH,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_NODE_TYPES,
    NAPISTU_GRAPH_VERTICES,
)

from napistu_torch.napistu_data import NapistuData


def test_feature_names(napistu_data):
    """Test that feature names are stored and retrievable."""
    vertex_names = napistu_data.get_vertex_feature_names()
    edge_names = napistu_data.get_edge_feature_names()

    assert vertex_names is not None
    assert edge_names is not None
    assert len(vertex_names) == napistu_data.num_node_features
    assert len(edge_names) == napistu_data.num_edge_features
    assert all(isinstance(name, str) for name in vertex_names)
    assert all(isinstance(name, str) for name in edge_names)


def test_summary(napistu_data):
    """Test the summary method."""
    summary = napistu_data.summary()

    assert isinstance(summary, dict)
    assert "num_nodes" in summary
    assert "num_edges" in summary
    assert "num_node_features" in summary
    assert "num_edge_features" in summary
    assert "has_vertex_feature_names" in summary
    assert "has_edge_feature_names" in summary

    assert summary["num_nodes"] == napistu_data.num_nodes
    assert summary["num_edges"] == napistu_data.num_edges
    assert summary["has_vertex_feature_names"] is True
    assert summary["has_edge_feature_names"] is True


def test_repr(napistu_data):
    """Test the string representation."""
    repr_str = repr(napistu_data)

    assert isinstance(repr_str, str)
    assert "NapistuData" in repr_str
    assert str(napistu_data.num_nodes) in repr_str
    assert str(napistu_data.num_edges) in repr_str


def test_save_load_roundtrip(napistu_data):
    """Test saving and loading NapistuData objects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_napistu_data.pt"

        # Save the data
        napistu_data.save(filepath)
        assert filepath.exists()

        # Load the data (should default to CPU)
        loaded_data = NapistuData.load(filepath)

        # Verify it's a NapistuData object
        assert isinstance(loaded_data, NapistuData)

        # Verify it loads to CPU by default
        assert loaded_data.x.device.type == "cpu"

        # Verify basic properties are preserved
        assert loaded_data.num_nodes == napistu_data.num_nodes
        assert loaded_data.num_edges == napistu_data.num_edges
        assert loaded_data.num_node_features == napistu_data.num_node_features
        assert loaded_data.num_edge_features == napistu_data.num_edge_features

        # Verify tensors are equal
        assert torch.equal(loaded_data.x, napistu_data.x)
        assert torch.equal(loaded_data.edge_index, napistu_data.edge_index)
        assert torch.equal(loaded_data.edge_attr, napistu_data.edge_attr)

        # Verify feature names are preserved
        assert (
            loaded_data.get_vertex_feature_names()
            == napistu_data.get_vertex_feature_names()
        )
        assert (
            loaded_data.get_edge_feature_names()
            == napistu_data.get_edge_feature_names()
        )


def test_load_nonexistent_file():
    """Test loading a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        NapistuData.load("/nonexistent/path.pt")


def test_load_invalid_file():
    """Test loading an invalid file raises appropriate error."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmpfile:
        # Write some invalid data
        tmpfile.write(b"invalid pickle data")
        tmpfile.flush()
        tmpfile_path = tmpfile.name

    # Ensure the file handle is closed before trying to load
    try:
        with pytest.raises(RuntimeError):
            NapistuData.load(tmpfile_path)
    finally:
        # Clean up with retry logic for Windows
        try:
            Path(tmpfile_path).unlink()
        except PermissionError:
            # On Windows, sometimes we need to wait a bit for the file to be released
            import time

            time.sleep(0.1)
            Path(tmpfile_path).unlink()


def test_directory_creation(napistu_data):
    """Test that save creates directories if they don't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a nested path that doesn't exist
        nested_path = Path(tmpdir) / "nested" / "directory" / "test.pt"

        # Save should create the directories
        napistu_data.save(nested_path)
        assert nested_path.exists()

        # Load should work
        loaded_data = NapistuData.load(nested_path)
        assert isinstance(loaded_data, NapistuData)


def test_supervised_data_ordering_consistency(supervised_napistu_data, napistu_graph):
    """Test that data ordering is consistent between NapistuGraph and NapistuData objects.

    This test verifies that when labels are decoded from the NapistuData object,
    they correspond to the correct vertices in the NapistuGraph based on the
    vertex ordering preserved in the NapistuData.ng_vertex_names attribute.
    """
    # Use the new _validate_labels method to test consistency
    supervised_napistu_data._validate_labels(
        napistu_graph, supervised_napistu_data.labeling_manager
    )


def test_vertex_feature_ordering_consistency(edge_masked_napistu_data, napistu_graph):
    """Test that vertex features are properly ordered between NapistuGraph and NapistuData.

    This test verifies that the one-hot encoded node_type features in NapistuData
    correspond to the correct vertices in the NapistuGraph based on the vertex ordering
    preserved in the NapistuData.ng_vertex_names attribute.
    """

    edge_masked_napistu_data._validate_vertex_encoding(
        napistu_graph, NAPISTU_GRAPH_VERTICES.NODE_TYPE
    )


def test_edge_feature_ordering_consistency(edge_masked_napistu_data, napistu_graph):
    """Test that edge features are properly ordered between NapistuGraph and NapistuData.

    This test verifies that the encoded edge features in NapistuData
    correspond to the correct edges in the NapistuGraph based on the edge ordering
    preserved in the NapistuData.ng_edge_names attribute.
    """

    edge_masked_napistu_data._validate_edge_encoding(
        napistu_graph, NAPISTU_GRAPH_EDGES.R_ISREVERSIBLE
    )


def test_unencode_features_node_type(napistu_data, napistu_graph):
    """Test unencode_features method for node_type attribute.

    This test verifies that the unencode_features method can successfully
    unencode the node_type attribute and that the output contains both
    species and reactions nodes.
    """

    # Unencode the node_type attribute from vertices
    unencoded_node_types = napistu_data.unencode_features(
        napistu_graph=napistu_graph,
        attribute_type=NAPISTU_GRAPH.VERTICES,
        attribute=NAPISTU_GRAPH_VERTICES.NODE_TYPE,
    )

    # Verify the output is a pandas Series
    assert isinstance(unencoded_node_types, pd.Series)
    assert unencoded_node_types.name == NAPISTU_GRAPH_VERTICES.NODE_TYPE

    # Verify the output contains both species and reactions
    unique_types = set(unencoded_node_types.dropna().unique())
    assert (
        NAPISTU_GRAPH_NODE_TYPES.SPECIES in unique_types
    ), "Output should contain species nodes"
    assert (
        NAPISTU_GRAPH_NODE_TYPES.REACTION in unique_types
    ), "Output should contain reaction nodes"

    # Verify we have a reasonable number of nodes
    assert len(unencoded_node_types) > 0, "Should have unencoded node types"
    assert len(unencoded_node_types.dropna()) > 0, "Should have non-null node types"
