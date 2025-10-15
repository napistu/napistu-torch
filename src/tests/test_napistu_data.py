"""
Tests for NapistuData class functionality.
"""

import tempfile
from pathlib import Path

import pytest
import torch

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

        with pytest.raises(RuntimeError):
            NapistuData.load(tmpfile.name)

        # Clean up
        Path(tmpfile.name).unlink()


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
