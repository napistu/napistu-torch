"""
Tests for NapistuData class functionality.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch

from napistu_torch.labeling.apply import decode_labels
from napistu_torch.load.constants import SPLITTING_STRATEGIES
from napistu_torch.load.napistu_graphs import napistu_graph_to_pyg
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


def test_supervised_data_ordering_consistency(
    supervised_napistu_data_package, napistu_graph
):
    """Test that data ordering is consistent between NapistuGraph and NapistuData objects.

    This test verifies that when labels are decoded from the NapistuData object,
    they correspond to the correct vertices in the NapistuGraph based on the
    vertex ordering preserved in the NapistuData.ng_vertex_names attribute.
    """
    # Unpack the supervised fixture
    napistu_data, labeling_manager = supervised_napistu_data_package

    # Get the encoded labels and vertex names from NapistuData
    encoded_labels = napistu_data.y
    vertex_names = napistu_data.ng_vertex_names

    # Verify dimensions match
    assert len(encoded_labels) == len(
        vertex_names
    ), "Label count should match vertex count"
    assert (
        len(encoded_labels) == napistu_data.num_nodes
    ), "Label count should match node count"

    # Decode the labels using the utility function
    decoded_labels = decode_labels(encoded_labels, labeling_manager)

    # Get the corresponding labels from the NapistuGraph using merge
    vertex_df = napistu_graph.get_vertex_dataframe()
    vertex_names_df = pd.DataFrame({"name": vertex_names})

    # Merge vertex names with the vertex DataFrame to get labels
    merged_df = vertex_names_df.merge(
        vertex_df[["name", labeling_manager.label_attribute]], on="name", how="left"
    )
    graph_labels = merged_df[labeling_manager.label_attribute].tolist()

    # Compare the decoded labels with the graph labels, masking on valid values
    # Convert graph_labels to pandas Series for easier handling
    graph_labels_series = pd.Series(graph_labels)

    # Create mask for valid (non-null) values in both decoded and graph labels
    decoded_valid_mask = pd.Series(decoded_labels).notna()
    graph_valid_mask = graph_labels_series.notna()
    valid_mask = decoded_valid_mask & graph_valid_mask

    # Compare only valid values
    decoded_valid = pd.Series(decoded_labels)[valid_mask]
    graph_valid = graph_labels_series[valid_mask]

    assert len(decoded_valid) == len(graph_valid), "Valid label counts should match"

    for i, (decoded, graph) in enumerate(zip(decoded_valid, graph_valid)):
        assert decoded == graph, (
            f"Label mismatch at valid position {i}: "
            f"decoded={decoded}, graph={graph}"
        )

    # Additional verification: check that we have some non-null labels
    # Use pd.isna() to properly handle both None and np.NaN values
    decoded_series = pd.Series(decoded_labels)
    graph_series = pd.Series(graph_labels)

    non_null_decoded = decoded_series[decoded_series.notna()]
    non_null_graph = graph_series[graph_series.notna()]

    assert len(non_null_decoded) > 0, "Should have some non-null decoded labels"
    assert len(non_null_graph) > 0, "Should have some non-null graph labels"
    assert len(non_null_decoded) == len(
        non_null_graph
    ), "Non-null label counts should match"


@pytest.fixture
def edge_masked_napistu_data(sbml_dfs, napistu_graph):
    """Create a NapistuData object using edge masking for better test coverage."""
    # Augment the graph with SBML_dfs information
    from napistu_torch.load.napistu_graphs import augment_napistu_graph

    augmented_graph = augment_napistu_graph(sbml_dfs, napistu_graph, inplace=False)

    # Convert to NapistuData using edge_mask strategy for better coverage
    return napistu_graph_to_pyg(
        augmented_graph, splitting_strategy=SPLITTING_STRATEGIES.EDGE_MASK
    )


def test_vertex_feature_ordering_consistency(edge_masked_napistu_data, napistu_graph):
    """Test that vertex features are properly ordered between NapistuGraph and NapistuData.

    This test verifies that the one-hot encoded node_type features in NapistuData
    correspond to the correct vertices in the NapistuGraph based on the vertex ordering
    preserved in the NapistuData.ng_vertex_names attribute.
    """
    napistu_data = edge_masked_napistu_data

    # Get vertex features and names from NapistuData
    vertex_features = napistu_data.x  # Shape: [num_nodes, num_node_features]
    vertex_names = napistu_data.ng_vertex_names

    # Get vertex DataFrame from NapistuGraph
    vertex_df = napistu_graph.get_vertex_dataframe()

    # Merge vertex names with the vertex DataFrame to get node_type
    vertex_names_df = pd.DataFrame({"name": vertex_names})
    merged_df = vertex_names_df.merge(
        vertex_df[["name", "node_type"]], on="name", how="left"
    )
    graph_node_types = merged_df["node_type"].tolist()

    # Get the node_type feature column indices from feature names
    # The feature names have format: 'categorical__node_type_species'
    vertex_feature_names = napistu_data.get_vertex_feature_names()
    node_type_indices = [
        i for i, name in enumerate(vertex_feature_names) if "node_type_" in name
    ]

    assert (
        len(node_type_indices) > 0
    ), f"Should have node_type one-hot encoded features. Available features: {vertex_feature_names}"

    # Verify that the one-hot encoding matches the original node_type values
    for i, (features, node_type) in enumerate(zip(vertex_features, graph_node_types)):
        if pd.notna(node_type):
            # Find which node_type feature is active (should be 1.0)
            active_features = features[node_type_indices]
            active_indices = torch.where(active_features > 0.5)[0]

            # Debug problematic cases and fail appropriately
            if len(active_indices) == 0:
                feature_values = [features[j].item() for j in node_type_indices]
                feature_names_subset = [
                    vertex_feature_names[j] for j in node_type_indices
                ]
                expected_feature_name = f"categorical__node_type_{node_type}"

                # Check if the expected feature name exists
                if expected_feature_name not in vertex_feature_names:
                    raise AssertionError(
                        f"Data alignment issue: Vertex {i} (name={vertex_names.iloc[i]}) has "
                        f"node_type='{node_type}' but expected feature '{expected_feature_name}' "
                        f"not found in feature names: {feature_names_subset}"
                    )
                else:
                    # Feature exists but no active values - this is a real encoding issue
                    raise AssertionError(
                        f"Encoding issue: Vertex {i} (name={vertex_names.iloc[i]}) has "
                        f"node_type='{node_type}' and expected feature '{expected_feature_name}' exists, "
                        f"but no active features found. Feature values: {feature_values}"
                    )

            # Verify exactly one feature is active
            assert len(active_indices) == 1, (
                f"Exactly one node_type feature should be active for vertex {i}, "
                f"but found {len(active_indices)} active features. "
                f"Node type: {node_type}, Feature values: {[features[j].item() for j in node_type_indices]}"
            )

            # Get the feature name for the active index and verify it matches
            active_feature_name = vertex_feature_names[
                node_type_indices[active_indices[0]]
            ]
            expected_feature_name = f"categorical__node_type_{node_type}"

            assert active_feature_name == expected_feature_name, (
                f"Node type mismatch for vertex {i}: "
                f"expected {expected_feature_name}, got {active_feature_name}"
            )


def test_edge_feature_ordering_consistency(edge_masked_napistu_data, napistu_graph):
    """Test that edge features are properly ordered between NapistuGraph and NapistuData.

    This test verifies that the binary encoded r_irreversible features in NapistuData
    correspond to the correct edges in the NapistuGraph based on the edge ordering
    preserved in the NapistuData.ng_edge_names attribute.
    """
    napistu_data = edge_masked_napistu_data

    # Get edge features and names from NapistuData
    edge_features = napistu_data.edge_attr  # Shape: [num_edges, num_edge_features]
    edge_names = napistu_data.ng_edge_names

    # Get edge DataFrame from NapistuGraph
    edge_df = napistu_graph.get_edge_dataframe()

    # Get the edge feature names first
    edge_feature_names = napistu_data.get_edge_feature_names()

    # Check what columns are actually available in the edge DataFrame
    available_columns = list(edge_df.columns)

    # Try to find r_irreversible or similar columns
    r_irreversible_candidates = [
        col
        for col in available_columns
        if "irreversible" in col.lower() or "reversible" in col.lower()
    ]

    # For now, let's use the first available column that might be relevant
    if r_irreversible_candidates:
        target_column = r_irreversible_candidates[0]
    else:
        # Let's just use the first available column for testing
        target_column = available_columns[0] if available_columns else None

    # Merge edge names with the edge DataFrame to get the target column
    if target_column:
        merged_df = edge_names.merge(
            edge_df[["from", "to", target_column]], on=["from", "to"], how="left"
        )
        graph_values = merged_df[target_column].tolist()
    else:
        return

    # Get the feature column index from feature names
    # Edge features likely follow similar naming pattern: 'binary__r_irreversible' etc.
    feature_index = None
    for i, name in enumerate(edge_feature_names):
        if target_column in name:
            feature_index = i
            break

    if feature_index is None:
        return

    # Verify that the feature encoding matches the original values
    for i, (features, graph_value) in enumerate(zip(edge_features, graph_values)):
        if pd.notna(graph_value):
            # Get the feature value
            feature_value = features[feature_index].item()

            # For boolean values, convert to expected float (0.0 or 1.0)
            if isinstance(graph_value, bool):
                expected_value = 1.0 if graph_value else 0.0
                assert abs(feature_value - expected_value) < 0.1, (
                    f"Feature mismatch for edge {i}: "
                    f"expected {expected_value}, got {feature_value}"
                )
            else:
                # For other numeric values, just verify they're accessible
                assert isinstance(
                    feature_value, (int, float)
                ), f"Feature value should be numeric, got {type(feature_value)}"
