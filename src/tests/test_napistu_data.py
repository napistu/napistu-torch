"""
Tests for NapistuData class functionality.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch
from napistu.constants import IDENTIFIERS, ONTOLOGIES
from napistu.network.constants import (
    NAPISTU_GRAPH,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_NODE_TYPES,
    NAPISTU_GRAPH_VERTICES,
)
from utils import assert_tensors_equal

from napistu_torch.constants import (
    NAPISTU_DATA,
    NAPISTU_DATA_SUMMARIES,
    NAPISTU_DATA_SUMMARY_TYPES,
    PYG,
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
    """Test the get_summary method."""
    summary = napistu_data.get_summary(NAPISTU_DATA_SUMMARY_TYPES.DETAILED)

    assert isinstance(summary, dict)
    assert PYG.NUM_NODES in summary
    assert PYG.NUM_EDGES in summary
    assert PYG.NUM_NODE_FEATURES in summary
    assert PYG.NUM_EDGE_FEATURES in summary
    assert NAPISTU_DATA_SUMMARIES.HAS_VERTEX_FEATURE_NAMES in summary
    assert NAPISTU_DATA_SUMMARIES.HAS_EDGE_FEATURE_NAMES in summary

    assert summary[PYG.NUM_NODES] == napistu_data.num_nodes
    assert summary[PYG.NUM_EDGES] == napistu_data.num_edges
    assert summary[NAPISTU_DATA_SUMMARIES.HAS_VERTEX_FEATURE_NAMES] is True
    assert summary[NAPISTU_DATA_SUMMARIES.HAS_EDGE_FEATURE_NAMES] is True


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

        # Verify tensors are equal (handles NaN values correctly)
        assert_tensors_equal(loaded_data.x, napistu_data.x)
        assert_tensors_equal(loaded_data.edge_index, napistu_data.edge_index)
        assert_tensors_equal(loaded_data.edge_attr, napistu_data.edge_attr)

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


def test_species_type_prediction_data_ordering_consistency(
    species_type_prediction_napistu_data, napistu_graph
):
    """Test that data ordering is consistent between NapistuGraph and NapistuData objects.

    This test verifies that when labels are decoded from the NapistuData object,
    they correspond to the correct vertices in the NapistuGraph based on the
    vertex ordering preserved in the NapistuData.ng_vertex_names attribute.
    """
    # Use the new _validate_labels method to test consistency
    species_type_prediction_napistu_data._validate_labels(
        napistu_graph, species_type_prediction_napistu_data.labeling_manager
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


def test_get_feature_by_name(napistu_data):
    """Test get_feature_by_name method with valid and invalid feature names."""

    # Get the vertex feature names to test with
    vertex_feature_names = napistu_data.get_vertex_feature_names()
    assert vertex_feature_names is not None, "Should have vertex feature names"

    # Test with a valid feature name
    valid_feature_name = vertex_feature_names[0]  # Use the first feature
    feature_tensor = napistu_data.get_feature_by_name(valid_feature_name)

    # Verify the returned tensor has the expected shape
    assert isinstance(feature_tensor, torch.Tensor)
    assert feature_tensor.shape == (napistu_data.num_nodes,)

    # Test with an invalid feature name
    invalid_feature_name = "nonexistent_feature"
    with pytest.raises(
        ValueError, match=f"Feature name {invalid_feature_name} not found"
    ):
        napistu_data.get_feature_by_name(invalid_feature_name)


def test_get_features_by_regex(napistu_data):
    """Test get_features_by_regex method with ontology regex pattern."""

    # Test 1: Match features containing "ontology" (lowercase)
    features, feature_names = napistu_data.get_features_by_regex(
        IDENTIFIERS.ONTOLOGY, return_suffixes=False
    )

    # Verify the returned types
    assert isinstance(features, torch.Tensor)
    assert isinstance(feature_names, list)

    # Verify the shape matches the number of matching features
    assert features.shape[0] == napistu_data.num_nodes
    assert features.shape[1] == len(feature_names)

    # Verify all feature names contain "ontology"
    for name in feature_names:
        assert (
            IDENTIFIERS.ONTOLOGY in name.lower()
        ), f"Feature name {name} should contain {IDENTIFIERS.ONTOLOGY}"

    # Test 2: Verify "chebi" is one of the matched entries and check total count
    chebi_matches = [name for name in feature_names if ONTOLOGIES.CHEBI in name.lower()]
    assert (
        len(chebi_matches) == 1
    ), f"Expected exactly 1 '{ONTOLOGIES.CHEBI}' feature, but found {len(chebi_matches)}: {chebi_matches}"
    assert (
        chebi_matches[0] == f"binary__{IDENTIFIERS.ONTOLOGY}_{ONTOLOGIES.CHEBI}"
    ), f"Expected 'binary__{IDENTIFIERS.ONTOLOGY}_{ONTOLOGIES.CHEBI}' but found {chebi_matches[0]}"

    # Test 3: Verify we have exactly 6 ontology features as shown in the output
    assert (
        len(feature_names) == 5
    ), f"Expected exactly 5 ontology features, but found {len(feature_names)}"

    # Test 4: Test with return_suffixes=True to extract suffixes after "ontology"
    masks, mask_names = napistu_data.get_features_by_regex(
        IDENTIFIERS.ONTOLOGY, return_suffixes=True
    )

    # Verify the returned types
    assert isinstance(masks, torch.Tensor)
    assert isinstance(mask_names, list)

    # Verify the same number of features are returned
    assert masks.shape == features.shape
    assert len(mask_names) == len(feature_names)

    # Verify suffixes are extracted correctly (should be the part after "ontology")
    # note that ONTOLOGIES.REACTOME is a constant and is dropped during encoding so it should not be in the mask names
    EXPECTED_MASK_NAMES = [
        ONTOLOGIES.EC_CODE,
        ONTOLOGIES.CHEBI,
        ONTOLOGIES.GO,
        ONTOLOGIES.PUBMED,
        ONTOLOGIES.UNIPROT,
    ]
    assert set(mask_names) == set(
        EXPECTED_MASK_NAMES
    ), f"Expected mask names {EXPECTED_MASK_NAMES}, but got {mask_names}"

    # Test 5: Test error case - no matching features
    with pytest.raises(ValueError, match="No features found with regex"):
        napistu_data.get_features_by_regex("nonexistent_pattern_12345")

    # Test 6: Test error case - regex with capturing groups when return_suffixes=True
    with pytest.raises(ValueError, match="already contains capturing groups"):
        napistu_data.get_features_by_regex(
            f"{IDENTIFIERS.ONTOLOGY}(.*)", return_suffixes=True
        )


def test_copy(napistu_data):
    """Test that copy creates a deep copy that is independent of the original."""
    copied = napistu_data.copy()

    # Verify it's a NapistuData object
    assert isinstance(copied, NapistuData)

    # Verify basic properties are the same
    assert copied.num_nodes == napistu_data.num_nodes
    assert copied.num_edges == napistu_data.num_edges

    # Verify tensors are equal but not the same object
    assert_tensors_equal(copied.x, napistu_data.x)
    assert copied.x is not napistu_data.x

    # Verify modifications to copy don't affect original
    copied.x[0, 0] = 999.0
    # After modification, tensors should not be equal
    # Use torch.equal directly here since we're checking they're different
    assert not torch.equal(copied.x, napistu_data.x)
    assert copied.x[0, 0] != napistu_data.x[0, 0]


def test_trim_default(napistu_data):
    """Test trim method with default settings keeps essential attributes."""
    trimmed = napistu_data.trim()

    # Verify core attributes are kept
    assert_tensors_equal(trimmed.x, napistu_data.x)
    assert_tensors_equal(trimmed.edge_index, napistu_data.edge_index)
    assert_tensors_equal(trimmed.edge_attr, napistu_data.edge_attr)

    # Verify edge_weight is preserved if it exists
    if hasattr(napistu_data, PYG.EDGE_WEIGHT) and napistu_data.edge_weight is not None:
        assert_tensors_equal(trimmed.edge_weight, napistu_data.edge_weight)

    # Verify metadata is removed
    assert not hasattr(trimmed, NAPISTU_DATA.NG_VERTEX_NAMES)
    assert not hasattr(trimmed, NAPISTU_DATA.NG_EDGE_NAMES)
    assert not hasattr(trimmed, NAPISTU_DATA.VERTEX_FEATURE_NAMES)
    assert not hasattr(trimmed, NAPISTU_DATA.EDGE_FEATURE_NAMES)

    # Verify name is set to trimmed
    assert trimmed.name == "default_trimmed"


def test_trim_no_edge_attr(napistu_data):
    """Test trim method with keep_edge_attr=False removes edge features."""
    trimmed = napistu_data.trim(keep_edge_attr=False)

    # Verify edge_attr is empty
    assert trimmed.edge_attr.shape == (napistu_data.num_edges, 0)
    assert trimmed.num_edge_features == 0


def test_trim_no_labels_masks(edge_masked_napistu_data):
    """Test trim method removes labels and masks when requested."""
    trimmed = edge_masked_napistu_data.trim(keep_labels=False, keep_masks=False)

    # Verify that trimmed data has core attributes
    assert hasattr(trimmed, PYG.X)
    assert hasattr(trimmed, PYG.EDGE_INDEX)
    assert hasattr(trimmed, PYG.EDGE_ATTR)

    # Verify masks are removed (check original had them, trimmed doesn't)
    assert hasattr(edge_masked_napistu_data, NAPISTU_DATA.TRAIN_MASK)
    assert hasattr(edge_masked_napistu_data, NAPISTU_DATA.VAL_MASK)
    assert hasattr(edge_masked_napistu_data, NAPISTU_DATA.TEST_MASK)
    assert not hasattr(trimmed, NAPISTU_DATA.TRAIN_MASK)
    assert not hasattr(trimmed, NAPISTU_DATA.VAL_MASK)
    assert not hasattr(trimmed, NAPISTU_DATA.TEST_MASK)

    # Verify metadata is removed
    assert not hasattr(trimmed, NAPISTU_DATA.VERTEX_FEATURE_NAMES)
    assert not hasattr(trimmed, NAPISTU_DATA.EDGE_FEATURE_NAMES)


def test_trim_no_relation_type(edge_prediction_with_sbo_relations):
    """Test trim method with keep_relation_type=False removes relation_type."""
    # Verify original data has relation_type
    assert hasattr(edge_prediction_with_sbo_relations, NAPISTU_DATA.RELATION_TYPE)
    assert edge_prediction_with_sbo_relations.relation_type is not None

    # Trim with keep_relation_type=False
    trimmed = edge_prediction_with_sbo_relations.trim(keep_relation_type=False)

    # Verify relation_type is removed
    assert not hasattr(trimmed, NAPISTU_DATA.RELATION_TYPE)

    # Verify relation_manager is also removed when relation_type is not kept
    assert hasattr(edge_prediction_with_sbo_relations, NAPISTU_DATA.RELATION_MANAGER)
    assert not hasattr(trimmed, NAPISTU_DATA.RELATION_MANAGER)

    # Verify core attributes are still present
    assert hasattr(trimmed, PYG.X)
    assert hasattr(trimmed, PYG.EDGE_INDEX)
    assert hasattr(trimmed, PYG.EDGE_ATTR)


def test_trim_keep_relation_type(edge_prediction_with_sbo_relations):
    """Test trim method with keep_relation_type=True preserves relation_type."""
    # Verify original data has relation_type
    assert hasattr(edge_prediction_with_sbo_relations, NAPISTU_DATA.RELATION_TYPE)
    original_relation_type = edge_prediction_with_sbo_relations.relation_type
    assert original_relation_type is not None

    # Trim with keep_relation_type=True (default)
    trimmed = edge_prediction_with_sbo_relations.trim(keep_relation_type=True)

    # Verify relation_type is preserved
    assert hasattr(trimmed, NAPISTU_DATA.RELATION_TYPE)
    assert trimmed.relation_type is not None
    assert torch.equal(trimmed.relation_type, original_relation_type)

    # Verify relation_manager is also preserved when relation_type is kept
    original_relation_manager = getattr(
        edge_prediction_with_sbo_relations, NAPISTU_DATA.RELATION_MANAGER
    )
    assert original_relation_manager is not None
    assert hasattr(trimmed, NAPISTU_DATA.RELATION_MANAGER)
    assert trimmed.relation_manager is not None
    assert trimmed.relation_manager is original_relation_manager

    # Verify get_num_relations still works
    num_relations = trimmed.get_num_relations()
    assert num_relations > 0


def test_trim_keep_labels(species_type_prediction_napistu_data):
    """Test trim method with keep_labels=True preserves labels and labeling_manager."""
    # Verify original data has labels
    assert hasattr(species_type_prediction_napistu_data, PYG.Y)
    original_labels = species_type_prediction_napistu_data.y
    assert original_labels is not None

    # Trim with keep_labels=True (default)
    trimmed = species_type_prediction_napistu_data.trim(keep_labels=True)

    # Verify labels are preserved
    assert hasattr(trimmed, PYG.Y)
    assert trimmed.y is not None
    assert torch.equal(trimmed.y, original_labels)

    # Verify labeling_manager is also preserved when labels are kept
    original_labeling_manager = getattr(
        species_type_prediction_napistu_data, NAPISTU_DATA.LABELING_MANAGER
    )
    assert original_labeling_manager is not None
    assert hasattr(trimmed, NAPISTU_DATA.LABELING_MANAGER)
    assert trimmed.labeling_manager is not None
    assert trimmed.labeling_manager is original_labeling_manager


def test_estimate_memory_footprint(napistu_data):
    """Test estimate_memory_footprint returns correct memory estimates."""
    footprint = napistu_data.estimate_memory_footprint()

    # Verify return type
    assert isinstance(footprint, dict)

    # Verify expected keys are present
    assert PYG.X in footprint
    assert PYG.EDGE_INDEX in footprint
    assert PYG.EDGE_ATTR in footprint
    assert NAPISTU_DATA.TRAIN_MASK in footprint
    assert NAPISTU_DATA.VAL_MASK in footprint
    assert NAPISTU_DATA.TEST_MASK in footprint
    assert "total" in footprint

    # Verify node features memory is calculated
    assert footprint[PYG.X] is not None
    assert footprint[PYG.X] > 0
    assert footprint[PYG.X] == napistu_data.x.element_size() * napistu_data.x.nelement()

    # Verify edge_index memory is calculated
    assert footprint[PYG.EDGE_INDEX] is not None
    assert footprint[PYG.EDGE_INDEX] > 0
    assert (
        footprint[PYG.EDGE_INDEX]
        == napistu_data.edge_index.element_size() * napistu_data.edge_index.nelement()
    )

    # Verify edge_attr memory is calculated (if present)
    if napistu_data.edge_attr is not None:
        assert footprint[PYG.EDGE_ATTR] is not None
        assert footprint[PYG.EDGE_ATTR] > 0
        assert (
            footprint[PYG.EDGE_ATTR]
            == napistu_data.edge_attr.element_size() * napistu_data.edge_attr.nelement()
        )

    # Verify total is sum of all components
    expected_total = (
        footprint[PYG.X]
        + footprint[PYG.EDGE_INDEX]
        + (footprint[PYG.EDGE_ATTR] or 0)
        + (footprint[NAPISTU_DATA.TRAIN_MASK] or 0)
        + (footprint[NAPISTU_DATA.VAL_MASK] or 0)
        + (footprint[NAPISTU_DATA.TEST_MASK] or 0)
    )
    assert footprint["total"] == expected_total
    assert footprint["total"] > 0

    # Call show_memory_footprint (just verify it doesn't raise, don't validate prints)
    napistu_data.show_memory_footprint()
