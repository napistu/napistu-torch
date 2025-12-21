"""Tests for NapistuData comparison and validation utilities."""

import copy

import pytest

from napistu_torch.constants import (
    NAPISTU_DATA,
    NAPISTU_DATA_SUMMARIES,
    NAPISTU_DATA_SUMMARY_TYPES,
)
from napistu_torch.data.compare_napistu_data import validate_same_data


@pytest.fixture
def relation_prediction_summary(edge_prediction_with_sbo_relations):
    """Fixture providing validation summary from relation prediction data."""
    return edge_prediction_with_sbo_relations.get_summary(
        NAPISTU_DATA_SUMMARY_TYPES.VALIDATION
    )


def test_validate_same_data_fails_on_mismatch(
    napistu_data, species_type_prediction_napistu_data
):
    """Test validate_same_data fails when data summaries don't match."""
    checkpoint_summary = napistu_data.get_summary(NAPISTU_DATA_SUMMARY_TYPES.VALIDATION)
    current_summary = species_type_prediction_napistu_data.get_summary(
        NAPISTU_DATA_SUMMARY_TYPES.VALIDATION
    )

    # Use a different NapistuData fixture with labels (has train/val/test splits)
    # This will cause extra fields in the current data summary
    # Should raise ValueError when data doesn't match
    with pytest.raises(ValueError, match="Data summary"):
        validate_same_data(checkpoint_summary, current_summary)


def test_validate_same_data_edge_vs_relation_prediction(
    edge_masked_napistu_data, edge_prediction_with_sbo_relations, caplog
):
    """Test validate_same_data warns when comparing edge and relation prediction fixtures."""
    checkpoint_summary = edge_masked_napistu_data.get_summary(
        NAPISTU_DATA_SUMMARY_TYPES.VALIDATION
    )
    current_summary = edge_prediction_with_sbo_relations.get_summary(
        NAPISTU_DATA_SUMMARY_TYPES.VALIDATION
    )

    # Should raise ValueError because the summaries differ (missing keys)
    # (edge_masked has no relations, edge_prediction_with_sbo_relations has relations)
    with pytest.raises(ValueError, match="Data summary"):
        validate_same_data(checkpoint_summary, current_summary, allow_missing_keys=[])

    # With allow_missing_keys including relation_type_labels, this should pass
    # but should warn about relation labels mismatch
    with caplog.at_level("WARNING"):
        validate_same_data(
            checkpoint_summary,
            current_summary,
            allow_missing_keys=[
                NAPISTU_DATA.RELATION_TYPE_LABELS,
                NAPISTU_DATA_SUMMARIES.NUM_UNIQUE_RELATIONS,
            ],
        )
        assert "Relation type labels mismatch" in caplog.text


def test_validate_relation_labels_shuffled(relation_prediction_summary):
    """Test that reordered relation type labels are detected."""
    reference_summary = relation_prediction_summary
    current_summary = copy.deepcopy(reference_summary)

    # Reverse order of relation type labels (ensures order changes)
    relation_labels = current_summary.get(NAPISTU_DATA.RELATION_TYPE_LABELS)
    if relation_labels and len(relation_labels) > 1:
        current_summary[NAPISTU_DATA.RELATION_TYPE_LABELS] = list(
            reversed(relation_labels)
        )

        with pytest.raises(ValueError, match="Relation type labels"):
            validate_same_data(current_summary, reference_summary)


def test_validate_relation_labels_reordered(relation_prediction_summary):
    """Test that reordered relation type labels are detected."""
    reference_summary = relation_prediction_summary
    current_summary = copy.deepcopy(reference_summary)

    # Reverse order of relation type labels
    relation_labels = current_summary.get(NAPISTU_DATA.RELATION_TYPE_LABELS)
    if relation_labels and len(relation_labels) > 1:
        current_summary[NAPISTU_DATA.RELATION_TYPE_LABELS] = list(
            reversed(relation_labels)
        )

        with pytest.raises(ValueError, match="Relation type labels"):
            validate_same_data(current_summary, reference_summary)


def test_validate_vertex_feature_names_reordered(relation_prediction_summary):
    """Test that reordered vertex feature names are detected."""
    reference_summary = relation_prediction_summary
    current_summary = copy.deepcopy(reference_summary)

    # Reverse order of vertex feature names
    vertex_features = current_summary.get(NAPISTU_DATA.VERTEX_FEATURE_NAMES)
    if vertex_features and len(vertex_features) > 1:
        current_summary[NAPISTU_DATA.VERTEX_FEATURE_NAMES] = list(
            reversed(vertex_features)
        )

        with pytest.raises(ValueError, match="Vertex feature names"):
            validate_same_data(current_summary, reference_summary)


def test_validate_edge_feature_names_reordered(relation_prediction_summary):
    """Test that reordered edge feature names are detected."""
    reference_summary = relation_prediction_summary
    current_summary = copy.deepcopy(reference_summary)

    # Reverse order of edge feature names
    edge_features = current_summary.get(NAPISTU_DATA.EDGE_FEATURE_NAMES)
    if edge_features and len(edge_features) > 1:
        current_summary[NAPISTU_DATA.EDGE_FEATURE_NAMES] = list(reversed(edge_features))

        with pytest.raises(ValueError, match="Edge feature names"):
            validate_same_data(current_summary, reference_summary)


def test_validate_structural_attributes_mismatch(relation_prediction_summary):
    """Test that structural attribute mismatches are detected."""
    reference_summary = relation_prediction_summary
    current_summary = copy.deepcopy(reference_summary)

    # Modify a structural attribute (num_nodes)
    if "num_nodes" in current_summary:
        current_summary["num_nodes"] = current_summary["num_nodes"] + 1

        with pytest.raises(ValueError, match="structural attributes"):
            validate_same_data(current_summary, reference_summary)


def test_validate_feature_aliases_mismatch(relation_prediction_summary):
    """Test that feature alias mismatches are detected."""
    reference_summary = relation_prediction_summary
    current_summary = copy.deepcopy(reference_summary)

    # Modify vertex feature aliases if they exist
    vertex_aliases = current_summary.get(NAPISTU_DATA.VERTEX_FEATURE_NAME_ALIASES)
    if vertex_aliases and len(vertex_aliases) > 0:
        # Change one alias value
        alias_key = list(vertex_aliases.keys())[0]
        modified_aliases = copy.deepcopy(vertex_aliases)
        modified_aliases[alias_key] = "modified_canonical_name"
        current_summary[NAPISTU_DATA.VERTEX_FEATURE_NAME_ALIASES] = modified_aliases

        with pytest.raises(ValueError, match="Vertex.*aliases"):
            validate_same_data(current_summary, reference_summary)


def test_validate_feature_aliases_missing_keys(relation_prediction_summary):
    """Test that missing alias keys are detected."""
    reference_summary = relation_prediction_summary
    current_summary = copy.deepcopy(reference_summary)

    # Remove a key from edge feature aliases if they exist
    edge_aliases = current_summary.get(NAPISTU_DATA.EDGE_FEATURE_NAME_ALIASES)
    if edge_aliases and len(edge_aliases) > 0:
        modified_aliases = copy.deepcopy(edge_aliases)
        alias_key = list(modified_aliases.keys())[0]
        del modified_aliases[alias_key]
        current_summary[NAPISTU_DATA.EDGE_FEATURE_NAME_ALIASES] = modified_aliases

        with pytest.raises(ValueError, match="Edge.*alias keys"):
            validate_same_data(current_summary, reference_summary)


def test_validate_keys_missing(relation_prediction_summary):
    """Test that missing required keys are detected."""
    reference_summary = relation_prediction_summary
    current_summary = copy.deepcopy(reference_summary)

    # Remove a required key
    if NAPISTU_DATA.VERTEX_FEATURE_NAMES in current_summary:
        del current_summary[NAPISTU_DATA.VERTEX_FEATURE_NAMES]

        with pytest.raises(ValueError, match="Data summary"):
            validate_same_data(
                current_summary, reference_summary, allow_missing_keys=[]
            )


def test_validate_same_data_passes_with_identical_summaries(
    relation_prediction_summary,
):
    """Test that identical summaries pass validation."""
    reference_summary = relation_prediction_summary
    current_summary = copy.deepcopy(reference_summary)

    # Should pass without raising
    validate_same_data(current_summary, reference_summary)
