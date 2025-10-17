"""Tests for labeling strategy configuration."""

import pytest

from napistu_torch.labeling.constants import LABEL_TYPE, VALID_LABEL_TYPES
from napistu_torch.labeling.labeling_manager import (
    LABELING_MANAGERS,
    LabelingManager,
)


def test_label_informed_featurization_completeness():
    """Test that all VALID_LABEL_TYPES are defined in LABELING_MANAGERS and vice versa."""
    # Get the keys from LABELING_MANAGERS
    defined_keys = set(LABELING_MANAGERS.keys())
    valid_label_types = set(VALID_LABEL_TYPES)

    # Check that all valid label types are defined
    missing_keys = valid_label_types - defined_keys
    assert not missing_keys, f"Missing label types in LABELING_MANAGERS: {missing_keys}"

    # Check that all defined keys are valid label types
    extra_keys = defined_keys - valid_label_types
    assert not extra_keys, f"Extra label types in LABELING_MANAGERS: {extra_keys}"

    # Verify they are exactly equal
    assert defined_keys == valid_label_types, (
        f"LABELING_MANAGERS keys ({defined_keys}) "
        f"do not match VALID_LABEL_TYPES ({valid_label_types})"
    )


def test_labeling_strategy_validation():
    """Test that LabelingManager validates correctly."""
    # Test valid strategy
    strategy = LabelingManager(
        label_attribute=LABEL_TYPE.SPECIES_TYPE,
        exclude_vertex_attributes=[LABEL_TYPE.SPECIES_TYPE],
        augment_summary_types=[],
    )
    assert strategy.label_attribute == LABEL_TYPE.SPECIES_TYPE

    # Test invalid summary type
    with pytest.raises(ValueError, match="Invalid summary_type"):
        LabelingManager(
            label_attribute=LABEL_TYPE.SPECIES_TYPE,
            exclude_vertex_attributes=[],
            augment_summary_types=["invalid_summary_type"],
        )


def test_labeling_strategy_serialization():
    """Test that LabelingManager can be serialized and deserialized."""
    original = LabelingManager(
        label_attribute=LABEL_TYPE.NODE_TYPE,
        exclude_vertex_attributes=[LABEL_TYPE.NODE_TYPE, LABEL_TYPE.SPECIES_TYPE],
        augment_summary_types=[],
    )

    # Test to_dict
    config_dict = original.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict["label_attribute"] == LABEL_TYPE.NODE_TYPE

    # Test from_dict
    reconstructed = LabelingManager.from_dict(config_dict)
    assert reconstructed.label_attribute == original.label_attribute
    assert reconstructed.exclude_vertex_attributes == original.exclude_vertex_attributes
    assert reconstructed.augment_summary_types == original.augment_summary_types


def test_labeling_manager_with_label_names():
    """Test that LabelingManager works with the optional label_names attribute."""
    # Test with label_names
    label_names = {0: "protein", 1: "metabolite", 2: "drug"}
    strategy = LabelingManager(
        label_attribute=LABEL_TYPE.SPECIES_TYPE,
        exclude_vertex_attributes=[LABEL_TYPE.SPECIES_TYPE],
        augment_summary_types=[],
        label_names=label_names,
    )
    assert strategy.label_names == label_names

    # Test without label_names (should be None by default)
    strategy_no_names = LabelingManager(
        label_attribute=LABEL_TYPE.SPECIES_TYPE,
        exclude_vertex_attributes=[LABEL_TYPE.SPECIES_TYPE],
        augment_summary_types=[],
    )
    assert strategy_no_names.label_names is None
