"""Tests for labeling strategy configuration."""

import pytest

from napistu_torch.labeling.constants import LABEL_TYPE, VALID_LABEL_TYPES
from napistu_torch.labeling.labeling_strategy import (
    LABEL_INFORMED_FEATURIZATION,
    LabelingStrategy,
)


def test_label_informed_featurization_completeness():
    """Test that all VALID_LABEL_TYPES are defined in LABEL_INFORMED_FEATURIZATION and vice versa."""
    # Get the keys from LABEL_INFORMED_FEATURIZATION
    defined_keys = set(LABEL_INFORMED_FEATURIZATION.keys())
    valid_label_types = set(VALID_LABEL_TYPES)

    # Check that all valid label types are defined
    missing_keys = valid_label_types - defined_keys
    assert (
        not missing_keys
    ), f"Missing label types in LABEL_INFORMED_FEATURIZATION: {missing_keys}"

    # Check that all defined keys are valid label types
    extra_keys = defined_keys - valid_label_types
    assert (
        not extra_keys
    ), f"Extra label types in LABEL_INFORMED_FEATURIZATION: {extra_keys}"

    # Verify they are exactly equal
    assert defined_keys == valid_label_types, (
        f"LABEL_INFORMED_FEATURIZATION keys ({defined_keys}) "
        f"do not match VALID_LABEL_TYPES ({valid_label_types})"
    )


def test_labeling_strategy_validation():
    """Test that LabelingStrategy validates correctly."""
    # Test valid strategy
    strategy = LabelingStrategy(
        label_attribute=LABEL_TYPE.SPECIES_TYPE,
        exclude_vertex_attributes=[LABEL_TYPE.SPECIES_TYPE],
        augment_summary_types=[],
    )
    assert strategy.label_attribute == LABEL_TYPE.SPECIES_TYPE

    # Test invalid summary type
    with pytest.raises(ValueError, match="Invalid summary_type"):
        LabelingStrategy(
            label_attribute=LABEL_TYPE.SPECIES_TYPE,
            exclude_vertex_attributes=[],
            augment_summary_types=["invalid_summary_type"],
        )


def test_labeling_strategy_serialization():
    """Test that LabelingStrategy can be serialized and deserialized."""
    original = LabelingStrategy(
        label_attribute=LABEL_TYPE.NODE_TYPE,
        exclude_vertex_attributes=[LABEL_TYPE.NODE_TYPE, LABEL_TYPE.SPECIES_TYPE],
        augment_summary_types=[],
    )

    # Test to_dict
    config_dict = original.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict["label_attribute"] == LABEL_TYPE.NODE_TYPE

    # Test from_dict
    reconstructed = LabelingStrategy.from_dict(config_dict)
    assert reconstructed.label_attribute == original.label_attribute
    assert reconstructed.exclude_vertex_attributes == original.exclude_vertex_attributes
    assert reconstructed.augment_summary_types == original.augment_summary_types
