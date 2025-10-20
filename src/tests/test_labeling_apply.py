"""Tests for labeling apply utilities."""

import pandas as pd

from napistu_torch.labeling.apply import decode_labels
from napistu_torch.labeling.constants import LABEL_TYPE
from napistu_torch.labeling.create import create_vertex_labels


def test_species_type_roundtrip_encoding(napistu_graph):
    """Test that species_type encoding and decoding is fully reversible.

    This test verifies that we can encode species_type labels from a NapistuGraph
    and then decode them back to their original values without any loss of information.
    """
    # Get the original species_type labels from the NapistuGraph
    original_labels = napistu_graph.get_vertex_series(LABEL_TYPE.SPECIES_TYPE)

    # Encode the labels using the labeling system
    encoded_labels, labeling_manager = create_vertex_labels(
        napistu_graph, label_type=LABEL_TYPE.SPECIES_TYPE
    )

    # Decode the labels back to original values
    decoded_labels = decode_labels(encoded_labels, labeling_manager)

    # Convert decoded labels to a pandas Series for comparison
    decoded_series = pd.Series(decoded_labels, index=original_labels.index)

    # Verify that the roundtrip preserves all information
    assert len(original_labels) == len(decoded_series), "Length should be preserved"

    # Compare original and decoded labels, masking on NaN values
    mask = pd.notna(original_labels)
    original_masked = original_labels[mask]
    decoded_masked = decoded_series[mask]

    for i, (original, decoded) in enumerate(zip(original_masked, decoded_masked)):
        assert original == decoded, (
            f"Roundtrip failed at position {i}: "
            f"original={original}, decoded={decoded}"
        )

    # verify that the masked nulls are decoded to None
    # Note: decoded values are None (not NaN), so we check for None values
    null_mask = pd.isna(original_labels)
    decoded_null_values = decoded_series[null_mask]
    assert all(
        pd.isna(decoded_null_values)
    ), "All masked nulls should be decoded to None/NaN"
