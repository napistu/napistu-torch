"""
Tests for VertexTensor class functionality.
"""

import pandas as pd
import pytest
import torch

from napistu_torch.vertex_tensor import VertexTensor


def test_align_to_napistu_data(napistu_data, comprehensive_source_membership):
    """Test align_to_napistu_data method with various alignment scenarios."""

    # Test Case 1: Already aligned (same order)
    # First, ensure they're aligned by calling align_to_napistu_data
    aligned_tensor = comprehensive_source_membership.align_to_napistu_data(
        napistu_data, inplace=False
    )

    # Test Case 2: Different order but same vertices
    # Create a reordered version of the comprehensive_source_membership
    vertex_names = comprehensive_source_membership.vertex_names
    data = comprehensive_source_membership.data

    # Reorder the data (reverse order)
    reorder_indices = list(range(len(vertex_names)))[::-1]
    reordered_data = data[reorder_indices]
    reordered_vertex_names = vertex_names.iloc[reorder_indices]

    reordered_tensor = VertexTensor(
        data=reordered_data,
        feature_names=comprehensive_source_membership.feature_names.copy(),
        vertex_names=reordered_vertex_names,
        name="reordered_tensor",
    )

    # Test alignment with reordered data
    aligned_reordered = reordered_tensor.align_to_napistu_data(
        napistu_data, inplace=False
    )

    # Verify the aligned tensor matches the original alignment
    assert torch.allclose(
        aligned_reordered.data, aligned_tensor.data
    ), "Reordered alignment should match original"
    assert (
        aligned_reordered.vertex_names.values.tolist()
        == aligned_tensor.vertex_names.values.tolist()
    ), "Vertex names should match"

    # Test Case 3: Subset of vertices
    # Create a subset tensor with only the first half of vertices
    subset_size = len(vertex_names) // 2
    subset_data = data[:subset_size]
    subset_vertex_names = vertex_names.iloc[:subset_size]

    subset_tensor = VertexTensor(
        data=subset_data,
        feature_names=comprehensive_source_membership.feature_names.copy(),
        vertex_names=subset_vertex_names,
        name="subset_tensor",
    )

    # Test that subset fails alignment due to different number of vertices
    with pytest.raises(ValueError, match="different numbers of rows"):
        subset_tensor.align_to_napistu_data(napistu_data)

    # Test Case 4: Different vertex names (should fail)
    # Create a tensor with completely different vertex names
    fake_vertex_names = pd.Index([f"fake_vertex_{i}" for i in range(len(vertex_names))])
    fake_tensor = VertexTensor(
        data=data,
        feature_names=comprehensive_source_membership.feature_names.copy(),
        vertex_names=fake_vertex_names,
        name="fake_tensor",
    )

    # Test that different vertex names fail alignment
    with pytest.raises(ValueError, match="Vertex names.*do not match"):
        fake_tensor.align_to_napistu_data(napistu_data)

    # Test Case 5: Test inplace=False vs inplace=True
    original_data = reordered_tensor.data.clone()
    original_vertex_names = reordered_tensor.vertex_names.copy()

    # Test inplace=True
    result_inplace = reordered_tensor.align_to_napistu_data(napistu_data, inplace=True)
    assert result_inplace is reordered_tensor, "Should return self when inplace=True"
    assert not torch.allclose(
        reordered_tensor.data, original_data
    ), "Data should be modified in place"
    assert not reordered_tensor.vertex_names.equals(
        original_vertex_names
    ), "Vertex names should be modified in place"

    # Test inplace=False (restore original first)
    reordered_tensor.data = original_data
    reordered_tensor.vertex_names = original_vertex_names
    result_copy = reordered_tensor.align_to_napistu_data(napistu_data, inplace=False)
    assert (
        result_copy is not reordered_tensor
    ), "Should return new instance when inplace=False"
    assert torch.allclose(
        reordered_tensor.data, original_data
    ), "Original data should be unchanged"
    assert reordered_tensor.vertex_names.equals(
        original_vertex_names
    ), "Original vertex names should be unchanged"
