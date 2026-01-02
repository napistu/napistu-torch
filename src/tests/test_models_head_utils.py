"""Tests for head utility functions."""

import pytest
import torch

from napistu_torch.models.head_utils import (
    compute_rotate_distance,
    validate_symmetric_relation_indices,
)


def test_compute_rotate_distance():
    """Test compute_rotate_distance functionality."""
    torch.manual_seed(42)

    num_edges = 10
    embedding_dim = 64

    head_embeddings = torch.randn(num_edges, embedding_dim)
    head_embeddings = head_embeddings / head_embeddings.norm(dim=-1, keepdim=True)
    tail_embeddings = torch.randn(num_edges, embedding_dim)
    tail_embeddings = tail_embeddings / tail_embeddings.norm(dim=-1, keepdim=True)
    relation_phase = torch.randn(num_edges, embedding_dim // 2)

    distance = compute_rotate_distance(head_embeddings, tail_embeddings, relation_phase)

    assert distance.shape == (num_edges,)
    assert not torch.isnan(distance).any()
    assert not torch.isinf(distance).any()
    assert (distance >= 0).all()
    assert (distance <= 2.0).all()

    # Test perfect match with identity rotation
    head_same = head_embeddings[:3]
    tail_same = head_same.clone()
    phase_zero = torch.zeros(3, embedding_dim // 2)
    distance_zero = compute_rotate_distance(head_same, tail_same, phase_zero)
    assert torch.allclose(distance_zero, torch.zeros(3), atol=1e-5)


def test_validate_symmetric_relation_indices():
    """Test validate_symmetric_relation_indices validation."""
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_symmetric_relation_indices([], num_relations=5)
    with pytest.raises(ValueError, match="duplicates"):
        validate_symmetric_relation_indices([0, 1, 0], num_relations=5)
    with pytest.raises(ValueError, match="invalid values"):
        validate_symmetric_relation_indices([0, 5], num_relations=5)
    with pytest.raises(ValueError, match="All.*relations are symmetric"):
        validate_symmetric_relation_indices([0, 1, 2, 3, 4], num_relations=5)
    validate_symmetric_relation_indices([0, 2], num_relations=5)
