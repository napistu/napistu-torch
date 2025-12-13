"""Tests for binary classification loss functions."""

import pytest
import torch

from napistu_torch.ml.losses import compute_simple_bce_loss, compute_weighted_bce_loss


def test_loss_functions_basic():
    """Test basic functionality of both loss functions."""
    torch.manual_seed(42)

    pos_scores = torch.randn(10) * 2.0
    neg_scores = torch.randn(10) * 2.0

    # Test simple loss
    loss_simple = compute_simple_bce_loss(pos_scores, neg_scores)
    assert isinstance(loss_simple, torch.Tensor)
    assert loss_simple.dim() == 0  # Scalar
    assert loss_simple.item() > 0
    assert not torch.isnan(loss_simple)
    assert not torch.isinf(loss_simple)

    # Test weighted loss without weights
    loss_weighted = compute_weighted_bce_loss(pos_scores, neg_scores)
    assert isinstance(loss_weighted, torch.Tensor)
    assert loss_weighted.dim() == 0
    assert loss_weighted.item() > 0
    assert not torch.isnan(loss_weighted)
    assert not torch.isinf(loss_weighted)


def test_weighted_loss_equivalence_and_custom_weights():
    """Test that weighted loss with uniform weights equals simple loss, and custom weights work."""
    torch.manual_seed(42)

    pos_scores = torch.randn(10) * 2.0
    neg_scores = torch.randn(10) * 2.0

    # Uniform weights should equal simple loss
    pos_weights = torch.ones(10)
    neg_weights = torch.ones(10)
    loss_weighted_uniform = compute_weighted_bce_loss(
        pos_scores, neg_scores, pos_weights=pos_weights, neg_weights=neg_weights
    )
    loss_simple = compute_simple_bce_loss(pos_scores, neg_scores)
    assert torch.isclose(loss_weighted_uniform, loss_simple, atol=1e-5)

    # Custom weights should produce different loss
    pos_weights_custom = torch.ones(10) * 2.0
    neg_weights_custom = torch.ones(10) * 0.5
    loss_weighted_custom = compute_weighted_bce_loss(
        pos_scores,
        neg_scores,
        pos_weights=pos_weights_custom,
        neg_weights=neg_weights_custom,
    )
    assert not torch.isclose(loss_weighted_custom, loss_simple, atol=1e-6)
    assert not torch.isnan(loss_weighted_custom)
    assert not torch.isinf(loss_weighted_custom)


def test_loss_extreme_predictions():
    """Test loss behavior with perfect vs wrong predictions."""
    # Perfect predictions: high scores for positives, low scores for negatives
    pos_scores_perfect = torch.ones(10) * 10.0
    neg_scores_perfect = torch.ones(10) * -10.0
    loss_perfect = compute_simple_bce_loss(pos_scores_perfect, neg_scores_perfect)
    assert loss_perfect.item() < 0.01  # Very small loss
    assert not torch.isnan(loss_perfect)
    assert not torch.isinf(loss_perfect)

    # Wrong predictions: low scores for positives, high scores for negatives
    pos_scores_wrong = torch.ones(10) * -10.0
    neg_scores_wrong = torch.ones(10) * 10.0
    loss_wrong = compute_simple_bce_loss(pos_scores_wrong, neg_scores_wrong)
    assert loss_wrong.item() > 10.0  # Very large loss
    assert not torch.isnan(loss_wrong)
    assert not torch.isinf(loss_wrong)


def test_loss_edge_cases():
    """Test edge cases: zero weights, different sample sizes, shape mismatch."""
    torch.manual_seed(42)

    # Zero weights
    pos_scores = torch.randn(10) * 2.0
    neg_scores = torch.randn(10) * 2.0
    pos_weights_zero = torch.zeros(10)
    neg_weights = torch.ones(10)
    loss_zero_weights = compute_weighted_bce_loss(
        pos_scores, neg_scores, pos_weights=pos_weights_zero, neg_weights=neg_weights
    )
    assert isinstance(loss_zero_weights, torch.Tensor)
    assert loss_zero_weights.dim() == 0
    assert not torch.isnan(loss_zero_weights)
    assert not torch.isinf(loss_zero_weights)

    # Different sample sizes
    pos_scores_large = torch.randn(20) * 2.0
    neg_scores_small = torch.randn(5) * 2.0
    loss_different_sizes = compute_weighted_bce_loss(pos_scores_large, neg_scores_small)
    assert isinstance(loss_different_sizes, torch.Tensor)
    assert loss_different_sizes.dim() == 0
    assert not torch.isnan(loss_different_sizes)
    assert not torch.isinf(loss_different_sizes)

    # Shape mismatch should raise error
    pos_scores = torch.randn(10) * 2.0
    neg_scores = torch.randn(10) * 2.0
    pos_weights_wrong = torch.ones(5)  # Wrong size
    with pytest.raises((RuntimeError, ValueError, IndexError)):
        compute_weighted_bce_loss(pos_scores, neg_scores, pos_weights=pos_weights_wrong)
