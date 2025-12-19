"""Tests for binary classification loss functions."""

import pytest
import torch

from napistu_torch.ml.losses import (
    compute_bce_loss,
    compute_margin_loss,
    compute_weighted_bce_loss,
    compute_weighted_margin_loss,
)


def test_loss_functions_basic():
    """Test basic functionality of both loss functions."""
    torch.manual_seed(42)

    pos_scores = torch.randn(10) * 2.0
    neg_scores = torch.randn(10) * 2.0

    # Test simple loss
    loss_simple = compute_bce_loss(pos_scores, neg_scores)
    assert isinstance(loss_simple, torch.Tensor)
    assert loss_simple.dim() == 0  # Scalar
    assert loss_simple.item() > 0
    assert not torch.isnan(loss_simple)
    assert not torch.isinf(loss_simple)

    # Test weighted loss with uniform weights
    pos_weights = torch.ones(10)
    neg_weights = torch.ones(10)
    loss_weighted = compute_weighted_bce_loss(
        pos_scores, neg_scores, pos_weights, neg_weights
    )
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
    loss_simple = compute_bce_loss(pos_scores, neg_scores)
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
    loss_perfect = compute_bce_loss(pos_scores_perfect, neg_scores_perfect)
    assert loss_perfect.item() < 0.01  # Very small loss
    assert not torch.isnan(loss_perfect)
    assert not torch.isinf(loss_perfect)

    # Wrong predictions: low scores for positives, high scores for negatives
    pos_scores_wrong = torch.ones(10) * -10.0
    neg_scores_wrong = torch.ones(10) * 10.0
    loss_wrong = compute_bce_loss(pos_scores_wrong, neg_scores_wrong)
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
    pos_weights_large = torch.ones(20)
    neg_weights_small = torch.ones(5)
    loss_different_sizes = compute_weighted_bce_loss(
        pos_scores_large, neg_scores_small, pos_weights_large, neg_weights_small
    )
    assert isinstance(loss_different_sizes, torch.Tensor)
    assert loss_different_sizes.dim() == 0
    assert not torch.isnan(loss_different_sizes)
    assert not torch.isinf(loss_different_sizes)

    # Shape mismatch should raise error
    pos_scores = torch.randn(10) * 2.0
    neg_scores = torch.randn(10) * 2.0
    pos_weights_wrong = torch.ones(5)  # Wrong size
    neg_weights = torch.ones(10)
    with pytest.raises((RuntimeError, ValueError, IndexError)):
        compute_weighted_bce_loss(
            pos_scores,
            neg_scores,
            pos_weights=pos_weights_wrong,
            neg_weights=neg_weights,
        )


def test_margin_loss_basic():
    """Test basic functionality and separation behavior of margin loss."""
    torch.manual_seed(42)

    margin = 1.0

    # Test with random scores
    pos_scores = torch.randn(10) * 0.5 - 0.5  # Negative distances, around -0.5
    neg_scores = torch.randn(10) * 0.5 - 1.0  # More negative (worse)
    loss = compute_margin_loss(pos_scores, neg_scores, margin)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.item() >= 0  # Margin loss is always non-negative
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Test good separation: pos_score > neg_score + margin (loss should be ~0)
    pos_scores_good = torch.ones(10) * -0.1  # Small distance (good)
    neg_scores_good = torch.ones(10) * -2.0  # Large distance (bad)
    loss_good = compute_margin_loss(pos_scores_good, neg_scores_good, margin)
    assert loss_good.item() < 0.01  # Should be very small (clamped to 0)

    # Test poor separation: pos_score < neg_score + margin (loss should be positive)
    pos_scores_poor = torch.ones(10) * -2.0  # Large distance (bad)
    neg_scores_poor = torch.ones(10) * -0.5  # Small distance (good - wrong!)
    loss_poor = compute_margin_loss(pos_scores_poor, neg_scores_poor, margin)
    assert loss_poor.item() > 0.5  # Should be positive
    assert not torch.isnan(loss_poor)
    assert not torch.isinf(loss_poor)


def test_weighted_margin_loss_equivalence_and_custom_weights():
    """Test that weighted margin loss with uniform weights equals simple loss, and custom weights work."""
    torch.manual_seed(42)

    pos_scores = torch.randn(10) * 0.5 - 0.5
    neg_scores = torch.randn(10) * 0.5 - 1.0
    margin = 1.0

    # Uniform weights should equal simple loss
    pos_weights_uniform = torch.ones(10)
    neg_weights_uniform = torch.ones(10)
    loss_weighted_uniform = compute_weighted_margin_loss(
        pos_scores, neg_scores, margin, pos_weights_uniform, neg_weights_uniform
    )
    loss_simple = compute_margin_loss(pos_scores, neg_scores, margin)
    # Note: sorting in weighted version might cause slight differences, so use larger tolerance
    assert torch.isclose(loss_weighted_uniform, loss_simple, atol=1e-4)

    # Custom weights should produce different loss
    # Use non-uniform weights to actually test weighting (uniform weights factor out)
    pos_weights_custom = torch.tensor(
        [2.0, 2.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0]
    )
    neg_weights_custom = torch.tensor(
        [0.5, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.5, 0.5, 1.0]
    )
    loss_weighted_custom = compute_weighted_margin_loss(
        pos_scores, neg_scores, margin, pos_weights_custom, neg_weights_custom
    )
    if loss_simple.item() > 1e-6:  # Only check if loss is non-zero
        assert not torch.isclose(loss_weighted_custom, loss_simple, atol=1e-6)
    assert not torch.isnan(loss_weighted_custom)
    assert not torch.isinf(loss_weighted_custom)
