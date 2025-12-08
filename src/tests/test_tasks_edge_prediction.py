"""Tests for edge prediction task functions."""

import torch

from napistu_torch.ml.constants import TRAINING
from napistu_torch.models.constants import ENCODERS, HEADS
from napistu_torch.models.heads import Decoder
from napistu_torch.models.message_passing_encoder import MessagePassingEncoder
from napistu_torch.tasks.constants import EDGE_PREDICTION_BATCH
from napistu_torch.tasks.edge_prediction import EdgePredictionTask, get_relation_weights


def _normalize_to_mean_one(weights: torch.Tensor) -> torch.Tensor:
    """Normalize weights to have mean=1.0."""
    return weights * (len(weights) / weights.sum())


def test_get_relation_weights():
    """Test get_relation_weights with different alpha values.

    Tests with category counts [1, 4, 9, 16]:
    - alpha=0: all weights = 1.0 (uniform)
    - alpha=0.5: normalized weights proportional to [1, 2, 3, 4]
    - alpha=1: normalized weights proportional to [1, 0.25, 1/9, 1/16]

    The function computes: weights = 1.0 / (counts ** alpha), then normalizes to mean=1.0.
    """
    # Category counts: [1, 4, 9, 16]
    counts = torch.tensor([1, 4, 9, 16], dtype=torch.long)

    # Test alpha = 0: uniform weighting (all weights = 1.0)
    weights_0 = get_relation_weights(counts, alpha=0.0)
    assert torch.allclose(weights_0, torch.ones(4), rtol=1e-5)
    assert torch.allclose(weights_0.mean(), torch.tensor(1.0), rtol=1e-5)

    # Test alpha = 0.5
    # User expects normalized weights proportional to [1, 2, 3, 4]
    weights_05 = get_relation_weights(counts, alpha=0.5)
    assert torch.allclose(weights_05.mean(), torch.tensor(1.0), rtol=1e-5)
    expected_05 = torch.tensor([1.0, 0.5, 1 / 3, 0.25], dtype=torch.float32)
    expected_05_normalized = _normalize_to_mean_one(expected_05)
    assert torch.allclose(weights_05, expected_05_normalized, rtol=1e-5)

    # Test alpha = 1.0: inverse frequency
    # User expects normalized weights proportional to [1, 0.25, 1/9, 1/16]
    weights_1 = get_relation_weights(counts, alpha=1.0)
    assert torch.allclose(weights_1.mean(), torch.tensor(1.0), rtol=1e-5)
    expected_1 = torch.tensor([1.0, 0.25, 1 / 9, 1 / 16], dtype=torch.float32)
    expected_1_normalized = _normalize_to_mean_one(expected_1)
    assert torch.allclose(weights_1, expected_1_normalized, rtol=1e-5)

    # Verify that weights are in correct order (first category gets highest weight)
    assert weights_1[0] > weights_1[1] > weights_1[2] > weights_1[3]
    assert weights_05[0] > weights_05[1] > weights_05[2] > weights_05[3]


def test_edge_prediction_task_lazy_initialization(edge_prediction_with_sbo_relations):
    """Test that EdgePredictionTask lazily initializes negative sampler and relation weights."""
    # Create encoder and head
    encoder = MessagePassingEncoder(
        in_channels=edge_prediction_with_sbo_relations.num_node_features,
        hidden_channels=32,
        num_layers=2,
        encoder_type=ENCODERS.SAGE,
    )
    # Use Decoder class (not raw head) for Test 1 - non-relation-aware head
    head1 = Decoder(hidden_channels=32, head_type=HEADS.DOT_PRODUCT)

    # Test 1: Negative sampler initialization (without relation weights)
    task1 = EdgePredictionTask(encoder, head1, weight_loss_by_relation_frequency=False)

    # Before prepare_batch: sampler should not be initialized
    assert task1.negative_sampler is None
    assert not task1._sampler_initialized
    assert not task1._relation_weights_initialized

    # Call prepare_batch - should trigger lazy initialization
    batch = task1.prepare_batch(
        edge_prediction_with_sbo_relations, split=TRAINING.TRAIN
    )

    # After prepare_batch: sampler should be initialized
    assert task1.negative_sampler is not None
    assert task1._sampler_initialized
    # Relation weights should be initialized to None (since weighting is disabled)
    assert task1._relation_weights_initialized
    assert task1._relation_weights is None

    # Verify batch was created successfully
    assert EDGE_PREDICTION_BATCH.POS_EDGES in batch
    assert EDGE_PREDICTION_BATCH.NEG_EDGES in batch

    # Test 2: Relation weights initialization (with relation data)
    head2 = Decoder(
        hidden_channels=32, head_type=HEADS.DOT_PRODUCT
    )  # relatons should be initialized if used for scoring even for non-relation-aware head
    task2 = EdgePredictionTask(
        encoder, head2, weight_loss_by_relation_frequency=True, loss_weight_alpha=0.5
    )

    # Before prepare_batch: nothing should be initialized
    assert task2.negative_sampler is None
    assert not task2._sampler_initialized
    assert not task2._relation_weights_initialized

    # Call prepare_batch with data that has relations
    batch2 = task2.prepare_batch(
        edge_prediction_with_sbo_relations, split=TRAINING.TRAIN
    )

    # After prepare_batch: both sampler and relation weights should be initialized
    assert task2.negative_sampler is not None
    assert task2._sampler_initialized
    assert task2._relation_weights_initialized
    assert task2._relation_weights is not None
    assert isinstance(task2._relation_weights, torch.Tensor)
    # Relation weights should have mean=1.0 (normalized)
    assert torch.allclose(task2._relation_weights.mean(), torch.tensor(1.0), rtol=1e-5)

    # Verify batch was created successfully
    assert EDGE_PREDICTION_BATCH.POS_EDGES in batch2
    assert EDGE_PREDICTION_BATCH.NEG_EDGES in batch2

    # Test 3: Multiple calls to prepare_batch should not re-initialize
    # (idempotency)
    original_sampler = task2.negative_sampler
    original_weights = task2._relation_weights.clone()

    # Call prepare_batch again
    _ = task2.prepare_batch(edge_prediction_with_sbo_relations, split=TRAINING.TRAIN)

    # Sampler and weights should be the same objects (not re-initialized)
    assert task2.negative_sampler is original_sampler
    assert torch.equal(task2._relation_weights, original_weights)
