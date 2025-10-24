"""Tests for Lightning task functionality."""

import pytest
import torch

from napistu_torch.lightning.data_module import NapistuDataModule
from napistu_torch.lightning.tasks import EdgePredictionLightning
from napistu_torch.ml.constants import TRAINING
from napistu_torch.models.constants import ENCODERS
from napistu_torch.models.gnns import GNNEncoder
from napistu_torch.models.heads import DotProductHead
from napistu_torch.tasks.edge_prediction import EdgePredictionTask


def test_edge_prediction_lightning_integration(
    edge_masked_napistu_data, experiment_config, data_config
):
    # Create encoder and head
    encoder = GNNEncoder(
        in_channels=edge_masked_napistu_data.num_node_features,
        hidden_channels=32,
        num_layers=2,
        encoder=ENCODERS.SAGE,
    )
    head = DotProductHead()

    # Create task
    task = EdgePredictionTask(encoder, head)

    # Create Lightning module
    lightning_task = EdgePredictionLightning(task, experiment_config.training)

    # Create data module
    dm = NapistuDataModule(edge_masked_napistu_data, data_config)
    dm.setup()

    # Test training step
    train_dl = dm.train_dataloader()
    train_batch = next(iter(train_dl))

    # Should not raise AttributeError
    loss = lightning_task.training_step(train_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad

    # Test validation step
    val_dl = dm.val_dataloader()
    val_batch = next(iter(val_dl))

    # Should not raise AttributeError
    metrics = lightning_task.validation_step(val_batch, batch_idx=0)
    assert isinstance(metrics, dict)
    assert "auc" in metrics
    assert "ap" in metrics


def test_edge_prediction_task_prepare_batch(edge_masked_napistu_data):
    """Test EdgePredictionTask.prepare_batch method with different splits."""

    # Create encoder and head
    encoder = GNNEncoder(
        in_channels=edge_masked_napistu_data.num_node_features,
        hidden_channels=32,
        num_layers=2,
        encoder=ENCODERS.SAGE,
    )
    head = DotProductHead()

    # Create task
    task = EdgePredictionTask(encoder, head)

    # Test prepare_batch for different splits
    for split in [TRAINING.TRAIN, TRAINING.VALIDATION, TRAINING.TEST]:

        batch = task.prepare_batch(edge_masked_napistu_data, split=split)

        # Check that batch contains expected keys
        expected_keys = [
            "x",
            "supervision_edges",
            "pos_edges",
            "neg_edges",
            "edge_weight",
        ]
        for key in expected_keys:
            assert key in batch

        # Check that tensors have correct shapes
        assert batch["x"].shape[0] == edge_masked_napistu_data.num_nodes
        assert batch["pos_edges"].shape[0] == 2
        assert batch["neg_edges"].shape[0] == 2


def test_lightning_task_batch_validation(edge_masked_napistu_data, experiment_config):
    """Test that Lightning tasks properly validate batch types."""
    # Create encoder and head
    encoder = GNNEncoder(
        in_channels=edge_masked_napistu_data.num_node_features,
        hidden_channels=32,
        num_layers=2,
        encoder=ENCODERS.SAGE,
    )
    head = DotProductHead()

    # Create task
    task = EdgePredictionTask(encoder, head)
    lightning_task = EdgePredictionLightning(task, experiment_config.training)

    # Test with valid NapistuData batch
    lightning_task.training_step(edge_masked_napistu_data, batch_idx=0)

    # Test with invalid batch type
    with pytest.raises(AssertionError, match="Expected NapistuData"):
        lightning_task.training_step([edge_masked_napistu_data], batch_idx=0)

    with pytest.raises(AssertionError, match="Expected NapistuData"):
        lightning_task.training_step("invalid_batch", batch_idx=0)
