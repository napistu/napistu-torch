"""Tests for Lightning task functionality."""

import pytest
import torch

from napistu_torch.configs import ModelConfig
from napistu_torch.lightning.full_graph_datamodule import FullGraphDataModule
from napistu_torch.lightning.tasks import EdgePredictionLightning
from napistu_torch.ml.constants import TRAINING
from napistu_torch.models.constants import ENCODERS
from napistu_torch.models.heads import DotProductHead
from napistu_torch.models.message_passing_encoder import MessagePassingEncoder
from napistu_torch.tasks.edge_prediction import EdgePredictionTask


def test_edge_prediction_lightning_integration(
    edge_masked_napistu_data, experiment_config, data_config
):
    # Create encoder and head
    encoder = MessagePassingEncoder(
        in_channels=edge_masked_napistu_data.num_node_features,
        hidden_channels=32,
        num_layers=2,
        encoder_type=ENCODERS.SAGE,
    )
    head = DotProductHead()

    # Create task
    task = EdgePredictionTask(encoder, head)

    # Create Lightning module
    lightning_task = EdgePredictionLightning(task, experiment_config.training)

    # Create data module
    dm = FullGraphDataModule(data_config, napistu_data=edge_masked_napistu_data)
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
    encoder = MessagePassingEncoder(
        in_channels=edge_masked_napistu_data.num_node_features,
        hidden_channels=32,
        num_layers=2,
        encoder_type=ENCODERS.SAGE,
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
            "edge_data",
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
    encoder = MessagePassingEncoder(
        in_channels=edge_masked_napistu_data.num_node_features,
        hidden_channels=32,
        num_layers=2,
        encoder_type=ENCODERS.SAGE,
    )
    head = DotProductHead()

    # Create task
    task = EdgePredictionTask(encoder, head)
    lightning_task = EdgePredictionLightning(task, experiment_config.training)

    # Test with valid NapistuData batch
    lightning_task.training_step(edge_masked_napistu_data, batch_idx=0)

    # Test with invalid batch type
    with pytest.raises(
        ValueError,
        match="Unexpected batch type in training: <class 'list'>. Expected NapistuData or torch.Tensor.",
    ):
        lightning_task.training_step([edge_masked_napistu_data], batch_idx=0)

    with pytest.raises(
        ValueError,
        match="Unexpected batch type in training: <class 'str'>. Expected NapistuData or torch.Tensor.",
    ):
        lightning_task.training_step("invalid_batch", batch_idx=0)


def test_edge_prediction_with_edge_encoder(edge_masked_napistu_data, experiment_config):
    """Test EdgePredictionTask with EdgeEncoder for edge weight support."""

    # Create model config with edge encoder enabled
    model_config = ModelConfig(
        encoder=ENCODERS.GCN,  # GCN supports edge weights
        hidden_channels=32,
        num_layers=2,
        use_edge_encoder=True,
        edge_encoder_dim=16,
        edge_encoder_dropout=0.1,
    )

    # Add dummy edge attributes to test data
    edge_masked_napistu_data.edge_attr = torch.randn(
        edge_masked_napistu_data.edge_index.size(1), 10
    )

    # Create encoder and head with edge encoder
    encoder = MessagePassingEncoder.from_config(
        model_config,
        in_channels=edge_masked_napistu_data.num_node_features,
        edge_in_channels=edge_masked_napistu_data.num_edge_features,
    )
    head = DotProductHead()

    # Create task directly
    task = EdgePredictionTask(
        encoder=encoder,
        head=head,
    )

    # Create Lightning module
    lightning_task = EdgePredictionLightning(task, experiment_config.training)

    # Test training step with edge weights
    loss = lightning_task.training_step(edge_masked_napistu_data, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0  # Loss should be non-negative

    # Test validation step
    metrics = lightning_task.validation_step(edge_masked_napistu_data, batch_idx=0)
    assert isinstance(metrics, dict)
    assert "auc" in metrics
    assert "ap" in metrics

    # Verify edge encoder was created and integrated in the encoder
    from napistu_torch.models.constants import EDGE_WEIGHTING_TYPE, ENCODER_DEFS

    assert hasattr(encoder, ENCODER_DEFS.EDGE_WEIGHTING_TYPE)
    assert (
        getattr(encoder, ENCODER_DEFS.EDGE_WEIGHTING_TYPE)
        == EDGE_WEIGHTING_TYPE.LEARNED_ENCODER
    )
    assert hasattr(encoder, ENCODER_DEFS.EDGE_WEIGHTING_VALUE)
    edge_encoder = getattr(encoder, ENCODER_DEFS.EDGE_WEIGHTING_VALUE)
    assert edge_encoder is not None
    assert edge_encoder.edge_dim == 10
    assert edge_encoder.hidden_dim == 16

    # Verify edge encoder is properly integrated into the MessagePassingEncoder
    # The edge encoder is stored in edge_weighting_value when type is LEARNED_ENCODER
    assert edge_encoder is getattr(encoder, ENCODER_DEFS.EDGE_WEIGHTING_VALUE)

    # Verify that the encoder uses the edge encoder during forward pass
    # by checking that it requires edge_data when edge encoder is present
    with pytest.raises(
        ValueError, match="edge_data required when using learnable edge encoder"
    ):
        encoder.forward(edge_masked_napistu_data.x, edge_masked_napistu_data.edge_index)

    # Test that forward pass works with edge_data
    output = encoder.forward(
        edge_masked_napistu_data.x,
        edge_masked_napistu_data.edge_index,
        edge_masked_napistu_data.edge_attr,
    )
    assert output.shape == (
        edge_masked_napistu_data.x.shape[0],
        32,
    )  # hidden_channels=32
