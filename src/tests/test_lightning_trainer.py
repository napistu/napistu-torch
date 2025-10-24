"""Tests for Lightning trainer integration - actual model fitting."""

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from napistu_torch.configs import DataConfig, ExperimentConfig
from napistu_torch.lightning.data_module import NapistuDataModule
from napistu_torch.lightning.tasks import EdgePredictionLightning
from napistu_torch.load.constants import SPLITTING_STRATEGIES
from napistu_torch.models.constants import ENCODERS
from napistu_torch.models.gnns import GNNEncoder
from napistu_torch.models.heads import DotProductHead
from napistu_torch.tasks.edge_prediction import EdgePredictionTask


def test_edge_prediction_trainer_fit(edge_masked_napistu_data):
    """Test that we can actually fit an edge prediction model with Lightning trainer."""

    # Create a minimal experiment config for fast testing
    experiment_config = ExperimentConfig(
        name="trainer_test",
        seed=42,
        deterministic=True,
        fast_dev_run=True,  # Only run 1 batch for testing
        limit_train_batches=0.1,  # Limit to 10% of batches
        limit_val_batches=0.1,
    )

    # Create data config
    data_config = DataConfig(
        splitting_strategy=SPLITTING_STRATEGIES.EDGE_MASK,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
    )

    # Create encoder and head
    encoder = GNNEncoder(
        in_channels=edge_masked_napistu_data.num_node_features,
        hidden_channels=32,  # Small for fast testing
        num_layers=2,
        encoder_type=ENCODERS.SAGE,
    )
    head = DotProductHead()

    # Create task
    task = EdgePredictionTask(encoder, head)

    # Create Lightning module
    lightning_task = EdgePredictionLightning(task, experiment_config.training)

    # Create data module
    dm = NapistuDataModule(edge_masked_napistu_data, data_config)

    # Create trainer with minimal settings for testing
    trainer = pl.Trainer(
        max_epochs=2,  # Just 2 epochs for testing
        accelerator="cpu",  # Use CPU for testing
        devices=1,
        enable_checkpointing=False,  # Disable checkpointing for testing
        enable_progress_bar=False,  # Disable progress bar for cleaner output
        enable_model_summary=False,  # Disable model summary for cleaner output
        logger=False,  # Disable logging for testing
        fast_dev_run=True,  # Override to ensure fast run
    )

    # This should not raise any errors
    trainer.fit(lightning_task, dm)

    # Verify the model was actually trained (has gradients)
    # Check that parameters exist and have gradients
    encoder_params = list(lightning_task.task.encoder.parameters())

    assert len(encoder_params) > 0, "Encoder should have parameters"

    # Check that at least one parameter requires grad (was trained)
    has_gradients = any(param.requires_grad for param in encoder_params)
    assert has_gradients, "Model should have trainable parameters with gradients"

    # Note: DotProductHead has no learnable parameters (it's just dot product)
    # So we only check encoder parameters


def test_edge_prediction_trainer_fit_with_callbacks(edge_masked_napistu_data):
    """Test trainer fit with callbacks enabled."""

    # Create experiment config
    experiment_config = ExperimentConfig(
        name="trainer_test_with_callbacks",
        seed=42,
        deterministic=True,
        fast_dev_run=True,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
    )

    # Create data config
    data_config = DataConfig(
        splitting_strategy=SPLITTING_STRATEGIES.EDGE_MASK,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
    )

    # Create encoder and head
    encoder = GNNEncoder(
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
    dm = NapistuDataModule(edge_masked_napistu_data, data_config)

    # Create callbacks
    early_stopping = EarlyStopping(
        monitor="val_auc",
        patience=2,
        mode="max",
    )

    model_checkpoint = ModelCheckpoint(
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        filename="best_model",
    )

    # Create trainer with callbacks
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="cpu",
        devices=1,
        callbacks=[early_stopping, model_checkpoint],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        fast_dev_run=True,
    )

    # This should not raise any errors
    trainer.fit(lightning_task, dm)

    # Verify callbacks were created
    assert len(trainer.callbacks) == 2


def test_edge_prediction_trainer_test(edge_masked_napistu_data):
    """Test that we can run test after fitting."""

    # Create experiment config
    experiment_config = ExperimentConfig(
        name="trainer_test_eval",
        seed=42,
        deterministic=True,
        fast_dev_run=True,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
    )

    # Create data config
    data_config = DataConfig(
        splitting_strategy=SPLITTING_STRATEGIES.EDGE_MASK,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
    )

    # Create encoder and head
    encoder = GNNEncoder(
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
    dm = NapistuDataModule(edge_masked_napistu_data, data_config)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        fast_dev_run=True,
    )

    # Fit the model
    trainer.fit(lightning_task, dm)

    # Test the model
    test_results = trainer.test(lightning_task, dm)

    # Verify test results structure
    assert isinstance(test_results, list)
    assert len(test_results) > 0
    assert isinstance(test_results[0], dict)

    # Should have test metrics
    assert "test_auc" in test_results[0]
    assert "test_ap" in test_results[0]


def test_edge_prediction_trainer_different_encoders(edge_masked_napistu_data):
    """Test trainer with different encoder types."""

    encoders_to_test = ["sage", "gcn", "gat"]

    for encoder_type in encoders_to_test:
        # Create experiment config
        experiment_config = ExperimentConfig(
            name=f"trainer_test_{encoder_type}",
            seed=42,
            deterministic=True,
            fast_dev_run=True,
            limit_train_batches=0.05,  # Even smaller for multiple tests
            limit_val_batches=0.05,
        )

        # Create data config
        data_config = DataConfig(
            splitting_strategy=SPLITTING_STRATEGIES.EDGE_MASK,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
        )

        # Create encoder and head
        encoder = GNNEncoder(
            in_channels=edge_masked_napistu_data.num_node_features,
            hidden_channels=16,  # Small for fast testing
            num_layers=1,  # Minimal layers
            encoder_type=encoder_type,
        )
        head = DotProductHead()

        # Create task
        task = EdgePredictionTask(encoder, head)

        # Create Lightning module
        lightning_task = EdgePredictionLightning(task, experiment_config.training)

        # Create data module
        dm = NapistuDataModule(edge_masked_napistu_data, data_config)

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            fast_dev_run=True,
        )

        # This should not raise any errors for any encoder type
        trainer.fit(lightning_task, dm)

        # Verify the model was trained
        encoder_params = list(lightning_task.task.encoder.parameters())

        assert len(encoder_params) > 0, f"{encoder_type} encoder should have parameters"

        # Check that at least one parameter requires grad (was trained)
        has_gradients = any(param.requires_grad for param in encoder_params)
        assert (
            has_gradients
        ), f"{encoder_type} model should have trainable parameters with gradients"

        # Note: DotProductHead has no learnable parameters (it's just dot product)
        # So we only check encoder parameters


def test_edge_prediction_trainer_gpu_if_available(edge_masked_napistu_data):
    """Test trainer on GPU if available, otherwise skip."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create experiment config
    experiment_config = ExperimentConfig(
        name="trainer_test_gpu",
        seed=42,
        deterministic=True,
        fast_dev_run=True,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
    )

    # Create data config
    data_config = DataConfig(
        splitting_strategy=SPLITTING_STRATEGIES.EDGE_MASK,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
    )

    # Create encoder and head
    encoder = GNNEncoder(
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
    dm = NapistuDataModule(edge_masked_napistu_data, data_config)

    # Create trainer with GPU
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        fast_dev_run=True,
    )

    # This should not raise any errors
    trainer.fit(lightning_task, dm)

    # Verify model is on GPU
    encoder_params = list(lightning_task.task.encoder.parameters())

    # Check that at least one parameter is on GPU
    has_gpu_params = any(param.is_cuda for param in encoder_params)
    assert has_gpu_params, "Model parameters should be on GPU"
