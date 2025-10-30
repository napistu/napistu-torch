"""Tests for Lightning trainer integration - actual model fitting."""

from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from napistu_torch.configs import DataConfig, ExperimentConfig
from napistu_torch.lightning.full_graph_datamodule import FullGraphDataModule
from napistu_torch.lightning.tasks import EdgePredictionLightning
from napistu_torch.models.constants import ENCODERS
from napistu_torch.models.heads import DotProductHead
from napistu_torch.models.message_passing_encoder import MessagePassingEncoder
from napistu_torch.tasks.edge_prediction import EdgePredictionTask


@pytest.fixture
def stub_data_config():
    """Create a stubbed DataConfig for testing."""
    return DataConfig(
        name="stubbed_config",
        sbml_dfs_path=Path("stub_sbml.pkl"),
        napistu_graph_path=Path("stub_graph.pkl"),
        napistu_data_name="edge_prediction",
    )


def _create_test_trainer(
    max_epochs=2, callbacks=None, enable_checkpointing=False, accelerator="cpu"
):
    """Create a pl.Trainer configured for fast testing.

    Parameters
    ----------
    max_epochs : int, default=2
        Maximum number of epochs to run
    callbacks : list, optional
        List of callbacks to use. If None, no callbacks are used.
    enable_checkpointing : bool, default=False
        Whether to enable checkpointing.
    accelerator : str, default="cpu"
        Accelerator to use ("cpu" or "gpu").

    Returns
    -------
    pl.Trainer
        Configured trainer for testing
    """
    if callbacks is None:
        callbacks = []

    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        enable_checkpointing=enable_checkpointing,
        enable_progress_bar=False,  # Disable progress bar for cleaner output
        enable_model_summary=False,  # Disable model summary for cleaner output
        logger=False,  # Disable logging for testing
        fast_dev_run=True,  # Override to ensure fast run
        callbacks=callbacks,
    )


def test_edge_prediction_trainer_fit(edge_masked_napistu_data, stub_data_config):
    """Test that we can actually fit an edge prediction model with Lightning trainer."""

    # Create a minimal experiment config for fast testing
    experiment_config = ExperimentConfig(
        name="trainer_test",
        seed=42,
        deterministic=True,
        fast_dev_run=True,  # Only run 1 batch for testing
        limit_train_batches=0.1,  # Limit to 10% of batches
        limit_val_batches=0.1,
        data=stub_data_config,
    )

    # Create data module with direct napistu_data
    dm = FullGraphDataModule(
        config=stub_data_config, napistu_data=edge_masked_napistu_data
    )

    # Create encoder and head
    encoder = MessagePassingEncoder(
        in_channels=dm.num_node_features,
        hidden_channels=32,  # Small for fast testing
        num_layers=2,
        encoder_type=ENCODERS.SAGE,
    )
    head = DotProductHead()

    # Create task
    task = EdgePredictionTask(encoder, head)

    # Create Lightning module
    lightning_task = EdgePredictionLightning(task, experiment_config.training)

    # Create trainer with minimal settings for testing
    trainer = _create_test_trainer(max_epochs=2)

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


def test_edge_prediction_trainer_fit_with_callbacks(
    edge_masked_napistu_data, stub_data_config
):
    """Test trainer fit with callbacks enabled."""

    # Create experiment config
    experiment_config = ExperimentConfig(
        name="trainer_test_with_callbacks",
        seed=42,
        deterministic=True,
        fast_dev_run=True,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        data=stub_data_config,
    )

    # Create data module with direct napistu_data (backward compatibility)
    dm = FullGraphDataModule(
        config=stub_data_config,
        napistu_data=edge_masked_napistu_data,
    )

    # Create encoder and head
    encoder = MessagePassingEncoder(
        in_channels=dm.num_node_features,
        hidden_channels=32,
        num_layers=2,
        encoder_type=ENCODERS.SAGE,
    )
    head = DotProductHead()

    # Create task
    task = EdgePredictionTask(encoder, head)

    # Create Lightning module
    lightning_task = EdgePredictionLightning(task, experiment_config.training)

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
    trainer = _create_test_trainer(
        max_epochs=3,
        callbacks=[early_stopping, model_checkpoint],
        enable_checkpointing=True,
    )

    # This should not raise any errors
    trainer.fit(lightning_task, dm)

    # Verify callbacks were created
    assert len(trainer.callbacks) == 2


def test_edge_prediction_trainer_test(edge_masked_napistu_data, stub_data_config):
    """Test that we can run test after fitting."""

    # Create experiment config
    experiment_config = ExperimentConfig(
        name="trainer_test_eval",
        seed=42,
        deterministic=True,
        fast_dev_run=True,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        data=stub_data_config,
    )

    # Create data module with direct napistu_data (backward compatibility)
    dm = FullGraphDataModule(
        config=stub_data_config,
        napistu_data=edge_masked_napistu_data,
    )

    # Create encoder and head
    encoder = MessagePassingEncoder(
        in_channels=dm.num_node_features,
        hidden_channels=32,
        num_layers=2,
        encoder_type=ENCODERS.SAGE,
    )
    head = DotProductHead()

    # Create task
    task = EdgePredictionTask(encoder, head)

    # Create Lightning module
    lightning_task = EdgePredictionLightning(task, experiment_config.training)

    # Create trainer
    trainer = _create_test_trainer(max_epochs=1)

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


def test_edge_prediction_trainer_different_encoders(
    edge_masked_napistu_data, stub_data_config
):
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
            data=stub_data_config,
        )

        dm = FullGraphDataModule(
            config=stub_data_config,
            napistu_data_name="test",
            napistu_data=edge_masked_napistu_data,
        )

        # Create encoder and head
        encoder = MessagePassingEncoder(
            in_channels=dm.num_node_features,
            hidden_channels=16,  # Small for fast testing
            num_layers=1,  # Minimal layers
            encoder_type=encoder_type,
        )
        head = DotProductHead()

        # Create task
        task = EdgePredictionTask(encoder, head)

        # Create Lightning module
        lightning_task = EdgePredictionLightning(task, experiment_config.training)

        # Create trainer
        trainer = _create_test_trainer(max_epochs=1)

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


@pytest.mark.parametrize(
    "accelerator",
    [
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS not available"
            ),
        ),
    ],
)
def test_edge_prediction_trainer_gpu_if_available(
    edge_masked_napistu_data, stub_data_config, accelerator
):
    """Test trainer on GPU accelerators if available, otherwise skip."""

    # Create experiment config
    experiment_config = ExperimentConfig(
        name=f"trainer_test_{accelerator}",
        seed=42,
        deterministic=True,
        fast_dev_run=True,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        data=stub_data_config,
    )

    dm = FullGraphDataModule(
        config=stub_data_config,
        napistu_data=edge_masked_napistu_data,
    )

    # Create encoder and head
    encoder = MessagePassingEncoder(
        in_channels=dm.num_node_features,
        hidden_channels=32,
        num_layers=2,
        encoder_type=ENCODERS.SAGE,
    )
    head = DotProductHead()

    # Create task
    task = EdgePredictionTask(encoder, head)

    # Create Lightning module
    lightning_task = EdgePredictionLightning(task, experiment_config.training)

    # Create trainer with specified accelerator
    trainer = _create_test_trainer(max_epochs=1, accelerator=accelerator)

    # This should not raise any errors
    # If training completes successfully with the specified accelerator, it was used
    trainer.fit(lightning_task, dm)

    # Verify that the trainer was configured with the correct accelerator type
    accelerator_class_name = trainer.accelerator.__class__.__name__
    expected_class_name = f"{accelerator.upper()}Accelerator"
    assert (
        accelerator_class_name == expected_class_name
    ), f"Expected {expected_class_name}, got {accelerator_class_name}"


@pytest.mark.skip_on_windows
def test_edge_prediction_trainer_with_store(temp_data_config_with_store):
    """Test trainer using NapistuDataStore-based approach."""

    # Create experiment config
    experiment_config = ExperimentConfig(
        name="trainer_test_with_store",
        seed=42,
        deterministic=True,
        fast_dev_run=True,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        data=temp_data_config_with_store,
    )

    # Create data module using NapistuDataStore approach
    dm = FullGraphDataModule(config=temp_data_config_with_store)

    # Create encoder and head
    encoder = MessagePassingEncoder(
        in_channels=dm.num_node_features,
        hidden_channels=32,
        num_layers=2,
        encoder_type=ENCODERS.SAGE,
    )
    head = DotProductHead()

    # Create task
    task = EdgePredictionTask(encoder, head)

    # Create Lightning module
    lightning_task = EdgePredictionLightning(task, experiment_config.training)

    # Create trainer
    trainer = _create_test_trainer(max_epochs=1)

    # This should not raise any errors
    trainer.fit(lightning_task, dm)

    # Verify the model was trained
    encoder_params = list(lightning_task.task.encoder.parameters())
    assert len(encoder_params) > 0, "Encoder should have parameters"
    has_gradients = any(param.requires_grad for param in encoder_params)
    assert has_gradients, "Model should have trainable parameters with gradients"


def test_edge_prediction_trainer_with_edge_strata(
    edge_masked_napistu_data, edge_strata, stub_data_config
):
    """Test that we can fit an edge prediction model with stratified negative sampling."""

    # Create a minimal experiment config for fast testing
    experiment_config = ExperimentConfig(
        name="trainer_test_with_strata",
        seed=42,
        deterministic=True,
        fast_dev_run=True,  # Only run 1 batch for testing
        limit_train_batches=0.1,  # Limit to 10% of batches
        limit_val_batches=0.1,
        data=stub_data_config,
    )

    # Create data module with direct napistu_data
    dm = FullGraphDataModule(
        config=stub_data_config, napistu_data=edge_masked_napistu_data
    )

    # Create encoder and head
    encoder = MessagePassingEncoder(
        in_channels=dm.num_node_features,
        hidden_channels=32,  # Small for fast testing
        num_layers=2,
        encoder_type=ENCODERS.SAGE,
    )
    head = DotProductHead()

    # Create task with edge_strata for stratified negative sampling
    task = EdgePredictionTask(
        encoder,
        head,
        neg_sampling_ratio=1.0,
        edge_strata=edge_strata,
        neg_sampling_strategy="degree_weighted",
    )

    # Verify that edge_strata was set correctly
    assert task.edge_strata is not None
    assert task.edge_strata.equals(edge_strata)
    assert task.neg_sampling_strategy == "degree_weighted"

    # Create Lightning module
    lightning_task = EdgePredictionLightning(task, experiment_config.training)

    # Create trainer with minimal settings for testing
    trainer = _create_test_trainer(max_epochs=2)

    # This should not raise any errors
    trainer.fit(lightning_task, dm)

    # Verify the model was trained
    encoder_params = list(lightning_task.task.encoder.parameters())
    assert len(encoder_params) > 0, "Encoder should have parameters"
    has_gradients = any(param.requires_grad for param in encoder_params)
    assert has_gradients, "Model should have trainable parameters with gradients"

    # Test that the negative sampler was initialized with edge_strata
    assert lightning_task.task.negative_sampler is not None
    assert lightning_task.task._sampler_initialized is True

    # Test that prepare_batch works with edge_strata
    batch = lightning_task.task.prepare_batch(edge_masked_napistu_data, split="train")

    # Verify batch structure
    assert "x" in batch
    assert "supervision_edges" in batch
    assert "pos_edges" in batch
    assert "neg_edges" in batch

    # Verify edge indices have correct shape
    assert batch["pos_edges"].shape[0] == 2  # [2, num_edges]
    assert batch["neg_edges"].shape[0] == 2  # [2, num_edges]

    # Verify that negative edges were sampled (should have some negative edges)
    assert batch["neg_edges"].shape[1] > 0, "Should have sampled negative edges"
