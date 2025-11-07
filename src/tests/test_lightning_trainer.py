"""Tests for Lightning trainer integration - actual model fitting."""

from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from napistu_torch.configs import DataConfig, ExperimentConfig, TaskConfig
from napistu_torch.lightning.edge_batch_datamodule import EdgeBatchDataModule
from napistu_torch.lightning.full_graph_datamodule import FullGraphDataModule
from napistu_torch.lightning.tasks import EdgePredictionLightning
from napistu_torch.load.constants import STRATIFY_BY
from napistu_torch.models.constants import ENCODERS
from napistu_torch.models.heads import DotProductHead
from napistu_torch.models.message_passing_encoder import MessagePassingEncoder
from napistu_torch.tasks.edge_prediction import EdgePredictionTask


@pytest.fixture
def stub_data_config():
    """Create a stubbed DataConfig for testing."""
    return DataConfig(
        sbml_dfs_path=Path("stub_sbml.pkl"),
        napistu_graph_path=Path("stub_graph.pkl"),
        napistu_data_name="edge_prediction",
    )


def _create_test_experiment_config(
    name: str,
    data_config: DataConfig,
    limit_train_batches: float = 0.1,
    limit_val_batches: float = 0.1,
    fast_dev_run: bool = True,
    **kwargs,
) -> ExperimentConfig:
    """Create a standardized ExperimentConfig for testing.

    This function abstracts away the common ExperimentConfig setup patterns
    used across multiple tests, reducing duplication and making tests more
    maintainable.

    Parameters
    ----------
    name : str
        Name for the experiment
    data_config : DataConfig
        Data configuration to use
    limit_train_batches : float, default=0.1
        Fraction of training batches to use (0.1 = 10%)
    limit_val_batches : float, default=0.1
        Fraction of validation batches to use (0.1 = 10%)
    fast_dev_run : bool, default=True
        Whether to use fast dev run mode
    **kwargs
        Additional keyword arguments to pass to ExperimentConfig

    Returns
    -------
    ExperimentConfig
        Configured experiment config for testing

    Examples
    --------
    Basic usage:
    >>> config = create_test_experiment_config("test", data_config)

    With custom parameters:
    >>> config = create_test_experiment_config(
    ...     "test", data_config,
    ...     fast_dev_run=False,
    ...     max_epochs=5,
    ...     limit_train_batches=1.0
    ... )
    """
    return ExperimentConfig(
        name=name,
        seed=42,
        deterministic=True,
        fast_dev_run=fast_dev_run,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        data=data_config,
        task=TaskConfig(edge_prediction_neg_sampling_stratify_by="none"),
        **kwargs,
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
    experiment_config = _create_test_experiment_config(
        name="trainer_test",
        data_config=stub_data_config,
        limit_train_batches=0.1,  # Limit to 10% of batches
        limit_val_batches=0.1,
    )

    # Create data module with direct napistu_data
    dm = FullGraphDataModule(
        config=experiment_config, napistu_data=edge_masked_napistu_data
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
    experiment_config = _create_test_experiment_config(
        name="trainer_test_with_callbacks",
        data_config=stub_data_config,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
    )

    # Create data module with direct napistu_data
    dm = FullGraphDataModule(
        config=experiment_config,
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
    experiment_config = _create_test_experiment_config(
        name="trainer_test_eval",
        data_config=stub_data_config,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
    )

    # Create data module with direct napistu_data
    dm = FullGraphDataModule(
        config=experiment_config,
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
        experiment_config = _create_test_experiment_config(
            name=f"trainer_test_{encoder_type}",
            data_config=stub_data_config,
            limit_train_batches=0.05,  # Even smaller for multiple tests
            limit_val_batches=0.05,
        )

        dm = FullGraphDataModule(
            config=experiment_config,
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
    experiment_config = _create_test_experiment_config(
        name=f"trainer_test_{accelerator}",
        data_config=stub_data_config,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
    )

    dm = FullGraphDataModule(
        config=experiment_config,
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
    experiment_config = _create_test_experiment_config(
        name="trainer_test_with_store",
        data_config=temp_data_config_with_store,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
    )

    # Create data module using NapistuDataStore approach
    dm = FullGraphDataModule(config=experiment_config)

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
        # set this temporarily here to avoid the dependency on a working NapistuDataStore for retriving strata. We're pasing them in directly.
        task=TaskConfig(edge_prediction_neg_sampling_stratify_by="none"),
    )

    # Create data module with direct napistu_data
    dm = FullGraphDataModule(
        config=experiment_config, napistu_data=edge_masked_napistu_data
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


def test_edge_prediction_trainer_with_edge_batch_datamodule(
    edge_masked_napistu_data, stub_data_config
):
    """Test full training with EdgeBatchDataModule for mini-batch updates."""

    # Create experiment config
    experiment_config = _create_test_experiment_config(
        name="trainer_test_edge_batch", data_config=stub_data_config
    )

    # Create mini-batch datamodule
    batches_per_epoch = 5
    dm = EdgeBatchDataModule(
        config=experiment_config,
        napistu_data=edge_masked_napistu_data,
        batches_per_epoch=batches_per_epoch,
        shuffle=True,
    )

    # replace the Task config with the correct one
    # we make this change here to avoid the dependency on a working NapistuDataStore for retriving strata in EdgeBatchDataModule.
    # we'll be passing the strata in directly to the task.
    experiment_config.task = TaskConfig(
        edge_prediction_neg_sampling_stratify_by=STRATIFY_BY.NODE_TYPE
    )

    # Create encoder, head, task
    encoder = MessagePassingEncoder(
        in_channels=edge_masked_napistu_data.num_node_features,
        hidden_channels=32,
        num_layers=2,
        encoder_type=ENCODERS.SAGE,
    )
    head = DotProductHead()
    task = EdgePredictionTask(encoder, head, neg_sampling_ratio=1.0)
    lightning_task = EdgePredictionLightning(task, experiment_config.training)

    # Track training steps
    class StepCounter(pl.Callback):
        def __init__(self):
            self.train_steps = 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.train_steps += 1

    step_counter = StepCounter()

    # Train
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[step_counter],
    )
    trainer.fit(lightning_task, dm)

    # Verify multiple gradient updates per epoch
    assert (
        step_counter.train_steps == batches_per_epoch
    ), f"Expected {batches_per_epoch} training steps, got {step_counter.train_steps}"

    # Verify model was trained
    encoder_params = list(lightning_task.task.encoder.parameters())
    assert len(encoder_params) > 0
    assert any(param.requires_grad for param in encoder_params)

    # Verify validation works
    val_results = trainer.validate(lightning_task, dm)
    assert "val_auc" in val_results[0]
    assert 0 <= val_results[0]["val_auc"] <= 1
