"""Tests for Lightning DataModule functionality."""

import pytest
import torch

from napistu_torch.configs import ExperimentConfig, ModelConfig, TaskConfig
from napistu_torch.constants import PYG
from napistu_torch.lightning.full_graph_datamodule import FullGraphDataModule
from napistu_torch.ml.constants import (
    SPLIT_TO_MASK,
    TRAINING,
)
from napistu_torch.models.constants import ENCODERS
from napistu_torch.napistu_data import NapistuData
from napistu_torch.tasks.constants import TASKS


def test_datamodule_returns_dataloaders(edge_masked_napistu_data, experiment_config):
    """Test that FullGraphDataModule returns proper DataLoader objects."""

    dm = FullGraphDataModule(
        config=experiment_config, napistu_data=edge_masked_napistu_data
    )
    dm.setup()

    # Test that dataloaders return DataLoader objects
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()

    assert isinstance(train_dl, torch.utils.data.DataLoader)
    assert isinstance(val_dl, torch.utils.data.DataLoader)
    assert isinstance(test_dl, torch.utils.data.DataLoader)

    # Test that DataLoader yields NapistuData objects
    train_batch = next(iter(train_dl))
    val_batch = next(iter(val_dl))
    test_batch = next(iter(test_dl))

    assert isinstance(train_batch, NapistuData)
    assert isinstance(val_batch, NapistuData)
    assert isinstance(test_batch, NapistuData)

    # Test that masks are accessible
    assert hasattr(train_batch, SPLIT_TO_MASK[TRAINING.TRAIN])
    assert hasattr(val_batch, SPLIT_TO_MASK[TRAINING.VALIDATION])
    assert hasattr(test_batch, SPLIT_TO_MASK[TRAINING.TEST])


def test_datamodule_num_node_features(edge_masked_napistu_data, experiment_config):
    """Test that num_node_features property works correctly."""
    dm = FullGraphDataModule(
        config=experiment_config, napistu_data=edge_masked_napistu_data
    )

    # Should work before setup
    expected_features = edge_masked_napistu_data.num_node_features
    assert dm.num_node_features == expected_features

    # Should still work after setup
    dm.setup()
    assert dm.num_node_features == expected_features


def test_datamodule_setup_idempotent(edge_masked_napistu_data, experiment_config):
    """Test that setup can be called multiple times safely."""
    dm = FullGraphDataModule(
        config=experiment_config, napistu_data=edge_masked_napistu_data
    )

    # First setup
    dm.setup()
    first_data = dm.data

    # Second setup should not change anything
    dm.setup()
    second_data = dm.data

    assert first_data is second_data  # Should be the same object


@pytest.mark.skip_on_windows
def test_datamodule_trimming_from_store(
    temp_data_config_with_store, edge_masked_napistu_data
):
    """Test that loading napistu_data from store applies attribute trimming based on config."""
    # Verify original data in store has expected attributes
    assert hasattr(
        edge_masked_napistu_data, PYG.EDGE_ATTR
    ), "Original data should have edge_attr"
    assert hasattr(
        edge_masked_napistu_data, SPLIT_TO_MASK[TRAINING.TRAIN]
    ), "Original data should have train_mask"
    assert hasattr(
        edge_masked_napistu_data, SPLIT_TO_MASK[TRAINING.VALIDATION]
    ), "Original data should have val_mask"
    assert hasattr(
        edge_masked_napistu_data, SPLIT_TO_MASK[TRAINING.TEST]
    ), "Original data should have test_mask"

    # Test 1: Edge prediction without edge encoder -> should trim edge_attr, keep masks
    config = ExperimentConfig(
        name="test_trimming",
        data=temp_data_config_with_store,
        task=TaskConfig(
            task=TASKS.EDGE_PREDICTION, edge_prediction_neg_sampling_stratify_by="none"
        ),
        model=ModelConfig(use_edge_encoder=False),
    )

    dm = FullGraphDataModule(config=config)
    dm.setup()

    loaded_data = dm.data
    # edge_attr is always present but should have 0 features when trimmed
    assert (
        loaded_data.num_edge_features == 0
    ), "edge_attr should be trimmed (0 features) when use_edge_encoder=False"
    assert hasattr(
        loaded_data, SPLIT_TO_MASK[TRAINING.TRAIN]
    ), "train_mask should be kept for edge_prediction task"
    assert hasattr(
        loaded_data, SPLIT_TO_MASK[TRAINING.VALIDATION]
    ), "val_mask should be kept for edge_prediction task"
    assert hasattr(
        loaded_data, SPLIT_TO_MASK[TRAINING.TEST]
    ), "test_mask should be kept for edge_prediction task"

    # Test 2: Edge encoder that supports weighting -> should keep edge_attr
    config_edge_encoder = ExperimentConfig(
        name="test_trimming_edge_encoder",
        data=temp_data_config_with_store,
        task=TaskConfig(
            task=TASKS.EDGE_PREDICTION, edge_prediction_neg_sampling_stratify_by="none"
        ),
        model=ModelConfig(use_edge_encoder=True, encoder=ENCODERS.GCN),
    )

    dm_edge_encoder = FullGraphDataModule(config=config_edge_encoder)
    dm_edge_encoder.setup()

    loaded_data_edge_encoder = dm_edge_encoder.data
    # edge_attr should have features when keep_edge_attr=True
    assert (
        loaded_data_edge_encoder.num_edge_features > 0
    ), "edge_attr should be kept (have features) when use_edge_encoder=True with supported encoder"
    assert hasattr(
        loaded_data_edge_encoder, SPLIT_TO_MASK[TRAINING.TRAIN]
    ), "train_mask should be kept for edge_prediction task"
