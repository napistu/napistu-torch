"""Tests for Lightning DataModule functionality."""

import pytest
import torch

from napistu_torch.lightning.data_module import NapistuDataModule
from napistu_torch.napistu_data import NapistuData

from napistu_torch.ml.constants import (
    SPLIT_TO_MASK,
    TRAINING,
)


def test_datamodule_returns_dataloaders(edge_masked_napistu_data, data_config):
    """Test that NapistuDataModule returns proper DataLoader objects."""

    dm = NapistuDataModule(edge_masked_napistu_data, data_config)
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


def test_datamodule_num_node_features(edge_masked_napistu_data, data_config):
    """Test that num_node_features property works correctly."""
    dm = NapistuDataModule(edge_masked_napistu_data, data_config)

    # Should work before setup
    expected_features = edge_masked_napistu_data.num_node_features
    assert dm.num_node_features == expected_features

    # Should still work after setup
    dm.setup()
    assert dm.num_node_features == expected_features


def test_datamodule_setup_idempotent(edge_masked_napistu_data, data_config):
    """Test that setup can be called multiple times safely."""
    dm = NapistuDataModule(edge_masked_napistu_data, data_config)

    # First setup
    dm.setup()
    first_data = dm.data

    # Second setup should not change anything
    dm.setup()
    second_data = dm.data

    assert first_data is second_data  # Should be the same object
