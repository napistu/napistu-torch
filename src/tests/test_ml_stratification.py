"""Tests for stratification functions."""

import pandas as pd
import pytest
import torch

from napistu_torch.ml.constants import TRAINING
from napistu_torch.ml.stratification import create_split_masks, train_test_val_split


def test_train_test_val_split_basic():
    """Test basic train/test/val split returns correct proportions."""
    df = pd.DataFrame({"value": range(100)})

    train, test, val = train_test_val_split(
        df, train_size=0.7, test_size=0.15, val_size=0.15
    )

    assert len(train) == 70
    assert len(test) == 15
    assert len(val) == 15
    assert len(train) + len(test) + len(val) == len(df)


def test_train_test_val_split_return_dict():
    """Test return_dict option returns dictionary."""
    df = pd.DataFrame({"value": range(100)})

    splits = train_test_val_split(df, return_dict=True)

    assert isinstance(splits, dict)
    assert TRAINING.TRAIN in splits
    assert TRAINING.TEST in splits
    assert TRAINING.VALIDATION in splits
    assert len(splits[TRAINING.TRAIN]) + len(splits[TRAINING.TEST]) + len(
        splits[TRAINING.VALIDATION]
    ) == len(df)


def test_train_test_val_split_invalid_proportions():
    """Test that invalid proportions raise ValueError."""
    df = pd.DataFrame({"value": range(100)})

    with pytest.raises(ValueError, match="must sum to 1.0"):
        train_test_val_split(df, train_size=0.5, test_size=0.3, val_size=0.3)


def test_create_split_masks_basic():
    """Test create_split_masks generates correct boolean masks."""
    vertex_df = pd.DataFrame({"value": range(100)})

    train, test, val = train_test_val_split(vertex_df)
    splits_dict = {TRAINING.TRAIN: train, TRAINING.TEST: test, TRAINING.VALIDATION: val}

    masks = create_split_masks(vertex_df, splits_dict)

    assert f"{TRAINING.TRAIN}_mask" in masks
    assert f"{TRAINING.TEST}_mask" in masks
    assert f"{TRAINING.VALIDATION}_mask" in masks
    assert all(isinstance(mask, torch.Tensor) for mask in masks.values())
    assert all(mask.dtype == torch.bool for mask in masks.values())
    assert all(len(mask) == len(vertex_df) for mask in masks.values())
    assert masks[f"{TRAINING.TRAIN}_mask"].sum() == len(train)
    assert masks[f"{TRAINING.TEST}_mask"].sum() == len(test)
    assert masks[f"{TRAINING.VALIDATION}_mask"].sum() == len(val)
