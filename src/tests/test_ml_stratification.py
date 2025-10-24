"""Tests for stratification functions."""

import pandas as pd
import pytest
import torch

from napistu_torch.ml.constants import TRAINING, SPLIT_TO_MASK
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


def test_train_test_val_split_zero_val_size():
    """Test train_test_val_split with val_size=0 creates empty validation set."""
    df = pd.DataFrame({"value": range(100)})

    train, test, val = train_test_val_split(
        df, train_size=0.8, test_size=0.2, val_size=0.0
    )

    assert len(train) == 80
    assert len(test) == 20
    assert len(val) == 0  # Empty validation set
    assert len(train) + len(test) + len(val) == len(df)

    # Verify val DataFrame has correct columns but 0 rows
    assert list(val.columns) == list(df.columns)
    assert val.empty


def test_train_test_val_split_mutually_exclusive_collectively_exhaustive():
    """Test that splits are mutually exclusive and collectively exhaustive."""
    df = pd.DataFrame({"value": range(100)})

    splits_dict = train_test_val_split(df, return_dict=True)
    masks = create_split_masks(df, splits_dict)

    train_mask = masks[SPLIT_TO_MASK[TRAINING.TRAIN]]
    test_mask = masks[SPLIT_TO_MASK[TRAINING.TEST]]
    val_mask = masks[SPLIT_TO_MASK[TRAINING.VALIDATION]]

    # Verify masks are mutually exclusive and collectively exhaustive
    # Sum of all masks should be exactly 1 for each element
    mask_sum = train_mask.int() + test_mask.int() + val_mask.int()
    assert torch.all(mask_sum == 1)


def test_train_test_val_split_zero_val_size_masks():
    """Test that val_size=0 creates correct masks with blank val_mask."""
    df = pd.DataFrame({"value": range(100)})

    splits_dict = train_test_val_split(
        df, train_size=0.8, test_size=0.2, val_size=0.0, return_dict=True
    )
    masks = create_split_masks(df, splits_dict)

    train_mask = masks[SPLIT_TO_MASK[TRAINING.TRAIN]]
    test_mask = masks[SPLIT_TO_MASK[TRAINING.TEST]]
    val_mask = masks[SPLIT_TO_MASK[TRAINING.VALIDATION]]

    # Verify val_mask is all zeros
    assert torch.all(~val_mask)
    assert val_mask.sum().item() == 0

    # Verify train and test masks are still mutually exclusive and collectively exhaustive
    # Sum of all masks should be exactly 1 for each element
    mask_sum = train_mask.int() + test_mask.int() + val_mask.int()
    assert torch.all(mask_sum == 1)


def test_create_split_masks_basic():
    """Test create_split_masks generates correct boolean masks."""
    vertex_df = pd.DataFrame({"value": range(100)})

    train, test, val = train_test_val_split(vertex_df)
    splits_dict = {TRAINING.TRAIN: train, TRAINING.TEST: test, TRAINING.VALIDATION: val}

    masks = create_split_masks(vertex_df, splits_dict)

    assert SPLIT_TO_MASK[TRAINING.TRAIN] in masks
    assert SPLIT_TO_MASK[TRAINING.TEST] in masks
    assert SPLIT_TO_MASK[TRAINING.VALIDATION] in masks
    assert all(isinstance(mask, torch.Tensor) for mask in masks.values())
    assert all(mask.dtype == torch.bool for mask in masks.values())
    assert all(len(mask) == len(vertex_df) for mask in masks.values())
    assert masks[SPLIT_TO_MASK[TRAINING.TRAIN]].sum() == len(train)
    assert masks[SPLIT_TO_MASK[TRAINING.TEST]].sum() == len(test)
    assert masks[SPLIT_TO_MASK[TRAINING.VALIDATION]].sum() == len(val)
