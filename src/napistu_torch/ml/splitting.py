from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from napistu_torch.ml.constants import SPLIT_TO_MASK, TRAINING


def train_test_val_split(
    df: pd.DataFrame,
    train_size: float = 0.7,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    shuffle: bool = True,
    stratify: pd.Series | None = None,
    return_dict: bool = False,
) -> (
    pd.DataFrame
    | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    | dict[str, pd.DataFrame]
):
    """
    Split DataFrame into train, test, and validation sets.

    This is an extension of sklearn's train_test_split for three-way splits.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to split
    train_size : float, default=0.7
        Proportion of data for training (0.0 to 1.0)
    test_size : float, default=0.15
        Proportion of data for testing (0.0 to 1.0)
    val_size : float, default=0.15
        Proportion of data for validation (0.0 to 1.0)
    random_state : int, default=42
        Random seed for reproducibility
    shuffle : bool, default=True
        Whether to shuffle the data before splitting
    stratify : array-like, optional
        If not None, data is split in a stratified fashion using this as class labels
    return_dict : bool, default=False
        If True, return a dictionary with keys for each split

    Returns
    -------
    If return_dict is False:

    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    val_df : pd.DataFrame
        Validation data

    If return_dict is True:
        dict
            A dictionary with keys for each split
            - train : pd.DataFrame
                Training data
            - test : pd.DataFrame
                Test data
            - val : pd.DataFrame
                Validation data

    Examples
    --------
    >>> # Basic usage
    >>> train, test, val = train_test_val_split(df)
    >>>
    >>> # Custom split ratios
    >>> train, test, val = train_test_val_split(df, train_size=0.8, test_size=0.1, val_size=0.1)
    >>>
    >>> # Stratified split by edge type
    >>> train, test, val = train_test_val_split(df, stratify=df['edge_type'])
    >>>
    >>> # No shuffling (preserve order)
    >>> train, test, val = train_test_val_split(df, shuffle=False)
    """
    # Validate proportions sum to 1.0
    if not np.isclose(train_size + test_size + val_size, 1.0):
        raise ValueError(
            f"train_size, test_size, and val_size must sum to 1.0, "
            f"got {train_size + test_size + val_size}"
        )

    # First split: separate out training data
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )

    # Second split: split remaining data into test and val
    if val_size == 0:
        # If no validation set needed, all remaining data goes to test
        test_df = temp_df
        val_df = pd.DataFrame(columns=df.columns)  # Empty DataFrame with same columns
    else:
        # Adjust test_size to be relative to the remaining data
        relative_test_size = test_size / (test_size + val_size)

        # Handle stratification for second split
        stratify_temp = None
        if stratify is not None:
            stratify_temp = stratify.loc[temp_df.index]

        test_df, val_df = train_test_split(
            temp_df,
            train_size=relative_test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_temp,
        )

    if return_dict:
        return {
            TRAINING.TRAIN: train_df,
            TRAINING.TEST: test_df,
            TRAINING.VALIDATION: val_df,
        }

    return train_df, test_df, val_df


def create_split_masks(
    df: pd.DataFrame, splits_dict: Dict[str, pd.DataFrame]
) -> Dict[str, torch.Tensor]:
    """
    Create train/test/val masks from split DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Full  DataFrame (before splitting)
    splits_dict : Dict[str, pd.DataFrame]
        Dictionary with 'train', 'test', 'validation' keys containing split DataFrames

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with 'train_mask', 'test_mask', 'validation_mask' boolean tensors
    """
    n = df.shape[0]
    masks = {}

    for split_name in [TRAINING.TRAIN, TRAINING.TEST, TRAINING.VALIDATION]:
        mask = torch.zeros(n, dtype=torch.bool)
        if split_name in splits_dict and not splits_dict[split_name].empty:
            mask[splits_dict[split_name].index] = True
        masks[SPLIT_TO_MASK[split_name]] = mask

    return masks
