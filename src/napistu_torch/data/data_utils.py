# In src/napistu_torch/data/data_utils.py

"""
Utility functions for data loading and batching.

Contains shared helpers for DataLoaders and collate functions.
"""

from typing import List

from torch.utils.data import DataLoader

from napistu_torch.data.dataset import SingleGraphDataset
from napistu_torch.napistu_data import NapistuData


def identity_collate(batch: List[NapistuData]) -> NapistuData:
    """
    Custom collate function that returns NapistuData unchanged.

    For single-graph training, we don't want PyG's batching behavior.
    This function extracts the single NapistuData object from the batch list.

    Parameters
    ----------
    batch : List[NapistuData]
        Batch from DataLoader (should contain exactly 1 NapistuData object)

    Returns
    -------
    NapistuData
        The single NapistuData object

    Raises
    ------
    AssertionError
        If batch is not a list with exactly 1 NapistuData object

    Examples
    --------
    >>> dataset = SingleGraphDataset(data)
    >>> loader = DataLoader(dataset, batch_size=1, collate_fn=identity_collate)
    >>> for batch in loader:
    ...     assert isinstance(batch, NapistuData)
    """
    if not isinstance(batch, list):
        raise ValueError(f"Expected list, got {type(batch)}")
    if len(batch) != 1:
        raise ValueError(f"Expected batch of size 1, got {len(batch)}")
    if not isinstance(batch[0], NapistuData):
        raise ValueError(f"Expected NapistuData, got {type(batch[0])}")
    return batch[0]


def create_single_graph_dataloader(
    dataset: SingleGraphDataset,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for single graph datasets.

    Returns a DataLoader configured for single-graph training that:
    - Uses batch_size=1 (only one graph per batch)
    - Uses identity_collate to avoid PyG batching
    - Returns NapistuData objects directly

    Parameters
    ----------
    dataset : SingleGraphDataset
        Dataset wrapping a single NapistuData object
    shuffle : bool, optional
        Whether to shuffle (not useful for single graph). Default False.
    num_workers : int, optional
        Number of worker processes. Default 0 (single graph doesn't benefit from workers).

    Returns
    -------
    DataLoader
        Configured DataLoader that yields NapistuData objects

    Examples
    --------
    >>> dataset = SingleGraphDataset(data)
    >>> loader = create_single_graph_dataloader(dataset)
    >>> batch = next(iter(loader))
    >>> assert isinstance(batch, NapistuData)
    """
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=identity_collate,
        num_workers=num_workers,
    )
