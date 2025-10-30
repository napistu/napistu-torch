# src/napistu_torch/data/dataset.py
"""Datasets for edge prediction training."""

import torch
from torch.utils.data import Dataset

from napistu_torch.napistu_data import NapistuData


class SingleGraphDataset(Dataset):
    """
    Wrapper to make a single NapistuData object work with DataLoader.

    This is necessary because DataLoader expects a Dataset interface.
    For full-batch training on a single graph, this just returns the same
    graph every time (batch_size should be 1).
    """

    def __init__(self, data: NapistuData):
        self.data = data

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data


class EdgeBatchDataset(Dataset):
    """
    Dataset that splits edge indices into mini-batches for training.

    Unlike sharding (parallel processing), batches are processed sequentially
    with weight updates between them.

    Parameters
    ----------
    edge_indices : torch.Tensor
        All edge indices to batch [num_edges]
    batches_per_epoch : int
        Number of mini-batches per epoch

    Examples
    --------
    >>> # 80M training edges split into 10 mini-batches
    >>> train_indices = torch.where(data.train_mask)[0]
    >>> dataset = EdgeBatchDataset(train_indices, batches_per_epoch=10)
    >>> len(dataset)  # 10
    >>> dataset[0].shape  # torch.Size([8000000]) - first mini-batch
    """

    def __init__(self, edge_indices: torch.Tensor, batches_per_epoch: int):
        self.edge_indices = edge_indices
        self.batches_per_epoch = batches_per_epoch

        # Calculate batch size
        self.batch_size = len(edge_indices) // batches_per_epoch

    def __len__(self) -> int:
        """Number of mini-batches."""
        return self.batches_per_epoch

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get one mini-batch of edge indices.

        Parameters
        ----------
        idx : int
            Batch index (0 to batches_per_epoch-1)

        Returns
        -------
        torch.Tensor
            Edge indices for this mini-batch
        """
        start = idx * self.batch_size

        # Last batch gets any remaining edges
        if idx == self.batches_per_epoch - 1:
            end = len(self.edge_indices)
        else:
            end = start + self.batch_size

        return self.edge_indices[start:end]
