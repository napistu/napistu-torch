"""
Lightning DataModule for Napistu data.

This is a thin adapter that works with pre-processed NapistuData objects.
All data loading/splitting logic should be done using napistu_torch.load.napistu_graphs
functions before passing the data to this DataModule (prevents data leakage).
"""

from typing import Dict, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from napistu_torch.configs import DataConfig
from napistu_torch.ml.constants import TRAINING
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


class NapistuDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for Napistu biological networks.

    This is a thin adapter that works with existing NapistuData objects.
    The data should be pre-processed and split using napistu_torch.load.napistu_graphs
    functions before being passed to this DataModule.

    Supports both transductive (mask-based) and inductive (separate graphs) splits.

    Parameters
    ----------
    napistu_data : Union[NapistuData, Dict[str, NapistuData]]
        Pre-processed NapistuData object(s). For transductive splits, pass a single
        NapistuData object. For inductive splits, pass a dictionary with keys
        TRAINING.TRAIN, TRAINING.VALIDATION, TRAINING.TEST (or subset thereof).
    config : DataConfig
        Pydantic data configuration (used for validation and logging)

    Examples
    --------
    >>> # Using pre-processed transductive data
    >>> data = construct_unsupervised_pyg_data(sbml_dfs, ng, splitting_strategy='edge_mask')
    >>> config = DataConfig(sbml_dfs_path=Path("data.pkl"), napistu_graph_path=Path("graph.pkl"))
    >>> dm = NapistuDataModule(data, config)
    >>>
    >>> # Using pre-processed inductive data
    >>> data_dict = construct_unsupervised_pyg_data(sbml_dfs, ng, splitting_strategy='inductive')
    >>> config = DataConfig(sbml_dfs_path=Path("data.pkl"), napistu_graph_path=Path("graph.pkl"))
    >>> dm = NapistuDataModule(data_dict, config)
    """

    def __init__(
        self,
        napistu_data: Union[NapistuData, Dict[str, NapistuData]],
        config: DataConfig,
    ):
        super().__init__()
        self.napistu_data = napistu_data
        self.config = config

        # Will be set in setup()
        self.data: Optional[NapistuData] = None
        self.train_data: Optional[NapistuData] = None
        self.val_data: Optional[NapistuData] = None
        self.test_data: Optional[NapistuData] = None

    @property
    def num_node_features(self) -> int:
        """
        Get the number of node features from the data.

        This works before setup() is called by accessing the raw napistu_data.
        For inductive splits, returns the number of features from the training data.

        Returns
        -------
        int
            Number of node features (in_channels for encoders)

        Examples
        --------
        >>> dm = NapistuDataModule(napistu_data, config)
        >>> in_channels = dm.num_node_features
        >>> encoder = GNNEncoder.from_config(model_config, in_channels=in_channels)
        """
        # Determine splitting strategy from the data itself
        if isinstance(self.napistu_data, dict):
            # For inductive splits, get features from training data
            return self.napistu_data[TRAINING.TRAIN].num_node_features
        elif isinstance(self.napistu_data, NapistuData):
            # For transductive splits, get features from the single data object
            return self.napistu_data.num_node_features
        else:
            raise ValueError(
                f"napistu_data must be either a NapistuData object or a dictionary "
                f"with keys {TRAINING.TRAIN}, {TRAINING.VALIDATION}, {TRAINING.TEST}, "
                f"but got {type(self.napistu_data)}"
            )

    def setup(self, stage: Optional[str] = None):
        """
        Set up NapistuData object(s) from the provided data.

        Uses the pre-processed NapistuData object(s) passed to the constructor.
        No data creation or processing happens here - just assignment.
        """
        if self.data is not None or self.train_data is not None:
            return  # Already set up

        # Determine splitting strategy from the data structure
        if isinstance(self.napistu_data, dict):
            # Separate graphs for each split (inductive)
            self.train_data = self.napistu_data[TRAINING.TRAIN]
            self.val_data = self.napistu_data.get(TRAINING.VALIDATION)
            self.test_data = self.napistu_data.get(TRAINING.TEST)
        elif isinstance(self.napistu_data, NapistuData):
            # Single graph with masks (transductive)
            self.data = self.napistu_data
        else:
            raise ValueError(
                f"napistu_data must be either a NapistuData object or a dictionary "
                f"with keys {TRAINING.TRAIN}, {TRAINING.VALIDATION}, {TRAINING.TEST}, "
                f"but got {type(self.napistu_data)}"
            )

    def train_dataloader(self):
        """
        Return a DataLoader for training.

        For single-graph training, this returns a DataLoader with batch_size=1
        that yields the same graph on each iteration. The Lightning Trainer
        will only iterate once per epoch.
        """
        if self.train_data is not None:
            # Inductive split - use separate training data
            dataset = SingleGraphDataset(self.train_data)
        else:
            # Transductive split - use single graph with masks
            dataset = SingleGraphDataset(self.data)

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,  # No shuffling for single graph
            collate_fn=identity_collate,  # Don't use PyG's batching
            num_workers=0,  # Single graph, no benefit from workers
        )

    def val_dataloader(self):
        """Return a DataLoader for validation."""
        if self.val_data is not None:
            # Inductive split - use separate validation data
            dataset = SingleGraphDataset(self.val_data)
        else:
            # Transductive split - use single graph with masks
            dataset = SingleGraphDataset(self.data)

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=identity_collate,
            num_workers=0,
        )

    def test_dataloader(self):
        """Return a DataLoader for testing."""
        if self.test_data is not None:
            # Inductive split - use separate test data
            dataset = SingleGraphDataset(self.test_data)
        else:
            # Transductive split - use single graph with masks
            dataset = SingleGraphDataset(self.data)

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=identity_collate,
            num_workers=0,
        )

    def predict_dataloader(self):
        """Return a DataLoader for prediction."""
        if self.test_data is not None:
            # Inductive split - use separate test data for prediction
            dataset = SingleGraphDataset(self.test_data)
        else:
            # Transductive split - use single graph with masks
            dataset = SingleGraphDataset(self.data)

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=identity_collate,
            num_workers=0,
        )


def identity_collate(batch):
    """
    Custom collate function that returns the NapistuData object unchanged.

    For single-graph training, we don't want PyG's batching behavior.

    This function is STRICT - it expects exactly what SingleGraphDataset
    produces: a list with one NapistuData object.
    """
    assert isinstance(batch, list), f"Expected list, got {type(batch)}"
    assert len(batch) == 1, f"Expected batch of size 1, got {len(batch)}"
    assert isinstance(
        batch[0], NapistuData
    ), f"Expected NapistuData, got {type(batch[0])}"
    return batch[0]
