"""
Lightning DataModule for Napistu data.

This is a thin adapter that works with pre-processed NapistuData objects.
All data loading/splitting logic should be done using napistu_torch.load.napistu_graphs
functions before passing the data to this DataModule (prevents data leakage).
"""

from typing import Optional, Union, Dict
import pytorch_lightning as pl

from napistu_torch.napistu_data import NapistuData
from napistu_torch.load.constants import SPLITTING_STRATEGIES
from napistu_torch.configs import DataConfig


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
        'train', 'val', 'test' (or subset thereof).
    config : DataConfig
        Pydantic data configuration (used for validation and logging)

    Examples
    --------
    >>> # Using pre-processed transductive data
    >>> data = construct_unsupervised_pyg_data(sbml_dfs, ng, splitting_strategy='edge_mask')
    >>> config = DataConfig(splitting_strategy='edge_mask', train_size=0.7)
    >>> dm = NapistuDataModule(data, config)
    >>>
    >>> # Using pre-processed inductive data
    >>> data_dict = construct_unsupervised_pyg_data(sbml_dfs, ng, splitting_strategy='inductive')
    >>> config = DataConfig(splitting_strategy='inductive')
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
        if self.config.splitting_strategy == SPLITTING_STRATEGIES.INDUCTIVE:
            # For inductive splits, get features from training data
            if isinstance(self.napistu_data, dict):
                return self.napistu_data["train"].num_node_features
            else:
                raise ValueError(
                    f"For inductive splitting strategy, napistu_data must be a dictionary "
                    f"with keys 'train', 'val', 'test', but got {type(self.napistu_data)}"
                )
        else:
            # For transductive splits, get features from the single data object
            if isinstance(self.napistu_data, NapistuData):
                return self.napistu_data.num_node_features
            else:
                raise ValueError(
                    f"For transductive splitting strategies, napistu_data must be a single "
                    f"NapistuData object, but got {type(self.napistu_data)}"
                )

    def setup(self, stage: Optional[str] = None):
        """
        Set up NapistuData object(s) from the provided data.

        Uses the pre-processed NapistuData object(s) passed to the constructor.
        No data creation or processing happens here - just assignment.
        """
        if self.data is not None or self.train_data is not None:
            return  # Already set up

        # Handle different splitting strategies
        if self.config.splitting_strategy == SPLITTING_STRATEGIES.INDUCTIVE:
            # Separate graphs for each split
            if isinstance(self.napistu_data, dict):
                self.train_data = self.napistu_data["train"]
                self.val_data = self.napistu_data.get("val")
                self.test_data = self.napistu_data.get("test")
            else:
                raise ValueError(
                    f"For inductive splitting strategy, napistu_data must be a dictionary "
                    f"with keys 'train', 'val', 'test', but got {type(self.napistu_data)}"
                )
        else:
            # Single graph with masks (transductive)
            if isinstance(self.napistu_data, NapistuData):
                self.data = self.napistu_data
            else:
                raise ValueError(
                    f"For transductive splitting strategies, napistu_data must be a single "
                    f"NapistuData object, but got {type(self.napistu_data)}"
                )

    def train_dataloader(self):
        """Return training data (list with single item for iteration)."""
        if self.config.splitting_strategy == SPLITTING_STRATEGIES.INDUCTIVE:
            return [self.train_data]
        else:
            return [self.data]  # Lightning module uses train_mask

    def val_dataloader(self):
        """Return validation data."""
        if self.config.splitting_strategy == SPLITTING_STRATEGIES.INDUCTIVE:
            return [self.val_data] if self.val_data else None
        else:
            return [self.data]  # Lightning module uses val_mask

    def test_dataloader(self):
        """Return test data."""
        if self.config.splitting_strategy == SPLITTING_STRATEGIES.INDUCTIVE:
            return [self.test_data] if self.test_data else None
        else:
            return [self.data]  # Lightning module uses test_mask
