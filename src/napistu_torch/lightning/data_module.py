"""
Lightning DataModule for Napistu data.

This is a thin adapter that works with pre-processed NapistuData objects.
All data loading/splitting logic should be done using napistu_torch.load.napistu_graphs
functions before passing the data to this DataModule (prevents data leakage).
"""

import logging
from typing import Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from napistu_torch.configs import DataConfig
from napistu_torch.constants import ARTIFACT_TYPES
from napistu_torch.load.artifacts import DEFAULT_ARTIFACT_REGISTRY, ArtifactDefinition
from napistu_torch.ml.constants import TRAINING
from napistu_torch.napistu_data import NapistuData
from napistu_torch.napistu_data_store import NapistuDataStore

logger = logging.getLogger(__name__)


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

    Loading Strategy:
    1. Check if artifact exists in store -> load it
    2. If not in store, check if it's in registry -> create it
    3. If neither, raise error with helpful message

    Parameters
    ----------
    config : DataConfig
        Pydantic data configuration
    napistu_data_name : str
        Name of the NapistuData artifact to use for training.
        Can be either:
        - A standard artifact from the registry (e.g., "unsupervised", "edge_prediction")
        - A custom artifact already saved in the store
    store : Optional[NapistuDataStore]
        Pre-initialized store. If None, will be created from config.
    napistu_data : Optional[NapistuData]
        Direct NapistuData object for testing/backward compatibility.
        If provided, store and napistu_data_name are ignored.
    artifact_registry : Optional[Dict[str, ArtifactDefinition]]
        Registry of artifact definitions. If None, the default registry will be used.
    overwrite_artifacts : bool, default=False
        If True, recreate artifact even if it exists

    Examples
    --------
    >>> # Using config to create everything automatically
    >>> config = DataConfig(
    ...     store_dir=".store/ecoli",
    ...     sbml_dfs_path=Path("/data/ecoli_sbml_dfs.pkl"),
    ...     napistu_graph_path=Path("/data/ecoli_ng.pkl"),
    ...     required_artifacts=["edge_prediction"]
    ... )
    >>> dm = NapistuDataModule(config, napistu_data_name="edge_prediction")
    >>>
    >>> # Using pre-initialized store
    >>> store = NapistuDataStore.from_config(config)
    >>> dm = NapistuDataModule(config, napistu_data_name="unsupervised", store=store)
    >>>
    >>> # Using direct napistu_data (for testing/backward compatibility)
    >>> dm = NapistuDataModule(config, napistu_data_name="test", napistu_data=my_napistu_data)
    """

    def __init__(
        self,
        config: DataConfig,
        napistu_data_name: str,
        store: Optional[NapistuDataStore] = None,
        napistu_data: Optional[NapistuData] = None,
        artifact_registry: Optional[
            Dict[str, ArtifactDefinition]
        ] = DEFAULT_ARTIFACT_REGISTRY,
        overwrite_artifacts: bool = False,
    ):
        super().__init__()
        self.config = config
        self.napistu_data_name = napistu_data_name
        self.artifact_registry = artifact_registry
        self.overwrite_artifacts = overwrite_artifacts

        # Handle direct napistu_data input (for testing/backward compatibility)
        if napistu_data is not None:
            logger.info("Using provided napistu_data directly")
            self.store = None  # No store needed when data is provided directly
            self.data = napistu_data
        else:
            # Create/load store
            if store is None:
                logger.info("Creating/loading store from config")
                self.store = NapistuDataStore.from_config(config)
            else:
                logger.info("Using provided store")
                self.store = store

            # Validate that we can either load or create this artifact
            # This uses the store's validation method which checks both
            # store and registry, and validates the artifact type
            self.store.validate_artifact_name(
                self.napistu_data_name,
                artifact_registry=self.artifact_registry,
                required_type=ARTIFACT_TYPES.NAPISTU_DATA,
            )

            logger.info(f"Artifact '{self.napistu_data_name}' validated successfully")

            # Load the data immediately
            self._load_data()

        # Set up train/val/test splits
        self._setup_splits()

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
        >>> dm = NapistuDataModule(config, napistu_data_name="edge_prediction")
        >>> in_channels = dm.num_node_features
        >>> encoder = GNNEncoder.from_config(model_config, in_channels=in_channels)
        """
        # Determine splitting strategy from the data itself
        if isinstance(self.data, dict):
            # For inductive splits, get features from training data
            return self.data[TRAINING.TRAIN].num_node_features
        elif isinstance(self.data, NapistuData):
            # For transductive splits, get features from the single data object
            return self.data.num_node_features
        else:
            raise ValueError(
                f"data must be either a NapistuData object or a dictionary "
                f"with keys {TRAINING.TRAIN}, {TRAINING.VALIDATION}, {TRAINING.TEST}, "
                f"but got {type(self.data)}"
            )

    def setup(self, stage: Optional[str] = None):
        """
        Set up NapistuData object(s) - data is already loaded during __init__.

        This method is kept for Lightning compatibility but does nothing
        since data loading happens during initialization.
        """
        # Data is already loaded and splits are already set up in __init__
        pass

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

    # private methods

    def _load_data(self) -> None:
        """Load data from the store."""
        # Check if artifact exists in store
        if self.napistu_data_name not in self.store.list_napistu_datas():
            # Not in store - create it using registry
            logger.info(f"Creating artifact '{self.napistu_data_name}' from registry")
            self.store.ensure_artifacts(
                [self.napistu_data_name],
                artifact_registry=self.artifact_registry,
                overwrite=self.overwrite_artifacts,
            )

        # Load it (now guaranteed to exist)
        logger.info(f"Loading artifact '{self.napistu_data_name}' from store")
        self.data = self.store.load_napistu_data(self.napistu_data_name)

    def _setup_splits(self) -> None:
        """Set up train/val/test splits based on data type."""
        if isinstance(self.data, dict):
            # Separate graphs for each split (inductive)
            self.train_data = self.data[TRAINING.TRAIN]
            self.val_data = self.data.get(TRAINING.VALIDATION)
            self.test_data = self.data.get(TRAINING.TEST)
        elif isinstance(self.data, NapistuData):
            # Single graph with masks (transductive)
            # train_data, val_data, test_data remain None - will use self.data
            self.train_data = None
            self.val_data = None
            self.test_data = None
        else:
            raise ValueError(
                f"data must be either a NapistuData object or a dictionary "
                f"with keys {TRAINING.TRAIN}, {TRAINING.VALIDATION}, {TRAINING.TEST}, "
                f"but got {type(self.data)}"
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
