"""
Lightning DataModule for Napistu data.

This is a thin adapter that works with pre-processed NapistuData objects.
All data loading/splitting logic should be done using napistu_torch.load.napistu_graphs
functions before passing the data to this DataModule (prevents data leakage).
"""

import logging
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from napistu_torch.configs import DataConfig, TaskConfig
from napistu_torch.constants import ARTIFACT_TYPES
from napistu_torch.load.artifacts import DEFAULT_ARTIFACT_REGISTRY, ArtifactDefinition
from napistu_torch.load.constants import STRATIFY_BY_ARTIFACT_NAMES
from napistu_torch.ml.constants import TRAINING
from napistu_torch.napistu_data import NapistuData
from napistu_torch.napistu_data_store import NapistuDataStore
from napistu_torch.tasks.constants import TASKS

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
    task_config : Optional[TaskConfig]
        Pydantic task configuration
    napistu_data_name : Optional[str]
        Name of the NapistuData artifact to use for training.
        If None, uses config.napistu_data_name.
        Can be either:
        - A standard artifact from the registry (e.g., "unsupervised", "edge_prediction")
        - A custom artifact already saved in the store
    other_artifacts : Optional[List[str]]
        List of other artifact names needed for the experiment.
        If None, uses config.other_artifacts.
        If None, no other artifacts will be loaded.
    napistu_data : Optional[NapistuData]
        Direct NapistuData object for testing/backward compatibility.
        If provided, napistu_data_name will be ignored and store will be ignored if other_artifacts is None.
    store : Optional[NapistuDataStore]
        Pre-initialized store. If None, will be created from config.
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
    ...     napistu_data_name="edge_prediction",
    ...     other_artifacts=["unsupervised"]
    ... )
    >>> dm = NapistuDataModule(config)
    >>>
    >>> # Using pre-initialized store
    >>> store = NapistuDataStore.from_config(config)
    >>> dm = NapistuDataModule(config, store=store)
    >>>
    >>> # Using direct napistu_data (for testing/backward compatibility)
    >>> dm = NapistuDataModule(config, napistu_data=my_napistu_data)
    """

    def __init__(
        self,
        config: DataConfig,
        task_config: Optional[TaskConfig] = None,
        napistu_data_name: Optional[str] = None,
        other_artifacts: Optional[List[str]] = None,
        napistu_data: Optional[NapistuData] = None,
        store: Optional[NapistuDataStore] = None,
        artifact_registry: Optional[
            Dict[str, ArtifactDefinition]
        ] = DEFAULT_ARTIFACT_REGISTRY,
        overwrite_artifacts: bool = False,
    ):
        super().__init__()
        self.config = config

        # Use DataConfig values if not provided explicitly (for backward compatibility)
        if napistu_data_name is None:
            napistu_data_name = config.napistu_data_name
        if other_artifacts is None:
            other_artifacts = config.other_artifacts

        # Add additional artifacts required by the task
        if task_config is not None:
            task_artifacts = _task_config_to_artifact_names(task_config)
            other_artifacts = list[str](
                set[str](other_artifacts) | set[str](task_artifacts)
            )

        # Create or load the store from config
        if store is None:
            # Determine if we need a store
            need_store = (napistu_data is None) or (len(other_artifacts) > 0)

            if need_store:
                logger.info("Creating/loading store from config")
                # Only ensure artifacts if we're not side-loading napistu_data OR we need other_artifacts
                ensure_artifacts = (napistu_data is None) or (len(other_artifacts) > 0)
                napistu_data_store = NapistuDataStore.from_config(
                    config, ensure_artifacts=ensure_artifacts
                )
            else:
                logger.info(
                    "No store needed - side-loading napistu_data with no other artifacts"
                )
                napistu_data_store = None
        else:
            if not isinstance(store, NapistuDataStore):
                raise ValueError("store must be a NapistuDataStore object")
            logger.info("Using provided store")
            napistu_data_store = store
            need_store = True

        # Handle direct napistu_data input (for testing/backward compatibility)
        if napistu_data is not None:
            logger.info("Using provided napistu_data directly")
            self.napistu_data = napistu_data
            required_artifacts = other_artifacts
        else:
            # Validate that we can either load or create this artifact
            # This uses the store's validation method which checks both
            # store and registry, and validates the artifact type
            napistu_data_store.validate_artifact_name(
                napistu_data_name,
                artifact_registry=artifact_registry,
                required_type=ARTIFACT_TYPES.NAPISTU_DATA,
            )

            required_artifacts = [napistu_data_name] + other_artifacts

        # Ensure all required artifacts exist on disk
        if need_store:
            napistu_data_store.ensure_artifacts(
                required_artifacts,
                artifact_registry=artifact_registry,
                overwrite=overwrite_artifacts,
            )

        # Load napistu_data from store if not provided directly
        if napistu_data is None:
            logger.info(f"Loading napistu_data '{napistu_data_name}' from store")
            self.napistu_data = napistu_data_store.load_napistu_data(napistu_data_name)

        # load the other artifacts from the store
        self.other_artifacts = dict[str, Any]()
        for artifact_name in other_artifacts:
            artifact_type = artifact_registry[artifact_name].artifact_type
            self.other_artifacts[artifact_name] = napistu_data_store.load_artifact(
                artifact_name, artifact_type
            )

        # This ensures hasattr() works correctly and setup() can check them
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

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
        >>> encoder = MessagePassingEncoder.from_config(model_config, in_channels=in_channels)
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
                f"data must be either a NapistuData object or a dictionary "
                f"with keys {TRAINING.TRAIN}, {TRAINING.VALIDATION}, {TRAINING.TEST}, "
                f"but got {type(self.napistu_data)}"
            )

    def setup(self, stage: Optional[str] = None):
        """
        Set up NapistuData object(s) from the provided data.

        Uses the pre-processed NapistuData object(s) passed to the constructor.
        No data creation or processing happens here - just assignment.
        """
        if hasattr(self, "data") and self.data is not None:
            return  # Already set up
        if hasattr(self, "train_data") and self.train_data is not None:
            return  # Already set up

        if isinstance(self.napistu_data, dict):
            # Inductive split - use separate data for each split
            self.train_data = self.napistu_data[TRAINING.TRAIN]
            self.val_data = self.napistu_data.get(TRAINING.VALIDATION)
            self.test_data = self.napistu_data.get(TRAINING.TEST)
        elif isinstance(self.napistu_data, NapistuData):
            # Single graph with masks (transductive)
            self.data = self.napistu_data
        else:
            raise ValueError(
                f"napistu_data must be a dictionary or a NapistuData object, but got {type(self.napistu_data)}"
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

        return _get_dataloader(dataset)

    def val_dataloader(self):
        """Return a DataLoader for validation."""
        if self.val_data is not None:
            # Inductive split - use separate validation data
            dataset = SingleGraphDataset(self.val_data)
        else:
            # Transductive split - use single graph with masks
            dataset = SingleGraphDataset(self.data)

        return _get_dataloader(dataset)

    def test_dataloader(self):
        """Return a DataLoader for testing."""
        if self.test_data is not None:
            # Inductive split - use separate test data
            dataset = SingleGraphDataset(self.test_data)
        else:
            # Transductive split - use single graph with masks
            dataset = SingleGraphDataset(self.data)

        return _get_dataloader(dataset)

    def predict_dataloader(self):
        """Return a DataLoader for prediction."""
        if self.test_data is not None:
            # Inductive split - use separate test data for prediction
            dataset = SingleGraphDataset(self.test_data)
        else:
            # Transductive split - use single graph with masks
            dataset = SingleGraphDataset(self.data)

        return _get_dataloader(dataset)


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


def _get_dataloader(dataset: SingleGraphDataset) -> DataLoader:

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,  # No shuffling for single graph
        collate_fn=identity_collate,  # Don't use PyG's batching
        num_workers=0,  # Single graph, no benefit from workers
    )


def _task_config_to_artifact_names(task_config: TaskConfig) -> List[str]:
    """
    Convert a TaskConfig to a list of artifact names.
    """

    if task_config.task == TASKS.EDGE_PREDICTION:
        return _task_config_to_artifact_names_edge_prediction(task_config)
    else:
        return list()


def _task_config_to_artifact_names_edge_prediction(
    task_config: TaskConfig,
) -> List[str]:
    """
    Convert a TaskConfig to a list of artifact names for edge prediction.
    """

    ALL_VALID = {"none"} | STRATIFY_BY_ARTIFACT_NAMES
    if task_config.edge_prediction_neg_sampling_stratify_by not in ALL_VALID:
        raise ValueError(
            f"Invalid stratify_by value: {task_config.edge_prediction_neg_sampling_stratify_by}. Must be one of: {ALL_VALID}"
        )
    if task_config.edge_prediction_neg_sampling_stratify_by == "none":
        return list()
    else:
        return [task_config.edge_prediction_neg_sampling_stratify_by]
