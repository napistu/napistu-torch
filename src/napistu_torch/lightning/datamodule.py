"""
Base Lightning DataModule for Napistu networks.

This module provides the abstract base class and shared infrastructure.
Concrete implementations should subclass and implement the dataloader methods.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from napistu_torch.configs import (
    ExperimentConfig,
    config_to_data_trimming_spec,
    task_config_to_artifact_names,
)
from napistu_torch.constants import (
    ARTIFACT_TYPES,
    DATA_CONFIG,
    NAPISTU_DATA_TRIM_ARGS,
)
from napistu_torch.load.artifacts import DEFAULT_ARTIFACT_REGISTRY, ArtifactDefinition
from napistu_torch.ml.constants import TRAINING
from napistu_torch.napistu_data import NapistuData
from napistu_torch.napistu_data_store import NapistuDataStore

logger = logging.getLogger(__name__)


class NapistuDataModule(pl.LightningDataModule, ABC):
    """
    Abstract base class for Napistu Lightning DataModules.

    Provides shared infrastructure for all Napistu DataModules:
    - Artifact loading from configs and store
    - Setup logic for transductive/inductive splits
    - Property access (num_node_features, num_edge_features)

    Subclasses must implement:
    - train_dataloader()
    - val_dataloader()
    - test_dataloader()
    - predict_dataloader()

    Do not instantiate this class directly. Use concrete implementations:
    - FullGraphDataModule: Returns full NapistuData objects (full-batch training)
    - EdgeBatchDataModule: Returns edge indices (mini-batch training for edge prediction)

    Parameters
    ----------
    config : ExperimentConfig
        Pydantic experiment configuration containing data, model, task, and training configs
    napistu_data_name : Optional[str]
        Name of the NapistuData artifact to use for training
    other_artifacts : Optional[List[str]]
        List of other artifact names needed for the experiment
    napistu_data : Optional[NapistuData]
        Direct NapistuData object for testing/backward compatibility
    store : Optional[NapistuDataStore]
        Pre-initialized store
    artifact_registry : Optional[Dict[str, ArtifactDefinition]]
        Registry of artifact definitions
    overwrite_artifacts : bool, default=False
        If True, recreate artifact even if it exists
    """

    def __init__(
        self,
        config: ExperimentConfig,
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

        # Ensure config is an ExperimentConfig instance
        if not isinstance(config, ExperimentConfig):
            raise TypeError(
                f"config must be an ExperimentConfig instance, got {type(config)}"
            )
        self.config = config

        # Use DataConfig values if not provided explicitly
        if napistu_data_name is None:
            napistu_data_name = getattr(config.data, DATA_CONFIG.NAPISTU_DATA_NAME)
        if other_artifacts is None:
            other_artifacts = getattr(config.data, DATA_CONFIG.OTHER_ARTIFACTS)

        # Add additional artifacts required by the task
        task_artifacts = task_config_to_artifact_names(config.task)
        other_artifacts = list(set(other_artifacts) | set(task_artifacts))

        # Create or load the store from config
        if store is None:
            need_store = (napistu_data is None) or (len(other_artifacts) > 0)
            if need_store:
                logger.info("Creating/loading store from config")
                ensure_artifacts = (napistu_data is None) or (len(other_artifacts) > 0)
                napistu_data_store = NapistuDataStore.from_config(
                    config.data,
                    task_config=config.task,
                    ensure_artifacts=ensure_artifacts,
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

        # Handle direct napistu_data input
        if napistu_data is not None:
            logger.info("Using provided napistu_data directly")
            self.napistu_data = napistu_data
            required_artifacts = other_artifacts
        else:
            napistu_data_store.validate_artifact_name(
                napistu_data_name,
                artifact_registry=artifact_registry,
                required_type=ARTIFACT_TYPES.NAPISTU_DATA,
            )
            required_artifacts = [napistu_data_name] + other_artifacts

        # Ensure all required artifacts exist
        if need_store:
            napistu_data_store.ensure_artifacts(
                required_artifacts,
                artifact_registry=artifact_registry,
                overwrite=overwrite_artifacts,
            )

        # Load napistu_data from store if not provided directly
        if napistu_data is None:
            logger.info(f"Loading napistu_data '{napistu_data_name}' from store")
            napistu_data = napistu_data_store.load_napistu_data(napistu_data_name)

            # filter to just the required attributes (ignored if side-loading NapistuData)
            data_trimming_spec = config_to_data_trimming_spec(config)
            napistu_data.trim(
                keep_edge_attr=data_trimming_spec[
                    NAPISTU_DATA_TRIM_ARGS.KEEP_EDGE_ATTR
                ],
                keep_labels=data_trimming_spec[NAPISTU_DATA_TRIM_ARGS.KEEP_LABELS],
                keep_masks=data_trimming_spec[NAPISTU_DATA_TRIM_ARGS.KEEP_MASKS],
                inplace=True,
            )
            self.napistu_data = napistu_data

        # Load other artifacts
        self.other_artifacts = {}
        for artifact_name in other_artifacts:
            artifact_type = artifact_registry[artifact_name].artifact_type
            self.other_artifacts[artifact_name] = napistu_data_store.load_artifact(
                artifact_name, artifact_type
            )

        # Initialize data attributes for setup()
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    @property
    def num_edge_features(self) -> int:
        """Get the number of edge features from the data."""
        # ... KEEP YOUR EXISTING IMPLEMENTATION ...
        if isinstance(self.napistu_data, dict):
            return self.napistu_data[TRAINING.TRAIN].num_edge_features
        elif isinstance(self.napistu_data, NapistuData):
            return self.napistu_data.num_edge_features
        else:
            raise ValueError(
                f"data must be either a NapistuData object or a dictionary, "
                f"but got {type(self.napistu_data)}"
            )

    @property
    def num_node_features(self) -> int:
        """Get the number of node features from the data."""
        # ... KEEP YOUR EXISTING IMPLEMENTATION ...
        if isinstance(self.napistu_data, dict):
            return self.napistu_data[TRAINING.TRAIN].num_node_features
        elif isinstance(self.napistu_data, NapistuData):
            return self.napistu_data.num_node_features
        else:
            raise ValueError(
                f"data must be either a NapistuData object or a dictionary, "
                f"but got {type(self.napistu_data)}"
            )

    def setup(self, stage: Optional[str] = None):
        """
        Set up NapistuData object(s) from the provided data.

        Shared setup logic for all subclasses.
        """
        # ... KEEP YOUR EXISTING IMPLEMENTATION ...
        if hasattr(self, "data") and self.data is not None:
            return
        if hasattr(self, "train_data") and self.train_data is not None:
            return

        if isinstance(self.napistu_data, dict):
            # Inductive split
            self.train_data = self.napistu_data[TRAINING.TRAIN]
            self.val_data = self.napistu_data.get(TRAINING.VALIDATION)
            self.test_data = self.napistu_data.get(TRAINING.TEST)
        elif isinstance(self.napistu_data, NapistuData):
            # Transductive split
            self.data = self.napistu_data
        else:
            raise ValueError(
                f"napistu_data must be a dictionary or a NapistuData object, "
                f"but got {type(self.napistu_data)}"
            )

    # Abstract methods - subclasses must implement
    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader. Must be implemented by subclasses."""
        pass
