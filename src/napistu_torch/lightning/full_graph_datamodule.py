# In src/napistu_torch/lightning/full_graph_datamodule.py

"""
DataModule for full-batch training on a single graph.

Returns complete NapistuData objects - the entire graph is processed at once.
This is the traditional approach and matches the original NapistuDataModule behavior.
"""

import logging
from typing import Dict, List, Optional

from torch.utils.data import DataLoader

from napistu_torch.configs import ExperimentConfig
from napistu_torch.data.data_utils import create_single_graph_dataloader
from napistu_torch.data.dataset import SingleGraphDataset
from napistu_torch.lightning.datamodule import NapistuDataModule
from napistu_torch.load.artifacts import DEFAULT_ARTIFACT_REGISTRY, ArtifactDefinition
from napistu_torch.napistu_data import NapistuData
from napistu_torch.napistu_data_store import NapistuDataStore

logger = logging.getLogger(__name__)


class FullGraphDataModule(NapistuDataModule):
    """
    DataModule for full-batch training on a single graph.
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
        """
        Initialize FullGraphDataModule.

        Parameters
        ----------
        config : ExperimentConfig
            Pydantic experiment configuration
        napistu_data_name : Optional[str]
            Name of the NapistuData artifact to use
        other_artifacts : Optional[List[str]]
            List of other artifact names needed
        napistu_data : Optional[NapistuData]
            Direct NapistuData object for testing
        store : Optional[NapistuDataStore]
            Pre-initialized store
        artifact_registry : Optional[Dict[str, ArtifactDefinition]]
            Registry of artifact definitions
        overwrite_artifacts : bool
            If True, recreate artifacts even if they exist
        """
        super().__init__(
            config=config,
            napistu_data_name=napistu_data_name,
            other_artifacts=other_artifacts,
            napistu_data=napistu_data,
            store=store,
            artifact_registry=artifact_registry,
            overwrite_artifacts=overwrite_artifacts,
        )
        logger.info("FullGraphDataModule initialized (full-batch training)")

    def train_dataloader(self) -> DataLoader:
        """
        Return training dataloader.

        Returns
        -------
        DataLoader
            DataLoader that yields complete NapistuData object.
        """
        if self.train_data is not None:
            # Inductive split
            dataset = SingleGraphDataset(self.train_data)
        else:
            # Transductive split
            dataset = SingleGraphDataset(self.data)

        return create_single_graph_dataloader(dataset)

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.val_data is not None:
            dataset = SingleGraphDataset(self.val_data)
        else:
            dataset = SingleGraphDataset(self.data)

        return create_single_graph_dataloader(dataset)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        if self.test_data is not None:
            dataset = SingleGraphDataset(self.test_data)
        else:
            dataset = SingleGraphDataset(self.data)

        return create_single_graph_dataloader(dataset)

    def predict_dataloader(self) -> DataLoader:
        """Return prediction dataloader."""
        if self.test_data is not None:
            dataset = SingleGraphDataset(self.test_data)
        else:
            dataset = SingleGraphDataset(self.data)

        return create_single_graph_dataloader(dataset)
