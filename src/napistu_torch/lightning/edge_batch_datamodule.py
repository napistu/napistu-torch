# In src/napistu_torch/lightning/edge_batch_datamodule.py

"""
DataModule for mini-batch training on edge prediction tasks.

Returns edge indices instead of full graphs, enabling multiple gradient
updates per epoch for better optimization on large graphs.
"""

import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from napistu_torch.configs import ExperimentConfig
from napistu_torch.data.data_utils import create_single_graph_dataloader
from napistu_torch.data.dataset import EdgeBatchDataset, SingleGraphDataset
from napistu_torch.lightning.datamodule import NapistuDataModule
from napistu_torch.load.artifacts import DEFAULT_ARTIFACT_REGISTRY, ArtifactDefinition
from napistu_torch.napistu_data import NapistuData
from napistu_torch.napistu_data_store import NapistuDataStore

logger = logging.getLogger(__name__)


class EdgeBatchDataModule(NapistuDataModule):
    """
    DataModule for edge prediction with mini-batch training.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        batches_per_epoch: Optional[int] = None,
        shuffle: bool = True,
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
        Initialize EdgeBatchDataModule.

        Parameters
        ----------
        config : ExperimentConfig
            Pydantic experiment configuration
        batches_per_epoch : Optional[int]
            Number of mini-batches per epoch. If None, uses config.training.batches_per_epoch
        shuffle : bool
            Whether to shuffle mini-batch order. Default True.
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

        self.batches_per_epoch = (
            batches_per_epoch
            if batches_per_epoch is not None
            else self.config.training.batches_per_epoch
        )
        self.shuffle = shuffle

        # Validate transductive split
        if isinstance(self.napistu_data, dict):
            raise ValueError(
                "EdgeBatchDataModule only supports transductive splits "
                "(single graph with train/val/test masks). "
                "For inductive splits, use FullGraphDataModule."
            )

        # Log configuration
        num_train_edges = self.napistu_data.train_mask.sum().item()
        edges_per_batch = num_train_edges // batches_per_epoch

        logger.info(
            f"EdgeBatchDataModule initialized:\n"
            f"  Total training edges: {num_train_edges:,}\n"
            f"  Mini-batches per epoch: {batches_per_epoch}\n"
            f"  Edges per mini-batch: {edges_per_batch:,}\n"
            f"  Shuffle mini-batches: {shuffle}"
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create training dataloader with edge mini-batching.

        Returns
        -------
        DataLoader
            DataLoader that yields edge indices tensors for each mini-batch.
        """

        train_edge_indices = torch.where(self.data.train_mask)[0]
        dataset = EdgeBatchDataset(train_edge_indices, self.batches_per_epoch)

        logger.info(f"Creating train dataloader with {self.batches_per_epoch} batches")

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=self.shuffle,
            num_workers=0,
            collate_fn=lambda x: x[0],
        )

    def val_dataloader(self) -> DataLoader:
        """
        Validation uses full graph (same as FullGraphDataModule).

        Returns
        -------
        DataLoader
            DataLoader that yields complete NapistuData object.
        """
        dataset = SingleGraphDataset(self.data)
        return create_single_graph_dataloader(dataset)

    def test_dataloader(self) -> DataLoader:
        """
        Test uses full graph (same as FullGraphDataModule).

        Returns
        -------
        DataLoader
            DataLoader that yields complete NapistuData object.
        """
        dataset = SingleGraphDataset(self.data)
        return create_single_graph_dataloader(dataset)

    def predict_dataloader(self) -> DataLoader:
        """
        Prediction uses full graph (same as FullGraphDataModule).

        Returns
        -------
        DataLoader
            DataLoader that yields complete NapistuData object.
        """
        dataset = SingleGraphDataset(self.data)
        return create_single_graph_dataloader(dataset)
