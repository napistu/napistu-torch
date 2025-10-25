from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn

from napistu_torch.ml.constants import TRAINING
from napistu_torch.napistu_data import NapistuData


class BaseTask(ABC, nn.Module):
    """
    Base class for all Napistu learning tasks.

    This defines the interface that all tasks must implement.
    No Lightning dependency - pure PyTorch.

    Tasks handle:
    - Data preparation (e.g., negative sampling)
    - Loss computation
    - Evaluation metrics

    Training infrastructure (optimizers, schedulers, logging) is handled
    by the Lightning adapter in napistu_torch.lightning
    """

    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x, edge_index, edge_weight=None):
        """Standard forward pass - encode nodes."""
        return self.encoder.encode(x, edge_index, edge_weight)

    @abstractmethod
    def prepare_batch(
        self, data: NapistuData, split: str = TRAINING.TRAIN
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare data batch for this task.

        Task-specific data transformations (e.g., negative sampling for
        edge prediction, masking for node classification).
        """
        pass

    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute task-specific loss."""
        pass

    @abstractmethod
    def compute_metrics(
        self, data: NapistuData, split: str = TRAINING.VALIDATION
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Returns dictionary of metric_name -> value.
        """
        pass

    # ========================================================================
    # Interface for Lightning adapters
    # ========================================================================

    def training_step(self, data: NapistuData) -> torch.Tensor:
        """
        Training step - called by Lightning adapter.

        This is the interface Lightning expects.
        """
        batch = self.prepare_batch(data, split=TRAINING.TRAIN)
        loss = self.compute_loss(batch)
        return loss

    def validation_step(self, data: NapistuData) -> Dict[str, float]:
        """
        Validation step - called by Lightning adapter.
        """
        return self.compute_metrics(data, split=TRAINING.VALIDATION)

    def test_step(self, data: NapistuData) -> Dict[str, float]:
        """
        Test step - called by Lightning adapter.
        """
        return self.compute_metrics(data, split=TRAINING.TEST)

    # ========================================================================
    # Convenience methods for inference (no Lightning needed!)
    # ========================================================================

    def predict(self, data: NapistuData) -> torch.Tensor:
        """
        Make predictions (inference mode).

        This can be used WITHOUT Lightning for production/inference.
        """
        self.eval()
        with torch.no_grad():
            return self._predict_impl(data)

    @abstractmethod
    def _predict_impl(self, data: NapistuData) -> torch.Tensor:
        """Implementation of prediction logic."""
        pass
