from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn

from napistu_torch.ml.constants import TRAINING
from napistu_torch.models.constants import EDGE_WEIGHTING_TYPE, ENCODER_DEFS
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

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):
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

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get node embeddings from the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node features [num_nodes, num_features]
        edge_index : torch.Tensor
            Edge connectivity [2, num_edges]
        edge_data : torch.Tensor, optional
            Edge data for weighting (attributes or weights). When using a learned
            edge encoder, pass edge attributes; when using static weights, pass the
            weight tensor.

        Returns
        -------
        torch.Tensor
            Node embeddings [num_nodes, hidden_channels]
        """
        return self.encoder.encode(x, edge_index, edge_data)

    def get_learned_edge_weights(
        self,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute learned edge weights using the encoder's edge encoder.

        Parameters
        ----------
        edge_attr : torch.Tensor
            Edge attributes used by the learned edge encoder. Shape [num_edges, edge_dim].

        Returns
        -------
        torch.Tensor
            Learned edge weights in the range [0, 1] with shape [num_edges].

        Raises
        ------
        ValueError
            If the encoder does not provide a learned edge encoder, if it is missing,
            or if edge_attr is not provided.
        """
        if edge_attr is None:
            raise ValueError("edge_attr is required to compute learned edge weights.")

        edge_weighting_type = getattr(
            self.encoder, ENCODER_DEFS.EDGE_WEIGHTING_TYPE, EDGE_WEIGHTING_TYPE.NONE
        )
        if edge_weighting_type != EDGE_WEIGHTING_TYPE.LEARNED_ENCODER:
            raise ValueError(
                "Encoder is not configured with a learned edge encoder; "
                "learned edge weights are unavailable."
            )

        edge_encoder = getattr(self.encoder, ENCODER_DEFS.EDGE_WEIGHTING_VALUE, None)
        if edge_encoder is None or not isinstance(edge_encoder, nn.Module):
            raise ValueError(
                "Encoder edge encoder module is missing; cannot compute learned edge weights."
            )

        encoder_param = next(edge_encoder.parameters(), None)
        if encoder_param is not None and edge_attr.device != encoder_param.device:
            edge_attr = edge_attr.to(encoder_param.device)

        return edge_encoder(edge_attr)

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
