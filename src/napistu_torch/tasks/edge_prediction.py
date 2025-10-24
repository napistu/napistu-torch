from typing import Dict, List

import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling

from napistu_torch.ml.constants import SPLIT_TO_MASK, TRAINING
from napistu_torch.napistu_data import NapistuData
from napistu_torch.tasks.base import BaseTask


class EdgePredictionTask(BaseTask):
    """
    Edge prediction (link prediction) task.

    Predicts whether edges exist between node pairs using:
    1. Node embeddings from encoder
    2. Edge scores from head (dot product, MLP, etc.)
    3. Negative sampling for training

    This class is Lightning-free - pure PyTorch logic.
    Use EdgePredictionLightning (in napistu_torch.lightning) for training.

    Parameters
    ----------
    encoder : nn.Module
        Graph encoder (SAGE, GCN, GAT, etc.)
    head : nn.Module
        Edge decoder (DotProduct, MLP, Bilinear, etc.)
    neg_sampling_ratio : float
        Ratio of negative to positive samples
    metrics : List[str]
        Metrics to compute ('auc', 'ap', etc.)

    Examples
    --------
    >>> # Create task (no Lightning!)
    >>> from napistu_torch.models.gnns import GNNEncoder
    >>> from napistu_torch.models.heads import DotProductHead
    >>>
    >>> encoder = GNNEncoder(in_channels=128, hidden_channels=256, num_layers=3, encoder='sage')
    >>> head = DotProductHead()
    >>> task = EdgePredictionTask(encoder, head)
    >>>
    >>> # Use for inference (no Lightning!)
    >>> predictions = task.predict(data)
    >>>
    >>> # Or wrap for training with Lightning
    >>> from napistu_torch.lightning import EdgePredictionLightning
    >>> lightning_task = EdgePredictionLightning(task, training_config)
    >>> trainer.fit(lightning_task, datamodule)
    """

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        neg_sampling_ratio: float = 1.0,
        metrics: List[str] = None,
    ):
        super().__init__(encoder, head)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.neg_sampling_ratio = neg_sampling_ratio
        self.metrics = metrics or ["auc", "ap"]

    def prepare_batch(
        self, data: NapistuData, split: str = "train"
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for edge prediction.

        For transductive (mask-based) splits, this:
        1. Gets positive edges from the split mask
        2. Gets supervision edges (for message passing)
        3. Samples negative edges

        Returns
        -------
        Dict with keys:
            - x: Node features
            - supervision_edges: Edges for message passing
            - pos_edges: Positive edges to predict
            - neg_edges: Negative edges to predict
            - edge_weight: Edge weights (optional)
        """
        # Get the right mask for this split
        mask_attr = SPLIT_TO_MASK[split]
        mask = getattr(data, mask_attr)
        pos_edge_index = data.edge_index[:, mask]

        # Edges for message passing (exclude test/val during training)
        if split == TRAINING.TRAIN:
            supervision_edges = data.edge_index[:, data.train_mask]
        else:
            # During validation/test, use training edges only
            supervision_edges = data.edge_index[:, data.train_mask]

        # Sample negative edges
        num_neg = int(pos_edge_index.size(1) * self.neg_sampling_ratio)
        neg_edge_index = negative_sampling(
            edge_index=supervision_edges,
            num_nodes=data.num_nodes,
            num_neg_samples=num_neg,
        )

        return {
            "x": data.x,
            "supervision_edges": supervision_edges,
            "pos_edges": pos_edge_index,
            "neg_edges": neg_edge_index,
            "edge_weight": getattr(data, "edge_weight", None),
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute BCE loss for edge prediction.

        Steps:
        1. Encode nodes using supervision edges
        2. Score positive and negative edges
        3. Compute binary cross-entropy loss
        """
        # Encode nodes
        z = self.encoder.encode(
            batch["x"],
            batch["supervision_edges"],
        )

        # Score positive and negative edges
        pos_scores = self.head(z, batch["pos_edges"])
        neg_scores = self.head(z, batch["neg_edges"])

        # Binary classification loss
        pos_loss = self.loss_fn(pos_scores, torch.ones_like(pos_scores))
        neg_loss = self.loss_fn(neg_scores, torch.zeros_like(neg_scores))

        return pos_loss + neg_loss

    def compute_metrics(
        self,
        data: NapistuData,
        split: str = TRAINING.VALIDATION,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics (AUC, AP, etc.).

        This runs in eval mode (no gradients).
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        self.eval()
        with torch.no_grad():
            # Prepare batch
            batch = self.prepare_batch(data, split=split)

            # Encode nodes
            z = self.encoder.encode(
                batch["x"],
                batch["supervision_edges"],
            )

            # Score positive and negative edges
            pos_scores = torch.sigmoid(self.head(z, batch["pos_edges"]))
            neg_scores = torch.sigmoid(self.head(z, batch["neg_edges"]))

            # Combine predictions and labels
            y_pred = torch.cat([pos_scores, neg_scores]).cpu().numpy()
            y_true = torch.cat(
                [
                    torch.ones(pos_scores.size(0)),
                    torch.zeros(neg_scores.size(0)),
                ]
            ).numpy()

            # Compute metrics
            results = {}
            if "auc" in self.metrics:
                results["auc"] = roc_auc_score(y_true, y_pred)
            if "ap" in self.metrics:
                results["ap"] = average_precision_score(y_true, y_pred)

            return results

    def _predict_impl(self, data: NapistuData) -> torch.Tensor:
        """
        Predict scores for all edges in data.

        This is for inference - no training, no negative sampling.
        """
        # Encode nodes using all edges
        z = self.encoder.encode(
            data.x,
            data.edge_index,
            getattr(data, "edge_weight", None),
        )

        # Score all edges
        scores = torch.sigmoid(self.head(z, data.edge_index))

        return scores

    def predict_edge_scores(
        self,
        data: NapistuData,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict scores for specific edge pairs.

        Useful for predicting on new/unseen edges.

        Parameters
        ----------
        data : NapistuData
            Graph data (for node features and structure)
        edge_index : torch.Tensor
            Edge pairs to score [2, num_edges]

        Returns
        -------
        torch.Tensor
            Edge scores [num_edges]

        Examples
        --------
        >>> # Predict on new edge candidates
        >>> task = EdgePredictionTask(encoder, head)
        >>> new_edges = torch.tensor([[0, 1], [2, 3], [4, 5]]).T
        >>> scores = task.predict_edge_scores(data, new_edges)
        """
        self.eval()
        with torch.no_grad():
            # Encode nodes
            z = self.encoder.encode(
                data.x,
                data.edge_index,
                getattr(data, "edge_weight", None),
            )

            # Score the specified edges
            scores = torch.sigmoid(self.head(z, edge_index))

            return scores
