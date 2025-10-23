from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling

from napistu_torch.napistu_data import NapistuData
from napistu_torch.tasks.base import BaseTask


class NodeClassificationTask(BaseTask):
    """
    Node classification task.

    Predicts node labels using node features and graph structure.

    This class is Lightning-free - pure PyTorch logic.
    """

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        num_classes: int,
        metrics: List[str] = None,
    ):
        super().__init__(encoder, head)
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = metrics or ["accuracy", "f1_macro"]

    def prepare_batch(
        self,
        data: NapistuData,
        split: str = "train",
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for node classification.

        For transductive learning, returns full graph with mask.
        """
        mask = getattr(data, f"{split}_mask")

        return {
            "x": data.x,
            "edge_index": data.edge_index,
            "edge_weight": getattr(data, "edge_weight", None),
            "y": data.y,
            "mask": mask,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute cross-entropy loss for node classification.
        """
        # Encode all nodes
        z = self.encoder.encode(
            batch["x"],
            batch["edge_index"],
            batch.get("edge_weight"),
        )

        # Classify nodes
        logits = self.head(z)

        # Compute loss only on training nodes
        loss = self.loss_fn(logits[batch["mask"]], batch["y"][batch["mask"]])

        return loss

    def compute_metrics(
        self,
        data: NapistuData,
        split: str = "val",
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        """
        from sklearn.metrics import accuracy_score, f1_score

        self.eval()
        with torch.no_grad():
            batch = self.prepare_batch(data, split=split)

            # Encode and classify
            z = self.encoder.encode(
                batch["x"],
                batch["edge_index"],
                batch.get("edge_weight"),
            )
            logits = self.head(z)

            # Get predictions for this split
            preds = logits[batch["mask"]].argmax(dim=1).cpu().numpy()
            labels = batch["y"][batch["mask"]].cpu().numpy()

            # Compute metrics
            results = {}
            if "accuracy" in self.metrics:
                results["accuracy"] = accuracy_score(labels, preds)
            if "f1_macro" in self.metrics:
                results["f1_macro"] = f1_score(labels, preds, average="macro")

            return results

    def _predict_impl(self, data: NapistuData) -> torch.Tensor:
        """
        Predict class labels for all nodes.
        """
        z = self.encoder.encode(
            data.x,
            data.edge_index,
            getattr(data, "edge_weight", None),
        )
        logits = self.head(z)
        return logits.argmax(dim=1)
