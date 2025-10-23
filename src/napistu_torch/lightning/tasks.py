"""
Lightning task adapters.

These are thin wrappers around your core task logic.
They adapt your domain logic to Lightning's interface.
"""

from typing import List, Optional, Dict, Any
import torch
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from napistu_torch.tasks.edge_prediction import EdgePredictionTask
from napistu_torch.tasks.node_classification import NodeClassificationTask
from napistu_torch.configs import TrainingConfig


class BaseLightningTask(pl.LightningModule):
    """
    Base class for Lightning task adapters.

    This handles all the Lightning boilerplate (optimizer config, logging, etc.)
    so your task-specific classes can focus on task logic.

    Subclasses just need to implement:
    - training_step()
    - validation_step()
    """

    def __init__(
        self,
        task,  # Your core task (no Lightning dependency)
        config: TrainingConfig,
    ):
        super().__init__()
        self.task = task
        self.config = config

        # Save hyperparameters (logged to W&B automatically)
        self.save_hyperparameters(ignore=["task"])

    def configure_optimizers(self):
        """
        Shared optimizer configuration.

        This is the same across all tasks, so it lives in the base class.
        """
        # Get all parameters
        params = self.task.parameters()

        # Create optimizer
        if self.config.optimizer == "adam":
            optimizer = Adam(
                params,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamw":
            optimizer = AdamW(
                params,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        # Optional scheduler
        if self.config.scheduler is None:
            return optimizer

        elif self.config.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=10,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.config.early_stopping_metric,
                    "interval": "epoch",
                },
            }

        elif self.config.scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.config.lr * 0.01,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")


class EdgePredictionLightning(BaseLightningTask):
    """
    Lightning adapter for edge prediction.

    This is a thin wrapper around EdgePredictionTask (your core logic).
    It just adapts your task to Lightning's training/validation interface.

    Parameters
    ----------
    task : EdgePredictionTask
        Your core edge prediction task (no Lightning dependency)
    config : TrainingConfig
        Training configuration

    Examples
    --------
    >>> # Your core task (no Lightning)
    >>> from napistu_torch.tasks import EdgePredictionTask
    >>> task = EdgePredictionTask(encoder, head)
    >>>
    >>> # Wrap for Lightning
    >>> from napistu_torch.lightning import EdgePredictionLightning
    >>> lightning_task = EdgePredictionLightning(task, training_config)
    >>>
    >>> # Train with Lightning
    >>> trainer = pl.Trainer(...)
    >>> trainer.fit(lightning_task, datamodule)
    """

    def __init__(
        self,
        task: EdgePredictionTask,
        config: TrainingConfig,
    ):
        super().__init__(task, config)

    def training_step(self, batch, batch_idx):
        """
        Delegate to your core task logic.

        This just adapts Lightning's interface to your task's interface.
        """
        # batch is a list with single NapistuData object
        data = batch[0] if isinstance(batch, list) else batch

        # Delegate to your core task
        loss = self.task.training_step(data)

        # Lightning handles logging
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Delegate to your core task logic.
        """
        data = batch[0] if isinstance(batch, list) else batch

        # Delegate to your core task
        metrics = self.task.validation_step(data)

        # Log all metrics
        for metric_name, value in metrics.items():
            self.log(f"val_{metric_name}", value, prog_bar=True, on_epoch=True)

        return metrics

    def test_step(self, batch, batch_idx):
        """
        Delegate to your core task logic.
        """
        data = batch[0] if isinstance(batch, list) else batch

        # Delegate to your core task
        metrics = self.task.test_step(data)

        # Log all metrics
        for metric_name, value in metrics.items():
            self.log(f"test_{metric_name}", value, on_epoch=True)

        return metrics


class NodeClassificationLightning(BaseLightningTask):
    """
    Lightning adapter for node classification.

    Same pattern as EdgePredictionLightning but for node classification.
    """

    def __init__(
        self,
        task: NodeClassificationTask,
        config: TrainingConfig,
    ):
        super().__init__(task, config)

    def training_step(self, batch, batch_idx):
        data = batch[0] if isinstance(batch, list) else batch
        loss = self.task.training_step(data)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[0] if isinstance(batch, list) else batch
        metrics = self.task.validation_step(data)
        for metric_name, value in metrics.items():
            self.log(f"val_{metric_name}", value, prog_bar=True, on_epoch=True)
        return metrics

    def test_step(self, batch, batch_idx):
        data = batch[0] if isinstance(batch, list) else batch
        metrics = self.task.test_step(data)
        for metric_name, value in metrics.items():
            self.log(f"test_{metric_name}", value, on_epoch=True)
        return metrics
