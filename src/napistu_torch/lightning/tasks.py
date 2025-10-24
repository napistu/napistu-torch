"""
Simplified Lightning task adapters that handle single-graph batches correctly.
"""

import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from napistu_torch.configs import TrainingConfig
from napistu_torch.napistu_data import NapistuData
from napistu_torch.tasks.edge_prediction import EdgePredictionTask
from napistu_torch.tasks.node_classification import NodeClassificationTask


class BaseLightningTask(pl.LightningModule):
    """
    Base class for Lightning task adapters.

    This handles all the Lightning boilerplate (optimizer config, logging, etc.)
    so your task-specific classes can focus on task logic.
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
        """Shared optimizer configuration."""
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

    This wraps EdgePredictionTask and handles the DataLoader interface.
    """

    def __init__(
        self,
        task: EdgePredictionTask,
        config: TrainingConfig,
    ):
        super().__init__(task, config)

    def training_step(self, batch, batch_idx):
        """
        Training step - batch should be a NapistuData object.

        Our DataModule with identity_collate ensures batch is a NapistuData
        object directly, not wrapped in any container.
        """
        assert isinstance(batch, NapistuData), (
            f"Expected NapistuData, got {type(batch)}. "
            f"Check your DataModule's collate_fn."
        )

        # Delegate to your core task
        loss = self.task.training_step(batch)

        # Lightning handles logging
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - batch should be a NapistuData object."""
        assert isinstance(batch, NapistuData), (
            f"Expected NapistuData, got {type(batch)}. "
            f"Check your DataModule's collate_fn."
        )

        # Delegate to your core task
        metrics = self.task.validation_step(batch)

        # Log all metrics
        for metric_name, value in metrics.items():
            self.log(f"val_{metric_name}", value, prog_bar=True, on_epoch=True)

        return metrics

    def test_step(self, batch, batch_idx):
        """Test step - batch should be a NapistuData object."""
        assert isinstance(batch, NapistuData), (
            f"Expected NapistuData, got {type(batch)}. "
            f"Check your DataModule's collate_fn."
        )

        # Delegate to your core task
        metrics = self.task.test_step(batch)

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
        """Training step - batch should be a NapistuData object."""
        assert isinstance(batch, NapistuData), (
            f"Expected NapistuData, got {type(batch)}. "
            f"Check your DataModule's collate_fn."
        )
        loss = self.task.training_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - batch should be a NapistuData object."""
        assert isinstance(batch, NapistuData), (
            f"Expected NapistuData, got {type(batch)}. "
            f"Check your DataModule's collate_fn."
        )
        metrics = self.task.validation_step(batch)
        for metric_name, value in metrics.items():
            self.log(f"val_{metric_name}", value, prog_bar=True, on_epoch=True)
        return metrics

    def test_step(self, batch, batch_idx):
        """Test step - batch should be a NapistuData object."""
        assert isinstance(batch, NapistuData), (
            f"Expected NapistuData, got {type(batch)}. "
            f"Check your DataModule's collate_fn."
        )
        metrics = self.task.test_step(batch)
        for metric_name, value in metrics.items():
            self.log(f"test_{metric_name}", value, on_epoch=True)
        return metrics
