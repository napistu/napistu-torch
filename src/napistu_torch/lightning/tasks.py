"""
Simplified Lightning task adapters that handle single-graph batches correctly.
"""

import gc
import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from napistu_torch.configs import TrainingConfig
from napistu_torch.constants import NAPISTU_DATA
from napistu_torch.ml.constants import TRAINING
from napistu_torch.models.constants import EDGE_WEIGHTING_TYPE, ENCODER_DEFS
from napistu_torch.napistu_data import NapistuData
from napistu_torch.tasks.edge_prediction import EdgePredictionTask
from napistu_torch.tasks.node_classification import NodeClassificationTask

logger = logging.getLogger(__name__)


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
                patience=self.config.early_stopping_patience,
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

    @torch.no_grad()
    def get_embeddings(self, data: NapistuData) -> torch.Tensor:
        """
        Extract node embeddings for the given data.

        Parameters
        ----------
        data : NapistuData
            Graph data object

        Returns
        -------
        torch.Tensor
            Node embeddings [num_nodes, hidden_channels] on CPU
        """
        self.eval()
        data = data.to(self.device)

        # Handle edge data based on encoder configuration
        edge_data = None
        if hasattr(self.task.encoder, ENCODER_DEFS.EDGE_WEIGHTING_TYPE):
            if (
                self.task.encoder.edge_weighting_type
                == EDGE_WEIGHTING_TYPE.LEARNED_ENCODER
            ):
                edge_data = getattr(data, NAPISTU_DATA.EDGE_ATTR, None)
            elif (
                self.task.encoder.edge_weighting_type
                == EDGE_WEIGHTING_TYPE.STATIC_WEIGHTS
            ):
                edge_data = getattr(
                    self.task.encoder, ENCODER_DEFS.EDGE_WEIGHTING_VALUE, None
                )

        embeddings = self.task.get_embeddings(data.x, data.edge_index, edge_data)
        return embeddings.cpu()

    @torch.no_grad()
    def get_learned_edge_weights(self, data: NapistuData) -> torch.Tensor:
        """
        Extract learned edge weights for the given data if a learned edge encoder is present.

        Parameters
        ----------
        data : NapistuData
            Graph data object containing edge attributes.

        Returns
        -------
        torch.Tensor
            Learned edge weights [num_edges] on CPU.

        Raises
        ------
        ValueError
            If the encoder is not configured with a learned edge encoder or if edge attributes are missing.
        """
        if (
            getattr(
                self.task.encoder,
                ENCODER_DEFS.EDGE_WEIGHTING_TYPE,
                EDGE_WEIGHTING_TYPE.NONE,
            )
            != EDGE_WEIGHTING_TYPE.LEARNED_ENCODER
        ):
            raise ValueError(
                "Encoder is not configured with a learned edge encoder; "
                "learned edge weights are unavailable."
            )

        self.eval()
        data = data.to(self.device)
        edge_attr = getattr(data, NAPISTU_DATA.EDGE_ATTR, None)
        if edge_attr is None:
            raise ValueError(
                "NapistuData.edge_attr is required to compute learned edge weights."
            )

        learned_weights = self.task.get_learned_edge_weights(edge_attr)
        return learned_weights.detach().cpu()

    def training_step(self, batch, batch_idx):
        """
        Training step - handles both full-batch and mini-batch modes.

        Parameters
        ----------
        batch : NapistuData or torch.Tensor
            Either:
            - NapistuData object (from FullGraphDataModule)
            - Edge indices tensor (from EdgeBatchDataModule)
        batch_idx : int
            Batch index within epoch

        Returns
        -------
        torch.Tensor
            Loss value
        """
        # Detect batch type and extract data
        data, edge_indices = self._unpack_batch(batch)

        # Prepare batch for the task
        prepared_batch = self.task.prepare_batch(
            data,
            split=TRAINING.TRAIN,
            edge_indices=edge_indices,
        )

        # Compute loss
        loss = self.task.compute_loss(prepared_batch)

        # Log
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - batch should be a NapistuData object."""
        _validate_is_napistu_data(batch, "validation_step")

        # Delegates to the core task
        metrics = self.task.validation_step(batch)

        # Log all metrics
        for metric_name, value in metrics.items():
            self.log(f"val_{metric_name}", value, prog_bar=True, on_epoch=True)

        return metrics

    def test_step(self, batch, batch_idx):
        """Test step - batch should be a NapistuData object."""
        _validate_is_napistu_data(batch, "test_step")

        # Delegates to the core task
        metrics = self.task.test_step(batch)

        # Log all metrics
        for metric_name, value in metrics.items():
            self.log(f"test_{metric_name}", value, on_epoch=True)

        return metrics

    def predict_step(self, batch, batch_idx):
        """
        Predict step - returns per-edge predictions for analysis.

        This method is called by trainer.predict() and returns predictions
        for each edge in the batch, which can be used for analysis.

        Returns
        -------
        torch.Tensor
            The actual model predictions (sigmoid probabilities).
            Higher scores = more likely to be a real edge.
        """
        _validate_is_napistu_data(batch, "predict_step")
        return self.task.predict(batch)

    def on_batch_end(self):
        _cleanup()

    def on_train_epoch_end(self):
        _cleanup()

    def on_validation_epoch_end(self):
        _cleanup()

    def _unpack_batch(self, batch):
        """
        Unpack training batch into (data, edge_indices).

        Only used in training_step. Val/test always receive NapistuData.

        Parameters
        ----------
        batch : NapistuData or torch.Tensor
            Batch from training dataloader

        Returns
        -------
        tuple[NapistuData, Optional[torch.Tensor]]
            (data, edge_indices) where:
            - data: NapistuData object
            - edge_indices: None (full-batch) or tensor (mini-batch)
        """
        if isinstance(batch, NapistuData):
            # Full-batch mode (FullGraphDataModule)
            logger.debug("Full-batch mode: batch is NapistuData")
            return batch, None

        elif isinstance(batch, torch.Tensor):
            # Mini-batch mode (EdgeBatchDataModule)
            logger.debug(f"Mini-batch mode: {len(batch)} edge indices")

            # Get full data from datamodule
            data = self.trainer.datamodule.data

            return data, batch

        else:
            raise ValueError(
                f"Unexpected batch type in training: {type(batch)}. "
                f"Expected NapistuData or torch.Tensor."
            )


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
        _validate_is_napistu_data(batch, "training_step")
        loss = self.task.training_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - batch should be a NapistuData object."""
        _validate_is_napistu_data(batch, "validation_step")
        metrics = self.task.validation_step(batch)
        for metric_name, value in metrics.items():
            self.log(f"val_{metric_name}", value, prog_bar=True, on_epoch=True)
        return metrics

    def test_step(self, batch, batch_idx):
        """Test step - batch should be a NapistuData object."""
        _validate_is_napistu_data(batch, "test_step")
        metrics = self.task.test_step(batch)
        for metric_name, value in metrics.items():
            self.log(f"test_{metric_name}", value, on_epoch=True)
        return metrics


# public functions


def get_edge_encoder(model: nn.Module) -> nn.Module:
    """
    Retrieve the learned edge encoder module from a trained model.

    Parameters
    ----------
    model : nn.Module
        Model whose task encoder defines ``edge_weighting_value``.

    Returns
    -------
    nn.Module
        The learned edge encoder set to evaluation mode.

    Raises
    ------
    ValueError
        If the model does not include a learned edge encoder.
    """
    edge_encoder = getattr(model.task.encoder, "edge_weighting_value", None)
    if edge_encoder is None or not isinstance(edge_encoder, nn.Module):
        raise ValueError("Experiment does not include a learned edge encoder.")
    edge_encoder.eval()
    return edge_encoder


# private functions


def _validate_is_napistu_data(batch, method_name: str):
    """
    Utility function to validate that batch is a NapistuData object.

    Parameters
    ----------
    batch : Any
        The batch to validate
    method_name : str
        Name of the method calling this validation (for error messages)

    Raises
    ------
    AssertionError
        If batch is not a NapistuData object
    """
    if not isinstance(batch, NapistuData):
        raise AssertionError(
            f"Expected NapistuData, got {type(batch)}. "
            f"Check your DataModule's collate_fn in {method_name}."
        )


def _cleanup():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
