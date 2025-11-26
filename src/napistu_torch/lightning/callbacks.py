"""Custom Lightning callbacks for Napistu-Torch."""

import logging
import time
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from napistu_torch.load.checkpoints import CheckpointHyperparameters
from napistu_torch.ml.constants import TRAINING
from napistu_torch.utils.environment_info import EnvironmentInfo

logger = logging.getLogger(__name__)


class ExperimentTimingCallback(Callback):
    """Track detailed timing for architecture comparison."""

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.epoch_times = []
        self.epoch_start = None

    def on_train_start(self, trainer, pl_module):
        # Only initialize if not resuming (start_time will be restored from checkpoint)
        if self.start_time is None:
            self.start_time = time.time()
        if not hasattr(self, "epoch_times") or self.epoch_times is None:
            self.epoch_times = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        # epoch_start should be set by on_train_epoch_start, but handle edge cases
        if not hasattr(self, "epoch_start") or self.epoch_start is None:
            # This can happen if resuming and the epoch started before the callback was restored
            # Use a small default duration to avoid errors
            logger.debug("epoch_start not set, skipping epoch timing for this epoch")
            return

        epoch_duration = time.time() - self.epoch_start
        if not hasattr(self, "epoch_times") or self.epoch_times is None:
            self.epoch_times = []
        self.epoch_times.append(epoch_duration)

        # Log per-epoch timing (only if logger exists)
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log(
                {
                    "epoch_duration_seconds": epoch_duration,
                    "avg_epoch_duration": sum(self.epoch_times) / len(self.epoch_times),
                }
            )

    def on_train_end(self, trainer, pl_module):
        if self.start_time is None:
            logger.warning("start_time not set, cannot compute total training time")
            return

        total_time = time.time() - self.start_time

        # Log summary statistics (only if logger exists)
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            if self.epoch_times:
                trainer.logger.experiment.log(
                    {
                        "total_train_time_minutes": total_time / 60,
                        "total_epochs_completed": len(self.epoch_times),
                        "time_per_epoch_avg": sum(self.epoch_times)
                        / len(self.epoch_times),
                        "time_per_epoch_std": np.std(self.epoch_times),
                    }
                )

    def state_dict(self) -> Dict[str, Any]:
        """Save callback state for checkpointing."""
        return {
            "start_time": self.start_time,
            "epoch_times": self.epoch_times,
            "epoch_start": getattr(self, "epoch_start", None),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        self.start_time = state_dict.get("start_time")
        self.epoch_times = state_dict.get("epoch_times", [])
        self.epoch_start = state_dict.get("epoch_start")


class ModelMetadataCallback(Callback):
    """
    Save model metadata to checkpoint for easier loading.

    Extracts metadata from:
    - task.get_summary() → Model architecture (encoder, head, edge_encoder)
    - napistu_data.get_summary(simplify=True) → Data statistics

    The metadata is validated using Pydantic models before saving to ensure
    compatibility with the Checkpoint loading system.

    Raises
    ------
    AttributeError
        If pl_module doesn't have a task attribute
    ValueError
        If datamodule or NapistuData cannot be found
    """

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Extract and save metadata at the start of training."""

        # Validate we have what we need - fail fast if not
        if not hasattr(pl_module, "task"):
            raise AttributeError(
                "pl_module must have a 'task' attribute. "
                "Cannot save model metadata without task."
            )

        if trainer.datamodule is None:
            raise ValueError(
                "trainer.datamodule is None. "
                "Cannot save data metadata without datamodule."
            )

        # Get NapistuData
        napistu_data = self._get_training_data(trainer.datamodule)
        if napistu_data is None:
            raise ValueError(
                "Could not extract NapistuData from datamodule. "
                "Cannot save data metadata. "
                "Datamodule must have one of: 'napistu_data', 'train_data', or 'data' attributes."
            )

        # Build and validate hyperparameters dict
        # Note: 'config' will be added by Lightning's save_hyperparameters()
        hparams_dict = CheckpointHyperparameters.from_task_and_data(
            task=pl_module.task,
            napistu_data=napistu_data,
            training_config=pl_module.hparams.get("config"),
            environment=EnvironmentInfo.from_current_env(),
        )

        # Update pl_module.hparams
        # This merges with existing hparams (like 'config' added by Lightning)
        for key, value in hparams_dict.items():
            pl_module.hparams[key] = value

        logger.info(
            f"Saved metadata: "
            f"encoder_type={hparams_dict['model'].get('encoder', {}).get('encoder_type')}, "
            f"head_type={hparams_dict['model'].get('head', {}).get('head_type')}, "
            f"data_name={hparams_dict['data'].get('name')}"
        )

    def _get_training_data(self, datamodule):
        """
        Get training NapistuData (handles transductive/inductive).

        Parameters
        ----------
        datamodule : NapistuDataModule
            The datamodule to extract NapistuData from

        Returns
        -------
        NapistuData or None
            Training data if found, None otherwise
        """
        if hasattr(datamodule, "napistu_data"):
            napistu_data = datamodule.napistu_data
            # Inductive: dict with train/val/test
            if isinstance(napistu_data, dict):
                return napistu_data.get(TRAINING.TRAIN)
            # Transductive: single NapistuData
            return napistu_data

        # Fallback after setup()
        if hasattr(datamodule, "train_data"):
            return datamodule.train_data
        if hasattr(datamodule, "data"):
            return datamodule.data

        return None
