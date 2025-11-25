"""Custom Lightning callbacks for Napistu-Torch."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from napistu_torch.ml.constants import TRAINING

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
    - Encoder.get_summary()
    - Head.get_summary()
    - NapistuData.get_summary()
    """

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Extract and save metadata at the start of training."""

        # Organize metadata into separate groups
        model_metadata = {}
        data_metadata = {}

        # Get task-level configuration (includes encoder, edge encoder, and head)
        if hasattr(pl_module, "task") and hasattr(
            pl_module.task, "to_model_config_dict"
        ):
            task_config = pl_module.task.to_model_config_dict()
            model_metadata.update(task_config)

        # Get data metadata
        if trainer.datamodule is not None:
            napistu_data = self._get_training_data(trainer.datamodule)
            if napistu_data is not None:
                data_summary = napistu_data.get_summary(simplify=True)
                data_metadata.update(data_summary)

        # Save to hyperparameters with organized structure
        if model_metadata or data_metadata:
            if model_metadata:
                pl_module.hparams["model"] = model_metadata
            if data_metadata:
                pl_module.hparams["data"] = data_metadata

            logger.info(
                f"Saved metadata: model={list(model_metadata.keys())}, "
                f"data={list(data_metadata.keys())}"
            )
        else:
            logger.warning("No metadata extracted")

    def _get_training_data(self, datamodule):
        """Get training NapistuData (handles transductive/inductive)."""
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


def validate_callback_state(checkpoint_path: Optional[Path]) -> None:
    """
    Verify that checkpoint contains callback state for EarlyStopping.

    Lightning automatically restores callback state when ckpt_path is passed to fit().
    This function verifies the checkpoint has the expected state and logs warnings if not.

    Parameters
    ----------
    checkpoint_path : Optional[Path]
        Path to the checkpoint file to verify
    """
    if checkpoint_path is None or not checkpoint_path.exists():
        return

    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

        # Check if checkpoint has callback states
        if "callbacks" not in checkpoint:
            logger.warning(
                f"Checkpoint {checkpoint_path.name} does not contain callback states. "
                f"EarlyStopping state will not be restored automatically."
            )
            return

        # Check for EarlyStopping callback state
        early_stopping_found = False
        for callback_key in checkpoint["callbacks"].keys():
            if "EarlyStopping" in callback_key:
                early_stopping_found = True
                callback_state = checkpoint["callbacks"][callback_key]
                if "best_score" in callback_state:
                    best_score = callback_state["best_score"]
                    # best_score might be a tensor
                    if hasattr(best_score, "item"):
                        best_score = best_score.item()
                    logger.info(
                        f"Checkpoint contains EarlyStopping state: best_score={best_score:.4f}, "
                        f"wait_count={callback_state.get('wait_count', 0)}"
                    )
                break

        if not early_stopping_found:
            logger.warning(
                f"Checkpoint {checkpoint_path.name} does not contain EarlyStopping callback state. "
                f"This may happen if the checkpoint was saved before EarlyStopping was added."
            )
    except Exception as e:
        logger.debug(f"Could not verify checkpoint callback state: {e}")
