"""Custom Lightning callbacks for Napistu-Torch."""

import logging
import time
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from napistu_torch.constants import NAPISTU_DATA
from napistu_torch.lightning.constants import (
    EMBEDDING_NORM_STATS,
    EXPERIMENT_TIMING_STATS,
    NAPISTU_DATA_MODULE,
)
from napistu_torch.load.checkpoints import CheckpointHyperparameters
from napistu_torch.load.constants import CHECKPOINT_HYPERPARAMETERS
from napistu_torch.ml.constants import SCORE_DISTRIBUTION_STATS, TRAINING
from napistu_torch.models.constants import MODEL_DEFS

logger = logging.getLogger(__name__)


class EmbeddingNormCallback(Callback):
    """Monitor embedding norm statistics during training."""

    def __init__(self, log_every_n_epochs: int = 10):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log embedding statistics periodically."""
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Get validation data
        val_data = trainer.datamodule.data
        val_data = val_data.to(pl_module.device)

        # Get embeddings
        with torch.no_grad():
            embeddings = pl_module.get_embeddings(val_data)
            norms = embeddings.norm(dim=1)

            stats = {
                EMBEDDING_NORM_STATS.EMBEDDING_NORM_MEAN: norms.mean().item(),
                EMBEDDING_NORM_STATS.EMBEDDING_NORM_MEDIAN: norms.median().item(),
                EMBEDDING_NORM_STATS.EMBEDDING_NORM_STD: norms.std().item(),
                EMBEDDING_NORM_STATS.EMBEDDING_NORM_MAX: norms.max().item(),
            }

        # Log to wandb
        if trainer.logger is not None:
            trainer.logger.experiment.log(
                {f"embeddings/{k}": v for k, v in stats.items()},
                step=trainer.global_step,
            )

        # Print summary
        logger.info(
            f"\n=== Embedding Norm Statistics (epoch {trainer.current_epoch}) ===\n"
            f"Mean ± Std: {stats[EMBEDDING_NORM_STATS.EMBEDDING_NORM_MEAN]:.3f} ± {stats[EMBEDDING_NORM_STATS.EMBEDDING_NORM_STD]:.3f}\n"
            f"Median: {stats[EMBEDDING_NORM_STATS.EMBEDDING_NORM_MEDIAN]:.3f}\n"
            f"Max: {stats[EMBEDDING_NORM_STATS.EMBEDDING_NORM_MAX]:.3f}\n"
        )


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
                    EXPERIMENT_TIMING_STATS.EPOCH_DURATION_SECONDS: epoch_duration,
                    EXPERIMENT_TIMING_STATS.AVG_EPOCH_DURATION: sum(self.epoch_times)
                    / len(self.epoch_times),
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
                        EXPERIMENT_TIMING_STATS.TOTAL_TRAIN_TIME_MINUTES: total_time
                        / 60,
                        EXPERIMENT_TIMING_STATS.TOTAL_EPOCHS_COMPLETED: len(
                            self.epoch_times
                        ),
                        EXPERIMENT_TIMING_STATS.TIME_PER_EPOCH_AVG: sum(
                            self.epoch_times
                        )
                        / len(self.epoch_times),
                        EXPERIMENT_TIMING_STATS.TIME_PER_EPOCH_STD: np.std(
                            self.epoch_times
                        ),
                    }
                )

    def state_dict(self) -> Dict[str, Any]:
        """Save callback state for checkpointing."""
        return {
            EXPERIMENT_TIMING_STATS.START_TIME: self.start_time,
            EXPERIMENT_TIMING_STATS.EPOCH_TIMES: self.epoch_times,
            EXPERIMENT_TIMING_STATS.EPOCH_START: getattr(self, "epoch_start", None),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        self.start_time = state_dict.get(EXPERIMENT_TIMING_STATS.START_TIME)
        self.epoch_times = state_dict.get(EXPERIMENT_TIMING_STATS.EPOCH_TIMES, [])
        self.epoch_start = state_dict.get(EXPERIMENT_TIMING_STATS.EPOCH_START)


class ScoreDistributionMonitoringCallback(Callback):
    """Monitor score distribution statistics periodically during training."""

    def __init__(self, log_every_n_epochs: int = 5):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            # Get validation data
            val_data = trainer.datamodule.data  # transductive

            # Get score distribution statistics
            score_distributions = pl_module.get_score_distributions(
                val_data, split=TRAINING.VALIDATION
            )

            # Log to wandb
            if trainer.logger is not None:
                trainer.logger.experiment.log(
                    {
                        f"score_distributions/{k}": v
                        for k, v in score_distributions.items()
                    },
                    step=trainer.global_step,
                )

            # Print summary
            logger.info(
                f"\n=== Score Distribution Statistics (epoch {trainer.current_epoch}) ===\n"
                f"Head: {score_distributions[SCORE_DISTRIBUTION_STATS.HEAD_TYPE]}\n"
                f"Pos scores: {score_distributions[SCORE_DISTRIBUTION_STATS.POS_SCORE_MEAN]:.2f} ± {score_distributions[SCORE_DISTRIBUTION_STATS.POS_SCORE_STD]:.2f} "
                f"[{score_distributions[SCORE_DISTRIBUTION_STATS.POS_SCORE_MIN]:.2f}, {score_distributions[SCORE_DISTRIBUTION_STATS.POS_SCORE_MAX]:.2f}]\n"
                f"Neg scores: {score_distributions[SCORE_DISTRIBUTION_STATS.NEG_SCORE_MEAN]:.2f} ± {score_distributions[SCORE_DISTRIBUTION_STATS.NEG_SCORE_STD]:.2f} "
                f"[{score_distributions[SCORE_DISTRIBUTION_STATS.NEG_SCORE_MIN]:.2f}, {score_distributions[SCORE_DISTRIBUTION_STATS.NEG_SCORE_MAX]:.2f}]\n"
                f"Separation (Cohen's d): {score_distributions[SCORE_DISTRIBUTION_STATS.SEPARATION_COHENS_D]:.2f}\n"
                f"Saturated: pos={score_distributions[SCORE_DISTRIBUTION_STATS.POS_SATURATED_PCT]:.1%}, neg={score_distributions[SCORE_DISTRIBUTION_STATS.NEG_SATURATED_PCT]:.1%}\n"
                f"Rank corr with dot product: {score_distributions[SCORE_DISTRIBUTION_STATS.RANK_CORR_WITH_DOTPROD]:.3f}\n"
            )


class SetHyperparametersCallback(Callback):
    """
    Set hyperparameters in Lightning module for checkpointing and logging.

    Extracts metadata from:
    - task.get_summary() → Model architecture (encoder, head, edge_encoder)
    - napistu_data.get_summary(simplify=True) → Data statistics
    - pl_module.config → Training configuration

    The metadata is validated using Pydantic models before saving to ensure
    compatibility with the Checkpoint loading system.

    Raises
    ------
    AttributeError
        If pl_module doesn't have a task or config attribute
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

        # Get config from pl_module.config attribute (set in BaseLightningTask.__init__)
        if not hasattr(pl_module, "config"):
            raise AttributeError(
                "pl_module must have a 'config' attribute. "
                "This should be set in BaseLightningTask.__init__()."
            )
        training_config = pl_module.config

        hparams_dict = CheckpointHyperparameters.from_task_and_data(
            task=pl_module.task,
            napistu_data=napistu_data,
            training_config=training_config,
            capture_environment=True,
        )

        # Update pl_module.hparams
        # This merges with existing hparams (like 'config' added by Lightning)
        for key, value in hparams_dict.items():
            pl_module.hparams[key] = value

        logger.info(
            f"Saved metadata: "
            f"encoder_type={hparams_dict[CHECKPOINT_HYPERPARAMETERS.MODEL].get(MODEL_DEFS.ENCODER, {}).get(MODEL_DEFS.ENCODER)}, "
            f"head_type={hparams_dict[CHECKPOINT_HYPERPARAMETERS.MODEL].get(MODEL_DEFS.HEAD, {}).get(MODEL_DEFS.HEAD)}, "
            f"data_name={hparams_dict[CHECKPOINT_HYPERPARAMETERS.DATA].get(NAPISTU_DATA.NAME)}"
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
        if hasattr(datamodule, NAPISTU_DATA_MODULE.NAPISTU_DATA):
            napistu_data = datamodule.napistu_data
            # Inductive: dict with train/val/test
            if isinstance(napistu_data, dict):
                return napistu_data.get(TRAINING.TRAIN)
            # Transductive: single NapistuData
            return napistu_data

        # Fallback after setup()
        if hasattr(datamodule, NAPISTU_DATA_MODULE.TRAIN_DATA):
            return datamodule.train_data
        if hasattr(datamodule, NAPISTU_DATA_MODULE.DATA):
            return datamodule.data

        return None
