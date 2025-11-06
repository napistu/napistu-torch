"""
Config-aware Trainer for Napistu-Torch.

Provides a NapistuTrainer class that wraps PyTorch Lightning Trainer
with Napistu-specific configurations and conveniences.
"""

import time
from typing import List, Optional, Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from napistu_torch.configs import ExperimentConfig
from napistu_torch.lightning.constants import (
    TRAINER_MODES,
    VALID_TRAINER_MODES,
)


class ExperimentTimingCallback(Callback):
    """Track detailed timing for architecture comparison."""

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.epoch_times = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - self.epoch_start
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
        total_time = time.time() - self.start_time

        # Log summary statistics (only if logger exists)
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log(
                {
                    "total_train_time_minutes": total_time / 60,
                    "total_epochs_completed": len(self.epoch_times),
                    "time_per_epoch_avg": sum(self.epoch_times) / len(self.epoch_times),
                    "time_per_epoch_std": np.std(self.epoch_times),
                }
            )


class NapistuTrainer:
    """
    Napistu-specific PyTorch Lightning Trainer wrapper.

    This class provides a convenient interface for creating and using
    PyTorch Lightning Trainers with Napistu-specific configurations.

    Parameters
    ----------
    config : ExperimentConfig
        Your Pydantic experiment configuration
    callbacks : List[pl.Callback], optional
        Additional custom callbacks

    Examples
    --------
    >>> from napistu_torch.config import ExperimentConfig
    >>> from napistu_torch.lightning import NapistuTrainer
    >>>
    >>> config = ExperimentConfig.from_yaml('experiment.yaml')
    >>> trainer = NapistuTrainer(config)
    >>>
    >>> # Train
    >>> trainer.fit(lightning_task, datamodule)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        mode: str = TRAINER_MODES.TRAIN,
        wandb_logger: Optional[WandbLogger] = None,
        callbacks: Optional[List[pl.Callback]] = None,
    ):
        self.config = config
        self._user_callbacks = callbacks or []
        # Store wandb_logger to use existing one if provided
        self.wandb_logger = wandb_logger

        # Create the underlying Lightning trainer
        if mode == TRAINER_MODES.TRAIN:
            self._trainer = self._create_trainer()
        elif mode == TRAINER_MODES.EVAL:
            self._trainer = self._create_eval_trainer()
        else:
            raise ValueError(
                f"Invalid trainer mode: {mode}; valid modes are {VALID_TRAINER_MODES}"
            )

    def _create_eval_trainer(self) -> pl.Trainer:
        """Create a minimal trainer for testing and evaluation (no callbacks)."""
        # Setup logger - use existing one if provided, otherwise create new one
        wandb_logger = self.wandb_logger or self._create_wandb_logger()

        trainer = pl.Trainer(
            accelerator=self.config.training.accelerator,
            devices=self.config.training.devices,
            precision=self.config.training.precision,
            logger=wandb_logger,
            enable_progress_bar=True,
            enable_model_summary=False,  # Already trained
            deterministic=self.config.deterministic,
        )

        return trainer

    def _create_trainer(self) -> pl.Trainer:
        """Create the underlying PyTorch Lightning Trainer."""
        # Setup logger - use existing one if provided, otherwise create new one
        wandb_logger = self.wandb_logger or self._create_wandb_logger()

        # Setup callbacks
        all_callbacks = self._create_callbacks()
        all_callbacks.extend(self._user_callbacks)

        # Create Trainer
        trainer = pl.Trainer(
            # Training config
            max_epochs=self.config.training.epochs,
            accelerator=self.config.training.accelerator,
            devices=self.config.training.devices,
            precision=self.config.training.precision,
            # Logging and callbacks
            logger=wandb_logger,
            callbacks=all_callbacks,
            log_every_n_steps=10,
            # Reproducibility
            deterministic=self.config.deterministic,
            # Debug options
            fast_dev_run=self.config.fast_dev_run,
            limit_train_batches=self.config.limit_train_batches,
            limit_val_batches=self.config.limit_val_batches,
            # Other useful defaults
            enable_progress_bar=True,
            enable_model_summary=True,
        )

        return trainer

    def _create_wandb_logger(self) -> Optional[WandbLogger]:
        """Create W&B logger from config.

        Returns None if wandb mode is disabled to avoid initializing wandb
        and triggering sentry/analytics.
        """
        if self.config.wandb.mode == "disabled":
            return None

        save_dir = self.config.wandb.get_save_dir(self.config.output_dir)
        return WandbLogger(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=self.config.name,
            group=self.config.wandb.group,
            tags=self.config.wandb.tags,
            save_dir=str(save_dir),
            log_model=self.config.wandb.log_model,
            offline=(self.config.wandb.mode == "offline"),
        )

    def _create_callbacks(self) -> List[pl.Callback]:
        """Create callbacks from config."""
        callbacks = []

        # Early stopping
        if self.config.training.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=self.config.training.early_stopping_metric,
                    patience=self.config.training.early_stopping_patience,
                    mode="max",  # Assuming metric like AUC (higher is better)
                    verbose=True,
                )
            )

        # Model checkpointing
        if self.config.training.save_checkpoints:
            checkpoint_dir = self.config.training.get_checkpoint_dir(
                self.config.output_dir
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="best-{epoch}-{val_auc:.4f}",
                    monitor=self.config.training.checkpoint_metric,
                    mode="max",
                    save_top_k=1,
                    save_last=True,
                    verbose=True,
                )
            )

        # Learning rate monitoring (always useful)
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        # Timing callback (always useful)
        callbacks.append(ExperimentTimingCallback())

        return callbacks

    # ========================================================================
    # Delegate methods to underlying trainer
    # ========================================================================

    def fit(
        self,
        model: pl.LightningModule,
        datamodule: Optional[pl.LightningDataModule] = None,
        train_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        **kwargs,
    ):
        """Train the model."""
        return self._trainer.fit(
            model, datamodule, train_dataloaders, val_dataloaders, **kwargs
        )

    def test(
        self,
        model: pl.LightningModule,
        datamodule: Optional[pl.LightningDataModule] = None,
        dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        **kwargs,
    ):
        """Test the model."""
        return self._trainer.test(
            model, dataloaders=dataloaders, datamodule=datamodule, **kwargs
        )

    def validate(
        self,
        model: pl.LightningModule,
        datamodule: Optional[pl.LightningDataModule] = None,
        dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        **kwargs,
    ):
        """Validate the model."""
        return self._trainer.validate(model, datamodule, dataloaders, **kwargs)

    def predict(
        self,
        model: pl.LightningModule,
        datamodule: Optional[pl.LightningDataModule] = None,
        dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        **kwargs,
    ):
        """Make predictions."""
        return self._trainer.predict(model, datamodule, dataloaders, **kwargs)

    # ========================================================================
    # Convenience properties
    # ========================================================================

    @property
    def trainer(self) -> pl.Trainer:
        """Access the underlying PyTorch Lightning Trainer."""
        return self._trainer

    @property
    def logger(self) -> Optional[WandbLogger]:
        """Access the W&B logger (may be None if wandb is disabled)."""
        return self._trainer.logger

    @property
    def callbacks(self) -> List[pl.Callback]:
        """Access the callbacks."""
        return self._trainer.callbacks
