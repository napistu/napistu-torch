"""Workflows for configuring, training and evaluating models"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pydantic import BaseModel, ConfigDict, field_validator

from napistu_torch.configs import ExperimentConfig
from napistu_torch.lightning.constants import EXPERIMENT_DICT
from napistu_torch.lightning.edge_batch_datamodule import EdgeBatchDataModule
from napistu_torch.lightning.full_graph_datamodule import FullGraphDataModule
from napistu_torch.lightning.tasks import EdgePredictionLightning
from napistu_torch.lightning.trainer import NapistuTrainer
from napistu_torch.ml.wandb import setup_wandb_logger
from napistu_torch.models.heads import Decoder
from napistu_torch.models.message_passing_encoder import MessagePassingEncoder
from napistu_torch.tasks.edge_prediction import (
    EdgePredictionTask,
    get_edge_strata_from_artifacts,
)


class ExperimentDict(BaseModel):
    """
    Pydantic model for validating experiment_dict structure.

    Ensures all required components are present and of correct types.
    """

    data_module: Any
    model: Any
    trainer: Any
    wandb_logger: Any

    @field_validator(EXPERIMENT_DICT.DATA_MODULE)
    @classmethod
    def validate_data_module(cls, v):
        """Validate that data_module is a LightningDataModule."""
        if not isinstance(v, pl.LightningDataModule):
            raise TypeError(
                f"data_module must be a LightningDataModule, got {type(v).__name__}"
            )
        if not isinstance(v, (FullGraphDataModule, EdgeBatchDataModule)):
            raise TypeError(
                f"data_module must be FullGraphDataModule or EdgeBatchDataModule, "
                f"got {type(v).__name__}"
            )
        return v

    @field_validator(EXPERIMENT_DICT.MODEL)
    @classmethod
    def validate_model(cls, v):
        """Validate that model is a LightningModule."""
        if not isinstance(v, pl.LightningModule):
            raise TypeError(f"model must be a LightningModule, got {type(v).__name__}")
        return v

    @field_validator(EXPERIMENT_DICT.TRAINER)
    @classmethod
    def validate_trainer(cls, v):
        """Validate that trainer is a NapistuTrainer."""
        if not isinstance(v, NapistuTrainer):
            raise TypeError(f"trainer must be a NapistuTrainer, got {type(v).__name__}")
        return v

    @field_validator(EXPERIMENT_DICT.WANDB_LOGGER)
    @classmethod
    def validate_wandb_logger(cls, v):
        """Validate that wandb_logger is a WandbLogger (check by class name)."""
        # Just check the class name to avoid import path issues
        if v is None or "WandbLogger" not in type(v).__name__:
            raise TypeError(
                f"wandb_logger must be a WandbLogger, got {type(v).__name__}"
            )
        return v

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )


def fit_model(
    experiment_dict: Dict[str, Any],
    resume_from: Optional[Path] = None,
    logger: Optional = None,
) -> NapistuTrainer:
    """
    Train a model using the provided experiment dictionary.

    Parameters
    ----------
    experiment_dict : Dict[str, Any]
        Dictionary containing the experiment components:
        - data_module : Union[FullGraphDataModule, EdgeBatchDataModule]
        - model : pl.LightningModule (e.g., EdgePredictionLightning)
        - trainer : NapistuTrainer
        - wandb_logger : WandbLogger
    resume_from : Path, optional
        Path to a checkpoint to resume from
    logger : logging.Logger, optional
        Logger instance to use

    Returns
    -------
    NapistuTrainer
        The trainer instance
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    # Validate experiment_dict structure - Pydantic will raise ValidationError with detailed info
    ExperimentDict(
        data_module=experiment_dict[EXPERIMENT_DICT.DATA_MODULE],
        model=experiment_dict[EXPERIMENT_DICT.MODEL],
        trainer=experiment_dict[EXPERIMENT_DICT.TRAINER],
        wandb_logger=experiment_dict[EXPERIMENT_DICT.WANDB_LOGGER],
    )

    logger.info("Starting training...")
    experiment_dict[EXPERIMENT_DICT.TRAINER].fit(
        experiment_dict[EXPERIMENT_DICT.MODEL],
        datamodule=experiment_dict[EXPERIMENT_DICT.DATA_MODULE],
        ckpt_path=resume_from,
    )

    logger.info("Training workflow completed")
    return experiment_dict[EXPERIMENT_DICT.TRAINER]


def prepare_experiment(
    config: ExperimentConfig, logger: Optional = None
) -> Dict[str, Any]:
    """
    Prepare the experiment for training.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration for the experiment
    logger : logging.Logger, optional
        Logger instance to use

    Returns
    -------
    experiment_dict : Dict[str, Any]
        Dictionary containing the experiment components:
        - data_module : Union[FullGraphDataModule, EdgeBatchDataModule]
        - model : pl.LightningModule (e.g., EdgePredictionLightning)
        - trainer : NapistuTrainer
        - wandb_logger : WandbLogger
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    # Set seed
    logger.info(f"Setting random seed: {config.seed}")
    pl.seed_everything(config.seed, workers=True)

    # 1. Setup W&B Logger
    logger.info("Setting up W&B logger")
    wandb_logger = setup_wandb_logger(config)

    # 2. Create Data Modul
    batches_per_epoch = config.training.batches_per_epoch
    logger.info(f"Setting up data module with {batches_per_epoch} batches per epoch")
    if batches_per_epoch == 1:
        data_module = FullGraphDataModule(config, task_config=config.task)
    else:
        data_module = EdgeBatchDataModule(
            config=config, batches_per_epoch=batches_per_epoch
        )

    # define the strata for negative sampling
    stratify_by = config.task.edge_prediction_neg_sampling_stratify_by
    logger.info(f"Defining strata for negative sampling: {stratify_by}")
    edge_strata = get_edge_strata_from_artifacts(
        stratify_by=stratify_by,
        artifacts=data_module.other_artifacts,
    )

    # 3. create model
    # a. encoder
    logger.info("Preparing the encoder...")
    encoder = MessagePassingEncoder.from_config(
        config.model,
        data_module.num_node_features,
        edge_in_channels=data_module.num_edge_features,
    )
    # b. decoder/head
    logger.info("Preparing the head...")
    head = Decoder.from_config(config.model)
    logger.info("Preparing the task...")
    task = EdgePredictionTask(encoder, head, edge_strata=edge_strata)

    # 4. create lightning module
    logger.info("Creating lightning module...")
    model = EdgePredictionLightning(
        task,
        config=config.training,
    )

    # 5. trainer
    logger.info("Creating trainer...")
    trainer = NapistuTrainer(config)

    return {
        EXPERIMENT_DICT.DATA_MODULE: data_module,
        EXPERIMENT_DICT.MODEL: model,
        EXPERIMENT_DICT.TRAINER: trainer,
        EXPERIMENT_DICT.WANDB_LOGGER: wandb_logger,
    }
