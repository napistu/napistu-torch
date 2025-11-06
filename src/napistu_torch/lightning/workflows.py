"""Workflows for configuring, training and evaluating models"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pydantic import BaseModel, ConfigDict, field_validator

from napistu_torch.configs import ExperimentConfig, RunManifest
from napistu_torch.lightning.constants import EXPERIMENT_DICT
from napistu_torch.lightning.edge_batch_datamodule import EdgeBatchDataModule
from napistu_torch.lightning.full_graph_datamodule import FullGraphDataModule
from napistu_torch.lightning.tasks import EdgePredictionLightning
from napistu_torch.lightning.trainer import NapistuTrainer
from napistu_torch.ml.wandb import (
    get_wandb_run_id_and_url,
    prepare_wandb_config,
    setup_wandb_logger,
)
from napistu_torch.models.heads import Decoder
from napistu_torch.models.message_passing_encoder import MessagePassingEncoder
from napistu_torch.tasks.edge_prediction import (
    EdgePredictionTask,
    get_edge_strata_from_artifacts,
)

logger = logging.getLogger(__name__)


class ExperimentDict(BaseModel):
    """
    Pydantic model for validating experiment_dict structure.

    Ensures all required components are present and of correct types.
    """

    data_module: Any
    model: Any
    run_manifest: Any
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

    @field_validator(EXPERIMENT_DICT.RUN_MANIFEST)
    @classmethod
    def validate_run_manifest(cls, v):
        """Validate that run_manifest is a RunManifest."""
        if not isinstance(v, RunManifest):
            raise TypeError(
                f"run_manifest must be a RunManifest, got {type(v).__name__}"
            )
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
        """Validate that wandb_logger is a WandbLogger or None (when disabled)."""
        # None is allowed when wandb is disabled
        if v is None:
            return v
        # Just check the class name to avoid import path issues
        if "WandbLogger" not in type(v).__name__:
            raise TypeError(
                f"wandb_logger must be a WandbLogger or None, got {type(v).__name__}"
            )
        return v

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )


def fit_model(
    experiment_dict: Dict[str, Any],
    resume_from: Optional[Path] = None,
    logger: Optional = logger,
) -> NapistuTrainer:
    """
    Train a model using the provided experiment dictionary.

    Parameters
    ----------
    experiment_dict : Dict[str, Any]
        Dictionary containing the experiment components:
        - data_module : Union[FullGraphDataModule, EdgeBatchDataModule]
        - model : pl.LightningModule (e.g., EdgePredictionLightning)
        - run_manifest : RunManifest
        - trainer : NapistuTrainer
        - wandb_logger : Optional[WandbLogger] (None when wandb is disabled)
    resume_from : Path, optional
        Path to a checkpoint to resume from
    logger : logging.Logger, optional
        Logger instance to use

    Returns
    -------
    NapistuTrainer
        The trainer instance
    """

    # Validate experiment_dict structure - Pydantic will raise ValidationError with detailed info
    ExperimentDict(
        data_module=experiment_dict[EXPERIMENT_DICT.DATA_MODULE],
        model=experiment_dict[EXPERIMENT_DICT.MODEL],
        run_manifest=experiment_dict[EXPERIMENT_DICT.RUN_MANIFEST],
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


def log_experiment_overview(
    experiment_dict: Dict[str, Any], logger: logging.Logger = logger
) -> None:
    """
    Log a comprehensive overview of the experiment configuration.

    Parameters
    ----------
    experiment_dict : Dict[str, Any]
        Dictionary containing the experiment components (from prepare_experiment),
        including the run_manifest
    logger : logging.Logger, optional
        Logger instance to use
    """
    data_module = experiment_dict[EXPERIMENT_DICT.DATA_MODULE]
    run_manifest = experiment_dict[EXPERIMENT_DICT.RUN_MANIFEST]
    config_dict = run_manifest.experiment_config

    # Extract config values from the manifest's experiment_config dict
    task = config_dict.get("task", {}).get("task", "unknown")
    model_config = config_dict.get("model", {})
    model_encoder = model_config.get("encoder", "unknown")
    model_head = model_config.get("head", "unknown")
    model_hidden_channels = model_config.get("hidden_channels", "unknown")
    model_num_layers = model_config.get("num_layers", "unknown")
    model_use_edge_encoder = model_config.get("use_edge_encoder", False)
    model_edge_encoder_dim = model_config.get("edge_encoder_dim", None)
    training_epochs = config_dict.get("training", {}).get("epochs", "unknown")
    training_lr = config_dict.get("training", {}).get("lr", "unknown")
    training_batches_per_epoch = config_dict.get("training", {}).get(
        "batches_per_epoch", "unknown"
    )
    seed = config_dict.get("seed", "unknown")
    wandb_project = run_manifest.wandb_project or config_dict.get("wandb", {}).get(
        "project", "unknown"
    )
    wandb_mode = config_dict.get("wandb", {}).get("mode", "unknown")

    # Get batches_per_epoch from data module or fallback to config
    batches_per_epoch = getattr(data_module, "batches_per_epoch", None)
    if batches_per_epoch is None:
        batches_per_epoch = training_batches_per_epoch

    logger.info("=" * 80)
    logger.info("Experiment Overview:")
    logger.info(f"  Experiment Name: {run_manifest.experiment_name or 'unnamed'}")
    logger.info(f"  Task: {task}")
    logger.info("  Model:")
    logger.info(
        f"    Encoder: {model_encoder}, Hidden Channels: {model_hidden_channels}, Layers: {model_num_layers}"
    )
    if model_use_edge_encoder:
        logger.info(f"    Edge Encoder: dim={model_edge_encoder_dim}")
    logger.info(f"    Head: {model_head}")
    logger.info(
        f"  Training: {training_epochs} epochs, lr={training_lr}, batches_per_epoch={training_batches_per_epoch}"
    )
    logger.info(f"  Seed: {seed}")
    logger.info(f"  W&B: project={wandb_project}, mode={wandb_mode}")
    if run_manifest.wandb_run_id:
        logger.info(f"  W&B Run ID: {run_manifest.wandb_run_id}")
    if run_manifest.wandb_run_url:
        logger.info(f"  W&B Run URL: {run_manifest.wandb_run_url}")
    logger.info(
        f"  Data Module: {type(data_module).__name__} ({batches_per_epoch} batches per epoch)"
    )
    logger.info("=" * 80)


def prepare_experiment(
    config: ExperimentConfig,
    logger: logging.Logger = logger,
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
        - run_manifest : RunManifest
        - trainer : NapistuTrainer
        - wandb_logger : Optional[WandbLogger] (None when wandb is disabled)
    """

    # Set seed
    pl.seed_everything(config.seed, workers=True)

    # 1. Setup W&B Logger
    # create an output directory and update the wandb config based on the model and training configs
    prepare_wandb_config(config)
    # create the actual wandb logger
    logger.info("Setting up W&B logger...")
    wandb_logger = setup_wandb_logger(config)

    # Initialize wandb by accessing the experiment (this triggers lazy initialization)
    if wandb_logger is not None:
        _ = wandb_logger.experiment
        wandb_run_id, wandb_run_url = get_wandb_run_id_and_url(wandb_logger, config)
    else:
        wandb_run_id, wandb_run_url = None, None

    # 2. Create Data Module
    batches_per_epoch = config.training.batches_per_epoch
    if batches_per_epoch == 1:
        logger.info("Creating FullGraphDataModule...")
        data_module = FullGraphDataModule(config)
    else:
        logger.info(
            "Creating EdgeBatchDataModule with batches_per_epoch = %s...",
            batches_per_epoch,
        )
        data_module = EdgeBatchDataModule(
            config=config, batches_per_epoch=batches_per_epoch
        )

    # define the strata for negative sampling
    stratify_by = config.task.edge_prediction_neg_sampling_stratify_by
    logger.info("Getting edge strata from artifacts...")
    edge_strata = get_edge_strata_from_artifacts(
        stratify_by=stratify_by,
        artifacts=data_module.other_artifacts,
    )

    # 3. create model
    # a. encoder
    logger.info("Creating MessagePassingEncoder from config...")
    encoder = MessagePassingEncoder.from_config(
        config.model,
        data_module.num_node_features,
        edge_in_channels=data_module.num_edge_features,
    )
    # b. decoder/head
    logger.info("Creating Decoder from config...")
    head = Decoder.from_config(config.model)
    task = EdgePredictionTask(encoder, head, edge_strata=edge_strata)

    # 4. create lightning module
    logger.info("Creating EdgePredictionLightning from task and config...")
    model = EdgePredictionLightning(
        task,
        config=config.training,
    )

    # 5. trainer
    logger.info("Creating NapistuTrainer from config...")
    trainer = NapistuTrainer(config)

    # 6. create a run manifest
    # Use the same naming scheme as wandb: config.name or generated name
    experiment_name = config.name or config.get_experiment_name()
    logger.info("Creating RunManifest with experiment_name = %s...", experiment_name)
    run_manifest = RunManifest(
        experiment_name=experiment_name,
        wandb_run_id=wandb_run_id,
        wandb_run_url=wandb_run_url,
        wandb_project=config.wandb.project,
        wandb_entity=config.wandb.entity,
        experiment_config=config.model_dump(mode="json"),
    )

    experiment_dict = {
        EXPERIMENT_DICT.DATA_MODULE: data_module,
        EXPERIMENT_DICT.MODEL: model,
        EXPERIMENT_DICT.TRAINER: trainer,
        EXPERIMENT_DICT.RUN_MANIFEST: run_manifest,
        EXPERIMENT_DICT.WANDB_LOGGER: wandb_logger,
    }

    return experiment_dict
