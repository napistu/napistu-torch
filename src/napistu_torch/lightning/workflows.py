"""Workflows for configuring, training and evaluating models"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
from pydantic import BaseModel, ConfigDict, field_validator

from napistu_torch.configs import ExperimentConfig, RunManifest
from napistu_torch.constants import (
    EXPERIMENT_CONFIG,
    MODEL_CONFIG,
    TASK_CONFIG,
    TRAINING_CONFIG,
    WANDB_CONFIG,
)
from napistu_torch.evaluation.constants import EVALUATION_MANAGER
from napistu_torch.evaluation.evaluation_manager import EvaluationManager
from napistu_torch.lightning.constants import (
    EXPERIMENT_DICT,
    TRAINER_MODES,
)
from napistu_torch.lightning.edge_batch_datamodule import EdgeBatchDataModule
from napistu_torch.lightning.full_graph_datamodule import FullGraphDataModule
from napistu_torch.lightning.tasks import EdgePredictionLightning
from napistu_torch.lightning.trainer import NapistuTrainer
from napistu_torch.ml.wandb import (
    get_wandb_run_id_and_url,
    prepare_wandb_config,
    resume_wandb_logger,
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


# public functions


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
    config = run_manifest.experiment_config

    # Extract config values from the manifest's experiment_config dict
    task_config = getattr(config, EXPERIMENT_CONFIG.TASK)
    task = getattr(task_config, TASK_CONFIG.TASK)

    model_config = getattr(config, EXPERIMENT_CONFIG.MODEL)
    model_encoder = getattr(model_config, MODEL_CONFIG.ENCODER)
    model_head = getattr(model_config, MODEL_CONFIG.HEAD)
    model_hidden_channels = getattr(model_config, MODEL_CONFIG.HIDDEN_CHANNELS)
    model_num_layers = getattr(model_config, MODEL_CONFIG.NUM_LAYERS)
    model_use_edge_encoder = getattr(model_config, MODEL_CONFIG.USE_EDGE_ENCODER)
    model_edge_encoder_dim = getattr(model_config, MODEL_CONFIG.EDGE_ENCODER_DIM)

    training_config = getattr(config, EXPERIMENT_CONFIG.TRAINING)
    training_epochs = getattr(training_config, TRAINING_CONFIG.EPOCHS)
    training_lr = getattr(training_config, TRAINING_CONFIG.LR)
    training_batches_per_epoch = getattr(
        training_config, TRAINING_CONFIG.BATCHES_PER_EPOCH
    )

    seed = getattr(config, EXPERIMENT_CONFIG.SEED)
    wandb_config = getattr(config, EXPERIMENT_CONFIG.WANDB)
    wandb_project = run_manifest.wandb_project or getattr(
        wandb_config, WANDB_CONFIG.PROJECT
    )
    wandb_mode = getattr(wandb_config, WANDB_CONFIG.MODE)

    # Get batches_per_epoch from data module or fallback to config
    batches_per_epoch = getattr(data_module, TRAINING_CONFIG.BATCHES_PER_EPOCH, None)
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


def predict(
    experiment_dict: ExperimentDict,
    checkpoint: Optional[Path] = None,
) -> list[dict]:
    """
    Predict using the provided experiment dictionary.

    Parameters
    ----------
    experiment_dict: ExperimentDict
        Dictionary containing the experiment components:
        - data_module : Union[FullGraphDataModule, EdgeBatchDataModule]
        - model : pl.LightningModule (e.g., EdgePredictionLightning)
        - run_manifest : RunManifest
        - trainer : NapistuTrainer
        - wandb_logger : Optional[WandbLogger] (None when wandb is disabled)
    checkpoint: Optional[Path] = None
        Path to a checkpoint to use for prediction (if None, uses last checkpoint)

    Returns
    -------
    list[dict]
        List of dictionaries containing the predictions
    """

    if checkpoint is None:
        checkpoint = "last"
        logger.warning("No checkpoint provided, using last checkpoint")
    else:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint file not found at path: {checkpoint}")

    return experiment_dict[EXPERIMENT_DICT.TRAINER].predict(
        model=experiment_dict[EXPERIMENT_DICT.MODEL],
        datamodule=experiment_dict[EXPERIMENT_DICT.DATA_MODULE],
        ckpt_path=checkpoint,
    )


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
    logger.info("Creating Data Module from config...")
    data_module = _create_data_module(config)

    # define the strata for negative sampling
    stratify_by = config.task.edge_prediction_neg_sampling_stratify_by
    logger.info("Getting edge strata from artifacts...")
    edge_strata = get_edge_strata_from_artifacts(
        stratify_by=stratify_by,
        artifacts=data_module.other_artifacts,
    )

    # 3. create model
    model = _create_model(config, data_module, edge_strata)

    # 4. trainer
    logger.info("Creating NapistuTrainer from config...")
    trainer = NapistuTrainer(config)

    # 5. create a run manifest
    # Use the same naming scheme as wandb: config.name or generated name
    experiment_name = config.name or config.get_experiment_name()
    logger.info("Creating RunManifest with experiment_name = %s...", experiment_name)
    run_manifest = RunManifest(
        experiment_name=experiment_name,
        wandb_run_id=wandb_run_id,
        wandb_run_url=wandb_run_url,
        wandb_project=config.wandb.project,
        wandb_entity=config.wandb.entity,
        experiment_config=config,
    )

    experiment_dict = {
        EXPERIMENT_DICT.DATA_MODULE: data_module,
        EXPERIMENT_DICT.MODEL: model,
        EXPERIMENT_DICT.TRAINER: trainer,
        EXPERIMENT_DICT.RUN_MANIFEST: run_manifest,
        EXPERIMENT_DICT.WANDB_LOGGER: wandb_logger,
    }

    return experiment_dict


def resume_experiment(
    evaluation_manager: EvaluationManager,
    logger: logging.Logger = logger,
) -> Dict[str, Any]:
    """
    Resume an experiment using its EvaluationManager manifest.

    Parameters
    ----------
    evaluation_manager: EvaluationManager
        The evaluation manager
    logger: logging.Logger, optional
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

    manifest = getattr(evaluation_manager, EVALUATION_MANAGER.MANIFEST)
    experiment_config = getattr(
        evaluation_manager, EVALUATION_MANAGER.EXPERIMENT_CONFIG
    )

    # 1. Resume W&B Logger
    wandb_logger = resume_wandb_logger(manifest)

    # 2. Create Data Module
    data_module = _create_data_module(experiment_config)

    stratify_by = experiment_config.task.edge_prediction_neg_sampling_stratify_by
    logger.info("Getting edge strata from artifacts...")
    edge_strata = get_edge_strata_from_artifacts(
        stratify_by=stratify_by,
        artifacts=data_module.other_artifacts,
    )

    # 3. create model
    model = _create_model(experiment_config, data_module, edge_strata)

    # 4. trainer
    logger.info("Creating NapistuTrainer from config...")
    trainer = NapistuTrainer(
        experiment_config, mode=TRAINER_MODES.EVAL, wandb_logger=wandb_logger
    )

    experiment_dict = {
        EXPERIMENT_DICT.DATA_MODULE: data_module,
        EXPERIMENT_DICT.MODEL: model,
        EXPERIMENT_DICT.TRAINER: trainer,
        EXPERIMENT_DICT.RUN_MANIFEST: manifest,
        EXPERIMENT_DICT.WANDB_LOGGER: wandb_logger,
    }

    return experiment_dict


def test(
    experiment_dict: ExperimentDict, checkpoint: Optional[Path] = None
) -> list[dict]:

    if checkpoint is None:
        checkpoint = "last"
        logger.warning("No checkpoint provided, using last checkpoint")
    else:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

    test_results = experiment_dict[EXPERIMENT_DICT.TRAINER].test(
        model=experiment_dict[EXPERIMENT_DICT.MODEL],
        datamodule=experiment_dict[EXPERIMENT_DICT.DATA_MODULE],
        ckpt_path=checkpoint,
    )

    for key, value in test_results[0].items():
        if experiment_dict[EXPERIMENT_DICT.WANDB_LOGGER] is not None:
            experiment_dict[EXPERIMENT_DICT.WANDB_LOGGER].experiment.summary[
                key
            ] = value

    return test_results


# private functions


def _create_data_module(
    config: ExperimentConfig,
) -> Union[FullGraphDataModule, EdgeBatchDataModule]:
    """Create the appropriate data module based on the configuration."""
    batches_per_epoch = config.training.batches_per_epoch
    if batches_per_epoch == 1:
        logger.info("Creating FullGraphDataModule...")
        return FullGraphDataModule(config)
    else:
        logger.info(
            "Creating EdgeBatchDataModule with batches_per_epoch = %s...",
            batches_per_epoch,
        )
        return EdgeBatchDataModule(config=config, batches_per_epoch=batches_per_epoch)


def _create_model(
    config: ExperimentConfig,
    data_module: Union[FullGraphDataModule, EdgeBatchDataModule],
    edge_strata: Optional[Dict[str, Any]] = None,
) -> EdgePredictionLightning:
    """Create the model based on the configuration."""
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

    return model
