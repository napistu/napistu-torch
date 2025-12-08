"""Workflows for configuring, training and evaluating models"""

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from pydantic import BaseModel, ConfigDict, field_validator

from napistu_torch.configs import ExperimentConfig, ModelConfig, RunManifest
from napistu_torch.constants import (
    EXPERIMENT_CONFIG,
    MODEL_CONFIG,
    PRETRAINED_COMPONENT_SOURCES,
    TASK_CONFIG,
    TRAINING_CONFIG,
    VALID_PRETRAINED_COMPONENT_SOURCES,
    WANDB_CONFIG,
)
from napistu_torch.evaluation.evaluation_manager import find_best_checkpoint
from napistu_torch.lightning.constants import (
    EXPERIMENT_DICT,
    TRAINER_MODES,
)
from napistu_torch.lightning.edge_batch_datamodule import EdgeBatchDataModule
from napistu_torch.lightning.full_graph_datamodule import FullGraphDataModule
from napistu_torch.lightning.tasks import EdgePredictionLightning
from napistu_torch.lightning.trainer import NapistuTrainer
from napistu_torch.load.checkpoints import Checkpoint
from napistu_torch.ml.hugging_face import HuggingFaceLoader
from napistu_torch.ml.wandb import (
    get_wandb_run_id_and_url,
    prepare_wandb_config,
    resume_wandb_logger,
    setup_wandb_logger,
)
from napistu_torch.models.constants import RELATION_AWARE_HEADS
from napistu_torch.models.heads import Decoder
from napistu_torch.models.message_passing_encoder import MessagePassingEncoder
from napistu_torch.tasks.edge_prediction import (
    EdgePredictionTask,
    get_edge_strata_from_artifacts,
)
from napistu_torch.utils.base_utils import CorruptionError

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
    run_manifest: RunManifest,
    resume_from: Optional[Path] = None,
    max_corruption_restarts: int = 30,
    logger: Optional[logging.Logger] = logger,
) -> NapistuTrainer:
    """
    Train a model using the provided run manifest.

    Handles both initial training and resuming from checkpoints, with automatic
    corruption recovery that reloads the experiment from manifest.

    Parameters
    ----------
    run_manifest : RunManifest
        Run manifest containing experiment configuration and metadata
    resume_from : Path, optional
        Path to a checkpoint to resume from (if None, starts from scratch)
    max_corruption_restarts : int, default=30
        Maximum number of times to restart after corruption detection
    logger : logging.Logger, optional
        Logger instance to use

    Returns
    -------
    NapistuTrainer
        The trained model's trainer instance
    """

    # Get checkpoint directory
    config = run_manifest.experiment_config
    checkpoint_dir = config.training.get_checkpoint_dir(config.output_dir)

    # Cleanup strategy depends on whether we're resuming
    if resume_from is None:
        # Fresh training - clean everything unless keep_best specified
        _cleanup_stale_checkpoints(
            checkpoint_dir, keep_checkpoint=None, keep_best=False, logger=logger
        )
    else:
        # Resuming - always preserve the checkpoint we're resuming from and the best
        _cleanup_stale_checkpoints(
            checkpoint_dir, keep_checkpoint=resume_from, keep_best=True, logger=logger
        )

    restart_count = 0
    current_checkpoint = resume_from
    experiment_dict = None

    while restart_count <= max_corruption_restarts:
        try:
            # Create/reload experiment dict for this attempt
            if experiment_dict is None:
                # we are resuming an experiment regardless of whether its fresh because we are working off of a manifest
                experiment_dict = resume_experiment(
                    run_manifest,
                    mode=TRAINER_MODES.TRAIN,
                    logger=logger,
                )
            else:
                # Retry after corruption - reload everything with resume_experiment
                logger.info("Reloading experiment from manifest for clean state...")
                experiment_dict = resume_experiment(
                    run_manifest,
                    mode=TRAINER_MODES.TRAIN,
                    logger=logger,
                )

            # Train with the current checkpoint
            logger.info("Starting training...")
            experiment_dict[EXPERIMENT_DICT.TRAINER].fit(
                experiment_dict[EXPERIMENT_DICT.MODEL],
                datamodule=experiment_dict[EXPERIMENT_DICT.DATA_MODULE],
                ckpt_path=current_checkpoint,
            )

            logger.info("Training workflow completed successfully")
            return experiment_dict[EXPERIMENT_DICT.TRAINER]

        except CorruptionError as e:
            if restart_count >= max_corruption_restarts:
                logger.error(
                    f"MPS memory corruption persisted after {max_corruption_restarts} restarts. "
                    f"Consider reducing memory pressure."
                )
                raise

            restart_count += 1
            logger.warning(
                f"MPS memory corruption detected: {e}\n"
                f"Restart {restart_count}/{max_corruption_restarts} - attempting recovery..."
            )

            # Find last checkpoint to resume from
            last_ckpt = checkpoint_dir / "last.ckpt"
            if last_ckpt.exists():
                current_checkpoint = last_ckpt
                logger.info(f"Will resume from: {current_checkpoint}")

                # Verify checkpoint state
                ckpt_data = torch.load(
                    current_checkpoint, weights_only=False, map_location="cpu"
                )
                logger.info(
                    f"Checkpoint state: epoch={ckpt_data['epoch']}, "
                    f"global_step={ckpt_data['global_step']}"
                )
            else:
                # No checkpoint exists (e.g., corruption during first epoch)
                # Set to None to start from scratch
                current_checkpoint = None
                logger.warning(
                    "No last.ckpt found for corruption recovery. Either checkpoints were not saved (enable save_checkpoints in the config) or the failure happened before the first checkpoint was saved. Will start training from scratch."
                )

            # Log corruption event to WandB if available
            if experiment_dict and experiment_dict.get(EXPERIMENT_DICT.WANDB_LOGGER):
                _log_corruption_to_wandb(
                    experiment_dict[EXPERIMENT_DICT.WANDB_LOGGER], restart_count, e
                )

            # Aggressive MPS cleanup
            if torch.backends.mps.is_available():
                logger.info("Cleaning up MPS cache...")
                torch.mps.synchronize()
                torch.mps.empty_cache()
            gc.collect()
            time.sleep(1)

            logger.info("Resuming training after cleanup...")
            continue

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
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Prepare the experiment for training.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration for the experiment
    logger : logging.Logger, optional
        Logger instance to use
    verbose: bool, default=True
        Whether to log verbose information

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
    if verbose:
        logger.info("Setting up W&B logger...")
    wandb_logger = setup_wandb_logger(config)

    # Initialize wandb by accessing the experiment (this triggers lazy initialization)
    if wandb_logger is not None:
        _ = wandb_logger.experiment
        wandb_run_id, wandb_run_url = get_wandb_run_id_and_url(wandb_logger, config)
    else:
        wandb_run_id, wandb_run_url = None, None

    # 2. Create Data Module
    if verbose:
        logger.info("Creating Data Module from config...")
    data_module = _create_data_module(config, logger=logger, verbose=verbose)

    # define the strata for negative sampling
    stratify_by = config.task.edge_prediction_neg_sampling_stratify_by
    if verbose:
        logger.info("Getting edge strata from artifacts...")
    edge_strata = get_edge_strata_from_artifacts(
        stratify_by=stratify_by,
        artifacts=data_module.other_artifacts,
    )

    # 3 create model
    # 3a. load a pretrained checkpoint if specified
    if config.model.use_pretrained_model:
        if verbose:
            logger.info("Loading pretrained checkpoint...")
        pretrained_checkpoint = _load_pretrained_checkpoint(
            config.model, logger=logger, verbose=verbose
        )
        # validate compatibility of checkpoint with data module (they should be trained on the same dataset or at least have compatible channels)
        pretrained_checkpoint.assert_same_napistu_data(data_module.napistu_data)
        # update model config so its attribute reflect the loaded encoder and head
        pretrained_checkpoint.update_model_config(config.model, inplace=True)

    # 3b. create the model based on the model config (which may have been updated based on the pretrained checkpoint)
    model = _create_model(
        config, data_module, edge_strata, logger=logger, verbose=verbose
    )

    # 3c. load pretrained weights if specified
    if config.model.use_pretrained_model:
        if verbose:
            logger.info("Loading pretrained weights...")
        _load_pretrained_weights(
            model, pretrained_checkpoint, config.model, logger=logger, verbose=verbose
        )

    # 4. trainer

    if verbose:
        logger.info("Creating NapistuTrainer from config...")
    trainer = NapistuTrainer(config)

    # 5. create a run manifest
    # Use the same naming scheme as wandb: config.name or generated name
    experiment_name = config.name or config.get_experiment_name()
    if verbose:
        logger.info(
            "Creating RunManifest with experiment_name = %s...", experiment_name
        )
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
    run_manifest: RunManifest,
    mode: str = TRAINER_MODES.EVAL,
    logger: logging.Logger = logger,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Resume an experiment using its run manifest.

    Parameters
    ----------
    run_manifest: RunManifest
        The run manifest
    mode: str, default=TRAINER_MODES.EVAL
        Trainer mode: TRAINER_MODES.TRAIN for training, TRAINER_MODES.EVAL for evaluation
    logger: logging.Logger, optional
        Logger instance to use
    verbose: bool, default=False
        Whether to log verbose information

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

    experiment_config = run_manifest.experiment_config

    # 1. Resume W&B Logger
    wandb_logger = resume_wandb_logger(run_manifest)

    # 2. Create Data Module
    data_module = _create_data_module(experiment_config, logger=logger)

    stratify_by = experiment_config.task.edge_prediction_neg_sampling_stratify_by
    if verbose:
        logger.info("Getting edge strata from artifacts...")
    edge_strata = get_edge_strata_from_artifacts(
        stratify_by=stratify_by,
        artifacts=data_module.other_artifacts,
    )

    # 3. create model
    model = _create_model(
        experiment_config, data_module, edge_strata, logger=logger, verbose=verbose
    )

    # 3a. load pretrained weights if specified
    if experiment_config.model.use_pretrained_model:
        if verbose:
            logger.info("Loading pretrained checkpoint...")
        pretrained_checkpoint = _load_pretrained_checkpoint(
            experiment_config.model, logger=logger, verbose=verbose
        )
        if verbose:
            logger.info("Loading pretrained weights...")
        _load_pretrained_weights(
            model,
            pretrained_checkpoint,
            experiment_config.model,
            logger=logger,
            verbose=verbose,
        )

    # 4. trainer
    if verbose:
        logger.info(f"Creating NapistuTrainer from config (mode={mode})...")
    trainer = NapistuTrainer(experiment_config, mode=mode, wandb_logger=wandb_logger)

    experiment_dict = {
        EXPERIMENT_DICT.DATA_MODULE: data_module,
        EXPERIMENT_DICT.MODEL: model,
        EXPERIMENT_DICT.TRAINER: trainer,
        EXPERIMENT_DICT.RUN_MANIFEST: run_manifest,
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


def _cleanup_stale_checkpoints(
    checkpoint_dir: Path,
    keep_checkpoint: Optional[Path] = None,
    keep_best: bool = False,
    logger: logging.Logger = logger,
) -> None:
    """
    Remove old checkpoints from previous runs before starting new training.

    This prevents accidentally resuming from stale checkpoints during corruption recovery.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing checkpoints
    keep_checkpoint : Path, optional
        Specific checkpoint to preserve (e.g., if explicitly resuming from it)
    keep_best : bool, default=False
        Whether to retain the best checkpoint (highest validation AUC). This is helpful if keep_checkpoint is 'last' and we
        also want to retain the best checkpoint for early stopping.
    logger : logging.Logger
        Logger instance
    """
    if not checkpoint_dir.exists():
        logger.info(f"Checkpoint directory does not exist yet: {checkpoint_dir}")
        return

    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoint_files:
        logger.info("No existing checkpoints to clean up")
        return

    # Determine which checkpoint to keep (if any)
    current_checkpoint_paths = []
    if keep_checkpoint:
        current_checkpoint_paths.append(keep_checkpoint.resolve())

    if keep_best:
        best_checkpoint_result = find_best_checkpoint(checkpoint_dir)
        if best_checkpoint_result is not None:
            best_checkpoint, _ = best_checkpoint_result
            best_checkpoint_resolved = best_checkpoint.resolve()
            if best_checkpoint_resolved not in current_checkpoint_paths:
                current_checkpoint_paths.append(best_checkpoint_resolved)
                logger.info(f"Preserving best checkpoint: {best_checkpoint.name}")
            else:
                logger.debug(
                    f"Best checkpoint already preserved: {best_checkpoint.name}"
                )
        else:
            logger.warning(
                f"No best checkpoint found in {checkpoint_dir}; ignoring `keep_best` = True"
            )

    removed_count = 0
    for ckpt_file in checkpoint_files:
        if ckpt_file.resolve() in current_checkpoint_paths:
            logger.info(f"Preserving checkpoint for resume: {ckpt_file.name}")
            continue

        try:
            ckpt_file.unlink()
            removed_count += 1
            logger.debug(f"Removed stale checkpoint: {ckpt_file.name}")
        except Exception as e:
            logger.warning(f"Failed to remove {ckpt_file.name}: {e}")

    if removed_count > 0:
        logger.info(
            f"Cleaned up {removed_count} stale checkpoint(s) from previous runs"
        )


def _create_data_module(
    config: ExperimentConfig,
    logger: logging.Logger = logger,
    verbose: bool = True,
) -> Union[FullGraphDataModule, EdgeBatchDataModule]:
    """Create the appropriate data module based on the configuration."""
    batches_per_epoch = config.training.batches_per_epoch
    if batches_per_epoch == 1:
        if verbose:
            logger.info("Creating FullGraphDataModule...")
        return FullGraphDataModule(config)
    else:
        if verbose:
            logger.info(
                "Creating EdgeBatchDataModule with batches_per_epoch = %s...",
                batches_per_epoch,
            )
        return EdgeBatchDataModule(config=config, batches_per_epoch=batches_per_epoch)


def _create_model(
    config: ExperimentConfig,
    data_module: Union[FullGraphDataModule, EdgeBatchDataModule],
    edge_strata: Optional[Union[pd.Series, pd.DataFrame]] = None,
    logger: logging.Logger = logger,
    verbose: bool = True,
) -> EdgePredictionLightning:
    """Create the model based on the configuration."""

    # a. encoder
    if verbose:
        logger.info("Creating MessagePassingEncoder from config...")
    encoder = MessagePassingEncoder.from_config(
        config.model,
        data_module.num_node_features,
        edge_in_channels=data_module.num_edge_features,
    )

    # b. decoder/head
    if verbose:
        logger.info("Creating Decoder from config...")
    if config.model.head in RELATION_AWARE_HEADS:
        num_relations = data_module.napistu_data.get_num_relations()
        if verbose:
            logger.info(
                f"Using relation-aware head '{config.model.head}' with {num_relations} relations"
            )
    else:
        num_relations = None

    head = Decoder.from_config(config.model, num_relations=num_relations)
    task = EdgePredictionTask(
        encoder,
        head,
        neg_sampling_ratio=config.task.edge_prediction_neg_sampling_ratio,
        edge_strata=edge_strata,
        neg_sampling_strategy=config.task.edge_prediction_neg_sampling_strategy,
        metrics=config.task.metrics,
        weight_loss_by_relation_frequency=config.task.weight_loss_by_relation_frequency,
        loss_weight_alpha=config.task.loss_weight_alpha,
    )

    # 4. create lightning module
    if verbose:
        logger.info("Creating EdgePredictionLightning from task and config...")
    model = EdgePredictionLightning(
        task,
        config=config.training,
    )

    return model


def _load_pretrained_checkpoint(
    model_config: ModelConfig,
    logger: logging.Logger = logger,
    verbose: bool = True,
) -> Checkpoint:
    """
    Load a pretrained model's checkpoint from HuggingFace or local file.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration containing pretrained model settings
    logger : logging.Logger, optional
        Logger instance to use

    Returns
    -------
    Checkpoint
        Loaded checkpoint object

    Raises
    ------
    ValueError
        If use_pretrained_model is False or pretrained_model_source is invalid
    FileNotFoundError
        If local checkpoint file doesn't exist
    """
    if not getattr(model_config, MODEL_CONFIG.USE_PRETRAINED_MODEL):
        raise ValueError(
            "use_pretrained_model must be True to load pretrained checkpoint"
        )

    pretrained_model_source = getattr(
        model_config, MODEL_CONFIG.PRETRAINED_MODEL_SOURCE
    )
    pretrained_model_path = getattr(model_config, MODEL_CONFIG.PRETRAINED_MODEL_PATH)
    pretrained_model_revision = getattr(
        model_config, MODEL_CONFIG.PRETRAINED_MODEL_REVISION
    )

    if pretrained_model_source == PRETRAINED_COMPONENT_SOURCES.HUGGINGFACE:
        if verbose:
            logger.info(
                f"Loading pretrained checkpoint from HuggingFace: {pretrained_model_path} "
                f"(revision: {pretrained_model_revision})"
            )
        hf_loader = HuggingFaceLoader(
            repo_id=pretrained_model_path, revision=pretrained_model_revision
        )
        checkpoint = hf_loader.load_checkpoint(raw_checkpoint=False)
    elif pretrained_model_source == PRETRAINED_COMPONENT_SOURCES.LOCAL:
        if verbose:
            logger.info(
                f"Loading pretrained checkpoint from local file: {pretrained_model_path}"
            )
        checkpoint = Checkpoint.load(pretrained_model_path)
    else:
        raise ValueError(
            f"Invalid pretrained model source: {pretrained_model_source}. "
            f"Valid sources are: {VALID_PRETRAINED_COMPONENT_SOURCES}"
        )

    return checkpoint


def _load_pretrained_weights(
    model: EdgePredictionLightning,
    checkpoint: Checkpoint,
    model_config: ModelConfig,
    logger: logging.Logger = logger,
    verbose: bool = True,
) -> None:
    """
    Load pretrained weights from a checkpoint and apply them to the model.

    This function:
    1. Extracts encoder and optionally head state dicts from the checkpoint
    2. Loads them into the model with proper prefix handling
    3. Freezes weights if specified in model_config

    Parameters
    ----------
    model : EdgePredictionLightning
        The Lightning model to load weights into
    checkpoint : Checkpoint
        Already loaded checkpoint object containing the pretrained weights
    model_config : ModelConfig
        Model configuration containing freezing settings
    logger : logging.Logger, optional
        Logger instance to use
    verbose : bool, default=True
        Whether to log verbose information

    Raises
    ------
    ValueError
        If no encoder or head state dict is found when required
    RuntimeError
        If any required encoder or head weights are missing from the checkpoint
    """

    if verbose:
        logger.info("Applying pretrained weights to model...")

    # Extract state dicts with proper prefix handling
    # Lightning checkpoints store weights with "task." prefix
    encoder_state_dict = {}
    head_state_dict = {}

    for key, value in checkpoint.state_dict.items():
        if key.startswith("task.encoder."):
            # Remove "task.encoder." prefix - encoder expects keys without prefix
            # e.g., "task.encoder.convs.0.lin_rel.weight" -> "convs.0.lin_rel.weight"
            encoder_key = key.replace("task.encoder.", "")
            encoder_state_dict[encoder_key] = value
        elif key.startswith("task.head."):
            # Remove "task.head." prefix - head expects keys without prefix
            # e.g., "task.head.bilinear.weight" -> "bilinear.weight"
            head_key = key.replace("task.head.", "")
            head_state_dict[head_key] = value

    # Load encoder weights
    if encoder_state_dict:
        if verbose:
            logger.info(f"Loading {len(encoder_state_dict)} encoder parameters...")
        missing_keys, unexpected_keys = model.task.encoder.load_state_dict(
            encoder_state_dict, strict=False
        )
        if missing_keys:
            raise RuntimeError(
                f"Failed to load pretrained encoder weights. Missing keys: {missing_keys}"
            )
        if unexpected_keys:
            logger.warning(f"Unexpected encoder keys (ignored): {unexpected_keys}")

        # Freeze encoder if requested
        if model_config.pretrained_model_freeze_encoder_weights:
            if verbose:
                logger.info("Freezing encoder weights")
            for param in model.task.encoder.parameters():
                param.requires_grad = False
    else:
        raise ValueError("No encoder state dict found in checkpoint")

    # Load head weights if requested
    if model_config.pretrained_model_load_head:
        # Check if head has any parameters (some heads like DotProductHead have none)
        head_has_params = len(list(model.task.head.parameters())) > 0

        if not head_has_params:
            if verbose:
                logger.info(
                    "Head has no trainable parameters (e.g., DotProductHead). "
                    "Skipping head weight loading."
                )
        elif head_state_dict:
            if verbose:
                logger.info(f"Loading {len(head_state_dict)} head parameters...")
            missing_keys, unexpected_keys = model.task.head.load_state_dict(
                head_state_dict, strict=False
            )
            if missing_keys:
                raise RuntimeError(
                    f"Failed to load pretrained head weights. Missing keys: {missing_keys}"
                )
            if unexpected_keys:
                logger.warning(f"Unexpected head keys (ignored): {unexpected_keys}")

            # Freeze head if requested
            if model_config.pretrained_model_freeze_head_weights:
                if verbose:
                    logger.info("Freezing head weights")
                for param in model.task.head.parameters():
                    param.requires_grad = False
        else:
            # Head has parameters but no state dict found in checkpoint
            raise ValueError(
                "No head state dict found in checkpoint but pretrained_model_load_head is True. "
                "The head has trainable parameters that need to be loaded."
            )

    if verbose:
        logger.info("✓ Pretrained weights loaded successfully")

    return None


def _log_corruption_to_wandb(
    wandb_logger: Optional[Any],
    restart_count: int,
    error: Exception,
) -> None:
    """
    Log corruption event to WandB if logger is available.

    Parameters
    ----------
    wandb_logger : Optional[Any]
        WandB logger instance, or None if not available
    restart_count : int
        Current restart attempt number
    error : Exception
        The corruption error that occurred
    """
    if wandb_logger is not None:
        try:
            wandb_logger.experiment.log(
                {
                    "corruption_restart": restart_count,
                    "corruption_error": str(error),
                }
            )
        except Exception as wandb_error:
            logger.warning(f"Failed to log corruption event to WandB: {wandb_error}")
