import logging
from typing import Optional, Tuple

from lightning.pytorch.loggers import WandbLogger

from napistu_torch.configs import (
    ExperimentConfig,
    RunManifest,
)
from napistu_torch.constants import (
    DATA_CONFIG,
    EXPERIMENT_CONFIG,
    MODEL_CONFIG,
    RUN_MANIFEST,
    TASK_CONFIG,
    TRAINING_CONFIG,
    WANDB_CONFIG,
)

logger = logging.getLogger(__name__)


def get_wandb_run_id_and_url(
    wandb_logger: Optional[WandbLogger], cfg: ExperimentConfig
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get the wandb run ID and URL from a WandbLogger.

    Parameters
    ----------
    wandb_logger : Optional[WandbLogger]
        The wandb logger instance (may be None if wandb is disabled)
    cfg : ExperimentConfig
        Experiment configuration containing wandb project and entity info

    Returns
    -------
    Tuple[Optional[str], Optional[str]]
        A tuple of (run_id, run_url). Both may be None if:
        - wandb_logger is None
        - The experiment hasn't been initialized yet
        - An error occurred accessing the run ID
    """
    wandb_run_id = None
    wandb_run_url = None

    if wandb_logger is not None:
        try:
            # Get run ID and URL directly from wandb API (most reliable)
            import wandb

            if wandb.run is not None:
                wandb_run_id = wandb.run.id
                wandb_run_url = wandb.run.url
                return wandb_run_id, wandb_run_url
        except (ImportError, AttributeError, RuntimeError):
            # Fallback: get from logger's experiment if available
            try:
                if (
                    hasattr(wandb_logger, "experiment")
                    and wandb_logger.experiment is not None
                ):
                    wandb_run_id = wandb_logger.experiment.id
                    # Try to get URL from experiment
                    if hasattr(wandb_logger.experiment, "url"):
                        wandb_run_url = wandb_logger.experiment.url
                    elif hasattr(wandb_logger.experiment, "get_url"):
                        wandb_run_url = wandb_logger.experiment.get_url()
                    else:
                        # Last resort: construct URL using config values (entity has default)
                        if wandb_run_id and cfg.wandb.project and cfg.wandb.entity:
                            wandb_run_url = f"https://wandb.ai/{cfg.wandb.entity}/{cfg.wandb.project}/runs/{wandb_run_id}"
            except (AttributeError, RuntimeError):
                logger.warning("Failed to get wandb run ID and URL")
                pass

    return wandb_run_id, wandb_run_url


def prepare_wandb_config(cfg: ExperimentConfig) -> None:
    """
    Prepare WandB configuration by computing and setting derived values.

    Modifies cfg.wandb in-place to set:
    - Enhanced tags based on model, task, and training config
    - Save directory (either user-specified or checkpoint_dir/wandb)

    Also creates the save directory if it doesn't exist.

    Parameters
    ----------
    cfg : ExperimentConfig
        Your experiment configuration (modified in-place)
    """
    # Compute and set enhanced tags
    cfg.wandb.tags = cfg.wandb.get_enhanced_tags(cfg.model, cfg.task)
    cfg.wandb.tags.extend([f"lr_{cfg.training.lr}", f"epochs_{cfg.training.epochs}"])

    # Compute and set save directory
    save_dir = cfg.wandb.get_save_dir(cfg.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    return None


def resume_wandb_logger(
    manifest: RunManifest,
) -> Optional[WandbLogger]:
    """
    Resume a W&B run using the run ID from the manifest.

    Parameters
    ----------
    manifest : RunManifest
        The original run manifest containing W&B run ID
    logger : logging.Logger
        Logger instance

    Returns
    -------
    Optional[WandbLogger]
        Resumed WandbLogger, or None if wandb is disabled or run ID missing
    """

    config = getattr(manifest, RUN_MANIFEST.EXPERIMENT_CONFIG)

    # If wandb is disabled in config, don't create logger
    wandb_config = getattr(config, EXPERIMENT_CONFIG.WANDB)
    if getattr(wandb_config, WANDB_CONFIG.MODE) == "disabled":
        logger.info("W&B logging disabled in config")
        return None

    # Need a run ID to resume
    run_id = getattr(manifest, RUN_MANIFEST.WANDB_RUN_ID)
    if run_id is None:
        logger.warning(
            "No W&B run ID found in manifest. " "Testing without W&B logging."
        )
        return None

    # Get save directory
    save_dir = wandb_config.get_save_dir(getattr(config, EXPERIMENT_CONFIG.OUTPUT_DIR))
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Resuming W&B run: {manifest.wandb_run_id}")

    # Create logger with resume='must' to ensure we resume the existing run
    wandb_logger = WandbLogger(
        project=getattr(manifest, RUN_MANIFEST.WANDB_PROJECT)
        or getattr(wandb_config, WANDB_CONFIG.PROJECT),
        entity=getattr(manifest, RUN_MANIFEST.WANDB_ENTITY)
        or getattr(wandb_config, WANDB_CONFIG.ENTITY),
        id=run_id,  # CRITICAL: This resumes the run
        resume="must",  # CRITICAL: Must resume, fail if run doesn't exist
        save_dir=str(save_dir),
        offline=getattr(wandb_config, WANDB_CONFIG.MODE) == "offline",
    )

    logger.info(f"Successfully resumed W&B run: {wandb_logger.experiment.url}")

    return wandb_logger


def setup_wandb_logger(cfg: ExperimentConfig) -> Optional[WandbLogger]:
    """
    Setup WandbLogger with configuration.

    Note: Call prepare_wandb_config() first to ensure cfg.wandb has all
    computed values set.

    If wandb mode is "disabled", returns None to avoid initializing wandb
    and triggering sentry/analytics.

    Parameters
    ----------
    cfg : ExperimentConfig
        Your experiment configuration (should be prepared with prepare_wandb_config)

    Returns
    -------
    Optional[WandbLogger]
        Configured WandbLogger instance, or None if wandb is disabled
    """
    # If wandb is disabled, don't create the logger at all
    wandb_config = getattr(cfg, EXPERIMENT_CONFIG.WANDB)
    if getattr(wandb_config, WANDB_CONFIG.MODE) == "disabled":
        return None

    # Use the config's built-in method for run name
    experiment_name = getattr(cfg, EXPERIMENT_CONFIG.NAME) or cfg.get_experiment_name()

    # Get the save directory using the config method
    save_dir = wandb_config.get_save_dir(getattr(cfg, EXPERIMENT_CONFIG.OUTPUT_DIR))

    # Create the logger with the config values
    wandb_logger = WandbLogger(
        project=getattr(wandb_config, WANDB_CONFIG.PROJECT),
        name=experiment_name,
        group=getattr(wandb_config, WANDB_CONFIG.GROUP),
        tags=getattr(wandb_config, WANDB_CONFIG.TAGS),
        save_dir=save_dir,
        log_model=getattr(wandb_config, WANDB_CONFIG.LOG_MODEL),
        config=_define_minimal_experiment_summaries(cfg),
        entity=getattr(wandb_config, WANDB_CONFIG.ENTITY),
        notes=f"Training {getattr(cfg, EXPERIMENT_CONFIG.MODEL).encoder} for {getattr(cfg, EXPERIMENT_CONFIG.TASK).task}",
        reinit=True,
        offline=getattr(wandb_config, WANDB_CONFIG.MODE)
        == "offline",  # Set offline mode if needed
    )

    return wandb_logger


def _define_minimal_experiment_summaries(cfg: ExperimentConfig) -> dict:
    """
    Extract only the key hyperparameters for W&B logging.

    This keeps the W&B UI clean by excluding paths, infrastructure settings,
    and other non-essential metadata.
    """

    model_config = getattr(cfg, EXPERIMENT_CONFIG.MODEL)
    task_config = getattr(cfg, EXPERIMENT_CONFIG.TASK)
    training_config = getattr(cfg, EXPERIMENT_CONFIG.TRAINING)

    data_config = getattr(cfg, EXPERIMENT_CONFIG.DATA)

    return {
        # Experiment metadata
        EXPERIMENT_CONFIG.NAME: getattr(cfg, EXPERIMENT_CONFIG.NAME),
        EXPERIMENT_CONFIG.SEED: getattr(cfg, EXPERIMENT_CONFIG.SEED),
        # Model architecture
        MODEL_CONFIG.ENCODER: getattr(model_config, MODEL_CONFIG.ENCODER),
        MODEL_CONFIG.HEAD: getattr(model_config, MODEL_CONFIG.HEAD),
        MODEL_CONFIG.HIDDEN_CHANNELS: getattr(
            model_config, MODEL_CONFIG.HIDDEN_CHANNELS
        ),
        MODEL_CONFIG.NUM_LAYERS: getattr(model_config, MODEL_CONFIG.NUM_LAYERS),
        MODEL_CONFIG.DROPOUT: getattr(model_config, MODEL_CONFIG.DROPOUT),
        MODEL_CONFIG.USE_EDGE_ENCODER: getattr(
            model_config, MODEL_CONFIG.USE_EDGE_ENCODER
        ),
        MODEL_CONFIG.EDGE_ENCODER_DIM: getattr(
            model_config, MODEL_CONFIG.EDGE_ENCODER_DIM
        ),
        # Task config
        TASK_CONFIG.TASK: getattr(task_config, TASK_CONFIG.TASK),
        TASK_CONFIG.METRICS: getattr(task_config, TASK_CONFIG.METRICS),
        # Training hyperparameters
        TRAINING_CONFIG.LR: getattr(training_config, TRAINING_CONFIG.LR),
        TRAINING_CONFIG.WEIGHT_DECAY: getattr(
            training_config, TRAINING_CONFIG.WEIGHT_DECAY
        ),
        TRAINING_CONFIG.OPTIMIZER: getattr(training_config, TRAINING_CONFIG.OPTIMIZER),
        TRAINING_CONFIG.SCHEDULER: getattr(training_config, TRAINING_CONFIG.SCHEDULER),
        TRAINING_CONFIG.EPOCHS: getattr(training_config, TRAINING_CONFIG.EPOCHS),
        TRAINING_CONFIG.BATCHES_PER_EPOCH: getattr(
            training_config, TRAINING_CONFIG.BATCHES_PER_EPOCH
        ),
        # Data config (just the name, not paths)
        DATA_CONFIG.NAPISTU_DATA_NAME: getattr(
            data_config, DATA_CONFIG.NAPISTU_DATA_NAME
        ),
    }
