import logging
from typing import Optional, Tuple

from lightning.pytorch.loggers import WandbLogger

from napistu_torch.configs import ExperimentConfig

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
    if cfg.wandb.mode == "disabled":
        return None

    # Use the config's built-in method for run name
    experiment_name = cfg.name or cfg.get_experiment_name()

    # Get the save directory using the config method
    save_dir = cfg.wandb.get_save_dir(cfg.output_dir)

    # Create the logger with the config values
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=experiment_name,
        group=cfg.wandb.group,
        tags=cfg.wandb.tags,
        save_dir=save_dir,
        log_model=cfg.wandb.log_model,
        config=cfg.to_dict(),
        entity=cfg.wandb.entity,
        notes=f"Training {cfg.model.encoder} for {cfg.task.task}",
        reinit=True,
        offline=cfg.wandb.mode == "offline",  # Set offline mode if needed
    )

    return wandb_logger
