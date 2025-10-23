from napistu_torch.configs import ExperimentConfig
from lightning.pytorch.loggers import WandbLogger


def setup_wandb_logger(cfg: ExperimentConfig) -> WandbLogger:
    """
    Setup WandbLogger with comprehensive configuration.

    Parameters
    ----------
    cfg : ExperimentConfig
        Your experiment configuration

    Returns
    -------
    WandbLogger
        Configured WandbLogger instance
    """

    # Use the config's built-in methods for better organization
    run_name = cfg.name or cfg.wandb.get_run_name(cfg.model, cfg.task)
    enhanced_tags = cfg.wandb.get_enhanced_tags(cfg.model, cfg.task)

    # Add training-specific tags
    enhanced_tags.extend([f"lr_{cfg.training.lr}", f"epochs_{cfg.training.epochs}"])

    # Create the logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_name,
        group=cfg.wandb.group,
        tags=enhanced_tags,
        save_dir=cfg.wandb.save_dir,
        log_model=cfg.wandb.log_model,
        config=cfg.to_dict(),
        # Additional useful parameters
        entity=cfg.wandb.entity,  # Your wandb username/team
        notes=f"Training {cfg.model.encoder} for {cfg.task.task}",
        reinit=True,  # Allow reinitializing if needed
    )

    return wandb_logger
