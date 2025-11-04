"""CLI for Napistu-Torch training"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from napistu_torch._cli import (
    prepare_config,
    setup_logging,
    verbosity_option,
)
from napistu_torch.lightning.workflows import fit_model, prepare_experiment


@click.group()
def cli():
    """Napistu-Torch: GNN training for network integration"""
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--seed", type=int, help="Override random seed")
@click.option(
    "--wandb-mode",
    type=click.Choice(["online", "offline", "disabled"], case_sensitive=False),
    help="Override W&B logging mode",
)
@click.option("--fast-dev-run", is_flag=True, help="Run 1 batch for quick debugging")
@click.option(
    "--log-dir",
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Directory for log files (default: current directory)",
)
@click.option(
    "--resume",
    type=click.Path(exists=True, path_type=Path),
    help="Resume training from checkpoint",
)
@click.option(
    "--set",
    "overrides",
    multiple=True,
    help="Override config values (e.g., --set training.epochs=100 --set model.hidden_channels=256)",
)
@verbosity_option
def train(
    config_path: Path,
    seed: Optional[int],
    wandb_mode: Optional[str],
    fast_dev_run: bool,
    log_dir: Path,
    resume: Optional[Path],
    overrides: tuple[str, ...],
    verbosity: str,
):
    """
    Train a GNN model using the specified configuration.

    CONFIG_PATH: Path to YAML configuration file

    \b
    Examples:
        # Basic training
        $ napistu-torch train config.yaml

        # Override specific config values
        $ napistu-torch train config.yaml --set wandb.mode=disabled --set training.epochs=50

        # Quick debug run
        $ napistu-torch train config.yaml --fast-dev-run --wandb-mode disabled

        # Resume from checkpoint
        $ napistu-torch train config.yaml --resume checkpoints/best-epoch=10-val_auc=0.85.ckpt

        # Save logs to specific directory
        $ napistu-torch train config.yaml --log-dir ./logs/experiment1
    """
    # Setup logging first
    logger, _ = setup_logging(
        log_dir=log_dir,
        verbosity=verbosity,
    )

    config = prepare_config(
        config_path=config_path,
        seed=seed,
        wandb_mode=wandb_mode,
        fast_dev_run=fast_dev_run,
        overrides=overrides,
        logger=logger,
    )
    if resume:
        logger.info(f"  Resume from: {resume}")
    logger.info("=" * 80)

    # Run training workflow
    try:
        logger.info("Starting training workflow...")

        experiment_dict = prepare_experiment(config, logger=logger)
        fit_model(experiment_dict, resume_from=resume, logger=logger)

        logger.info("Training completed successfully! 🎉")

    except click.Abort:
        # User-friendly abort (already logged)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Training failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
