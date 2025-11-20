"""CLI for Napistu-Torch training"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from napistu_torch._cli import (
    log_deferred_messages,
    prepare_config,
    setup_logging,
    verbosity_option,
)
from napistu_torch.configs import create_template_yaml
from napistu_torch.constants import RUN_MANIFEST, RUN_MANIFEST_DEFAULTS
from napistu_torch.evaluation.evaluation_manager import EvaluationManager
from napistu_torch.lightning.constants import EXPERIMENT_DICT
from napistu_torch.lightning.workflows import (
    fit_model,
    log_experiment_overview,
    prepare_experiment,
    resume_experiment,
)
from napistu_torch.lightning.workflows import (
    test as run_test_workflow,
)

# Module-level logger and console - will be initialized when CLI is invoked
logger = None
console = None


@click.group()
def cli():
    """Napistu-Torch: GNN training for network integration"""
    # Set up logging only when CLI is actually invoked, not at import time
    # This prevents interfering with pytest's caplog fixture during tests
    # Note: Individual commands may set up their own logging (e.g., train command)
    global logger, console
    if logger is None:
        # Use napistu's setup_logging for basic CLI logging
        from napistu._cli import setup_logging as napistu_setup_logging

        logger, console = napistu_setup_logging()


@cli.command()
@click.argument(
    "experiment_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional checkpoint path to use instead of the best checkpoint discovered",
)
def test(experiment_dir: Path, checkpoint: Optional[Path]):
    """Run evaluation for a finished experiment located at EXPERIMENT_DIR."""

    evaluation_manager = EvaluationManager(experiment_dir)
    checkpoint_path = checkpoint or evaluation_manager.best_checkpoint_path

    if checkpoint_path is None:
        raise click.ClickException(
            "No checkpoint found. Provide --checkpoint or ensure checkpoints exist."
        )

    experiment_dict = resume_experiment(evaluation_manager)
    run_test_workflow(experiment_dict, checkpoint_path)


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
    "--out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for all run artifacts (logs, checkpoints, manifest). "
    "If not specified, uses checkpoint_dir from config.",
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
    out_dir: Optional[Path],
    resume: Optional[Path],
    overrides: tuple[str, ...],
    verbosity: str,
):
    """
    Train a GNN model using the specified configuration.

    CONFIG_PATH: Path to YAML configuration file

    \b
    Examples:
        # Basic training (outputs to checkpoint_dir from config)
        $ napistu-torch train config.yaml

        # Specify custom output directory
        $ napistu-torch train config.yaml --out-dir ./experiments/run_001

        # Override specific config values
        $ napistu-torch train config.yaml --set training.epochs=50 --out-dir ./quick_test

        # Quick debug run
        $ napistu-torch train config.yaml --fast-dev-run --wandb-mode disabled

        # Resume from checkpoint
        $ napistu-torch train config.yaml --resume checkpoints/best.ckpt
    """

    # Prepare config to respect named and wildcard overrides
    config, config_messages = prepare_config(
        config_path=config_path,
        seed=seed,
        wandb_mode=wandb_mode,
        fast_dev_run=fast_dev_run,
        overrides=overrides,
    )

    # Override output_dir if --out-dir provided
    if out_dir is not None:
        config_messages.append(f"Overriding output_dir with --out-dir: {out_dir}")
        config.output_dir = out_dir.resolve()
    else:
        config.output_dir = config.output_dir.resolve()

    # Compute derived directories
    checkpoint_dir = config.training.get_checkpoint_dir(config.output_dir)
    log_dir = config.output_dir / "logs"
    wandb_dir = config.wandb.get_save_dir(config.output_dir)

    # Setup logging
    logger, _ = setup_logging(
        log_dir=log_dir,
        verbosity=verbosity,
    )

    # Log all deferred messages
    log_deferred_messages(
        logger=logger,
        config_messages=config_messages,
        config=config,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        wandb_dir=wandb_dir,
        resume=resume,
    )

    if resume:
        logger.info(f"  Resume from: {resume}")
    logger.info("=" * 80)

    # Run training workflow
    try:
        logger.info("Starting training workflow...")

        experiment_dict = prepare_experiment(config, logger=logger)
        log_experiment_overview(experiment_dict, logger=logger)

        # save manifest to file
        manifest_path = (
            config.output_dir / RUN_MANIFEST_DEFAULTS[RUN_MANIFEST.MANIFEST_FILENAME]
        )
        experiment_dict[EXPERIMENT_DICT.RUN_MANIFEST].to_yaml(manifest_path)
        logger.info(f"Saved run manifest to {manifest_path}")

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


@cli.group()
def utils():
    """Utility commands for Napistu-Torch"""
    pass


@utils.command("create-template-yaml")
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--sbml-dfs-path",
    type=click.Path(path_type=Path),
    help="Path to SBML_dfs pickle file (default: placeholder)",
)
@click.option(
    "--napistu-graph-path",
    type=click.Path(path_type=Path),
    help="Path to NapistuGraph pickle file (default: placeholder)",
)
@click.option(
    "--name",
    type=str,
    help="Experiment name (default: omitted)",
)
def create_template_yaml_cmd(
    output_path: Path,
    sbml_dfs_path: Optional[Path],
    napistu_graph_path: Optional[Path],
    name: Optional[str],
):
    """
    Create a minimal YAML template file for experiment configuration.

    OUTPUT_PATH: Path where the YAML template file will be written

    \b
    Examples:
        # Create template with placeholder paths
        $ napistu-torch utils create-template-yaml config.yaml

        # Create template with specific paths
        $ napistu-torch utils create-template-yaml config.yaml \\
            --sbml-dfs-path data/sbml_dfs.pkl \\
            --napistu-graph-path data/graph.pkl \\
            --name my_experiment
    """
    try:
        create_template_yaml(
            output_path=output_path,
            sbml_dfs_path=sbml_dfs_path,
            napistu_graph_path=napistu_graph_path,
            name=name,
        )
        click.echo(f"✓ Template created at: {output_path}")
        click.echo("  You can now edit this file and use it with 'napistu-torch train'")
    except Exception as e:
        click.echo(f"✗ Failed to create template: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
