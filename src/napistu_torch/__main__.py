"""CLI for Napistu-Torch training"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from napistu_torch._cli import (
    format_named_overrides,
    log_deferred_messages,
    prepare_config,
    setup_logging,
    verbosity_option,
)
from napistu_torch.configs import create_template_yaml
from napistu_torch.constants import RUN_MANIFEST, RUN_MANIFEST_DEFAULTS
from napistu_torch.evaluation.evaluation_manager import EvaluationManager
from napistu_torch.lightning.constants import EXPERIMENT_DICT, TRAINER_MODES
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
    # Also allows sphinx-click to introspect the CLI without executing this code
    global logger, console
    if logger is None:
        try:
            # Use napistu's setup_logging for basic CLI logging
            from napistu._cli import setup_logging as napistu_setup_logging

            logger, console = napistu_setup_logging()
        except ImportError:
            # Fallback for documentation builds or when napistu isn't available
            import logging

            logger = logging.getLogger(__name__)
            console = None


@cli.command()
@click.argument(
    "experiment_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("repo_id", type=str)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Checkpoint to publish (default: best checkpoint)",
)
@click.option(
    "--message",
    type=str,
    default=None,
    help="Custom commit message (default: auto-generated from experiment)",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Explicitly confirm overwriting existing model in repo",
)
def publish(
    experiment_dir: Path,
    repo_id: str,
    checkpoint: Optional[Path],
    message: Optional[str],
    overwrite: bool,
):
    """
    Publish a trained model to HuggingFace Hub.

    Automatically creates a private repository if it doesn't exist.
    Repositories can be made public manually on huggingface.co after curation.

    EXPERIMENT_DIR: Path to experiment directory containing manifest and checkpoints

    REPO_ID: HuggingFace repository in format 'username/repo-name'

    \b
    Examples:
        # First upload - creates new private repo
        $ napistu-torch publish experiments/run_001 shackett/napistu-sage-octopus

        # Update existing model
        $ napistu-torch publish experiments/run_002 shackett/napistu-sage-octopus --overwrite

        # Publish specific checkpoint with custom message
        $ napistu-torch publish experiments/run_001 shackett/napistu-transe-v1 \\
            --checkpoint experiments/run_001/checkpoints/epoch=50.ckpt \\
            --message "Improved model with increased dropout"

    \b
    Note: Requires 'napistu-torch[lightning]' to be installed.
          Authenticate first with: huggingface-cli login
    """
    # Import EvaluationManager
    from napistu_torch.evaluation.evaluation_manager import EvaluationManager

    # Initialize manager
    evaluation_manager = EvaluationManager(experiment_dir)

    # Determine checkpoint path
    checkpoint_path = checkpoint or evaluation_manager.best_checkpoint_path
    if checkpoint_path is None:
        raise click.ClickException(
            "No checkpoint found. Provide --checkpoint or ensure checkpoints exist."
        )

    # Publish to HuggingFace Hub
    try:
        repo_url = evaluation_manager.publish_to_huggingface(
            repo_id=repo_id,
            checkpoint_path=checkpoint,
            commit_message=message,
            overwrite=overwrite,
        )

        # Success output
        click.echo()
        click.echo(click.style("✅ Published successfully!", fg="green", bold=True))
        click.echo(f"   URL: {repo_url}")
        click.echo()
        click.echo("   Uploaded files:")
        click.echo("     • model.ckpt")
        click.echo("     • config.json")
        click.echo("     • README.md")
        click.echo()
        click.echo(click.style("💡 Repository is private by default", fg="yellow"))
        click.echo(f"   Make public at: {repo_url}/settings")

    except RuntimeError as e:
        # Authentication error
        if "authentication" in str(e).lower():
            click.echo(click.style("\n✗ HuggingFace authentication failed", fg="red"))
            click.echo("\nAuthenticate with: huggingface-cli login")
        raise click.ClickException(str(e))
    except ValueError as e:
        # Validation error (e.g., repo exists without --overwrite)
        raise click.ClickException(str(e))


@cli.command()
@click.argument(
    "out_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--checkpoint",
    type=str,
    default="last",
    help="Checkpoint name or path. Can be 'last' (default), a checkpoint filename (e.g., 'best-epoch=50-val_auc=0.85.ckpt'), or a full path.",
)
@verbosity_option
def resume(out_dir: Path, checkpoint: str, verbosity: str):
    """
    Resume training for an existing experiment.

    Resumes training from the last checkpoint (or specified checkpoint) and reuses
    the existing WandB run ID from the experiment manifest.

    OUT_DIR: Path to experiment directory containing manifest and checkpoints

    \b
    Examples:
        # Resume from last checkpoint (default)
        $ napistu-torch resume ./experiments/run_001

        # Resume from best checkpoint
        $ napistu-torch resume ./experiments/run_001 --checkpoint best-epoch=50-val_auc=0.85.ckpt

        # Resume from specific checkpoint by filename
        $ napistu-torch resume ./experiments/run_001 --checkpoint best-epoch=34-val_auc=0.7533.ckpt

        # Resume from checkpoint by full path
        $ napistu-torch resume ./experiments/run_001 --checkpoint /path/to/checkpoint.ckpt
    """
    # Setup logging
    log_dir = out_dir / "logs"
    logger, _ = setup_logging(log_dir=log_dir, verbosity=verbosity)

    logger.info("=" * 80)
    logger.info("Resuming experiment...")
    logger.info(f"  Experiment directory: {out_dir}")

    try:
        # Load experiment from manifest
        evaluation_manager = EvaluationManager(out_dir)

        # Determine checkpoint to use
        try:
            checkpoint_path = evaluation_manager._resolve_checkpoint_path(checkpoint)
        except ValueError as e:
            raise click.ClickException(
                "No checkpoint found. Provide --checkpoint or ensure checkpoints exist."
            ) from e
        except FileNotFoundError as e:
            raise click.ClickException(str(e)) from e

        logger.info(f"  Resuming from: {checkpoint_path}")
        logger.info("=" * 80)

        # Resume experiment (in train mode)
        experiment_dict = resume_experiment(
            evaluation_manager, mode=TRAINER_MODES.TRAIN, logger=logger
        )

        # Continue training
        fit_model(experiment_dict, resume_from=checkpoint_path, logger=logger)

        logger.info("Training resumed and completed successfully! 🎉")

    except click.Abort:
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Failed to resume experiment: {e}")
        sys.exit(1)


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
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for all run artifacts (logs, checkpoints, manifest). "
    "If not specified, uses checkpoint_dir from config.",
)
@click.option("--seed", type=int, help="Override random seed")
@click.option("--fast_dev_run", is_flag=True, help="Run 1 batch for quick debugging")
@click.option(
    "--encoder", type=str, help="The model encoder (e.g., sage, gcn, gat, graph_conv)"
)
@click.option(
    "--head",
    type=str,
    help="The model head (e.g., node_classification, dot_product, mlp, bilinear, rotate, transe, distmult)",
)
@click.option("--hidden_channels", "hidden_channels", type=int, help="Hidden channels")
@click.option("--dropout", type=float, help="Dropout rate")
@click.option("--lr", type=float, help="Learning rate")
@click.option("--weight_decay", "weight_decay", type=float, help="Weight decay")
@click.option("--optimizer", type=str, help="Optimizer")
@click.option("--epochs", type=int, help="Number of epochs")
@click.option("--wandb_group", type=str, help="WandB group")
@click.option(
    "--wandb_mode",
    type=click.Choice(["online", "offline", "disabled"], case_sensitive=False),
    help="Override W&B logging mode",
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
    out_dir: Optional[Path],
    seed: Optional[int],
    fast_dev_run: bool,
    encoder: Optional[str],
    head: Optional[str],
    hidden_channels: Optional[int],
    dropout: Optional[float],
    lr: Optional[float],
    weight_decay: Optional[float],
    optimizer: Optional[str],
    epochs: Optional[int],
    wandb_mode: Optional[str],
    wandb_group: Optional[str],
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
        $ napistu-torch train config.yaml --fast_dev_run --wandb_mode disabled
    """

    # Convert all named CLI parameters to overrides
    named_overrides, named_messages = format_named_overrides(
        seed=seed,
        fast_dev_run=fast_dev_run,
        encoder=encoder,
        head=head,
        hidden_channels=hidden_channels,
        dropout=dropout,
        lr=lr,
        optimizer=optimizer,
        weight_decay=weight_decay,
        epochs=epochs,
        wandb_group=wandb_group,
        wandb_mode=wandb_mode,
    )

    # Combine with existing --set overrides
    all_overrides = tuple(list(overrides) + named_overrides)

    # Prepare config - now much simpler!
    config, config_messages = prepare_config(
        config_path=config_path,
        overrides=all_overrides,
    )

    # Combine all messages (named overrides + config messages)
    all_messages = named_messages + config_messages

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
        config_messages=all_messages,
        config=config,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        wandb_dir=wandb_dir,
    )

    logger.info("=" * 80)

    # Run training workflow
    try:
        logger.info("Starting training workflow...")

        # Prepare new experiment
        manifest_path = (
            config.output_dir / RUN_MANIFEST_DEFAULTS[RUN_MANIFEST.MANIFEST_FILENAME]
        )
        experiment_dict = prepare_experiment(config, logger=logger)
        log_experiment_overview(experiment_dict, logger=logger)
        experiment_dict[EXPERIMENT_DICT.RUN_MANIFEST].to_yaml(manifest_path)
        logger.info(f"Saved run manifest to {manifest_path}")

        fit_model(experiment_dict, resume_from=None, logger=logger)

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
