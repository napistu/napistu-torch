"""Shared CLI utilities for Napistu-Torch CLI"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

import click
from pydantic import ValidationError
from rich.console import Console
from rich.logging import RichHandler

import napistu_torch
from napistu_torch.configs import ExperimentConfig
from napistu_torch.constants import (
    PARAM_OVERRIDE_MAP,
    TRAIN_BOOLEAN_FLAGS,
    TRAIN_NAMED_ARGS,
)


def format_named_overrides(
    seed: Optional[int] = None,
    fast_dev_run: bool = False,
    encoder: Optional[str] = None,
    head: Optional[str] = None,
    hidden_channels: Optional[int] = None,
    dropout: Optional[float] = None,
    init_head_as_identity: bool = False,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    optimizer: Optional[str] = None,
    scheduler: Optional[str] = None,
    epochs: Optional[int] = None,
    wandb_group: Optional[str] = None,
    wandb_mode: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    """
    Convert named CLI parameters to config overrides and log messages.

    This utility converts direct CLI parameters (typically from WandB sweeps
    or manual CLI usage) into config overrides and corresponding log messages.

    Parameters
    ----------
    seed : Optional[int]
        Random seed
    fast_dev_run : bool
        Fast dev run flag
    encoder : Optional[str]
        Model encoder (e.g., sage, gcn, gat, graph_conv)
    head : Optional[str]
        Model head (e.g., node_classification, dot_product, mlp, attention, rotate, transe, distmult)
    hidden_channels : Optional[int]
        Model hidden channels
    dropout : Optional[float]
        Dropout rate
    init_head_as_identity : bool
        Whether to initialize the head to approximate an identity transformation
    lr : Optional[float]
        Learning rate
    weight_decay : Optional[float]
        Weight decay
    optimizer : Optional[str]
        Optimizer name (adam, adamw)
    scheduler : Optional[str]
        Learning rate scheduler (cosine, onecycle, plateau)
    epochs : Optional[int]
        Number of training epochs
    wandb_group : Optional[str]
        WandB group name
    wandb_mode : Optional[str]
        WandB mode (online, offline, disabled)

    Returns
    -------
    tuple[list[str], list[str]]
        - List of override strings in "key.subkey=value" format
        - List of log messages describing the overrides

    Examples
    --------
    >>> overrides, messages = format_named_overrides(
    ...     lr=0.001,
    ...     optimizer="adam",
    ...     seed=42
    ... )
    >>> overrides
    ['training.lr=0.001', 'training.optimizer=adam', 'seed=42']
    >>> messages
    ['CLI override: training.lr=0.001', 'CLI override: training.optimizer=adam', 'CLI override: seed=42']
    """
    overrides = []
    messages = []

    # Get all kwargs passed to the function
    kwargs = {
        TRAIN_NAMED_ARGS.SEED: seed,
        TRAIN_NAMED_ARGS.FAST_DEV_RUN: fast_dev_run,
        TRAIN_NAMED_ARGS.ENCODER: encoder,
        TRAIN_NAMED_ARGS.HEAD: head,
        TRAIN_NAMED_ARGS.HIDDEN_CHANNELS: hidden_channels,
        TRAIN_NAMED_ARGS.DROPOUT: dropout,
        TRAIN_NAMED_ARGS.INIT_HEAD_AS_IDENTITY: init_head_as_identity,
        TRAIN_NAMED_ARGS.LR: lr,
        TRAIN_NAMED_ARGS.WEIGHT_DECAY: weight_decay,
        TRAIN_NAMED_ARGS.OPTIMIZER: optimizer,
        TRAIN_NAMED_ARGS.SCHEDULER: scheduler,
        TRAIN_NAMED_ARGS.EPOCHS: epochs,
        TRAIN_NAMED_ARGS.WANDB_GROUP: wandb_group,
        TRAIN_NAMED_ARGS.WANDB_MODE: wandb_mode,
    }

    # Loop through the mapping and build overrides
    for param_name, override_path in PARAM_OVERRIDE_MAP.items():
        value = kwargs[param_name]

        # Skip None values
        if value is None:
            continue

        # Handle boolean flags - only add override if True
        if param_name in TRAIN_BOOLEAN_FLAGS and not value:
            continue

        # Format the override string
        override_str = f"{override_path}={value}"
        overrides.append(override_str)
        messages.append(f"CLI override: {override_str}")

    return overrides, messages


def log_deferred_messages(
    logger: logging.Logger,
    config_messages: list[str],
    config: ExperimentConfig,
    checkpoint_dir: Path,
    log_dir: Path,
    wandb_dir: Path,
) -> None:
    """
    Log deferred configuration messages and run information.

    Parameters
    ----------
    logger : logging.Logger
        Logger to use for output
    config_messages : list[str]
        Messages collected during config preparation
    config : ExperimentConfig
        The experiment configuration
    checkpoint_dir : Path
        Checkpoint directory path
    log_dir : Path
        Log directory path
    wandb_dir : Path
        WandB directory path
    resume : Optional[Path]
        Checkpoint path to resume from, if any
    """
    logger.info("=" * 80)
    logger.info("Napistu-Torch Training")
    logger.info("=" * 80)

    # Log config preparation messages
    for msg in config_messages:
        logger.info(f"  {msg}")

    logger.info("=" * 80)

    # Log directory information
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Checkpoints: {checkpoint_dir}")
    logger.info(f"  Logs: {log_dir}")
    logger.info(f"  WandB: {wandb_dir}")
    logger.info("=" * 80)


def prepare_config(
    config_path: Path,
    overrides: tuple[str] = (),
) -> tuple[ExperimentConfig, list[str]]:
    """
    Prepare the configuration for training.

    Parameters
    ----------
    config_path: Path
        Path to the configuration file
    overrides: tuple[str]
        A tuple of strings in the format "key.subkey=value" to override the values in the loaded config file (including defaults)

    Returns
    -------
    tuple[ExperimentConfig, list[str]]
        The prepared configuration and a list of log messages to emit after logger setup
    """
    messages = []

    messages.append(f"Loading config from: {config_path}")
    config = ExperimentConfig.from_yaml(config_path)
    messages.append(f"Config loaded: {config.name or 'unnamed experiment'}")

    # Try to validate original config first (catch config file issues early)
    try:
        config.model_validate(config.model_dump())
        messages.append("Original config validated successfully")
    except ValidationError as e:
        # Validation errors need to be raised immediately (can't defer these)
        print(f"ERROR: Config file validation failed:\n{e}")
        raise click.Abort()

    # Apply --set overrides (now includes named CLI params too)
    if overrides:
        config, override_messages = _apply_config_overrides(config, overrides)
        messages.extend(override_messages)

    # Validate config after all overrides
    try:
        config.model_validate(config.model_dump())
        messages.append("Config validation passed")
    except ValidationError as e:
        # Validation errors need to be raised immediately (can't defer these)
        print(
            f"ERROR: Config validation failed after applying overrides:\n{e}\n\n"
            f"Original config was valid, so the issue is with one of your overrides."
        )
        raise click.Abort()

    # Check that required data paths exist
    if not config.data.sbml_dfs_path.exists():
        print(f"ERROR: SBML_dfs file not found: {config.data.sbml_dfs_path}")
        raise click.Abort()

    if not config.data.napistu_graph_path.exists():
        print(f"ERROR: NapistuGraph file not found: {config.data.napistu_graph_path}")
        raise click.Abort()

    messages.append("All data paths validated")

    return config, messages


def setup_logging(
    log_dir: Path, verbosity: str = "INFO"
) -> tuple[logging.Logger, Console]:
    """
    Set up logging for training runs.

    Creates both console output (Rich) and file logging with timestamps.

    Parameters
    ----------
    log_dir : Path
        Directory to write log files to
    verbosity : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns
    -------
    tuple[logging.Logger, Console]
        Configured logger and Rich console
    """
    # Create log directory if needed
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    # Rich console for pretty output
    console = Console(width=120)

    # Console handler (what user sees in terminal)
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,  # Cleaner for console
        markup=True,
        log_time_format="[%m/%d %H:%M]",
    )
    console_handler.setLevel(getattr(logging, verbosity.upper()))

    # File handler (everything goes here at DEBUG level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Configure main logger
    logger = logging.getLogger(napistu_torch.__name__)
    logger.setLevel(logging.DEBUG)  # Capture everything
    logger.propagate = False
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Silence noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    logger.info(f"Logging to file: {log_file}")

    return logger, console


def verbosity_option(f: Callable) -> Callable:
    """
    Decorator that adds --verbosity option.

    This controls the console output level. File logs are always DEBUG.

    Note: This is a simplified version for napistu-torch that returns the verbosity
    value as a parameter. For napistu-py style verbosity that auto-configures logging,
    use napistu._cli.verbosity_option instead.
    """
    return click.option(
        "--verbosity",
        "-v",
        type=click.Choice(
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
        ),
        default="INFO",
        help="Console logging verbosity (file logs always DEBUG)",
    )(f)


# private utils


def _apply_config_overrides(
    config: ExperimentConfig, overrides: tuple[str]
) -> tuple[ExperimentConfig, list[str]]:
    """
    Apply --set style config overrides.

    Parameters
    ----------
    config : ExperimentConfig
        The config to modify
    overrides : tuple[str]
        Tuples of "key.subkey=value" strings

    Returns
    -------
    tuple[ExperimentConfig, list[str]]
        The modified config and a list of log messages

    Raises
    ------
    click.Abort
        If a parameter is specified multiple times in overrides
    """
    messages = []
    messages.append(f"Applying {len(overrides)} config override(s):")

    # Track which parameters have been overridden to detect conflicts
    overridden_params = set()

    for override in overrides:
        try:
            key, value = override.split("=", 1)

            # Check for conflicts
            if key in overridden_params:
                print(
                    f"ERROR: Parameter '{key}' specified multiple times in overrides.\n"
                    f"This can happen when using both --set and named CLI parameters (e.g., --lr).\n"
                    f"Please use only one method to set each parameter."
                )
                raise click.Abort()

            # Track this parameter
            overridden_params.add(key)

            messages.append(f"  {key} = {value}")

            # Parse the nested key path (e.g., "training.lr" -> ["training", "lr"])
            keys = key.split(".")

            # Navigate to the target and set the value
            target = config
            for k in keys[:-1]:
                target = getattr(target, k)

            # Convert value to appropriate type
            current_value = getattr(target, keys[-1])
            if isinstance(current_value, bool):
                converted_value = value.lower() in ("true", "1", "yes")
            elif isinstance(current_value, int):
                converted_value = int(value)
            elif isinstance(current_value, float):
                converted_value = float(value)
            elif isinstance(current_value, Path):
                converted_value = Path(value)
            else:
                # Handle Optional[str] fields: convert "none" or empty string to None
                # This is needed for WandB sweeps compatibility where we need to explicitly
                # set scheduler=None to override a config that has a scheduler
                if (
                    current_value is None or isinstance(current_value, str)
                ) and value.lower() in ("none", ""):
                    converted_value = None
                else:
                    converted_value = value

            setattr(target, keys[-1], converted_value)

        except (ValueError, AttributeError) as e:
            print(f"ERROR: Failed to apply override '{override}': {e}")
            raise click.Abort()

    return config, messages


def _convert_value(value_str: str, field_type: Any) -> Any:
    """
    Convert string value to appropriate type based on Pydantic field type.

    Parameters
    ----------
    value_str : str
        String value from CLI
    field_type : Any
        Expected type from Pydantic model

    Returns
    -------
    Any
        Converted value

    Raises
    ------
    ValueError
        If conversion fails
    """
    # Handle Optional types
    if hasattr(field_type, "__origin__"):
        if field_type.__origin__ is Union:
            # Get non-None type from Optional
            types = [t for t in field_type.__args__ if t is not type(None)]
            if types:
                field_type = types[0]

    # Convert based on type
    if field_type is bool:
        return value_str.lower() in ("true", "1", "yes", "on")
    elif field_type is int:
        return int(value_str)
    elif field_type is float:
        return float(value_str)
    elif field_type is Path:
        return Path(value_str)
    elif hasattr(field_type, "__origin__") and field_type.__origin__ is list:
        # Simple list parsing: "a,b,c" -> ["a", "b", "c"]
        return value_str.split(",")
    else:
        # String or custom type
        return value_str
