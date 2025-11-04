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


def prepare_config(
    config_path: Path,
    seed: Optional[int] = None,
    wandb_mode: Optional[str] = None,
    fast_dev_run: bool = False,
    overrides: tuple[str] = (),
    logger: logging.Logger = logging.getLogger(__name__),
) -> ExperimentConfig:
    """
    Prepare the configuration for training.

    Parameters
    ----------
    config_path: Path
        Path to the configuration file
    seed: Optional[int]
        An optional random seed to override the one in the loaded config file (including defaults)
    wandb_mode: Optional[str]
        W&B mode to use for the experiment to override the one in the loaded config file (including defaults)
    fast_dev_run: bool
        Whether to run a fast development run (1 batch per epoch)
    overrides: tuple[str]
        A tuple of strings in the format "key.subkey=value" to override the values in the loaded config file (including defaults)
    logger: logging.Logger
        The logger to use for logging

    Returns
    -------
    ExperimentConfig
        The prepared configuration
    """
    logger.info(f"Loading config from: {config_path}")
    config = ExperimentConfig.from_yaml(config_path)
    logger.info(f"Config loaded: {config.name or 'unnamed experiment'}")

    # Try to validate original config first (catch config file issues early)
    try:
        config.model_validate(config.model_dump())
        logger.debug("Original config validated successfully")
    except ValidationError as e:
        logger.error(f"Config file validation failed:\n{e}")
        raise click.Abort()

    # Apply explicit CLI flags (these take precedence over --set)
    if seed is not None:
        logger.info(f"Overriding seed: {config.seed} → {seed}")
        config.seed = seed

    if wandb_mode is not None:
        logger.info(f"Overriding W&B mode: {config.wandb.mode} → {wandb_mode}")
        config.wandb.mode = wandb_mode

    if fast_dev_run:
        logger.info("Fast dev run enabled (1 batch per epoch)")
        config.fast_dev_run = True

    # Apply --set overrides
    if overrides:
        config = _apply_config_overrides(config, overrides, logger)

    # Validate config after all overrides
    try:
        config.model_validate(config.model_dump())
        logger.info("Config validation passed")
    except ValidationError as e:
        logger.error(
            f"Config validation failed after applying overrides:\n{e}\n\n"
            f"Original config was valid, so the issue is with one of your overrides."
        )
        raise click.Abort()

    # Check that required data paths exist
    if not config.data.sbml_dfs_path.exists():
        logger.error(f"SBML_dfs file not found: {config.data.sbml_dfs_path}")
        raise click.Abort()

    if not config.data.napistu_graph_path.exists():
        logger.error(f"NapistuGraph file not found: {config.data.napistu_graph_path}")
        raise click.Abort()

    logger.info("All data paths validated")

    # Display key configuration
    logger.info("=" * 80)
    logger.info("Training Configuration:")
    logger.info(f"  Task: {config.task.task}")
    logger.info(
        f"  Model: {config.model.encoder} (hidden={config.model.hidden_channels}, layers={config.model.num_layers})"
    )
    logger.info(
        f"  Training: {config.training.epochs} epochs, lr={config.training.lr}, batches_per_epoch={config.training.batches_per_epoch}"
    )
    logger.info(f"  W&B: project={config.wandb.project}, mode={config.wandb.mode}")
    logger.info(f"  Seed: {config.seed}")

    return config


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
    config: ExperimentConfig, overrides: tuple[str], logger: logging.Logger
) -> ExperimentConfig:
    """
    Apply dot-notation overrides to a Pydantic config.

    Parameters
    ----------
    config : ExperimentConfig
        Base configuration object
    overrides : tuple[str]
        Override strings in format "key.subkey=value"
    logger : logging.Logger
        Logger for reporting changes

    Returns
    -------
    ExperimentConfig
        Config with overrides applied

    Examples
    --------
    >>> config = apply_config_overrides(
    ...     config,
    ...     ("wandb.mode=disabled", "training.epochs=100"),
    ...     logger
    ... )
    """
    if not overrides:
        return config

    logger.info("Applying config overrides:")

    for override in overrides:
        try:
            if "=" not in override:
                raise ValueError(
                    f"Override must be in format 'key=value', got: {override}"
                )

            key_path, value = override.split("=", 1)
            keys = key_path.split(".")

            # Navigate to parent object
            obj = config
            for key in keys[:-1]:
                if not hasattr(obj, key):
                    raise AttributeError(
                        f"Config has no attribute '{key}' in path '{key_path}'"
                    )
                obj = getattr(obj, key)

            # Get field info for type conversion
            field_name = keys[-1]
            if not hasattr(obj, field_name):
                raise AttributeError(f"Config has no attribute '{field_name}'")

            # Get the field type from Pydantic model
            field_info = obj.model_fields[field_name]
            field_type = field_info.annotation

            # Convert and set value
            old_value = getattr(obj, field_name)
            converted_value = _convert_value(value, field_type)
            setattr(obj, field_name, converted_value)

            logger.info(f"  {key_path}: {old_value} → {converted_value}")

        except Exception as e:
            raise click.BadParameter(
                f"Invalid override '{override}': {e}\n"
                f"Use format: --set key.subkey=value"
            )

    return config


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
