"""Manager for organizing experiments' metadata, data, models, and evaluation results."""

import logging
import re
from pathlib import Path
from typing import Tuple, Union

from pydantic import ValidationError

from napistu_torch.configs import RunManifest
from napistu_torch.constants import (
    RUN_MANIFEST,
    RUN_MANIFEST_DEFAULTS,
)

logger = logging.getLogger(__name__)


class EvaluationManager:
    """Manage the evaluation of an experiment."""

    def __init__(self, experiment_dir: Union[Path, str]):

        if isinstance(experiment_dir, str):
            experiment_dir = Path(experiment_dir)
        elif not isinstance(experiment_dir, Path):
            raise TypeError(
                f"Experiment directory must be a Path or string, got {type(experiment_dir)}"
            )

        if not experiment_dir.exists():
            raise FileNotFoundError(
                f"Experiment directory {experiment_dir} does not exist"
            )
        self.experiment_dir = experiment_dir

        # manifest_path = experiment_dir / RUN_MANIFEST_DEFAULTS[RUN_MANIFEST.MANIFEST_FILENAME]
        manifest_path = (
            experiment_dir
            / "logs"
            / RUN_MANIFEST_DEFAULTS[RUN_MANIFEST.MANIFEST_FILENAME]
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file {manifest_path} does not exist")
        try:
            self.manifest = RunManifest.from_yaml(manifest_path)
        except ValidationError as e:
            raise ValueError(f"Invalid manifest file {manifest_path}: {e}")

        # set attributes based on manifest
        self.experiment_name = self.manifest.experiment_name
        self.wandb_run_id = self.manifest.wandb_run_id
        self.wandb_run_url = self.manifest.wandb_run_url
        self.wandb_project = self.manifest.wandb_project
        self.wandb_entity = self.manifest.wandb_entity

        # Get ExperimentConfig from manifest (already reconstructed by RunManifest.from_yaml)
        self.experiment_config = self.manifest.experiment_config
        # Replace output_dir with experiment_dir so paths will appropriately resolve
        self.experiment_config.output_dir = experiment_dir

        # set checkpoint directory
        self.checkpoint_dir = self.experiment_config.training.get_checkpoint_dir(
            experiment_dir
        )
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory {self.checkpoint_dir} does not exist"
            )

        best_checkpoint = find_best_checkpoint(self.checkpoint_dir)
        if best_checkpoint is None:
            self.best_checkpoint_path, self.best_checkpoint_val_auc = None, None
        else:
            self.best_checkpoint_path, self.best_checkpoint_val_auc = best_checkpoint


# public functions


def find_best_checkpoint(checkpoint_dir: Path) -> Tuple[Path, float] | None:
    """Get the best checkpoint from a directory of checkpoints."""
    # Get all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))

    # If no checkpoints found, return None
    if not checkpoint_files:
        logger.warning(f"No checkpoints found in {checkpoint_dir}; returning None")
        return None

    # Sort checkpoints by validation loss (assumes loss is stored in filename)
    best_checkpoint = None
    for file in checkpoint_files:
        result = _parse_checkpoint_filename(file)
        if result is None:
            continue
        _, val_auc = result
        if best_checkpoint is None or val_auc > best_checkpoint[1]:
            best_checkpoint = (file, val_auc)

    if best_checkpoint is None:
        logger.warning(
            f"No valid checkpoints found in {checkpoint_dir}; returning None"
        )
        return None

    # Return the best checkpoint
    return best_checkpoint


# private functions


def _parse_checkpoint_filename(filename: str | Path) -> Tuple[int, float] | None:
    """
    Extract epoch number and validation AUC from checkpoint filename.

    Parameters
    ----------
    filename: str | Path
        Checkpoint filename like "best-epoch=120-val_auc=0.7604.ckpt"

    Returns
    -------
    epoch: int
        Epoch number
    val_auc: float
        Validation AUC

    Example:
        >>> parse_checkpoint_filename("best-epoch=120-val_auc=0.7604.ckpt")
        {'epoch': 120, 'val_auc': 0.7604}
    """
    # Convert Path to string and extract just the filename
    if isinstance(filename, Path):
        filename_str = filename.name
    else:
        filename_str = str(filename)

    match = re.search(r"epoch=(\d+)-val_auc=(0\.[\d]+)", filename_str)

    if not match:
        return None

    return int(match.group(1)), float(match.group(2))
