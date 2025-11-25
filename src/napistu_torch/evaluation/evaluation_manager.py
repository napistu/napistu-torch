"""Manager for organizing experiments' metadata, data, models, and evaluation results."""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from pydantic import ValidationError

if TYPE_CHECKING:  # for static analysis only
    from pytorch_lightning import LightningModule
else:
    LightningModule = object

from napistu_torch.configs import RunManifest
from napistu_torch.constants import (
    RUN_MANIFEST,
    RUN_MANIFEST_DEFAULTS,
)
from napistu_torch.lightning.constants import EXPERIMENT_DICT
from napistu_torch.napistu_data import NapistuData
from napistu_torch.napistu_data_store import NapistuDataStore
from napistu_torch.utils.optional import import_lightning, require_lightning

logger = logging.getLogger(__name__)


class EvaluationManager:
    """Manage the evaluation of an experiment."""

    def __init__(self, experiment_dir: Union[Path, str]):

        if isinstance(experiment_dir, str):
            experiment_dir = Path(experiment_dir).expanduser()
        elif not isinstance(experiment_dir, Path):
            raise TypeError(
                f"Experiment directory must be a Path or string, got {type(experiment_dir)}"
            )

        if not experiment_dir.exists():
            raise FileNotFoundError(
                f"Experiment directory {experiment_dir} does not exist"
            )
        self.experiment_dir = experiment_dir

        manifest_path = (
            experiment_dir / RUN_MANIFEST_DEFAULTS[RUN_MANIFEST.MANIFEST_FILENAME]
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

        self.experiment_dict = None
        self.napistu_data_store = None

    @require_lightning
    def get_experiment_dict(self) -> dict:
        from napistu_torch.lightning.workflows import (
            resume_experiment,  # import here to avoid circular import
        )

        if self.experiment_dict is None:
            self.experiment_dict = resume_experiment(self)

        return self.experiment_dict

    def get_store(self) -> NapistuDataStore:

        if self.napistu_data_store is None:
            self.napistu_data_store = NapistuDataStore(
                self.experiment_config.data.store_dir
            )

        return self.napistu_data_store

    def get_summary_string(self) -> str:
        """
        Generate a descriptive summary string from experiment metadata.

        Examples:
        - "model: sage-octopus-baseline (WandB: abc123)"
        - "model: transe-256-hidden"

        Returns
        -------
        str
            Formatted commit message string
        """
        parts = []

        # Experiment name
        if self.manifest.experiment_name:
            parts.append(f"model: {self.manifest.experiment_name}")
        else:
            parts.append("Napistu-Torch model")

        # Model architecture info
        arch_info = self.experiment_config.model.get_architecture_string()
        parts.append(f"({arch_info})")

        # WandB run ID
        if self.manifest.wandb_run_id:
            parts.append(f"WandB: {self.manifest.wandb_run_id}")

        return " | ".join(parts)

    def get_run_summary(self) -> dict:
        from wandb import Api  # optional dependency

        api = Api()

        run_path = f"{self.wandb_entity}/{self.wandb_project}/{self.wandb_run_id}"
        run = api.run(run_path)

        # Extract summary metrics
        summary = run.summary._json_dict
        return summary

    @require_lightning
    def load_model_from_checkpoint(
        self, checkpoint_path: Optional[Path] = None
    ) -> LightningModule:
        import_lightning()

        if checkpoint_path is None:
            checkpoint_path = self.best_checkpoint_path
            if checkpoint_path is None:
                raise ValueError(
                    "No checkpoint path provided and no best checkpoint found"
                )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint file not found at path: {checkpoint_path}"
            )

        experiment_dict = self.get_experiment_dict()

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model = experiment_dict[EXPERIMENT_DICT.MODEL]
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return model

    def load_napistu_data(self, napistu_data_name: Optional[str] = None) -> NapistuData:
        if napistu_data_name is None:
            napistu_data_name = self.experiment_config.data.napistu_data_name
        napistu_data_store = self.get_store()
        return napistu_data_store.load_napistu_data(napistu_data_name)

    def publish_to_huggingface(
        self,
        repo_id: str,
        checkpoint_path: Optional[Path] = None,
        commit_message: Optional[str] = None,
        overwrite: bool = False,
        token: Optional[str] = None,
    ) -> str:
        """
        Publish this experiment's model to HuggingFace Hub.

        Creates a private repository if it doesn't exist. Repositories can be
        made public manually on huggingface.co after curation.

        Parameters
        ----------
        repo_id : str
            Repository ID in format "username/repo-name"
        checkpoint_path : Optional[Path]
            Checkpoint to publish. If None, uses best checkpoint.
        commit_message : Optional[str]
            Custom commit message (default: auto-generated)
        overwrite : bool
            Explicitly confirm overwriting existing model (default: False)
        token : Optional[str]
            HuggingFace API token (default: uses `huggingface-cli login` token)

        Returns
        -------
        str
            URL to the published model on HuggingFace Hub

        Examples
        --------
        >>> manager = EvaluationManager("experiments/my_run")
        >>> # First upload
        >>> url = manager.publish_to_huggingface("shackett/napistu-sage-octopus")
        >>> # Update same repo
        >>> url = manager.publish_to_huggingface("shackett/napistu-sage-octopus", overwrite=True)
        """
        from napistu_torch.ml.hugging_face import HuggingFacePublisher

        # Use best checkpoint if not specified
        if checkpoint_path is None:
            checkpoint_path = self.best_checkpoint_path
            if checkpoint_path is None:
                raise ValueError(
                    "No checkpoint path provided and no best checkpoint found. "
                    "Specify checkpoint_path explicitly."
                )

        if commit_message is None:
            commit_message = self.get_summary_string()

        # Initialize publisher
        publisher = HuggingFacePublisher(token=token)

        # Publish
        return publisher.publish_model(
            repo_id=repo_id,
            checkpoint_path=checkpoint_path,
            manifest=self.manifest,
            commit_message=commit_message,
            overwrite=overwrite,
        )


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
