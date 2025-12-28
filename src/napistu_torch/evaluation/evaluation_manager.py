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
from napistu_torch.ml.constants import METRIC_SUMMARIES
from napistu_torch.napistu_data import NapistuData
from napistu_torch.napistu_data_store import NapistuDataStore
from napistu_torch.utils.optional import import_lightning, require_lightning

logger = logging.getLogger(__name__)


class EvaluationManager:
    """
    Manager for post-training evaluation of a model.

    This class provides a unified interface for accessing experiment artifacts,
    loading models from checkpoints, publishing to HuggingFace Hub, and managing
    experiment metadata. It loads the experiment manifest and provides convenient
    access to checkpoints, WandB information, and data stores.

    Parameters
    ----------
    experiment_dir : Union[Path, str]
        Path to the experiment directory containing the manifest file and checkpoints.
        Must contain a `run_manifest.yaml` file.

    Attributes
    ----------
    experiment_dir : Path
        Path to the experiment directory
    manifest : RunManifest
        The experiment manifest containing metadata and configuration
    experiment_name : Optional[str]
        Name of the experiment from the manifest
    wandb_run_id : Optional[str]
        WandB run ID from the manifest
    wandb_run_url : Optional[str]
        WandB run URL from the manifest
    wandb_project : Optional[str]
        WandB project name from the manifest
    wandb_entity : Optional[str]
        WandB entity (username/team) from the manifest
    experiment_config : ExperimentConfig
        The experiment configuration from the manifest
    checkpoint_dir : Path
        Directory containing model checkpoints
    best_checkpoint_path : Optional[Path]
        Path to the best checkpoint (highest validation AUC)
    best_checkpoint_val_auc : Optional[float]
        Validation AUC of the best checkpoint

    Public Methods
    --------------
    get_experiment_dict()
        Get the experiment dictionary with model, data module, trainer, etc.
    get_store()
        Get the NapistuDataStore for this experiment's data
    get_summary_string()
        Generate a descriptive summary string from experiment metadata
    get_run_summary()
        Get summary metrics from WandB for this experiment
    load_model_from_checkpoint(checkpoint_name=None)
        Load a trained model from a checkpoint file
    load_napistu_data(napistu_data_name=None)
        Load the NapistuData object used for this experiment
    publish_to_huggingface(repo_id, checkpoint_path=None, commit_message=None, overwrite=False, token=None)
        Publish this experiment's model to HuggingFace Hub

    Private Methods
    ---------------
    _resolve_checkpoint_path(checkpoint_name=None)
        Resolve a checkpoint name or path to an actual checkpoint file path

    Examples
    --------
    >>> # Load an experiment
    >>> manager = EvaluationManager("experiments/my_run")
    >>>
    >>> # Load the model from best checkpoint
    >>> model = manager.load_model_from_checkpoint()
    >>>
    >>> # Load from specific checkpoint
    >>> model = manager.load_model_from_checkpoint("last")
    >>> model = manager.load_model_from_checkpoint("best-epoch=50-val_auc=0.85.ckpt")
    >>>
    >>> # Get experiment summary
    >>> summary = manager.get_summary_string()
    >>> print(summary)  # "model: sage-octopus-baseline (sage-dot_product_h128_l3) | WandB: abc123"
    >>>
    >>> # Publish to HuggingFace
    >>> url = manager.publish_to_huggingface("username/model-name")
    """

    def __init__(self, experiment_dir: Union[Path, str]):
        """
        Initialize EvaluationManager from an experiment directory.

        Parameters
        ----------
        experiment_dir : Union[Path, str]
            Path to experiment directory containing manifest and checkpoints.
            Must contain a `run_manifest.yaml` file.

        Raises
        ------
        FileNotFoundError
            If experiment directory or manifest file doesn't exist
        ValueError
            If manifest file is invalid or cannot be parsed
        """

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
        """
        Get the experiment dictionary with all experiment components.

        The experiment dictionary contains the model, data module, trainer,
        run manifest, and WandB logger. This is lazily loaded and cached.

        Returns
        -------
        dict
            Experiment dictionary containing:
            - data_module : Union[FullGraphDataModule, EdgeBatchDataModule]
            - model : pl.LightningModule (e.g., EdgePredictionLightning)
            - trainer : NapistuTrainer
            - run_manifest : RunManifest
            - wandb_logger : Optional[WandbLogger]

        Examples
        --------
        >>> manager = EvaluationManager("experiments/my_run")
        >>> experiment_dict = manager.get_experiment_dict()
        >>> model = experiment_dict[EXPERIMENT_DICT.MODEL]
        """
        from napistu_torch.lightning.workflows import (
            resume_experiment,  # import here to avoid circular import
        )

        if self.experiment_dict is None:
            self.experiment_dict = resume_experiment(self)

        return self.experiment_dict

    def get_store(self) -> NapistuDataStore:
        """
        Get the NapistuDataStore for this experiment's data.

        The data store is lazily loaded and cached. It provides access to
        NapistuData objects, vertex tensors, and pandas DataFrames stored
        for this experiment.

        Returns
        -------
        NapistuDataStore
            The data store instance for this experiment

        Examples
        --------
        >>> manager = EvaluationManager("experiments/my_run")
        >>> store = manager.get_store()
        >>> napistu_data = store.load_napistu_data("edge_prediction")
        """

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
        """
        Get summary metrics from WandB for this experiment.

        Retrieves the summary metrics (final values) from the WandB run
        associated with this experiment.

        Returns
        -------
        dict
            Dictionary containing summary metrics from WandB (e.g., final
            validation AUC, training loss, etc.)

        Raises
        ------
        ValueError
            If WandB run ID is not available
        RuntimeError
            If WandB API access fails

        Examples
        --------
        >>> manager = EvaluationManager("experiments/my_run")
        >>> summary = manager.get_run_summary()
        >>> print(summary["val_auc"])  # Final validation AUC
        """
        return self.manifest.get_run_summary()

    @require_lightning
    def load_model_from_checkpoint(
        self, checkpoint_name: Optional[Union[Path, str]] = None
    ) -> LightningModule:
        """
        Load a trained model from a checkpoint file.

        The checkpoint name can be:
        - None: Uses the best checkpoint (highest validation AUC)
        - A string matching a checkpoint filename in checkpoint_dir (e.g., "last.ckpt", "best-epoch=50-val_auc=0.85.ckpt")
        - The string "last": Resolves to "last.ckpt" in checkpoint_dir
        - A Path object or string path to a checkpoint file

        Parameters
        ----------
        checkpoint_name : Optional[Union[Path, str]], default=None
            Checkpoint name or path. If None, uses best checkpoint.
            If a string, first checks if it matches a file in checkpoint_dir,
            otherwise treats it as a file path.

        Returns
        -------
        LightningModule
            The loaded model in evaluation mode

        Raises
        ------
        ValueError
            If no checkpoint is found and none is provided
        FileNotFoundError
            If the specified checkpoint file doesn't exist

        Examples
        --------
        >>> manager = EvaluationManager("experiments/my_run")
        >>>
        >>> # Load from best checkpoint
        >>> model = manager.load_model_from_checkpoint()
        >>>
        >>> # Load from last checkpoint
        >>> model = manager.load_model_from_checkpoint("last")
        >>>
        >>> # Load from specific checkpoint by name
        >>> model = manager.load_model_from_checkpoint("best-epoch=50-val_auc=0.85.ckpt")
        >>>
        >>> # Load from absolute path
        >>> model = manager.load_model_from_checkpoint("/path/to/checkpoint.ckpt")
        """
        import_lightning()

        checkpoint_path = self._resolve_checkpoint_path(checkpoint_name)

        experiment_dict = self.get_experiment_dict()

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model = experiment_dict[EXPERIMENT_DICT.MODEL]
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return model

    def load_napistu_data(self, napistu_data_name: Optional[str] = None) -> NapistuData:
        """
        Load the NapistuData object used for this experiment.

        Loads the NapistuData object from the experiment's data store.
        If no name is provided, uses the name from the experiment configuration.

        Parameters
        ----------
        napistu_data_name : Optional[str], default=None
            Name of the NapistuData object to load. If None, uses the name
            from the experiment configuration.

        Returns
        -------
        NapistuData
            The loaded NapistuData object

        Examples
        --------
        >>> manager = EvaluationManager("experiments/my_run")
        >>> # Load using name from config
        >>> data = manager.load_napistu_data()
        >>> # Load specific artifact
        >>> data = manager.load_napistu_data("edge_prediction")
        """
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
        tag: Optional[str] = None,
        tag_message: Optional[str] = None,
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
        tag : Optional[str]
            Tag name to create after all assets are uploaded (e.g., "v1.0")
        tag_message : Optional[str]
            Optional message for the tag

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
        from napistu_torch.ml.hugging_face import HFModelPublisher

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
        publisher = HFModelPublisher(token=token)

        # Publish
        return publisher.publish_model(
            repo_id=repo_id,
            checkpoint_path=checkpoint_path,
            manifest=self.manifest,
            commit_message=commit_message,
            overwrite=overwrite,
            tag=tag,
            tag_message=tag_message,
        )

    # private methods

    def _resolve_checkpoint_path(
        self, checkpoint_name: Optional[Union[Path, str]] = None
    ) -> Path:
        """
        Resolve a checkpoint name or path to an actual checkpoint file path.

        Handles various input formats:
        - None: Uses the best checkpoint (highest validation AUC)
        - String matching a checkpoint filename in checkpoint_dir (e.g., "last.ckpt", "best-epoch=50-val_auc=0.85.ckpt")
        - The string "last": Resolves to "last.ckpt" in checkpoint_dir
        - A Path object or string path to a checkpoint file

        Parameters
        ----------
        checkpoint_name : Optional[Union[Path, str]], default=None
            Checkpoint name or path. If None, uses best checkpoint.
            If a string, first checks if it matches a file in checkpoint_dir,
            otherwise treats it as a file path.

        Returns
        -------
        Path
            Resolved path to the checkpoint file

        Raises
        ------
        ValueError
            If no checkpoint is found and none is provided
        FileNotFoundError
            If the specified checkpoint file doesn't exist
        """
        if checkpoint_name is None:
            checkpoint_path = self.best_checkpoint_path
            if checkpoint_path is None:
                raise ValueError(
                    "No checkpoint name provided and no best checkpoint found"
                )
            return checkpoint_path

        if isinstance(checkpoint_name, str):
            # First, check if the string matches a file in checkpoint_dir
            potential_path = self.checkpoint_dir / checkpoint_name
            if potential_path.is_file():
                return potential_path
            elif checkpoint_name == "last":
                # Special case: resolve "last" to last.ckpt
                checkpoint_path = self.checkpoint_dir / "last.ckpt"
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Last checkpoint not found at: {checkpoint_path}"
                    )
                return checkpoint_path
            else:
                # Treat as a path string
                checkpoint_path = Path(checkpoint_name)
        else:
            # Already a Path object
            checkpoint_path = checkpoint_name

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint file not found treating it as a named artifact in self.checkpoint_dir or as a path: {checkpoint_path}"
            )

        return checkpoint_path


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

    match = re.search(
        rf"epoch=(\d+)-{METRIC_SUMMARIES.VAL_AUC}=(0\.[\d]+)", filename_str
    )

    if not match:
        return None

    return int(match.group(1)), float(match.group(2))
