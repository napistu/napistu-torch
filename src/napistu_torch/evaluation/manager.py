"""Manager for organizing experiments' metadata, data, models, and evaluation results."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import torch
from pydantic import ValidationError

if TYPE_CHECKING:  # for static analysis only
    from pytorch_lightning import LightningModule

    from napistu_torch.ml.hugging_face import HFModelLoader
else:
    LightningModule = object

from napistu_torch.configs import ExperimentConfig, RunManifest
from napistu_torch.constants import (
    NAPISTU_DATA_STORE_STRUCTURE,
    RUN_MANIFEST,
    RUN_MANIFEST_DEFAULTS,
)
from napistu_torch.lightning.constants import EXPERIMENT_DICT
from napistu_torch.napistu_data import NapistuData
from napistu_torch.napistu_data_store import NapistuDataStore
from napistu_torch.utils.optional import require_lightning

logger = logging.getLogger(__name__)


class EvaluationManager(ABC):
    """
    Base class for evaluation managers.

    Provides a unified interface for accessing experiment artifacts, loading models,
    and managing experiment metadata. Both local and remote (HuggingFace) evaluation
    managers share this common interface built around the RunManifest abstraction.

    Attributes
    ----------
    manifest : RunManifest
        The experiment manifest containing metadata and configuration
    napistu_data_store : Optional[NapistuDataStore]
        Data store for accessing NapistuData objects and other artifacts
    experiment_dict : Optional[dict]
        Cached experiment dictionary with model, data module, trainer, etc.

    Properties (derived from manifest)
    ----------------------------------
    experiment_config : ExperimentConfig
        The experiment configuration
    experiment_name : Optional[str]
        Name of the experiment
    wandb_run_id : Optional[str]
        WandB run ID
    wandb_run_url : Optional[str]
        WandB run URL
    wandb_project : Optional[str]
        WandB project name
    wandb_entity : Optional[str]
        WandB entity (username/team)

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
        Load a trained model from a checkpoint file (abstract, subclass-specific)
    load_napistu_data(napistu_data_name=None)
        Load the NapistuData object used for this experiment
    """

    # Core attributes that all subclasses must provide
    manifest: RunManifest
    napistu_data_store: Optional[NapistuDataStore]
    experiment_dict: Optional[dict]

    # Properties derived from manifest for convenience
    @property
    def experiment_config(self) -> ExperimentConfig:
        """Get the experiment configuration from the manifest."""
        return self.manifest.experiment_config

    @property
    def experiment_name(self) -> Optional[str]:
        """Get the experiment name from the manifest."""
        return self.manifest.experiment_name

    @property
    def wandb_run_id(self) -> Optional[str]:
        """Get the WandB run ID from the manifest."""
        return self.manifest.wandb_run_id

    @property
    def wandb_run_url(self) -> Optional[str]:
        """Get the WandB run URL from the manifest."""
        return self.manifest.wandb_run_url

    @property
    def wandb_project(self) -> Optional[str]:
        """Get the WandB project name from the manifest."""
        return self.manifest.wandb_project

    @property
    def wandb_entity(self) -> Optional[str]:
        """Get the WandB entity from the manifest."""
        return self.manifest.wandb_entity

    # Concrete methods (work identically for both Local and Remote)

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
        >>> manager = LocalEvaluationManager("experiments/my_run")
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

        The data store is lazily loaded and cached. For LocalEvaluationManager,
        it loads from experiment_config.data.store_dir on first access. For
        RemoteEvaluationManager, it's already loaded during initialization.

        Returns
        -------
        NapistuDataStore
            The data store instance for this experiment

        Raises
        ------
        ValueError
            If no data store is available

        Examples
        --------
        >>> manager = LocalEvaluationManager("experiments/my_run")
        >>> store = manager.get_store()
        >>> napistu_data = store.load_napistu_data("edge_prediction")
        >>>
        >>> manager = RemoteEvaluationManager.from_huggingface(
        ...     repo_id="shackett/sage-octopus",
        ...     data_store_dir=Path("./store")
        ... )
        >>> store = manager.get_store()
        """
        if self.napistu_data_store is None:
            # Lazy load from config (LocalEvaluationManager path)
            store_dir = self.experiment_config.data.store_dir
            if store_dir is None:
                raise ValueError(
                    "No data store available: experiment_config.data.store_dir is None"
                )
            logger.info(f"Lazy loading data store from {store_dir}")
            self.napistu_data_store = NapistuDataStore(store_dir)

        return self.napistu_data_store

    def get_summary_string(self) -> str:
        """
        Generate a descriptive summary string from experiment metadata.

        Examples:
        - "model: sage-octopus-baseline (sage-dot_product_h128_l3) | WandB: abc123"
        - "model: transe-256-hidden"

        Returns
        -------
        str
            Formatted summary string
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
        >>> manager = LocalEvaluationManager("experiments/my_run")
        >>> summary = manager.get_run_summary()
        >>> print(summary["val_auc"])  # Final validation AUC
        """
        return self.manifest.get_run_summary()

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
        >>> manager = LocalEvaluationManager("experiments/my_run")
        >>> # Load using name from config
        >>> data = manager.load_napistu_data()
        >>> # Load specific artifact
        >>> data = manager.load_napistu_data("edge_prediction")
        """
        if napistu_data_name is None:
            napistu_data_name = self.experiment_config.data.napistu_data_name
        napistu_data_store = self.get_store()
        return napistu_data_store.load_napistu_data(napistu_data_name)

    # Abstract methods (must be implemented by subclasses)

    @abstractmethod
    def load_model_from_checkpoint(
        self, checkpoint_name: Optional[Union[Path, str]] = None
    ) -> LightningModule:
        """
        Load a trained model from a checkpoint file.

        This method is implemented differently for local and remote managers:
        - LocalEvaluationManager: discovers and loads from checkpoint directory
        - RemoteEvaluationManager: loads the single published checkpoint from HuggingFace

        Parameters
        ----------
        checkpoint_name : Optional[Union[Path, str]], default=None
            Checkpoint identifier (interpretation depends on subclass)

        Returns
        -------
        LightningModule
            The loaded model in evaluation mode

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by subclasses
        """
        raise NotImplementedError(
            "load_model_from_checkpoint must be implemented by subclass"
        )


class LocalEvaluationManager(EvaluationManager):
    """
    Manager for post-training evaluation of a locally-stored model.

    This class provides a unified interface for accessing experiment artifacts,
    loading models from checkpoints, publishing to HuggingFace Hub, and managing
    experiment metadata. It loads the experiment manifest from a local directory
    and provides convenient access to checkpoints, WandB information, and data stores.

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
    checkpoint_dir : Path
        Directory containing model checkpoints
    best_checkpoint_path : Optional[Path]
        Path to the best checkpoint (highest validation AUC)
    best_checkpoint_val_auc : Optional[float]
        Validation AUC of the best checkpoint
    napistu_data_store : Optional[NapistuDataStore]
        The data store instance (lazily loaded)
    experiment_dict : Optional[dict]
        Cached experiment dictionary (lazily loaded)

    Public Methods
    --------------
    load_model_from_checkpoint(checkpoint_name=None)
        Load a trained model from a checkpoint file
    publish_to_huggingface(repo_id, checkpoint_path=None, commit_message=None, overwrite=False, token=None)
        Publish this experiment's model to HuggingFace Hub

    Private Methods
    ---------------
    _resolve_checkpoint_path(checkpoint_name=None)
        Resolve a checkpoint name or path to an actual checkpoint file path

    Examples
    --------
    >>> # Load an experiment
    >>> manager = LocalEvaluationManager("experiments/my_run")
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
        Initialize LocalEvaluationManager from an experiment directory.

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

        # Replace output_dir with experiment_dir so paths will appropriately resolve
        self.manifest.experiment_config.output_dir = experiment_dir

        # Set checkpoint directory
        self.checkpoint_dir = (
            self.manifest.experiment_config.training.get_checkpoint_dir(experiment_dir)
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

        # Initialize base class attributes
        self.experiment_dict = None
        self.napistu_data_store = None

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
        >>> manager = LocalEvaluationManager("experiments/my_run")
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
        from napistu_torch.utils.optional import import_lightning

        import_lightning()

        checkpoint_path = self._resolve_checkpoint_path(checkpoint_name)

        experiment_dict = self.get_experiment_dict()

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model = experiment_dict[EXPERIMENT_DICT.MODEL]
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return model

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
        >>> manager = LocalEvaluationManager("experiments/my_run")
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

    # Private methods

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


class RemoteEvaluationManager(EvaluationManager):
    """
    Manager for evaluation of models loaded from HuggingFace Hub.

    This class provides evaluation capabilities for models published to HuggingFace,
    loading the model checkpoint, configuration, and optionally data from remote
    repositories. It shares the same interface as LocalEvaluationManager but with
    remote-specific implementation details.

    Parameters
    ----------
    repo_id : str
        HuggingFace model repository ID (e.g., "username/model-name")
    model_loader : HFModelLoader
        Loader instance with downloaded model artifacts
    data_store : Optional[NapistuDataStore]
        Data store for accessing NapistuData objects (optional)

    Attributes
    ----------
    repo_id : str
        HuggingFace repository ID
    revision : str
        Git revision (branch, tag, or commit) used for loading
    manifest : RunManifest
        Reconstructed run manifest from HuggingFace artifacts
    checkpoint : Checkpoint
        Loaded model checkpoint
    checkpoint_path : Path
        Path to cached checkpoint file
    napistu_data_store : Optional[NapistuDataStore]
        The data store instance (may be None)
    experiment_dict : Optional[dict]
        Cached experiment dictionary (lazily loaded)

    Public Methods
    --------------
    from_huggingface(repo_id, revision=None, data_repo_id=None, data_store_dir=None, ...)
        Load a model and optionally data from HuggingFace Hub
    load_model_from_checkpoint(checkpoint_name=None)
        Load the published model checkpoint

    Properties (Remote-specific, raise AttributeError)
    --------------------------------------------------
    experiment_dir
        Not available for remote models
    checkpoint_dir
        Not available for remote models
    best_checkpoint_path
        Not available for remote models (only one checkpoint exists)

    Examples
    --------
    >>> # Load model only
    >>> manager = RemoteEvaluationManager.from_huggingface(
    ...     repo_id="shackett/sage-octopus"
    ... )
    >>>
    >>> # Load model with data
    >>> manager = RemoteEvaluationManager.from_huggingface(
    ...     repo_id="shackett/sage-octopus",
    ...     data_repo_id="shackett/octopus-consensus-v1"
    ... )
    >>>
    >>> # Load model and use it
    >>> model = manager.load_model_from_checkpoint()
    >>> summary = manager.get_summary_string()
    """

    def __init__(
        self,
        repo_id: str,
        model_loader: HFModelLoader,
        data_store: Optional[NapistuDataStore] = None,
    ):
        """
        Initialize RemoteEvaluationManager from HuggingFace artifacts.

        Parameters
        ----------
        repo_id : str
            HuggingFace repository ID
        model_loader : HFModelLoader
            Loader instance with downloaded model artifacts
        data_store : Optional[NapistuDataStore]
            Data store for accessing NapistuData objects
        """
        self.repo_id = repo_id
        self.revision = model_loader.revision
        self._model_loader = model_loader

        # Load checkpoint
        self.checkpoint = model_loader.load_checkpoint()
        self.checkpoint_path = model_loader._checkpoint_path

        # Reconstruct manifest from HuggingFace artifacts
        self.manifest = RunManifest.from_huggingface(model_loader, repo_id)

        # Initialize base class attributes
        self.napistu_data_store = data_store
        self.experiment_dict = None

    def get_run_summary(self, from_wandb: bool = False) -> dict:
        """
        Get summary metrics from HuggingFace (default) or WandB for this experiment.

        By default, retrieves the summary metrics from the HuggingFace repository
        by loading the wandb_run_info.yaml file. This avoids needing WandB API access
        for remote models. Optionally, can load directly from WandB if from_wandb=True.

        Parameters
        ----------
        from_wandb : bool, optional
            If True, load summary from WandB API instead of HuggingFace.
            Default is False (load from HuggingFace).

        Returns
        -------
        dict
            Dictionary containing summary metrics (e.g., final validation AUC,
            training loss, etc.)

        Raises
        ------
        RuntimeError
            If HuggingFace API access fails or run info is not available
        ValueError
            If from_wandb=True but WandB run ID is not available
        RuntimeError
            If WandB API access fails when from_wandb=True

        Examples
        --------
        >>> manager = RemoteEvaluationManager.from_huggingface(
        ...     repo_id="shackett/sage-octopus",
        ...     data_store_dir=Path("./store")
        ... )
        >>> # Load from HuggingFace (default)
        >>> summary = manager.get_run_summary()
        >>> print(summary["val_auc"])  # Final validation AUC
        >>>
        >>> # Load from WandB instead
        >>> summary = manager.get_run_summary(from_wandb=True)
        """
        if from_wandb:
            # Load from WandB API (same as base class implementation)
            return self.manifest.get_run_summary()
        else:
            # Load from HuggingFace (default)
            try:
                run_info = self._model_loader.load_run_info()
                return run_info.run_summaries
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load run summary from HuggingFace repository '{self.repo_id}': {e}"
                ) from e

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str,
        data_store_dir: Path,
        revision: Optional[str] = None,
        data_repo_id: Optional[str] = None,
        data_revision: Optional[str] = None,
        model_cache_dir: Optional[Path] = None,
        token: Optional[str] = None,
    ) -> "RemoteEvaluationManager":
        """
        Load a model and data from HuggingFace Hub.

        Parameters
        ----------
        repo_id : str
            Model repository ID (e.g., "shackett/sage-octopus")
        data_store_dir : Path
            Directory for the data store. If it exists, will be loaded as-is.
            If it doesn't exist, will be downloaded from HuggingFace using
            data_repo_id (or config.data.hf_repo_id if not provided).
        revision : Optional[str]
            Model revision (branch/tag/commit). Default: "main"
        data_repo_id : Optional[str]
            Data repository ID. If None and data_store_dir doesn't exist,
            will try to use config.data.hf_repo_id.
        data_revision : Optional[str]
            Data revision (branch/tag/commit). Default: uses config.data.hf_revision
            or "main" if not specified.
        model_cache_dir : Optional[Path]
            Where to cache model files. Default: HF default cache
        token : Optional[str]
            HuggingFace token for private repos

        Returns
        -------
        RemoteEvaluationManager
            Manager instance with model and data loaded

        Raises
        ------
        ValueError
            If data_store_dir doesn't exist and no HF repo info is available

        Examples
        --------
        >>> # Use existing local data store
        >>> manager = RemoteEvaluationManager.from_huggingface(
        ...     repo_id="shackett/sage-octopus",
        ...     data_store_dir=Path("./existing_store")
        ... )
        >>>
        >>> # Download data from HF to new location
        >>> manager = RemoteEvaluationManager.from_huggingface(
        ...     repo_id="shackett/sage-octopus",
        ...     data_store_dir=Path("./new_store"),
        ...     data_repo_id="shackett/octopus-consensus-v1"
        ... )
        >>>
        >>> # Let config determine data repo (if available)
        >>> manager = RemoteEvaluationManager.from_huggingface(
        ...     repo_id="shackett/sage-octopus",
        ...     data_store_dir=Path("./new_store")
        ... )
        >>>
        >>> # Pinned versions
        >>> manager = RemoteEvaluationManager.from_huggingface(
        ...     repo_id="shackett/sage-octopus",
        ...     revision="v1.0",
        ...     data_store_dir=Path("./store"),
        ...     data_repo_id="shackett/octopus-consensus-v1",
        ...     data_revision="v1.0"
        ... )
        """
        from napistu_torch.ml.hugging_face import HFModelLoader

        # Load model
        model_loader = HFModelLoader(
            repo_id=repo_id,
            revision=revision,
            cache_dir=model_cache_dir,
            token=token,
        )

        # Load experiment config to get data repo info if needed
        experiment_config = model_loader.load_config()

        # Resolve data store (handles existing vs. download from HF)
        data_store = _resolve_data_store_for_remote(
            data_store_dir=data_store_dir,
            experiment_config=experiment_config,
            data_repo_id=data_repo_id,
            data_revision=data_revision,
            token=token,
        )

        return cls(
            repo_id=repo_id,
            model_loader=model_loader,
            data_store=data_store,
        )

    @require_lightning
    def load_model_from_checkpoint(
        self, checkpoint_name: Optional[Union[Path, str]] = None
    ) -> LightningModule:
        """
        Load the published model checkpoint from HuggingFace Hub.

        RemoteEvaluationManager only contains the single published checkpoint,
        so checkpoint_name should not be provided.

        Parameters
        ----------
        checkpoint_name : Optional[Union[Path, str]], default=None
            Must be None. Remote models only have one checkpoint.

        Returns
        -------
        LightningModule
            The loaded model in evaluation mode

        Raises
        ------
        ValueError
            If checkpoint_name is provided (not supported for remote models)

        Examples
        --------
        >>> manager = RemoteEvaluationManager.from_huggingface("shackett/sage-octopus")
        >>> model = manager.load_model_from_checkpoint()
        """
        from napistu_torch.utils.optional import import_lightning

        import_lightning()

        if checkpoint_name is not None:
            raise ValueError(
                "RemoteEvaluationManager only contains the published checkpoint. "
                "Call load_model_from_checkpoint() without arguments to load it. "
                f"Attempted to load: {checkpoint_name}"
            )

        experiment_dict = self.get_experiment_dict()

        # Load state dict from the checkpoint we already have
        checkpoint_dict = torch.load(self.checkpoint_path, weights_only=False)
        model = experiment_dict[EXPERIMENT_DICT.MODEL]
        model.load_state_dict(checkpoint_dict["state_dict"])
        model.eval()

        return model

    # Properties that don't exist for remote models

    @property
    def experiment_dir(self) -> Path:
        """Not available for remote models."""
        raise AttributeError(
            "RemoteEvaluationManager does not have an experiment_dir. "
            "The model is loaded from HuggingFace Hub. "
            f"Model repository: https://huggingface.co/{self.repo_id}"
        )

    @property
    def checkpoint_dir(self) -> Path:
        """Not available for remote models."""
        raise AttributeError(
            "RemoteEvaluationManager does not have a checkpoint_dir. "
            "Only the published checkpoint is available. "
            "Use load_model_from_checkpoint() to load it."
        )

    @property
    def best_checkpoint_path(self) -> Optional[Path]:
        """Not available for remote models."""
        raise AttributeError(
            "RemoteEvaluationManager does not track multiple checkpoints. "
            "Only the published checkpoint is available. "
            "Access it via: manager.checkpoint_path"
        )

    @property
    def best_checkpoint_val_auc(self) -> Optional[float]:
        """Not available for remote models."""
        raise AttributeError(
            "RemoteEvaluationManager does not track checkpoint metrics. "
            "Use get_run_summary() to access WandB metrics instead."
        )

    @property
    def repo_url(self) -> str:
        """URL to the HuggingFace model repository."""
        return f"https://huggingface.co/{self.repo_id}"

    # Methods that don't make sense for remote models

    def publish_to_huggingface(self, *args, **kwargs) -> str:
        """Not supported for remote models (already published)."""
        raise NotImplementedError(
            f"Model is already published on HuggingFace at {self.repo_url}. "
            "To update, use a LocalEvaluationManager with the original experiment "
            "directory and call publish_to_huggingface() again with overwrite=True."
        )

    def __repr__(self) -> str:
        """String representation of remote manager."""
        data_status = "with data" if self.napistu_data_store else "without data"
        return (
            f"RemoteEvaluationManager(repo_id='{self.repo_id}', "
            f"revision='{self.revision}', {data_status})"
        )


# Public functions


def find_best_checkpoint(checkpoint_dir: Path) -> tuple[Path, float] | None:
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


# Private functions


def _parse_checkpoint_filename(filename: str | Path) -> tuple[int, float] | None:
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
        >>> _parse_checkpoint_filename("best-epoch=120-val_auc=0.7604.ckpt")
        (120, 0.7604)
    """
    import re

    from napistu_torch.ml.constants import METRIC_SUMMARIES

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


def _resolve_data_store_for_remote(
    data_store_dir: Path,
    experiment_config: ExperimentConfig,
    data_repo_id: Optional[str] = None,
    data_revision: Optional[str] = None,
    token: Optional[str] = None,
) -> NapistuDataStore:
    """
    Resolve and load a NapistuDataStore for remote evaluation.

    Handles three scenarios:
    1. data_store_dir exists -> load existing store
    2. data_store_dir missing + HF repo info available -> download from HF
    3. data_store_dir missing + no HF info -> error

    Parameters
    ----------
    data_store_dir : Path
        Directory for the data store (may or may not exist)
    experiment_config : ExperimentConfig
        Experiment configuration (may contain HF repo info)
    data_repo_id : Optional[str]
        HuggingFace data repository ID (overrides config)
    data_revision : Optional[str]
        Data revision (overrides config)
    token : Optional[str]
        HuggingFace token for private repos

    Returns
    -------
    NapistuDataStore
        Loaded data store

    Raises
    ------
    ValueError
        If data_store_dir doesn't exist and no HF repo info is available

    Examples
    --------
    >>> # Existing store
    >>> store = _resolve_data_store_for_remote(
    ...     data_store_dir=Path("./existing_store"),
    ...     experiment_config=config
    ... )
    >>>
    >>> # Download from HF
    >>> store = _resolve_data_store_for_remote(
    ...     data_store_dir=Path("./new_store"),
    ...     experiment_config=config,
    ...     data_repo_id="username/data-repo"
    ... )
    """
    # Check if store already exists
    registry_path = data_store_dir / NAPISTU_DATA_STORE_STRUCTURE.REGISTRY_FILE

    if registry_path.exists():
        logger.info(f"Loading existing data store from {data_store_dir}")
        return NapistuDataStore(data_store_dir)
    elif os.path.exists(data_store_dir):
        raise ValueError(
            f"Data store found at {data_store_dir}, but it is not a valid NapistuDataStore"
        )

    # Store doesn't exist - need to download from HF
    logger.info(
        f"Data store not found at {data_store_dir}, attempting to download from HuggingFace Hub"
    )

    # Try to get HF repo info from arguments or config
    if data_repo_id is None:
        data_repo_id = experiment_config.data.hf_repo_id

    if data_revision is None:
        data_revision = experiment_config.data.hf_revision

    # Validate we have what we need
    if data_repo_id is None:
        raise ValueError(
            f"Cannot load data store: {data_store_dir} does not exist and no "
            "HuggingFace repository specified.\n\n"
            "To fix this, either:\n"
            "1. Provide data_repo_id when creating the manager:\n"
            f"   RemoteEvaluationManager.from_huggingface(\n"
            f"       repo_id='...',\n"
            f"       data_store_dir='{data_store_dir}',\n"
            f"       data_repo_id='username/data-repo-name'\n"
            f"   )\n\n"
            "2. Or use an existing local data store by pointing data_store_dir to it."
        )

    # Download from HuggingFace
    logger.info(
        f"Downloading data store from {data_repo_id} "
        f"(revision: {data_revision or 'main'}) to {data_store_dir}..."
    )

    return NapistuDataStore.from_huggingface(
        repo_id=data_repo_id,
        store_dir=data_store_dir,
        revision=data_revision,
        token=token,
    )
