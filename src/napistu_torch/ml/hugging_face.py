"""HuggingFace Hub integration for publishing Napistu-Torch models."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from tqdm import tqdm

try:
    from huggingface_hub import HfApi, create_repo, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError as e:
    raise ImportError(
        "HuggingFace Hub integration requires additional dependencies.\n"
        "Install with: pip install napistu-torch[lightning]\n"
        "Or install huggingface_hub directly: pip install 'huggingface_hub[cli]>=0.20.0'"
    ) from e

from napistu_torch.configs import ExperimentConfig, RunManifest
from napistu_torch.constants import (
    NAPISTU_DATA_STORE,
    NAPISTU_DATA_STORE_STRUCTURE,
)
from napistu_torch.load.checkpoints import Checkpoint
from napistu_torch.ml.constants import (
    DEVICE,
    HUGGING_FACE_REPOS,
    VALID_HUGGING_FACE_REPOS,
)
from napistu_torch.ml.wandb import get_wandb_metrics_table
from napistu_torch.models.constants import RELATION_AWARE_HEADS
from napistu_torch.napistu_data_store import NapistuDataStore
from napistu_torch.tasks.constants import TASK_DESCRIPTIONS
from napistu_torch.utils.table_utils import format_metrics_as_markdown

logger = logging.getLogger(__name__)


class HFClient:
    """
    Base client for interacting with HuggingFace Hub.

    Provides common functionality for authentication, validation, and repo operations.
    Subclassed by HFModelPublisher and HFModelLoader for specific workflows.
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initialize HuggingFace client.

        Parameters
        ----------
        token : Optional[str]
            HuggingFace API token. If None, uses token from `huggingface-cli login`.
        """
        self.api = HfApi(token=token)
        self._token = token
        self._validate_authentication()

    def _check_repo_exists(
        self, repo_id: str, repo_type: str = HUGGING_FACE_REPOS.MODEL
    ) -> bool:
        """
        Check if repository exists without raising errors.

        Parameters
        ----------
        repo_id : str
            Repository ID to check
        repo_type : str
            Type of repository ("model" or "dataset")

        Returns
        -------
        bool
            True if repository exists, False otherwise
        """
        if repo_type not in VALID_HUGGING_FACE_REPOS:
            raise ValueError(f"Invalid repository type: {repo_type}")
        try:
            self.api.repo_info(repo_id, repo_type=repo_type)
            return True
        except RepositoryNotFoundError:
            return False

    def _validate_authentication(self) -> None:
        """Verify HuggingFace authentication is working."""
        try:
            # Simple check: attempt to get user info
            self.api.whoami()
            logger.info("✓ HuggingFace authentication verified")
        except Exception as e:
            raise RuntimeError(
                "HuggingFace authentication failed. Please run:\n"
                "  huggingface-cli login\n"
                f"Error: {e}"
            )

    def _validate_repo_id(self, repo_id: str) -> None:
        """
        Validate repository ID format.

        Parameters
        ----------
        repo_id : str
            Repository ID to validate

        Raises
        ------
        ValueError
            If repo_id format is invalid
        """
        if "/" not in repo_id:
            raise ValueError(
                f"Invalid repo_id format: '{repo_id}'\n"
                f"Expected format: 'username/repo-name'"
            )
        parts = repo_id.split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"Invalid repo_id format: '{repo_id}'\n"
                f"Expected format: 'username/repo-name'"
            )

    def _get_repo_url(
        self,
        repo_id: str,
        repo_type: str = HUGGING_FACE_REPOS.MODEL,
        overwrite: bool = False,
    ) -> str:
        """
        Create or get repository URL.

        Parameters
        ----------
        repo_id : str
            Repository ID in format "username/repo-name"
        repo_type : str
            Type of repository ("model" or "dataset")
        overwrite : bool
            If True, allow overwriting existing repository

        Returns
        -------
        str
            URL to the repository

        Raises
        ------
        ValueError
            If repo exists and overwrite=False
        """
        repo_exists = self._check_repo_exists(repo_id, repo_type=repo_type)

        if repo_exists and not overwrite:
            raise ValueError(
                f"Repository '{repo_id}' already exists.\n"
                f"To update it, call the function with overwrite=True."
            )

        # Create repo if needed (always private)
        if not repo_exists:
            logger.info(f"Creating private {repo_type} repository: {repo_id}")
            repo_url = create_repo(
                repo_id,
                private=True,
                repo_type=repo_type,
                token=self.api.token,
                exist_ok=True,
            )
            logger.info(f"✓ Created private repository: {repo_url}")
            return repo_url
        else:
            repo_type_prefix = "datasets" if repo_type == "dataset" else ""
            repo_url = f"https://huggingface.co/{repo_type_prefix}/{repo_id}".replace(
                "//", "/"
            )
            logger.info(f"Updating existing repository: {repo_id}")
            return repo_url


class HFDatasetLoader(HFClient):
    """
    Load NapistuDataStore from HuggingFace Hub.

    This class handles downloading a complete NapistuDataStore (registry.json and
    all artifacts) from a HuggingFace dataset repository and creating a local store.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository in format "username/repo-name"
    store_dir : Path
        Local directory where the store will be created
    revision : Optional[str]
        Git revision (branch, tag, or commit hash). Defaults to "main"
    token : Optional[str]
        HuggingFace access token for private repositories

    Public Methods
    --------------
    load_store()
        Download and create a NapistuDataStore from HuggingFace Hub.
        Returns the path to the store directory.

    Private Methods
    ---------------
    _list_repo_files()
        List all files in the repository
    _download_registry()
        Download registry.json from HuggingFace Hub
    _download_artifacts()
        Download all artifact files based on registry
    _create_store_structure()
        Create the local store directory structure

    Examples
    --------
    >>> from napistu_torch.ml.hugging_face import HFDatasetLoader
    >>> from pathlib import Path
    >>>
    >>> # Load store from HuggingFace Hub
    >>> loader = HFDatasetLoader(
    ...     repo_id="username/my-dataset",
    ...     store_dir=Path("./local_store")
    ... )
    >>> store = loader.load_store()
    >>>
    >>> # Load from specific revision
    >>> loader = HFDatasetLoader(
    ...     repo_id="username/my-dataset",
    ...     store_dir=Path("./local_store"),
    ...     revision="v1.0"
    ... )
    >>> store = loader.load_store()
    """

    def __init__(
        self,
        repo_id: str,
        store_dir: Path,
        revision: Optional[str] = None,
        token: Optional[str] = None,
    ):
        super().__init__(token=token)
        self.repo_id = repo_id
        self.store_dir = Path(store_dir)
        self.revision = revision or "main"

        # Validate repo_id format
        self._validate_repo_id(repo_id)

        # Check if repo exists
        if not self._check_repo_exists(repo_id, repo_type=HUGGING_FACE_REPOS.DATASET):
            raise ValueError(
                f"Repository '{repo_id}' not found on HuggingFace Hub. "
                f"Please check the repository name and ensure you have access."
            )

        # Cache for downloaded registry
        self._registry: Optional[Dict] = None

    def load_store(self) -> Path:
        """
        Download and create a NapistuDataStore from HuggingFace Hub.

        Downloads registry.json and all artifact files, creates the local store
        directory structure, and returns the path to the store directory.

        Returns
        -------
        Path
            Path to the store directory (ready to be loaded with NapistuDataStore)

        Raises
        ------
        ValueError
            If repository doesn't exist or registry is invalid
        FileNotFoundError
            If required files are missing from the repository
        """
        logger.info(
            f"Loading NapistuDataStore from {self.repo_id} (revision: {self.revision})..."
        )

        # Create store directory structure
        self._create_store_structure()

        # Download registry first to know what artifacts to download
        registry = self._download_registry()

        # Download all artifacts
        self._download_artifacts(registry)

        # Return the store directory path
        return self.store_dir

    # Private methods

    def _create_store_structure(self) -> None:
        """Create the local store directory structure."""
        # Create main store directory
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for artifacts
        (self.store_dir / NAPISTU_DATA_STORE_STRUCTURE.NAPISTU_DATA).mkdir(
            parents=True, exist_ok=True
        )
        (self.store_dir / NAPISTU_DATA_STORE_STRUCTURE.VERTEX_TENSORS).mkdir(
            parents=True, exist_ok=True
        )
        (self.store_dir / NAPISTU_DATA_STORE_STRUCTURE.PANDAS_DFS).mkdir(
            parents=True, exist_ok=True
        )

    def _download_registry(self) -> Dict:
        """
        Download registry.json from HuggingFace Hub.

        Returns
        -------
        Dict
            Registry dictionary
        """
        if self._registry is None:
            logger.info("Downloading registry.json...")

            registry_path = Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename="registry.json",
                    revision=self.revision,
                    repo_type=HUGGING_FACE_REPOS.DATASET,
                    token=self._token,
                )
            )

            # Read and parse registry
            import json

            with open(registry_path, "r") as f:
                self._registry = json.load(f)

            # Copy registry to store directory
            store_registry_path = (
                self.store_dir / NAPISTU_DATA_STORE_STRUCTURE.REGISTRY_FILE
            )
            shutil.copy2(registry_path, store_registry_path)

            logger.info(f"✓ Downloaded registry.json to {store_registry_path}")

        return self._registry

    def _download_artifacts(self, registry: Dict) -> None:
        """
        Download all artifact files based on registry.

        Parameters
        ----------
        registry : Dict
            Registry dictionary containing artifact metadata
        """
        # Collect all files to download
        files_to_download = []

        # Define artifact types to process
        artifact_types = [
            (
                NAPISTU_DATA_STORE.NAPISTU_DATA,
                NAPISTU_DATA_STORE_STRUCTURE.NAPISTU_DATA,
            ),
            (
                NAPISTU_DATA_STORE.VERTEX_TENSORS,
                NAPISTU_DATA_STORE_STRUCTURE.VERTEX_TENSORS,
            ),
            (
                NAPISTU_DATA_STORE.PANDAS_DFS,
                NAPISTU_DATA_STORE_STRUCTURE.PANDAS_DFS,
            ),
        ]

        # Collect files from all artifact types
        for registry_key, directory_name in artifact_types:
            artifact_registry = registry.get(registry_key, {})
            for artifact_name, entry in artifact_registry.items():
                filename = entry[NAPISTU_DATA_STORE.FILENAME]
                files_to_download.append(
                    (
                        f"{directory_name}/{filename}",
                        self.store_dir / directory_name / filename,
                    )
                )

        # Download files with progress bar (tqdm is only for visual feedback, not speed)
        if files_to_download:
            logger.info(f"Downloading {len(files_to_download)} artifact files...")

            iterator = tqdm(files_to_download, desc="Downloading artifacts")

            for path_in_repo, local_path in iterator:
                # Skip if file already exists
                if local_path.exists():
                    logger.debug(f"Skipping {path_in_repo} (already exists)")
                    continue

                try:
                    downloaded_path = Path(
                        hf_hub_download(
                            repo_id=self.repo_id,
                            filename=path_in_repo,
                            revision=self.revision,
                            repo_type=HUGGING_FACE_REPOS.DATASET,
                            token=self._token,
                        )
                    )

                    # Copy to store directory (hf_hub_download may cache elsewhere)
                    shutil.copy2(downloaded_path, local_path)
                    logger.debug(f"✓ Downloaded {path_in_repo}")

                except Exception as e:
                    logger.warning(
                        f"Failed to download {path_in_repo}: {e}. Continuing..."
                    )

            logger.info(f"✓ Downloaded {len(files_to_download)} artifact files")
        else:
            logger.info("No artifacts found in registry")


class HFDatasetPublisher(HFClient):
    """
    Publish NapistuDataStore to HuggingFace Hub as a dataset repository.

    This class handles uploading an entire NapistuDataStore (all artifacts) to
    HuggingFace Hub. The published store will be read-only (sbml_dfs_path and
    napistu_graph_path set to None).

    Parameters
    ----------
    token : Optional[str]
        HuggingFace API token. If None, uses token from `huggingface-cli login`.

    Public Methods
    --------------
    publish_store(repo_id, store, revision=None, overwrite=False, commit_message=None)
        Publish entire store to HuggingFace Hub

    Private Methods
    ---------------
    _create_read_only_registry(store)
        Create a read-only registry with paths set to None
    _upload_all_artifacts(repo_id, store, commit_message)
        Upload all artifact files from the store
    _upload_registry(repo_id, registry, commit_message)
        Upload registry.json to HuggingFace Hub
    _upload_dataset_card(repo_id, dataset_card, commit_message)
        Upload dataset card (README.md) to HuggingFace Hub
    """

    def publish_store(
        self,
        repo_id: str,
        store: NapistuDataStore,
        revision: Optional[str] = None,
        overwrite: bool = False,
        commit_message: Optional[str] = None,
    ) -> str:
        """
        Publish entire NapistuDataStore to HuggingFace Hub.

        Uploads all artifacts from the store to a HuggingFace dataset repository.
        The published registry will have sbml_dfs_path and napistu_graph_path set
        to None, making it a read-only store.

        Parameters
        ----------
        repo_id : str
            Repository ID in format "username/repo-name"
        store : NapistuDataStore
            Store to publish
        revision : Optional[str]
            Git revision (branch, tag, or commit hash). Defaults to "main"
        overwrite : bool
            Explicitly confirm overwriting existing dataset (default: False)
        commit_message : Optional[str]
            Custom commit message (default: auto-generated)

        Returns
        -------
        str
            URL to the published dataset on HuggingFace Hub

        Raises
        ------
        ValueError
            If repo_id format is invalid or if repo exists and overwrite=False

        Examples
        --------
        >>> from napistu_torch.ml.hugging_face import HFDatasetPublisher
        >>> from napistu_torch.napistu_data_store import NapistuDataStore
        >>>
        >>> store = NapistuDataStore("path/to/store")
        >>> publisher = HFDatasetPublisher()
        >>> url = publisher.publish_store("username/my-dataset", store)
        """
        # Validate inputs
        self._validate_repo_id(repo_id)
        revision = revision or "main"

        # Get or create repository URL
        repo_url = self._get_repo_url(
            repo_id, repo_type=HUGGING_FACE_REPOS.DATASET, overwrite=overwrite
        )

        # Generate commit message if not provided
        if commit_message is None:
            commit_message = f"Publish NapistuDataStore: {store.store_dir.name}"

        # Upload all artifacts
        logger.info("Uploading all artifacts from store...")
        self._upload_all_artifacts(repo_id, store, commit_message)

        # Create read-only registry (with paths set to None)
        registry = self._create_read_only_registry(store)
        logger.info("Uploading registry.json (read-only)...")
        self._upload_registry(repo_id, registry, commit_message)

        # Generate and upload dataset card
        dataset_card = generate_dataset_card(store)
        logger.info("Uploading dataset card (README.md)...")
        self._upload_dataset_card(repo_id, dataset_card, commit_message)

        return repo_url

    # Private methods

    def _create_read_only_registry(self, store: NapistuDataStore) -> Dict:
        """
        Create a read-only registry with sbml_dfs_path and napistu_graph_path set to None.

        This allows the published store to be loaded as a read-only store.

        Parameters
        ----------
        store : NapistuDataStore
            Store to create registry from

        Returns
        -------
        Dict
            Registry dictionary with paths set to None
        """
        # Copy the registry and set paths to None for read-only mode
        registry = store.registry.copy()
        registry[NAPISTU_DATA_STORE.READ_ONLY] = True
        registry[NAPISTU_DATA_STORE.NAPISTU_RAW] = {
            NAPISTU_DATA_STORE.SBML_DFS: None,
            NAPISTU_DATA_STORE.NAPISTU_GRAPH: None,
        }

        return registry

    def _upload_all_artifacts(
        self, repo_id: str, store: NapistuDataStore, commit_message: str
    ) -> None:
        """
        Upload all artifact files from the store.

        Parameters
        ----------
        repo_id : str
            Repository ID
        store : NapistuDataStore
            Store containing artifacts to upload
        commit_message : str
            Commit message
        """
        # Upload NapistuData artifacts
        self._upload_artifact_type(
            repo_id=repo_id,
            store=store,
            artifact_names=store.list_napistu_datas(),
            registry_key=NAPISTU_DATA_STORE.NAPISTU_DATA,
            directory_name=NAPISTU_DATA_STORE_STRUCTURE.NAPISTU_DATA,
            commit_message=commit_message,
        )

        # Upload VertexTensor artifacts
        self._upload_artifact_type(
            repo_id=repo_id,
            store=store,
            artifact_names=store.list_vertex_tensors(),
            registry_key=NAPISTU_DATA_STORE.VERTEX_TENSORS,
            directory_name=NAPISTU_DATA_STORE_STRUCTURE.VERTEX_TENSORS,
            commit_message=commit_message,
        )

        # Upload pandas DataFrame artifacts
        self._upload_artifact_type(
            repo_id=repo_id,
            store=store,
            artifact_names=store.list_pandas_dfs(),
            registry_key=NAPISTU_DATA_STORE.PANDAS_DFS,
            directory_name=NAPISTU_DATA_STORE_STRUCTURE.PANDAS_DFS,
            commit_message=commit_message,
        )

    def _upload_artifact_type(
        self,
        repo_id: str,
        store: NapistuDataStore,
        artifact_names: list,
        registry_key: str,
        directory_name: str,
        commit_message: str,
    ) -> None:
        """
        Upload artifacts of a specific type.

        Parameters
        ----------
        repo_id : str
            Repository ID
        store : NapistuDataStore
            Store containing artifacts to upload
        artifact_names : list
            List of artifact names to upload
        registry_key : str
            Key in the registry for this artifact type
        directory_name : str
            Directory name in the store structure
        commit_message : str
            Commit message
        """
        for artifact_name in artifact_names:
            entry = store.registry[registry_key][artifact_name]
            filename = entry[NAPISTU_DATA_STORE.FILENAME]
            filepath = store.store_dir / directory_name / filename
            if filepath.exists():
                path_in_repo = f"{directory_name}/{filename}"
                self.api.upload_file(
                    path_or_fileobj=str(filepath),
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type=HUGGING_FACE_REPOS.DATASET,
                    commit_message=commit_message,
                )
                logger.info(f"✓ Uploaded: {path_in_repo}")

    def _upload_registry(
        self, repo_id: str, registry: Dict, commit_message: str
    ) -> None:
        """Upload registry.json to HuggingFace Hub."""
        import json

        registry_json = json.dumps(registry, indent=2)

        self.api.upload_file(
            path_or_fileobj=registry_json.encode("utf-8"),
            path_in_repo="registry.json",
            repo_id=repo_id,
            repo_type=HUGGING_FACE_REPOS.DATASET,
            commit_message=commit_message,
        )
        logger.info("✓ Uploaded registry.json")

    def _upload_dataset_card(
        self, repo_id: str, dataset_card: str, commit_message: str
    ) -> None:
        """Upload dataset card (README.md) to HuggingFace Hub."""
        self.api.upload_file(
            path_or_fileobj=dataset_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=HUGGING_FACE_REPOS.DATASET,
            commit_message=commit_message,
        )
        logger.info("✓ Uploaded dataset card: README.md")


class HFModelLoader(HFClient):
    """
    Load model components from HuggingFace Hub.

    This class handles downloading and reconstructing Napistu-Torch model
    components (encoders, heads) from published HuggingFace repositories.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository in format "username/repo-name"
    revision : str, optional
        Git revision (branch, tag, or commit hash). Defaults to "main"
    cache_dir : Path, optional
        Local cache directory for downloaded files. If None, uses HuggingFace's
        default cache (~/.cache/huggingface/hub/)
    token : str, optional
        HuggingFace access token for private repositories

    Public Methods
    --------------
    load_checkpoint()
        Load model checkpoint from HuggingFace Hub

    Private Methods
    ---------------
    _download_checkpoint()
        Download model checkpoint from HuggingFace Hub
    _download_config()
        Download config.json from HuggingFace Hub
    _get_checkpoint()
        Download and load checkpoint, with caching
    _get_config()
        Download and parse config, with caching

    Examples
    --------
    >>> from napistu_torch.ml.hugging_face import HFModelLoader
    >>>
    >>> # Load complete encoder
    >>> loader = HFModelLoader("shackett/napistu-sage-baseline-v1")
    >>> encoder = loader.load_encoder()
    >>>
    >>> # Load from specific revision
    >>> loader = HFModelLoader("shackett/model-v1", revision="v1.0")
    >>> head = loader.load_head()
    >>>
    >>> # Load both components
    >>> encoder = loader.load_encoder()
    >>> head = loader.load_head()
    """

    def __init__(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        token: Optional[str] = None,
    ):
        super().__init__(token=token)
        self.repo_id = repo_id
        self.revision = revision or "main"
        self.cache_dir = cache_dir

        # Validate repo_id format
        self._validate_repo_id(repo_id)

        # Check if repo exists
        if not self._check_repo_exists(repo_id):
            raise ValueError(
                f"Repository '{repo_id}' not found on HuggingFace Hub. "
                f"Please check the repository name and ensure you have access."
            )

        # Cache for downloaded files
        self._checkpoint_path: Optional[Path] = None
        self._config_path: Optional[Path] = None

    def load_checkpoint(
        self, raw_checkpoint: bool = False
    ) -> Union[Checkpoint, Dict[str, Any]]:
        """
        Load model checkpoint from HuggingFace Hub.

        Parameters
        ----------
        raw_checkpoint : bool, optional
            If True, return the raw checkpoint dictionary instead of a Checkpoint object.
            Defaults to False.

        Returns
        -------
        Checkpoint
            PyTorch checkpoint dictionary
        """

        if self._checkpoint_path is None:
            # download/load cache
            self._download_checkpoint()

        if raw_checkpoint:
            return torch.load(
                self._checkpoint_path, weights_only=False, map_location=DEVICE.CPU
            )
        else:
            return Checkpoint.load(self._checkpoint_path)

    def load_config(self) -> ExperimentConfig:
        """
        Download and parse config, with caching.

        Returns
        -------
        ExperimentConfig
            Parsed experiment configuration
        """
        if self._config_path is None:
            # download/load cache
            self._download_config()

        return ExperimentConfig.from_json(self._config_path)

    # private methods

    def _download_checkpoint(self) -> None:
        """
        Download model checkpoint from HuggingFace Hub and set the _checkpoint_path attribute.
        """
        if self._checkpoint_path is None:
            logger.info(
                f"Downloading checkpoint from {self.repo_id} (revision: {self.revision})..."
            )

            self._checkpoint_path = Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename="model.ckpt",
                    revision=self.revision,
                    cache_dir=self.cache_dir,
                    repo_type="model",
                    token=self._token,
                )
            )

            logger.info(f"Checkpoint cached at: {self._checkpoint_path}")

        return None

    def _download_config(self) -> None:
        """
        Download config.json from HuggingFace Hub and set the _config_path attribute.
        """
        if self._config_path is None:
            logger.info(
                f"Downloading config from {self.repo_id} (revision: {self.revision})..."
            )

            self._config_path = Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename="config.json",
                    revision=self.revision,
                    cache_dir=self.cache_dir,
                    repo_type="model",
                    token=self._token,
                )
            )

            logger.info(f"Config cached at: {self._config_path}")

        return None


class HFModelPublisher(HFClient):
    """
    Handles publishing models to HuggingFace Hub

    Parameters
    ----------
    token : Optional[str]
        HuggingFace API token. If None, uses token from `huggingface-cli login`.

    Public Methods
    --------------
    publish_model(repo_id, checkpoint_path, manifest, commit_message=None, overwrite=False)
        Upload model checkpoint and metadata to HuggingFace Hub

    Private Methods
    ---------------
    _upload_checkpoint(repo_id, checkpoint_path, commit_message)
        Upload model checkpoint file
    _upload_config(repo_id, config, commit_message)
        Upload model configuration as JSON
    _upload_model_card(repo_id, manifest, checkpoint_path, commit_message)
        Generate and upload model card (README.md)
    """

    # public methods

    def publish_model(
        self,
        repo_id: str,
        checkpoint_path: Path,
        manifest: RunManifest,
        commit_message: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        """
        Upload model checkpoint and metadata to HuggingFace Hub.

        Creates a private repository if it doesn't exist. If the repository
        already exists, requires overwrite=True to confirm updating.

        Parameters
        ----------
        repo_id : str
            Repository ID in format "username/repo-name"
        checkpoint_path : Path
            Path to model checkpoint (.ckpt file)
        manifest : RunManifest
            Run manifest with metadata
        commit_message : Optional[str]
            Custom commit message (default: auto-generated from manifest)
        overwrite : bool
            Explicitly confirm overwriting existing model (default: False)

        Returns
        -------
        str
            URL to the published model on HuggingFace Hub

        Raises
        ------
        ValueError
            If repo_id format is invalid or if repo exists and overwrite=False
        FileNotFoundError
            If checkpoint doesn't exist
        """

        config = manifest.experiment_config

        # Validate inputs
        self._validate_repo_id(repo_id)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Get or create repository URL
        repo_url = self._get_repo_url(repo_id, repo_type="model", overwrite=overwrite)

        # Generate commit message if not provided
        if commit_message is None:
            logger.warning("No commit message provided, using default generation")
            commit_message = "Default commit message"

        # Upload files
        logger.info("Uploading checkpoint...")
        self._upload_checkpoint(repo_id, checkpoint_path, commit_message)

        logger.info("Uploading config...")
        self._upload_config(repo_id, config, commit_message)

        logger.info("Uploading model card...")
        self._upload_model_card(repo_id, manifest, checkpoint_path, commit_message)

        return repo_url

    # private methods

    def _upload_checkpoint(
        self, repo_id: str, checkpoint_path: Path, commit_message: str
    ) -> None:
        """
        Upload model checkpoint file.

        Parameters
        ----------
        repo_id : str
            Repository ID
        checkpoint_path : Path
            Path to checkpoint file
        commit_message : str
            Commit message
        """
        self.api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo="model.ckpt",
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        logger.info("✓ Uploaded checkpoint: model.ckpt")

    def _upload_config(
        self, repo_id: str, config: ExperimentConfig, commit_message: str
    ) -> None:
        """
        Upload model configuration as JSON.

        Parameters
        ----------
        repo_id : str
            Repository ID
        config : ExperimentConfig
            Experiment configuration
        commit_message : str
            Commit message
        """
        # Anonymize config to mask local file paths before uploading
        anonymized_config = config.anonymize(inplace=False)
        # Use Pydantic's model_dump_json() which automatically handles Path serialization
        config_json = anonymized_config.model_dump_json(indent=2)

        self.api.upload_file(
            path_or_fileobj=config_json.encode("utf-8"),
            path_in_repo="config.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        logger.info("✓ Uploaded config: config.json")

    def _upload_model_card(
        self,
        repo_id: str,
        manifest: RunManifest,
        checkpoint_path: Path,
        commit_message: str,
    ) -> None:
        """
        Generate and upload model card (README.md).

        Parameters
        ----------
        repo_id : str
            Repository ID
        config : ExperimentConfig
            Experiment configuration
        manifest : RunManifest
            Run manifest with metadata
        checkpoint_path : Path
            Path to checkpoint file
        commit_message : str
            Commit message
        """
        model_card = generate_model_card(manifest, repo_id, checkpoint_path)

        self.api.upload_file(
            path_or_fileobj=model_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        logger.info("✓ Uploaded model card: README.md")


# public functions


def generate_model_card(
    manifest: RunManifest, repo_id: str, checkpoint_path: Path
) -> str:
    """
    Generate a comprehensive HuggingFace model card from run metadata.

    Parameters
    ----------
    manifest : RunManifest
        Run manifest with metadata
    repo_id : str
        Repository ID in format "username/repo-name"
    checkpoint_path : Path
        Path to checkpoint file

    Returns
    -------
    str
        Model card as a string
    """

    config = manifest.experiment_config
    model_config = config.model
    task_config = config.task
    experiment_name = manifest.experiment_name or "Napistu-Torch Model"

    # Extract model details
    encoder = model_config.encoder
    head = model_config.head
    task = task_config.task

    # Detect relation-aware head
    is_relation_aware = head in RELATION_AWARE_HEADS

    # Build tags
    tags = [
        "graph-neural-networks",
        "biological-networks",
        "napistu",
        "pytorch",
        encoder,
        head,
        task,
    ]
    if is_relation_aware:
        tags.append("relation-aware")

    # Get architecture details from ModelConfig __repr__
    arch_details = repr(model_config)

    # Get task description
    task_description = TASK_DESCRIPTIONS[task]

    # Get metrics table
    metrics_table = get_wandb_metrics_table(
        wandb_entity=manifest.wandb_entity,
        wandb_project=manifest.wandb_project,
        wandb_run_id=manifest.wandb_run_id,
    )
    metrics_markdown = format_metrics_as_markdown(metrics_table)

    # Build W&B link if available
    wandb_link = ""
    if manifest.wandb_run_url:
        wandb_link = f"- 📊 [W&B Run]({manifest.wandb_run_url})"

    # get installation directions
    checkpoint = Checkpoint.load(checkpoint_path)
    installation_directions = checkpoint.environment_info.get_install_directions()

    # Build the model card
    card = f"""---
tags: {tags}
library_name: napistu-torch
license: mit
metrics:
- auc
- average_precision
---

# {experiment_name}

This model was trained using [Napistu-Torch](https://www.shackett.org/napistu_torch/), a PyTorch framework for training graph neural networks on biological pathway networks.

The dataset used for training is the 8-source ["Octopus" human consensus network](https://www.shackett.org/octopus_network/), which integrates pathway data from STRING, OmniPath, Reactome, and others. The network encompasses ~50K genes, metabolites, and complexes connected by ~8M interactions.

## Task

{task_description}

## Model Description

{arch_details}

**Training Date**: {manifest.created_at.strftime('%Y-%m-%d')}

For detailed experiment and training settings see this repository's `config.json` file.

## Performance

{metrics_markdown}

## Links

{wandb_link}
- 💻 [GitHub Repository](https://github.com/napistu/Napistu-Torch)
- 📖 [Read the Docs](https://napistu-torch.readthedocs.io/en/latest)
- 📚 [Napistu Wiki](https://github.com/napistu/napistu/wiki)

## Usage

### 1. Setup Environment

To reproduce the environment used for training, run the following commands:

```bash
{installation_directions}
```

### 2. Setup Data Store

First, download the Octopus consensus network data to create a local `NapistuDataStore`:
```python
from napistu_torch.load.gcs import gcs_model_to_store

# Download data and create store
napistu_data_store = gcs_model_to_store(
    napistu_data_dir="path/to/napistu_data",
    store_dir="path/to/store",
    asset_name="human_consensus",
    # Pin to stable version for reproducibility
    asset_version="20250923"
)
```

### 3. Load Pretrained Model from HuggingFace Hub
```python
from napistu_torch.ml.hugging_face import HFModelLoader

# Load checkpoint
loader = HFModelLoader("{repo_id}")
checkpoint = loader.load_checkpoint()

# Load config to reproduce experiment
experiment_config = loader.load_config()
```

### 4. Use Pretrained Model for Training

You can use this pretrained model as initialization for training via the CLI:
```bash
# Create a training config that uses the pretrained model
cat > my_config.yaml << EOF
name: my_finetuned_model

model:
  use_pretrained_model: true
  pretrained_model_source: huggingface
  pretrained_model_path: {repo_id}
  pretrained_model_freeze_encoder_weights: false  # Allow fine-tuning

data:
  sbml_dfs_path: path/to/sbml_dfs.pkl
  napistu_graph_path: path/to/graph.pkl
  napistu_data_name: edge_prediction

training:
  epochs: 100
  lr: 0.001
EOF

# Train with pretrained weights
napistu-torch train my_config.yaml
```

## Citation

If you use this model, please cite:
```bibtex
@software{{napistu_torch,
  title = {{Napistu-Torch: Graph Neural Networks for Biological Pathway Analysis}},
  author = {{Hackett, Sean R.}},
  url = {{https://github.com/napistu/Napistu-Torch}},
  year = {{2025}},
  note = {{Model: {experiment_name}}}
}}
```

## License

MIT License - See [LICENSE](https://github.com/napistu/Napistu-Torch/blob/main/LICENSE) for details.
"""
    return card


def generate_dataset_card(store: NapistuDataStore) -> str:
    """
    Generate a dataset card (README.md) for a NapistuDataStore.

    Parameters
    ----------
    store : NapistuDataStore
        Store to generate card for

    Returns
    -------
    str
        Dataset card as markdown string
    """
    napistu_data_count = len(store.list_napistu_datas())
    vertex_tensor_count = len(store.list_vertex_tensors())
    pandas_df_count = len(store.list_pandas_dfs())

    # Build tags
    tags = [
        "napistu",
        "biological-networks",
        "graph-neural-networks",
        "napistu-data-store",
    ]

    # Build dataset card
    card = f"""---
tags: {tags}
library_name: napistu-torch
license: mit
---

# NapistuDataStore Dataset

This dataset contains a complete NapistuDataStore with all artifacts published as a read-only store.

## Store Contents

- **NapistuData artifacts**: {napistu_data_count}
- **VertexTensor artifacts**: {vertex_tensor_count}
- **pandas DataFrame artifacts**: {pandas_df_count}
- **Total artifacts**: {napistu_data_count + vertex_tensor_count + pandas_df_count}

## Artifacts

### NapistuData
{chr(10).join(f"- `{name}`" for name in store.list_napistu_datas()) if store.list_napistu_datas() else "- None"}

### VertexTensors
{chr(10).join(f"- `{name}`" for name in store.list_vertex_tensors()) if store.list_vertex_tensors() else "- None"}

### pandas DataFrames
{chr(10).join(f"- `{name}`" for name in store.list_pandas_dfs()) if store.list_pandas_dfs() else "- None"}

## Usage

### Load from HuggingFace Hub

```python
from huggingface_hub import hf_hub_download
from napistu_torch.napistu_data_store import NapistuDataStore
import tempfile
import shutil

# Download registry.json and create a temporary store directory
with tempfile.TemporaryDirectory() as temp_dir:
    # Download registry.json
    registry_path = hf_hub_download(
        repo_id="username/repo-name",
        filename="registry.json",
        repo_type="dataset",
        local_dir=temp_dir
    )
    
    # Create store from downloaded registry
    store = NapistuDataStore(temp_dir)
    
    # Use the store (read-only)
    napistu_data = store.load_napistu_data("edge_prediction")
```

## Note

This is a read-only store. The `sbml_dfs_path` and `napistu_graph_path` are set to None,
so you cannot create new artifacts that require the raw SBML_dfs or NapistuGraph files.
You can still load and use all existing artifacts.

## Citation

If you use this dataset, please cite:

```bibtex
@software{{napistu_torch,
  title = {{Napistu-Torch: Graph Neural Networks for Biological Pathway Analysis}},
  author = {{Hackett, Sean R.}},
  url = {{https://github.com/napistu/Napistu-Torch}},
  year = {{2025}}
}}
```

## License

MIT License - See [LICENSE](https://github.com/napistu/Napistu-Torch/blob/main/LICENSE) for details.
"""
    return card
