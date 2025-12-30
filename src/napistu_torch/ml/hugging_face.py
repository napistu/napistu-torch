"""HuggingFace Hub integration for publishing Napistu-Torch models."""

from __future__ import annotations

import json
import logging
from json import load as json_load
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, Optional, Union

import yaml
from torch import load as torch_load
from tqdm import tqdm

try:
    from huggingface_hub import (
        HfApi,
        create_repo,
        create_tag,
        hf_hub_download,
        list_repo_refs,
    )
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
    DEFAULT_HUGGING_FACE_TAGS,
    DEVICE,
    HUGGING_FACE_REPOS,
    VALID_HUGGING_FACE_REPOS,
)
from napistu_torch.ml.wandb import WandbRunInfo, get_wandb_metrics_table
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

    Attributes
    ----------
    api : HfApi
        HuggingFace API client with an initialized token
    _token : Optional[str]
        HuggingFace API token
    _validate_authentication() : None
        Verify HuggingFace authentication is working

    Private Methods
    ---------------
    _check_repo_exists(repo_id, repo_type=HUGGING_FACE_REPOS.MODEL) : bool
        Check if repository exists without raising errors
    _check_tag_exists(repo_id, tag, repo_type=HUGGING_FACE_REPOS.MODEL) : bool
        Check if a tag already exists in the HuggingFace repository
    _create_tag(repo_id, tag, tag_message=None, repo_type=HUGGING_FACE_REPOS.MODEL, revision=None) : None
        Create a git tag in the HuggingFace repository
    _get_repo_url(repo_id, repo_type=HUGGING_FACE_REPOS.MODEL, overwrite=False) : str
        Create or get repository URL
    _validate_authentication() : None
        Verify HuggingFace authentication is working
    _validate_repo_id(repo_id) : None
        Validate repository ID format
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

    def _check_tag_exists(
        self,
        repo_id: str,
        tag: str,
        repo_type: str = HUGGING_FACE_REPOS.MODEL,
    ) -> bool:
        """
        Check if a tag already exists in the HuggingFace repository.

        Parameters
        ----------
        repo_id : str
            Repository ID
        tag : str
            Tag name to check (e.g., "v1.0")
        repo_type : str
            Type of repository. Defaults to HUGGING_FACE_REPOS.MODEL

        Returns
        -------
        bool
            True if tag exists, False otherwise
        """
        try:
            refs = list_repo_refs(
                repo_id=repo_id,
                repo_type=repo_type,
                token=self._token,
            )
            existing_tags = [ref.ref for ref in refs.tags]
            return tag in existing_tags
        except Exception as e:
            logger.warning(f"Failed to check for existing tags: {e}")
            # If we can't check, assume tag doesn't exist to allow creation attempt
            return False

    def _create_tag(
        self,
        repo_id: str,
        tag: str,
        tag_message: Optional[str] = None,
        repo_type: str = HUGGING_FACE_REPOS.MODEL,
        revision: Optional[str] = None,
    ) -> None:
        """
        Create a git tag in the HuggingFace repository.

        Parameters
        ----------
        repo_id : str
            Repository ID
        tag : str
            Tag name (e.g., "v1.0")
        tag_message : Optional[str]
            Optional message for the tag
        repo_type : str
            Type of repository. Defaults to HUGGING_FACE_REPOS.MODEL
        revision : Optional[str]
            Git revision to tag (branch name or commit hash).
            If None, tags the latest commit on the default branch.

        Raises
        ------
        ValueError
            If the tag already exists in the repository
        """
        # Check if tag already exists
        if self._check_tag_exists(repo_id, tag, repo_type=repo_type):
            raise ValueError(
                f"Tag '{tag}' already exists in repository '{repo_id}'. "
                f"Cannot create duplicate tag."
            )

        try:
            create_tag(
                repo_id=repo_id,
                tag=tag,
                tag_message=tag_message,
                repo_type=repo_type,
                revision=revision,
                token=self._token,
            )
            logger.info(f"✓ Created tag: {tag}")
        except Exception as e:
            logger.error(f"Failed to create tag {tag}: {e}")
            raise

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
            with open(registry_path, "r") as f:
                self._registry = json_load(f)

            # Copy registry to store directory
            store_registry_path = (
                self.store_dir / NAPISTU_DATA_STORE_STRUCTURE.REGISTRY_FILE
            )
            copy2(registry_path, store_registry_path)

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
                    copy2(downloaded_path, local_path)
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
        asset_name: Optional[str] = None,
        asset_version: Optional[str] = None,
        tag: Optional[str] = None,
        tag_message: Optional[str] = None,
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
        asset_name : Optional[str]
            Name of the GCS asset used to create the store (for documentation)
        asset_version : Optional[str]
            Version of the GCS asset used to create the store (for documentation)
        tag : Optional[str]
            Tag name to create after all assets are uploaded (e.g., "v1.0")
        tag_message : Optional[str]
            Optional message for the tag

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
        >>> url = publisher.publish_store(
        ...     "username/my-dataset",
        ...     store,
        ...     asset_name="human_consensus",
        ...     asset_version="v1.0",
        ...     tag="v1.0"
        ... )
        """
        # Validate inputs
        self._validate_repo_id(repo_id)
        revision = revision or "main"

        # Validate that store has at least one NapistuData artifact
        napistu_data_list = store.list_napistu_datas()
        if len(napistu_data_list) == 0:
            raise ValueError(
                "Cannot publish store: Store must contain at least one NapistuData artifact. "
                "Publishing currently requires one or more NapistuData objects."
            )

        # Check if tag already exists (fail early before uploading)
        if tag:
            if self._check_tag_exists(
                repo_id, tag, repo_type=HUGGING_FACE_REPOS.DATASET
            ):
                raise ValueError(
                    f"Tag '{tag}' already exists in repository '{repo_id}'. "
                    f"Cannot publish with duplicate tag."
                )

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
        dataset_card = generate_dataset_card(
            store,
            repo_id=repo_id,
            revision=revision,
            asset_name=asset_name,
            asset_version=asset_version,
        )
        logger.info("Uploading dataset card (README.md)...")
        self._upload_dataset_card(repo_id, dataset_card, commit_message)

        # Create tag if requested
        if tag:
            logger.info(f"Creating tag: {tag}")
            self._create_tag(
                repo_id, tag, tag_message, repo_type=HUGGING_FACE_REPOS.DATASET
            )

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
    load_config()
        Load experiment configuration from HuggingFace Hub
    load_run_info()
        Load WandB run information from HuggingFace Hub

    Private Methods
    ---------------
    _download_checkpoint()
        Download model checkpoint from HuggingFace Hub
    _download_config()
        Download config.json from HuggingFace Hub
    _download_run_info()
        Download wandb_run_info.yaml from HuggingFace Hub

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
        self._run_info_path: Optional[Path] = None

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
            return torch_load(
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

    def load_run_info(self) -> WandbRunInfo:
        """
        Load WandB run information from HuggingFace Hub.

        Downloads and parses the wandb_run_info.yaml file from the model repository.

        Returns
        -------
        WandbRunInfo
            WandB run information including summaries and metadata

        Examples
        --------
        >>> loader = HFModelLoader("username/model-name")
        >>> run_info = loader.load_run_info()
        >>> print(run_info.run_path)
        >>> print(run_info.run_summaries)
        """
        if self._run_info_path is None:
            self._download_run_info()

        return WandbRunInfo.from_yaml(self._run_info_path)

    # private methods

    def _download_checkpoint(self) -> None:
        """
        Download model checkpoint from HuggingFace Hub and set the _checkpoint_path attribute.
        """
        self._download_file("model.ckpt", "checkpoint", "_checkpoint_path")
        return None

    def _download_config(self) -> None:
        """
        Download config.json from HuggingFace Hub and set the _config_path attribute.
        """
        self._download_file("config.json", "config", "_config_path")
        return None

    def _download_file(self, filename: str, description: str, cache_attr: str) -> Path:
        """
        Download a file from HuggingFace Hub and cache it.

        Parameters
        ----------
        filename : str
            Name of the file to download
        description : str
            Human-readable description for logging (e.g., "checkpoint", "config", "run info")
        cache_attr : str
            Name of the instance attribute to store the cached path (e.g., "_checkpoint_path")

        Returns
        -------
        Path
            Path to the downloaded file
        """
        # Get current cached path
        cached_path = getattr(self, cache_attr)

        if cached_path is None:
            logger.info(
                f"Downloading {description} from {self.repo_id} (revision: {self.revision})..."
            )

            downloaded_path = Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=filename,
                    revision=self.revision,
                    cache_dir=self.cache_dir,
                    repo_type=HUGGING_FACE_REPOS.MODEL,
                    token=self._token,
                )
            )

            # Set the cache attribute
            setattr(self, cache_attr, downloaded_path)

            logger.info(f"{description.capitalize()} cached at: {downloaded_path}")
            return downloaded_path

        return cached_path

    def _download_run_info(self) -> None:
        """
        Download wandb_run_info.yaml from HuggingFace Hub and set the _run_info_path attribute.
        """
        self._download_file("wandb_run_info.yaml", "run info", "_run_info_path")
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
    publish_model(repo_id, checkpoint_path, manifest, commit_message, overwrite, tag, tag_message)
        Upload model checkpoint and metadata to HuggingFace Hub

    Private Methods
    ---------------
    _upload_checkpoint(repo_id, checkpoint_path, commit_message)
        Upload model checkpoint file
    _upload_config(repo_id, config, commit_message)
        Upload model configuration as JSON
    _upload_model_card(repo_id, manifest, checkpoint_path, commit_message)
        Generate and upload model card (README.md)
    _upload_run_info(repo_id, manifest, commit_message, wandb_run_summaries)
        Upload WandB run information as YAML
    """

    # public methods

    def publish_model(
        self,
        repo_id: str,
        checkpoint_path: Path,
        manifest: RunManifest,
        commit_message: Optional[str] = None,
        overwrite: bool = False,
        tag: Optional[str] = None,
        tag_message: Optional[str] = None,
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
        tag : Optional[str]
            Tag name to create after all assets are uploaded (e.g., "v1.0")
        tag_message : Optional[str]
            Optional message for the tag

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
        from napistu_torch.ml.wandb import get_wandb_run_summaries

        config = manifest.experiment_config

        # Validate inputs
        self._validate_repo_id(repo_id)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Check if tag already exists (fail early before uploading)
        if tag:
            if self._check_tag_exists(repo_id, tag, repo_type=HUGGING_FACE_REPOS.MODEL):
                raise ValueError(
                    f"Tag '{tag}' already exists in repository '{repo_id}'. "
                    f"Cannot publish with duplicate tag."
                )

        # Get or create repository URL
        repo_url = self._get_repo_url(
            repo_id, repo_type=HUGGING_FACE_REPOS.MODEL, overwrite=overwrite
        )

        # Generate commit message if not provided
        if commit_message is None:
            logger.warning("No commit message provided, using default generation")
            commit_message = "Default commit message"

        # Get WandB run summaries
        wandb_run_summaries = get_wandb_run_summaries(
            wandb_entity=manifest.wandb_entity,
            wandb_project=manifest.wandb_project,
            wandb_run_id=manifest.wandb_run_id,
        )

        # Upload files
        logger.info("Uploading checkpoint...")
        self._upload_checkpoint(repo_id, checkpoint_path, commit_message)

        logger.info("Uploading config...")
        self._upload_config(repo_id, config, commit_message)

        logger.info("Uploading model card...")
        self._upload_model_card(
            repo_id, manifest, checkpoint_path, commit_message, wandb_run_summaries
        )

        logger.info("Uploading run info...")
        self._upload_run_info(repo_id, manifest, commit_message, wandb_run_summaries)

        # Create tag if requested
        if tag:
            logger.info(f"Creating tag: {tag}")
            self._create_tag(
                repo_id, tag, tag_message, repo_type=HUGGING_FACE_REPOS.MODEL
            )

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
            repo_type=HUGGING_FACE_REPOS.MODEL,
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
            repo_type=HUGGING_FACE_REPOS.MODEL,
            commit_message=commit_message,
        )
        logger.info("✓ Uploaded config: config.json")

    def _upload_model_card(
        self,
        repo_id: str,
        manifest: RunManifest,
        checkpoint_path: Path,
        commit_message: str,
        wandb_run_summaries: Dict[str, Any],
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
        wandb_run_summaries : Dict[str, Any]
            WandB run summaries
        """
        model_card = generate_model_card(
            manifest, repo_id, checkpoint_path, wandb_run_summaries
        )

        self.api.upload_file(
            path_or_fileobj=model_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=HUGGING_FACE_REPOS.MODEL,
            commit_message=commit_message,
        )
        logger.info("✓ Uploaded model card: README.md")

    def _upload_run_info(
        self,
        repo_id: str,
        manifest: RunManifest,
        commit_message: str,
        wandb_run_summaries: Dict[str, Any],
    ) -> None:
        """
        Upload WandB run information as YAML to the model repository.

        Parameters
        ----------
        repo_id : str
            Repository ID
        manifest : RunManifest
            Run manifest with metadata
        commit_message : str
            Commit message
        wandb_run_summaries : Dict[str, Any]
            WandB run summaries
        """
        run_info = WandbRunInfo(
            run_summaries=wandb_run_summaries,
            wandb_entity=manifest.wandb_entity,
            wandb_project=manifest.wandb_project,
            wandb_run_id=manifest.wandb_run_id,
        )

        # Serialize to YAML string directly
        data = run_info.model_dump(mode="json")
        run_info_yaml = yaml.dump(data, default_flow_style=False, sort_keys=False)

        self.api.upload_file(
            path_or_fileobj=run_info_yaml.encode("utf-8"),
            path_in_repo="wandb_run_info.yaml",
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        logger.info("✓ Uploaded run info: wandb_run_info.yaml")


# public functions


def generate_model_card(
    manifest: RunManifest,
    repo_id: str,
    checkpoint_path: Path,
    wandb_run_summaries: Dict[str, Any],
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
    tags = DEFAULT_HUGGING_FACE_TAGS + [
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
    metrics_table = get_wandb_metrics_table(wandb_run_summaries)
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
- 🌐 [Napistu](https://napistu.com)
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


def generate_dataset_card(
    store: NapistuDataStore,
    repo_id: str,
    revision: Optional[str] = None,
    asset_name: Optional[str] = None,
    asset_version: Optional[str] = None,
) -> str:
    """
    Generate a dataset card (README.md) for a NapistuDataStore.

    Parameters
    ----------
    store : NapistuDataStore
        Store to generate card for
    repo_id : str
        HuggingFace repository ID (for usage examples)
    revision : Optional[str]
        Git revision (branch, tag, or commit hash) for usage examples
    asset_name : Optional[str]
        Name of the GCS asset used to create the store (for documentation)
    asset_version : Optional[str]
        Version of the GCS asset used to create the store (for documentation)

    Returns
    -------
    str
        Dataset card as markdown string
    """
    napistu_data_list = store.list_napistu_datas()
    napistu_data_count = len(napistu_data_list)
    vertex_tensor_count = len(store.list_vertex_tensors())
    pandas_df_count = len(store.list_pandas_dfs())

    # Validate that at least one NapistuData artifact exists
    if napistu_data_count == 0:
        raise ValueError(
            "Cannot generate dataset card: Store must contain at least one NapistuData artifact. "
            "Publishing currently requires one or more NapistuData objects."
        )

    # Get the first NapistuData artifact name for use in examples
    first_napistu_data_name = napistu_data_list[0]

    # Build tags
    tags = DEFAULT_HUGGING_FACE_TAGS + [
        "napistu-data-store",
    ]

    # Build source information section if asset details are provided
    source_section = ""
    if asset_name is not None:
        source_section = "\n## Source Data\n\n"
        source_section += f"This store was created from GCS asset: **{asset_name}**"
        if asset_version is not None:
            source_section += f" (version: **{asset_version}**)"
        source_section += "\n"

    # Build revision snippets for code examples
    revision_snippet = ""
    if revision is not None:
        revision_snippet = f',\n    revision="{revision}"'

    revision_config_yaml_snippet = ""
    if revision is not None:
        revision_config_yaml_snippet = f'\n  hf_revision: "{revision}"'

    # Build GCS section if asset details are provided
    gcs_section = ""
    if asset_name is not None:
        # Build version string for code examples
        version_str = f'"{asset_version}"' if asset_version else "None"

        gcs_section = f"""
### Load Raw Data from GCS (Optional)

If you need to create new artifacts, you can convert this read-only store to a non-read-only store
by loading the raw data from GCS and passing the paths directly to `from_huggingface`:

```python
from napistu_torch.napistu_data_store import NapistuDataStore
from napistu.gcs.downloads import load_public_napistu_asset
from napistu.gcs.constants import GCS_SUBASSET_NAMES
from pathlib import Path
import tempfile

# Download raw data from GCS
with tempfile.TemporaryDirectory() as temp_data_dir:
    sbml_dfs_path = load_public_napistu_asset(
        "{asset_name}",
        temp_data_dir,
        subasset=GCS_SUBASSET_NAMES.SBML_DFS,
        version={version_str},
    )
    napistu_graph_path = load_public_napistu_asset(
        "{asset_name}",
        temp_data_dir,
        subasset=GCS_SUBASSET_NAMES.NAPISTU_GRAPH,
        version={version_str},
    )
    
    # Load and convert to non-read-only in one step
    store = NapistuDataStore.from_huggingface(
        repo_id="{repo_id}",
        store_dir=Path("./local_store"){revision_snippet},
        sbml_dfs_path=sbml_dfs_path,
        napistu_graph_path=napistu_graph_path,
    )
    
    # Now you can create new artifacts
    store.ensure_artifacts(["new_artifact_name"])
```
"""

    # Build dataset card
    card = f"""---
tags: {tags}
library_name: napistu-torch
license: mit
---

# NapistuDataStore Dataset

This dataset contains a complete NapistuDataStore with all artifacts published as a read-only store.
{source_section}
## Artifacts

### NapistuData ({napistu_data_count})
{chr(10).join(f"- `{name}`" for name in store.list_napistu_datas()) if store.list_napistu_datas() else "- None"}

### VertexTensor ({vertex_tensor_count})
{chr(10).join(f"- `{name}`" for name in store.list_vertex_tensors()) if store.list_vertex_tensors() else "- None"}

### Pandas DataFrame ({pandas_df_count})
{chr(10).join(f"- `{name}`" for name in store.list_pandas_dfs()) if store.list_pandas_dfs() else "- None"}

## Usage

### Load from HuggingFace Hub

The easiest way to load this dataset is using the `from_huggingface` class method:

```python
from napistu_torch.napistu_data_store import NapistuDataStore
from pathlib import Path

# Load read-only store from HuggingFace Hub
store = NapistuDataStore.from_huggingface(
    repo_id="{repo_id}",
    store_dir=Path("./local_store"){revision_snippet}
)

# Use the store (read-only)
napistu_data = store.load_napistu_data("{first_napistu_data_name}")
```

### Configure DataConfig

You can also use this dataset in your `DataConfig` YAML for PyTorch Lightning experiments:

```yaml
data:
  store_dir: "./local_store"
  hf_repo_id: "{repo_id}"{revision_config_yaml_snippet}
  napistu_data_name: "{first_napistu_data_name}"
```

To make the store writable (non-read-only), provide paths to the raw data files:

```yaml
data:
  store_dir: "./local_store"
  hf_repo_id: "{repo_id}"{revision_config_yaml_snippet}
  sbml_dfs_path: "/path/to/sbml_dfs.pkl"
  napistu_graph_path: "/path/to/napistu_graph.pkl"
  napistu_data_name: "{first_napistu_data_name}"
```
{gcs_section}
## Links

- 🌐 [Napistu](https://napistu.com)
- 💻 [GitHub Repository](https://github.com/napistu/Napistu-Torch)
- 📚 [Napistu Wiki](https://github.com/napistu/napistu/wiki)

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
