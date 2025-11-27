"""HuggingFace Hub integration for publishing Napistu-Torch models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

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
from napistu_torch.load.checkpoints import Checkpoint
from napistu_torch.ml.constants import DEVICE

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """
    Base client for interacting with HuggingFace Hub.

    Provides common functionality for authentication, validation, and repo operations.
    Subclassed by HuggingFacePublisher and HuggingFaceLoader for specific workflows.
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

    def _check_repo_exists(self, repo_id: str) -> bool:
        """
        Check if repository exists without raising errors.

        Parameters
        ----------
        repo_id : str
            Repository ID to check

        Returns
        -------
        bool
            True if repository exists, False otherwise
        """
        try:
            self.api.repo_info(repo_id, repo_type="model")
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


class HuggingFaceLoader(HuggingFaceClient):
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
    >>> from napistu_torch.ml.hugging_face import HuggingFaceLoader
    >>>
    >>> # Load complete encoder
    >>> loader = HuggingFaceLoader("shackett/napistu-sage-baseline-v1")
    >>> encoder = loader.load_encoder()
    >>>
    >>> # Load from specific revision
    >>> loader = HuggingFaceLoader("shackett/model-v1", revision="v1.0")
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


class HuggingFacePublisher(HuggingFaceClient):
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
    _upload_model_card(repo_id, manifest, commit_message)
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

        # Check if repo exists
        repo_exists = self._check_repo_exists(repo_id)

        if repo_exists and not overwrite:
            raise ValueError(
                f"Repository '{repo_id}' already exists.\n"
                f"To update it, call the function with overwrite=True or use the --overwrite CLI option."
            )

        # Create repo if needed (always private)
        if not repo_exists:
            logger.info(f"Creating private repository: {repo_id}")
            repo_url = create_repo(
                repo_id,
                private=True,  # Always private by default
                repo_type="model",
                token=self.api.token,
                exist_ok=True,  # Don't fail if created between check and now
            )
            logger.info(f"✓ Created private repository: {repo_url}")
        else:
            repo_url = f"https://huggingface.co/{repo_id}"
            logger.info(f"Updating existing repository: {repo_id}")

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
        self._upload_model_card(repo_id, manifest, commit_message)

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
        commit_message : str
            Commit message
        """
        model_card = generate_model_card(manifest)

        self.api.upload_file(
            path_or_fileobj=model_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        logger.info("✓ Uploaded model card: README.md")


# public functions


def generate_model_card(manifest: RunManifest) -> str:
    """
    Generate a basic model card in Markdown format.

    This creates a minimal card that users can enhance on HuggingFace.

    Parameters
    ----------
    manifest : RunManifest
        Run manifest with metadata

    Returns
    -------
    str
        Model card content in Markdown format
    """
    # Extract key config info
    config = manifest.experiment_config
    model_config = config.model
    encoder = model_config.encoder
    head = model_config.head
    hidden_channels = model_config.hidden_channels

    # Build model card
    card = f"""---
tags:
- graph-neural-network
- biological-networks
- napistu
- pytorch
library_name: napistu-torch
---

# {manifest.experiment_name or 'Napistu-Torch Model'}

This model was trained using [Napistu-Torch](https://github.com/napistu/Napistu-Torch), a framework for training graph neural networks on biological pathway networks.

## Model Details

- **Encoder**: {encoder}
- **Head**: {head}
- **Hidden Channels**: {hidden_channels}
- **Training Date**: {manifest.created_at}

## Training Configuration

```json
{{
  "encoder": "{encoder}",
  "head": "{head}",
  "hidden_channels": {hidden_channels},
  "dropout": {model_config.dropout},
  "num_layers": {model_config.num_layers}
}}
```

## Weights & Biases

Training tracked at: [{manifest.wandb_run_url}]({manifest.wandb_run_url})

## Usage

```python
import torch
from napistu_torch.evaluation import EvaluationManager

# Download checkpoint
checkpoint_path = "model.ckpt"

# Load checkpoint
checkpoint = torch.load(checkpoint_path)

# Use with your NapistuData and model architecture
# (See Napistu-Torch documentation for details)
```

## Model Card Authors

This model card was auto-generated. Please edit to add:
- Model description and use case
- Training data details
- Performance metrics
- Limitations and biases
- Citation information

## Citation

If you use this model, please cite:

```bibtex
@software{{napistu_torch,
  title = {{Napistu-Torch}},
  author = {{Sean R Hackett}},
  url = {{https://github.com/napistu/Napistu-Torch}},
  year = {{2025}}
}}
```
"""
    return card
