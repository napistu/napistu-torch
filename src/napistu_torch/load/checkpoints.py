"""
Checkpoint loading and validation utilities.

This module provides utilities for loading and validating pretrained Napistu-Torch models.

Classes
-------
Checkpoint
    Manager for PyTorch Lightning checkpoint loading and validation.
DataMetadata
    Metadata about the NapistuData used during training.
EdgeEncoderMetadata
    Metadata about the edge encoder component.
EncoderMetadata
    Metadata about the encoder component.
HeadMetadata
    Metadata about the head component.
ModelMetadata
    Metadata about the complete model architecture.
CheckpointHyperparameters
    Hyperparameters stored in the checkpoint.
CheckpointStructure
    Structure definition for checkpoint validation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel, Field, field_validator

from napistu_torch.configs import ModelConfig, TrainingConfig
from napistu_torch.constants import (
    MODEL_COMPONENTS,
    MODEL_CONFIG,
    NAPISTU_DATA_SUMMARY_TYPES,
)
from napistu_torch.data.compare_napistu_data import validate_same_data
from napistu_torch.load.constants import (
    CHECKPOINT_HYPERPARAMETERS,
    CHECKPOINT_STRUCTURE,
)
from napistu_torch.ml.constants import DEVICE
from napistu_torch.models.constants import MODEL_DEFS
from napistu_torch.napistu_data import NapistuData
from napistu_torch.utils.environment_info import EnvironmentInfo

logger = logging.getLogger(__name__)


class Checkpoint:
    """
    Manager for PyTorch Lightning checkpoint loading and validation.

    This class handles loading checkpoints, extracting metadata, validating
    compatibility with current data, and reconstructing model components.

    Parameters
    ----------
    checkpoint_dict : Dict[str, Any]
        PyTorch Lightning checkpoint dictionary (validated via Pydantic)

    Public Methods
    --------------
    assert_same_napistu_data(napistu_data: NapistuData) -> None:
        Validate that current NapistuData is compatible with checkpoint.
    load(checkpoint_path: Union[str, Path], map_location: str = DEVICE.CPU) -> "Checkpoint":
        Load and validate a checkpoint from a local file.
    get_encoder_config() -> Dict[str, Any]:
        Get encoder configuration as dictionary.
    get_head_config() -> Dict[str, Any]:
        Get head configuration as dictionary.
    get_data_summary() -> Dict[str, Any]:
        Get data summary as dictionary.
    update_model_config(model_config: ModelConfig, inplace: bool = True) -> ModelConfig:
        Update a ModelConfig instance with settings from the checkpoint.

    Private Methods
    ---------------
    _update_model_config_with_encoder(model_config: ModelConfig, inplace: bool = True) -> Optional[ModelConfig]:
        Update a ModelConfig instance with encoder configuration from checkpoint.
    _update_model_config_with_head(model_config: ModelConfig, inplace: bool = True) -> Optional[ModelConfig]:
        Update a ModelConfig instance with head configuration from checkpoint.

    Examples
    --------
    >>> # Load from local file (automatically validated)
    >>> checkpoint = Checkpoint.load("path/to/checkpoint.ckpt")
    >>>
    >>> # Validate compatibility with current data
    >>> checkpoint.assert_same_napistu_data(current_data)
    >>>
    >>> # Access validated configurations
    >>> encoder_config = checkpoint.encoder_metadata
    >>> head_config = checkpoint.head_metadata
    >>> data_config = checkpoint.data_metadata
    """

    def __init__(self, checkpoint_dict: Dict[str, Any]):
        """
        Initialize Checkpoint from a checkpoint dictionary.

        Parameters
        ----------
        checkpoint_dict : Dict[str, Any]
            PyTorch Lightning checkpoint dictionary

        Raises
        ------
        ValidationError
            If checkpoint structure is invalid
        """
        # Validate checkpoint structure using Pydantic
        validated = CheckpointStructure.model_validate(checkpoint_dict)

        # Store original dict for state_dict access
        self.checkpoint = checkpoint_dict
        self.state_dict = checkpoint_dict[CHECKPOINT_STRUCTURE.STATE_DICT]

        # Expose validated metadata as properties
        self.hyper_parameters = validated.hyper_parameters
        self.model_metadata = self.hyper_parameters.model
        self.data_metadata = self.hyper_parameters.data
        self.encoder_metadata = self.model_metadata.encoder
        self.head_metadata = self.model_metadata.head
        self.edge_encoder_metadata = self.model_metadata.edge_encoder
        self.environment_info = self.hyper_parameters.environment

    def assert_same_napistu_data(self, napistu_data: NapistuData) -> None:
        """
        Validate that current NapistuData is compatible with checkpoint.

        Compares the data summary from the checkpoint with a summary
        generated from the provided NapistuData object.

        Parameters
        ----------
        napistu_data : NapistuData
            Current NapistuData object to validate against checkpoint

        Raises
        ------
        ValueError
            If data summaries don't match

        Examples
        --------
        >>> checkpoint = Checkpoint.load("model.ckpt")
        >>> checkpoint.assert_same_napistu_data(current_data)
        """
        # Get checkpoint data summary (already validated by Pydantic)
        checkpoint_summary = self.get_data_summary()

        # Get current data summary (simplified, matching checkpoint format)
        current_summary = napistu_data.get_summary(
            NAPISTU_DATA_SUMMARY_TYPES.VALIDATION
        )

        validate_same_data(
            current_summary=current_summary,
            reference_summary=checkpoint_summary,
        )

    def get_edge_encoder_config(self) -> Dict[str, Any]:
        """
        Get edge encoder configuration as dictionary.

        Returns
        -------
        Dict[str, Any]
            Edge encoder configuration dictionary
        """
        return self.edge_encoder_metadata.model_dump()

    def get_encoder_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration as dictionary.

        Returns
        -------
        Dict[str, Any]
            Encoder configuration dictionary
        """
        return self.encoder_metadata.model_dump()

    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get environment information as dictionary.

        Returns
        -------
        Dict[str, Any]
            Environment information dictionary
        """
        return self.environment_info.model_dump()

    def get_head_config(self) -> Dict[str, Any]:
        """
        Get head configuration as dictionary.

        Returns
        -------
        Dict[str, Any]
            Head configuration dictionary
        """
        return self.head_metadata.model_dump()

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get data summary as dictionary.

        Returns
        -------
        Dict[str, Any]
            Data summary dictionary
        """
        return self.data_metadata.model_dump(exclude_none=True)

    @classmethod
    def load(
        cls, checkpoint_path: Union[str, Path], map_location: str = DEVICE.CPU
    ) -> "Checkpoint":
        """
        Load and validate a checkpoint from a local file.

        Parameters
        ----------
        checkpoint_path : Union[str, Path]
            Path to the checkpoint file (.ckpt)
        map_location : str, optional
            Device to load tensors to, by default 'cpu'

        Returns
        -------
        Checkpoint
            Loaded and validated checkpoint object

        Raises
        ------
        FileNotFoundError
            If checkpoint file doesn't exist
        RuntimeError
            If checkpoint loading fails
        ValidationError
            If checkpoint structure is invalid

        Examples
        --------
        >>> checkpoint = Checkpoint.load("model.ckpt")
        >>> checkpoint = Checkpoint.load("model.ckpt", map_location="cuda:0")
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            # Load with weights_only=False for Lightning compatibility
            checkpoint_dict = torch.load(
                checkpoint_path, map_location=map_location, weights_only=False
            )

            # Validation happens in __init__ via Pydantic
            return cls(checkpoint_dict)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from {checkpoint_path}: {e}"
            ) from e

    def update_model_config(
        self, model_config: ModelConfig, inplace: bool = True
    ) -> ModelConfig:
        """
        Update a ModelConfig instance with settings from the checkpoint.

        Updates encoder configuration and optionally head configuration from the checkpoint
        metadata. This is useful when reconstructing a model from a checkpoint or when
        loading a pretrained model.

        The head is only updated if `pretrained_model_load_head` is True in the model_config.

        Parameters
        ----------
        model_config : ModelConfig
            ModelConfig instance to update
        inplace : bool, optional
            If True, modify the ModelConfig in place. If False, create a copy and return it.
            Default is True.

        Returns
        -------
        ModelConfig
            The updated ModelConfig instance. If inplace=True, returns the same object (modified in place).
            If inplace=False, returns a new ModelConfig instance with updated settings.

        Examples
        --------
        >>> checkpoint = Checkpoint.load("model.ckpt")
        >>> model_config = ModelConfig()
        >>> checkpoint.update_model_config(model_config)
        >>> # model_config now has encoder and head settings from checkpoint
        >>>
        >>> # Create a copy instead
        >>> updated_config = checkpoint.update_model_config(model_config, inplace=False)
        """
        if not inplace:
            model_config = model_config.model_copy(deep=True)

        # Update the encoder
        self._update_model_config_with_encoder(
            model_config, inplace=True
        )  # inplace is True here because we already made a copy if needed

        # Optionally update the head if pretrained_model_load_head is True
        pretrained_model_load_head = getattr(
            model_config, MODEL_CONFIG.PRETRAINED_MODEL_LOAD_HEAD
        )
        if pretrained_model_load_head:
            self._update_model_config_with_head(model_config, inplace=True)

        return None if inplace else model_config

    # private methods

    def _update_model_config_with_encoder(
        self, model_config: ModelConfig, inplace: bool = True
    ) -> Optional[ModelConfig]:
        """
        Update a ModelConfig instance with encoder configuration from checkpoint.

        Updates encoder-related fields in the ModelConfig based on the checkpoint's
        encoder metadata. This is useful when reconstructing a model from a checkpoint.

        Parameters
        ----------
        model_config : ModelConfig
            ModelConfig instance to update
        inplace : bool, optional
            If True, modify the ModelConfig in place. If False, create a copy and return it.
            Default is True.

        Returns
        -------
        ModelConfig
            The updated ModelConfig instance. If inplace=True, returns the same object.
            If inplace=False, returns a new ModelConfig instance.

        Examples
        --------
        >>> checkpoint = Checkpoint.load("model.ckpt")
        >>> model_config = ModelConfig()
        >>> checkpoint._update_model_config_with_encoder(model_config)
        >>> # model_config now has encoder settings from checkpoint
        >>>
        >>> # Create a copy instead
        >>> updated_config = checkpoint._update_model_config_with_encoder(model_config, inplace=False)
        """
        # Create a copy if not modifying in place
        if not inplace:
            model_config = model_config.model_copy(deep=True)

        # Update encoder fields from encoder_metadata
        encoder_dict = self.encoder_metadata.model_dump(exclude_none=True)
        encoder_fields = MODEL_COMPONENTS[MODEL_DEFS.ENCODER]

        # Update all encoder fields from MODEL_COMPONENTS
        for field in encoder_fields:
            if field in encoder_dict:
                setattr(model_config, field, encoder_dict[field])

        # Update edge encoder fields if edge encoder exists
        if self.edge_encoder_metadata is not None:
            edge_encoder_dict = self.edge_encoder_metadata.model_dump(exclude_none=True)
            edge_encoder_fields = MODEL_COMPONENTS[MODEL_DEFS.EDGE_ENCODER]
            model_config.use_edge_encoder = True

            for field in edge_encoder_fields:
                if field in edge_encoder_dict:
                    setattr(model_config, field, edge_encoder_dict[field])
        else:
            model_config.use_edge_encoder = False

        return None if inplace else model_config

    def _update_model_config_with_head(
        self, model_config: ModelConfig, inplace: bool = True
    ) -> Optional[ModelConfig]:
        """
        Update a ModelConfig instance with head configuration from checkpoint.

        Updates head-related fields in the ModelConfig based on the checkpoint's
        head metadata. This is useful when reconstructing a model from a checkpoint.

        Parameters
        ----------
        model_config : ModelConfig
            ModelConfig instance to update
        inplace : bool, optional
            If True, modify the ModelConfig in place. If False, create a copy and return it.
            Default is True.

        Returns
        -------
        ModelConfig
            The updated ModelConfig instance. If inplace=True, returns the same object.
            If inplace=False, returns a new ModelConfig instance.

        Examples
        --------
        >>> checkpoint = Checkpoint.load("model.ckpt")
        >>> model_config = ModelConfig()
        >>> checkpoint._update_model_config_with_head(model_config)
        >>> # model_config now has head settings from checkpoint
        >>>
        >>> # Create a copy instead
        >>> updated_config = checkpoint._update_model_config_with_head(model_config, inplace=False)
        """
        # Create a copy if not modifying in place
        if not inplace:
            model_config = model_config.model_copy(deep=True)

        # Update head fields from head_metadata
        head_dict = self.head_metadata.model_dump(exclude_none=True)
        head_fields = MODEL_COMPONENTS[MODEL_DEFS.HEAD]

        # Update all head fields from MODEL_COMPONENTS
        for field in head_fields:
            if field in head_dict:
                setattr(model_config, field, head_dict[field])

        return None if inplace else model_config

    def __repr__(self) -> str:
        """String representation of checkpoint."""
        return (
            f"Checkpoint(encoder={self.encoder_metadata.encoder}, "
            f"head={self.head_metadata.head}, "
            f"data={self.data_metadata.name})"
        )


class DataMetadata(BaseModel):
    """
    Validated metadata about the training data.

    This matches the structure saved by SetHyperparameters.
    """

    name: str = Field(..., description="Name of the NapistuData object")
    num_nodes: int = Field(..., ge=0, description="Number of nodes in the graph")
    num_edges: int = Field(..., ge=0, description="Number of edges in the graph")
    num_node_features: int = Field(..., ge=0, description="Number of node features")
    num_edge_features: int = Field(..., ge=0, description="Number of edge features")

    # Optional fields
    splitting_strategy: Optional[str] = Field(
        None, description="Data splitting strategy"
    )
    num_unique_relations: Optional[int] = Field(
        None, ge=0, description="Number of unique relation types"
    )
    num_train_edges: Optional[int] = Field(
        None, ge=0, description="Number of training edges"
    )
    num_val_edges: Optional[int] = Field(
        None, ge=0, description="Number of validation edges"
    )
    num_test_edges: Optional[int] = Field(
        None, ge=0, description="Number of test edges"
    )

    # Optional, feature metadata
    vertex_feature_names: Optional[List[str]] = Field(
        None, description="Ordered list of vertex feature names"
    )
    edge_feature_names: Optional[List[str]] = Field(
        None, description="Ordered list of edge feature names"
    )
    vertex_feature_name_aliases: Optional[Dict[str, str]] = Field(
        None, description="Mapping from aliased to canonical vertex feature names"
    )
    edge_feature_name_aliases: Optional[Dict[str, str]] = Field(
        None, description="Mapping from aliased to canonical edge feature names"
    )
    relation_type_labels: Optional[List[str]] = Field(
        None, description="Ordered list of relation type labels"
    )

    # Optional, mask hashes for detecting train/val/test split changes
    train_mask_hash: Optional[str] = Field(
        None, description="Hash of training mask for split verification"
    )
    val_mask_hash: Optional[str] = Field(
        None, description="Hash of validation mask for split verification"
    )
    test_mask_hash: Optional[str] = Field(
        None, description="Hash of test mask for split verification"
    )

    model_config = {"extra": "forbid"}


class EdgeEncoderMetadata(BaseModel):
    """
    Validated metadata about the edge encoder.

    This matches the structure from EdgeEncoder.get_summary() with to_model_config_names=True.
    """

    edge_in_channels: int = Field(..., ge=1, description="Edge feature dimension")
    edge_encoder_dim: int = Field(..., ge=1, description="Hidden layer dimension")
    edge_encoder_dropout: float = Field(
        ..., ge=0.0, le=1.0, description="Dropout probability"
    )
    edge_encoder_init_bias: Optional[float] = Field(
        None, description="Initial bias for output layer"
    )


class EncoderMetadata(BaseModel):
    """
    Validated metadata about the encoder.

    This matches the structure from MessagePassingEncoder.get_summary().
    """

    encoder: str = Field(..., description="Type of encoder (e.g., 'sage', 'gat')")
    in_channels: int = Field(..., ge=1, description="Input feature dimension")
    hidden_channels: int = Field(..., ge=1, description="Hidden layer dimension")
    num_layers: int = Field(..., ge=1, description="Number of GNN layers")

    # Optional fields
    edge_in_channels: Optional[int] = Field(
        None, ge=0, description="Edge feature dimension"
    )
    dropout: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Dropout probability"
    )

    # Encoder-specific parameters
    sage_aggregator: Optional[str] = Field(
        None, description="Aggregation method for SAGE"
    )
    graph_conv_aggregator: Optional[str] = Field(
        None, description="Aggregation method for GraphConv"
    )
    gat_heads: Optional[int] = Field(
        None, ge=1, description="Number of attention heads for GAT"
    )
    gat_concat: Optional[bool] = Field(
        None, description="Whether to concatenate attention heads in GAT"
    )

    model_config = {"extra": "forbid"}


class HeadMetadata(BaseModel):
    """
    Validated metadata about the head/decoder.

    This matches the structure from Decoder.get_summary().
    """

    head: str = Field(..., description="Type of head (e.g., 'dot_product', 'transe')")
    hidden_channels: int = Field(..., ge=1, description="Input embedding dimension")

    # Optional fields for different head types
    num_relations: Optional[int] = Field(
        None, ge=1, description="Number of relation types"
    )
    num_classes: Optional[int] = Field(
        None, ge=2, description="Number of output classes"
    )

    # Head-specific parameters
    mlp_hidden_dim: Optional[int] = Field(None, ge=1)
    mlp_num_layers: Optional[int] = Field(None, ge=1)
    mlp_dropout: Optional[float] = Field(None, ge=0.0, le=1.0)
    nc_dropout: Optional[float] = Field(None, ge=0.0, le=1.0)
    rotate_margin: Optional[float] = Field(None, gt=0.0)
    transe_margin: Optional[float] = Field(None, gt=0.0)

    model_config = {"extra": "allow"}  # Allow head-specific params


class ModelMetadata(BaseModel):
    """
    Validated metadata about the complete model.

    This matches the structure saved by ModelMetadataCallback under
    checkpoint['hyper_parameters']['model'].
    """

    encoder: EncoderMetadata = Field(..., description="Encoder configuration")
    head: HeadMetadata = Field(..., description="Head/decoder configuration")
    # Optional: edge encoder if present
    edge_encoder: Optional[EdgeEncoderMetadata] = Field(
        None, description="Edge encoder configuration"
    )

    model_config = {"extra": "forbid"}


class CheckpointHyperparameters(BaseModel):
    """
    Validated hyperparameters structure from Lightning checkpoint.

    This validates the checkpoint['hyper_parameters'] structure.
    """

    config: Dict[str, Any] = Field(
        ..., description="Training configuration as dictionary"
    )
    model: ModelMetadata = Field(..., description="Model architecture metadata")
    data: DataMetadata = Field(..., description="Training data metadata")
    environment: Optional[EnvironmentInfo] = Field(
        None, description="Python environment information for reproducibility"
    )

    model_config = {
        "extra": "allow"
    }  # Allow additional hparams like wandb config, etc.

    @classmethod
    def from_task_and_data(
        cls,
        task: Any,
        napistu_data: NapistuData,
        training_config: Optional[TrainingConfig] = None,
        capture_environment: bool = True,
        extra_packages: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create hyperparameters dict from task and data, with validation.

        This is used by SetHyperparameters to build the hyperparameters
        dict that will be saved to the checkpoint. The dict is validated
        against the CheckpointHyperparameters schema before returning.

        Parameters
        ----------
        task : Any
            Task object with get_summary() method that returns model metadata
        napistu_data : NapistuData
            NapistuData object to extract data metadata from
        training_config : Optional[TrainingConfig], optional
            Training configuration object (usually set by Lightning's
            save_hyperparameters()), by default None
        capture_environment : bool, optional
            Whether to capture Python environment info, by default True
        extra_packages : Optional[list[str]], optional
            Additional package names to capture versions for, by default None

        Returns
        -------
        Dict[str, Any]
            Validated hyperparameters dictionary ready to be saved to checkpoint

        Raises
        ------
        ValidationError
            If the constructed hyperparameters don't match the expected schema
        AttributeError
            If task doesn't have get_summary() method

        Examples
        --------
        >>> # In SetHyperparameters
        >>> hparams_dict = CheckpointHyperparameters.from_task_and_data(
        ...     task=pl_module.task,
        ...     napistu_data=napistu_data,
        ...     training_config=pl_module.hparams.get('config'),
        ...     extra_packages=['wandb', 'numpy']
        ... )
        >>> pl_module.hparams.update(hparams_dict)
        """
        # Get model metadata from task
        if not hasattr(task, "get_summary"):
            raise AttributeError(
                f"Task object must have get_summary() method, got {type(task)}"
            )

        model_metadata = task.get_summary()
        data_metadata = napistu_data.get_summary(NAPISTU_DATA_SUMMARY_TYPES.VALIDATION)

        # Build hyperparameters dict
        hparams = {
            CHECKPOINT_HYPERPARAMETERS.MODEL: model_metadata,
            CHECKPOINT_HYPERPARAMETERS.DATA: data_metadata,
        }

        # Add environment info
        if capture_environment:
            env_info = EnvironmentInfo.from_current_env(extra_packages=extra_packages)
            hparams[CHECKPOINT_HYPERPARAMETERS.ENVIRONMENT] = env_info.model_dump()

        # Add config if provided (will be added by Lightning's save_hyperparameters)
        if training_config is not None:
            if not isinstance(training_config, TrainingConfig):
                raise ValueError(
                    f"training_config must be a TrainingConfig, got {type(training_config)}"
                )
            # Convert TrainingConfig (Pydantic model) to dict
            hparams[CHECKPOINT_HYPERPARAMETERS.CONFIG] = training_config.model_dump()

        # Validate the structure
        if training_config is not None:
            # Full validation
            cls.model_validate(hparams)
        else:
            # Partial validation - just check model and data
            # config will be added later by Lightning
            ModelMetadata.model_validate(model_metadata)
            DataMetadata.model_validate(data_metadata)

        return hparams


class CheckpointStructure(BaseModel):
    """
    Validated structure of a Lightning checkpoint dictionary.

    This ensures the checkpoint has all required fields with correct types.
    """

    state_dict: Dict[str, Any] = Field(..., description="Model state dictionary")
    hyper_parameters: CheckpointHyperparameters = Field(
        ..., description="Training metadata"
    )

    # Optional Lightning fields
    epoch: Optional[int] = Field(None, ge=0)
    global_step: Optional[int] = Field(None, ge=0)
    pytorch_lightning_version: Optional[str] = None

    model_config = {"extra": "allow"}  # Allow other Lightning fields

    @field_validator(CHECKPOINT_STRUCTURE.STATE_DICT)
    @classmethod
    def validate_state_dict_not_empty(cls, v):
        """Ensure state_dict is not empty."""
        if not v:
            raise ValueError("state_dict cannot be empty")
        return v
