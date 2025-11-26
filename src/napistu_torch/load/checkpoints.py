"""
Checkpoint loading and validation utilities.

This module provides utilities for loading and validating pretrained Napistu-Torch models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from pydantic import BaseModel, Field, field_validator

from napistu_torch.ml.constants import DEVICE
from napistu_torch.napistu_data import NapistuData
from napistu_torch.utils.environment_info import EnvironmentInfo

logger = logging.getLogger(__name__)


class DataMetadata(BaseModel):
    """
    Validated metadata about the training data.

    This matches the structure saved by ModelMetadataCallback.
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

    model_config = {"extra": "forbid"}


class EncoderMetadata(BaseModel):
    """
    Validated metadata about the encoder.

    This matches the structure from MessagePassingEncoder.get_summary().
    """

    encoder_type: str = Field(..., description="Type of encoder (e.g., 'sage', 'gat')")
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


class HeadMetadata(BaseModel):
    """
    Validated metadata about the head/decoder.

    This matches the structure from Decoder.get_summary().
    """

    head_type: str = Field(
        ..., description="Type of head (e.g., 'dot_product', 'transe')"
    )
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
    bilinear_bias: Optional[bool] = None
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

    config: Any = Field(..., description="Training configuration (ExperimentConfig)")
    model: ModelMetadata = Field(..., description="Model architecture metadata")
    data: DataMetadata = Field(..., description="Training data metadata")
    environment: EnvironmentInfo = Field(..., description="Environment information")

    model_config = {
        "extra": "allow"
    }  # Allow additional hparams like wandb config, etc.


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

    @field_validator("state_dict")
    @classmethod
    def validate_state_dict_not_empty(cls, v):
        """Ensure state_dict is not empty."""
        if not v:
            raise ValueError("state_dict cannot be empty")
        return v


class Checkpoint:
    """
    Manager for PyTorch Lightning checkpoint loading and validation.

    This class handles loading checkpoints, extracting metadata, validating
    compatibility with current data, and reconstructing model components.

    Parameters
    ----------
    checkpoint_dict : Dict[str, Any]
        PyTorch Lightning checkpoint dictionary (validated via Pydantic)

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
        self.validated_checkpoint = CheckpointStructure.model_validate(checkpoint_dict)

        # Store original dict for state_dict access
        self.checkpoint = checkpoint_dict
        self.state_dict = checkpoint_dict["state_dict"]

        # Expose validated metadata as properties
        self.hyper_parameters = self.validated_checkpoint.hyper_parameters
        self.model_metadata = self.hyper_parameters.model
        self.data_metadata = self.hyper_parameters.data
        self.encoder_metadata = self.model_metadata.encoder
        self.head_metadata = self.model_metadata.head
        self.edge_encoder_metadata = self.model_metadata.edge_encoder

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

    def get_encoder_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration as dictionary.

        Returns
        -------
        Dict[str, Any]
            Encoder configuration dictionary
        """
        return self.encoder_metadata.model_dump()

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

    def assert_same_napistu_data(
        self, napistu_data: NapistuData, strict: bool = True
    ) -> None:
        """
        Validate that current NapistuData is compatible with checkpoint.

        Compares the data summary from the checkpoint with a summary
        generated from the provided NapistuData object.

        Parameters
        ----------
        napistu_data : NapistuData
            Current NapistuData object to validate against checkpoint
        strict : bool, optional
            If True, require exact match on all fields. If False, only
            validate critical dimensions (in_channels, edge_in_channels,
            num_relations), by default True

        Raises
        ------
        ValueError
            If data summaries don't match (strict=True) or if critical
            dimensions don't match (strict=False)

        Examples
        --------
        >>> checkpoint = Checkpoint.load("model.ckpt")
        >>> checkpoint.assert_same_napistu_data(current_data)
        >>>
        >>> # Less strict - only check critical dimensions
        >>> checkpoint.assert_same_napistu_data(current_data, strict=False)
        """
        # Get checkpoint data summary (already validated by Pydantic)
        checkpoint_summary = self.get_data_summary()

        # Get current data summary (simplified, matching checkpoint format)
        current_summary = _get_simplified_data_summary(napistu_data)

        if strict:
            # Strict mode: all fields must match
            _validate_strict_match(checkpoint_summary, current_summary)
        else:
            # Non-strict: only validate critical dimensions
            _validate_critical_dimensions(checkpoint_summary, current_summary)

    def __repr__(self) -> str:
        """String representation of checkpoint."""
        return (
            f"Checkpoint(encoder={self.encoder_metadata.encoder_type}, "
            f"head={self.head_metadata.head_type}, "
            f"data={self.data_metadata.name})"
        )


# Helper functions (can be used by ModelMetadataCallback too)


def _get_simplified_data_summary(napistu_data: NapistuData) -> Dict[str, Any]:
    """
    Get simplified summary matching checkpoint format.

    This mirrors the summary saved during training (from ModelMetadataCallback).

    Parameters
    ----------
    napistu_data : NapistuData
        NapistuData object to summarize

    Returns
    -------
    Dict[str, Any]
        Simplified summary with only essential fields
    """
    # Use the NapistuData.get_summary(simplify=True) method
    if hasattr(napistu_data, "get_summary"):
        summary = napistu_data.get_summary(simplify=True)
    else:
        # Fallback for older NapistuData without get_summary
        full_summary = napistu_data.summary()
        essential_keys = [
            "name",
            "num_nodes",
            "num_edges",
            "num_node_features",
            "num_edge_features",
            "splitting_strategy",
            "num_unique_relations",
            "num_train_edges",
            "num_val_edges",
            "num_test_edges",
        ]
        summary = {
            k: v
            for k, v in full_summary.items()
            if k in essential_keys and v is not None
        }

    return summary


def _validate_strict_match(
    checkpoint_summary: Dict[str, Any], current_summary: Dict[str, Any]
) -> None:
    """
    Validate exact match between checkpoint and current data.

    Parameters
    ----------
    checkpoint_summary : Dict[str, Any]
        Data summary from checkpoint
    current_summary : Dict[str, Any]
        Data summary from current NapistuData

    Raises
    ------
    ValueError
        If summaries don't match exactly
    """
    # Check for missing keys
    checkpoint_keys = set(checkpoint_summary.keys())
    current_keys = set(current_summary.keys())

    if checkpoint_keys != current_keys:
        missing_in_current = checkpoint_keys - current_keys
        extra_in_current = current_keys - checkpoint_keys

        msg_parts = ["Data summary mismatch:"]
        if missing_in_current:
            msg_parts.append(f"  Missing in current data: {sorted(missing_in_current)}")
        if extra_in_current:
            msg_parts.append(f"  Extra in current data: {sorted(extra_in_current)}")

        raise ValueError("\n".join(msg_parts))

    # Check for value mismatches
    mismatches = []
    for key in checkpoint_keys:
        ckpt_val = checkpoint_summary[key]
        curr_val = current_summary[key]

        if ckpt_val != curr_val:
            mismatches.append(f"  {key}: checkpoint={ckpt_val}, current={curr_val}")

    if mismatches:
        raise ValueError("Data summary values don't match:\n" + "\n".join(mismatches))

    logger.info("✓ Data validation passed (strict mode)")


def _validate_critical_dimensions(
    checkpoint_summary: Dict[str, Any], current_summary: Dict[str, Any]
) -> None:
    """
    Validate only critical dimensions required for model compatibility.

    Critical dimensions are those that affect model architecture:
    - num_node_features (in_channels)
    - num_edge_features (edge_in_channels)
    - num_unique_relations (for relation-aware heads)

    Parameters
    ----------
    checkpoint_summary : Dict[str, Any]
        Data summary from checkpoint
    current_summary : Dict[str, Any]
        Data summary from current NapistuData

    Raises
    ------
    ValueError
        If critical dimensions don't match
    """
    critical_fields = ["num_node_features", "num_edge_features", "num_unique_relations"]

    mismatches = []
    for field in critical_fields:
        # Skip if field not present in checkpoint (optional field)
        if field not in checkpoint_summary:
            continue

        ckpt_val = checkpoint_summary[field]
        curr_val = current_summary.get(field)

        if curr_val is None:
            mismatches.append(
                f"  {field}: checkpoint={ckpt_val}, current=None (missing)"
            )
        elif ckpt_val != curr_val:
            mismatches.append(f"  {field}: checkpoint={ckpt_val}, current={curr_val}")

    if mismatches:
        raise ValueError(
            "Critical dimension mismatch - cannot load pretrained model:\n"
            + "\n".join(mismatches)
            + "\n\nThe pretrained model was trained on data with different "
            "dimensions. You must either:\n"
            "1. Use data with matching dimensions, or\n"
            "2. Don't use the pretrained model"
        )

    # Log warnings for non-critical differences
    non_critical = ["num_nodes", "num_edges", "num_train_edges"]
    for field in non_critical:
        if field not in checkpoint_summary or field not in current_summary:
            continue

        ckpt_val = checkpoint_summary[field]
        curr_val = current_summary[field]

        if ckpt_val != curr_val:
            diff_pct = abs(curr_val - ckpt_val) / ckpt_val * 100
            logger.warning(
                f"⚠ {field} differs: checkpoint={ckpt_val:,}, "
                f"current={curr_val:,} ({diff_pct:.1f}% difference)"
            )

    logger.info("✓ Critical dimension validation passed")
