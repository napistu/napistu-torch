"""
Foundation model data structures and utilities for loading virtual cell models.

This module provides Pydantic-based classes for working with foundation model weights,
embeddings, and metadata in a standardized format.
"""

import json
import logging
import os
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from napistu.constants import ONTOLOGIES
from pydantic import BaseModel, Field, field_validator, model_validator

from napistu_torch.load.constants import FM_DEFS
from napistu_torch.utils.tensor_utils import (
    compute_cosine_distances_torch,
    compute_spearman_correlation_torch,
)
from napistu_torch.utils.torch_utils import (
    ensure_device,
    memory_manager,
    select_device,
)

logger = logging.getLogger(__name__)


class AttentionLayer(BaseModel):
    """Attention weights for a single transformer layer.

    Attributes
    ----------
    layer_idx : int
        Index of this layer in the model
    W_q : np.ndarray
        Query weight matrix of shape (embed_dim, d_k)
    W_k : np.ndarray
        Key weight matrix of shape (embed_dim, d_k)
    W_v : np.ndarray
        Value weight matrix of shape (embed_dim, d_v)
    W_o : np.ndarray
        Output projection weight matrix of shape (embed_dim, embed_dim)

    Methods
    -------
    compute_attention_pattern(embeddings, n_heads, average_heads, device)
        Compute attention pattern for this layer with multi-head handling.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    layer_idx: int
    W_q: np.ndarray
    W_k: np.ndarray
    W_v: np.ndarray
    W_o: np.ndarray

    @field_validator(FM_DEFS.W_Q, FM_DEFS.W_K, FM_DEFS.W_V, FM_DEFS.W_O)
    @classmethod
    def validate_weight_matrix(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError("Weight matrix must be a numpy array")
        if v.ndim != 2:
            raise ValueError("Weight matrix must be 2-dimensional")
        return v

    def compute_attention_pattern(
        self,
        embeddings: np.ndarray,
        n_heads: int,
        average_heads: bool = True,
        return_tensor: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute attention pattern for this layer with proper multi-head handling.

        Parameters
        ----------
        embeddings : np.ndarray
            Gene embeddings of shape (n_genes, d_model)
        n_heads : int
            Number of attention heads
        average_heads : bool, optional
            If True, return averaged attention across heads (default: True).
            If False, return list of per-head attention patterns.
        return_tensor : bool, optional
            If True, return a torch.Tensor. If False, return a numpy array. (default: False)
        device : str or torch.device, optional
            Device to perform computation on (default: None to decide automatically)

        Returns
        -------
        torch.Tensor
            If average_heads=True: attention matrix of shape (n_genes, n_genes)
            If average_heads=False: list of n_heads attention matrices, each (n_genes, n_genes)

        Examples
        --------
        >>> layer = model.weights.attention_layers[0]
        >>> attention = layer.compute_attention_pattern(embeddings, n_heads=4)
        >>> attention.shape
        torch.Size([15000, 15000])
        """

        if device is None:
            device = select_device()
        else:
            device = ensure_device(device)

        with memory_manager(device):
            # Convert to tensors
            emb = torch.from_numpy(embeddings).float().to(device)
            Wq = torch.from_numpy(self.W_q).float().to(device)
            Wk = torch.from_numpy(self.W_k).float().to(device)

            n_genes, d_model = emb.shape
            d_k = d_model // n_heads

            if d_model % n_heads != 0:
                raise ValueError(
                    f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
                )

            # Split by heads (row-wise)
            # W_q is (d_model, d_model), split rows into n_heads chunks of d_k rows each
            Wq_heads = Wq.reshape(n_heads, d_k, d_model)
            Wk_heads = Wk.reshape(n_heads, d_k, d_model)

            # Compute per-head attention
            head_attentions = []

            for h in range(n_heads):
                # Project embeddings for this head
                Q = emb @ Wq_heads[h].T  # (n_genes, d_k)
                K = emb @ Wk_heads[h].T  # (n_genes, d_k)

                # Scaled dot-product attention
                attn_scores = (Q @ K.T) / torch.sqrt(
                    torch.tensor(d_k, dtype=torch.float32, device=device)
                )
                attn_probs = torch.softmax(attn_scores, dim=-1)  # (n_genes, n_genes)

                head_attentions.append(attn_probs)

            if average_heads:
                # Average across heads
                results = torch.stack(head_attentions).mean(dim=0)
            else:
                results = head_attentions

            if return_tensor:
                return results
            else:
                return results.cpu().numpy()


class FoundationModelWeights(BaseModel):
    """Weight matrices from a foundation model.

    Attributes
    ----------
    gene_embedding : np.ndarray
        Gene embedding matrix of shape (n_vocab, embed_dim)
    attention_layers : List[AttentionLayer]
        List of attention layers, one per transformer layer

    Methods
    -------
    compute_attention_from_weights(layer_idx, device='cpu')
        Compute attention scores for a specific layer.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    gene_embedding: np.ndarray
    attention_layers: List[AttentionLayer]

    @field_validator(FM_DEFS.GENE_EMBEDDING)
    @classmethod
    def validate_gene_embedding(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError("gene_embedding must be a numpy array")
        if v.ndim != 2:
            raise ValueError("gene_embedding must be 2-dimensional")
        return v

    @field_validator(FM_DEFS.ATTENTION_LAYERS)
    @classmethod
    def validate_attention_weights_structure(cls, v):
        if not isinstance(v, list):
            raise ValueError("attention_layers must be a list")

        if not all(isinstance(layer, AttentionLayer) for layer in v):
            raise ValueError(
                "All elements in attention_layers must be AttentionLayer instances"
            )

        return v

    @model_validator(mode="after")
    def validate_embedding_attention_consistency(self):
        """Validate that embedding dimensions are consistent with attention weights."""
        embed_dim = self.gene_embedding.shape[1]

        # Check that all attention weight matrices have consistent dimensions
        for layer in self.attention_layers:
            for weight_name in [FM_DEFS.W_Q, FM_DEFS.W_K, FM_DEFS.W_V, FM_DEFS.W_O]:
                weight_matrix = getattr(layer, weight_name)
                if weight_matrix.shape[0] != embed_dim:
                    raise ValueError(
                        f"Attention weight {weight_name} in layer_{layer.layer_idx} has "
                        f"inconsistent dimension: expected {embed_dim}, got {weight_matrix.shape[0]}"
                    )

        return self

    def compute_attention_from_weights(
        self,
        layer_idx: int,
        n_heads: int,
        average_heads: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        vocab_mask: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Compute attention scores for a specific layer with proper multi-head handling.

        Parameters
        ----------
        layer_idx : int
            Index of the layer to compute attention for
        n_heads : int
            Number of attention heads in the model
        average_heads : bool, optional
            If True, return averaged attention across heads (default: True)
        vocab_mask : np.ndarray, optional
            Boolean mask of shape (n_vocab,) indicating which vocabulary items to include.
            If provided, only embeddings corresponding to True values will be used.
            Default: None.
        device : str or torch.device, optional
            Device to perform computation on (default: None, to automatically select a device)

        Returns
        -------
        torch.Tensor
            Attention scores matrix. If vocab_mask is provided, shape is (n_selected, n_selected),
            otherwise shape is (n_vocab, n_vocab). Softmax is applied.

        Raises
        ------
        ValueError
            If layer_idx is out of range or vocab_mask has incorrect shape

        Examples
        --------
        >>> attention = model.weights.compute_attention_from_weights(
        ...     layer_idx=0,
        ...     n_heads=model.n_heads
        ... )
        """
        if layer_idx >= len(self.attention_layers):
            raise ValueError(
                f"Layer index {layer_idx} out of range "
                f"(model has {len(self.attention_layers)} layers)"
            )

        # Apply vocab_mask to filter embeddings if provided
        embeddings = self.gene_embedding
        if vocab_mask is not None:
            vocab_mask = np.asarray(vocab_mask, dtype=bool)
            n_vocab = self.gene_embedding.shape[0]
            if vocab_mask.shape != (n_vocab,):
                raise ValueError(
                    f"vocab_mask must have shape ({n_vocab},), got {vocab_mask.shape}"
                )
            embeddings = embeddings[vocab_mask]

        layer = self.attention_layers[layer_idx]

        return layer.compute_attention_pattern(
            embeddings=embeddings,
            n_heads=n_heads,
            average_heads=average_heads,
            device=device,
        )


class GeneAnnotations(BaseModel):
    """Gene annotations DataFrame validator.

    Attributes
    ----------
    annotations : pd.DataFrame
        DataFrame with gene annotations containing at minimum:
        - vocab_name: Gene names as they appear in the model vocabulary
        - ensembl_gene: Ensembl gene identifiers
        - symbol (optional): Gene symbols

    Public Methods
    --------------
    None
        This class has no public methods.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    annotations: pd.DataFrame

    @field_validator("annotations")
    @classmethod
    def validate_annotations_structure(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError("annotations must be a pandas DataFrame")

        # Check required columns
        required_columns = [FM_DEFS.VOCAB_NAME, ONTOLOGIES.ENSEMBL_GENE]
        for col in required_columns:
            if col not in v.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

        # Validate vocab_name column
        if not pd.api.types.is_string_dtype(v[FM_DEFS.VOCAB_NAME]):
            raise ValueError(f"Column {FM_DEFS.VOCAB_NAME} must contain strings")

        # Check for unique vocab_name values
        if v[FM_DEFS.VOCAB_NAME].duplicated().any():
            raise ValueError(f"Column {FM_DEFS.VOCAB_NAME} must contain unique values")

        # Check for missing vocab_name values
        if v[FM_DEFS.VOCAB_NAME].isna().any():
            raise ValueError(
                f"Column {FM_DEFS.VOCAB_NAME} must not contain missing values"
            )

        # Validate ensembl_gene column
        if not pd.api.types.is_string_dtype(v[ONTOLOGIES.ENSEMBL_GENE]):
            raise ValueError(f"Column {ONTOLOGIES.ENSEMBL_GENE} must contain strings")

        return v


class ModelMetadata(BaseModel):
    """Model metadata validator.

    Attributes
    ----------
    model_name : str
        Name of the foundation model (e.g., 'scGPT', 'AIDOCell', 'scPRINT')
    model_varaints: Optional[str]
        Variant of the foundation model (e.g., 'aido_cell_3m', 'aido_cell_10m', 'aido_cell_100m')
    n_genes : int
        Number of actual genes (excluding special tokens)
    n_vocab : int
        Total vocabulary size (may include special tokens like <pad>, <cls>)
    ordered_vocabulary : list
        Vocabulary terms in same order as embedding matrix rows
    embed_dim : int
        Embedding dimension
    n_layers : int
        Number of transformer layers
    n_heads : int
        Number of attention heads per layer

    Public Methods
    --------------
    None
        This class has no public methods.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    model_name: str
    model_variant: Optional[str] = None
    n_genes: int
    n_vocab: int
    ordered_vocabulary: List[str] = Field(
        ...,
        description="Vocabulary terms in same order as embedding matrix rows (index i corresponds to embedding row i)",
    )
    embed_dim: int
    n_layers: int
    n_heads: int

    @field_validator(
        FM_DEFS.N_GENES,
        FM_DEFS.N_VOCAB,
        FM_DEFS.EMBED_DIM,
        FM_DEFS.N_LAYERS,
        FM_DEFS.N_HEADS,
    )
    @classmethod
    def validate_positive_integers(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"Value must be a positive integer, got: {v}")
        return v

    @field_validator(FM_DEFS.ORDERED_VOCABULARY)
    @classmethod
    def validate_ordered_vocabulary(cls, v):
        if not isinstance(v, list):
            raise ValueError("ordered_vocabulary must be a list")
        if not all(isinstance(item, str) for item in v):
            raise ValueError("ordered_vocabulary must contain only strings")
        return v

    @model_validator(mode="after")
    def validate_vocab_gene_relationship(self):
        """Validate that n_vocab >= n_genes and matches ordered_vocabulary length"""
        if self.n_vocab < self.n_genes:
            raise ValueError(
                f"n_vocab ({self.n_vocab}) must be >= n_genes ({self.n_genes})"
            )
        if len(self.ordered_vocabulary) != self.n_vocab:
            raise ValueError(
                f"ordered_vocabulary length ({len(self.ordered_vocabulary)}) "
                f"must match n_vocab ({self.n_vocab})"
            )
        return self


class FoundationModel(BaseModel):
    """Complete foundation model including weights, annotations, and metadata.

    Attributes
    ----------
    weights : FoundationModelWeights
        Model weight matrices (embeddings and attention layers)
    gene_annotations : pd.DataFrame
        Gene annotations with columns: vocab_name, ensembl_gene, symbol (optional)
    model_name : str
        Name of the foundation model (e.g., 'scGPT', 'AIDOCell', 'scPRINT')
    model_variant: Optional[str]
        Variant of the foundation model (e.g., 'aido_cell_3m', 'aido_cell_10m', 'aido_cell_100m')
    n_genes : int
        Number of actual genes (excluding special tokens)
    n_vocab : int
        Total vocabulary size (may include special tokens like <pad>, <cls>)
    ordered_vocabulary : List[str]
        Vocabulary terms in same order as embedding matrix rows
    embed_dim : int
        Embedding dimension
    n_layers : int
        Number of transformer layers
    n_heads : int
        Number of attention heads per layer

    Public Methods
    --------------
    compute_attention(layer_idx, average_heads=True, device='cpu')
        Compute attention scores for a specific layer
    full_name
        Property returning full unique identifier (model_name with model_variant if present)
    load(output_dir, prefix)
        Load foundation model from saved files (classmethod)
    save(output_dir, prefix)
        Save foundation model to files.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    # Core data
    weights: FoundationModelWeights
    gene_annotations: pd.DataFrame

    # Metadata as direct attributes
    model_name: str
    model_variant: Optional[str] = None
    n_genes: int
    n_vocab: int
    ordered_vocabulary: List[str]
    embed_dim: int
    n_layers: int
    n_heads: int

    def __init__(
        self,
        weights: FoundationModelWeights,
        gene_annotations: Union[pd.DataFrame, GeneAnnotations],
        model_metadata: Union[Dict[str, Any], ModelMetadata],
        **kwargs,
    ):
        """
        Initialize FoundationModel from weights, annotations, and metadata.

        Parameters
        ----------
        weights : FoundationModelWeights
            Model weight matrices
        gene_annotations : pd.DataFrame or GeneAnnotations
            Gene annotations
        model_metadata : dict or ModelMetadata
            Model metadata containing model_name, n_genes, n_vocab, ordered_vocabulary,
            embed_dim, n_layers, n_heads
        **kwargs
            Additional keyword arguments (ignored, for compatibility)

        Examples
        --------
        >>> # Using validated classes
        >>> gene_annot = GeneAnnotations(annotations=df)
        >>> metadata = ModelMetadata(model_name='scGPT', n_genes=1000, ...)
        >>> model = FoundationModel(weights, gene_annot, metadata)

        >>> # Using raw data
        >>> model = FoundationModel(weights, df, metadata_dict)
        """
        # Extract DataFrame from GeneAnnotations if needed
        if isinstance(gene_annotations, GeneAnnotations):
            gene_annotations_df = gene_annotations.annotations
        else:
            # Validate it
            GeneAnnotations(annotations=gene_annotations)
            gene_annotations_df = gene_annotations

        # Extract dict from ModelMetadata if needed
        if isinstance(model_metadata, ModelMetadata):
            metadata_dict = {
                FM_DEFS.MODEL_NAME: model_metadata.model_name,
                FM_DEFS.N_GENES: model_metadata.n_genes,
                FM_DEFS.N_VOCAB: model_metadata.n_vocab,
                FM_DEFS.ORDERED_VOCABULARY: model_metadata.ordered_vocabulary,
                FM_DEFS.EMBED_DIM: model_metadata.embed_dim,
                FM_DEFS.N_LAYERS: model_metadata.n_layers,
                FM_DEFS.N_HEADS: model_metadata.n_heads,
            }
        else:
            # Validate it
            ModelMetadata(**model_metadata)
            metadata_dict = model_metadata

        # Call parent __init__ with unpacked metadata
        super().__init__(
            weights=weights, gene_annotations=gene_annotations_df, **metadata_dict
        )

    @field_validator(FM_DEFS.GENE_ANNOTATIONS)
    @classmethod
    def validate_gene_annotations(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError("gene_annotations must be a pandas DataFrame")

        required_columns = [FM_DEFS.VOCAB_NAME, ONTOLOGIES.ENSEMBL_GENE]
        for col in required_columns:
            if col not in v.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

        return v

    @field_validator(
        FM_DEFS.N_GENES,
        FM_DEFS.N_VOCAB,
        FM_DEFS.EMBED_DIM,
        FM_DEFS.N_LAYERS,
        FM_DEFS.N_HEADS,
    )
    @classmethod
    def validate_positive_integers(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"Value must be a positive integer, got: {v}")
        return v

    @field_validator(FM_DEFS.ORDERED_VOCABULARY)
    @classmethod
    def validate_ordered_vocabulary(cls, v):
        if not isinstance(v, list):
            raise ValueError("ordered_vocabulary must be a list")
        if not all(isinstance(item, str) for item in v):
            raise ValueError("ordered_vocabulary must contain only strings")
        return v

    def compute_attention(
        self,
        layer_idx: int,
        average_heads: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        vocab_mask: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Compute attention scores for a specific layer using the model's n_heads.

        This is a convenience method that calls weights.compute_attention_from_weights
        with the model's n_heads attribute automatically provided.

        Parameters
        ----------
        layer_idx : int
            Index of the layer to compute attention for
        average_heads : bool, optional
            If True, return averaged attention across heads (default: True)
        device : str or torch.device, optional
            Device to perform computation on (default: None to automatically select a device)
        vocab_mask : np.ndarray, optional
            Boolean mask of shape (n_vocab,) indicating which vocabulary items to include.
            If provided, only embeddings corresponding to True values will be used.
            Default: None.

        Returns
        -------
        torch.Tensor
            Attention scores matrix. If vocab_mask is provided, shape is (n_selected, n_selected),
            otherwise shape is (n_vocab, n_vocab). Softmax is applied.

        Raises
        ------
        ValueError
            If layer_idx is out of range

        Examples
        --------
        >>> attention = model.compute_attention(layer_idx=0)
        >>> attention.shape
        torch.Size([15000, 15000])
        """
        return self.weights.compute_attention_from_weights(
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            average_heads=average_heads,
            device=device,
            vocab_mask=vocab_mask,
        )

    @property
    def full_name(self) -> str:
        """Get full unique identifier."""
        if self.model_variant:
            return f"{self.model_name}_{self.model_variant}"
        return self.model_name

    @classmethod
    def load(cls, output_dir: str, prefix: str) -> "FoundationModel":
        """
        Load foundation model from saved files.

        Parameters
        ----------
        output_dir : str
            Directory path containing the saved files
        prefix : str
            Prefix used for the saved files

        Returns
        -------
        FoundationModel
            Loaded foundation model instance

        Examples
        --------
        >>> model = FoundationModel.load('/path/to/output', 'scGPT')
        """
        weights_dict, gene_annotations, model_metadata = _load_results(
            output_dir, prefix
        )

        # Infer model_variant from prefix if not in metadata
        if (
            FM_DEFS.MODEL_VARIANT not in model_metadata
            or model_metadata[FM_DEFS.MODEL_VARIANT] is None
        ):
            model_name = model_metadata[FM_DEFS.MODEL_NAME]
            if prefix.startswith(f"{model_name}_"):
                model_metadata[FM_DEFS.MODEL_VARIANT] = prefix[len(model_name) + 1 :]
            else:
                model_metadata[FM_DEFS.MODEL_VARIANT] = None

        # Build AttentionLayer instances from weights_dict
        attention_layers = [
            AttentionLayer(
                layer_idx=int(layer_name.split("_")[1]),
                W_q=layer_weights[FM_DEFS.W_Q],
                W_k=layer_weights[FM_DEFS.W_K],
                W_v=layer_weights[FM_DEFS.W_V],
                W_o=layer_weights[FM_DEFS.W_O],
            )
            for layer_name, layer_weights in sorted(
                weights_dict[FM_DEFS.ATTENTION_WEIGHTS].items()
            )
        ]

        weights = FoundationModelWeights(
            gene_embedding=weights_dict[FM_DEFS.GENE_EMBEDDING],
            attention_layers=attention_layers,
        )

        return cls(
            weights=weights,
            gene_annotations=gene_annotations,
            model_metadata=model_metadata,
        )

    def save(self, output_dir: str, prefix: str) -> None:
        """
        Save foundation model to files.

        Creates two files:
        - {prefix}_weights.npz: Contains gene embeddings and attention weights
        - {prefix}_metadata.json: Contains gene annotations and model metadata

        Parameters
        ----------
        output_dir : str
            Directory path to save files
        prefix : str
            Prefix for output filenames (e.g., 'scGPT', 'AIDOCell_aido_cell_100m')

        Examples
        --------
        >>> model.save('/path/to/output', 'scGPT')
        # Creates: /path/to/output/scGPT_weights.npz
        #          /path/to/output/scGPT_metadata.json
        """
        os.makedirs(output_dir, exist_ok=True)

        weights_filename = FM_DEFS.WEIGHTS_TEMPLATE.format(prefix=prefix)
        metadata_filename = FM_DEFS.METADATA_TEMPLATE.format(prefix=prefix)
        weights_path = os.path.join(output_dir, weights_filename)
        metadata_path = os.path.join(output_dir, metadata_filename)

        logger.info(f"Saving weights to {weights_path}")
        logger.info(f"Saving metadata to {metadata_path}")

        # Reconstruct weights_dict format for saving
        attention_weights_dict = {
            FM_DEFS.LAYER_NAME_TEMPLATE.format(layer_idx=layer.layer_idx): {
                FM_DEFS.W_Q: layer.W_q,
                FM_DEFS.W_K: layer.W_k,
                FM_DEFS.W_V: layer.W_v,
                FM_DEFS.W_O: layer.W_o,
            }
            for layer in self.weights.attention_layers
        }

        weights_dict = {
            FM_DEFS.GENE_EMBEDDING: self.weights.gene_embedding,
            FM_DEFS.ATTENTION_WEIGHTS: attention_weights_dict,
        }

        # Save weights to npz
        np.savez(weights_path, **weights_dict)

        # Reconstruct metadata dict
        model_metadata = {
            FM_DEFS.MODEL_NAME: self.model_name,
            FM_DEFS.MODEL_VARIANT: self.model_variant,
            FM_DEFS.N_GENES: self.n_genes,
            FM_DEFS.N_VOCAB: self.n_vocab,
            FM_DEFS.ORDERED_VOCABULARY: self.ordered_vocabulary,
            FM_DEFS.EMBED_DIM: self.embed_dim,
            FM_DEFS.N_LAYERS: self.n_layers,
            FM_DEFS.N_HEADS: self.n_heads,
        }

        # Combine gene_annotations and model_metadata into single JSON
        combined_metadata = {
            FM_DEFS.MODEL_METADATA: model_metadata,
            FM_DEFS.GENE_ANNOTATIONS: self.gene_annotations.to_dict("records"),
        }

        with open(metadata_path, "w") as f:
            json.dump(combined_metadata, f, indent=2)

        logger.info("Successfully saved all results")


class FoundationModels(BaseModel):
    """Container for multiple foundation models with cross-model analysis capabilities.

    This class manages multiple FoundationModel instances and provides methods for
    cross-model comparisons and alignment operations.

    Attributes
    ----------
    models : List[FoundationModel]
        List of foundation model instances (minimum 2 required)

    Methods
    -------
    get_common_identifiers(ontology='ensembl_gene', verbose=True)
        Get common identifiers across all models.
    get_aligned_embeddings(common_identifiers, ontology='ensembl_gene')
        Align gene embeddings across all models based on common identifiers.
    load_multiple(output_dir, prefixes)
        Load multiple foundation models from saved files (classmethod).
    model_names
        Property returning list of model names.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    models: List[FoundationModel]

    @field_validator(FM_DEFS.MODELS)
    @classmethod
    def validate_models_list(cls, v):
        if not isinstance(v, list):
            raise ValueError("models must be a list")
        if len(v) < 2:
            raise ValueError("At least 2 models are required for cross-model analysis")
        if not all(isinstance(model, FoundationModel) for model in v):
            raise ValueError("All elements must be FoundationModel instances")
        return v

    def compare_embeddings(
        self, device: Optional[Union[str, torch.device]] = None, verbose: bool = False
    ) -> Dict[str, float]:
        """
        Compare the embeddings of all models.

        Aligns gene embeddings across all models based on common identifiers and then calculates Spearman correlations of distances between all pairs of models

        Parameters
        ----------
        device : Optional[Union[str, torch.device]]
            Device to use for the computation.
        verbose : bool
            Whether to print verbose output.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping model pair names to Spearman correlation coefficients.
        """

        # Get common identifiers across all models
        common_identifiers = self.get_common_identifiers(verbose=verbose)

        # pull out and align embeddings across models
        aligned_embeddings = self._align_embeddings(common_identifiers, verbose=verbose)

        # calculate each model's gene-gene distance matrix and then Spearman correlations of
        # distances between all pairs of models
        comparisons = _calculate_embedding_correlations(
            aligned_embeddings, common_identifiers, device, verbose
        )

        return comparisons

    def get_common_identifiers(
        self, ontology: str = ONTOLOGIES.ENSEMBL_GENE, verbose: bool = True
    ) -> List[str]:
        """
        Get common identifiers across all models.

        Parameters
        ----------
        ontology : str, optional
            The ontology column to use for common identifiers (default: 'ensembl_gene').
            This should be a column in every model's gene annotations.
        verbose : bool, optional
            Extra reporting (default: True)

        Returns
        -------
        List[str]
            List of common identifiers across all models

        Raises
        ------
        ValueError
            If ontology column is missing from any model's gene annotations

        Examples
        --------
        >>> models = FoundationModels(models=[model1, model2, model3])
        >>> common_genes = models.get_common_identifiers()
        >>> common_symbols = models.get_common_identifiers(ontology='symbol')
        """
        # Get common identifiers across all models
        common_identifiers = None
        for model in self.models:
            if ontology not in model.gene_annotations.columns:
                raise ValueError(
                    f"The ontology '{ontology}' is not a column in the gene annotations "
                    f"for the {model.model_name} model"
                )

            identifiers = set(model.gene_annotations[ontology])
            if common_identifiers is None:
                common_identifiers = identifiers
            else:
                common_identifiers = common_identifiers.intersection(identifiers)

        common_identifiers = list(common_identifiers)

        if verbose:
            logger.info(
                f"Found {len(common_identifiers)} identifiers (ontology: '{ontology}') "
                f"shared across {len(self.models)} models"
            )

        return common_identifiers

    @classmethod
    def load_multiple(cls, output_dir: str, prefixes: List[str]) -> "FoundationModels":
        """
        Load multiple foundation models from saved files.

        Parameters
        ----------
        output_dir : str
            Directory path containing the saved model files
        prefixes : List[str]
            List of prefixes for the models to load

        Returns
        -------
        FoundationModels
            Container with all loaded models

        Examples
        --------
        >>> models = FoundationModels.load_multiple(
        ...     '/path/to/output',
        ...     ['scGPT', 'AIDOCell_aido_cell_100m', 'scPRINT']
        ... )
        >>> common_ids = models.get_common_identifiers()
        """
        loaded_models = [
            FoundationModel.load(output_dir, prefix) for prefix in prefixes
        ]
        return cls(models=loaded_models)

    @property
    def model_names(self) -> List[str]:
        """Get list of model names."""
        return [model.full_name for model in self.models]

    # private methods

    def _align_embeddings(
        self,
        common_identifiers: List[str],
        ontology: str = ONTOLOGIES.ENSEMBL_GENE,
        verbose: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Align gene embeddings across all models based on common identifiers.

        This function aligns gene embeddings across all models by:
        1. Adding a positional index to the gene embeddings which maps each gene to a row in the embedding matrix.
        2. Filtering and reordering the gene annotations so they match the order of the common identifiers.
        3. Using the positional index to reorder the gene embeddings.

        Parameters
        ----------
        common_identifiers : List[str]
            List of common identifiers across all models. This will define the order of rows
            in the aligned embeddings. Typically obtained from get_common_identifiers().
        ontology : str, optional
            The ontology column to use for common identifiers (default: 'ensembl_gene').
            This should be a column in every model's gene annotations.
        verbose : bool, optional
            Extra reporting (default: False)

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping model names to aligned embedding arrays.
            Each array has shape (n_common_genes, embed_dim).

        Raises
        ------
        ValueError
            If ontology column is missing from any model's gene annotations

        Examples
        --------
        >>> models = FoundationModels(models=[model1, model2])
        >>> common_ids = models.get_common_identifiers()
        >>> aligned_embeddings = models.get_aligned_embeddings(common_ids)
        >>> aligned_embeddings['scGPT'].shape
        (15000, 512)
        """
        aligned_embeddings = {}

        for model in self.models:
            # Validate ontology column exists
            if ontology not in model.gene_annotations.columns:
                raise ValueError(
                    f"The ontology '{ontology}' is not a column in the gene annotations "
                    f"for the {model.model_name} model"
                )

            # Get gene embedding and annotations
            gene_embedding = model.weights.gene_embedding
            gene_annotations = model.gene_annotations
            ordered_vocab = model.ordered_vocabulary

            # Create vocab lookup with positional indices
            vocab_df = pd.DataFrame({FM_DEFS.VOCAB_NAME: ordered_vocab}).assign(
                index_position=range(len(ordered_vocab))
            )

            # Filter to common identifiers and add the ordering in the vocab (i.e., the rows in the embedding matrix)
            embedding_alignment_lookup_table = (
                gene_annotations.set_index(ontology)
                # filter to common identifiers and reorder based on common_identifiers' ordering
                .loc[common_identifiers].merge(
                    vocab_df, on=FM_DEFS.VOCAB_NAME, how="inner"
                )
            )

            # Extract the embeddings for the common identifiers in the order of common_identifiers
            aligned_embedding = gene_embedding[
                embedding_alignment_lookup_table["index_position"].values
            ]

            if verbose:
                logger.info(
                    f"{model.model_name}: Extracted a length {aligned_embedding.shape[1]} embedding "
                    f"for {aligned_embedding.shape[0]} common identifiers"
                )

            aligned_embeddings[model.full_name] = aligned_embedding

        return aligned_embeddings

    def __repr__(self) -> str:
        """String representation listing model names."""
        model_full_names_str = ", ".join(self.model_names)
        return f"FoundationModels(models=[{model_full_names_str}])"


# Private utility functions


def _calculate_embedding_correlations(
    aligned_embeddings: Dict[str, np.ndarray],
    common_identifiers: List[str],
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compare embeddings by calculating gene-gene distances and then Spearman correlations of distances between all pairs of models

    Parameters
    ----------
    aligned_embeddings : Dict[str, np.ndarray]
        Dictionary mapping model names to aligned embedding arrays.
    common_identifiers : List[str]
        List of common identifiers across all models.
    device : Optional[Union[str, torch.device]]
        Device to use for the computation.
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping model pair names to Spearman correlation coefficients.
    """

    # Check if MPS/GPU is available
    if device is None:
        device = select_device(mps_valid=True)
        if verbose:
            logger.info(f"No device specified, using default device: {device}")
    else:
        device = ensure_device(device)

    # Convert embeddings to PyTorch tensors and compute distances with memory management
    distances = {}
    with memory_manager(device):
        for model_name, embedding in aligned_embeddings.items():
            if verbose:
                logger.info(f"Computing distances for {model_name}...")
            distances[model_name] = compute_cosine_distances_torch(embedding, device)

    # Compare distance matrices pairwise - all unique pairs from model_prefixes
    # Use upper triangle only (exclude diagonal and avoid redundancy)
    mask = np.triu_indices(len(common_identifiers), k=1)  # k=1 excludes diagonal

    all_model_names = list(aligned_embeddings.keys())
    comparisons = {}
    with memory_manager(device):
        for model1, model2 in combinations(all_model_names, 2):
            if verbose:
                logger.info(f"Comparing {model1} vs {model2}...")

            dist1_flat = distances[model1][mask]
            dist2_flat = distances[model2][mask]

            # Spearman correlation using PyTorch
            rho = compute_spearman_correlation_torch(dist1_flat, dist2_flat, device)
            comparisons[f"{model1}_vs_{model2}"] = rho

            if verbose:
                logger.info(f"  {model1} vs {model2}: Spearman rho = {rho:.4f}")

    return comparisons


def _load_results(output_dir: str, prefix: str) -> Tuple[dict, pd.DataFrame, dict]:
    """
    Load foundation model results from files.

    Parameters
    ----------
    output_dir : str
        Directory path containing the saved files
    prefix : str
        Prefix used for the saved files

    Returns
    -------
    weights_dict : dict
        Dictionary containing gene_embedding and attention_weights numpy arrays
    gene_annotations : pandas.DataFrame
        DataFrame with gene annotations
    model_metadata : dict
        Dictionary with model metadata
    """
    weights_filename = FM_DEFS.WEIGHTS_TEMPLATE.format(prefix=prefix)
    metadata_filename = FM_DEFS.METADATA_TEMPLATE.format(prefix=prefix)
    weights_path = os.path.join(output_dir, weights_filename)
    metadata_path = os.path.join(output_dir, metadata_filename)

    logger.info(
        f"Loading weights ({weights_filename}) and metadata (  {metadata_filename}) from output_dir ({output_dir})"
    )

    # Load weights from npz
    weights_data = np.load(weights_path, allow_pickle=True)
    weights_dict = {}
    for key in weights_data.keys():
        value = weights_data[key]
        # Handle numpy arrays containing objects (like dictionaries)
        if isinstance(value, np.ndarray) and value.dtype == object:
            weights_dict[key] = value.item()
        else:
            weights_dict[key] = value

    # Load metadata from JSON
    with open(metadata_path, "r") as f:
        combined_metadata = json.load(f)

    model_metadata = combined_metadata[FM_DEFS.MODEL_METADATA]
    gene_annotations = pd.DataFrame(combined_metadata[FM_DEFS.GENE_ANNOTATIONS])

    logger.info("Successfully loaded all results")

    return weights_dict, gene_annotations, model_metadata
