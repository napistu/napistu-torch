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
from torch import Tensor

from napistu_torch.load.constants import FM_DEFS
from napistu_torch.utils.base_utils import normalize_and_validate_indices
from napistu_torch.utils.tensor_utils import (
    compute_cosine_distances_torch,
    compute_max_abs_across_layers,
    compute_spearman_correlation_torch,
    find_top_k,
)
from napistu_torch.utils.torch_utils import (
    empty_cache,
    ensure_device,
    memory_manager,
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
    compute_attention_pattern(embeddings, n_heads, apply_softmax=True, device='cpu')
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
        apply_softmax: bool = True,
        return_tensor: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Union[Tensor, np.ndarray]:
        """
        Compute attention pattern for this layer with proper multi-head handling.

        Uses incremental averaging to minimize memory usage.

        Parameters
        ----------
        embeddings : np.ndarray
            Gene embeddings of shape (n_genes, d_model)
        n_heads : int
            Number of attention heads
        return_tensor : bool, optional
            If True, return the attention scores as tensor instead of a numpy array
        device : str or torch.device, optional
            Device to perform computation on (default: 'cpu')
        apply_softmax : bool, optional
            If True, apply softmax to get attention probabilities (default: True).
            If False, return raw attention scores (Q @ K.T / sqrt(d_k))

        Returns
        -------
        torch.Tensor or np.ndarray
            Averaged attention matrix of shape (n_genes, n_genes).
            If apply_softmax=True, each row sums to 1 (probabilities).
            If apply_softmax=False, raw scores (unbounded).
        """

        device = ensure_device(device, allow_autoselect=True)

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
        Wq_heads = Wq.reshape(n_heads, d_k, d_model)
        Wk_heads = Wk.reshape(n_heads, d_k, d_model)

        # Initialize accumulator for average (stays on device)
        avg_attention = None

        # Compute per-head attention and accumulate
        for h in range(n_heads):
            # Project embeddings for this head
            Q = emb @ Wq_heads[h].T  # (n_genes, d_k)
            K = emb @ Wk_heads[h].T  # (n_genes, d_k)

            # Scaled dot-product attention
            attn_scores = (Q @ K.T) / torch.sqrt(
                torch.tensor(d_k, dtype=torch.float32, device=device)
            )

            # Optionally apply softmax
            if apply_softmax:
                attn = torch.softmax(attn_scores, dim=-1)  # (n_genes, n_genes)
            else:
                attn = attn_scores

            # Accumulate running average
            if avg_attention is None:
                avg_attention = attn / n_heads
            else:
                avg_attention += attn / n_heads

            # Explicitly clean up intermediate tensors
            del Q, K, attn_scores, attn

            # Clear cache if using MPS or CUDA
            empty_cache(device)

        if return_tensor:
            return avg_attention
        else:
            return avg_attention.cpu().numpy()


class FoundationModelWeights(BaseModel):
    """Weight matrices from a foundation model.

    Attributes
    ----------
    gene_embedding : np.ndarray
        Gene embedding matrix of shape (n_vocab, embed_dim)
    attention_layers : List[AttentionLayer]
        List of attention layers, one per transformer layer

    Public Methods
    --------------
    compute_attention_from_weights(layer_idx, vocab_mask=None, apply_softmax=True, device='cpu')
        Compute attention scores for a specific layer with optional vocabulary mask.
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
        vocab_mask: Optional[np.ndarray] = None,
        apply_softmax: bool = True,
        return_tensor: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Union[Tensor, np.ndarray]:
        """
        Compute attention scores for a specific layer with proper multi-head handling.

        Parameters
        ----------
        layer_idx : int
            Index of the layer to compute attention for
        n_heads : int
            Number of attention heads in the model
        vocab_mask : np.ndarray, optional
            Boolean mask of shape (n_vocab,) indicating which vocabulary items to include.
            If provided, only embeddings corresponding to True values will be used.
            Default: None.
        apply_softmax : bool, optional
            If True, apply softmax to get attention probabilities (default: True).
            If False, return raw attention scores (Q @ K.T / sqrt(d_k))
        return_tensor : bool, optional
            If True, return attention as torch.Tensor (default: False).
            If False, return as numpy array.
        device : str or torch.device, optional
            Device to perform computation on (default: None, to automatically select a device)

        Returns
        -------
        torch.Tensor or np.ndarray
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
            apply_softmax=apply_softmax,
            return_tensor=return_tensor,
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
    compute_reordered_attention(layer_idx, target_ids, gene_annotation_target_var='ensembl_gene', apply_softmax=True, device='cpu')
        Compute attention scores for a specific layer and reorder to match a target gene ordering
    full_name
        Property returning full unique identifier (model_name with model_variant if present)
    load(output_dir, prefix)
        Load foundation model from saved files (classmethod)
    save(output_dir, prefix)
        Save foundation model to files.

    Private Methods
    --------------
    _compute_attention(layer_idx, device='cpu', vocab_mask=None)
        Compute attention scores for a specific layer with optional vocabulary mask
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

    def compute_max_attention(
        self,
        target_ids: List[str],
        gene_annotation_target_var: str = ONTOLOGIES.ENSEMBL_GENE,
        apply_softmax: bool = False,
        return_layer_indices: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Compute maximum absolute attention across all layers for target genes.

        For each gene pair, finds the layer with the strongest attention (by absolute
        value) and returns that attention value with its original sign preserved.
        This identifies the most significant attention relationships across the model.

        Parameters
        ----------
        target_ids : List[str]
            Ordered list of gene identifiers to compute attention for
        gene_annotation_target_var : str, optional
            Column name in gene_annotations to match against target_ids
            (default: ONTOLOGIES.ENSEMBL_GENE)
        apply_softmax : bool, optional
            If True, apply softmax to get attention probabilities (default: False).
            If False, return raw attention scores (Q @ K.T / sqrt(d_k))
        return_layer_indices : bool, optional
            If True, also return which layer had max attention for each gene pair
            (default: False)
        device : str or torch.device, optional
            Device to perform computation on (default: None to automatically select)

        Returns
        -------
        torch.Tensor
            Maximum absolute attention with sign preserved, shape (len(target_ids), len(target_ids))
            where result[i, j] is the strongest attention from target_ids[i] to target_ids[j]
            across all layers
        torch.Tensor (optional)
            If return_layer_indices=True, also returns layer indices where max occurred,
            shape (len(target_ids), len(target_ids))

        Examples
        --------
        >>> # Find strongest attention relationships across all layers
        >>> common_genes = ['ENSG00000000003', 'ENSG00000000005', ...]
        >>> max_attn = model.compute_max_attention(common_genes)
        >>> # Identify which layer had the strongest attention
        >>> max_attn, layer_idx = model.compute_max_attention(common_genes, return_layer_indices=True)
        >>> # Compare top attention across models
        >>> top_attn1 = model1.compute_max_attention(common_genes)
        >>> top_attn2 = model2.compute_max_attention(common_genes)
        """
        # Pre-allocate 3D tensor: (n_layers, n_genes, n_genes)
        n_genes = len(target_ids)
        all_attention = torch.zeros(
            (self.n_layers, n_genes, n_genes), dtype=torch.float32
        )

        # Fill in layer by layer
        for layer_idx in range(self.n_layers):
            attention = self.compute_reordered_attention(
                layer_idx=layer_idx,
                target_ids=target_ids,
                gene_annotation_target_var=gene_annotation_target_var,
                apply_softmax=apply_softmax,
                return_tensor=True,
                device=device,
            )
            all_attention[layer_idx] = attention

        # Find maximum absolute values across layers, preserving sign
        return compute_max_abs_across_layers(
            all_attention, return_indices=return_layer_indices
        )

    def compute_reordered_attention(
        self,
        layer_idx: int,
        target_ids: List[str],
        gene_annotation_target_var: str = ONTOLOGIES.ENSEMBL_GENE,
        apply_softmax: bool = True,
        return_tensor: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Union[Tensor, np.ndarray]:
        """
        Compute attention scores reordered to match a target gene ordering.

        This method computes attention for genes in target_ids and reorders the
        resulting attention matrix to match the order of target_ids. This enables
        direct comparison of attention matrices across different models and layers.

        Parameters
        ----------
        layer_idx : int
            Index of the layer to compute attention for
        target_ids : List[str]
            Ordered list of gene identifiers to compute attention for.
            The output attention matrix will be ordered to match this list.
        gene_annotation_target_var : str, optional
            Column name in gene_annotations to match against target_ids
            (default: ONTOLOGIES.ENSEMBL_GENE)
        apply_softmax : bool, optional
            If True, apply softmax to get attention probabilities (default: True).
            If False, return raw attention scores (Q @ K.T / sqrt(d_k))
        return_tensor : bool, optional
            If True, return attention as torch.Tensor (default: False).
            If False, return as numpy array.
        device : str or torch.device, optional
            Device to perform computation on (default: None to automatically select)

        Returns
        -------
        Tensor or np.ndarray
            Attention scores matrix of shape (len(target_ids), len(target_ids))
            where reordered_attention[i, j] represents attention from target_ids[i]
            to target_ids[j]. Softmax is applied.

        Raises
        ------
        ValueError
            If layer_idx is out of range
            If gene_annotation_target_var is not a column in gene_annotations
            If any target_ids are not found in gene_annotations

        Examples
        --------
        >>> # Compare attention across models for same genes
        >>> common_genes = ['ENSG00000000003', 'ENSG00000000005', ...]
        >>> attn1 = model1.compute_attention_reordered(0, common_genes)
        >>> attn2 = model2.compute_attention_reordered(0, common_genes)
        >>> correlation = np.corrcoef(attn1.flatten(), attn2.flatten())[0, 1]
        """
        # Validate gene_annotation_target_var exists
        if gene_annotation_target_var not in self.gene_annotations.columns:
            raise ValueError(
                f"Column '{gene_annotation_target_var}' not found in gene_annotations. "
                f"Available columns: {list(self.gene_annotations.columns)}"
            )

        # Get gene annotations for genes in target_ids
        target_gene_annotations = self.gene_annotations.query(
            f"{gene_annotation_target_var} in @target_ids"
        ).copy()

        # Check that all target_ids were found
        found_ids = set(target_gene_annotations[gene_annotation_target_var])
        missing_ids = set(target_ids) - found_ids
        if missing_ids:
            raise ValueError(
                f"Could not find {len(missing_ids)} target_ids in gene_annotations. "
                f"First few missing: {list(missing_ids)[:5]}"
            )

        # Create vocab mask: which positions in ordered_vocabulary are in target_ids?
        target_vocab_set = set(target_gene_annotations[FM_DEFS.VOCAB_NAME])
        vocab_mask = [
            vocab_name in target_vocab_set for vocab_name in self.ordered_vocabulary
        ]

        # Compute attention for masked vocabulary
        attention = self._compute_attention(
            layer_idx=layer_idx,
            device=device,
            vocab_mask=vocab_mask,
            apply_softmax=apply_softmax,
            return_tensor=return_tensor,
        )

        # REORDERING: Map from attention matrix order to target_ids order

        # Step 1: Get vocab_names in attention matrix order (filtered ordered_vocabulary)
        attention_ordered_vocab = [
            vocab_name
            for vocab_name, mask_val in zip(self.ordered_vocabulary, vocab_mask)
            if mask_val
        ]

        # Step 2: Create lookup from vocab_name -> target identifier
        vocab_to_target = dict(
            zip(
                target_gene_annotations[FM_DEFS.VOCAB_NAME],
                target_gene_annotations[gene_annotation_target_var],
            )
        )

        # Step 3: For each position in attention matrix, find its position in target_ids
        attention_idx_to_target_idx = [
            target_ids.index(vocab_to_target[vocab_name])
            for vocab_name in attention_ordered_vocab
        ]

        # Step 4: Reorder both dimensions of attention matrix to match target_ids
        reordered_attention = attention[attention_idx_to_target_idx, :][
            :, attention_idx_to_target_idx
        ]

        return reordered_attention

    def find_top_k_attention_edges(
        self,
        k: int,
        layer_indices: Optional[List[int]] = None,
        target_ids: Optional[List[str]] = None,
        gene_annotation_target_var: str = ONTOLOGIES.ENSEMBL_GENE,
        apply_softmax: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Extract top-k strongest attention edges across all layers.

        For each layer, identifies the k gene pairs with highest absolute attention values
        and returns them as a DataFrame. Useful for network construction and identifying
        the most significant gene-gene relationships learned by the model.

        Parameters
        ----------
        k : int
            Number of top edges to extract per layer
        layer_indices : List[int], optional
            Layers to analyze. If None, uses all layers.
        target_ids : List[str], optional
            Gene identifiers to analyze. If None, uses all genes in the model.
        gene_annotation_target_var : str, optional
            Column name in gene_annotations to match against target_ids
            (default: ONTOLOGIES.ENSEMBL_GENE)
        apply_softmax : bool, optional
            If True, use softmax-normalized attention probabilities (default: False).
            If False, use raw attention scores for ranking.
        device : str or torch.device, optional
            Device to perform computation on (default: None to automatically select)
        verbose : bool, optional
            Whether to print verbose output (default: False)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - layer : int
                Layer index
            - from_idx : int
                Source gene index in target_ids
            - to_idx : int
                Target gene index in target_ids
            - from_gene : str
                Source gene identifier
            - to_gene : str
                Target gene identifier
            - attention : float
                Attention value (preserves sign if apply_softmax=False)
            Sorted by layer, then by descending absolute attention value.

        Examples
        --------
        >>> # Get top 1000 edges per layer for common genes
        >>> common_genes = ['ENSG00000000003', 'ENSG00000000005', ...]
        >>> top_edges = model.find_top_k_attention_edges(k = 1000, common_genes)
        """

        device = ensure_device(device, allow_autoselect=True)

        # Use all genes if target_ids not provided
        if target_ids is None:
            target_ids = list(
                self.gene_annotations[gene_annotation_target_var].unique()
            )

        results = []

        if layer_indices is None:
            layer_indices = list(range(self.n_layers))
        else:
            layer_indices = normalize_and_validate_indices(
                indices=layer_indices,
                max_value=self.n_layers,
                param_name="layer_indices",
            )

        with memory_manager(device):
            for layer_idx in layer_indices:
                if verbose:
                    logger.info(f"Extracting top-{k} edges from layer {layer_idx}...")

                # Get attention for this layer
                attention = self.compute_reordered_attention(
                    layer_idx=layer_idx,
                    target_ids=target_ids,
                    gene_annotation_target_var=gene_annotation_target_var,
                    apply_softmax=apply_softmax,
                    return_tensor=True,
                    device=device,
                )

                # Extract top edges
                layer_df = _find_top_k_edges_in_attention_layer(
                    attention=attention,
                    k=k,
                    layer_idx=layer_idx,
                    gene_ids=target_ids,
                )

                results.append(layer_df)

                # Clean up
                del attention
                empty_cache(device)

        # Combine all layers
        all_edges = pd.concat(results, ignore_index=True)

        if verbose:
            logger.info(
                f"Extracted {len(all_edges)} total edges across {self.n_layers} layers"
            )

        return all_edges

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

    def _compute_attention(
        self,
        layer_idx: int,
        apply_softmax: bool = True,
        vocab_mask: Optional[np.ndarray] = None,
        return_tensor: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Union[Tensor, np.ndarray]:
        """
        Compute attention scores for a specific layer using the model's n_heads.

        This is a convenience method that calls weights.compute_attention_from_weights
        with the model's n_heads attribute automatically provided.

        Parameters
        ----------
        layer_idx : int
            Index of the layer to compute attention for
        vocab_mask : np.ndarray, optional
            Boolean mask of shape (n_vocab,) indicating which vocabulary items to include.
            If provided, only embeddings corresponding to True values will be used.
            Default: None.
        apply_softmax : bool, optional
            If True, apply softmax to get attention probabilities (default: True).
            If False, return raw attention scores (Q @ K.T / sqrt(d_k))
        return_tensor : bool, optional
            If True, return attention as torch.Tensor (default: False).
            If False, return as numpy array.
        device : str or torch.device, optional
            Device to perform computation on (default: None to automatically select a device)

        Returns
        -------
        Tensor or np.ndarray
            Attention scores matrix. If vocab_mask is provided, shape is (n_selected, n_selected),
            otherwise shape is (n_vocab, n_vocab). Softmax is applied.

        Raises
        ------
        ValueError
            If layer_idx is out of range

        Examples
        --------
        >>> attention = model._compute_attention(layer_idx=0)
        >>> attention.shape
        torch.Size([15000, 15000])
        """
        return self.weights.compute_attention_from_weights(
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            apply_softmax=apply_softmax,
            vocab_mask=vocab_mask,
            return_tensor=return_tensor,
            device=device,
        )


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
    get_max_attentions()
        Compute maximum attention scores across all models for common genes.
    get_model(full_name)
        Get a specific model by its full_name attribute.
    get_top_attentions(top_k=10000, apply_softmax=False, verbose=False)
        Extract top-k attention edges across all models for common genes.
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

    def get_top_attentions(
        self,
        top_k: int = 10000,
        apply_softmax: bool = False,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Extract top-k attention edges across all models for common genes.

        For each model, identifies the k strongest attention relationships per layer
        among genes that are common across all models. This enables cross-model
        comparison of attention patterns by identifying the most significant
        gene-gene relationships learned by each model.

        Parameters
        ----------
        top_k : int, optional
            Number of top edges to extract per layer per model (default: 10000)
        apply_softmax : bool, optional
            If True, use softmax-normalized attention probabilities (default: False).
            If False, use raw attention scores for ranking.
        verbose : bool, optional
            Whether to print verbose output during computation (default: False)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - layer : int
                Layer index where attention was computed
            - from_idx : int
                Source gene index in common identifiers
            - to_idx : int
                Target gene index in common identifiers
            - from_gene : str
                Source gene identifier
            - to_gene : str
                Target gene identifier
            - attention : float
                Attention value (preserves sign if apply_softmax=False)
            - model : str
                Model name (e.g., 'scGPT', 'Geneformer')
            Sorted by model, then layer, then by descending absolute attention value.

        Examples
        --------
        >>> # Get top 1000 attention edges per layer for all models
        >>> models = FoundationModels.load_multiple('/path/to/output', ['scGPT', 'Geneformer'])
        >>> top_edges = models.get_top_attentions(top_k=1000)
        >>>
        >>> # Compare attention patterns between models
        >>> scgpt_edges = top_edges[top_edges['model'] == 'scGPT']
        >>> geneformer_edges = top_edges[top_edges['model'] == 'Geneformer']
        """
        common_ids = self.get_common_identifiers()
        n_models = len(self.models)

        top_attention_edges = list()
        for i in range(n_models):
            model = self.models[i]
            model_name = model.full_name

            logger.info(f"Computing top-k attention for {model_name}...")

            model_top_k_attention = model.find_top_k_attention_edges(
                k=top_k,
                target_ids=common_ids,
                apply_softmax=apply_softmax,
                verbose=verbose,
            ).assign(model=model_name)

            top_attention_edges.append(model_top_k_attention)

        return pd.concat(top_attention_edges, ignore_index=True)

    def get_max_attentions(
        self,
        apply_softmax: bool = False,
    ) -> Tensor:
        """
        Compute maximum attention scores across all models for common genes.

        For each model, computes the maximum absolute attention across all layers
        for genes that are common across all models. This enables cross-model
        comparison of attention patterns by identifying the strongest attention
        relationships in each model.

        Returns
        -------
        Tensor
            3D tensor of shape (n_models, n_genes, n_genes) containing maximum
            attention scores. The first dimension corresponds to each model in
            self.models, and the last two dimensions represent attention from
            gene i to gene j. Values are raw attention scores (no softmax applied).
        softmax : bool, optional
            If True, apply softmax to the attention scores (default: False).

        Examples
        --------
        >>> models = FoundationModels.load_multiple('/path/to/output', ['scGPT', 'Geneformer'])
        >>> max_attentions = models.get_max_attentions()
        >>> # Compare attention patterns between first two models
        >>> model1_attn = max_attentions[0]
        >>> model2_attn = max_attentions[1]
        >>> correlation = np.corrcoef(model1_attn.flatten(), model2_attn.flatten())[0, 1]
        """
        common_ids = self.get_common_identifiers()
        n_genes = len(common_ids)
        n_models = len(self.models)

        cross_model_attention = torch.zeros(
            (n_models, n_genes, n_genes), dtype=torch.float32
        )

        for i in range(n_models):
            model = self.models[i]
            logger.info(f"Computing max-attention for {model.full_name}...")

            attention = model.compute_max_attention(
                target_ids=common_ids,
                apply_softmax=apply_softmax,
            )

            cross_model_attention[i] = attention

        return cross_model_attention

    def get_model(self, full_name: str) -> FoundationModel:
        """
        Get a specific model by its full_name attribute.

        Parameters
        ----------
        full_name : str
            The full_name of the model to retrieve (e.g., "scGPT", "Geneformer_v1")

        Returns
        -------
        FoundationModel
            The FoundationModel instance with matching full_name

        Raises
        ------
        ValueError
            If no model with the given full_name is found

        Examples
        --------
        >>> models = FoundationModels.load_multiple('/path/to/output', ['scGPT', 'Geneformer'])
        >>> scgpt_model = models.get_model("scGPT")
        >>> geneformer_model = models.get_model("Geneformer")
        """
        for model in self.models:
            if model.full_name == full_name:
                return model

        available_models = ", ".join(self.model_names)
        raise ValueError(
            f"Model '{full_name}' not found. Available models: {available_models}"
        )

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

    device = ensure_device(device, allow_autoselect=True)

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


def _find_top_k_edges_in_attention_layer(
    attention: Tensor,
    k: int,
    layer_idx: Optional[int] = None,
    gene_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract top-k edges from an attention matrix by absolute value.

    Identifies the k gene pairs with highest absolute attention values
    and returns them as a DataFrame with gene indices and identifiers.

    Parameters
    ----------
    attention : torch.Tensor
        Attention matrix of shape (n_genes, n_genes)
    k : int
        Number of top edges to extract
    layer_idx : int, optional
        Layer index to include in output (default: None)
    gene_ids : List[str], optional
        Gene identifiers corresponding to attention matrix rows/cols.
        If provided, includes 'from_gene' and 'to_gene' columns in output.
        (default: None)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - layer : int (if layer_idx provided)
            Layer index
        - from_idx : int
            Source gene index
        - to_idx : int
            Target gene index
        - from_gene : str (if gene_ids provided)
            Source gene identifier
        - to_gene : str (if gene_ids provided)
            Target gene identifier
        - attention : float
            Attention value (preserves sign)
        Sorted by descending absolute attention value.

    Examples
    --------
    >>> # Basic usage
    >>> attention = model.compute_reordered_attention(0, common_genes, return_tensor=True)
    >>> top_edges = find_top_k_edges(attention, k=1000, layer_idx=0, gene_ids=common_genes)
    >>>
    >>> # Without gene IDs
    >>> top_edges = find_top_k_edges(attention, k=100)
    """

    # Extract top-k indices and values (stays on device)
    from_indices, to_indices, top_values = find_top_k(
        attention,
        k=k,
        by_absolute_value=True,
    )

    # Build base DataFrame with indices and attention
    df = pd.DataFrame(
        {
            "from_idx": from_indices.cpu().numpy(),
            "to_idx": to_indices.cpu().numpy(),
            "attention": top_values.cpu().numpy(),
        }
    )

    del from_indices, to_indices, top_values

    # Add layer if provided
    if layer_idx is not None:
        df["layer"] = layer_idx

    # Add gene IDs via merge if provided
    if gene_ids is not None:
        # Create lookup table mapping index to gene_id
        gene_lookup = pd.DataFrame({"idx": range(len(gene_ids)), "gene_id": gene_ids})

        # Merge for from_gene
        df = df.merge(
            gene_lookup.rename(columns={"idx": "from_idx", "gene_id": "from_gene"}),
            on="from_idx",
            how="left",
        )

        # Merge for to_gene
        df = df.merge(
            gene_lookup.rename(columns={"idx": "to_idx", "gene_id": "to_gene"}),
            on="to_idx",
            how="left",
        )

    # Order columns
    col_order = []
    if "layer" in df.columns:
        col_order.append("layer")
    col_order.extend(["from_idx", "to_idx"])
    if "from_gene" in df.columns:
        col_order.extend(["from_gene", "to_gene"])
    col_order.append("attention")

    return df[col_order]


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
