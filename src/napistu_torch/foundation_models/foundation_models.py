"""
Foundation model weights, annotations, metadata, and multi-model containers.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from napistu.ontologies.constants import ONTOLOGIES
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import Tensor

from napistu_torch.foundation_models.constants import FM_DEFS
from napistu_torch.foundation_models.gene_embeddings import (
    DatasetGeneEmbeddings,
    GeneEmbeddings,
    GeneEmbeddingsSet,
    _get_model_label,
)
from napistu_torch.utils.torch_utils import cleanup_tensors, ensure_device

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

    Public Methods
    --------------
    compute_attention_pattern(embeddings, n_heads, apply_softmax=True, return_tensor=False, device=None)
        Compute attention pattern for this layer with proper multi-head handling.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    layer_idx: int
    W_q: np.ndarray
    W_k: np.ndarray
    W_v: np.ndarray
    W_o: np.ndarray

    @field_validator(FM_DEFS.W_Q, FM_DEFS.W_K, FM_DEFS.W_V, FM_DEFS.W_O)
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

        d_model = emb.shape[1]
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
            cleanup_tensors(Q, K, attn_scores, attn)

        if return_tensor:
            return avg_attention
        else:
            out_attention = avg_attention.cpu().numpy()
            cleanup_tensors(avg_attention)
            return out_attention


class FoundationModelWeights(BaseModel):
    """
    Weight matrices from a foundation model.

    Attributes
    ----------
    static_gene_embeddings : GeneEmbeddings
        Static gene embeddings for the model vocabulary
    attention_layers : List[AttentionLayer]
        List of attention layers, one per transformer layer

    Public Methods
    --------------
    count_attention_parameters()
        Count the total number of parameters across all attention layers.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    static_gene_embeddings: GeneEmbeddings
    attention_layers: List[AttentionLayer]

    @field_validator(FM_DEFS.STATIC_GENE_EMBEDDINGS)
    def validate_static_gene_embeddings(cls, v):
        if not isinstance(v, GeneEmbeddings):
            raise ValueError("static_gene_embeddings must be a GeneEmbeddings instance")
        return v

    @field_validator(FM_DEFS.ATTENTION_LAYERS)
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
        embed_dim = self.static_gene_embeddings.embedding.shape[1]

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

    def count_attention_parameters(self) -> int:
        """
        Count the total number of parameters across all attention layers.

        Sums the parameters in W_q, W_k, W_v, and W_o matrices for each
        attention layer in the model.

        Returns
        -------
        int
            Total number of parameters in all attention layers

        Examples
        --------
        >>> n_params = model.weights.count_attention_parameters()
        >>> print(f"Total attention parameters: {n_params:,}")
        """
        total_params = 0
        for layer in self.attention_layers:
            # Count parameters in each weight matrix
            total_params += layer.W_q.size  # W_q: (embed_dim, d_k)
            total_params += layer.W_k.size  # W_k: (embed_dim, d_k)
            total_params += layer.W_v.size  # W_v: (embed_dim, d_v)
            total_params += layer.W_o.size  # W_o: (embed_dim, embed_dim)
        return total_params


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
    model_variant : Optional[str]
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
    def validate_positive_integers(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"Value must be a positive integer, got: {v}")
        return v

    @field_validator(FM_DEFS.ORDERED_VOCABULARY)
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
    """
    Complete foundation model including weights, annotations, and metadata.

    Attributes
    ----------
    dataset_expression_embeddings : Optional[DatasetGeneEmbeddings]
        Contextualized gene embeddings for 0+ datasets, keyed by dataset name.
        Each dataset contains a GeneEmbeddingsSet (one GeneEmbeddings per
        category/cell type), which may have different gene vocabularies across
        datasets (e.g., different HVG selections from scGPT).
    embed_dim : int
        Embedding dimension
    gene_annotations : pd.DataFrame
        Gene annotations with columns: vocab_name, ensembl_gene, symbol (optional)
    model_name : str
        Name of the foundation model (e.g., 'scGPT', 'AIDOCell', 'scPRINT')
    model_variant: Optional[str]
        Variant of the foundation model (e.g., 'aido_cell_3m', 'aido_cell_10m', 'aido_cell_100m')
    n_genes : int
        Number of actual genes (excluding special tokens)
    n_heads : int
        Number of attention heads per layer
    n_layers : int
        Number of transformer layers
    n_vocab : int
        Total vocabulary size (may include special tokens like <pad>, <cls>)
    ordered_vocabulary : List[str]
        Vocabulary terms in same order as embedding matrix rows
    weights : FoundationModelWeights
        Model weight matrices (embeddings and attention layers)

    Properties
    ----------
    disk_name : str
        Version of the model label which can be used for a filename.
    full_name : str
        Full unique identifier (model_name with model_variant if present).

    Public Methods
    --------------
    load(output_dir, prefix)
        Load foundation model from saved files (classmethod).
    save(output_dir, prefix)
        Save foundation model to files.
    validate_dataset_gene_embeddings(...)
        Post-load checks on dataset gene embeddings from ETL (layer coverage,
        variance, labels). Distinct from :meth:`validate_dataset_gene_embeddings_field`,
        which validates the field type at model construction.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    # Core data
    dataset_gene_embeddings: Optional[DatasetGeneEmbeddings] = None
    gene_annotations: pd.DataFrame
    weights: FoundationModelWeights

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
        dataset_gene_embeddings: Optional[DatasetGeneEmbeddings] = None,
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
        dataset_gene_embeddings : DatasetGeneEmbeddings, optional
            Contextualized gene embeddings for 0+ datasets. Each dataset is a
            GeneEmbeddingsSet containing one GeneEmbeddings per category.
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

        if dataset_gene_embeddings is not None:
            if not isinstance(dataset_gene_embeddings, DatasetGeneEmbeddings):
                raise ValueError(
                    "dataset_gene_embeddings must be DatasetGeneEmbeddings or None, "
                    f"got {type(dataset_gene_embeddings)}"
                )
        else:
            dataset_gene_embeddings = None

        # Call parent __init__ with unpacked metadata
        super().__init__(
            weights=weights,
            gene_annotations=gene_annotations_df,
            dataset_gene_embeddings=dataset_gene_embeddings,
            **metadata_dict,
        )

    @property
    def disk_name(self) -> str:
        """Get a version of the model label which can be used for a filename."""
        return _get_disk_name(self.model_name, self.model_variant)

    @property
    def full_name(self) -> str:
        """Get full unique identifier."""
        return _get_model_label(self.model_name, self.model_variant)

    @classmethod
    def load(
        cls,
        output_dir: str,
        prefix: str,
        verbose: bool = True,
    ) -> "FoundationModel":
        """
        Load foundation model from saved files.

        Parameters
        ----------
        output_dir : str
            Directory path containing the saved files
        prefix : str
            Prefix used for the saved files
        verbose : bool
            Extra reporting (default: True)

        Returns
        -------
        FoundationModel
            Loaded foundation model instance

        Examples
        --------
        >>> model = FoundationModel.load('/path/to/output', 'scGPT')
        """

        (
            weights_dict,
            gene_annotations,
            model_metadata,
            static_gene_embedding_metadata,
            dataset_gene_embeddings_metadata,
        ) = _load_results(output_dir, prefix, verbose=verbose)

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

        # Reconstruct static gene embedding as GeneEmbeddings
        if static_gene_embedding_metadata is None:
            raise ValueError(
                "Static gene embedding metadata not found in saved files. "
                "This file was saved with an older format that is no longer "
                "supported. Re-run the ETL pipeline to regenerate model outputs."
            )

        # Support both new and legacy keys for backward compatibility
        static_emb_array = weights_dict.get(FM_DEFS.STATIC_GENE_EMBEDDINGS)
        static_gene_embedding = _gene_embeddings_from_save_dict(
            embedding=static_emb_array,
            metadata=static_gene_embedding_metadata,
            fallback_metadata=model_metadata,
        )

        weights = FoundationModelWeights(
            static_gene_embeddings=static_gene_embedding,
            attention_layers=attention_layers,
        )

        # Reconstruct dataset gene embeddings
        dataset_gene_embeddings = None
        if dataset_gene_embeddings_metadata:
            if verbose:
                logger.info(
                    f"Loading {len(dataset_gene_embeddings_metadata)} dataset gene embeddings"
                )

            dataset_gene_embeddings_lists: Dict[str, List[GeneEmbeddings]] = {}

            for i, ge_emb_meta in enumerate(dataset_gene_embeddings_metadata):
                embeddings_key = f"dataset_gene_embeddings_{i}"
                if embeddings_key not in weights_dict:
                    logger.warning(
                        f"Expression embeddings metadata found but embeddings tensor "
                        f"'{embeddings_key}' not found in weights file"
                    )
                    continue

                ge = _gene_embeddings_from_save_dict(
                    embedding=weights_dict[embeddings_key],
                    metadata=ge_emb_meta,
                    fallback_metadata=model_metadata,
                )

                ds_name = ge.dataset_name or "unknown"
                if ds_name not in dataset_gene_embeddings_lists:
                    dataset_gene_embeddings_lists[ds_name] = []
                dataset_gene_embeddings_lists[ds_name].append(ge)

            # Build GeneEmbeddingsSet per dataset, then wrap in DatasetGeneEmbeddings
            if dataset_gene_embeddings_lists:
                dataset_sets: Dict[str, GeneEmbeddingsSet] = {}
                for ds_name, ge_list in dataset_gene_embeddings_lists.items():
                    dataset_sets[ds_name] = GeneEmbeddingsSet.from_gene_embeddings(
                        ge_list, verbose=verbose
                    )
                dataset_gene_embeddings = DatasetGeneEmbeddings(dataset_sets)

        return cls(
            weights=weights,
            gene_annotations=gene_annotations,
            model_metadata=model_metadata,
            dataset_gene_embeddings=dataset_gene_embeddings,
        )

    def save(self, store_or_dir: Union["FoundationModelStore", str, Path]) -> None:
        """Save model weights and metadata to a model directory.

        Creates:
            {model_dir}/weights.npz
            {model_dir}/metadata.json

        Residual streams are saved separately via save_category_residuals().

        Parameters
        ----------
        store_or_dir : FoundationModelStore, str, or Path
            Either an existing FoundationModelStore or a path to the model
            directory (created if it does not exist).
        """
        store = FoundationModelStore.ensure(store_or_dir)
        store.initialize()

        logger.info(f"Saving weights to {store.weights_path}")
        logger.info(f"Saving metadata to {store.metadata_path}")

        attention_weights_dict = {
            FM_DEFS.LAYER_NAME_TEMPLATE.format(layer_idx=layer.layer_idx): {
                FM_DEFS.W_Q: layer.W_q,
                FM_DEFS.W_K: layer.W_k,
                FM_DEFS.W_V: layer.W_v,
                FM_DEFS.W_O: layer.W_o,
            }
            for layer in self.weights.attention_layers
        }

        static_ge_meta = _gene_embeddings_to_save_dict(
            self.weights.static_gene_embeddings
        )

        weights_dict = {
            FM_DEFS.STATIC_GENE_EMBEDDINGS: self.weights.static_gene_embeddings.embedding,
            FM_DEFS.ATTENTION_WEIGHTS: attention_weights_dict,
        }

        np.savez(store.weights_path, **weights_dict)

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

        combined_metadata = {
            FM_DEFS.MODEL_METADATA: model_metadata,
            FM_DEFS.GENE_ANNOTATIONS: self.gene_annotations.to_dict("records"),
            FM_DEFS.STATIC_GENE_EMBEDDINGS: static_ge_meta,
        }

        with open(store.metadata_path, "w") as f:
            json.dump(combined_metadata, f, indent=2)

        logger.info("Successfully saved weights and metadata")

    def save_category_residuals(
        self,
        store_or_dir: Union["FoundationModelStore", str, Path],
        dataset_name: str,
        category: str,
    ) -> None:
        """Save per-layer residual stream embeddings for one category.

        Creates:
            {model_dir}/residuals/{stem}.npz
            {model_dir}/residuals/{stem}_metadata.json
        and updates:
            {model_dir}/residuals_index.yaml

        Parameters
        ----------
        store_or_dir : FoundationModelStore, str, or Path
            Either an existing FoundationModelStore or a path to the model
            directory. The store must already be initialized (i.e., save()
            called first).
        dataset_name : str
            Dataset name (e.g., 'efthymiou2025').
        category : str
            Category name (e.g., 'adipocyte (0)'). May contain spaces and
            special characters — sanitized for the filename, preserved
            exactly in the index.

        Raises
        ------
        ValueError
            If dataset_gene_embeddings is None, the dataset is not found,
            or the category is not found in the dataset.
        """
        if self.dataset_gene_embeddings is None:
            raise ValueError(
                f"Model '{self.full_name}' has no dataset_gene_embeddings. "
                f"Cannot save residuals."
            )
        if dataset_name not in self.dataset_gene_embeddings:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in model '{self.full_name}'. "
                f"Available: {self.dataset_gene_embeddings.dataset_names}"
            )

        ge_set = self.dataset_gene_embeddings[dataset_name]
        category_embeddings = sorted(
            [ge for ge in ge_set.values() if ge.category == category],
            key=lambda ge: ge.layer_idx,
        )

        if not category_embeddings:
            available = sorted(
                {ge.category for ge in ge_set.values() if ge.category is not None}
            )
            raise ValueError(
                f"Category '{category}' not found in dataset '{dataset_name}' "
                f"for model '{self.full_name}'. Available: {available}"
            )

        stem = f"{_sanitize_filename(dataset_name)}" f"_{_sanitize_filename(category)}"

        arrays = {f"layer_{ge.layer_idx}": ge.embedding for ge in category_embeddings}
        metadata_records = [
            _gene_embeddings_to_save_dict(ge) for ge in category_embeddings
        ]

        store = FoundationModelStore.ensure(store_or_dir)
        store.save_residual_arrays(stem, arrays, metadata_records)
        store.register_category(dataset_name, category, stem)

        logger.info(
            f"Saved {len(category_embeddings)} layer residuals for "
            f"'{dataset_name}/{category}'"
        )

    def validate_dataset_gene_embeddings(
        self,
        *,
        dataset_name: Optional[str] = None,
        expected_n_layers: Optional[int] = None,
        std_tol: float = 0.0,
        spot_check_layers: Optional[Tuple[int, int]] = None,
        verbose: bool = False,
        raise_on_fail: bool = False,
    ) -> Dict[str, Any]:
        """Validate dataset gene embeddings produced by ETL (layer exports).

        Distinct from :meth:`validate_dataset_gene_embeddings_field`, which only
        checks that ``dataset_gene_embeddings`` is ``None`` or a
        ``DatasetGeneEmbeddings`` instance.

        Delegates per dataset to ``GeneEmbeddingsSet._validate_expression_embeddings``.

        When every embedding sets ``layer_idx``, checks that distinct indices match
        ``range(expected_n_layers)`` (default: ``self.n_layers``), that
        ``source_label`` contains ``layer_{idx}``, and that each matrix has
        ``std > std_tol``. If **any** embedding omits ``layer_idx`` while others set it,
        validation fails (mixed mode). If **all** omit ``layer_idx``, validation fails:
        layer-wise residual exports are required.

        With ``verbose=True``, logs notebook-style summaries.

        Parameters
        ----------
        dataset_name : str, optional
            Single dataset to check. When ``None``, checks every dataset.
        expected_n_layers : int, optional
            Expected number of transformer layers (default: ``self.n_layers``).
        std_tol : float
            Minimum acceptable ``embedding.std()`` per embedding (default: ``0``).
        spot_check_layers : tuple of int, optional
            Layers ``(i, j)`` for an informational mean comparison. Default picks
            ``(0, expected_n_layers - 1)`` when ``expected_n_layers > 1``.
        verbose : bool
            Log per-embedding details (default: ``False``).
        raise_on_fail : bool
            If True, raise ``ValueError`` when any check fails (default: ``False``).

        Returns
        -------
        dict
            ``{"ok": bool, "datasets": [ {...}, ... ]}`` where each dataset row has
            ``dataset_name``, ``ok``, ``errors`` (list of str), counts,
            ``distinct_layer_indices``, and ``spot_check_note``.

        Raises
        ------
        KeyError
            If ``dataset_name`` is not present.
        ValueError
            If ``raise_on_fail`` is True and a check fails.
        """
        exp_n = _validate_expected_layer_count(expected_n_layers, self.n_layers)

        missing_row = DatasetGeneEmbeddings._validate_dataset_gene_embeddings_present(
            self.dataset_gene_embeddings
        )
        if missing_row is not None:
            out: Dict[str, Any] = {"ok": False, "datasets": [missing_row]}
            if raise_on_fail:
                _raise_validation_failures(out["datasets"])
            return out

        pairs = self.dataset_gene_embeddings._validate_resolve_dataset_slice(
            dataset_name
        )
        datasets = [
            ge_set._validate_expression_embeddings(
                dataset_name=nm,
                expected_n_layers=exp_n,
                std_tol=std_tol,
                spot_check_layers=spot_check_layers,
                verbose=verbose,
            )
            for nm, ge_set in pairs
        ]
        result = {"ok": all(d["ok"] for d in datasets), "datasets": datasets}
        if raise_on_fail and not result["ok"]:
            _raise_validation_failures(datasets)
        return result

    # pydantic validation

    @field_validator(FM_DEFS.DATASET_GENE_EMBEDDINGS)
    def validate_dataset_gene_embeddings_field(cls, v):
        if v is not None and not isinstance(v, DatasetGeneEmbeddings):
            raise ValueError(
                "dataset_gene_embeddings must be DatasetGeneEmbeddings or None, "
                f"got {type(v)}"
            )
        return v

    @field_validator(FM_DEFS.GENE_ANNOTATIONS)
    def validate_gene_annotations(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError("gene_annotations must be a pandas DataFrame")

        required_columns = [FM_DEFS.VOCAB_NAME, ONTOLOGIES.ENSEMBL_GENE]
        for col in required_columns:
            if col not in v.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

        return v

    @field_validator(FM_DEFS.ORDERED_VOCABULARY)
    def validate_ordered_vocabulary(cls, v):
        if not isinstance(v, list):
            raise ValueError("ordered_vocabulary must be a list")
        if not all(isinstance(item, str) for item in v):
            raise ValueError("ordered_vocabulary must contain only strings")
        return v

    @field_validator(
        FM_DEFS.N_GENES,
        FM_DEFS.N_VOCAB,
        FM_DEFS.EMBED_DIM,
        FM_DEFS.N_LAYERS,
        FM_DEFS.N_HEADS,
    )
    def validate_positive_integers(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"Value must be a positive integer, got: {v}")
        return v

    # dunder methods

    def __repr__(self) -> str:
        """String representation of the FoundationModel instance."""
        return (
            f"FoundationModel("
            f"name={self.full_name}, "
            f"n_genes={self.n_genes}, "
            f"n_layers={self.n_layers}, "
            f"embed_dim={self.embed_dim}, "
            f"n_heads={self.n_heads}"
            f")"
        )


class FoundationModels(BaseModel):
    """Container for one or more foundation models with cross-model analysis when applicable.

    This class manages ``FoundationModel`` instances and provides methods for
    comparisons and alignment operations across models (when multiple are present).

    Attributes
    ----------
    models : List[FoundationModel]
        Non-empty list of foundation model instances

    Properties
    ----------
    model_names : List[str]
        List of model names.

    Public Methods
    --------------
    get_common_identifiers(ontology='ensembl_gene', verbose=True)
        Get common identifiers across all models.
    get_model(full_name)
        Get a specific model by its full_name attribute.
    get_summary()
        Get a summary of model metadata.
    load_multiple(output_dir, prefixes)
        Load multiple foundation models from saved files (classmethod).

    Private Methods
    --------------
    _align_embeddings(common_identifiers, ontology='ensembl_gene', verbose=False)
        Align gene embeddings across all models based on common identifiers.
    _sort_models_by_parameters()
        Sort models by attention parameters, grouping by model_name and sorting by group max.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    models: List[FoundationModel]

    @field_validator(FM_DEFS.MODELS)
    def validate_models_list(cls, v):
        if not isinstance(v, list):
            raise ValueError("models must be a list")
        if len(v) < 1:
            raise ValueError("models must contain at least one FoundationModel")
        if not all(isinstance(model, FoundationModel) for model in v):
            raise ValueError("All elements must be FoundationModel instances")
        return v

    @property
    def model_names(self) -> List[str]:
        """Get list of model names."""
        return [model.full_name for model in self.models]

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

    def get_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "full_name": [x.full_name for x in self.models],
                "model": [x.model_name for x in self.models],
                "variant": [
                    x.model_variant if x.model_variant is not None else ""
                    for x in self.models
                ],
                "embed_dim": [x.embed_dim for x in self.models],
                "n_layers": [x.n_layers for x in self.models],
                "n_heads": [x.n_heads for x in self.models],
                "parameter_count": [
                    x.weights.count_attention_parameters() for x in self.models
                ],
            }
        )

    @classmethod
    def load_multiple(
        cls, output_dir: str, prefixes: List[str], verbose: bool = True
    ) -> "FoundationModels":
        """
        Load multiple foundation models from saved files.

        Parameters
        ----------
        output_dir : str
            Directory path containing the saved model files
        prefixes : List[str]
            List of prefixes for the models to load
        verbose : bool
            Extra reporting (default: True)

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
            FoundationModel.load(output_dir, prefix, verbose=verbose)
            for prefix in prefixes
        ]

        # Create instance and sort by parameters
        instance = cls(models=loaded_models)
        instance._sort_models_by_parameters()

        return instance

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

            # Get static gene embedding and annotations
            static_gene_embeddings = model.weights.static_gene_embeddings
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
            aligned_embedding = static_gene_embeddings.embedding[
                embedding_alignment_lookup_table["index_position"].values
            ]

            if verbose:
                logger.info(
                    f"{model.model_name}: Extracted a length {aligned_embedding.shape[1]} embedding "
                    f"for {aligned_embedding.shape[0]} common identifiers"
                )

            aligned_embeddings[model.full_name] = aligned_embedding

        return aligned_embeddings

    def _sort_models_by_parameters(self) -> None:
        """
        Sort models list by attention parameters.

        Groups models by model_name (not full_name, so variants are grouped together),
        finds the maximum number of parameters in each group, then sorts by:
        1. Group maximum (ascending)
        2. Within each group, by number of parameters (ascending)

        This results in smaller models appearing first and larger models appearing later.

        Modifies self.models in-place.

        Examples
        --------
        >>> models = FoundationModels(models=[model1, model2, model3])
        >>> models._sort_models_by_parameters()
        >>> # models.models is now sorted (smallest to largest)
        """
        # Calculate parameters for each model
        model_params = [
            model.weights.count_attention_parameters() for model in self.models
        ]

        # Group by model_name and find max in each group
        group_maxes = {}
        for model, n_params in zip(self.models, model_params):
            model_name = model.model_name
            if model_name not in group_maxes:
                group_maxes[model_name] = n_params
            else:
                group_maxes[model_name] = max(group_maxes[model_name], n_params)

        # Sort by: (group_max ascending, params ascending)
        # Larger models appear later in the list
        self.models.sort(
            key=lambda model: (
                group_maxes[model.model_name],
                model.weights.count_attention_parameters(),
            )
        )

    def __repr__(self) -> str:
        """String representation listing model names."""
        model_full_names_str = ", ".join(self.model_names)
        return f"FoundationModels(models=[{model_full_names_str}])"


class FoundationModelStore:
    """Manages the on-disk layout for one foundation model's saved artifacts.

    Owns path resolution, directory creation, and the residuals index for
    a single model directory. The index is cached in memory and written to
    disk on every mutation. Does not hold any model data itself.

    Layout
    ------
    {model_dir}/
        weights.npz
        metadata.json
        residuals_index.yaml
        residuals/
            {stem}.npz
            {stem}_metadata.json

    Parameters
    ----------
    model_dir : str or Path
        Path to the model-specific directory. Created on initialize() if
        it does not exist.

    Properties
    ----------
    index_path : Path
        Path to the residuals index yaml.
    metadata_path : Path
        Path to the metadata json.
    residuals_dir : Path
        Path to the residuals subdirectory.
    weights_path : Path
        Path to the weights npz.

    Public Methods
    --------------
    ensure(store_or_dir)
        Classmethod. Return a store, constructing one from a path if needed.
    get_stem(dataset_name, category)
        Look up the filename stem for a (dataset, category) pair.
    has_category(dataset_name, category)
        Return True if a category is registered in the index.
    initialize()
        Create the model directory and residuals subdirectory.
    is_initialized()
        Return True if the model directory exists on disk.
    list_categories(dataset_name)
        Return all categories registered for a dataset.
    list_datasets()
        Return all datasets present in the index.
    load_residual_arrays(stem)
        Load residual arrays and sidecar metadata from disk.
    register_category(dataset_name, category, stem)
        Add a (dataset, category) → stem entry and persist the index.
    residuals_metadata_path(stem)
        Return the path to the sidecar metadata json for a given stem.
    residuals_path(stem)
        Return the path to the residual npz for a given stem.
    save_residual_arrays(stem, arrays, metadata_records)
        Write residual arrays and sidecar metadata to disk.

    Examples
    --------
    >>> store = FoundationModelStore("/path/to/outputs/scGPT")
    >>> store.initialize()
    >>> fm.save(store)
    >>> fm.save_category_residuals(store, "ds1", "cluster_0")

    >>> store = FoundationModelStore.ensure(existing_store)   # no-op
    >>> store = FoundationModelStore.ensure("/path/to/scGPT") # constructs
    """

    def __init__(self, model_dir: Union[str, Path]):
        self.model_dir = Path(model_dir)
        self._index: Optional[Dict] = None

    @classmethod
    def ensure(
        cls,
        store_or_dir: Union["FoundationModelStore", str, Path],
    ) -> "FoundationModelStore":
        """Return a FoundationModelStore, constructing one if needed.

        Parameters
        ----------
        store_or_dir : FoundationModelStore, str, or Path
            Either an existing store (returned as-is) or a directory path
            from which a new store is constructed.

        Returns
        -------
        FoundationModelStore
        """
        if isinstance(store_or_dir, cls):
            return store_or_dir
        return cls(model_dir=store_or_dir)

    @property
    def index_path(self) -> Path:
        return self.model_dir / FM_DEFS.RESIDUALS_INDEX_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / FM_DEFS.METADATA_FILENAME

    @property
    def residuals_dir(self) -> Path:
        return self.model_dir / FM_DEFS.RESIDUALS_SUBDIR

    @property
    def weights_path(self) -> Path:
        return self.model_dir / FM_DEFS.WEIGHTS_FILENAME

    def residuals_metadata_path(self, stem: str) -> Path:
        return self.residuals_dir / f"{stem}_metadata.json"

    def residuals_path(self, stem: str) -> Path:
        return self.residuals_dir / f"{stem}.npz"

    def get_stem(self, dataset_name: str, category: str) -> Optional[str]:
        """Return the filename stem for a (dataset, category) pair, or None."""
        return self.index.get("datasets", {}).get(dataset_name, {}).get(category)

    def has_category(self, dataset_name: str, category: str) -> bool:
        """Return True if residuals for this (dataset, category) are registered."""
        return self.get_stem(dataset_name, category) is not None

    def initialize(self) -> None:
        """Create the model directory and residuals subdirectory if needed."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.residuals_dir.mkdir(parents=True, exist_ok=True)

    def is_initialized(self) -> bool:
        """Return True if the model directory exists on disk."""
        return self.model_dir.exists()

    def list_categories(self, dataset_name: str) -> List[str]:
        """Return all categories registered for a dataset."""
        return list(self.index.get("datasets", {}).get(dataset_name, {}).keys())

    def list_datasets(self) -> List[str]:
        """Return all datasets present in the index."""
        return list(self.index.get("datasets", {}).keys())

    def load_residual_arrays(
        self,
        stem: str,
    ) -> Tuple[Dict[str, np.ndarray], List[dict]]:
        """Load residual stream arrays and sidecar metadata from disk.

        Parameters
        ----------
        stem : str
            Sanitized filename stem.

        Returns
        -------
        arrays : Dict[str, np.ndarray]
        metadata_records : List[dict]

        Raises
        ------
        FileNotFoundError
            If the npz or sidecar metadata file does not exist.
        """
        npz_path = self.residuals_path(stem)
        meta_path = self.residuals_metadata_path(stem)

        if not npz_path.exists():
            raise FileNotFoundError(
                f"Residual arrays not found: {npz_path}. "
                f"Has save_category_residuals been called for this category?"
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Residual metadata not found: {meta_path}. "
                f"The npz exists but the sidecar is missing — the save may be incomplete."
            )

        data = np.load(npz_path, allow_pickle=True)
        arrays = {key: data[key] for key in data.files}

        with open(meta_path) as f:
            metadata_records = json.load(f)

        return arrays, metadata_records

    def register_category(
        self,
        dataset_name: str,
        category: str,
        stem: str,
    ) -> None:
        """Register a (dataset, category) → stem mapping and persist the index.

        Parameters
        ----------
        dataset_name : str
            Dataset name.
        category : str
            Category name. Preserved exactly as-is in the index.
        stem : str
            Sanitized filename stem.
        """
        if dataset_name not in self.index["datasets"]:
            self.index["datasets"][dataset_name] = {}
        self.index["datasets"][dataset_name][category] = stem
        self._persist_index()

    def save_residual_arrays(
        self,
        stem: str,
        arrays: Dict[str, np.ndarray],
        metadata_records: List[dict],
    ) -> None:
        """Write residual stream arrays and sidecar metadata to disk.

        Parameters
        ----------
        stem : str
            Sanitized filename stem.
        arrays : Dict[str, np.ndarray]
            Mapping from layer key (e.g., 'layer_0') to embedding array.
        metadata_records : List[dict]
            Per-layer metadata dicts, in layer order.
        """
        np.savez(self.residuals_path(stem), **arrays)
        with open(self.residuals_metadata_path(stem), "w") as f:
            json.dump(metadata_records, f, indent=2)
        logger.info(f"Saved residual arrays to {self.residuals_path(stem).name}")

    @property
    def index(self) -> Dict:
        """In-memory index, loaded lazily on first access."""
        if self._index is None:
            self._index = self._load_index()
        return self._index

    def _load_index(self) -> Dict:
        if self.index_path.exists():
            with open(self.index_path) as f:
                return yaml.safe_load(f) or {"datasets": {}}
        return {"datasets": {}}

    def _persist_index(self) -> None:
        with open(self.index_path, "w") as f:
            yaml.dump(self._index, f, default_flow_style=False, allow_unicode=True)

    def __repr__(self) -> str:
        initialized = self.is_initialized()
        n_datasets = len(self.list_datasets()) if initialized else 0
        return (
            f"FoundationModelStore("
            f"model_dir={self.model_dir}, "
            f"initialized={initialized}, "
            f"n_datasets={n_datasets}"
            f")"
        )


def _get_disk_name(
    model_name: str,
    model_variant: Optional[str] = None,
) -> str:
    """Get a version of the model label which can be used for a filename."""
    if model_variant is None:
        return model_name
    return f"{model_name}_{model_variant}"


def _gene_embeddings_from_save_dict(
    embedding: np.ndarray,
    metadata: dict,
    fallback_metadata: Optional[dict] = None,
) -> GeneEmbeddings:
    """Reconstruct a GeneEmbeddings from a saved embedding array and metadata dict.

    Parameters
    ----------
    embedding : np.ndarray
        The 2D embedding array
    metadata : dict
        Per-embedding metadata dict (from _gene_embeddings_to_save_dict)
    fallback_metadata : dict, optional
        Model-level metadata to use as fallback for model_name/model_variant
        if not present in per-embedding metadata

    Returns
    -------
    GeneEmbeddings
        Reconstructed instance

    Raises
    ------
    ValueError
        If gene_annotations is missing from metadata
    """
    fallback_metadata = fallback_metadata or {}

    ge_annotations = metadata.get(FM_DEFS.GENE_ANNOTATIONS)
    if ge_annotations is None:
        raise ValueError(
            "GeneEmbeddings metadata is missing gene_annotations. "
            "This file was saved with an older format that is no longer supported. "
            "Re-run the ETL pipeline to regenerate model outputs."
        )

    # Always use model-level metadata for model_name/model_variant; per-embedding
    # metadata may have been None (ETL never set it)
    model_name = fallback_metadata.get(FM_DEFS.MODEL_NAME)
    model_variant = fallback_metadata.get(FM_DEFS.MODEL_VARIANT)

    return GeneEmbeddings(
        embedding=embedding,
        ordered_gene_ids=metadata.get(FM_DEFS.ORDERED_GENES, []),
        gene_annotations=pd.DataFrame(ge_annotations),
        model_name=model_name,
        model_variant=model_variant,
        layer_idx=metadata.get(FM_DEFS.LAYER_IDX),
        dataset_name=metadata.get(FM_DEFS.DATASET_NAME),
        dataset_uri=metadata.get(FM_DEFS.DATASET_URI),
        category=metadata.get(FM_DEFS.CATEGORY),
    )


def _gene_embeddings_to_save_dict(ge: GeneEmbeddings) -> dict:
    """
    Serialize a GeneEmbeddings instance to a metadata dict.

    The embedding array itself is saved separately in the weights npz;
    this returns only the JSON-serializable metadata.

    Parameters
    ----------
    ge : GeneEmbeddings
        The gene embeddings to serialize

    Returns
    -------
    dict
        JSON-serializable metadata dictionary
    """
    return {
        FM_DEFS.ORDERED_GENES: ge.ordered_gene_ids,
        FM_DEFS.GENE_ANNOTATIONS: ge.gene_annotations.to_dict("records"),
        FM_DEFS.MODEL_NAME: ge.model_name,
        FM_DEFS.MODEL_VARIANT: ge.model_variant,
        FM_DEFS.LAYER_IDX: ge.layer_idx,
        FM_DEFS.DATASET_NAME: ge.dataset_name,
        FM_DEFS.DATASET_URI: ge.dataset_uri,
        FM_DEFS.CATEGORY: ge.category,
    }


def _load_results(
    output_dir: str,
    prefix: str,
    verbose: bool = True,
) -> Tuple[dict, pd.DataFrame, dict, Optional[dict], List[dict]]:
    """
    Load foundation model results from files.

    Parameters
    ----------
    output_dir : str
        Directory path containing the saved files
    prefix : str
        Prefix used for the saved files
    verbose : bool
        Extra reporting (default: True)

    Returns
    -------
    weights_dict : dict
        Dictionary containing static_gene_embeddings and attention_weights numpy arrays
    gene_annotations : pandas.DataFrame
        DataFrame with gene annotations
    model_metadata : dict
        Dictionary with model metadata
    static_gene_embeddings_metadata : dict
        Metadata for static gene embeddings
    dataset_gene_embeddings_metadata : List[dict]
        List of dictionaries containing dataset gene embeddings metadata
    """
    weights_filename = FM_DEFS.WEIGHTS_TEMPLATE.format(prefix=prefix)
    metadata_filename = FM_DEFS.METADATA_TEMPLATE.format(prefix=prefix)
    weights_path = os.path.join(output_dir, weights_filename)
    metadata_path = os.path.join(output_dir, metadata_filename)

    if verbose:
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

    # Load static gene embeddings metadata
    static_gene_embedding_metadata = combined_metadata.get(
        FM_DEFS.STATIC_GENE_EMBEDDINGS
    )

    # Load expression embeddings metadata (if present)
    dataset_gene_embeddings_metadata = combined_metadata.get(
        FM_DEFS.DATASET_GENE_EMBEDDINGS, []
    )

    if verbose:
        logger.info("Successfully loaded all results")

    return (
        weights_dict,
        gene_annotations,
        model_metadata,
        static_gene_embedding_metadata,
        dataset_gene_embeddings_metadata,
    )


def _sanitize_filename(s: str) -> str:
    """Sanitize a string for safe use as a filename component.

    Removes characters that are problematic in filenames, collapses
    whitespace to underscores, and strips leading/trailing separators.
    The original string is preserved in the residuals index — this is
    only used to construct the on-disk filename stem.

    Parameters
    ----------
    s : str
        Input string (e.g., a category name like 'adipocyte (0)').

    Returns
    -------
    str
        Filesystem-safe string (e.g., 'adipocyte_0').
    """
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s.strip("-_")


def _raise_validation_failures(dataset_reports: List[Dict[str, Any]]) -> None:
    """Raise ``ValueError`` summarizing failed dataset validation rows."""
    parts: List[str] = []
    for row in dataset_reports:
        ds_nm = row["dataset_name"]
        for err in row["errors"]:
            parts.append(f"[{ds_nm}] {err}")
    raise ValueError("Validation failed:\n" + "\n".join(parts))


def _validate_expected_layer_count(
    expected_n_layers: Optional[int], fallback: int
) -> int:
    """Resolve and sanity-check the transformer depth used for layer coverage."""
    exp_n = fallback if expected_n_layers is None else expected_n_layers
    if not isinstance(exp_n, int) or exp_n <= 0:
        raise ValueError(
            f"expected_n_layers must be a positive int, got {expected_n_layers!r}"
        )
    return exp_n
