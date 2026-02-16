"""
Foundation model data structures and utilities for loading virtual cell models.

This module provides Pydantic-based classes for working with foundation model weights,
embeddings, and metadata in a standardized format.

Classes
-------
GeneEmbeddings
    Data and metadata for a gene x latent space embedding.
GeneEmbeddingsSet
    Container for multiple GeneEmbeddings with aligned gene ids.
DatasetGeneEmbeddings
    Container for multiple GeneEmbeddingsSets keyed by dataset name.
AttentionLayer
    Attention weights for a single transformer layer.
FoundationModelWeights
    Weight matrices from a foundation model.
GeneAnnotations
    Gene annotations DataFrame validator.
ModelMetadata
    Model metadata validator.
FoundationModel
    Complete foundation model including weights, annotations, and metadata.
FoundationModels
    Container for multiple foundation models with cross-model analysis capabilities.
"""

import json
import logging
import os
from collections import Counter
from functools import cached_property
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from napistu.ontologies.constants import ONTOLOGIES
from pydantic import BaseModel, Field, field_validator, model_validator
from torch import Tensor

from napistu_torch.load.constants import (
    FM_DEFS,
    FM_EDGELIST,
    FM_LAYER_CONSENSUS_METHODS,
    VALID_FM_LAYER_CONSENSUS_METHODS,
)
from napistu_torch.utils.base_utils import normalize_and_validate_indices
from napistu_torch.utils.constants import CORRELATION_METHODS
from napistu_torch.utils.pd_utils import calculate_ranks
from napistu_torch.utils.tensor_utils import (
    compute_correlation,
    compute_cosine_distances_torch,
    compute_max_abs_over_z,
    compute_max_over_z,
    compute_tensor_ranks,
    compute_tensor_ranks_for_indices,
    find_top_k,
)
from napistu_torch.utils.torch_utils import (
    cleanup_tensors,
    ensure_device,
    memory_manager,
)

logger = logging.getLogger(__name__)


class GeneEmbeddings(BaseModel):
    """
    Immutable container co-locating a 2D embedding matrix with gene identifiers.

    Ensures that the embedding matrix rows always correspond 1:1 with
    ordered_gene_ids and gene_annotations rows. All reordering and filtering
    operations return new instances rather than mutating state.

    Attributes
    ----------
    embedding : np.ndarray
        Gene embedding matrix of shape (n_genes, embed_dim).
    ordered_gene_ids : List[str]
        Gene identifiers in the same order as embedding rows.
        Typically Ensembl gene IDs (e.g., 'ENSG00000141510').
    gene_annotations : pd.DataFrame
        Gene annotations with rows aligned to embedding rows.
        Must contain at minimum a column matching the identifiers in
        ordered_gene_ids (default: ONTOLOGIES.ENSEMBL_GENE).
    model_name : Optional[str]
        Name of the source foundation model (e.g., 'scGPT', 'AIDOCell').
    model_variant : Optional[str]
        Variant of the source model (e.g., 'aido_cell_100m').
    dataset_name : Optional[str]
        Name of the expression dataset used to contextualize embeddings.
        None for static (non-expression-aware) gene embeddings.
    dataset_uri : Optional[str]
        Path or URI to the source dataset (e.g., '/path/to/data.h5ad').
        None for static gene embeddings.
    category : Optional[str]
        Category within the dataset (e.g., cell type, cluster).
        None for static gene embeddings or dataset-level aggregations.

    Properties
    ----------
    n_genes : int
        Number of genes.
    embed_dim : int
        Embedding dimensionality.
    gene_ids_set : FrozenSet[str]
        Frozen set of gene IDs for O(1) membership checks.
    source_label : str
        Human-readable source descriptor combining model and dataset info.

    Public Methods
    --------------
    align_to(target_ids)
        Return a new GeneEmbeddings filtered and reordered to match target_ids.
    compute_pairwise_distances(device=None)
        Compute gene-gene cosine distance matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> gene_ids = ['ENSG00000141510', 'ENSG00000157764', 'ENSG00000171862']
    >>> embedding = np.random.randn(3, 512)
    >>> annotations = pd.DataFrame({
    ...     'ensembl_gene': gene_ids,
    ...     'symbol': ['TP53', 'BRAF', 'PTEN'],
    ... })
    >>> ge = GeneEmbeddings(
    ...     embedding=embedding,
    ...     ordered_gene_ids=gene_ids,
    ...     gene_annotations=annotations,
    ...     model_name='scGPT',
    ... )
    >>> ge.n_genes
    3
    >>> ge.embed_dim
    512
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    # Core data
    embedding: np.ndarray = Field(
        ...,
        description="Gene embedding matrix of shape (n_genes, embed_dim)",
    )
    ordered_gene_ids: List[str] = Field(
        ...,
        description="Gene identifiers aligned with embedding rows",
    )
    gene_annotations: pd.DataFrame = Field(
        ...,
        description="Gene annotations with rows aligned to embedding rows",
    )

    # Source metadata
    model_name: Optional[str] = None
    model_variant: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_uri: Optional[str] = None
    category: Optional[str] = None

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: np.ndarray) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            raise ValueError("embedding must be a numpy array")
        if v.ndim != 2:
            raise ValueError(
                f"embedding must be 2-dimensional (n_genes, embed_dim), "
                f"got shape {v.shape}"
            )
        if v.shape[0] == 0:
            raise ValueError("embedding must have at least one gene (row)")
        return v

    @field_validator("ordered_gene_ids")
    @classmethod
    def validate_ordered_gene_ids(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("ordered_gene_ids must be a list")
        if len(v) == 0:
            raise ValueError("ordered_gene_ids must not be empty")
        if not all(isinstance(gid, str) for gid in v):
            raise ValueError("ordered_gene_ids must contain only strings")
        if len(v) != len(set(v)):
            raise ValueError("ordered_gene_ids must contain unique values")
        return v

    @field_validator("gene_annotations")
    @classmethod
    def validate_gene_annotations(cls, v: pd.DataFrame) -> pd.DataFrame:
        GeneAnnotations(annotations=v)
        return v

    @model_validator(mode="after")
    def validate_row_correspondence(self) -> "GeneEmbeddings":
        """Validate that embedding rows, ordered_gene_ids, and gene_annotations are aligned."""
        n_embedding_rows = self.embedding.shape[0]
        n_gene_ids = len(self.ordered_gene_ids)
        n_annotation_rows = len(self.gene_annotations)

        if n_embedding_rows != n_gene_ids:
            raise ValueError(
                f"embedding has {n_embedding_rows} rows but "
                f"ordered_gene_ids has {n_gene_ids} entries"
            )

        if n_embedding_rows != n_annotation_rows:
            raise ValueError(
                f"embedding has {n_embedding_rows} rows but "
                f"gene_annotations has {n_annotation_rows} rows"
            )

        return self

    @property
    def n_genes(self) -> int:
        """Number of genes."""
        return self.embedding.shape[0]

    @property
    def embed_dim(self) -> int:
        """Embedding dimensionality."""
        return self.embedding.shape[1]

    @cached_property
    def gene_ids_set(self) -> FrozenSet[str]:
        """Frozen set of gene IDs for O(1) membership checks."""
        return frozenset(self.ordered_gene_ids)

    @property
    def source_label(self) -> str:
        """Human-readable source descriptor."""
        parts = []
        if self.model_name:
            label = self.model_name
            if self.model_variant:
                label = f"{label}_{self.model_variant}"
            parts.append(label)
        if self.dataset_name:
            parts.append(self.dataset_name)
        if self.category:
            parts.append(self.category)
        return "/".join(parts) if parts else "unknown"

    def align_to(self, target_ids: List[str]) -> "GeneEmbeddings":
        """Return a new GeneEmbeddings filtered and reordered to match target_ids.

        Only genes present in both self and target_ids are retained.
        The output order matches target_ids.

        Parameters
        ----------
        target_ids : List[str]
            Ordered list of gene identifiers to align to.
            Genes in target_ids that are not in self are silently skipped.

        Returns
        -------
        GeneEmbeddings
            New instance with rows filtered and reordered to match the
            intersection of target_ids and self.ordered_gene_ids,
            preserving the order from target_ids.

        Raises
        ------
        ValueError
            If no genes in target_ids are found in this embedding.

        Examples
        --------
        >>> common = ge1.common_genes(ge2)
        >>> ge1_aligned = ge1.align_to(common)
        >>> ge2_aligned = ge2.align_to(common)
        """
        # Filter target_ids to those present in self
        my_ids = self.gene_ids_set
        filtered_targets = [gid for gid in target_ids if gid in my_ids]

        if len(filtered_targets) == 0:
            raise ValueError(
                "No genes in target_ids are present in this GeneEmbeddings instance"
            )

        if len(filtered_targets) < len(target_ids):
            n_missing = len(target_ids) - len(filtered_targets)
            logger.warning(
                f"{n_missing} of {len(target_ids)} target_ids not found in "
                f"this GeneEmbeddings (source: {self.source_label}); "
                f"returning {len(filtered_targets)} genes"
            )

        # Build index mapping: gene_id -> current row position
        id_to_idx = {gid: idx for idx, gid in enumerate(self.ordered_gene_ids)}
        reorder_indices = np.array([id_to_idx[gid] for gid in filtered_targets])

        return GeneEmbeddings(
            embedding=self.embedding[reorder_indices],
            ordered_gene_ids=filtered_targets,
            gene_annotations=self.gene_annotations.iloc[reorder_indices].reset_index(
                drop=True
            ),
            model_name=self.model_name,
            model_variant=self.model_variant,
            dataset_name=self.dataset_name,
            category=self.category,
        )

    def compute_pairwise_distances(
        self,
        device: Optional[Union[str, torch.device]] = None,
    ) -> np.ndarray:
        """Compute gene-gene cosine distance matrix.

        Parameters
        ----------
        device : str or torch.device, optional
            Device for computation (default: None to auto-select).

        Returns
        -------
        np.ndarray
            Symmetric distance matrix of shape (n_genes, n_genes).
            Entry [i, j] is the cosine distance between gene i and gene j.
            Values range from 0 (identical) to 2 (opposite).
        """
        return compute_cosine_distances_torch(self.embedding, device=device)

    def gene_ids_in_ontology(self, ontology: str) -> List[str]:
        """Return gene identifiers in the given ontology column, in row order."""
        if ontology not in self.gene_annotations.columns:
            raise ValueError(
                f"Ontology '{ontology}' not in gene_annotations columns: "
                f"{list(self.gene_annotations.columns)}"
            )
        return self.gene_annotations[ontology].tolist()

    def __repr__(self) -> str:
        return (
            f"GeneEmbeddings("
            f"source={self.source_label}, "
            f"n_genes={self.n_genes}, "
            f"embed_dim={self.embed_dim}"
            f")"
        )

    def __len__(self) -> int:
        return self.n_genes

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GeneEmbeddings):
            return NotImplemented
        return self.ordered_gene_ids == other.ordered_gene_ids and np.array_equal(
            self.embedding, other.embedding
        )


class GeneEmbeddingsSet:
    """Container for one or more GeneEmbeddings instances aligned to a common list of genes.

    All stored embeddings have the same genes in the same row order (in terms of
    the alignment ontology), though each may retain its own native vocabulary in
    ``ordered_gene_ids``. This enables direct cross-embedding comparison without
    repeated alignment overhead.

    When constructed with a single embedding, it serves as a validated wrapper
    that still provides the same dict-like interface and summary properties.

    Attributes
    ----------
    common_gene_ids : List[str]
        Gene identifiers (in the alignment ontology) shared across all embeddings,
        in the order used for alignment.
    align_on : str
        The ontology column used for alignment.

    Properties
    ----------
    n_embeddings : int
        Number of stored embeddings.
    n_common_genes : int
        Number of common genes.
    summary : pd.DataFrame
        One row per embedding with source metadata and dimensionality info.

    Public Methods
    --------------
    from_gene_embeddings(embeddings, align_on='ensembl_gene')
        Classmethod: align embeddings and construct container.
    get(key)
        Return the aligned GeneEmbeddings for a given key.
    keys()
        Return embedding labels.
    values()
        Return GeneEmbeddings instances.
    items()
        Return (label, GeneEmbeddings) pairs.

    Examples
    --------
    >>> aligned = GeneEmbeddingsSet.from_gene_embeddings(
    ...     [ge_scgpt, ge_aido, ge_scprint]
    ... )
    >>> aligned.n_common_genes
    15234
    >>> aligned.summary
          key     model_name model_variant  ... n_genes embed_dim
    0  scGPT          scGPT          None  ...   15234       512
    1  AIDOCell_100m  AIDOCell  aido_cell_100m  ...   15234       768
    >>> scgpt_emb = aligned["scGPT"]
    """

    def __init__(
        self,
        data: Dict[str, GeneEmbeddings],
        common_gene_ids: List[str],
        align_on: str,
    ):
        """Initialize GeneEmbeddingsSet.

        Prefer using the ``from_gene_embeddings`` classmethod which handles
        alignment automatically. This constructor assumes embeddings are
        already aligned.

        Parameters
        ----------
        data : Dict[str, GeneEmbeddings]
            Mapping from label to aligned GeneEmbeddings.
        common_gene_ids : List[str]
            The common gene IDs (in the alignment ontology) shared by all
            embeddings, in the order used for alignment.
        align_on : str
            The ontology column that was used for alignment.

        Raises
        ------
        ValueError
            If data is empty.
            If any embedding's genes in the alignment ontology don't match
            common_gene_ids.
        """
        if len(data) == 0:
            raise ValueError("GeneEmbeddingsSet requires at least one embedding")

        # Validate alignment
        for key, emb in data.items():
            emb_ids = emb.gene_ids_in_ontology(align_on)
            if emb_ids != common_gene_ids:
                # Find first mismatch for a helpful error message
                for i, (a, b) in enumerate(zip(emb_ids, common_gene_ids)):
                    if a != b:
                        raise ValueError(
                            f"Embedding '{key}' gene IDs in ontology '{align_on}' "
                            f"do not match common_gene_ids. "
                            f"First mismatch at index {i}: '{a}' != '{b}'"
                        )
                # Length mismatch
                raise ValueError(
                    f"Embedding '{key}' has {len(emb_ids)} genes but "
                    f"common_gene_ids has {len(common_gene_ids)}"
                )

        self._data: Dict[str, GeneEmbeddings] = dict(data)
        self._common_gene_ids: List[str] = list(common_gene_ids)
        self._align_on: str = align_on

    @classmethod
    def from_gene_embeddings(
        cls,
        embeddings: List[GeneEmbeddings],
        align_on: str = ONTOLOGIES.ENSEMBL_GENE,
    ) -> "GeneEmbeddingsSet":
        """Align embeddings to common genes and construct container.

        Uses ``align_gene_embeddings`` to find common genes across all
        embeddings and reorder each to a consistent row ordering. Keys
        are derived from each embedding's ``source_label``.

        Parameters
        ----------
        embeddings : List[GeneEmbeddings]
            One or more GeneEmbeddings to align. Each must have unique
            ``source_label`` values. When a single embedding is provided,
            it is wrapped directly without alignment.
        align_on : str, optional
            Column in gene_annotations to align on (default: 'ensembl_gene').

        Returns
        -------
        GeneEmbeddingsSet
            Container with aligned embeddings.

        Raises
        ------
        ValueError
            If no embeddings are provided.
            If source_label values are not unique.
            If no common genes are found (when aligning 2+ embeddings).

        Examples
        --------
        >>> # Multiple embeddings — aligns to common genes
        >>> aligned = GeneEmbeddingsSet.from_gene_embeddings(
        ...     [ge_scgpt, ge_aido],
        ...     align_on='ensembl_gene',
        ... )
        >>> # Single embedding — wraps directly
        >>> single = GeneEmbeddingsSet.from_gene_embeddings(
        ...     [ge_scgpt],
        ...     align_on='ensembl_gene',
        ... )
        """
        if len(embeddings) == 0:
            raise ValueError("GeneEmbeddingsSet requires at least 1 embedding, got 0")

        # Validate unique source_labels
        labels = [emb.source_label for emb in embeddings]
        label_counts = Counter(labels)
        duplicates = {
            label: count for label, count in label_counts.items() if count > 1
        }
        if duplicates:
            raise ValueError(
                f"Duplicate source_label values: {duplicates}. "
                f"Set distinct model_name, model_variant, dataset_name, or category "
                f"on each GeneEmbeddings to produce unique labels."
            )

        if len(embeddings) == 1:
            # Single embedding: validate align_on column exists, wrap directly
            emb = embeddings[0]
            if align_on not in emb.gene_annotations.columns:
                raise ValueError(
                    f"Column '{align_on}' not found in gene_annotations for "
                    f"embedding '{emb.source_label}'. "
                    f"Available columns: {list(emb.gene_annotations.columns)}"
                )
            common_gene_ids = emb.gene_ids_in_ontology(align_on)
            data = {labels[0]: emb}
        else:
            # Multiple embeddings: align to common genes
            aligned_embeddings = _align_gene_embeddings(embeddings, align_on=align_on)
            common_gene_ids = aligned_embeddings[0].gene_ids_in_ontology(align_on)
            data = {
                label: aligned_emb
                for label, aligned_emb in zip(labels, aligned_embeddings)
            }

        return cls(
            data=data,
            common_gene_ids=common_gene_ids,
            align_on=align_on,
        )

    # --- Properties ---

    @property
    def common_gene_ids(self) -> List[str]:
        """Gene identifiers shared across all embeddings (in the alignment ontology)."""
        return list(self._common_gene_ids)

    @property
    def align_on(self) -> str:
        """The ontology column used for alignment."""
        return self._align_on

    @property
    def n_embeddings(self) -> int:
        """Number of stored embeddings."""
        return len(self._data)

    @property
    def n_common_genes(self) -> int:
        """Number of common genes."""
        return len(self._common_gene_ids)

    @property
    def summary(self) -> pd.DataFrame:
        """Summary DataFrame with one row per embedding.

        Returns
        -------
        pd.DataFrame
            Columns: key, model_name, model_variant, dataset_name, dataset_uri,
            category, n_genes, embed_dim.
        """
        rows = []
        for key, emb in self._data.items():
            rows.append(
                {
                    "key": key,
                    "model_name": emb.model_name,
                    "model_variant": emb.model_variant,
                    "dataset_name": emb.dataset_name,
                    "dataset_uri": emb.dataset_uri,
                    "category": emb.category,
                    "n_genes": emb.n_genes,
                    "embed_dim": emb.embed_dim,
                }
            )
        return pd.DataFrame(rows)

    # --- Dict-like access ---

    def get(self, key: str) -> GeneEmbeddings:
        """Get aligned GeneEmbeddings by label.

        Parameters
        ----------
        key : str
            Label (source_label) of the embedding to retrieve.

        Returns
        -------
        GeneEmbeddings
            The aligned embedding.

        Raises
        ------
        KeyError
            If key is not found.
        """
        if key not in self._data:
            raise KeyError(
                f"Embedding '{key}' not found. "
                f"Available keys: {list(self._data.keys())}"
            )
        return self._data[key]

    def __getitem__(self, key: str) -> GeneEmbeddings:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def keys(self):
        """Return embedding labels."""
        return self._data.keys()

    def values(self):
        """Return GeneEmbeddings instances."""
        return self._data.values()

    def items(self):
        """Return (label, GeneEmbeddings) pairs."""
        return self._data.items()

    # --- Dunder methods ---

    def __repr__(self) -> str:
        keys = list(self._data.keys())
        return (
            f"GeneEmbeddingsSet("
            f"n_embeddings={self.n_embeddings}, "
            f"n_common_genes={self.n_common_genes}, "
            f"align_on='{self._align_on}', "
            f"keys={keys}"
            f")"
        )


class DatasetGeneEmbeddings:
    """Container for expression-contextualized gene embeddings across multiple datasets.

    Maps dataset names to ``GeneEmbeddingsSet`` instances. Each dataset may have
    a different gene vocabulary (e.g., different highly-variable gene selections
    from scGPT), but within a dataset all categories share the same genes.

    Each dataset's embeddings are now stored as a ``GeneEmbeddingsSet`` (dict of 2D
    ``GeneEmbeddings``).

    Attributes
    ----------
    n_datasets : int
        Number of datasets.
    dataset_names : List[str]
        Ordered list of dataset names.
    summary : pd.DataFrame
        One row per dataset with gene count, embedding count, and category info.

    Public Methods
    --------------
    from_gene_embeddings_sets(sets)
        Classmethod: construct from a list of (dataset_name, GeneEmbeddingsSet) pairs.
    get(dataset_name)
        Return the GeneEmbeddingsSet for a given dataset.
    keys()
        Return dataset names.
    values()
        Return GeneEmbeddingsSet instances.
    items()
        Return (dataset_name, GeneEmbeddingsSet) pairs.
    all_gene_embeddings()
        Return a flat list of all GeneEmbeddings across all datasets.

    Examples
    --------
    >>> dge = DatasetGeneEmbeddings({
    ...     "pbmc": pbmc_gene_emb_set,
    ...     "tumor": tumor_gene_emb_set,
    ... })
    >>> dge.n_datasets
    2
    >>> dge["pbmc"].n_common_genes
    1200
    >>> dge["tumor"].n_common_genes
    1500
    >>> all_embs = dge.all_gene_embeddings()  # flat list for cross-dataset alignment
    """

    def __init__(self, data: Dict[str, GeneEmbeddingsSet]):
        """Initialize DatasetGeneEmbeddings.

        Parameters
        ----------
        data : Dict[str, GeneEmbeddingsSet]
            Mapping from dataset name to GeneEmbeddingsSet.

        Raises
        ------
        ValueError
            If data is empty.
            If any value is not a GeneEmbeddingsSet.
        """
        if len(data) == 0:
            raise ValueError("DatasetGeneEmbeddings requires at least one dataset")

        for name, ge_set in data.items():
            if not isinstance(ge_set, GeneEmbeddingsSet):
                raise ValueError(
                    f"Dataset '{name}' value must be a GeneEmbeddingsSet, "
                    f"got {type(ge_set)}"
                )

        self._data: Dict[str, GeneEmbeddingsSet] = dict(data)

    # --- Properties ---

    @property
    def n_datasets(self) -> int:
        """Number of datasets."""
        return len(self._data)

    @property
    def dataset_names(self) -> List[str]:
        """Ordered list of dataset names."""
        return list(self._data.keys())

    @property
    def summary(self) -> pd.DataFrame:
        """Summary DataFrame with one row per dataset.

        Returns
        -------
        pd.DataFrame
            Columns: dataset_name, n_embeddings, n_common_genes, align_on,
            embed_dims (set of unique embed_dims across categories).
        """
        rows = []
        for name, ge_set in self._data.items():
            embed_dims = {emb.embed_dim for emb in ge_set.values()}
            rows.append(
                {
                    "dataset_name": name,
                    "n_embeddings": ge_set.n_embeddings,
                    "n_common_genes": ge_set.n_common_genes,
                    "align_on": ge_set.align_on,
                    "embed_dims": embed_dims,
                }
            )
        return pd.DataFrame(rows)

    # --- Public methods ---

    def all_gene_embeddings(self) -> List[GeneEmbeddings]:
        """Return a flat list of all GeneEmbeddings across all datasets.

        Useful as input to ``GeneEmbeddingsSet.from_gene_embeddings`` when
        you want to align embeddings across datasets (which may require
        intersecting to common genes).

        Returns
        -------
        List[GeneEmbeddings]
            All GeneEmbeddings from all datasets, in dataset-then-category order.
        """
        result = []
        for ge_set in self._data.values():
            result.extend(ge_set.values())
        return result

    # --- Dict-like access ---

    def get(self, dataset_name: str) -> GeneEmbeddingsSet:
        """Get GeneEmbeddingsSet by dataset name.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.

        Returns
        -------
        GeneEmbeddingsSet
            The embeddings for that dataset.

        Raises
        ------
        KeyError
            If dataset_name is not found.
        """
        if dataset_name not in self._data:
            raise KeyError(
                f"Dataset '{dataset_name}' not found. "
                f"Available datasets: {list(self._data.keys())}"
            )
        return self._data[dataset_name]

    def __getitem__(self, dataset_name: str) -> GeneEmbeddingsSet:
        return self.get(dataset_name)

    def __contains__(self, dataset_name: str) -> bool:
        return dataset_name in self._data

    def __len__(self) -> int:
        return len(self._data)

    def keys(self):
        """Return dataset names."""
        return self._data.keys()

    def values(self):
        """Return GeneEmbeddingsSet instances."""
        return self._data.values()

    def items(self):
        """Return (dataset_name, GeneEmbeddingsSet) pairs."""
        return self._data.items()

    # --- Dunder methods ---

    def __repr__(self) -> str:
        datasets = list(self._data.keys())
        total_embeddings = sum(ge_set.n_embeddings for ge_set in self._data.values())
        return (
            f"DatasetGeneEmbeddings("
            f"n_datasets={self.n_datasets}, "
            f"total_embeddings={total_embeddings}, "
            f"datasets={datasets}"
            f")"
        )


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
    gene_embedding : np.ndarray
        Gene embedding matrix of shape (n_vocab, embed_dim)
    attention_layers : List[AttentionLayer]
        List of attention layers, one per transformer layer

    Public Methods
    --------------
    compute_attention_from_weights(layer_idx, n_heads, vocab_mask=None, apply_softmax=True, return_tensor=False, device=None)
        Compute attention scores for a specific layer with proper multi-head handling.
    count_attention_parameters()
        Count the total number of parameters across all attention layers.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    gene_embedding: np.ndarray
    attention_layers: List[AttentionLayer]

    @field_validator(FM_DEFS.GENE_EMBEDDING)
    def validate_gene_embedding(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError("gene_embedding must be a numpy array")
        if v.ndim != 2:
            raise ValueError("gene_embedding must be 2-dimensional")
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

    Public Methods
    --------------
    compute_consensus_attention(target_ids, consensus_method='absolute-argmax', gene_annotation_target_var='ensembl_gene', apply_softmax=False)
        Compute consensus attention across all layers for target genes.
    compute_reordered_attention(layer_idx, target_ids, gene_annotation_target_var='ensembl_gene', apply_softmax=True, return_tensor=False, device=None)
        Compute attention scores for a specific layer and reorder to match a target gene ordering.
    full_name
        Property returning full unique identifier (model_name with model_variant if present).
    get_specific_attentions(edge_list, layer_indices=None, target_ids=None, gene_annotation_target_var='ensembl_gene', apply_softmax=False, device=None, verbose=False)
        Extract specific attention values across specified layers for given edges.
    get_top_attentions(k, layer_indices=None, target_ids=None, gene_annotation_target_var='ensembl_gene', apply_softmax=False, device=None, verbose=False)
        Extract top-k strongest attention edges across all layers.
    load(output_dir, prefix)
        Load foundation model from saved files (classmethod).
    save(output_dir, prefix)
        Save foundation model to files.

    Private Methods
    --------------
    _compute_attention(layer_idx, apply_softmax=True, vocab_mask=None, return_tensor=False, device=None)
        Compute attention scores for a specific layer with optional vocabulary mask.
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

    @field_validator(FM_DEFS.GENE_ANNOTATIONS)
    def validate_gene_annotations(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError("gene_annotations must be a pandas DataFrame")

        required_columns = [FM_DEFS.VOCAB_NAME, ONTOLOGIES.ENSEMBL_GENE]
        for col in required_columns:
            if col not in v.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

        return v

    @field_validator(FM_DEFS.DATASET_GENE_EMBEDDINGS)
    def validate_dataset_gene_embeddings(cls, v):
        if v is not None and not isinstance(v, DatasetGeneEmbeddings):
            raise ValueError(
                "dataset_gene_embeddings must be DatasetGeneEmbeddings or None, "
                f"got {type(v)}"
            )
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

    @field_validator(FM_DEFS.ORDERED_VOCABULARY)
    def validate_ordered_vocabulary(cls, v):
        if not isinstance(v, list):
            raise ValueError("ordered_vocabulary must be a list")
        if not all(isinstance(item, str) for item in v):
            raise ValueError("ordered_vocabulary must contain only strings")
        return v

    def compute_consensus_attention(
        self,
        target_ids: List[str],
        consensus_method: str = "absolute-argmax",
        gene_annotation_target_var: str = ONTOLOGIES.ENSEMBL_GENE,
        apply_softmax: bool = False,
        return_layer_indices: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Compute consensus attention across all layers for target genes.

        For each gene pair, aggregates attention values across layers using the
        specified consensus method. The default method ("absolute-argmax") finds the layer
        with the strongest attention (by absolute value) and returns that attention
        value with its original sign preserved.

        Parameters
        ----------
        target_ids : List[str]
            Ordered list of gene identifiers to compute attention for
        consensus_method : str, optional
            Method for aggregating attention across layers to compute consensus.
            Currently supported:
            - "absolute-argmax" (default): Find layer with maximum absolute attention
                and return that value with sign preserved
            - "max": Find layer with maximum attention value (without taking absolute value)
                and return that value
            - "sum": Sum attention values across all layers
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
            Consensus attention, shape (len(target_ids), len(target_ids))
            where result[i, j] is the consensus attention from target_ids[i] to target_ids[j]
            across all layers
        torch.Tensor (optional)
            If return_layer_indices=True, also returns layer indices where max occurred,
            shape (len(target_ids), len(target_ids))

        Examples
        --------
        >>> # Find strongest attention relationships across all layers
        >>> common_genes = ['ENSG00000000003', 'ENSG00000000005', ...]
        >>> consensus_attn = model.compute_consensus_attention(common_genes)
        >>> # Identify which layer had the strongest attention
        >>> consensus_attn, layer_idx = model.compute_consensus_attention(common_genes, return_layer_indices=True)
        >>> # Compare consensus attention across models
        >>> consensus_attn1 = model1.compute_consensus_attention(common_genes)
        >>> consensus_attn2 = model2.compute_consensus_attention(common_genes)
        """
        # Pre-allocate 3D tensor: (n_genes, n_genes, n_layers)
        n_genes = len(target_ids)
        all_attention = torch.zeros(
            (n_genes, n_genes, self.n_layers), dtype=torch.float32
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
            all_attention[:, :, layer_idx] = attention

        # Aggregate attention across layers using specified consensus method
        if consensus_method == FM_LAYER_CONSENSUS_METHODS.ABSOLUTE_ARGMAX:
            return compute_max_abs_over_z(
                all_attention, return_indices=return_layer_indices
            )
        elif consensus_method == FM_LAYER_CONSENSUS_METHODS.MAX:
            return compute_max_over_z(
                all_attention, return_indices=return_layer_indices
            )
        elif consensus_method == FM_LAYER_CONSENSUS_METHODS.SUM:
            # Sum across layers (z dimension)
            if return_layer_indices:
                # For sum, indices don't make sense, but return zeros for consistency
                return all_attention.sum(dim=2), torch.zeros(
                    (n_genes, n_genes), dtype=torch.long
                )
            return all_attention.sum(dim=2)
        else:
            raise ValueError(
                f"Unknown consensus_method '{consensus_method}'. Supported methods: {VALID_FM_LAYER_CONSENSUS_METHODS}"
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

    def get_specific_attentions(
        self,
        edge_list: pd.DataFrame,
        layer_indices: Optional[List[int]] = None,
        target_ids: Optional[List[str]] = None,
        gene_annotation_target_var: str = ONTOLOGIES.ENSEMBL_GENE,
        apply_softmax: bool = False,
        compute_ranks: bool = False,
        by_absolute_value: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Extract specific attention values across layers for given edges.

        This complements find_top_k_attention_edges() by extracting the exact
        attention values for specific gene pairs across specified layers.
        Useful for analyzing how specific relationships vary across layers.

        Parameters
        ----------
        edge_list : pd.DataFrame
            DataFrame with at minimum 'from_gene' and 'to_gene' columns containing
            gene identifiers. Typically the output from find_top_k_attention_edges().
        layer_indices : List[int], optional
            Layers to extract from. If None, uses all layers.
        target_ids : List[str], optional
            Gene identifiers to use. If None, uses all genes in the model.
        gene_annotation_target_var : str, optional
            Column name in gene_annotations to match against target_ids
            (default: ONTOLOGIES.ENSEMBL_GENE)
        apply_softmax : bool, optional
            If True, use softmax-normalized attention probabilities (default: False).
            If False, use raw attention scores.
        compute_ranks : bool, optional
            If True, compute ranks of attention values relative to the full attention tensor
            for each layer and add them to the output table (default: False)
        by_absolute_value : bool, optional
            If True, rank by absolute value when calculating ranks (default: True).
            Only used if compute_ranks=True.
        device : str or torch.device, optional
            Device to perform computation on (default: None to automatically select)
        verbose : bool, optional
            Whether to print verbose output during computation (default: False)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - from_gene : str
                Source gene identifier
            - to_gene : str
                Target gene identifier
            - layer : int
                Layer index
            - attention : float
                Attention value for this edge in this layer
            - attention_rank : int (if compute_ranks=True)
                Integer rank compared to all attention values in the full tensor for this layer (rank 1 = highest)

        Examples
        --------
        >>> # Get top edges from one layer, then extract from all layers
        >>> top_edges = model.get_top_attentions(k=1000, layer_indices=[0,2,3])
        >>> unique_edges = top_edges[['from_gene', 'to_gene']].drop_duplicates()
        >>> all_layers = model.get_specific_attentions(unique_edges)
        >>>
        >>> # Analyze how attention varies across layers for same edges
        >>> pivot = all_layers.pivot_table(
        ...     values='attention',
        ...     index=['from_gene', 'to_gene'],
        ...     columns='layer'
        ... )
        """
        device = ensure_device(device, allow_autoselect=True)

        if target_ids is None:
            target_ids = list(
                self.gene_annotations[gene_annotation_target_var].unique()
            )

        # Convert edge list to indices ONCE
        edge_df = _edgelist_to_indices(
            edge_list=edge_list,
            gene_ids=target_ids,
            verbose=verbose,
        )

        if layer_indices is None:
            layer_indices = list(range(self.n_layers))
        else:
            layer_indices = normalize_and_validate_indices(
                indices=layer_indices,
                max_value=self.n_layers,
                param_name="layer_indices",
            )

        results = []

        with memory_manager(device):
            # Create index tensors on device inside memory_manager
            from_idx_tensor = (
                torch.from_numpy(edge_df[FM_EDGELIST.FROM_IDX].values).long().to(device)
            )
            to_idx_tensor = (
                torch.from_numpy(edge_df[FM_EDGELIST.TO_IDX].values).long().to(device)
            )

            for layer_idx in layer_indices:
                if verbose:
                    logger.info(f"Extracting attentions from layer {layer_idx}...")

                # Compute attention matrix
                attention = self.compute_reordered_attention(
                    layer_idx=layer_idx,
                    target_ids=target_ids,
                    gene_annotation_target_var=gene_annotation_target_var,
                    apply_softmax=apply_softmax,
                    return_tensor=True,
                    device=device,
                )

                if verbose:
                    logger.debug(f"Attention tensor shape: {attention.shape}")
                    logger.debug(f"From index tensor shape: {from_idx_tensor.shape}")
                    logger.debug(f"To index tensor shape: {to_idx_tensor.shape}")

                # Extract edges ON GPU using tensor indexing
                edge_attentions = attention[from_idx_tensor, to_idx_tensor]

                # Move only the extracted values to CPU
                layer_df = edge_df[[FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE]].copy()
                layer_df[FM_EDGELIST.LAYER] = layer_idx
                layer_df[FM_EDGELIST.ATTENTION] = edge_attentions.cpu().numpy()

                # Compute ranks if requested
                if compute_ranks:
                    if verbose:
                        logger.info(f"Calculating ranks for layer {layer_idx}...")

                    # Compute ranks only for the specific indices (memory-efficient)
                    edge_ranks = compute_tensor_ranks_for_indices(
                        attention,
                        (from_idx_tensor, to_idx_tensor),
                        by_absolute_value=by_absolute_value,
                    )
                    layer_df[FM_EDGELIST.ATTENTION_RANK] = edge_ranks.cpu().numpy()

                results.append(layer_df)

                # Clean up
                cleanup_tensors(attention, edge_attentions, edge_ranks)

        # Combine all layers
        all_attentions = pd.concat(results, ignore_index=True)

        if verbose:
            logger.info(
                f"Extracted {len(all_attentions)} total attention values "
                f"({len(edge_df)} edges × {len(layer_indices)} layers)"
            )

        return all_attentions

    def get_top_attentions(
        self,
        k: int,
        layer_indices: Optional[List[int]] = None,
        target_ids: Optional[List[str]] = None,
        gene_annotation_target_var: str = ONTOLOGIES.ENSEMBL_GENE,
        apply_softmax: bool = False,
        by_absolute_value: bool = True,
        compute_ranks: bool = False,
        ignore_self_attention: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Extract top-k strongest attention edges across all layers.

        For each layer, identifies the k gene pairs with highest attention values
        (by absolute value or raw value depending on by_absolute_value parameter)
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
        by_absolute_value : bool, optional
            If True, rank edges by absolute attention value (default: True).
            If False, rank edges by raw attention value.
        compute_ranks : bool, optional
            If True, compute ranks of attention values relative to the full attention tensor
            for each layer and add them to the output table (default: False)
        ignore_self_attention : bool, optional
            If True, exclude self-attention edges (where from_gene == to_gene) from
            top-k selection (default: False).
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
            - attention_rank : int (if compute_ranks=True)
                Integer rank compared to all attention values in the full tensor for this layer (rank 1 = highest)
            Sorted by layer, then by descending absolute attention value (if by_absolute_value=True)
            or descending raw attention value (if by_absolute_value=False).

        Examples
        --------
        >>> # Get top 1000 edges per layer for common genes
        >>> common_genes = ['ENSG00000000003', 'ENSG00000000005', ...]
        >>> top_edges = model.get_top_attentions(k=1000, target_ids=common_genes)
        >>>
        >>> # Rank by raw value instead of absolute value
        >>> top_edges = model.get_top_attentions(k=1000, by_absolute_value=False)
        >>>
        >>> # Exclude self-attention edges
        >>> top_edges = model.get_top_attentions(k=1000, ignore_self_attention=True)
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
                    value_type = "absolute value" if by_absolute_value else "raw value"
                    logger.info(
                        f"Extracting top-{k} edges from layer {layer_idx} by {value_type}..."
                    )

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
                    by_absolute_value=by_absolute_value,
                    ignore_self_attention=ignore_self_attention,
                )

                results.append(layer_df)

                # Clean up attention tensor
                cleanup_tensors(attention)

        # Combine all layers
        all_edges = pd.concat(results, ignore_index=True)

        # Add ranks if requested (rank within each layer)
        if compute_ranks:
            all_edges[FM_EDGELIST.ATTENTION_RANK] = calculate_ranks(
                df=all_edges,
                value_col=FM_EDGELIST.ATTENTION,
                by_absolute_value=by_absolute_value,
                grouping_vars=FM_EDGELIST.LAYER,
            )

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
        (
            weights_dict,
            gene_annotations,
            model_metadata,
            dataset_gene_embeddings_metadata,
        ) = _load_results(output_dir, prefix)

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

        dataset_gene_embeddings = None
        if dataset_gene_embeddings_metadata:
            logger.info(
                f"Loading {len(dataset_gene_embeddings_metadata)} dataset gene embeddings"
            )

            # Group saved entries by dataset_name to reconstruct per-dataset GeneEmbeddingSets
            # Each saved entry is one GeneEmbeddings (one category from one dataset)
            dataset_gene_embeddings_lists: Dict[str, List[GeneEmbeddings]] = {}

            for i, ge_emb_meta in enumerate(dataset_gene_embeddings_metadata):
                embeddings_key = f"dataset_gene_embeddings_{i}"
                if embeddings_key not in weights_dict:
                    logger.warning(
                        f"Expression embeddings metadata found but embeddings tensor "
                        f"'{embeddings_key}' not found in weights file"
                    )
                    continue

                embedding = weights_dict[embeddings_key]

                # Reconstruct gene_annotations DataFrame (required — no fallback)
                ge_annotations = ge_emb_meta.get(FM_DEFS.GENE_ANNOTATIONS)
                if ge_annotations is None:
                    raise ValueError(
                        f"Dataset gene embedding {i} is missing per-embedding "
                        f"gene_annotations. This file was saved with an older "
                        f"format that is no longer supported. Re-run the ETL "
                        f"pipeline to regenerate model outputs."
                    )
                ge_annotations_df = pd.DataFrame(ge_annotations)

                ge = GeneEmbeddings(
                    embedding=embedding,
                    ordered_gene_ids=ge_emb_meta.get(FM_DEFS.ORDERED_GENES, []),
                    gene_annotations=ge_annotations_df,
                    model_name=ge_emb_meta.get(
                        FM_DEFS.MODEL_NAME,
                        model_metadata.get(FM_DEFS.MODEL_NAME),
                    ),
                    model_variant=ge_emb_meta.get(
                        FM_DEFS.MODEL_VARIANT,
                        model_metadata.get(FM_DEFS.MODEL_VARIANT),
                    ),
                    dataset_name=ge_emb_meta.get(FM_DEFS.DATASET_NAME),
                    dataset_uri=ge_emb_meta.get(FM_DEFS.DATASET_URI),
                    category=ge_emb_meta.get(FM_DEFS.CATEGORY),
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
                        ge_list
                    )
                dataset_gene_embeddings = DatasetGeneEmbeddings(dataset_sets)

        return cls(
            weights=weights,
            gene_annotations=gene_annotations,
            model_metadata=model_metadata,
            dataset_gene_embeddings=dataset_gene_embeddings,
        )

    def save(self, output_dir: str, prefix: str) -> None:
        """
        Save foundation model to files.

        Creates two files:
        - {prefix}_weights.npz: Contains gene embeddings, attention weights, and expression embeddings tensors
        - {prefix}_metadata.json: Contains gene annotations, model metadata, and expression embeddings metadata

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

        # Iterate: DatasetGeneEmbeddings -> GeneEmbeddingsSet -> GeneEmbeddings
        # Each GeneEmbeddings gets a sequential index for the weights key
        dataset_gene_embeddings_metadata = []
        if self.dataset_gene_embeddings:
            all_gene_embeddings = self.dataset_gene_embeddings.all_gene_embeddings()
            logger.info(f"Saving {len(all_gene_embeddings)} dataset gene embeddings")

            for i, ge in enumerate(all_gene_embeddings):
                # Save 2D embedding array
                weights_dict[f"dataset_gene_embeddings_{i}"] = ge.embedding

                # Save metadata per GeneEmbeddings
                ge_emb_meta = {
                    FM_DEFS.ORDERED_GENES: ge.ordered_gene_ids,
                    FM_DEFS.GENE_ANNOTATIONS: ge.gene_annotations.to_dict("records"),
                    FM_DEFS.MODEL_NAME: ge.model_name,
                    FM_DEFS.MODEL_VARIANT: ge.model_variant,
                    FM_DEFS.DATASET_NAME: ge.dataset_name,
                    FM_DEFS.DATASET_URI: ge.dataset_uri,
                    FM_DEFS.CATEGORY: ge.category,
                }
                dataset_gene_embeddings_metadata.append(ge_emb_meta)

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
            FM_DEFS.DATASET_GENE_EMBEDDINGS: dataset_gene_embeddings_metadata,
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
    """Container for multiple foundation models with cross-model analysis capabilities.

    This class manages multiple FoundationModel instances and provides methods for
    cross-model comparisons and alignment operations.

    Attributes
    ----------
    models : List[FoundationModel]
        List of foundation model instances (minimum 2 required)

    Public Methods
    --------------
    compare_embeddings(device=None, verbose=False)
        Compare embeddings of all models using Spearman correlation of distance matrices.
    get_common_identifiers(ontology='ensembl_gene', verbose=True)
        Get common identifiers across all models.
    get_consensus_top_attentions(k=10000, consensus_method='absolute-argmax', apply_softmax=False, reextract_union=False, verbose=False)
        Compute consensus top-k attention edges across all models for common genes.
    get_consensus_attentions(consensus_method='absolute-argmax', apply_softmax=False)
        Compute consensus attention scores across all models for common genes.
    get_model(full_name)
        Get a specific model by its full_name attribute.
    get_specific_attentions(edge_list, apply_softmax=False, verbose=False)
        Extract specific attention values across all models and layers for given edges.
    get_top_attentions(k=10000, apply_softmax=False, reextract_union=False, verbose=False)
        Extract top-k attention edges across all models for common genes.
    load_multiple(output_dir, prefixes)
        Load multiple foundation models from saved files (classmethod).
    model_names
        Property returning list of model names.
    __repr__()
        String representation of the FoundationModels instance.

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

    def get_consensus_top_attentions(
        self,
        k: int = 10000,
        consensus_method: str = FM_LAYER_CONSENSUS_METHODS.ABSOLUTE_ARGMAX,
        by_absolute_value: bool = True,
        reextract_union: bool = False,
        apply_softmax: bool = False,
        compute_ranks: bool = False,
        ignore_self_attention: bool = False,
        return_original_and_reextracted: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Extract top-k consensus attention edges across models.

        For each model:
        1. Compute consensus attention across all layers
        2. Extract top-k strongest edges from consensus

        Optionally re-extract the union of all models' top edges
        from every model's consensus.

        Parameters
        ----------
        k : int, optional
            Number of top edges to extract per model (default: 10000)
        consensus_method : str, optional
            Method for aggregating attention across layers to compute consensus.
            Currently supported:
                - "absolute-argmax" (default): Find layer with maximum absolute attention
                and return that value with sign preserved
                - "max": Find layer with maximum attention value (without taking absolute value)
                and return that value
                - "sum": Sum attention values across all layers
                Options: 'absolute-argmax', 'max', 'sum'
        by_absolute_value : bool, optional
            If True, rank edges by absolute attention value (default: True).
            If False, rank edges by raw attention value.
        compute_ranks : bool, optional
            If True, compute ranks of attention values and add them to the output table.
        reextract_union : bool, optional
            If True, take union of all top edges and re-extract from all models
            (default: False)
        apply_softmax : bool, optional
            Whether to apply softmax before computing consensus (default: False)
        ignore_self_attention : bool, optional
            If True, exclude self-attention edges (where from_gene == to_gene) from
            top-k selection (default: False).
        return_original_and_reextracted : bool, optional
            If True and reextract_union=True, return tuple (original, reextracted).
            If False and reextract_union=True, return only reextracted DataFrame.
            Ignored if reextract_union=False (default: False)
        device: str or torch.device, optional
            Device to perform computation on (default: None to automatically select)
        verbose : bool, optional
            Print progress information (default: False)

        Returns
        -------
        pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
            If reextract_union=False:
                Single DataFrame with columns:
                - from_idx : int
                - to_idx : int
                - from_gene : str
                - to_gene : str
                - attention : float (consensus value)
                - model : str

            If reextract_union=True and return_original_and_reextracted=True:
                Tuple of (top_edges_df, reextracted_union_df) where:
                - top_edges_df: Same as above
                - reextracted_union_df: DataFrame with union edges extracted from all models
                    Columns: from_gene, to_gene, attention, model
                - attention_rank : int (if compute_ranks=True)
                    Integer rank compared to all attention values (rank 1 = highest)

            If reextract_union=True and return_original_and_reextracted=False:
                Single DataFrame with reextracted union edges (same structure as reextracted_union_df above)

        Examples
        --------
        >>> # Get top-1000 consensus edges per model
        >>> models = FoundationModels.load_multiple(dir, ['scGPT', 'scPRINT'])
        >>> top_consensus = models.get_consensus_top_attentions(k=1000)
        >>>
        >>> # With union re-extraction: Compare how models score same edges
        >>> top_edges, all_models_on_union = models.get_consensus_top_attentions(
        ...     k=1000,
        ...     reextract_union=True,
        ...     return_original_and_reextracted=True
        ... )
        >>>
        >>> # Just get reextracted union (without original)
        >>> reextracted = models.get_consensus_top_attentions(
        ...     k=1000,
        ...     reextract_union=True,
        ...     return_original_and_reextracted=False
        ... )
        >>>
        >>> # Analyze: Which edges are in multiple models' top-k?
        >>> from collections import Counter
        >>> edge_counts = Counter(
        ...     zip(top_edges['from_gene'], top_edges['to_gene'])
        ... )
        >>> shared_edges = {edge: count for edge, count in edge_counts.items()
        ...                 if count > 1}
        >>>
        >>> # Analyze: How do models differ on the union edges?
        >>> pivot = all_models_on_union.pivot_table(
        ...     values='attention',
        ...     index=['from_gene', 'to_gene'],
        ...     columns='model'
        ... )
        """
        # Get common genes across all models
        common_ids = self.get_common_identifiers(verbose=False)

        if verbose:
            logger.info(
                f"Computing consensus attention across {len(self.models)} models "
                f"for {len(common_ids)} common genes..."
            )

        # Phase 1: Compute consensus for ALL models at once
        # Returns: Tensor of shape (n_models, n_genes, n_genes)
        all_consensus = self.get_consensus_attentions(
            consensus_method=consensus_method,
            apply_softmax=apply_softmax,
        )

        # Extract top-k from each model's consensus
        top_edges_list = []

        for i, model in enumerate(self.models):
            if verbose:
                logger.info(f"Extracting top-{k} edges from {model.full_name}...")

            # Extract top-k using existing utility
            model_top_k = _find_top_k_edges_in_attention_layer(
                attention=all_consensus[i],
                k=k,
                layer_idx=None,  # No layer for consensus
                gene_ids=common_ids,
                by_absolute_value=by_absolute_value,
                ignore_self_attention=ignore_self_attention,
            )
            model_top_k[FM_EDGELIST.MODEL] = model.full_name

            top_edges_list.append(model_top_k)

        all_top_edges = pd.concat(top_edges_list, ignore_index=True)

        if verbose:
            logger.info(
                f"Extracted {len(all_top_edges)} total edges "
                f"({k} per model × {len(self.models)} models)"
            )

        if not reextract_union:
            if return_original_and_reextracted:
                logger.warning(
                    "return_original_and_reextracted=True but reextract_union=False, returning original top-k edges only"
                )

            return all_top_edges

        # Phase 2: Union re-extraction
        unique_edges = all_top_edges[
            [FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE]
        ].drop_duplicates()

        if verbose:
            logger.info(
                f"Re-extracting {len(unique_edges)} unique edges from all models..."
            )

        # Convert edges to indices ONCE
        edge_df = _edgelist_to_indices(
            edge_list=unique_edges,
            gene_ids=common_ids,
            verbose=verbose,
        )

        # Prepare attention tensors and metadata for utility function
        # REUSE consensus from Phase 1 - no recomputation!
        attention_tensors = [all_consensus[i] for i in range(len(self.models))]
        metadata = [{FM_EDGELIST.MODEL: model.full_name} for model in self.models]

        # Use utility to extract edges
        reextracted_union = _extract_edges_from_attention_tensors(
            edge_df=edge_df,
            attention_tensors=attention_tensors,
            metadata=metadata,
            compute_ranks=compute_ranks,
            by_absolute_value=by_absolute_value,
            device=device,
            verbose=verbose,
        )

        if verbose:
            logger.info(
                f"Extracted {len(reextracted_union)} total attention values "
                f"({len(unique_edges)} edges × {len(self.models)} models)"
            )

        if return_original_and_reextracted:
            return all_top_edges, reextracted_union
        else:
            return reextracted_union

    def get_specific_attentions(
        self,
        edge_list: pd.DataFrame,
        apply_softmax: bool = False,
        compute_ranks: bool = False,
        by_absolute_value: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Extract specific attention values across all models and layers for given edges.

        This complements get_top_attentions() by extracting the exact attention values
        for specific gene pairs across all models and layers. Useful for comparing how
        different models represent the same biological relationships.

        Parameters
        ----------
        edge_list : pd.DataFrame
            DataFrame with at minimum 'from_gene' and 'to_gene' columns containing
            gene identifiers. Typically the output from get_top_attentions().
        apply_softmax : bool, optional
            If True, use softmax-normalized attention probabilities (default: False).
            If False, use raw attention scores.
        compute_ranks : bool, optional
            If True, compute ranks of attention values relative to the full attention tensor
            for each model/layer and add them to the output table (default: False)
        by_absolute_value : bool, optional
            If True, rank by absolute value when calculating ranks (default: True).
            Only used if compute_ranks=True.
        verbose : bool, optional
            Whether to print verbose output during computation (default: False)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - from_gene : str
                Source gene identifier
            - to_gene : str
                Target gene identifier
            - model : str
                Model name
            - layer : int
                Layer index
            - attention : float
                Attention value for this edge in this model/layer
            - attention_rank : int (if compute_ranks=True)
                Integer rank compared to all attention values in the full tensor for this model/layer (rank 1 = highest)

        Examples
        --------
        >>> # Get top edges, then extract those same edges from all models/layers
        >>> top_edges = models.get_top_attentions(k=1000)
        >>> # Get unique edges (remove layer/model info)
        >>> unique_edges = top_edges[['from_gene', 'to_gene']].drop_duplicates()
        >>> # Extract these edges from all models and layers
        >>> all_attentions = models.get_specific_attentions(unique_edges)
        >>>
        >>> # Now analyze how attention varies across models for same edges
        >>> pivot = all_attentions.pivot_table(
        ...     values='attention',
        ...     index=['from_gene', 'to_gene', 'layer'],
        ...     columns='model'
        ... )
        """
        # Get common identifiers across all models
        common_ids = self.get_common_identifiers(verbose=False)

        results = []

        # Iterate over models - delegate to FoundationModel method
        for model in self.models:
            model_name = model.full_name

            if verbose:
                logger.info(f"Extracting attentions from {model_name}...")

            # Delegate to FoundationModel.get_specific_attentions()
            model_attentions = model.get_specific_attentions(
                edge_list=edge_list,
                layer_indices=None,  # Extract from all layers
                target_ids=common_ids,
                apply_softmax=apply_softmax,
                compute_ranks=compute_ranks,
                by_absolute_value=by_absolute_value,
                verbose=False,  # Suppress per-layer logging
            )

            # Add model name column
            model_attentions[FM_EDGELIST.MODEL] = model_name

            results.append(model_attentions)

        # Combine all results
        all_attentions = pd.concat(results, ignore_index=True)

        if verbose:
            n_edges = len(
                edge_list[
                    [FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE]
                ].drop_duplicates()
            )
            logger.info(
                f"Extracted {len(all_attentions)} total attention values "
                f"({n_edges} edges × {len(self.models)} models × "
                f"{self.models[0].n_layers} layers)"
            )

        return all_attentions

    def get_top_attentions(
        self,
        k: int = 10000,
        by_absolute_value: bool = True,
        reextract_union: bool = False,
        apply_softmax: bool = False,
        compute_ranks: bool = False,
        ignore_self_attention: bool = False,
        return_original_and_reextracted: bool = False,
        verbose: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Extract top-k attention edges across all models for common genes.

        For each model, identifies the k strongest attention relationships per layer
        among genes that are common across all models. This enables cross-model
        comparison of attention patterns by identifying the most significant
        gene-gene relationships learned by each model.

        Parameters
        ----------
        k : int, optional
            Number of top edges to extract per layer per model (default: 10000)
        by_absolute_value : bool, optional
            If True, rank edges by absolute attention value (default: True).
            If False, rank edges by raw attention value.
        reextract_union: bool, optional
            If True, take the union of top edges and extract them from every model and layer.
            If False, extract top edges from each model and layer separately.
        apply_softmax : bool, optional
            If True, use softmax-normalized attention probabilities (default: False).
            If False, use raw attention scores for ranking.
        compute_ranks : bool, optional
            If True, compute ranks of attention values and add them to the output table.
            Ranks are computed within each model and layer group (default: False)
        ignore_self_attention : bool, optional
            If True, exclude self-attention edges (where from_gene == to_gene) from
            top-k selection (default: False).
        return_original_and_reextracted : bool, optional
            If True and reextract_union=True, return tuple (original, reextracted).
            If False and reextract_union=True, return only reextracted DataFrame.
            Ignored if reextract_union=False (default: False)
        verbose : bool, optional
            Whether to print verbose output during computation (default: False)

        Returns
        -------
        pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
            If reextract_union is False, returns a single DataFrame with columns:
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
            - attention_rank : int (if compute_ranks=True)
                Integer rank compared to all attention values within the same model and layer (rank 1 = highest)
            Sorted by model, then layer, then by descending absolute attention value (if by_absolute_value=True)
            or descending raw attention value (if by_absolute_value=False).

        If reextract_union is True and return_original_and_reextracted is True:
            Returns a tuple of two DataFrames:
            - The first DataFrame is the same as above (original top edges).
            - The second DataFrame has the attention for each top edge across all models and layers.
                Includes 'attention_rank' column if compute_ranks=True (ranks within each model and layer).

        If reextract_union is True and return_original_and_reextracted is False:
            Returns a single DataFrame with reextracted union edges (same structure as second DataFrame above).
                Includes 'attention_rank' column if compute_ranks=True (ranks within each model and layer).

        Examples
        --------
        >>> # Get top 1000 attention edges per layer for all models
        >>> models = FoundationModels.load_multiple('/path/to/output', ['scGPT', 'Geneformer'])
        >>> top_edges = models.get_top_attentions(k=1000)
        >>>
        >>> # Compare attention patterns between models
        >>> scgpt_edges = top_edges[top_edges['model'] == 'scGPT']
        >>> geneformer_edges = top_edges[top_edges['model'] == 'Geneformer']
        >>>
        >>> # Get both original and reextracted union
        >>> original, reextracted = models.get_top_attentions(
        ...     k=1000,
        ...     reextract_union=True,
        ...     return_original_and_reextracted=True
        ... )
        >>>
        >>> # Get only reextracted union
        >>> reextracted = models.get_top_attentions(
        ...     k=1000,
        ...     reextract_union=True,
        ...     return_original_and_reextracted=False
        ... )
        >>>
        >>> # Rank by raw value instead of absolute value
        >>> top_edges = models.get_top_attentions(k=1000, by_absolute_value=False)
        """
        common_ids = self.get_common_identifiers()
        n_models = len(self.models)

        top_attention_edges = list()
        for i in range(n_models):
            model = self.models[i]
            model_name = model.full_name

            logger.info(f"Computing top-k attention for {model_name}...")

            model_top_k_attention = model.get_top_attentions(
                k=k,
                target_ids=common_ids,
                apply_softmax=apply_softmax,
                by_absolute_value=by_absolute_value,
                compute_ranks=compute_ranks,
                ignore_self_attention=ignore_self_attention,
                verbose=verbose,
            ).assign(model=model_name)

            top_attention_edges.append(model_top_k_attention)

        all_top_edges = pd.concat(top_attention_edges, ignore_index=True)

        if reextract_union:
            logger.info("Re-extracting top edges from every model and layer...")

            reextracted_top_edges = self.get_specific_attentions(
                all_top_edges,
                apply_softmax=apply_softmax,
                compute_ranks=compute_ranks,
                by_absolute_value=by_absolute_value,
                verbose=verbose,
            )

            if return_original_and_reextracted:
                return all_top_edges, reextracted_top_edges
            else:
                return reextracted_top_edges
        else:
            if return_original_and_reextracted:
                logger.warning(
                    "return_original_and_reextracted=True but reextract_union=False, returning original top-k edges only"
                )
            return all_top_edges

    def get_consensus_attentions(
        self,
        consensus_method: str = FM_LAYER_CONSENSUS_METHODS.ABSOLUTE_ARGMAX,
        apply_softmax: bool = False,
        device: Optional[Union[str, torch.device]] = None,
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
        consensus_method : str, optional
            Method for aggregating attention across layers to compute consensus.
            Currently supported:
            - "absolute-argmax" (default): Find layer with maximum absolute attention
              and return that value with sign preserved
            - "max": Find layer with maximum attention value (without taking absolute value)
              and return that value
            - "sum": Sum attention values across all layers
        softmax : bool, optional
            If True, apply softmax to the attention scores (default: False).
        device: str or torch.device, optional
            Device to perform computation on (default: None to automatically select)


        Examples
        --------
        >>> models = FoundationModels.load_multiple('/path/to/output', ['scGPT', 'Geneformer'])
        >>> consensus_attentions = models.get_consensus_attentions()
        >>> # Compare attention patterns between first two models
        >>> model1_attn = consensus_attentions[0]
        >>> model2_attn = consensus_attentions[1]
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
            logger.info(f"Computing consensus attention for {model.full_name}...")

            attention = model.compute_consensus_attention(
                target_ids=common_ids,
                consensus_method=consensus_method,
                apply_softmax=apply_softmax,
                device=device,
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

        # Create instance and sort by parameters
        instance = cls(models=loaded_models)
        instance._sort_models_by_parameters()

        return instance

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


# Private utility functions


def _align_gene_embeddings(
    embeddings: List[GeneEmbeddings],
    align_on: str = ONTOLOGIES.ENSEMBL_GENE,
) -> List[GeneEmbeddings]:
    """Align multiple GeneEmbeddings to their common genes via a shared ontology.

    Each GeneEmbeddings may use a different native vocabulary (symbols, ensembl
    IDs, etc.) in its ``ordered_gene_ids``. This function:

    1. Maps each embedding's genes to the ``align_on`` ontology via gene_annotations.
    2. Computes the intersection of mapped IDs across all embeddings.
    3. Filters and reorders each embedding so that its rows correspond to the
        common genes (in a consistent order), while preserving the native
        vocabulary in ``ordered_gene_ids``.

    Parameters
    ----------
    embeddings : List[GeneEmbeddings]
        Two or more GeneEmbeddings instances to align.
    align_on : str, optional
        Column name in gene_annotations to use as the common ontology
        (default: 'ensembl_gene'). Must be present in every embedding's
        gene_annotations.

    Returns
    -------
    List[GeneEmbeddings]
        New GeneEmbeddings instances, one per input, each filtered and reordered
        so that ``gene_ids_in_ontology(align_on)`` returns the same list for all.
        The native ``ordered_gene_ids`` are preserved (each embedding keeps its
        own vocabulary).

    Raises
    ------
    ValueError
        If fewer than 2 embeddings are provided.
        If ``align_on`` column is missing from any embedding's gene_annotations.
        If the intersection of common genes is empty.
        If any embedding has duplicate values in the ``align_on`` column
        (would make alignment ambiguous).

    Examples
    --------
    >>> # scGPT uses ensembl IDs natively, AIDOCell uses symbols
    >>> aligned = align_gene_embeddings([ge_scgpt, ge_aido])
    >>> # Both now cover the same genes in the same order
    >>> aligned[0].gene_ids_in_ontology('ensembl_gene')
    ['ENSG00000141510', 'ENSG00000157764', ...]
    >>> aligned[1].gene_ids_in_ontology('ensembl_gene')
    ['ENSG00000141510', 'ENSG00000157764', ...]
    >>> # But native vocabularies are preserved
    >>> aligned[0].ordered_gene_ids[:2]
    ['ENSG00000141510', 'ENSG00000157764']
    >>> aligned[1].ordered_gene_ids[:2]
    ['TP53', 'BRAF']
    """
    if len(embeddings) < 2:
        raise ValueError(
            f"align_gene_embeddings requires at least 2 embeddings, got {len(embeddings)}"
        )

    # --- Validate align_on column exists in all embeddings ---
    for emb in embeddings:
        if align_on not in emb.gene_annotations.columns:
            raise ValueError(
                f"Column '{align_on}' not found in gene_annotations for "
                f"embedding '{emb.source_label}'. "
                f"Available columns: {list(emb.gene_annotations.columns)}"
            )

    # --- Validate no duplicates in the align_on column ---
    for emb in embeddings:
        ontology_ids = emb.gene_annotations[align_on]
        duplicates = ontology_ids[ontology_ids.duplicated()].tolist()
        if duplicates:
            raise ValueError(
                f"Embedding '{emb.source_label}' has duplicate values in "
                f"'{align_on}' column: {duplicates[:5]}{'...' if len(duplicates) > 5 else ''}. "
                f"Alignment requires unique values in the align_on column."
            )

    # --- Compute intersection across all embeddings ---
    common_ids = None
    for emb in embeddings:
        ids = set(emb.gene_annotations[align_on].dropna())
        if common_ids is None:
            common_ids = ids
        else:
            common_ids = common_ids.intersection(ids)

    if not common_ids:
        raise ValueError(
            f"No common genes found across {len(embeddings)} embeddings "
            f"using ontology '{align_on}'"
        )

    # Use a stable sorted order for the common gene set
    common_ids_ordered = sorted(common_ids)

    logger.info(
        f"Found {len(common_ids_ordered)} common genes across "
        f"{len(embeddings)} embeddings (ontology: '{align_on}')"
    )

    # --- Filter and reorder each embedding ---
    aligned = []
    for emb in embeddings:
        # Build mapping: align_on value -> row index in this embedding
        ontology_values = emb.gene_annotations[align_on].tolist()
        ontology_to_row = {val: idx for idx, val in enumerate(ontology_values)}

        # Reorder indices to match common_ids_ordered
        reorder_indices = np.array([ontology_to_row[gid] for gid in common_ids_ordered])

        # Build new ordered_gene_ids in the native vocabulary, reordered
        native_ids = [emb.ordered_gene_ids[i] for i in reorder_indices]

        aligned_emb = GeneEmbeddings(
            embedding=emb.embedding[reorder_indices],
            ordered_gene_ids=native_ids,
            gene_annotations=emb.gene_annotations.iloc[reorder_indices].reset_index(
                drop=True
            ),
            model_name=emb.model_name,
            model_variant=emb.model_variant,
            dataset_name=emb.dataset_name,
            category=emb.category,
        )
        aligned.append(aligned_emb)

    return aligned


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
            rho = compute_correlation(
                dist1_flat,
                dist2_flat,
                method=CORRELATION_METHODS.SPEARMAN,
                device=device,
            )
            comparisons[f"{model1}_vs_{model2}"] = rho

            if verbose:
                logger.info(f"  {model1} vs {model2}: Spearman rho = {rho:.4f}")

    return comparisons


def _edgelist_to_indices(
    edge_list: pd.DataFrame,
    gene_ids: List[str],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Convert edge list with gene identifiers to indices.

    Parameters
    ----------
    edge_list : pd.DataFrame
        DataFrame with 'from_gene' and 'to_gene' columns
    gene_ids : List[str]
        Ordered list of gene identifiers (defines index mapping)
    verbose : bool, optional
        Whether to print warnings about filtered edges

    Returns
    -------
    pd.DataFrame
        DataFrame with 'from_gene', 'to_gene', 'from_idx', 'to_idx' columns,
        filtered to only edges where both genes are in gene_ids
    """
    # Validate input
    required_cols = [FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE]
    missing = [col for col in required_cols if col not in edge_list.columns]
    if missing:
        raise ValueError(f"edge_list must contain columns: {missing}")

    # Get unique edges
    unique_edges = edge_list[required_cols].drop_duplicates()
    n_edges = len(unique_edges)

    # Filter edges to only those in gene_ids
    edges_in_common = unique_edges[
        unique_edges[FM_EDGELIST.FROM_GENE].isin(gene_ids)
        & unique_edges[FM_EDGELIST.TO_GENE].isin(gene_ids)
    ].copy()

    if len(edges_in_common) < n_edges and verbose:
        logger.warning(
            f"Filtered from {n_edges} to {len(edges_in_common)} edges "
            f"(some genes not in gene_ids)"
        )

    # Create index mappings
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_ids)}
    edges_in_common[FM_EDGELIST.FROM_IDX] = edges_in_common[FM_EDGELIST.FROM_GENE].map(
        gene_to_idx
    )
    edges_in_common[FM_EDGELIST.TO_IDX] = edges_in_common[FM_EDGELIST.TO_GENE].map(
        gene_to_idx
    )

    return edges_in_common


def _extract_edges_from_attention_tensor(
    edge_df: pd.DataFrame,
    attention: Tensor,
    from_idx_tensor: Tensor,
    to_idx_tensor: Tensor,
    metadata: Optional[Dict[str, Any]] = None,
    rank_tensor: Optional[Tensor] = None,
) -> pd.DataFrame:
    """
    Extract specific edge values from a single attention tensor.

    Core utility for extracting edge attention values from an attention matrix
    using pre-computed index tensors. This function performs the actual GPU
    extraction and DataFrame construction.

    Parameters
    ----------
    edge_df : pd.DataFrame
        Must contain 'from_gene' and 'to_gene' columns.
        Typically output from _edgelist_to_indices().
    attention : Tensor
        Attention matrix of shape (n_genes, n_genes) on device.
    from_idx_tensor : Tensor
        Pre-computed source indices tensor on device.
    to_idx_tensor : Tensor
        Pre-computed target indices tensor on device.
    metadata : Dict[str, Any], optional
        Metadata dict whose keys become DataFrame columns (default: None).
        Example: {'model': 'scGPT', 'layer': 0}
    rank_tensor : Tensor, optional
        Pre-computed rank tensor from compute_tensor_ranks().
        If provided, adds 'attention_rank' column with integer ranks (rank 1 = highest).
        Uses fast O(n_edges) indexing (default: None)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - from_gene : str
        - to_gene : str
        - attention : float
        - attention_rank : int (if rank_tensor is provided)
            Integer rank compared to all attention values (rank 1 = highest)
        - <metadata columns> : any keys from metadata dict

    Examples
    --------
    >>> # Extract edges from a single attention tensor
    >>> from_idx = torch.from_numpy(edge_df['from_idx'].values).long().to(device)
    >>> to_idx = torch.from_numpy(edge_df['to_idx'].values).long().to(device)
    >>> result = _extract_edges_from_attention_tensor(
    ...     edge_df, attention, from_idx, to_idx, {'layer': 0}
    ... )
    >>>
    >>> # Extract with ranks: pre-compute rank tensor once, then use for multiple extractions
    >>> rank_tensor = compute_tensor_ranks(attention)
    >>> result = _extract_edges_from_attention_tensor(
    ...     edge_df, attention, from_idx, to_idx, {'layer': 0},
    ...     rank_tensor=rank_tensor
    ... )
    >>> # Filter to top 100 strongest relationships
    >>> top_100 = result[result['attention_rank'] <= 100]
    """
    # Extract edges ON GPU using pre-computed indices
    edge_attentions = attention[from_idx_tensor, to_idx_tensor]

    # Move only the extracted values to CPU
    result_df = edge_df[[FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE]].copy()
    result_df[FM_EDGELIST.ATTENTION] = edge_attentions.cpu().numpy()

    # Compute ranks if rank_tensor is provided
    if rank_tensor is not None:
        ranks = rank_tensor[from_idx_tensor, to_idx_tensor].cpu().numpy()
        result_df[FM_EDGELIST.ATTENTION_RANK] = ranks

    # Add metadata columns
    if metadata:
        for key, value in metadata.items():
            result_df[key] = value

    return result_df


def _extract_edges_from_attention_tensors(
    edge_df: pd.DataFrame,
    attention_tensors: List[Tensor],
    metadata: List[Dict[str, Any]],
    delete_as_processed: bool = False,
    compute_ranks: bool = False,
    by_absolute_value: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Extract specific edge values from multiple attention tensors.

    Generic utility for extracting edges from a sequence of attention matrices,
    whether they represent different layers, models, or other dimensions.

    Parameters
    ----------
    edge_df : pd.DataFrame
        Must contain 'from_idx', 'to_idx', 'from_gene', 'to_gene' columns.
        Typically output from _edgelist_to_indices().
    attention_tensors : List[Tensor]
        List of attention matrices, each of shape (n_genes, n_genes).
        All tensors must have the same shape.
    metadata : List[Dict[str, Any]]
        Metadata for each tensor. Must have same length as attention_tensors.
        Each dict's keys become DataFrame columns.
        Example: [{'model': 'scGPT', 'layer': 0}, ...]
    delete_as_processed : bool, optional
        If True, delete each attention tensor from the list immediately after processing
        to free GPU memory. Useful when processing many large tensors (default: True)
    compute_ranks : bool, optional
        If True, compute ranks of attention values and add them to the output table.
    by_absolute_value : bool, optional
        If True, rank by absolute value when calculating ranks (default: True).
        Only used if compute_ranks=True.
    device : str or torch.device, optional
        Device for computation (default: None to auto-select)
    verbose : bool, optional
        Print progress information (default: False)

    Returns
    -------
    pd.DataFrame
        Combined results with columns:
        - from_gene : str
        - to_gene : str
        - attention : float
        - <metadata columns> : any keys from metadata dicts

    Raises
    ------
    ValueError
        If attention_tensors and metadata have different lengths

    Examples
    --------
    >>> # Extract from multiple layers
    >>> attention_tensors = [layer0_attn, layer1_attn, layer2_attn]
    >>> metadata = [{'layer': 0}, {'layer': 1}, {'layer': 2}]
    >>> results = _extract_edges_from_attention_tensors(
    ...     edge_df, attention_tensors, metadata
    ... )

    >>> # Extract from multiple models
    >>> attention_tensors = [model1_consensus, model2_consensus]
    >>> metadata = [{'model': 'scGPT'}, {'model': 'scPRINT'}]
    >>> results = _extract_edges_from_attention_tensors(
    ...     edge_df, attention_tensors, metadata
    ... )
    """
    if len(attention_tensors) != len(metadata):
        raise ValueError(
            f"attention_tensors and metadata must have same length, "
            f"got {len(attention_tensors)} and {len(metadata)}"
        )

    device = ensure_device(device, allow_autoselect=True)
    results = []

    with memory_manager(device):
        # Create index tensors ONCE - stays on device for all iterations
        from_idx_tensor = (
            torch.from_numpy(edge_df[FM_EDGELIST.FROM_IDX].values).long().to(device)
        )
        to_idx_tensor = (
            torch.from_numpy(edge_df[FM_EDGELIST.TO_IDX].values).long().to(device)
        )

        # Iterate backwards to allow safe deletion from list while iterating
        # When delete_as_processed=False, this is equivalent to forward iteration
        for i in range(len(attention_tensors) - 1, -1, -1):
            attention = attention_tensors[i].to(device).clone()
            meta = metadata[i]

            if verbose:
                meta_str = ", ".join(f"{k}={v}" for k, v in meta.items())
                logger.info(
                    f"Extracting edges from tensor {len(attention_tensors) - i}/{len(attention_tensors)} "
                    f"({meta_str})..."
                )

            rank_tensor = None
            if compute_ranks:
                rank_tensor = compute_tensor_ranks(
                    attention, by_absolute_value=by_absolute_value
                )

            if delete_as_processed:
                if verbose:
                    logger.info(f"Deleting attention tensor {i} from list...")
                del attention_tensors[i]

            # Extract edges using utility function
            result_df = _extract_edges_from_attention_tensor(
                edge_df=edge_df,
                attention=attention,
                from_idx_tensor=from_idx_tensor,
                to_idx_tensor=to_idx_tensor,
                metadata=meta,
                rank_tensor=rank_tensor,
            )

            # Prepend to maintain order (since we're iterating backwards)
            results.insert(0, result_df)

            # Cleanup
            cleanup_tensors(attention)

    return pd.concat(results, ignore_index=True)


def _find_top_k_edges_in_attention_layer(
    attention: Tensor,
    k: int,
    layer_idx: Optional[int] = None,
    gene_ids: Optional[List[str]] = None,
    by_absolute_value: bool = True,
    ignore_self_attention: bool = False,
) -> pd.DataFrame:
    """
    Extract top-k edges from an attention matrix.

    Identifies the k gene pairs with highest attention values (by absolute value
    or raw value depending on by_absolute_value parameter) and returns them as a
    DataFrame with gene indices and identifiers.

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
    by_absolute_value : bool, optional
        If True, rank edges by absolute attention value (default: True).
        If False, rank edges by raw attention value.
    ignore_self_attention : bool, optional
        If True, exclude self-attention edges (where from_gene == to_gene) from
        top-k selection (default: False).

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
        Sorted by descending absolute attention value (if by_absolute_value=True)
        or descending raw attention value (if by_absolute_value=False).

    Examples
    --------
    >>> # Basic usage
    >>> attention = model.compute_reordered_attention(0, common_genes, return_tensor=True)
    >>> top_edges = find_top_k_edges(attention, k=1000, layer_idx=0, gene_ids=common_genes)
    >>>
    >>> # Without gene IDs
    >>> top_edges = find_top_k_edges(attention, k=100)
    >>>
    >>> # Rank by raw value instead of absolute value
    >>> top_edges = find_top_k_edges(attention, k=100, by_absolute_value=False)
    >>>
    >>> # Exclude self-attention edges
    >>> top_edges = find_top_k_edges(attention, k=1000, ignore_self_attention=True)
    """

    # Get device from attention tensor for memory management
    cleanup_device = attention.device

    # Extract top-k indices and values (stays on device)
    # Use memory_manager to ensure intermediate tensors are cleaned up
    with memory_manager(cleanup_device):
        from_indices, to_indices, top_values = find_top_k(
            attention,
            k=k,
            by_absolute_value=by_absolute_value,
            ignore_self_attention=ignore_self_attention,
        )

        # Build base DataFrame with indices and attention
        df = pd.DataFrame(
            {
                FM_EDGELIST.FROM_IDX: from_indices.cpu().numpy(),
                FM_EDGELIST.TO_IDX: to_indices.cpu().numpy(),
                FM_EDGELIST.ATTENTION: top_values.cpu().numpy(),
            }
        )

        # Clean up GPU tensors immediately
        del from_indices, to_indices, top_values

    # Add layer if provided
    if layer_idx is not None:
        df[FM_EDGELIST.LAYER] = layer_idx

    # Add gene IDs via merge if provided
    if gene_ids is not None:
        # Create lookup table mapping index to gene_id
        gene_lookup = pd.DataFrame({"idx": range(len(gene_ids)), "gene_id": gene_ids})

        # Merge for from_gene
        df = df.merge(
            gene_lookup.rename(
                columns={"idx": FM_EDGELIST.FROM_IDX, "gene_id": FM_EDGELIST.FROM_GENE}
            ),
            on=FM_EDGELIST.FROM_IDX,
            how="left",
        )

        # Merge for to_gene
        df = df.merge(
            gene_lookup.rename(
                columns={"idx": FM_EDGELIST.TO_IDX, "gene_id": FM_EDGELIST.TO_GENE}
            ),
            on=FM_EDGELIST.TO_IDX,
            how="left",
        )

    # Order columns
    col_order = []
    if FM_EDGELIST.LAYER in df.columns:
        col_order.append(FM_EDGELIST.LAYER)
    col_order.extend([FM_EDGELIST.FROM_IDX, FM_EDGELIST.TO_IDX])
    if FM_EDGELIST.FROM_GENE in df.columns:
        col_order.extend([FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE])
    col_order.append(FM_EDGELIST.ATTENTION)

    return df[col_order]


def _load_results(
    output_dir: str, prefix: str
) -> Tuple[dict, pd.DataFrame, dict, List[dict]]:
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
    dataset_gene_embeddings_metadata : List[dict]
        List of dictionaries containing dataset gene embeddings metadata
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

    # Load expression embeddings metadata (if present)
    dataset_gene_embeddings_metadata = combined_metadata.get(
        FM_DEFS.DATASET_GENE_EMBEDDINGS, []
    )

    logger.info("Successfully loaded all results")

    return (
        weights_dict,
        gene_annotations,
        model_metadata,
        dataset_gene_embeddings_metadata,
    )
