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

Class Relationships
-------------------
FoundationModels
    FoundationModel
        FoundationModelWeights
            GeneEmbeddings (static)
                GeneAnnotations
            List[AttentionLayer]
        DatasetGeneEmbeddings
            GeneEmbeddingsSet (all embeddings for a given dataset)
                GeneEmbeddings (a single 2D embedding matrix, e.g., of a cell type, cluster, or individual sample)
                    GeneAnnotations
        ModelMetadata

AttendedEmbeddingsSet
    AttendedEmbeddings
        GeneEmbeddingsSet (a set of embedding of interest - static, expression-based, etc.)
        FoundationModels (the models containing the attention weights for these embeddings)
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
    COMPARE_EMBEDDINGS_COMPARISONS,
    COMPARE_EMBEDDINGS_SETTINGS,
    EMBEDDING_METADATA_FIELDS,
    FM_DEFS,
    FM_EDGELIST,
    FM_LAYER_CONSENSUS_METHODS,
    MODEL_NICE_NAMES,
    SCOPING_FIELDS,
    VALID_COMPARE_EMBEDDINGS_COMPARISONS,
    VALID_FM_LAYER_CONSENSUS_METHODS,
)
from napistu_torch.utils.base_utils import normalize_and_validate_indices
from napistu_torch.utils.constants import CORRELATION_METHODS
from napistu_torch.utils.pd_utils import calculate_ranks
from napistu_torch.utils.statistics import compare_top_k_union_ranks
from napistu_torch.utils.tensor_utils import (
    compute_correlation,
    compute_correlation_matrix,
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
    layer_idx : Optional[int]
        Index of the transformer layer this embedding represents.
        None for non-transformer embeddings or models that capture
        a single activation per category.
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
    get_gene_mask(target_ids=None)
        Return a boolean mask and gene IDs for optionally restricting to target_ids.

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
    layer_idx: Optional[int] = None
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
        if self.layer_idx is not None:
            parts.append(f"layer_{self.layer_idx}")
        if self.dataset_name:
            parts.append(self.dataset_name)
        if self.category:
            parts.append(self.category)
        return "/".join(parts) if parts else "static"

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
            layer_idx=self.layer_idx,
            dataset_name=self.dataset_name,
            dataset_uri=self.dataset_uri,
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

    def get_gene_mask(
        self, target_ids: Optional[List[str]] = None
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Return a boolean mask and gene IDs for optionally restricting to target_ids.

        Parameters
        ----------
        target_ids : List[str], optional
            Subset of gene IDs to restrict to. Must all be present in this
            embedding. If None, uses all genes.

        Returns
        -------
        gene_mask : np.ndarray or None
            Boolean array of shape (n_genes,) with True for genes in target_ids,
            in the order of ordered_gene_ids. None if target_ids is None.
        gene_ids : List[str]
            ordered_gene_ids if target_ids is None, else target_ids (to preserve
            their order for downstream use).

        Raises
        ------
        ValueError
            If any target_ids are not found in this embedding.
        """
        gene_ids = self.ordered_gene_ids
        if target_ids is None:
            return None, gene_ids
        missing = [gid for gid in target_ids if gid not in self.gene_ids_set]
        if missing:
            raise ValueError(
                f"{len(missing)} target_ids not found in embedding. "
                f"First few: {missing[:5]}"
            )
        gene_mask = np.array([gid in set(target_ids) for gid in gene_ids], dtype=bool)
        return gene_mask, target_ids

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

    Keys are automatically scoped to the minimal distinguishing metadata. For example,
    if all embeddings share the same model and dataset, keys are just the category
    (e.g., "adipocyte (0)"). The constant parts are available as ``constant_label``
    for use in plot titles.

    Attributes
    ----------
    data : Dict[str, GeneEmbeddings]
        Mapping from scoped key to aligned GeneEmbeddings.
    common_gene_ids : List[str]
        Gene identifiers (in the alignment ontology) shared across all embeddings,
        in the order used for alignment.
    aligned_on : str
        The ontology column used for alignment.
    embedding_metadata : pd.DataFrame
        One row per embedding with source metadata, model_label, and scoped_key.
    constant_label : str
        Human-readable label for metadata that is constant across all embeddings.
        Useful for plot titles. e.g., "scGPT / efthymiou2025". Empty string if
        nothing is constant.

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
    compare_embeddings(device=None, verbose=False)
        Compare embeddings of all models using Spearman correlation of distance matrices.
    from_gene_embeddings(embeddings, align_on='ensembl_gene')
        Classmethod: align embeddings and construct container.

    Dictionary-like Methods
    -----------------------
    get(key)
        Return the aligned GeneEmbeddings for a given key.
    keys()
        Return embedding labels (scoped keys).
    values()
        Return GeneEmbeddings instances.
    items()
        Return (scoped_key, GeneEmbeddings) pairs.

    Examples
    --------
    >>> # Cross-model alignment — keys are model names
    >>> aligned = GeneEmbeddingsSet.from_gene_embeddings(
    ...     [ge_scgpt, ge_aido, ge_scprint]
    ... )
    >>> aligned.keys()
    ['scGPT', 'AIDOCell_aido_cell_100m', 'scPRINT']
    >>> aligned.constant_label
    ''
    >>>
    >>> # Within-model, within-dataset — keys are just categories
    >>> categories = GeneEmbeddingsSet.from_gene_embeddings(
    ...     [ge_adipocyte, ge_tcell, ge_bcell]
    ... )
    >>> categories.keys()
    ['adipocyte (0)', 'T_cell', 'B_cell']
    >>> categories.constant_label
    'scGPT / efthymiou2025'
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
            Mapping from source_label to aligned GeneEmbeddings. Will be
            re-keyed to scoped keys automatically.
        common_gene_ids : List[str]
            The common gene IDs (in the alignment ontology) shared by all
            embeddings, in the order used for alignment.
        aligned_on : str
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
                for i, (a, b) in enumerate(zip(emb_ids, common_gene_ids)):
                    if a != b:
                        raise ValueError(
                            f"Embedding '{key}' gene IDs in ontology '{align_on}' "
                            f"do not match common_gene_ids. "
                            f"First mismatch at index {i}: '{a}' != '{b}'"
                        )
                raise ValueError(
                    f"Embedding '{key}' has {len(emb_ids)} genes but "
                    f"common_gene_ids has {len(common_gene_ids)}"
                )

        # Build metadata and compute scoped keys
        embedding_metadata = _build_embedding_metadata(data)
        source_to_scoped, constant_label = _compute_scoped_keys(embedding_metadata)

        # Add scoped keys to metadata
        embedding_metadata[EMBEDDING_METADATA_FIELDS.SCOPED_KEY] = embedding_metadata[
            EMBEDDING_METADATA_FIELDS.SOURCE_LABEL
        ].map(source_to_scoped)

        # Re-key data dict using scoped keys
        self.data: Dict[str, GeneEmbeddings] = {
            source_to_scoped[source_label]: emb for source_label, emb in data.items()
        }
        self.common_gene_ids: List[str] = list(common_gene_ids)
        self.aligned_on: str = align_on
        self.embedding_metadata: pd.DataFrame = embedding_metadata
        self.constant_label: str = constant_label

    # --- Properties ---

    @property
    def n_embeddings(self) -> int:
        """Number of stored embeddings."""
        return len(self.data)

    @property
    def n_common_genes(self) -> int:
        """Number of common genes."""
        return len(self.common_gene_ids)

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
        for key, emb in self.data.items():
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

    def compare_embeddings(
        self,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Compare embeddings by Spearman correlation of pairwise distance matrices.

        For each pair of embeddings, computes gene-gene cosine distances and then
        calculates the Spearman correlation between the upper triangles of the
        two distance matrices.

        Parameters
        ----------
        device : str or torch.device, optional
            Device for computation (default: None to auto-select).
        verbose : bool, optional
            Whether to print progress (default: False).

        Returns
        -------
        Dict[str, float]
            Dictionary mapping embedding pair names to Spearman correlation
            coefficients (e.g., {"scGPT_vs_scPRINT": 0.42}).

        Raises
        ------
        ValueError
            If fewer than 2 embeddings are in the set.
        """
        if self.n_embeddings < 2:
            raise ValueError(
                "compare_embeddings requires at least 2 embeddings, "
                f"got {self.n_embeddings}"
            )

        aligned_embeddings = {key: emb.embedding for key, emb in self.data.items()}

        return _calculate_embedding_correlations(
            aligned_embeddings, device=device, verbose=verbose
        )

    @classmethod
    def from_gene_embeddings(
        cls,
        embeddings: List[GeneEmbeddings],
        align_on: str = ONTOLOGIES.ENSEMBL_GENE,
        verbose: bool = True,
    ) -> "GeneEmbeddingsSet":
        """Align embeddings to common genes and construct container.

        Uses ``_align_gene_embeddings`` to find common genes across all
        embeddings and reorder each to a consistent row ordering. Keys
        are derived from each embedding's ``source_label`` and then
        automatically scoped to minimal distinguishing labels.

        Parameters
        ----------
        embeddings : List[GeneEmbeddings]
            One or more GeneEmbeddings to align. Each must have unique
            ``source_label`` values. When a single embedding is provided,
            it is wrapped directly without alignment.
        align_on : str, optional
            Column in gene_annotations to align on (default: 'ensembl_gene').
        verbose : bool, optional
            Extra reporting (default: True)

        Returns
        -------
        GeneEmbeddingsSet
            Container with aligned embeddings and scoped keys.

        Raises
        ------
        ValueError
            If no embeddings are provided.
            If source_label values are not unique.
            If no common genes are found (when aligning 2+ embeddings).

        Examples
        --------
        >>> aligned = GeneEmbeddingsSet.from_gene_embeddings(
        ...     [ge_scgpt, ge_aido],
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
            aligned_embeddings = _align_gene_embeddings(
                embeddings, align_on=align_on, verbose=verbose
            )
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

    # --- Dict-like access ---

    def get(self, key: str) -> GeneEmbeddings:
        """Get aligned GeneEmbeddings by scoped key.

        Parameters
        ----------
        key : str
            Scoped key of the embedding to retrieve.

        Returns
        -------
        GeneEmbeddings
            The aligned embedding.

        Raises
        ------
        KeyError
            If key is not found.
        """
        if key not in self.data:
            raise KeyError(
                f"Embedding '{key}' not found. "
                f"Available keys: {list(self.data.keys())}"
            )
        return self.data[key]

    def __getitem__(self, key: str) -> GeneEmbeddings:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __len__(self) -> int:
        return len(self.data)

    def keys(self):
        """Return embedding labels (scoped keys)."""
        return self.data.keys()

    def values(self):
        """Return GeneEmbeddings instances."""
        return self.data.values()

    def items(self):
        """Return (scoped_key, GeneEmbeddings) pairs."""
        return self.data.items()

    # --- Dunder methods ---

    def __repr__(self) -> str:
        keys = list(self.data.keys())
        parts = [
            f"n_embeddings={self.n_embeddings}",
            f"n_common_genes={self.n_common_genes}",
            f"aligned_on='{self.aligned_on}'",
            f"keys={keys}",
        ]
        if self.constant_label:
            parts.insert(0, f"constant_label='{self.constant_label}'")
        return f"GeneEmbeddingsSet({', '.join(parts)})"


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
                    "aligned_on": ge_set.aligned_on,
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

    # properties

    @property
    def disk_name(self) -> str:
        """Get a version of the model label which can be used for a filename."""
        return _get_disk_name(self.model_name, self.model_variant)

    @property
    def full_name(self) -> str:
        """Get full unique identifier."""
        return _get_model_label(self.model_name, self.model_variant)

    # methods

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

        # Serialize static gene embedding
        static_ge_meta = _gene_embeddings_to_save_dict(
            self.weights.static_gene_embeddings
        )

        weights_dict = {
            FM_DEFS.STATIC_GENE_EMBEDDINGS: self.weights.static_gene_embeddings.embedding,
            FM_DEFS.ATTENTION_WEIGHTS: attention_weights_dict,
        }

        # Iterate: DatasetGeneEmbeddings -> GeneEmbeddingsSet -> GeneEmbeddings
        dataset_gene_embeddings_metadata = []
        if self.dataset_gene_embeddings:
            all_gene_embeddings = self.dataset_gene_embeddings.all_gene_embeddings()
            logger.info(f"Saving {len(all_gene_embeddings)} dataset gene embeddings")

            for i, ge in enumerate(all_gene_embeddings):
                weights_dict[f"dataset_gene_embeddings_{i}"] = ge.embedding
                dataset_gene_embeddings_metadata.append(
                    _gene_embeddings_to_save_dict(ge)
                )

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

        combined_metadata = {
            FM_DEFS.MODEL_METADATA: model_metadata,
            FM_DEFS.GENE_ANNOTATIONS: self.gene_annotations.to_dict("records"),
            FM_DEFS.STATIC_GENE_EMBEDDINGS: static_ge_meta,
            FM_DEFS.DATASET_GENE_EMBEDDINGS: dataset_gene_embeddings_metadata,
        }

        with open(metadata_path, "w") as f:
            json.dump(combined_metadata, f, indent=2)

        logger.info("Successfully saved all results")

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
        if len(v) < 2:
            raise ValueError("At least 2 models are required for cross-model analysis")
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


class AttendedEmbeddings:
    """A single gene embedding matrix paired with its model's attention machinery.

    Pairs one GeneEmbeddings instance (static or expression-contextualized)
    with a reference to the FoundationModel that produced it. This enables
    computing attention patterns using this specific embedding's vectors
    with the model's attention weight matrices.

    Typically created by AttendedEmbeddingsSet rather than directly by users.

    Attributes
    ----------
    gene_embeddings : GeneEmbeddings
        The gene embedding matrix and associated metadata.
    foundation_model : FoundationModel
        Reference to the source model (for attention layers, n_heads,
        gene_annotations, ordered_vocabulary).

    Properties
    ----------
    embedding : np.ndarray
        The embedding matrix (shortcut to gene_embeddings.embedding).
    n_genes : int
        Number of genes.
    embed_dim : int
        Embedding dimensionality.
    n_layers : int
        Number of attention layers (from the model).
    n_heads : int
        Number of attention heads (from the model).
    attention_layers : List[AttentionLayer]
        Attention layers (from the model).
    model_name : str
        Model display name (from the model).

    Examples
    --------
    >>> ae = AttendedEmbeddings(gene_embeddings=ge, foundation_model=model)
    >>> attn = ae.compute_attention(layer_idx=0)
    >>> consensus = ae.compute_consensus_attention()
    """

    def __init__(
        self,
        gene_embeddings: GeneEmbeddings,
        foundation_model: FoundationModel,
    ):
        if not isinstance(gene_embeddings, GeneEmbeddings):
            raise TypeError(
                f"gene_embeddings must be a GeneEmbeddings, "
                f"got {type(gene_embeddings)}"
            )
        if not isinstance(foundation_model, FoundationModel):
            raise TypeError(
                f"foundation_model must be a FoundationModel, "
                f"got {type(foundation_model)}"
            )

        # Validate that the embedding's model matches the foundation model
        emb_label = _get_model_label(
            gene_embeddings.model_name, gene_embeddings.model_variant
        )
        if emb_label != foundation_model.full_name:
            raise ValueError(
                f"Embedding model label '{emb_label}' does not match "
                f"foundation model '{foundation_model.full_name}'"
            )

        if len(foundation_model.weights.attention_layers) == 0:
            raise ValueError(
                f"Model '{foundation_model.full_name}' has no attention layers."
            )

        self.gene_embeddings = gene_embeddings
        self.foundation_model = foundation_model

    # --- Properties (shortcuts) ---

    @property
    def embedding(self) -> np.ndarray:
        """The embedding matrix."""
        return self.gene_embeddings.embedding

    @property
    def n_genes(self) -> int:
        """Number of genes."""
        return self.gene_embeddings.n_genes

    @property
    def embed_dim(self) -> int:
        """Embedding dimensionality."""
        return self.gene_embeddings.embed_dim

    @property
    def n_layers(self) -> int:
        """Number of attention layers."""
        return self.foundation_model.n_layers

    @property
    def n_heads(self) -> int:
        """Number of attention heads."""
        return self.foundation_model.n_heads

    @property
    def attention_layers(self) -> List[AttentionLayer]:
        """Attention layers from the model."""
        return self.foundation_model.weights.attention_layers

    @property
    def model_name(self) -> str:
        """Model display name."""
        return self.foundation_model.full_name

    def compare_layer_attention_consistency(
        self,
        top_k: int,
        by_absolute_value: bool,
        ignore_self_attention: bool,
        target_ids: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:

        _, gene_ids = self.gene_embeddings.get_gene_mask(target_ids)

        top_k_attention_edges = self.get_top_attentions(
            k=top_k,
            target_ids=gene_ids,
            by_absolute_value=by_absolute_value,
            ignore_self_attention=ignore_self_attention,
            verbose=verbose,
        )
        # re-extract the top-k edges across all layers
        distinct_top_edges = top_k_attention_edges[
            ["from_gene", "to_gene"]
        ].drop_duplicates()

        # extract the attention scores for the distinct top-k edges across all layers
        top_k_union = self.get_specific_attentions(
            distinct_top_edges,
            target_ids=gene_ids,
            compute_ranks=True,
            by_absolute_value=by_absolute_value,
            verbose=verbose,
        )

        wide_top_k_union = top_k_union.pivot(
            index=["from_gene", "to_gene"], columns="layer", values="attention"
        )

        corr = compute_correlation_matrix(wide_top_k_union.to_numpy())
        top_k_ranks = compare_top_k_union_ranks(
            top_k_union,
            grouping_vars=["layer"],
            defining_vars=["from_gene", "to_gene"],
            max_rank=len(gene_ids) ** 2,
            top_k=top_k,
            rank_col="attention_rank",
        )

        return corr, top_k_ranks

    def compute_attention(
        self,
        layer_idx: int,
        target_ids: Optional[List[str]] = None,
        apply_softmax: bool = True,
        return_tensor: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Union[Tensor, np.ndarray]:
        """Compute attention scores for a specific layer.

        Uses this instance's embedding matrix with the model's attention
        weight matrices to compute the attention pattern.

        Parameters
        ----------
        layer_idx : int
            Index of the layer to compute attention for.
        target_ids : List[str], optional
            Subset of gene IDs to compute attention for. Results will be
            ordered to match target_ids. Must be a subset of this embedding's
            gene IDs. If None, uses all genes in embedding order.
        apply_softmax : bool, optional
            If True, apply softmax to get attention probabilities (default: True).
            If False, return raw scores (Q @ K.T / sqrt(d_k)).
        return_tensor : bool, optional
            If True, return torch.Tensor (default: False).
        device : str or torch.device, optional
            Device for computation (default: None to auto-select).

        Returns
        -------
        torch.Tensor or np.ndarray
            Attention matrix of shape (n_target, n_target) if target_ids
            is provided, otherwise (n_genes, n_genes). Rows and columns
            are ordered to match target_ids when provided.

        Raises
        ------
        ValueError
            If layer_idx is out of range or target_ids contains unknown genes.
        """
        if layer_idx >= self.n_layers:
            raise ValueError(
                f"Layer index {layer_idx} out of range "
                f"(model has {self.n_layers} layers)"
            )

        gene_mask, gene_ids = self.gene_embeddings.get_gene_mask(target_ids)

        embeddings = self.embedding
        if gene_mask is not None:
            embeddings = embeddings[gene_mask]

        layer = self.attention_layers[layer_idx]

        attention = layer.compute_attention_pattern(
            embeddings=embeddings,
            n_heads=self.n_heads,
            apply_softmax=apply_softmax,
            return_tensor=True,
            device=device,
        )

        # Reorder from embedding order to target_ids order if needed
        if target_ids is not None:
            masked_gene_ids = [
                gid
                for gid, m in zip(self.gene_embeddings.ordered_gene_ids, gene_mask)
                if m
            ]
            masked_id_to_idx = {gid: i for i, gid in enumerate(masked_gene_ids)}
            reorder_indices = [masked_id_to_idx[gid] for gid in target_ids]
            attention = attention[reorder_indices, :][:, reorder_indices]

        if return_tensor:
            return attention
        else:
            return attention.cpu().numpy()

    def compute_consensus_attention(
        self,
        target_ids: Optional[List[str]] = None,
        consensus_method: str = FM_LAYER_CONSENSUS_METHODS.ABSOLUTE_ARGMAX,
        apply_softmax: bool = False,
        return_layer_indices: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute consensus attention across all layers.

        For each gene pair, aggregates attention values across layers using the
        specified consensus method. The default method ("absolute-argmax") finds
        the layer with the strongest attention (by absolute value) and returns
        that value with its original sign preserved.

        Parameters
        ----------
        target_ids : List[str], optional
            Subset of gene IDs to compute consensus attention for. Results will be
            ordered to match target_ids. Must be a subset of this embedding's
            gene IDs. If None, uses all genes in embedding order.
        consensus_method : str, optional
            Method for aggregating across layers:
            - "absolute-argmax" (default): Layer with max absolute value, sign preserved
            - "max": Layer with maximum value
            - "sum": Sum across all layers
        apply_softmax : bool, optional
            If True, apply softmax per layer (default: False).
        return_layer_indices : bool, optional
            If True, also return which layer had max attention (default: False).
        device : str or torch.device, optional
            Device for computation (default: None to auto-select).

        Returns
        -------
        torch.Tensor
            Consensus attention of shape (n_genes, n_genes).
        torch.Tensor (optional)
            If return_layer_indices=True, layer indices of shape (n_genes, n_genes).
        """

        all_attention = torch.zeros(
            (self.n_genes, self.n_genes, self.n_layers), dtype=torch.float32
        )

        for layer_idx in range(self.n_layers):
            attention = self.compute_attention(
                layer_idx=layer_idx,
                target_ids=target_ids,
                apply_softmax=apply_softmax,
                return_tensor=True,
                device=device,
            )
            all_attention[:, :, layer_idx] = attention

        if consensus_method == FM_LAYER_CONSENSUS_METHODS.ABSOLUTE_ARGMAX:
            return compute_max_abs_over_z(
                all_attention, return_indices=return_layer_indices
            )
        elif consensus_method == FM_LAYER_CONSENSUS_METHODS.MAX:
            return compute_max_over_z(
                all_attention, return_indices=return_layer_indices
            )
        elif consensus_method == FM_LAYER_CONSENSUS_METHODS.SUM:
            if return_layer_indices:
                return all_attention.sum(dim=2), torch.zeros(
                    (self.n_genes, self.n_genes), dtype=torch.long
                )
            return all_attention.sum(dim=2)
        else:
            raise ValueError(
                f"Unknown consensus_method '{consensus_method}'. "
                f"Supported methods: {VALID_FM_LAYER_CONSENSUS_METHODS}"
            )

    def get_specific_attentions(
        self,
        edge_list: pd.DataFrame,
        layer_indices: Optional[List[int]] = None,
        target_ids: Optional[List[str]] = None,
        apply_softmax: bool = False,
        compute_ranks: bool = False,
        by_absolute_value: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Extract specific attention values across layers for given edges.

        Extracts the exact attention values for specific gene pairs across
        specified layers. Useful for analyzing how specific relationships
        vary across layers.

        Parameters
        ----------
        edge_list : pd.DataFrame
            DataFrame with at minimum 'from_gene' and 'to_gene' columns.
        layer_indices : List[int], optional
            Layers to extract from. If None, uses all layers.
        target_ids : List[str], optional
            Subset of gene IDs to use for attention computation. Must be a
            subset of this embedding's gene IDs. If None, uses all genes.
        apply_softmax : bool, optional
            If True, use softmax-normalized attention (default: False).
        compute_ranks : bool, optional
            If True, add attention ranks to output (default: False).
        by_absolute_value : bool, optional
            If True, rank by absolute value (default: True).
            Only used if compute_ranks=True.
        device : str or torch.device, optional
            Device for computation (default: None to auto-select).
        verbose : bool, optional
            Print progress (default: False).

        Returns
        -------
        pd.DataFrame
            Columns: from_gene, to_gene, layer, attention,
            and optionally attention_rank.
        """
        device = ensure_device(device, allow_autoselect=True)

        _, gene_ids = self.gene_embeddings.get_gene_mask(target_ids)

        # Convert edge list to indices ONCE
        edge_df = _edgelist_to_indices(
            edge_list=edge_list,
            gene_ids=gene_ids,
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
            from_idx_tensor = (
                torch.from_numpy(edge_df[FM_EDGELIST.FROM_IDX].values).long().to(device)
            )
            to_idx_tensor = (
                torch.from_numpy(edge_df[FM_EDGELIST.TO_IDX].values).long().to(device)
            )

            for layer_idx in layer_indices:
                if verbose:
                    logger.info(f"Extracting attentions from layer {layer_idx}...")

                attention = self.compute_attention(
                    layer_idx=layer_idx,
                    target_ids=target_ids,
                    apply_softmax=apply_softmax,
                    return_tensor=True,
                    device=device,
                )

                # Extract edges using tensor indexing
                edge_attentions = attention[from_idx_tensor, to_idx_tensor]

                layer_df = edge_df[[FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE]].copy()
                layer_df[FM_EDGELIST.LAYER] = layer_idx
                layer_df[FM_EDGELIST.ATTENTION] = edge_attentions.cpu().numpy()

                if compute_ranks:
                    if verbose:
                        logger.info(f"Calculating ranks for layer {layer_idx}...")

                    edge_ranks = compute_tensor_ranks_for_indices(
                        attention,
                        (from_idx_tensor, to_idx_tensor),
                        by_absolute_value=by_absolute_value,
                    )
                    layer_df[FM_EDGELIST.ATTENTION_RANK] = edge_ranks.cpu().numpy()

                results.append(layer_df)
                cleanup_tensors(attention, edge_attentions)

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
        apply_softmax: bool = False,
        by_absolute_value: bool = True,
        compute_ranks: bool = False,
        ignore_self_attention: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Extract top-k strongest attention edges across layers.

        For each layer, identifies the k gene pairs with highest attention
        values and returns them as a DataFrame.

        Parameters
        ----------
        k : int
            Number of top edges to extract per layer.
        layer_indices : List[int], optional
            Layers to analyze. If None, uses all layers.
        target_ids : List[str], optional
            Subset of gene identifiers to analyze. Must be a subset of
            this embedding's gene IDs. If None, uses all genes.
        apply_softmax : bool, optional
            If True, use softmax-normalized attention (default: False).
        by_absolute_value : bool, optional
            If True, rank by absolute attention value (default: True).
        compute_ranks : bool, optional
            If True, add attention ranks to output (default: False).
        ignore_self_attention : bool, optional
            If True, exclude self-attention edges (default: False).
        device : str or torch.device, optional
            Device for computation (default: None to auto-select).
        verbose : bool, optional
            Print progress (default: False).

        Returns
        -------
        pd.DataFrame
            Columns: layer, from_idx, to_idx, from_gene, to_gene, attention,
            and optionally attention_rank.
        """

        device = ensure_device(device, allow_autoselect=True)

        _, gene_ids = self.gene_embeddings.get_gene_mask(target_ids)

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
            for layer_idx in layer_indices:
                if verbose:
                    value_type = "absolute value" if by_absolute_value else "raw value"
                    logger.info(
                        f"Extracting top-{k} edges from layer {layer_idx} "
                        f"by {value_type}..."
                    )

                attention = self.compute_attention(
                    layer_idx=layer_idx,
                    target_ids=target_ids,
                    apply_softmax=apply_softmax,
                    return_tensor=True,
                    device=device,
                )

                layer_df = _find_top_k_edges_in_attention_layer(
                    attention=attention,
                    k=k,
                    layer_idx=layer_idx,
                    gene_ids=gene_ids,
                    by_absolute_value=by_absolute_value,
                    ignore_self_attention=ignore_self_attention,
                )

                results.append(layer_df)
                cleanup_tensors(attention)

        all_edges = pd.concat(results, ignore_index=True)

        if compute_ranks:
            all_edges[FM_EDGELIST.ATTENTION_RANK] = calculate_ranks(
                df=all_edges,
                value_col=FM_EDGELIST.ATTENTION,
                by_absolute_value=by_absolute_value,
                grouping_vars=FM_EDGELIST.LAYER,
            )

        if verbose:
            logger.info(
                f"Extracted {len(all_edges)} total edges "
                f"across {len(layer_indices)} layers"
            )

        return all_edges

    # --- Dunder ---

    def __repr__(self) -> str:
        return (
            f"AttendedEmbeddings("
            f"model={self.model_name}, "
            f"source={self.gene_embeddings.source_label}, "
            f"n_genes={self.n_genes}, "
            f"embed_dim={self.embed_dim}, "
            f"n_layers={self.n_layers}"
            f")"
        )


class AttendedEmbeddingsSet:
    """Aligned gene embeddings paired with attention machinery for cross-embedding analysis.

    Bundles a GeneEmbeddingsSet (aligned embeddings from one or more models/datasets)
    with references to the FoundationModel instances that produced them. This enables
    computing attention patterns from any embedding using its model's attention weights,
    and comparing those patterns across embeddings.

    The key insight is that attention computation requires:
    1. An embedding matrix (from GeneEmbeddings — could be static or expression-contextualized)
    2. Attention weight matrices (W_q, W_k, W_v, W_o from AttentionLayer)
    3. n_heads (from the model metadata)

    Items 2 and 3 are always model-level properties, while item 1 varies per embedding.
    This class manages the mapping from each embedding to its corresponding model.

    Parameters
    ----------
    embeddings_set : GeneEmbeddingsSet
        Aligned gene embeddings. All embeddings must share the same common genes
        in the same row order.
    foundation_models : FoundationModels
        Container of FoundationModel instances. Each embedding in embeddings_set
        must map to exactly one model (via model_name + model_variant -> full_name).

    Attributes
    ----------
    embeddings_set : GeneEmbeddingsSet
        The aligned embeddings.
    foundation_models : FoundationModels
        The source foundation models (held by reference).
    common_gene_ids : List[str]
        Gene IDs shared across all embeddings (delegates to embeddings_set).
    embedding_to_model_map : Dict[str, str]
        Mapping from embedding key (source_label) to model full_name.

    Properties
    ----------
    n_embeddings : int
        Number of embeddings in the set.
    n_common_genes : int
        Number of common genes.
    embedding_keys : List[str]
        Labels for each embedding.
    model_names : List[str]
        Unique model names referenced by embeddings.

    Public Methods
    --------------
    from_static(foundation_models: FoundationModels, align_on: str = ONTOLOGIES.ENSEMBL_GENE, verbose: bool = True) -> "AttendedEmbeddingsSet":
        Create AttendedEmbeddings from the static gene embeddings of all models.
    from_expression(foundation_models: FoundationModels, dataset_name: str, category: str, align_on: str = ONTOLOGIES.ENSEMBL_GENE, verbose: bool = True) -> "AttendedEmbeddingsSet":
        Create AttendedEmbeddings from expression-contextualized embeddings.
    get_consensus_attention(k: int = 10000, target_ids: Optional[List[str]] = None, consensus_method: str = FM_LAYER_CONSENSUS_METHODS.ABSOLUTE_ARGMAX, by_absolute_value: bool = True, reextract_union: bool = False, apply_softmax: bool = False, compute_ranks: bool = False, ignore_self_attention: bool = False, return_original_and_reextracted: bool = False, device: Optional[Union[str, torch.device]] = None, verbose: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        Compute consensus attention across all layers for each embedding.
    get_consensus_top_attentions(k: int = 10000, target_ids: Optional[List[str]] = None, consensus_method: str = FM_LAYER_CONSENSUS_METHODS.ABSOLUTE_ARGMAX, by_absolute_value: bool = True, reextract_union: bool = False, apply_softmax: bool = False, compute_ranks: bool = False, ignore_self_attention: bool = False, return_original_and_reextracted: bool = False, device: Optional[Union[str, torch.device]] = None, verbose: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        Extract top-k consensus attention edges across embeddings.
    get_specific_attentions(edges: pd.DataFrame, target_ids: Optional[List[str]] = None, apply_softmax: bool = False, compute_ranks: bool = False, by_absolute_value: bool = True, verbose: bool = False) -> pd.DataFrame:
        Extract specific attention edges across layers.
    get_top_attentions(k: int, layer_indices: Optional[List[int]] = None, target_ids: Optional[List[str]] = None, apply_softmax: bool = False, by_absolute_value: bool = True, compute_ranks: bool = False, ignore_self_attention: bool = False, device: Optional[Union[str, torch.device]] = None, verbose: bool = False) -> pd.DataFrame:
        Extract top-k strongest attention edges across layers.

    Examples
    --------
    >>> # From static embeddings (most common entry point)
    >>> attended = AttendedEmbeddingsSet.from_static(foundation_models)
    >>> attended.n_common_genes
    15234

    >>> # From expression-contextualized embeddings
    >>> attended = AttendedEmbeddingsSet.from_expression(
    ...     foundation_models, dataset_name="efthymiou2025", category="adipocyte (0)"
    ... )

    >>> # From a pre-built GeneEmbeddingsSet
    >>> attended = AttendedEmbeddingsSet(embeddings_set, foundation_models)
    """

    def __init__(
        self,
        embeddings_set: GeneEmbeddingsSet,
        foundation_models: FoundationModels,
    ):
        # Validate types
        if not isinstance(embeddings_set, GeneEmbeddingsSet):
            raise TypeError(
                f"embeddings_set must be a GeneEmbeddingsSet, "
                f"got {type(embeddings_set)}"
            )
        if not isinstance(foundation_models, FoundationModels):
            raise TypeError(
                f"foundation_models must be a FoundationModels, "
                f"got {type(foundation_models)}"
            )

        # Build mapping: embedding key -> model full_name
        # Each GeneEmbeddings carries model_name and model_variant;
        # combine them the same way FoundationModel.full_name does.
        embedding_to_model: Dict[str, str] = {}
        available_model_names = set(foundation_models.model_names)

        for key, emb in embeddings_set.items():
            if emb.model_name is None:
                raise ValueError(
                    f"Embedding '{key}' has no model_name set. "
                    f"Cannot map to a FoundationModel."
                )

            full_name = _get_model_label(emb.model_name, emb.model_variant)

            if full_name not in available_model_names:
                raise ValueError(
                    f"Embedding '{key}' maps to model '{full_name}', "
                    f"but no matching FoundationModel was found. "
                    f"Available models: {sorted(available_model_names)}"
                )

            embedding_to_model[key] = full_name

        # Validate that every referenced model has attention layers
        referenced_models = set(embedding_to_model.values())
        for model_name in referenced_models:
            model = foundation_models.get_model(model_name)
            if len(model.weights.attention_layers) == 0:
                raise ValueError(
                    f"Model '{model_name}' has no attention layers. "
                    f"AttendedEmbeddings requires models with attention weights."
                )

        # Build AttendedEmbeddings instances
        attended: Dict[str, AttendedEmbeddings] = {}
        for key in embeddings_set.keys():
            full_name = embedding_to_model[key]
            model = foundation_models.get_model(full_name)
            attended[key] = AttendedEmbeddings(
                gene_embeddings=embeddings_set[key],
                foundation_model=model,
            )

        self.embeddings_set = embeddings_set
        self.attended_embeddings = attended

    # --- Properties ---

    @property
    def common_gene_ids(self) -> List[str]:
        """Gene IDs shared across all embeddings."""
        return self.embeddings_set.common_gene_ids

    @property
    def n_embeddings(self) -> int:
        """Number of embeddings."""
        return len(self.attended_embeddings)

    @property
    def n_common_genes(self) -> int:
        """Number of common genes."""
        return self.embeddings_set.n_common_genes

    @property
    def embedding_keys(self) -> List[str]:
        """Labels for each embedding (scoped keys)."""
        return list(self.attended_embeddings.keys())

    @property
    def model_names(self) -> List[str]:
        """Unique model names referenced by embeddings (preserves order)."""
        seen = set()
        result = []
        for ae in self.attended_embeddings.values():
            if ae.model_name not in seen:
                seen.add(ae.model_name)
                result.append(ae.model_name)
        return result

    def compare(
        self,
        comparison_types: List[str] = VALID_COMPARE_EMBEDDINGS_COMPARISONS,
        top_k: int = 10000,
        consensus_method: str = FM_LAYER_CONSENSUS_METHODS.SUM,
        by_absolute_value: bool = False,
        ignore_self_attention: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:

        invalid_comparison_types = set(comparison_types) - set(
            VALID_COMPARE_EMBEDDINGS_COMPARISONS
        )
        if invalid_comparison_types:
            raise ValueError(
                f"The following requested comparison types are not valid: {invalid_comparison_types}. Valid comparison types: {VALID_COMPARE_EMBEDDINGS_COMPARISONS}."
            )

        if consensus_method not in VALID_FM_LAYER_CONSENSUS_METHODS:
            raise ValueError(
                f"The requested consensus method is not valid: {consensus_method}. Valid consensus methods: {VALID_FM_LAYER_CONSENSUS_METHODS}."
            )

        # precalculate and cache operations that take more than a minute or so
        comparisons = dict()
        n_genes = self.n_common_genes

        # gene embedding correlations
        if (
            COMPARE_EMBEDDINGS_COMPARISONS.GENE_EMBEDDING_CORRELATIONS
            in comparison_types
        ):
            logger.info("Calculating gene embedding correlations...")
            comparisons[COMPARE_EMBEDDINGS_COMPARISONS.GENE_EMBEDDING_CORRELATIONS] = (
                self.embeddings_set.compare_embeddings(verbose=verbose)
            )
        else:
            comparisons[COMPARE_EMBEDDINGS_COMPARISONS.GENE_EMBEDDING_CORRELATIONS] = (
                None
            )

        # within model layer x layer comparisons (correlations and rank agreement)
        if COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_CORRELATIONS in comparison_types:
            logger.info("Calculating within model layer x layer comparisons...")
            (
                comparisons[COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_CORRELATIONS],
                comparisons[COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_RANK_AGREEMENT],
            ) = self.get_within_embeddings_layer_comparisons(
                top_k, ignore_self_attention, by_absolute_value, verbose
            )
        else:
            comparisons[COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_CORRELATIONS] = None
            comparisons[COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_RANK_AGREEMENT] = (
                None
            )

        # cross model x layer top attentions
        if (
            COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_TOP_ATTENTIONS
            in comparison_types
        ):
            logger.info("Calculating cross model x layer top attentions...")
            comparisons[
                COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_TOP_ATTENTIONS
            ] = self.get_top_attentions(
                k=top_k,
                by_absolute_value=by_absolute_value,
                reextract_union=True,
                compute_ranks=True,
                ignore_self_attention=ignore_self_attention,
                verbose=verbose,
            )
        else:
            comparisons[
                COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_TOP_ATTENTIONS
            ] = None

        if (
            COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_RANK_AGREEMENT
            in comparison_types
        ):
            logger.info(
                "Comparing ranks of topK attentions in one model/layer to all other models/layers..."
            )
            comparisons[
                COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_RANK_AGREEMENT
            ] = compare_top_k_union_ranks(
                comparisons[
                    COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_TOP_ATTENTIONS
                ],
                grouping_vars=[FM_EDGELIST.MODEL, FM_EDGELIST.LAYER],
                defining_vars=[FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE],
                max_rank=n_genes**2,
                top_k=top_k,
                rank_col=FM_EDGELIST.ATTENTION_RANK,
            )
        else:
            comparisons[
                COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_RANK_AGREEMENT
            ] = None

        if (
            COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS
            in comparison_types
        ):
            logger.info("Calculating cross model consensus top attentions...")
            comparisons[
                COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS
            ] = self.get_consensus_top_attentions(
                k=top_k,
                consensus_method=consensus_method,
                by_absolute_value=by_absolute_value,
                reextract_union=True,
                compute_ranks=True,
                ignore_self_attention=ignore_self_attention,
                verbose=verbose,
            )
        else:
            comparisons[
                COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS
            ] = None

        if (
            COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS_RANK_AGREEMENT
            in comparison_types
        ):
            comparisons[
                COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS_RANK_AGREEMENT
            ] = compare_top_k_union_ranks(
                comparisons[
                    COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS
                ],
                grouping_vars=[FM_EDGELIST.MODEL],
                defining_vars=[FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE],
                max_rank=n_genes**2,
                top_k=top_k,
                rank_col=FM_EDGELIST.ATTENTION_RANK,
            )
        else:
            comparisons[
                COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS_RANK_AGREEMENT
            ] = None

        comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS] = {
            COMPARE_EMBEDDINGS_SETTINGS.TOP_K: top_k,
            COMPARE_EMBEDDINGS_SETTINGS.CONSENSUS_METHOD: consensus_method,
            COMPARE_EMBEDDINGS_SETTINGS.BY_ABSOLUTE_VALUE: by_absolute_value,
            COMPARE_EMBEDDINGS_SETTINGS.IGNORE_SELF_ATTENTION: ignore_self_attention,
            COMPARE_EMBEDDINGS_SETTINGS.EMBEDDING_KEYS: self.embedding_keys,
            COMPARE_EMBEDDINGS_SETTINGS.N_GENES: n_genes,
        }

        return comparisons

    @classmethod
    def from_expression(
        cls,
        foundation_models: FoundationModels,
        dataset_name: str,
        category: str,
        align_on: str = ONTOLOGIES.ENSEMBL_GENE,
        verbose: bool = True,
    ) -> "AttendedEmbeddingsSet":
        """Create AttendedEmbeddings from expression-contextualized embeddings.

        For each model, retrieves the GeneEmbeddings for the specified
        dataset and category, aligns them to common genes, and wires up
        the attention references.

        Parameters
        ----------
        foundation_models : FoundationModels
            Container with 2+ loaded foundation models. Each model must have
            dataset_gene_embeddings containing the specified dataset and category.
        dataset_name : str
            Name of the expression dataset (e.g., 'efthymiou2025').
        category : str
            Category within the dataset (e.g., 'adipocyte (0)', 'T_cell').
        align_on : str, optional
            Ontology column for gene alignment (default: 'ensembl_gene').
        verbose : bool, optional
            Extra reporting (default: True)

        Returns
        -------
        AttendedEmbeddingsSet
            Ready for analysis with aligned expression embeddings.

        Raises
        ------
        ValueError
            If any model lacks dataset_gene_embeddings.
            If the specified dataset is not found in any model.
            If the specified category is not found in any model's dataset.

        Examples
        --------
        >>> models = FoundationModels.load_multiple(dir, ['scGPT', 'scPRINT'])
        >>> attended = AttendedEmbeddingsSet.from_expression(
        ...     models, dataset_name="efthymiou2025", category="adipocyte (0)"
        ... )
        >>> attended.n_embeddings
        2
        """
        if not isinstance(foundation_models, FoundationModels):
            raise TypeError(
                f"foundation_models must be a FoundationModels, "
                f"got {type(foundation_models)}"
            )

        expression_embeddings: List[GeneEmbeddings] = []

        for model in foundation_models.models:
            # Validate model has expression embeddings
            if model.dataset_gene_embeddings is None:
                raise ValueError(
                    f"Model '{model.full_name}' has no dataset_gene_embeddings. "
                    f"Cannot extract expression embeddings."
                )

            # Validate dataset exists
            if dataset_name not in model.dataset_gene_embeddings:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found in model "
                    f"'{model.full_name}'. Available datasets: "
                    f"{model.dataset_gene_embeddings.dataset_names}"
                )

            ge_set = model.dataset_gene_embeddings[dataset_name]

            if category not in ge_set:
                raise ValueError(
                    f"Category '{category}' not found in dataset '{dataset_name}' "
                    f"for model '{model.full_name}'. "
                    f"Available categories: {list(ge_set.keys())}"
                )

            expression_embeddings.append(ge_set[category])

        embeddings_set = GeneEmbeddingsSet.from_gene_embeddings(
            expression_embeddings, align_on=align_on, verbose=verbose
        )

        return cls(
            embeddings_set=embeddings_set,
            foundation_models=foundation_models,
        )

    @classmethod
    def from_static(
        cls,
        foundation_models: FoundationModels,
        align_on: str = ONTOLOGIES.ENSEMBL_GENE,
        verbose: bool = True,
    ) -> "AttendedEmbeddingsSet":
        """Create AttendedEmbeddings from the static gene embeddings of all models.

        Extracts each model's static gene embedding, aligns them to common genes,
        and wires up the attention references. This replicates the current
        FoundationModels cross-model analysis workflow.

        Parameters
        ----------
        foundation_models : FoundationModels
            Container with 2+ loaded foundation models.
        align_on : str, optional
            Ontology column for gene alignment (default: 'ensembl_gene').
        verbose : bool, optional
            Extra reporting (default: True)

        Returns
        -------
        AttendedEmbeddingsSet
            Ready for analysis with aligned static embeddings.

        Examples
        --------
        >>> models = FoundationModels.load_multiple(dir, ['scGPT', 'scPRINT'])
        >>> attended = AttendedEmbeddingsSet.from_static(models)
        >>> attended.n_common_genes
        15234
        >>> attended.embedding_keys
        ['scGPT', 'scPRINT']
        """
        if not isinstance(foundation_models, FoundationModels):
            raise TypeError(
                f"foundation_models must be a FoundationModels, "
                f"got {type(foundation_models)}"
            )

        static_embeddings = [
            model.weights.static_gene_embeddings for model in foundation_models.models
        ]

        embeddings_set = GeneEmbeddingsSet.from_gene_embeddings(
            static_embeddings, align_on=align_on, verbose=verbose
        )

        return cls(
            embeddings_set=embeddings_set,
            foundation_models=foundation_models,
        )

    def get_consensus_attentions(
        self,
        target_ids: Optional[List[str]] = None,
        consensus_method: str = FM_LAYER_CONSENSUS_METHODS.ABSOLUTE_ARGMAX,
        apply_softmax: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tensor:
        """Compute consensus attention for each embedding in the set.

        For each embedding, computes consensus attention across all layers
        using the specified method.

        Parameters
        ----------
        target_ids : List[str], optional
            Subset of gene IDs. If None, uses all common genes.
        consensus_method : str, optional
            Aggregation method across layers (default: 'absolute-argmax').
        apply_softmax : bool, optional
            If True, apply softmax per layer (default: False).
        device : str or torch.device, optional
            Device for computation (default: None to auto-select).

        Returns
        -------
        Tensor
            3D tensor of shape (n_embeddings, n_genes, n_genes).
        """
        gene_ids = target_ids if target_ids is not None else self.common_gene_ids
        n_genes = len(gene_ids)

        cross_embedding_attention = torch.zeros(
            (self.n_embeddings, n_genes, n_genes), dtype=torch.float32
        )

        for i, (key, ae) in enumerate(self.attended_embeddings.items()):
            logger.info(f"Computing consensus attention for {key}...")

            attention = ae.compute_consensus_attention(
                target_ids=target_ids,
                consensus_method=consensus_method,
                apply_softmax=apply_softmax,
                device=device,
            )

            cross_embedding_attention[i] = attention

        return cross_embedding_attention

    def get_consensus_top_attentions(
        self,
        k: int = 10000,
        target_ids: Optional[List[str]] = None,
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
        """Extract top-k consensus attention edges across embeddings.

        For each embedding:
        1. Compute consensus attention across all layers
        2. Extract top-k strongest edges

        Optionally re-extract the union of all top edges from every
        embedding's consensus.

        Parameters
        ----------
        k : int, optional
            Number of top edges per embedding (default: 10000).
        target_ids : List[str], optional
            Subset of gene IDs. If None, uses all common genes.
        consensus_method : str, optional
            Aggregation method across layers (default: 'absolute-argmax').
        by_absolute_value : bool, optional
            If True, rank by absolute value (default: True).
        reextract_union : bool, optional
            If True, re-extract union from all embeddings (default: False).
        apply_softmax : bool, optional
            If True, use softmax attention (default: False).
        compute_ranks : bool, optional
            If True, add ranks to output (default: False).
        ignore_self_attention : bool, optional
            If True, exclude self-attention edges (default: False).
        return_original_and_reextracted : bool, optional
            If True and reextract_union=True, return tuple (default: False).
        device : str or torch.device, optional
            Device for computation (default: None to auto-select).
        verbose : bool, optional
            Print progress (default: False).

        Returns
        -------
        pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
            DataFrame with columns: from_idx, to_idx, from_gene, to_gene,
            attention, model, and optionally attention_rank.
        """
        gene_ids = target_ids if target_ids is not None else self.common_gene_ids

        if verbose:
            logger.info(
                f"Computing consensus attention across {self.n_embeddings} embeddings "
                f"for {len(gene_ids)} genes..."
            )

        # Phase 1: Compute consensus for all embeddings
        all_consensus = self.get_consensus_attentions(
            target_ids=target_ids,
            consensus_method=consensus_method,
            apply_softmax=apply_softmax,
            device=device,
        )

        # Extract top-k from each embedding's consensus
        top_edges_list = []
        keys = list(self.attended_embeddings.keys())

        for i, key in enumerate(keys):
            ae = self.attended_embeddings[key]
            if verbose:
                logger.info(f"Extracting top-{k} edges from {key}...")

            model_top_k = _find_top_k_edges_in_attention_layer(
                attention=all_consensus[i],
                k=k,
                layer_idx=None,
                gene_ids=gene_ids,
                by_absolute_value=by_absolute_value,
                ignore_self_attention=ignore_self_attention,
            )
            model_top_k[FM_EDGELIST.MODEL] = ae.model_name

            top_edges_list.append(model_top_k)

        all_top_edges = pd.concat(top_edges_list, ignore_index=True)

        if verbose:
            logger.info(
                f"Extracted {len(all_top_edges)} total edges "
                f"({k} per embedding × {self.n_embeddings} embeddings)"
            )

        if not reextract_union:
            if return_original_and_reextracted:
                logger.warning(
                    "return_original_and_reextracted=True but reextract_union=False, "
                    "returning original top-k edges only"
                )
            return all_top_edges

        # Phase 2: Union re-extraction
        unique_edges = all_top_edges[
            [FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE]
        ].drop_duplicates()

        if verbose:
            logger.info(
                f"Re-extracting {len(unique_edges)} unique edges from all embeddings..."
            )

        edge_df = _edgelist_to_indices(
            edge_list=unique_edges,
            gene_ids=gene_ids,
            verbose=verbose,
        )

        # Reuse consensus tensors from Phase 1
        attention_tensors = [all_consensus[i] for i in range(self.n_embeddings)]
        metadata = [
            {FM_EDGELIST.MODEL: self.attended_embeddings[key].model_name}
            for key in keys
        ]

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
                f"({len(unique_edges)} edges × {self.n_embeddings} embeddings)"
            )

        if return_original_and_reextracted:
            return all_top_edges, reextracted_union
        return reextracted_union

    def get_model_for_embedding(self, embedding_key: str) -> FoundationModel:
        if embedding_key not in self.attended_embeddings:
            raise KeyError(
                f"Embedding '{embedding_key}' not found. "
                f"Available keys: {self.embedding_keys}"
            )
        return self.attended_embeddings[embedding_key].foundation_model

    def get_specific_attentions(
        self,
        edge_list: pd.DataFrame,
        target_ids: Optional[List[str]] = None,
        apply_softmax: bool = False,
        compute_ranks: bool = False,
        by_absolute_value: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Extract specific attention values across all embeddings and layers.

        Parameters
        ----------
        edge_list : pd.DataFrame
            DataFrame with 'from_gene' and 'to_gene' columns.
        target_ids : List[str], optional
            Subset of gene IDs. If None, uses all common genes.
        apply_softmax : bool, optional
            If True, use softmax attention (default: False).
        compute_ranks : bool, optional
            If True, add ranks to output (default: False).
        by_absolute_value : bool, optional
            If True, rank by absolute value (default: True).
        verbose : bool, optional
            Print progress (default: False).

        Returns
        -------
        pd.DataFrame
            Columns: from_gene, to_gene, model, layer, attention,
            and optionally attention_rank.
        """
        results = []

        for key, ae in self.attended_embeddings.items():
            if verbose:
                logger.info(f"Extracting attentions from {key}...")

            model_attentions = ae.get_specific_attentions(
                edge_list=edge_list,
                target_ids=target_ids,
                apply_softmax=apply_softmax,
                compute_ranks=compute_ranks,
                by_absolute_value=by_absolute_value,
                verbose=False,
            )

            model_attentions[FM_EDGELIST.MODEL] = ae.model_name

            results.append(model_attentions)

        all_attentions = pd.concat(results, ignore_index=True)

        if verbose:
            n_edges = len(
                edge_list[
                    [FM_EDGELIST.FROM_GENE, FM_EDGELIST.TO_GENE]
                ].drop_duplicates()
            )
            logger.info(
                f"Extracted {len(all_attentions)} total attention values "
                f"({n_edges} edges × {self.n_embeddings} embeddings)"
            )

        return all_attentions

    def get_top_attentions(
        self,
        k: int = 10000,
        target_ids: Optional[List[str]] = None,
        by_absolute_value: bool = True,
        reextract_union: bool = False,
        apply_softmax: bool = False,
        compute_ranks: bool = False,
        ignore_self_attention: bool = False,
        return_original_and_reextracted: bool = False,
        verbose: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Extract top-k attention edges across all embeddings.

        For each embedding, identifies the k strongest attention relationships
        per layer. Enables cross-embedding comparison of attention patterns.

        Parameters
        ----------
        k : int, optional
            Number of top edges per layer per embedding (default: 10000).
        target_ids : List[str], optional
            Subset of gene IDs. If None, uses all common genes.
        by_absolute_value : bool, optional
            If True, rank by absolute value (default: True).
        reextract_union : bool, optional
            If True, re-extract union of top edges from all embeddings (default: False).
        apply_softmax : bool, optional
            If True, use softmax attention (default: False).
        compute_ranks : bool, optional
            If True, add ranks to output (default: False).
        ignore_self_attention : bool, optional
            If True, exclude self-attention edges (default: False).
        return_original_and_reextracted : bool, optional
            If True and reextract_union=True, return tuple (default: False).
        verbose : bool, optional
            Print progress (default: False).

        Returns
        -------
        pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
            DataFrame with columns: layer, from_idx, to_idx, from_gene,
            to_gene, attention, model, and optionally attention_rank.
        """
        top_attention_edges = []

        for key, ae in self.attended_embeddings.items():
            logger.info(f"Computing top-k attention for {key}...")

            model_top_k = ae.get_top_attentions(
                k=k,
                target_ids=target_ids,
                apply_softmax=apply_softmax,
                by_absolute_value=by_absolute_value,
                compute_ranks=compute_ranks,
                ignore_self_attention=ignore_self_attention,
                verbose=verbose,
            ).assign(**{FM_EDGELIST.MODEL: ae.model_name})

            top_attention_edges.append(model_top_k)

        all_top_edges = pd.concat(top_attention_edges, ignore_index=True)

        if reextract_union:
            logger.info("Re-extracting top edges from every embedding and layer...")

            reextracted = self.get_specific_attentions(
                all_top_edges,
                target_ids=target_ids,
                apply_softmax=apply_softmax,
                compute_ranks=compute_ranks,
                by_absolute_value=by_absolute_value,
                verbose=verbose,
            )

            if return_original_and_reextracted:
                return all_top_edges, reextracted
            return reextracted

        if return_original_and_reextracted:
            logger.warning(
                "return_original_and_reextracted=True but reextract_union=False, "
                "returning original top-k edges only"
            )
        return all_top_edges

    # --- Dunder ---

    def __getitem__(self, key: str) -> AttendedEmbeddings:
        if key not in self.attended_embeddings:
            raise KeyError(
                f"Embedding '{key}' not found. "
                f"Available keys: {self.embedding_keys}"
            )
        return self.attended_embeddings[key]

    def __len__(self) -> int:
        return self.n_embeddings

    def __repr__(self) -> str:
        return (
            f"AttendedEmbeddingsSet("
            f"n_embeddings={self.n_embeddings}, "
            f"n_common_genes={self.n_common_genes}, "
            f"models={self.model_names}, "
            f"keys={self.embedding_keys}"
            f")"
        )

    def get_within_embeddings_layer_comparisons(
        self,
        top_k: int,
        ignore_self_attention: bool = True,
        by_absolute_value: bool = False,
        verbose: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get within model layer x layer comparisons.

        For each of the embeddings in the set calculate cross-layer attention consistency.

        Parameters
        ----------
        top_k : int
            The number of attention pairs to summarize for "top" summaries.
        ignore_self_attention : bool
            Should self-attention (i.e., gene A - gene A) be ignored when summarizing top attention pairs. Default is True.
        by_absolute_value : bool
            Should the absolute value of the attention be used when summarizing top attention pairs. If False (default) then top-attention is selected from the most positive attention values. Default is False.
        verbose : bool
            Should verbose output be printed. Default is False.

        Returns
        -------
        tuple
            A tuple of two dictionaries. The first contains the model x layer correlations and the second contains the model x layer rank agreement.

        """

        model_layer_correlations = {}
        model_layer_rank_agreement = {}

        for an_attended_embeddings in self.attended_embeddings.values():

            model_name = an_attended_embeddings.model_name
            logger.info(
                f"Summarizing cross-layer attention consistency for {model_name}..."
            )

            (
                model_layer_correlations[model_name],
                model_layer_rank_agreement[model_name],
            ) = an_attended_embeddings.compare_layer_attention_consistency(
                top_k=top_k,
                ignore_self_attention=ignore_self_attention,
                by_absolute_value=by_absolute_value,
                verbose=verbose,
            )

        return model_layer_correlations, model_layer_rank_agreement


# Public functions


def aggregate_embedding_comparisons_over_categories(
    embedding_comparisons: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Combine a dictionary of foundation model comparison dicts into a single dict of summaries.

    For matrix-based summaries, take the median across categories.
    For DataFrame-based summaries, concatenate the DataFrames across categories and add the category as a column.
    Each comparison field is either present (dict/DataFrame) or None for a category; only whole-field None is checked.
    If all categories have None for a field, that field is omitted.

    Parameters
    ----------
    embedding_comparisons : dict
        A dictionary of embedding comparison dicts (e.g. from AttendedEmbeddingsSet.compare() per category).

    Returns
    -------
    dict
        A dictionary of aggregated embedding comparisons (no SETTINGS roll-up; callers have per-category settings).
    """
    comparisons = dict()
    categories = list(embedding_comparisons.keys())
    if not categories:
        return comparisons

    def _non_none(key: str):
        return [
            embedding_comparisons[cat][key]
            for cat in categories
            if embedding_comparisons[cat].get(key) is not None
        ]

    def _concat_categories(key: str):
        parts = [
            embedding_comparisons[cat][key].assign(category=cat)
            for cat in categories
            if embedding_comparisons[cat].get(key) is not None
        ]
        if not parts:
            return None
        return pd.concat(parts, ignore_index=True)

    # gene_embedding_correlations: median over categories (structure from first non-None)
    vals = _non_none(COMPARE_EMBEDDINGS_COMPARISONS.GENE_EMBEDDING_CORRELATIONS)
    if vals:
        comparisons[COMPARE_EMBEDDINGS_COMPARISONS.GENE_EMBEDDING_CORRELATIONS] = {
            key: np.median([cat[key] for cat in vals]) for key in vals[0]
        }
    else:
        logger.debug(
            "Omitting %s: all categories had None",
            COMPARE_EMBEDDINGS_COMPARISONS.GENE_EMBEDDING_CORRELATIONS,
        )

    # model_layer_correlations: median over categories per model (structure from first non-None)
    vals = _non_none(COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_CORRELATIONS)
    if vals:
        comparisons[COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_CORRELATIONS] = {
            model: np.median([v[model] for v in vals], axis=0) for model in vals[0]
        }
    else:
        logger.debug(
            "Omitting %s: all categories had None",
            COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_CORRELATIONS,
        )

    # model_layer_rank_agreement: concat DataFrames per model (structure from first non-None)
    vals = _non_none(COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_RANK_AGREEMENT)
    if vals:
        key_mlra = COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_RANK_AGREEMENT
        comparisons[key_mlra] = {
            model: pd.concat(
                [
                    embedding_comparisons[cat][key_mlra][model].assign(category=cat)
                    for cat in categories
                    if embedding_comparisons[cat].get(key_mlra) is not None
                ],
                ignore_index=True,
            )
            for model in vals[0]
        }
    else:
        logger.debug(
            "Omitting %s: all categories had None",
            COMPARE_EMBEDDINGS_COMPARISONS.MODEL_LAYER_RANK_AGREEMENT,
        )

    # DataFrame fields
    for key in (
        COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_TOP_ATTENTIONS,
        COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_X_LAYER_RANK_AGREEMENT,
        COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS,
        COMPARE_EMBEDDINGS_COMPARISONS.CROSS_MODEL_CONSENSUS_TOP_ATTENTIONS_RANK_AGREEMENT,
    ):
        result = _concat_categories(key)
        if result is not None:
            comparisons[key] = result
        else:
            logger.debug("Omitting %s: all categories had None", key)

    return comparisons


def validate_embedding_comparisons_settings(
    embedding_comparisons: Dict[str, Any],
    top_k: int,
    consensus_method: str,
    by_absolute_value: bool,
    ignore_self_attention: bool,
    embeddings_keys: Optional[List[str]] = None,
) -> None:
    """
    Validate embedding comparisons to ensure that their recorded settings agree with the provided settings.

    Parameters
    ----------
    embedding_comparisons : Dict[str, Any]
        The comparisons to validate. Created by AttendedEmbeddingsSet.compare().
    top_k : int
        The number of top-k attention pairs to summarize.
    consensus_method : str
        The consensus method to use. Valid options are: {VALID_FM_LAYER_CONSENSUS_METHODS}.
    by_absolute_value : bool
        Whether to use the absolute value of the attention.
    ignore_self_attention : bool
        Whether to ignore self-attention.
    embeddings_keys : Optional[List[str]]
        The embeddings keys to validate. If None, all embeddings keys will be validated.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the comparisons do not match the provided settings.
    """

    # check compatibility between the loaded results and the current settings
    if (
        top_k
        != embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][
            COMPARE_EMBEDDINGS_SETTINGS.TOP_K
        ]
    ):
        logger.warning(
            f"TOP_K mismatch: {top_k} != {embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][COMPARE_EMBEDDINGS_SETTINGS.TOP_K]}"
        )
    if (
        consensus_method
        != embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][
            COMPARE_EMBEDDINGS_SETTINGS.CONSENSUS_METHOD
        ]
    ):
        logger.warning(
            f"CONSENSUS_METHOD mismatch: {consensus_method} != {embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][COMPARE_EMBEDDINGS_SETTINGS.CONSENSUS_METHOD]}"
        )
    if (
        by_absolute_value
        != embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][
            COMPARE_EMBEDDINGS_SETTINGS.BY_ABSOLUTE_VALUE
        ]
    ):
        logger.warning(
            f"BY_ABSOLUTE_VALUE mismatch: {by_absolute_value} != {embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][COMPARE_EMBEDDINGS_SETTINGS.BY_ABSOLUTE_VALUE]}"
        )
    if (
        ignore_self_attention
        != embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][
            COMPARE_EMBEDDINGS_SETTINGS.IGNORE_SELF_ATTENTION
        ]
    ):
        logger.warning(
            f"IGNORE_SELF_ATTENTION mismatch: {ignore_self_attention} != {embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][COMPARE_EMBEDDINGS_SETTINGS.IGNORE_SELF_ATTENTION]}"
        )
    if embeddings_keys is not None:
        if set(embeddings_keys) != set(
            embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][
                COMPARE_EMBEDDINGS_SETTINGS.EMBEDDING_KEYS
            ]
        ):
            extra_models = set(
                embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][
                    COMPARE_EMBEDDINGS_SETTINGS.EMBEDDING_KEYS
                ]
            ) - set(embeddings_keys)
            missing_models = set(embeddings_keys) - set(
                embedding_comparisons[COMPARE_EMBEDDINGS_COMPARISONS.SETTINGS][
                    COMPARE_EMBEDDINGS_SETTINGS.EMBEDDING_KEYS
                ]
            )
            logger.warning(f"Extra models: {extra_models}")
            logger.warning(f"Missing models: {missing_models}")


# Private utility functions


def _align_gene_embeddings(
    embeddings: List[GeneEmbeddings],
    align_on: str = ONTOLOGIES.ENSEMBL_GENE,
    verbose: bool = True,
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
    verbose : bool, optional
        Extra reporting (default: True)

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

    if verbose:
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

        aligned_emb = GeneEmbeddings(
            embedding=emb.embedding[reorder_indices],
            ordered_gene_ids=common_ids_ordered,
            gene_annotations=emb.gene_annotations.iloc[reorder_indices].reset_index(
                drop=True
            ),
            model_name=emb.model_name,
            model_variant=emb.model_variant,
            layer_idx=emb.layer_idx,
            dataset_name=emb.dataset_name,
            dataset_uri=emb.dataset_uri,
            category=emb.category,
        )
        aligned.append(aligned_emb)

    return aligned


def _build_embedding_metadata(
    embeddings: Dict[str, GeneEmbeddings],
) -> pd.DataFrame:
    """Build a metadata DataFrame from a dict of GeneEmbeddings.

    Parameters
    ----------
    embeddings : Dict[str, GeneEmbeddings]
        Mapping from source_label to GeneEmbeddings.

    Returns
    -------
    pd.DataFrame
        One row per embedding with columns: source_label, model_name,
        model_variant, model_label, dataset_name, category.
        model_label combines model_name and model_variant with "_"
        (e.g., "scGPT", "AIDOCell_aido_cell_100m"). None variant is omitted.
    """
    rows = []
    for source_label, emb in embeddings.items():
        model_label = _get_model_label(emb.model_name, emb.model_variant)

        rows.append(
            {
                EMBEDDING_METADATA_FIELDS.SOURCE_LABEL: source_label,
                EMBEDDING_METADATA_FIELDS.MODEL_NAME: emb.model_name,
                EMBEDDING_METADATA_FIELDS.MODEL_VARIANT: emb.model_variant,
                EMBEDDING_METADATA_FIELDS.MODEL_LABEL: model_label,
                EMBEDDING_METADATA_FIELDS.LAYER_IDX: emb.layer_idx,
                EMBEDDING_METADATA_FIELDS.DATASET_NAME: emb.dataset_name,
                EMBEDDING_METADATA_FIELDS.CATEGORY: emb.category,
            }
        )
    return pd.DataFrame(rows)


def _calculate_embedding_correlations(
    aligned_embeddings: Dict[str, np.ndarray],
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compare embeddings by calculating gene-gene distances and then Spearman correlations of distances between all pairs of models

    Parameters
    ----------
    aligned_embeddings : Dict[str, np.ndarray]
        Dictionary mapping model names to aligned embedding arrays.
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
    n_genes = next(iter(aligned_embeddings.values())).shape[0]
    mask = np.triu_indices(n_genes, k=1)  # k=1 excludes diagonal

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


def _compute_scoped_keys(
    embedding_metadata: pd.DataFrame,
) -> Tuple[Dict[str, str], str]:
    """Compute minimal scoped keys and a constant label from embedding metadata.

    For each scoping field (model_label, dataset_name, category, layer_idx):
    - If the field is constant and non-None across all embeddings, it goes into
      the constant_label and is excluded from keys.
    - If the field varies (or is a mix of None and values), it stays in the keys.

    None values are excluded from both keys and labels.

    Parameters
    ----------
    embedding_metadata : pd.DataFrame
        Output of _build_embedding_metadata. Must contain columns:
        source_label, model_label, dataset_name, category, layer_idx.

    Returns
    -------
    source_to_scoped : Dict[str, str]
        Mapping from source_label to scoped key.
    constant_label : str
        "/"-joined label of constant non-None fields
        (e.g., "scGPT / efthymiou2025"). Empty string if nothing is constant.

    Raises
    ------
    ValueError
        If scoped keys are not unique.

    Examples
    --------
    >>> # Same model + dataset, different categories
    >>> # constant_label = "scGPT / efthymiou2025"
    >>> # keys: "adipocyte (0)", "T_cell"

    >>> # Different models, no dataset/category
    >>> # constant_label = ""
    >>> # keys: "scGPT", "scPRINT", "AIDOCell_aido_cell_100m"

    >>> # Same model + dataset + category, multiple layers
    >>> # constant_label = "scGPT / efthymiou2025 / adipocyte (0)"
    >>> # keys: "layer_0", "layer_5", "layer_11"
    """
    constant_parts = []
    varying_fields = []

    for field in SCOPING_FIELDS:
        non_null_values = embedding_metadata[field].dropna().unique()

        if len(non_null_values) == 0:
            # All None — skip
            continue
        elif len(non_null_values) == 1 and embedding_metadata[field].notna().all():
            # Constant non-None across all embeddings
            constant_parts.append(_format_scoping_value(field, non_null_values[0]))
        else:
            # Varies across embeddings
            varying_fields.append(field)

    # Build scoped keys from varying fields
    source_to_scoped = {}
    for _, row in embedding_metadata.iterrows():
        key_parts = []
        for field in varying_fields:
            val = row[field]
            if pd.notna(val) and val is not None:
                key_parts.append(_format_scoping_value(field, val))

        # Fallback to source_label if no varying fields produce a key
        # (e.g., single embedding where everything is constant)
        scoped_key = (
            "/".join(key_parts)
            if key_parts
            else row[EMBEDDING_METADATA_FIELDS.SOURCE_LABEL]
        )
        source_to_scoped[row[EMBEDDING_METADATA_FIELDS.SOURCE_LABEL]] = scoped_key

    # Validate uniqueness
    scoped_values = list(source_to_scoped.values())
    if len(scoped_values) != len(set(scoped_values)):
        counts = Counter(scoped_values)
        duplicates = {k: v for k, v in counts.items() if v > 1}
        raise ValueError(
            f"Scoped keys are not unique: {duplicates}. "
            f"This indicates a bug in the scoping logic or duplicate embeddings."
        )

    constant_label = " / ".join(constant_parts)

    return source_to_scoped, constant_label


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


def _format_scoping_value(field: str, val: Any) -> str:
    """Format a scoping field's value for use in scoped keys or constant labels.

    Most fields stringify directly. layer_idx gets a ``"layer_"`` prefix for
    readability: a key like ``"adipocyte (0)/layer_5"`` is more self-documenting
    than ``"adipocyte (0)/5"``.

    Parameters
    ----------
    field : str
        Field name from SCOPING_FIELDS.
    val : Any
        Non-None field value.

    Returns
    -------
    str
        Formatted string.
    """
    if field == EMBEDDING_METADATA_FIELDS.LAYER_IDX:
        return f"layer_{int(val)}"
    return str(val)


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


def _get_disk_name(
    model_name: str,
    model_variant: Optional[str] = None,
) -> str:
    """Get a version of the model label which can be used for a filename."""
    if model_variant is None:
        return model_name
    return f"{model_name}_{model_variant}"


def _get_model_label(
    model_name: str,
    model_variant: Optional[str] = None,
) -> str:
    """Get a human-readable label for a foundation model.

    Looks up (model_name, model_variant) in a curated table of display names.
    Falls back to an auto-generated label if no curated name exists.

    Auto-generated format:
    - "model_name (model_variant)" if model_variant is not None
    - "model_name" if model_variant is None

    Parameters
    ----------
    model_name : str
        Name of the foundation model (e.g., "scGPT", "AIDOCell").
    model_variant : str, optional
        Variant of the model (e.g., "aido_cell_100m", "small").

    Returns
    -------
    str
        Human-readable model label.

    Examples
    --------
    >>> get_model_label("AIDOCell", "aido_cell_100m")
    'AIDO.Cell (100M)'
    >>> get_model_label("scGPT")
    'scGPT'
    >>> get_model_label("scPRINT", "small")
    'scPRINT (small)'
    >>> get_model_label("NewModel", "v2")
    'NewModel (v2)'
    >>> get_model_label("NewModel")
    'NewModel'
    """
    key = (model_name, model_variant)
    if key in MODEL_NICE_NAMES:
        return MODEL_NICE_NAMES[key]

    # Auto-generate
    if model_variant is not None:
        return f"{model_name} ({model_variant})"
    return model_name


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
