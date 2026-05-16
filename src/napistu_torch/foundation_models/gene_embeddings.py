"""
Gene embeddings: aligned matrices with metadata and dataset containers.
"""

from __future__ import annotations

import logging
from collections import Counter
from functools import cached_property
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from napistu.ontologies.constants import ONTOLOGIES
from pydantic import BaseModel, Field, field_validator, model_validator

from napistu_torch.foundation_models.constants import (
    EMBEDDING_METADATA_FIELDS,
    GROUP_SCOPING_FIELDS,
    MODEL_NICE_NAMES,
    SCOPING_FIELDS,
)
from napistu_torch.utils.constants import CORRELATION_METHODS
from napistu_torch.utils.tensor_utils import (
    compute_correlation,
    compute_cosine_distances_torch,
)
from napistu_torch.utils.torch_utils import ensure_device, memory_manager

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
        from napistu_torch.foundation_models.foundation_models import GeneAnnotations

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

    Private Methods (expression validation)
    ---------------------------------------
    _log_expression_validation_detail(...)
        Log detailed validation information.
    _validate_expression_embeddings(...)
        Bundle used by ``FoundationModel.validate_dataset_gene_embeddings``; calls layered helpers below.
    _validate_expression_layers(...)
        ``layer_idx`` coverage and ``source_label`` checks.
    _validate_expression_spot_means(...)
        Optional informational comparison of mean activation between two layers.
    _validate_expression_variance(...)
        Fail embeddings whose pooled standard deviation is at or below ``std_tol``.

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
        source_to_scoped, constant_label = _compute_scoped_keys_for_fields(
            embedding_metadata, SCOPING_FIELDS
        )

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

    def _log_expression_validation_detail(
        self,
        *,
        dataset_name: str,
        distinct_sorted: Optional[Tuple[int, ...]],
    ) -> None:
        logger.info("Dataset: %s", dataset_name)
        logger.info("  n_embeddings: %s", self.n_embeddings)
        logger.info("  n_common_genes: %s", self.n_common_genes)
        logger.info("--- Per-embedding check ---")
        for key, ge in self.items():
            logger.info(
                "  key=%r layer_idx=%s source_label=%r embedding.shape=%s",
                key,
                ge.layer_idx,
                ge.source_label,
                getattr(ge.embedding, "shape", None),
            )
        if distinct_sorted is not None:
            logger.info("Unique layer_idx values: %s", list(distinct_sorted))

    def _validate_expression_embeddings(
        self,
        *,
        dataset_name: str,
        expected_n_layers: int,
        std_tol: float,
        spot_check_layers: Optional[Tuple[int, int]],
        verbose: bool,
    ) -> Dict[str, Any]:
        """Run layer, variance, and spot checks (used by ``FoundationModel.validate_dataset_gene_embeddings``)."""
        layer_errors, distinct_sorted = self._validate_expression_layers(
            dataset_name=dataset_name,
            expected_n_layers=expected_n_layers,
            verbose=verbose,
        )
        errors = list(layer_errors)
        errors.extend(self._validate_expression_variance(std_tol=std_tol))

        if verbose:
            self._log_expression_validation_detail(
                dataset_name=dataset_name,
                distinct_sorted=distinct_sorted,
            )

        spot_note = self._validate_expression_spot_means(
            expected_n_layers=expected_n_layers,
            spot_check_layers=spot_check_layers,
            errors=errors,
            verbose=verbose,
        )

        if verbose and len(errors) == 0:
            logger.info("✓ Validation passed for dataset %r", dataset_name)

        return {
            "dataset_name": dataset_name,
            "ok": len(errors) == 0,
            "errors": errors,
            "n_embeddings": self.n_embeddings,
            "n_common_genes": self.n_common_genes,
            "distinct_layer_indices": distinct_sorted,
            "spot_check_note": spot_note,
        }

    def _validate_expression_layers(
        self,
        *,
        dataset_name: str,
        expected_n_layers: int,
        verbose: bool,
    ) -> Tuple[List[str], Optional[Tuple[int, ...]]]:
        """Check ``layer_idx`` coverage and ``source_label`` layer segments."""
        errors: List[str] = []
        distinct_sorted: Optional[Tuple[int, ...]] = None
        layer_vals = [ge.layer_idx for ge in self.values()]
        if any(li is None for li in layer_vals) and any(
            li is not None for li in layer_vals
        ):
            errors.append(
                "Mixed layer_idx: some embeddings set layer_idx and others leave it None."
            )
            return errors, distinct_sorted

        if all(li is None for li in layer_vals):
            errors.append(
                "No layer-wise residual stream: all embeddings have layer_idx=None. "
                "Per-layer exports must set layer_idx to an integer in "
                f"range({expected_n_layers}) for each embedding tensor."
            )
            return errors, distinct_sorted

        layers_seen = {li for li in layer_vals if li is not None}
        distinct_sorted = tuple(sorted(layers_seen))
        expected = set(range(expected_n_layers))
        if layers_seen != expected:
            missing = sorted(expected - layers_seen)
            extra = sorted(layers_seen - expected)
            msg_parts = []
            if missing:
                msg_parts.append(f"missing layers {missing}")
            if extra:
                msg_parts.append(f"unexpected layers {extra}")
            errors.append(
                f"layer_idx coverage mismatch ({'; '.join(msg_parts)}); "
                f"expected {{0..{expected_n_layers - 1}}}"
            )

        for key, ge in self.items():
            if ge.layer_idx is None:
                errors.append(
                    f"embedding key={key!r}: layer_idx is None inside layered set"
                )
            elif f"layer_{ge.layer_idx}" not in ge.source_label:
                errors.append(
                    f"key={key!r}: source_label {ge.source_label!r} does not contain "
                    f"'layer_{ge.layer_idx}'"
                )
        return errors, distinct_sorted

    def _validate_expression_spot_means(
        self,
        *,
        expected_n_layers: int,
        spot_check_layers: Optional[Tuple[int, int]],
        errors: List[str],
        verbose: bool,
    ) -> Optional[str]:
        """Optional informational comparison of mean activation between two layers."""
        resolved_spot = spot_check_layers
        if resolved_spot is None and expected_n_layers > 1:
            resolved_spot = (0, expected_n_layers - 1)

        if resolved_spot is None or errors:
            return None

        li_a, li_b = resolved_spot
        try:
            key_a = next(k for k, ge in self.items() if ge.layer_idx == li_a)
            key_b = next(k for k, ge in self.items() if ge.layer_idx == li_b)
        except StopIteration:
            note = f"spot-check skipped (no embeddings found for layers {li_a} and/or {li_b})"
            if verbose:
                logger.info("%s", note)
            return note

        mean_a = float(np.asarray(self[key_a].embedding).mean())
        mean_b = float(np.asarray(self[key_b].embedding).mean())
        note = (
            f"Spot-check means — layer {li_a}: {mean_a:.4f}, "
            f"layer {li_b}: {mean_b:.4f} "
            f"(ordering depends on cell type)"
        )
        if verbose:
            logger.info("%s", note)
        return note

    def _validate_expression_variance(self, *, std_tol: float) -> List[str]:
        """Fail embeddings whose pooled standard deviation is at or below ``std_tol``."""
        errors: List[str] = []
        for key, ge in self.items():
            std = float(np.asarray(ge.embedding).std())
            if std <= std_tol:
                errors.append(
                    f"key={key!r}: embedding.std()={std} is not above std_tol ({std_tol!r})"
                )
        return errors

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

    Private Methods
    ---------------
    _validate_dataset_gene_embeddings_present(dge)
        Classmethod: if ``dge`` is ``None``, return a synthetic validation failure row.
    _validate_resolve_dataset_slice(dataset_name)
        Resolve which ``(dataset_name, GeneEmbeddingsSet)`` pairs to validate.

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

    # --- Private methods ---

    @classmethod
    def _validate_dataset_gene_embeddings_present(
        cls,
        dge: Optional["DatasetGeneEmbeddings"],
    ) -> Optional[Dict[str, Any]]:
        """If embeddings are missing, return a synthetic failure row; else ``None``."""
        if dge is None:
            return {
                "dataset_name": "__missing__",
                "ok": False,
                "errors": ["dataset_gene_embeddings is None"],
                "n_embeddings": 0,
                "n_common_genes": 0,
                "distinct_layer_indices": None,
                "spot_check_note": None,
            }
        return None

    def _validate_resolve_dataset_slice(
        self,
        dataset_name: Optional[str],
    ) -> List[Tuple[str, GeneEmbeddingsSet]]:
        """Resolve which (dataset_name, GeneEmbeddingsSet) pairs to validate."""
        if dataset_name is not None:
            return [(dataset_name, self[dataset_name])]
        return list(self.items())

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


def _build_group_scoped_keys(
    groups: Dict[Tuple[str, str], Dict[int, "GeneEmbeddings"]],
) -> Tuple[Dict[str, str], str]:
    group_representative_embeddings = {
        f"{full_name}/{category}": next(iter(layer_embeddings.values()))
        for (full_name, category), layer_embeddings in groups.items()
    }
    group_metadata = _build_embedding_metadata(group_representative_embeddings)
    return _compute_scoped_keys_for_fields(group_metadata, GROUP_SCOPING_FIELDS)


def _compute_scoped_keys_for_fields(
    embedding_metadata: pd.DataFrame,
    scoping_fields: List[str],
) -> Tuple[Dict[str, str], str]:
    """Compute minimal scoped keys using a specified set of scoping fields.

    Core implementation shared by _compute_scoped_keys (full, with layer_idx)
    and group-level scoping in AttentionPatternsInputs (without layer_idx).

    Parameters
    ----------
    embedding_metadata : pd.DataFrame
        Output of _build_embedding_metadata.
    scoping_fields : List[str]
        Ordered fields to consider, e.g. SCOPING_FIELDS or GROUP_SCOPING_FIELDS.

    Returns
    -------
    source_to_scoped : Dict[str, str]
        Mapping from source_label to scoped key.
    constant_label : str
        " / "-joined label of constant non-None fields.

    Raises
    ------
    ValueError
        If scoped keys are not unique.
    """
    constant_parts = []
    varying_fields = []

    for field in scoping_fields:
        non_null_values = embedding_metadata[field].dropna().unique()

        if len(non_null_values) == 0:
            continue
        elif len(non_null_values) == 1 and embedding_metadata[field].notna().all():
            constant_parts.append(_format_scoping_value(field, non_null_values[0]))
        else:
            varying_fields.append(field)

    source_to_scoped = {}
    for _, row in embedding_metadata.iterrows():
        key_parts = []
        for field in varying_fields:
            val = row[field]
            if pd.notna(val) and val is not None:
                key_parts.append(_format_scoping_value(field, val))

        scoped_key = (
            "/".join(key_parts)
            if key_parts
            else row[EMBEDDING_METADATA_FIELDS.SOURCE_LABEL]
        )
        source_to_scoped[row[EMBEDDING_METADATA_FIELDS.SOURCE_LABEL]] = scoped_key

    scoped_values = list(source_to_scoped.values())
    if len(scoped_values) != len(set(scoped_values)):
        counts = Counter(scoped_values)
        duplicates = {k: v for k, v in counts.items() if v > 1}
        raise ValueError(
            f"Scoped keys are not unique: {duplicates}. "
            f"This indicates a bug in the scoping logic or duplicate embeddings."
        )

    return source_to_scoped, " / ".join(constant_parts)


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
