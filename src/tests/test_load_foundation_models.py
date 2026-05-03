"""Tests for foundation model data structures and validation."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from napistu.ontologies.constants import ONTOLOGIES

from napistu_torch.load.constants import FM_DEFS
from napistu_torch.load.foundation_models import (
    AttendedEmbeddings,
    AttendedEmbeddingsSet,
    AttentionLayer,
    DatasetGeneEmbeddings,
    FoundationModel,
    FoundationModels,
    FoundationModelWeights,
    GeneEmbeddings,
    GeneEmbeddingsSet,
    ModelMetadata,
    _build_embedding_metadata,
    _compute_scoped_keys,
)

# ---------------------------------------------------------------------------
# Low-level factories
# ---------------------------------------------------------------------------


def make_gene_ids(
    n: int,
    prefix: str = "ENSG",
    start: int = 0,
) -> List[str]:
    """Create deterministic gene IDs.

    Parameters
    ----------
    n : int
        Number of gene IDs to create.
    prefix : str
        Prefix for each ID (default: "ENSG").
    start : int
        Starting index (default: 0).

    Returns
    -------
    List[str]
        e.g., ["ENSG00000", "ENSG00001", ...]
    """
    return [f"{prefix}{str(i).zfill(5)}" for i in range(start, start + n)]


def make_gene_annotations(
    gene_ids: List[str],
    vocab_names: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create a gene annotations DataFrame.

    Parameters
    ----------
    gene_ids : List[str]
        Ensembl gene IDs.
    vocab_names : List[str], optional
        Vocabulary names. If None, uses gene_ids.
    symbols : List[str], optional
        Gene symbols. If None, generates SYM_00000, SYM_00001, ...

    Returns
    -------
    pd.DataFrame
        With columns: vocab_name, ensembl_gene, symbol.
    """
    if vocab_names is None:
        vocab_names = list(gene_ids)
    if symbols is None:
        symbols = [f"SYM_{str(i).zfill(5)}" for i in range(len(gene_ids))]

    return pd.DataFrame(
        {
            "vocab_name": vocab_names,
            ONTOLOGIES.ENSEMBL_GENE: gene_ids,
            "symbol": symbols,
        }
    )


def make_gene_embeddings(
    n_genes: int = 10,
    embed_dim: int = 16,
    gene_ids: Optional[List[str]] = None,
    vocab_names: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    model_variant: Optional[str] = None,
    dataset_name: Optional[str] = None,
    category: Optional[str] = None,
    seed: int = 42,
) -> GeneEmbeddings:
    """Create a GeneEmbeddings with deterministic random data.

    Parameters
    ----------
    n_genes : int
        Number of genes (default: 10).
    embed_dim : int
        Embedding dimensionality (default: 16).
    gene_ids : List[str], optional
        Gene IDs. If None, generates ENSG00000, ENSG00001, ...
    vocab_names : List[str], optional
        Vocabulary names. If None, uses gene_ids.
    model_name : str, optional
        Source model name.
    model_variant : str, optional
        Source model variant.
    dataset_name : str, optional
        Source dataset name.
    category : str, optional
        Category within dataset.
    seed : int
        Random seed for reproducibility (default: 42).

    Returns
    -------
    GeneEmbeddings
    """
    rng = np.random.default_rng(seed)

    if gene_ids is None:
        gene_ids = make_gene_ids(n_genes)
    else:
        n_genes = len(gene_ids)

    annotations = make_gene_annotations(gene_ids=gene_ids, vocab_names=vocab_names)
    embedding = rng.standard_normal((n_genes, embed_dim)).astype(np.float32)

    return GeneEmbeddings(
        embedding=embedding,
        ordered_gene_ids=gene_ids,
        gene_annotations=annotations,
        model_name=model_name,
        model_variant=model_variant,
        dataset_name=dataset_name,
        category=category,
    )


def make_attention_layer(
    embed_dim: int = 16,
    layer_idx: int = 0,
    seed: int = 42,
) -> AttentionLayer:
    """Create an AttentionLayer with deterministic random weights.

    Parameters
    ----------
    embed_dim : int
        Model embedding dimension (default: 16).
    layer_idx : int
        Layer index (default: 0).
    seed : int
        Random seed (default: 42).

    Returns
    -------
    AttentionLayer
    """
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(embed_dim)

    return AttentionLayer(
        layer_idx=layer_idx,
        W_q=rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale,
        W_k=rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale,
        W_v=rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale,
        W_o=rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale,
    )


# ---------------------------------------------------------------------------
# Mid-level factories
# ---------------------------------------------------------------------------


def make_foundation_model(
    n_genes: int = 20,
    embed_dim: int = 16,
    n_layers: int = 2,
    n_heads: int = 2,
    model_name: str = "TestModel",
    model_variant: Optional[str] = None,
    gene_ids: Optional[List[str]] = None,
    vocab_names: Optional[List[str]] = None,
    extra_vocab: int = 2,
    seed: int = 42,
) -> FoundationModel:
    """Create a complete FoundationModel with deterministic data.

    Generates a model with `n_genes` real genes plus `extra_vocab` special tokens
    (e.g., <pad>, <cls>). The static gene embedding covers the full vocabulary.

    Parameters
    ----------
    n_genes : int
        Number of real genes (default: 20).
    embed_dim : int
        Embedding dimension (default: 16).
    n_layers : int
        Number of attention layers (default: 2).
    n_heads : int
        Number of attention heads (default: 2).
    model_name : str
        Model name (default: "TestModel").
    model_variant : str, optional
        Model variant.
    gene_ids : List[str], optional
        Gene IDs. If None, generates ENSG00000 through ENSG{n_genes-1}.
    vocab_names : List[str], optional
        Vocabulary names for the genes. If None, uses gene_ids.
    extra_vocab : int
        Number of extra vocabulary tokens (default: 2).
    seed : int
        Random seed (default: 42).

    Returns
    -------
    FoundationModel
    """
    rng = np.random.default_rng(seed)

    if gene_ids is None:
        gene_ids = make_gene_ids(n_genes)
    else:
        n_genes = len(gene_ids)

    if vocab_names is None:
        vocab_names = list(gene_ids)

    # Build full vocabulary: genes + special tokens
    special_tokens = [f"<special_{i}>" for i in range(extra_vocab)]
    full_vocab = vocab_names + special_tokens
    n_vocab = len(full_vocab)

    # Gene annotations cover only the real genes
    annotations = make_gene_annotations(gene_ids=gene_ids, vocab_names=vocab_names)

    # Static embedding covers full vocabulary
    full_embedding = rng.standard_normal((n_vocab, embed_dim)).astype(np.float32)

    # The GeneEmbeddings for static uses the gene portion only
    static_ge = GeneEmbeddings(
        embedding=full_embedding[:n_genes],
        ordered_gene_ids=gene_ids,
        gene_annotations=annotations,
        model_name=model_name,
        model_variant=model_variant,
    )

    # Attention layers with different seeds per layer
    attention_layers = [
        make_attention_layer(
            embed_dim=embed_dim,
            layer_idx=i,
            seed=seed + i + 1,
        )
        for i in range(n_layers)
    ]

    weights = FoundationModelWeights(
        static_gene_embeddings=static_ge,
        attention_layers=attention_layers,
    )

    metadata = ModelMetadata(
        model_name=model_name,
        model_variant=model_variant,
        n_genes=n_genes,
        n_vocab=n_vocab,
        ordered_vocabulary=full_vocab,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
    )

    return FoundationModel(
        weights=weights,
        gene_annotations=annotations,
        model_metadata=metadata,
    )


def make_foundation_model_pair(
    n_shared: int = 15,
    n_unique_a: int = 5,
    n_unique_b: int = 5,
    embed_dim: int = 16,
    n_layers_a: int = 2,
    n_layers_b: int = 3,
    n_heads: int = 2,
    model_name_a: str = "ModelA",
    model_name_b: str = "ModelB",
    use_different_vocab: bool = False,
    seed: int = 42,
) -> Tuple[FoundationModel, FoundationModel, List[str]]:
    """Create two FoundationModels with partially overlapping gene sets.

    This is the key fixture for testing cross-model alignment. The two models
    share `n_shared` genes and each has unique genes the other doesn't have.

    Parameters
    ----------
    n_shared : int
        Number of shared genes (default: 15).
    n_unique_a : int
        Genes only in model A (default: 5).
    n_unique_b : int
        Genes only in model B (default: 5).
    embed_dim : int
        Embedding dimension (default: 16).
    n_layers_a : int
        Layers for model A (default: 2).
    n_layers_b : int
        Layers for model B (default: 3).
    n_heads : int
        Attention heads (default: 2).
    model_name_a : str
        Name for model A (default: "ModelA").
    model_name_b : str
        Name for model B (default: "ModelB").
    use_different_vocab : bool
        If True, model B uses symbol-based vocab names instead of Ensembl IDs.
        This tests alignment across different native vocabularies (default: False).
    seed : int
        Random seed (default: 42).

    Returns
    -------
    model_a : FoundationModel
    model_b : FoundationModel
    shared_gene_ids : List[str]
        The Ensembl gene IDs shared by both models.
    """
    shared_ids = make_gene_ids(n_shared, prefix="ENSG", start=0)
    unique_a_ids = make_gene_ids(n_unique_a, prefix="ENSG", start=n_shared)
    unique_b_ids = make_gene_ids(n_unique_b, prefix="ENSG", start=n_shared + n_unique_a)

    gene_ids_a = unique_a_ids + shared_ids  # unique first, shared after
    gene_ids_b = shared_ids + unique_b_ids  # shared first, unique after

    vocab_names_b = None
    if use_different_vocab:
        # Model B uses symbols as vocab names
        symbols_b = [f"SYM_{str(i).zfill(5)}" for i in range(len(gene_ids_b))]
        vocab_names_b = symbols_b

    model_a = make_foundation_model(
        gene_ids=gene_ids_a,
        embed_dim=embed_dim,
        n_layers=n_layers_a,
        n_heads=n_heads,
        model_name=model_name_a,
        seed=seed,
    )

    model_b = make_foundation_model(
        gene_ids=gene_ids_b,
        vocab_names=vocab_names_b,
        embed_dim=embed_dim,
        n_layers=n_layers_b,
        n_heads=n_heads,
        model_name=model_name_b,
        seed=seed + 1000,
    )

    return model_a, model_b, shared_ids


# ---------------------------------------------------------------------------
# High-level factories
# ---------------------------------------------------------------------------


def make_foundation_models(
    n_models: int = 3,
    n_shared: int = 15,
    n_unique_per_model: int = 5,
    embed_dim: int = 16,
    n_layers: int = 2,
    n_heads: int = 2,
    seed: int = 42,
) -> Tuple[FoundationModels, List[str]]:
    """Create a FoundationModels container with multiple models.

    All models share `n_shared` genes and each has `n_unique_per_model`
    unique genes.

    Parameters
    ----------
    n_models : int
        Number of models (default: 3, minimum 2).
    n_shared : int
        Genes shared across all models (default: 15).
    n_unique_per_model : int
        Unique genes per model (default: 5).
    embed_dim : int
        Embedding dimension (default: 16).
    n_layers : int
        Layers per model (default: 2).
    n_heads : int
        Heads per model (default: 2).
    seed : int
        Random seed (default: 42).

    Returns
    -------
    foundation_models : FoundationModels
    shared_gene_ids : List[str]
        Ensembl gene IDs shared by all models.
    """
    shared_ids = make_gene_ids(n_shared, prefix="ENSG", start=0)

    models = []
    for i in range(n_models):
        unique_start = n_shared + i * n_unique_per_model
        unique_ids = make_gene_ids(
            n_unique_per_model, prefix="ENSG", start=unique_start
        )
        # Interleave unique and shared to test ordering robustness
        gene_ids = unique_ids + shared_ids

        model = make_foundation_model(
            gene_ids=gene_ids,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            model_name=f"TestModel{i}",
            seed=seed + i * 1000,
        )
        models.append(model)

    return FoundationModels(models=models), shared_ids


def make_attended_embeddings(
    n_genes: int = 10,
    embed_dim: int = 16,
    n_layers: int = 2,
    n_heads: int = 2,
    model_name: str = "TestModel",
    seed: int = 42,
) -> AttendedEmbeddings:
    """Create a single AttendedEmbeddings for unit testing.

    Parameters
    ----------
    n_genes : int
        Number of genes (default: 10).
    embed_dim : int
        Embedding dimension (default: 16).
    n_layers : int
        Number of layers (default: 2).
    n_heads : int
        Number of heads (default: 2).
    model_name : str
        Model name (default: "TestModel").
    seed : int
        Random seed (default: 42).

    Returns
    -------
    AttendedEmbeddings
    """
    model = make_foundation_model(
        n_genes=n_genes,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        model_name=model_name,
        seed=seed,
    )

    return AttendedEmbeddings(
        gene_embeddings=model.weights.static_gene_embeddings,
        foundation_model=model,
    )


def make_attended_embeddings_set(
    n_models: int = 3,
    n_shared: int = 15,
    n_unique_per_model: int = 5,
    embed_dim: int = 16,
    n_layers: int = 2,
    n_heads: int = 2,
    seed: int = 42,
) -> Tuple[AttendedEmbeddingsSet, List[str]]:
    """Create an AttendedEmbeddingsSet for cross-model testing.

    Parameters
    ----------
    n_models : int
        Number of models (default: 3, minimum 2).
    n_shared : int
        Shared genes (default: 15).
    n_unique_per_model : int
        Unique genes per model (default: 5).
    embed_dim : int
        Embedding dimension (default: 16).
    n_layers : int
        Layers per model (default: 2).
    n_heads : int
        Heads per model (default: 2).
    seed : int
        Random seed (default: 42).

    Returns
    -------
    attended_set : AttendedEmbeddingsSet
    shared_gene_ids : List[str]
        Gene IDs common to all models.
    """
    fm, shared_ids = make_foundation_models(
        n_models=n_models,
        n_shared=n_shared,
        n_unique_per_model=n_unique_per_model,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        seed=seed,
    )

    attended_set = AttendedEmbeddingsSet.from_static(
        fm, align_on=ONTOLOGIES.ENSEMBL_GENE
    )

    return attended_set, shared_ids


# utility functions

# Ensembl IDs shared by scoping tests (same genes; source_labels differ by
# model/dataset/category/layer).
SCOPING_TEST_GENES = [
    "ENSG00000000001",
    "ENSG00000000002",
    "ENSG00000000003",
]


def _make_gene_annotations(genes):
    """DataFrame valid for GeneAnnotations (vocab_name + ensembl_gene)."""
    return pd.DataFrame(
        {
            FM_DEFS.VOCAB_NAME: list(genes),
            ONTOLOGIES.ENSEMBL_GENE: list(genes),
        }
    )


def _make_gene_emb(
    genes,
    embed_dim=4,
    model_name=None,
    dataset_name=None,
    category=None,
    layer_idx=None,
):
    """Minimal GeneEmbeddings for tests."""
    n = len(genes)
    return GeneEmbeddings(
        embedding=np.arange(n * embed_dim, dtype=np.float64).reshape(n, embed_dim),
        ordered_gene_ids=list(genes),
        gene_annotations=_make_gene_annotations(genes),
        model_name=model_name,
        layer_idx=layer_idx,
        dataset_name=dataset_name,
        category=category,
    )


def _scope(embeddings):
    data = {emb.source_label: emb for emb in embeddings}
    metadata = _build_embedding_metadata(data)
    return _compute_scoped_keys(metadata)


# tests


def test_gene_embeddings_field_validation():
    """Test GeneEmbeddings embedding field validation: shape and type."""
    genes = ["g1", "g2", "g3"]
    ann = _make_gene_annotations(genes)

    # Valid 2D numpy array
    valid_embeddings = np.random.randn(3, 32)
    ge = GeneEmbeddings(
        embedding=valid_embeddings,
        ordered_gene_ids=genes,
        gene_annotations=ann,
    )
    assert ge.embedding.shape == (3, 32)
    assert ge.n_genes == 3
    assert ge.embed_dim == 32

    # Invalid: 3D array
    with pytest.raises(ValueError) as exc_info:
        GeneEmbeddings(
            embedding=np.random.randn(2, 10, 32),
            ordered_gene_ids=genes[:2],
            gene_annotations=_make_gene_annotations(genes[:2]),
        )
    assert "2-dimensional" in str(exc_info.value)
    assert "got shape (2, 10, 32)" in str(exc_info.value)

    # Invalid: 1D array
    with pytest.raises(ValueError) as exc_info:
        GeneEmbeddings(
            embedding=np.random.randn(32),
            ordered_gene_ids=genes,
            gene_annotations=ann,
        )
    assert "2-dimensional" in str(exc_info.value)


def test_gene_embeddings_set_validation():
    """Test GeneEmbeddingsSet: from_gene_embeddings, unique source_label, empty list."""
    genes = ["g1", "g2", "g3"]
    ge1 = _make_gene_emb(genes, embed_dim=8, model_name="ModelA", category="type1")
    ge2 = _make_gene_emb(genes, embed_dim=8, model_name="ModelA", category="type2")

    # Single embedding: wraps directly
    ge_set = GeneEmbeddingsSet.from_gene_embeddings(
        [ge1], align_on=ONTOLOGIES.ENSEMBL_GENE
    )
    assert ge_set.n_embeddings == 1
    assert ge_set.n_common_genes == 3
    assert ge_set.get("ModelA/type1") is ge1

    # Multiple embeddings with unique source_label: aligns to common genes
    ge_set2 = GeneEmbeddingsSet.from_gene_embeddings([ge1, ge2])
    assert ge_set2.n_embeddings == 2
    assert ge_set2.n_common_genes == 3
    assert set(ge_set2.keys()) == {"type1", "type2"}

    # Invalid: duplicate source_label
    ge_dup = _make_gene_emb(genes, embed_dim=8, model_name="ModelA", category="type1")
    with pytest.raises(ValueError) as exc_info:
        GeneEmbeddingsSet.from_gene_embeddings([ge1, ge_dup])
    assert "Duplicate source_label" in str(exc_info.value)

    # Invalid: empty list
    with pytest.raises(ValueError) as exc_info:
        GeneEmbeddingsSet.from_gene_embeddings([])
    assert "requires at least 1 embedding" in str(exc_info.value)


def test_dataset_gene_embeddings():
    """Test DatasetGeneEmbeddings: get, keys, values, items, dict init."""
    genes = ["g1", "g2", "g3"]
    ge1a = _make_gene_emb(genes, embed_dim=32, model_name="scGPT", category="type1")
    ge1b = _make_gene_emb(genes, embed_dim=32, model_name="scGPT", category="type2")
    set1 = GeneEmbeddingsSet.from_gene_embeddings([ge1a, ge1b])

    ge2a = _make_gene_emb(genes, embed_dim=32, model_name="scGPT", category="a")
    ge2b = _make_gene_emb(genes, embed_dim=32, model_name="scGPT", category="b")
    ge2c = _make_gene_emb(genes, embed_dim=32, model_name="scGPT", category="c")
    set2 = GeneEmbeddingsSet.from_gene_embeddings([ge2a, ge2b, ge2c])

    # Init from dict
    container = DatasetGeneEmbeddings({"efthymiou": set1, "tabula_sapiens": set2})
    assert container.get("efthymiou") is set1
    assert container["efthymiou"] is set1
    assert container.get("tabula_sapiens") is set2
    assert "efthymiou" in container
    assert "missing" not in container
    assert set(container.keys()) == {"efthymiou", "tabula_sapiens"}
    assert list(container.values()) == [set1, set2]
    assert len(list(container.items())) == 2
    assert "DatasetGeneEmbeddings" in repr(container)

    # KeyError when not found
    with pytest.raises(KeyError) as exc_info:
        container.get("nonexistent")
    assert "nonexistent" in str(exc_info.value)
    assert "Available datasets" in str(exc_info.value)

    # Invalid: value must be GeneEmbeddingsSet
    with pytest.raises(ValueError) as exc_info:
        DatasetGeneEmbeddings(
            {"bad": ge1a}
        )  # ge1a is GeneEmbeddings, not GeneEmbeddingsSet
    assert "must be a GeneEmbeddingsSet" in str(exc_info.value)

    # Invalid: empty dict
    with pytest.raises(ValueError) as exc_info:
        DatasetGeneEmbeddings({})
    assert "requires at least one dataset" in str(exc_info.value)


# --- GeneEmbeddings ---


def test_gene_embeddings_construction():
    """GeneEmbeddings construction: valid args and properties."""
    genes = ["g1", "g2", "g3"]
    ge = _make_gene_emb(genes, embed_dim=8, model_name="scGPT")
    assert ge.n_genes == 3
    assert ge.embed_dim == 8
    assert ge.ordered_gene_ids == genes
    assert ge.gene_ids_set == frozenset(genes)
    assert ge.source_label == "scGPT"
    assert ge.embedding.shape == (3, 8)

    ge = _make_gene_emb(genes, embed_dim=8, model_name="scGPT", layer_idx=0)
    assert ge.layer_idx == 0
    assert ge.source_label == "scGPT/layer_0"
    assert ge.embedding.shape == (3, 8)
    assert ge.gene_ids_set == frozenset(genes)
    assert ge.gene_annotations.shape == (3, 2)
    assert set(ge.gene_annotations.columns) == {
        FM_DEFS.VOCAB_NAME,
        ONTOLOGIES.ENSEMBL_GENE,
    }


def test_gene_embeddings_construction_validation():
    """GeneEmbeddings rejects bad embedding shape and row/gene count mismatch."""
    genes = ["g1", "g2"]
    ann = _make_gene_annotations(genes)
    with pytest.raises(ValueError) as e:
        GeneEmbeddings(
            embedding=np.random.randn(2, 3, 4),
            ordered_gene_ids=genes,
            gene_annotations=ann,
        )
    assert "2-dimensional" in str(e.value)
    with pytest.raises(ValueError) as e:
        GeneEmbeddings(
            embedding=np.random.randn(2, 4),
            ordered_gene_ids=genes,
            gene_annotations=_make_gene_annotations(["g1"]),
        )
    assert "gene_annotations has 1 rows" in str(e.value)
    with pytest.raises(ValueError) as e:
        GeneEmbeddings(
            embedding=np.random.randn(2, 4),
            ordered_gene_ids=["g1", "g1"],
            gene_annotations=ann,
        )
    assert "unique" in str(e.value).lower()

    # GeneAnnotations validation: missing required column
    with pytest.raises(ValueError) as e:
        GeneEmbeddings(
            embedding=np.random.randn(2, 4),
            ordered_gene_ids=genes,
            gene_annotations=pd.DataFrame({ONTOLOGIES.ENSEMBL_GENE: genes}),
        )
    assert "missing required column" in str(e.value).lower()


def test_gene_embeddings_align_to_full_match_preserves_order():
    """align_to with full target list reorders to target order."""
    ge = _make_gene_emb(["a", "b", "c"], embed_dim=2)
    aligned = ge.align_to(["c", "a", "b"])
    assert aligned.ordered_gene_ids == ["c", "a", "b"]
    assert aligned.n_genes == 3
    # Row 0 was c (index 2), row 1 was a (0), row 2 was b (1)
    np.testing.assert_array_equal(aligned.embedding[0], ge.embedding[2])
    np.testing.assert_array_equal(aligned.embedding[1], ge.embedding[0])
    np.testing.assert_array_equal(aligned.embedding[2], ge.embedding[1])


def test_gene_embeddings_align_to_subset_and_reorder():
    """align_to keeps only target genes present in self, in target order."""
    ge = _make_gene_emb(["a", "b", "c"], embed_dim=2)
    aligned = ge.align_to(["c", "x", "a"])  # x not in ge
    assert aligned.ordered_gene_ids == ["c", "a"]
    assert aligned.n_genes == 2
    np.testing.assert_array_equal(aligned.embedding[0], ge.embedding[2])
    np.testing.assert_array_equal(aligned.embedding[1], ge.embedding[0])


def test_gene_embeddings_align_to_no_overlap_raises():
    """align_to raises when no target_ids are in this embedding."""
    ge = _make_gene_emb(["a", "b"])
    with pytest.raises(ValueError) as e:
        ge.align_to(["x", "y"])
    assert "No genes in target_ids" in str(e.value)


def test_gene_embeddings_set_unaligned_vocab_output_aligned_and_gene_dim_matches():
    """From unaligned embeddings with different vocabs, output is aligned and gene dim matches."""
    # Shared ontology: ensembl_gene. Different native vocabs and order/subset per embedding.
    e1, e2, e3, e4 = "ens_a", "ens_b", "ens_c", "ens_d"
    # Embedding A: 4 genes, native vocab vA_*
    ann_a = pd.DataFrame(
        {
            FM_DEFS.VOCAB_NAME: ["vA_1", "vA_2", "vA_3", "vA_4"],
            ONTOLOGIES.ENSEMBL_GENE: [e1, e2, e3, e4],
        }
    )
    ge_a = GeneEmbeddings(
        embedding=np.arange(4 * 2, dtype=np.float64).reshape(4, 2),
        ordered_gene_ids=ann_a[FM_DEFS.VOCAB_NAME].tolist(),
        gene_annotations=ann_a,
        model_name="ModelA",
    )
    # Embedding B: 3 genes, different order (e2, e1, e3), native vocab vB_*
    ann_b = pd.DataFrame(
        {
            FM_DEFS.VOCAB_NAME: ["vB_1", "vB_2", "vB_3"],
            ONTOLOGIES.ENSEMBL_GENE: [e2, e1, e3],
        }
    )
    ge_b = GeneEmbeddings(
        embedding=np.arange(3 * 3, dtype=np.float64).reshape(3, 3),
        ordered_gene_ids=ann_b[FM_DEFS.VOCAB_NAME].tolist(),
        gene_annotations=ann_b,
        model_name="ModelB",
    )

    aligned_set = GeneEmbeddingsSet.from_gene_embeddings(
        [ge_a, ge_b], align_on=ONTOLOGIES.ENSEMBL_GENE
    )

    common = aligned_set.common_gene_ids
    assert len(common) == 3
    assert set(common) == {e1, e2, e3}
    assert aligned_set.n_common_genes == 3

    for key, emb in aligned_set.items():
        assert emb.gene_ids_in_ontology(ONTOLOGIES.ENSEMBL_GENE) == common
        assert emb.n_genes == aligned_set.n_common_genes

    summary = aligned_set.summary
    assert list(summary["n_genes"]) == [3, 3]
    assert list(summary["key"]) == ["ModelA", "ModelB"]


class TestComputeAttentionReordering:
    """Test that compute_attention with target_ids produces correctly ordered results."""

    def test_target_ids_none_matches_full(self):
        """Attention with target_ids=None equals full computation."""
        ae = make_attended_embeddings(n_genes=10, seed=42)
        full = ae.compute_attention(layer_idx=0, target_ids=None)
        also_full = ae.compute_attention(
            layer_idx=0, target_ids=ae.gene_embeddings.ordered_gene_ids
        )
        np.testing.assert_allclose(full, also_full, atol=1e-5)

    def test_target_ids_subset_values_match(self):
        """Subset attention values match corresponding entries in full matrix."""
        ae = make_attended_embeddings(n_genes=10, seed=42)
        gene_ids = ae.gene_embeddings.ordered_gene_ids

        full = ae.compute_attention(layer_idx=0, target_ids=None, apply_softmax=False)
        subset_ids = [gene_ids[2], gene_ids[5], gene_ids[7]]
        subset = ae.compute_attention(
            layer_idx=0, target_ids=subset_ids, apply_softmax=False
        )

        idx = [2, 5, 7]
        expected = full[np.ix_(idx, idx)]
        np.testing.assert_allclose(subset, expected, atol=1e-5)

    def test_target_ids_reordering(self):
        """Reversed target_ids produces transposed-like reordering."""
        ae = make_attended_embeddings(n_genes=10, seed=42)
        gene_ids = ae.gene_embeddings.ordered_gene_ids

        forward = ae.compute_attention(layer_idx=0, target_ids=gene_ids[:5])
        backward = ae.compute_attention(layer_idx=0, target_ids=gene_ids[:5][::-1])

        # backward should be forward with rows and cols reversed
        np.testing.assert_allclose(backward, forward[::-1, ::-1], atol=1e-5)


class TestGetSpecificAttentionsConsistency:
    """Test that get_specific_attentions returns correct values."""

    def test_values_match_direct_computation(self):
        """Edge values from get_specific_attentions match direct indexing."""
        ae = make_attended_embeddings(n_genes=10, n_layers=2, seed=42)
        gene_ids = ae.gene_embeddings.ordered_gene_ids

        edges = pd.DataFrame(
            {
                "from_gene": [gene_ids[0], gene_ids[3]],
                "to_gene": [gene_ids[1], gene_ids[7]],
            }
        )

        result = ae.get_specific_attentions(
            edges, target_ids=gene_ids, apply_softmax=False
        )

        for layer_idx in range(ae.n_layers):
            full_attn = ae.compute_attention(
                layer_idx=layer_idx, target_ids=gene_ids, apply_softmax=False
            )
            layer_rows = result[result["layer"] == layer_idx]

            for _, row in layer_rows.iterrows():
                i = gene_ids.index(row["from_gene"])
                j = gene_ids.index(row["to_gene"])
                np.testing.assert_allclose(row["attention"], full_attn[i, j], atol=1e-5)


class TestCrossModelReextractionCompleteness:
    """Test that cross-model re-extraction produces no NaNs."""

    def test_reextract_union_no_nans(self):
        """get_top_attentions with reextract_union=True has no NaN values."""
        attended_set, shared_ids = make_attended_embeddings_set(
            n_models=2, n_shared=15, n_unique_per_model=5, seed=42
        )

        result = attended_set.get_top_attentions(
            k=20,
            reextract_union=True,
            compute_ranks=True,
        )

        assert (
            not result["attention"].isna().any()
        ), f"Found {result['attention'].isna().sum()} NaN attention values"
        assert (
            not result["attention_rank"].isna().any()
        ), f"Found {result['attention_rank'].isna().sum()} NaN rank values"

    def test_reextract_union_pivot_complete(self):
        """Pivoted re-extraction has no NaN cells (all model×layer combos present)."""
        attended_set, shared_ids = make_attended_embeddings_set(
            n_models=2, n_shared=15, n_unique_per_model=5, seed=42
        )

        result = attended_set.get_top_attentions(
            k=20,
            reextract_union=True,
        )

        pivot = result.pivot(
            index=["from_gene", "to_gene"],
            columns=["model", "layer"],
            values="attention",
        )

        n_nans = pivot.isna().sum().sum()
        assert n_nans == 0, (
            f"Pivot has {n_nans} NaN cells. "
            f"Shape: {pivot.shape}, "
            f"NaN per column:\n{pivot.isna().sum()}"
        )

    def test_all_edges_use_common_genes_only(self):
        """Re-extracted edges only reference genes in the common set."""
        attended_set, shared_ids = make_attended_embeddings_set(
            n_models=2, n_shared=15, n_unique_per_model=5, seed=42
        )

        result = attended_set.get_top_attentions(
            k=20,
            reextract_union=True,
        )

        common_set = set(attended_set.common_gene_ids)
        from_genes = set(result["from_gene"].unique())
        to_genes = set(result["to_gene"].unique())

        assert from_genes.issubset(common_set), (
            f"from_gene contains genes not in common set: " f"{from_genes - common_set}"
        )
        assert to_genes.issubset(common_set), (
            f"to_gene contains genes not in common set: " f"{to_genes - common_set}"
        )


class TestConsensusTopAttentionsCompleteness:
    """Same completeness tests for consensus path."""

    def test_consensus_reextract_no_nans(self):
        """get_consensus_top_attentions with reextract_union has no NaNs."""
        attended_set, shared_ids = make_attended_embeddings_set(
            n_models=2, n_shared=15, n_unique_per_model=5, seed=42
        )

        result = attended_set.get_consensus_top_attentions(
            k=20,
            reextract_union=True,
        )

        assert not result["attention"].isna().any()


class TestWithDifferentVocabularies:
    """Test cross-model operations when models use different native vocabularies."""

    def test_different_vocab_no_nans(self):
        """Models with different vocab names still produce complete re-extraction."""
        model_a, model_b, shared_ids = make_foundation_model_pair(
            n_shared=15,
            n_unique_a=5,
            n_unique_b=5,
            use_different_vocab=True,
            seed=42,
        )

        fm = FoundationModels(models=[model_a, model_b])
        attended_set = AttendedEmbeddingsSet.from_static(fm)

        result = attended_set.get_top_attentions(
            k=20,
            reextract_union=True,
        )

        pivot = result.pivot(
            index=["from_gene", "to_gene"],
            columns=["model", "layer"],
            values="attention",
        )

        assert (
            pivot.isna().sum().sum() == 0
        ), "NaN values found with different vocabulary models"


def test_scoping_current_behavior():
    """Baseline: constant fields fold into label, varying fields go in keys."""
    # Varying category only
    scoped, constant = _scope(
        [
            _make_gene_emb(
                SCOPING_TEST_GENES,
                model_name="scGPT",
                dataset_name="ds1",
                category="adipocyte",
            ),
            _make_gene_emb(
                SCOPING_TEST_GENES,
                model_name="scGPT",
                dataset_name="ds1",
                category="T_cell",
            ),
        ]
    )
    assert set(scoped.values()) == {"adipocyte", "T_cell"}
    assert constant == "scGPT / ds1"

    # Varying model only
    scoped, constant = _scope(
        [
            _make_gene_emb(SCOPING_TEST_GENES, model_name="scGPT"),
            _make_gene_emb(SCOPING_TEST_GENES, model_name="scPRINT"),
        ]
    )
    assert set(scoped.values()) == {"scGPT", "scPRINT"}
    assert constant == ""

    # Two varying fields compose
    scoped, _ = _scope(
        [
            _make_gene_emb(
                SCOPING_TEST_GENES, model_name="scGPT", category="adipocyte"
            ),
            _make_gene_emb(SCOPING_TEST_GENES, model_name="scGPT", category="T_cell"),
            _make_gene_emb(
                SCOPING_TEST_GENES, model_name="scPRINT", category="adipocyte"
            ),
            _make_gene_emb(SCOPING_TEST_GENES, model_name="scPRINT", category="T_cell"),
        ]
    )
    assert set(scoped.values()) == {
        "scGPT/adipocyte",
        "scGPT/T_cell",
        "scPRINT/adipocyte",
        "scPRINT/T_cell",
    }


def test_scoping_with_layer_idx():
    """After adding layer_idx to SCOPING_FIELDS: constant layer folds, varying layer appears in keys."""
    # Constant layer_idx=0 across all embeddings -> keys unchanged from pre-migration
    scoped, _ = _scope(
        [
            _make_gene_emb(
                SCOPING_TEST_GENES,
                model_name="scGPT",
                dataset_name="ds1",
                category="adipocyte",
                layer_idx=0,
            ),
            _make_gene_emb(
                SCOPING_TEST_GENES,
                model_name="scGPT",
                dataset_name="ds1",
                category="T_cell",
                layer_idx=0,
            ),
        ]
    )

    assert set(scoped.values()) == {"adipocyte", "T_cell"}

    # Varying layer -> layer in keys
    scoped, constant = _scope(
        [
            _make_gene_emb(
                SCOPING_TEST_GENES,
                model_name="scGPT",
                dataset_name="ds1",
                category="adipocyte",
                layer_idx=0,
            ),
            _make_gene_emb(
                SCOPING_TEST_GENES,
                model_name="scGPT",
                dataset_name="ds1",
                category="adipocyte",
                layer_idx=5,
            ),
            _make_gene_emb(
                SCOPING_TEST_GENES,
                model_name="scGPT",
                dataset_name="ds1",
                category="adipocyte",
                layer_idx=11,
            ),
        ]
    )

    assert len(set(scoped.values())) == 3
    assert constant == "scGPT / ds1 / adipocyte"


def _metadata_dict_from_model(fm: FoundationModel) -> dict:
    """Rebuild metadata dict for cloning a FoundationModel in tests."""
    return {
        FM_DEFS.MODEL_NAME: fm.model_name,
        FM_DEFS.MODEL_VARIANT: fm.model_variant,
        FM_DEFS.N_GENES: fm.n_genes,
        FM_DEFS.N_VOCAB: fm.n_vocab,
        FM_DEFS.ORDERED_VOCABULARY: fm.ordered_vocabulary,
        FM_DEFS.EMBED_DIM: fm.embed_dim,
        FM_DEFS.N_LAYERS: fm.n_layers,
        FM_DEFS.N_HEADS: fm.n_heads,
    }


def _clone_fm_with_dge(
    fm: FoundationModel, dge: DatasetGeneEmbeddings
) -> FoundationModel:
    return FoundationModel(
        weights=fm.weights,
        gene_annotations=fm.gene_annotations,
        model_metadata=_metadata_dict_from_model(fm),
        dataset_gene_embeddings=dge,
    )


def _make_layer_grid_dge(
    *,
    n_layers: int,
    n_categories: int,
    gene_ids: List[str],
    embed_dim: int = 8,
    layers_per_emb: Optional[List[int]] = None,
) -> DatasetGeneEmbeddings:
    """Expression embeddings: each category repeats one matrix per listed layer."""
    annotations = make_gene_annotations(gene_ids)
    rng = np.random.default_rng(42)
    mats: List[GeneEmbeddings] = []
    layers_cycle = (
        layers_per_emb if layers_per_emb is not None else list(range(n_layers))
    )
    for ci in range(n_categories):
        for layer in layers_cycle:
            emb_mx = rng.standard_normal((len(gene_ids), embed_dim)).astype(np.float32)
            mats.append(
                GeneEmbeddings(
                    embedding=emb_mx,
                    ordered_gene_ids=list(gene_ids),
                    gene_annotations=annotations,
                    model_name="TestModel",
                    layer_idx=layer,
                    dataset_name="ds1",
                    category=f"cluster_{ci}",
                )
            )
    ges = GeneEmbeddingsSet.from_gene_embeddings(mats)
    return DatasetGeneEmbeddings({"ds1": ges})


def test_foundation_model_validate_dataset_gene_embeddings_passes_for_full_layer_grid():
    """Verification succeeds when every layer index 0..n_layers-1 appears across embeddings."""
    gene_ids = make_gene_ids(6)
    dge = _make_layer_grid_dge(n_layers=3, n_categories=2, gene_ids=gene_ids)
    template = make_foundation_model(
        n_genes=6,
        embed_dim=8,
        n_layers=3,
        gene_ids=gene_ids,
    )
    fm = _clone_fm_with_dge(template, dge)
    report = fm.validate_dataset_gene_embeddings(verbose=False)
    assert report["ok"]
    assert len(report["datasets"]) == 1
    assert report["datasets"][0]["distinct_layer_indices"] == (0, 1, 2)
    assert report["datasets"][0]["spot_check_note"] is not None


def test_foundation_model_validate_dataset_gene_embeddings_fails_without_dataset_embeddings():
    """Verification reports failure when dataset_gene_embeddings was never attached."""
    fm = make_foundation_model()
    report = fm.validate_dataset_gene_embeddings()
    assert not report["ok"]
    assert "dataset_gene_embeddings is None" in report["datasets"][0]["errors"][0]


def test_foundation_model_validate_dataset_gene_embeddings_raise_on_fail():
    """raise_on_fail surfaces dataset embedding absence."""
    fm = make_foundation_model()
    with pytest.raises(ValueError, match="Validation failed"):
        fm.validate_dataset_gene_embeddings(raise_on_fail=True)


def test_foundation_model_validate_dataset_gene_embeddings_missing_layer_indices():
    """Verification fails when union of layer_idx omits a layer."""
    gene_ids = make_gene_ids(5)
    dge = _make_layer_grid_dge(
        n_layers=3,
        n_categories=2,
        gene_ids=gene_ids,
        layers_per_emb=[0, 2],
    )
    template = make_foundation_model(
        n_genes=5,
        embed_dim=8,
        n_layers=3,
        gene_ids=gene_ids,
    )
    fm = _clone_fm_with_dge(template, dge)
    report = fm.validate_dataset_gene_embeddings(verbose=False)
    assert not report["ok"]
    assert any("missing layers [1]" in err for err in report["datasets"][0]["errors"])


def test_foundation_model_validate_dataset_gene_embeddings_detects_zero_variance_embedding():
    """Verification fails when an embedding matrix is constant (std == 0)."""
    gene_ids = make_gene_ids(4)
    annotations = make_gene_annotations(gene_ids)
    zero_mat = np.zeros((4, 8), dtype=np.float32)
    mats = [
        GeneEmbeddings(
            embedding=zero_mat,
            ordered_gene_ids=list(gene_ids),
            gene_annotations=annotations,
            model_name="TestModel",
            layer_idx=0,
            dataset_name="ds1",
            category="c0",
        ),
        GeneEmbeddings(
            embedding=np.ones((4, 8), dtype=np.float32),
            ordered_gene_ids=list(gene_ids),
            gene_annotations=annotations,
            model_name="TestModel",
            layer_idx=1,
            dataset_name="ds1",
            category="c0",
        ),
        GeneEmbeddings(
            embedding=np.ones((4, 8), dtype=np.float32) * 2,
            ordered_gene_ids=list(gene_ids),
            gene_annotations=annotations,
            model_name="TestModel",
            layer_idx=2,
            dataset_name="ds1",
            category="c0",
        ),
    ]
    ges = GeneEmbeddingsSet.from_gene_embeddings(mats)
    dge = DatasetGeneEmbeddings({"ds1": ges})
    template = make_foundation_model(
        n_genes=4,
        embed_dim=8,
        n_layers=3,
        gene_ids=gene_ids,
    )
    fm = _clone_fm_with_dge(template, dge)
    report = fm.validate_dataset_gene_embeddings(verbose=False)
    assert not report["ok"]
    assert any("embedding.std()" in err for err in report["datasets"][0]["errors"])


def test_foundation_model_validate_dataset_gene_embeddings_unknown_dataset_key_raises():
    """validate_dataset_gene_embeddings(dataset_name=...) raises KeyError for missing dataset."""
    gene_ids = make_gene_ids(4)
    dge = _make_layer_grid_dge(n_layers=2, n_categories=1, gene_ids=gene_ids)
    template = make_foundation_model(
        n_genes=4,
        embed_dim=8,
        n_layers=2,
        gene_ids=gene_ids,
    )
    fm = _clone_fm_with_dge(template, dge)
    with pytest.raises(KeyError):
        fm.validate_dataset_gene_embeddings(dataset_name="missing_dataset")
