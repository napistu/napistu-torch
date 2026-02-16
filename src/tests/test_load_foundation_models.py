"""Tests for foundation model data structures and validation."""

import numpy as np
import pandas as pd
import pytest
from napistu.ontologies.constants import ONTOLOGIES

from napistu_torch.load.constants import FM_DEFS
from napistu_torch.load.foundation_models import (
    DatasetGeneEmbeddings,
    GeneEmbeddings,
    GeneEmbeddingsSet,
)

# utility functions


def _make_gene_annotations(genes):
    """DataFrame valid for GeneAnnotations (vocab_name + ensembl_gene)."""
    return pd.DataFrame(
        {
            FM_DEFS.VOCAB_NAME: list(genes),
            ONTOLOGIES.ENSEMBL_GENE: list(genes),
        }
    )


def _make_gene_emb(genes, embed_dim=4, model_name=None, category=None):
    """Minimal GeneEmbeddings for tests."""
    n = len(genes)
    return GeneEmbeddings(
        embedding=np.arange(n * embed_dim, dtype=np.float64).reshape(n, embed_dim),
        ordered_gene_ids=list(genes),
        gene_annotations=_make_gene_annotations(genes),
        model_name=model_name,
        category=category,
    )


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
    assert set(ge_set2.keys()) == {"ModelA/type1", "ModelA/type2"}

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


# --- GeneEmbeddingsSet ---


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
