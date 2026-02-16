"""Tests for foundation model data structures and validation."""

import numpy as np
import pandas as pd
import pytest
import torch
from napistu.ontologies.constants import ONTOLOGIES
from pydantic import ValidationError

from napistu_torch.load.constants import FM_DEFS
from napistu_torch.load.foundation_models import (
    DatasetExpressionEmbeddings,
    ExpressionEmbeddings,
    GeneEmbeddings,
    GeneEmbeddingsSet,
)


def test_embeddings_field_validation():
    """Test embeddings field validation: shape, type, and conversion."""
    # Valid 3D numpy array (with category_dict for multi-category)
    valid_embeddings = np.random.randn(2, 10, 32)
    expr_emb = ExpressionEmbeddings(
        embeddings=valid_embeddings,
        ordered_genes=[f"gene_{i}" for i in range(10)],
        category_dict={0: "type1", 1: "type2"},
    )
    assert expr_emb.embeddings.shape == (2, 10, 32)

    # Valid torch.Tensor
    torch_embeddings = torch.randn(2, 10, 32)
    expr_emb = ExpressionEmbeddings(
        embeddings=torch_embeddings,
        ordered_genes=[f"gene_{i}" for i in range(10)],
        category_dict={0: "type1", 1: "type2"},
    )
    assert isinstance(expr_emb.embeddings, np.ndarray)
    assert expr_emb.embeddings.shape == (2, 10, 32)

    # Numpy converted to torch before validation (single category, no ordered_genes needed)
    numpy_embeddings = np.random.randn(1, 5, 16)
    expr_emb = ExpressionEmbeddings(embeddings=numpy_embeddings)
    assert isinstance(expr_emb.embeddings, np.ndarray)
    assert expr_emb.embeddings.shape == (1, 5, 16)

    # Invalid: 2D array (validation runs in __init__ before accessing shape[0])
    with pytest.raises(ValueError) as exc_info:
        ExpressionEmbeddings(embeddings=np.random.randn(10, 32))
    assert "3-dimensional" in str(exc_info.value)
    assert "got shape (10, 32)" in str(exc_info.value)

    # Invalid: 1D array
    with pytest.raises(ValueError) as exc_info:
        ExpressionEmbeddings(embeddings=np.random.randn(32))
    assert "3-dimensional" in str(exc_info.value)

    # Invalid: 4D array
    with pytest.raises(ValueError) as exc_info:
        ExpressionEmbeddings(embeddings=np.random.randn(2, 10, 32, 1))
    assert "3-dimensional" in str(exc_info.value)


def test_model_validation_and_defaults():
    """Test model-level validation: category_dict defaults and consistency checks."""
    # Single category: defaults to {0: "category_0"}
    embeddings = np.random.randn(1, 10, 32)
    expr_emb = ExpressionEmbeddings(
        embeddings=embeddings, ordered_genes=[f"gene_{i}" for i in range(10)]
    )
    assert expr_emb.category_dict == {0: "category_0"}
    assert expr_emb.n_categories == 1
    assert expr_emb.n_genes == 10
    assert expr_emb.embed_dim == 32

    # Multi-category: defaults to {0: "category_0", 1: "category_1", ...}
    embeddings = np.random.randn(3, 10, 32)
    expr_emb = ExpressionEmbeddings(
        embeddings=embeddings, ordered_genes=[f"gene_{i}" for i in range(10)]
    )
    assert expr_emb.category_dict == {0: "category_0", 1: "category_1", 2: "category_2"}

    # Valid multi-category with correct category_dict
    category_dict = {0: "type1", 1: "type2", 2: "type3"}
    expr_emb = ExpressionEmbeddings(
        embeddings=embeddings,
        ordered_genes=[f"gene_{i}" for i in range(10)],
        category_dict=category_dict,
    )
    assert expr_emb.category_dict == category_dict

    # Invalid: category_dict missing key
    with pytest.raises(ValidationError) as exc_info:
        ExpressionEmbeddings(
            embeddings=embeddings,
            ordered_genes=[f"gene_{i}" for i in range(10)],
            category_dict={0: "type1", 1: "type2"},  # Missing key 2
        )
    assert "category_dict must have keys 0 to 2" in str(exc_info.value)

    # Invalid: category_dict extra key
    with pytest.raises(ValidationError) as exc_info:
        ExpressionEmbeddings(
            embeddings=embeddings,
            ordered_genes=[f"gene_{i}" for i in range(10)],
            category_dict={0: "type1", 1: "type2", 2: "type3", 3: "type4"},  # Extra key
        )
    assert "category_dict must have keys 0 to 2" in str(exc_info.value)

    # Invalid: ordered_genes length mismatch
    with pytest.raises(ValidationError) as exc_info:
        ExpressionEmbeddings(
            embeddings=embeddings,
            ordered_genes=[f"gene_{i}" for i in range(5)],  # Wrong length
            category_dict=category_dict,
        )
    assert "ordered_genes has 5 entries but embeddings has 10 genes" in str(
        exc_info.value
    )


def test_dataset_expression_embeddings():
    """Test DatasetExpressionEmbeddings: get, keys, values, items, dict and list init."""
    emb1 = ExpressionEmbeddings(
        embeddings=np.random.randn(2, 10, 32),
        ordered_genes=[f"gene_{i}" for i in range(10)],
        category_dict={0: "type1", 1: "type2"},
        dataset_name="efthymiou",
    )
    emb2 = ExpressionEmbeddings(
        embeddings=np.random.randn(3, 10, 32),
        ordered_genes=[f"gene_{i}" for i in range(10)],
        category_dict={0: "a", 1: "b", 2: "c"},
        dataset_name="tabula_sapiens",
    )

    # Init from dict
    container = DatasetExpressionEmbeddings({"efthymiou": emb1, "tabula_sapiens": emb2})
    assert container.get("efthymiou") is emb1
    assert container["efthymiou"] is emb1
    assert container.get("tabula_sapiens") is emb2
    assert "efthymiou" in container
    assert "missing" not in container
    assert set(container.keys()) == {"efthymiou", "tabula_sapiens"}
    assert list(container.values()) == [emb1, emb2]
    assert len(list(container.items())) == 2
    assert "DatasetExpressionEmbeddings" in repr(container)

    # Init from list (keys from dataset_name)
    container2 = DatasetExpressionEmbeddings([emb1, emb2])
    assert container2.get("efthymiou") is emb1
    assert container2.get("tabula_sapiens") is emb2

    # KeyError when not found
    with pytest.raises(KeyError) as exc_info:
        container.get("nonexistent")
    assert "nonexistent" in str(exc_info.value)
    assert "Available datasets" in str(exc_info.value)

    # Duplicate dataset_name in list raises
    emb_dup = ExpressionEmbeddings(
        embeddings=np.random.randn(1, 10, 32),
        dataset_name="efthymiou",
    )
    with pytest.raises(ValueError) as exc_info:
        DatasetExpressionEmbeddings([emb1, emb_dup])
    assert "Duplicate dataset name" in str(exc_info.value)


# --- GeneEmbeddings ---


def _make_gene_annotations(genes):
    """DataFrame valid for GeneAnnotations (vocab_name + ensembl_gene)."""
    return pd.DataFrame(
        {
            FM_DEFS.VOCAB_NAME: list(genes),
            ONTOLOGIES.ENSEMBL_GENE: list(genes),
        }
    )


def _make_gene_emb(genes, embed_dim=4, model_name=None):
    """Minimal GeneEmbeddings for tests."""
    n = len(genes)
    return GeneEmbeddings(
        embedding=np.arange(n * embed_dim, dtype=np.float64).reshape(n, embed_dim),
        ordered_gene_ids=list(genes),
        gene_annotations=_make_gene_annotations(genes),
        model_name=model_name,
    )


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
