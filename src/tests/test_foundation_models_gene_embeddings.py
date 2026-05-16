"""Tests for ``napistu_torch.foundation_models.gene_embeddings``."""

import numpy as np
import pandas as pd
import pytest
from foundation_model_factories import (
    SCOPING_TEST_GENES,
    _make_gene_annotations,
    _make_gene_emb,
)
from napistu.ontologies.constants import ONTOLOGIES

from napistu_torch.foundation_models.constants import FM_DEFS, SCOPING_FIELDS
from napistu_torch.foundation_models.gene_embeddings import (
    GeneEmbeddings,
    GeneEmbeddingsSet,
    _build_embedding_metadata,
    _compute_scoped_keys_for_fields,
)


def _scope(embeddings):
    data = {emb.source_label: emb for emb in embeddings}
    metadata = _build_embedding_metadata(data)
    return _compute_scoped_keys_for_fields(metadata, SCOPING_FIELDS)


def test_gene_embeddings_rejects_3d_array():
    genes = ["g1", "g2"]
    with pytest.raises(ValueError, match="2-dimensional"):
        GeneEmbeddings(
            embedding=np.random.randn(2, 3, 4),
            ordered_gene_ids=genes,
            gene_annotations=_make_gene_annotations(genes),
        )


def test_gene_embeddings_rejects_annotation_mismatch():
    genes = ["g1", "g2"]
    with pytest.raises(ValueError, match="gene_annotations has 1 rows"):
        GeneEmbeddings(
            embedding=np.random.randn(2, 4),
            ordered_gene_ids=genes,
            gene_annotations=_make_gene_annotations(["g1"]),
        )


def test_gene_embeddings_rejects_duplicate_ids():
    genes = ["g1", "g2"]
    with pytest.raises(ValueError, match="unique"):
        GeneEmbeddings(
            embedding=np.random.randn(2, 4),
            ordered_gene_ids=["g1", "g1"],
            gene_annotations=_make_gene_annotations(genes),
        )


def test_gene_embeddings_source_label_with_layer():
    ge = _make_gene_emb(["g1", "g2", "g3"], model_name="scGPT", layer_idx=0)
    assert ge.source_label == "scGPT/layer_0"
    assert ge.layer_idx == 0


def test_gene_embeddings_set_single_wraps_directly():
    genes = ["g1", "g2", "g3"]
    ge = _make_gene_emb(genes, model_name="ModelA", category="type1")
    ge_set = GeneEmbeddingsSet.from_gene_embeddings([ge])
    assert ge_set.n_embeddings == 1
    assert ge_set.n_common_genes == 3


def test_gene_embeddings_set_rejects_duplicates():
    genes = ["g1", "g2", "g3"]
    ge1 = _make_gene_emb(genes, model_name="ModelA", category="type1")
    ge2 = _make_gene_emb(genes, model_name="ModelA", category="type1")
    with pytest.raises(ValueError, match="Duplicate source_label"):
        GeneEmbeddingsSet.from_gene_embeddings([ge1, ge2])


def test_gene_embeddings_set_aligns_across_vocabs():
    e1, e2, e3, e4 = "ens_a", "ens_b", "ens_c", "ens_d"
    ann_a = pd.DataFrame(
        {
            FM_DEFS.VOCAB_NAME: ["vA_1", "vA_2", "vA_3", "vA_4"],
            ONTOLOGIES.ENSEMBL_GENE: [e1, e2, e3, e4],
        }
    )
    ge_a = GeneEmbeddings(
        embedding=np.arange(8, dtype=np.float64).reshape(4, 2),
        ordered_gene_ids=ann_a[FM_DEFS.VOCAB_NAME].tolist(),
        gene_annotations=ann_a,
        model_name="ModelA",
    )
    ann_b = pd.DataFrame(
        {
            FM_DEFS.VOCAB_NAME: ["vB_1", "vB_2", "vB_3"],
            ONTOLOGIES.ENSEMBL_GENE: [e2, e1, e3],
        }
    )
    ge_b = GeneEmbeddings(
        embedding=np.arange(9, dtype=np.float64).reshape(3, 3),
        ordered_gene_ids=ann_b[FM_DEFS.VOCAB_NAME].tolist(),
        gene_annotations=ann_b,
        model_name="ModelB",
    )
    aligned = GeneEmbeddingsSet.from_gene_embeddings(
        [ge_a, ge_b], align_on=ONTOLOGIES.ENSEMBL_GENE
    )
    assert aligned.n_common_genes == 3
    assert set(aligned.common_gene_ids) == {e1, e2, e3}
    for _, emb in aligned.items():
        assert (
            emb.gene_ids_in_ontology(ONTOLOGIES.ENSEMBL_GENE) == aligned.common_gene_ids
        )


def test_scoping_varying_category_only():
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


def test_scoping_varying_model_only():
    scoped, constant = _scope(
        [
            _make_gene_emb(SCOPING_TEST_GENES, model_name="scGPT"),
            _make_gene_emb(SCOPING_TEST_GENES, model_name="scPRINT"),
        ]
    )
    assert set(scoped.values()) == {"scGPT", "scPRINT"}
    assert constant == ""


def test_scoping_varying_layer():
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
