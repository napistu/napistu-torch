"""Tests for foundation model data structures and validation."""

import numpy as np
import pandas as pd
import pytest
from foundation_model_factories import (
    SCOPING_TEST_GENES,
    _clone_fm_with_dge,
    _make_all_layer_idx_none_dge,
    _make_gene_annotations,
    _make_gene_emb,
    _make_layer_grid_dge,
    make_foundation_model,
    make_foundation_model_pair,
    make_gene_annotations,
    make_gene_ids,
)
from napistu.ontologies.constants import ONTOLOGIES

from napistu_torch.load.constants import FM_DEFS
from napistu_torch.load.foundation_models import (
    AttendedEmbeddingsSet,
    FoundationModels,
    GeneEmbeddings,
    GeneEmbeddingsSet,
    _build_embedding_metadata,
    _compute_scoped_keys,
    _get_model_label,
    _group_embeddings_by_model_and_category,
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _scope(embeddings):
    data = {emb.source_label: emb for emb in embeddings}
    metadata = _build_embedding_metadata(data)
    return _compute_scoped_keys(metadata)


# ---------------------------------------------------------------------------
# GeneEmbeddings
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# GeneEmbeddingsSet
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Scoping
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# validate_dataset_gene_embeddings
# ---------------------------------------------------------------------------


def test_validate_dge_passes_full_layer_grid():
    gene_ids = make_gene_ids(6)
    dge = _make_layer_grid_dge(
        n_layers=3, n_categories=2, gene_ids=gene_ids, model_name="TestModel"
    )
    fm = _clone_fm_with_dge(
        make_foundation_model(n_genes=6, embed_dim=8, n_layers=3, gene_ids=gene_ids),
        dge,
    )
    report = fm.validate_dataset_gene_embeddings(verbose=False)
    assert report["ok"]
    assert report["datasets"][0]["distinct_layer_indices"] == (0, 1, 2)


def test_validate_dge_fails_without_embeddings():
    fm = make_foundation_model()
    report = fm.validate_dataset_gene_embeddings()
    assert not report["ok"]
    assert "dataset_gene_embeddings is None" in report["datasets"][0]["errors"][0]


def test_validate_dge_fails_all_layer_idx_none():
    gene_ids = make_gene_ids(5)
    dge = _make_all_layer_idx_none_dge(gene_ids=gene_ids, n_embeddings=3)
    fm = _clone_fm_with_dge(
        make_foundation_model(n_genes=5, embed_dim=8, n_layers=12, gene_ids=gene_ids),
        dge,
    )
    report = fm.validate_dataset_gene_embeddings(verbose=False)
    assert not report["ok"]
    assert any(
        "layer_idx=None" in e and "residual" in e
        for e in report["datasets"][0]["errors"]
    )


def test_validate_dge_fails_missing_layer():
    gene_ids = make_gene_ids(5)
    dge = _make_layer_grid_dge(
        n_layers=3,
        n_categories=2,
        gene_ids=gene_ids,
        layers_per_emb=[0, 2],
        model_name="TestModel",
    )
    fm = _clone_fm_with_dge(
        make_foundation_model(n_genes=5, embed_dim=8, n_layers=3, gene_ids=gene_ids),
        dge,
    )
    report = fm.validate_dataset_gene_embeddings(verbose=False)
    assert not report["ok"]
    assert any("missing layers [1]" in e for e in report["datasets"][0]["errors"])


def test_validate_dge_raises_on_missing_dataset_key():
    gene_ids = make_gene_ids(4)
    dge = _make_layer_grid_dge(
        n_layers=2, n_categories=1, gene_ids=gene_ids, model_name="TestModel"
    )
    fm = _clone_fm_with_dge(
        make_foundation_model(n_genes=4, embed_dim=8, n_layers=2, gene_ids=gene_ids),
        dge,
    )
    with pytest.raises(KeyError):
        fm.validate_dataset_gene_embeddings(dataset_name="missing_dataset")


# ---------------------------------------------------------------------------
# LayerwiseAttentionInputs
# ---------------------------------------------------------------------------


def test_compute_attention_full_equals_explicit_target(attended_embeddings):
    ae = attended_embeddings
    full = ae.compute_attention(layer_idx=0)
    also_full = ae.compute_attention(layer_idx=0, target_ids=ae.ordered_gene_ids)
    np.testing.assert_allclose(full, also_full, atol=1e-5)


def test_compute_attention_subset_matches_full(attended_embeddings):
    ae = attended_embeddings
    gene_ids = ae.ordered_gene_ids
    full = ae.compute_attention(layer_idx=0, apply_softmax=False)
    subset = ae.compute_attention(
        layer_idx=0,
        target_ids=[gene_ids[2], gene_ids[5], gene_ids[7]],
        apply_softmax=False,
    )
    np.testing.assert_allclose(subset, full[np.ix_([2, 5, 7], [2, 5, 7])], atol=1e-5)


def test_compute_attention_reversed_target_ids(attended_embeddings):
    ae = attended_embeddings
    gene_ids = ae.ordered_gene_ids[:5]
    forward = ae.compute_attention(layer_idx=0, target_ids=gene_ids)
    backward = ae.compute_attention(layer_idx=0, target_ids=gene_ids[::-1])
    np.testing.assert_allclose(backward, forward[::-1, ::-1], atol=1e-5)


def test_get_specific_attentions_match_direct(attended_embeddings):
    ae = attended_embeddings
    gene_ids = ae.ordered_gene_ids
    edges = pd.DataFrame(
        {
            "from_gene": [gene_ids[0], gene_ids[3]],
            "to_gene": [gene_ids[1], gene_ids[7]],
        }
    )
    result = ae.get_specific_attentions(edges, target_ids=gene_ids, apply_softmax=False)
    for layer_idx in range(ae.n_layers):
        full = ae.compute_attention(
            layer_idx=layer_idx, target_ids=gene_ids, apply_softmax=False
        )
        for _, row in result[result["layer"] == layer_idx].iterrows():
            i = gene_ids.index(row["from_gene"])
            j = gene_ids.index(row["to_gene"])
            np.testing.assert_allclose(row["attention"], full[i, j], atol=1e-5)


# ---------------------------------------------------------------------------
# AttendedEmbeddingsSet
# ---------------------------------------------------------------------------


def test_reextract_union_no_nans(attended_embeddings_set):
    result = attended_embeddings_set.get_top_attentions(
        k=20, reextract_union=True, compute_ranks=True
    )
    assert not result["attention"].isna().any()
    assert not result["attention_rank"].isna().any()


def test_reextract_union_pivot_complete(attended_embeddings_set):
    result = attended_embeddings_set.get_top_attentions(k=20, reextract_union=True)
    pivot = result.pivot(
        index=["from_gene", "to_gene"],
        columns=["model", "layer"],
        values="attention",
    )
    assert pivot.isna().sum().sum() == 0


def test_edges_use_common_genes_only(attended_embeddings_set):
    result = attended_embeddings_set.get_top_attentions(k=20, reextract_union=True)
    common = set(attended_embeddings_set.common_gene_ids)
    assert set(result["from_gene"].unique()).issubset(common)
    assert set(result["to_gene"].unique()).issubset(common)


def test_consensus_reextract_no_nans(attended_embeddings_set):
    result = attended_embeddings_set.get_consensus_top_attentions(
        k=20, reextract_union=True
    )
    assert not result["attention"].isna().any()


def test_different_vocab_alignment_no_nans():
    """Models with different native vocabs align correctly via Ensembl IDs."""
    model_a, model_b, _ = make_foundation_model_pair(
        n_shared=15,
        n_unique_a=5,
        n_unique_b=5,
        n_layers_a=2,
        n_layers_b=2,
        use_different_vocab=True,
        seed=42,
    )
    models_with_dge = []
    for m in (model_a, model_b):
        gene_ids = m.gene_annotations[ONTOLOGIES.ENSEMBL_GENE].tolist()
        dge = _make_layer_grid_dge(
            n_layers=m.n_layers,
            n_categories=1,
            gene_ids=gene_ids,
            embed_dim=m.embed_dim,
            model_name=m.model_name,
            model_variant=m.model_variant,
            gene_annotations=m.gene_annotations,
        )
        models_with_dge.append(_clone_fm_with_dge(m, dge))
    fm = FoundationModels(models=models_with_dge)
    attended_set = AttendedEmbeddingsSet.from_expression(
        fm,
        dataset_name="ds1",
        category="cluster_0",
        verbose=False,
    )
    result = attended_set.get_top_attentions(k=20, reextract_union=True)
    pivot = result.pivot(
        index=["from_gene", "to_gene"],
        columns=["model", "layer"],
        values="attention",
    )
    assert pivot.isna().sum().sum() == 0


def test_group_embeddings_by_model_and_category():
    """Groups correctly and rejects bad inputs."""

    genes = make_gene_ids(5)
    annotations = make_gene_annotations(genes)

    # Two models, two layers each, one category — happy path
    embeddings = []
    for model_name in ("ModelA", "ModelB"):
        for layer_idx in (0, 1):
            embeddings.append(
                GeneEmbeddings(
                    embedding=np.random.randn(5, 8).astype(np.float32),
                    ordered_gene_ids=genes,
                    gene_annotations=annotations,
                    model_name=model_name,
                    layer_idx=layer_idx,
                    dataset_name="ds1",
                    category="cluster_0",
                )
            )

    ge_set = GeneEmbeddingsSet.from_gene_embeddings(embeddings)
    embedding_to_model = {
        key: _get_model_label(emb.model_name, emb.model_variant)
        for key, emb in ge_set.items()
    }

    groups = _group_embeddings_by_model_and_category(ge_set, embedding_to_model)

    assert set(groups.keys()) == {("ModelA", "cluster_0"), ("ModelB", "cluster_0")}
    for (model_name, category), layer_dict in groups.items():
        assert set(layer_dict.keys()) == {0, 1}
        for layer_idx, ge in layer_dict.items():
            assert ge.layer_idx == layer_idx
            assert ge.model_name == model_name
            assert ge.category == category

    # layer_idx=None is rejected
    ge_none = GeneEmbeddings(
        embedding=np.random.randn(5, 8).astype(np.float32),
        ordered_gene_ids=genes,
        gene_annotations=annotations,
        model_name="ModelA",
        layer_idx=None,
        dataset_name="ds1",
        category="cluster_0",
    )
    ge_set_none = GeneEmbeddingsSet.from_gene_embeddings([ge_none])
    with pytest.raises(ValueError, match="layer_idx=None"):
        _group_embeddings_by_model_and_category(
            ge_set_none,
            {list(ge_set_none.keys())[0]: "ModelA"},
        )

    # Mismatch between embedding model_name and embedding_to_model is caught
    ge_single = GeneEmbeddingsSet.from_gene_embeddings([embeddings[0]])
    with pytest.raises(ValueError, match="bug in the embedding_to_model mapping"):
        _group_embeddings_by_model_and_category(
            ge_single,
            {list(ge_single.keys())[0]: "ModelB"},  # deliberate mismatch
        )
