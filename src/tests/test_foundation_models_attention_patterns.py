"""Tests for ``napistu_torch.foundation_models.attention_patterns``."""

import numpy as np
import pandas as pd
import pytest
from foundation_model_factories import (
    _make_layer_grid_embeddings,
    make_foundation_model_pair,
    make_gene_annotations,
    make_gene_ids,
)
from napistu.ontologies.constants import ONTOLOGIES

from napistu_torch.foundation_models.attention_patterns import (
    AttentionPatternsInputs,
    _group_embeddings_by_model_and_category,
)
from napistu_torch.foundation_models.foundation_models import FoundationModels
from napistu_torch.foundation_models.gene_embeddings import (
    GeneEmbeddings,
    GeneEmbeddingsSet,
    _get_model_label,
)


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
    fm = FoundationModels(models=[model_a, model_b])

    all_embeddings = []
    for m in (model_a, model_b):
        gene_ids = m.gene_annotations[ONTOLOGIES.ENSEMBL_GENE].tolist()
        embeddings = _make_layer_grid_embeddings(
            n_layers=m.n_layers,
            n_categories=1,
            gene_ids=gene_ids,
            embed_dim=m.embed_dim,
            model_name=m.model_name,
            model_variant=m.model_variant,
            gene_annotations=m.gene_annotations,
        )
        all_embeddings.extend(embeddings)

    category_embeddings = [ge for ge in all_embeddings if ge.category == "cluster_0"]
    embeddings_set = GeneEmbeddingsSet.from_gene_embeddings(
        category_embeddings,
        align_on=ONTOLOGIES.ENSEMBL_GENE,
        verbose=False,
    )
    attended_set = AttentionPatternsInputs(
        embeddings_set=embeddings_set,
        foundation_models=fm,
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
