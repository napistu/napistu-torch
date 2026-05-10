"""Tests for ``napistu_torch.foundation_models.foundation_models``."""

import pytest
from foundation_model_factories import (
    _clone_fm_with_dge,
    _make_all_layer_idx_none_dge,
    _make_layer_grid_dge,
    make_foundation_model,
    make_gene_ids,
)


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
