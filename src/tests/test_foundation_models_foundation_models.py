"""Tests for ``napistu_torch.foundation_models.foundation_models``."""

import json

import numpy as np
import pytest
from foundation_model_factories import (
    _make_layer_grid_embeddings,
    make_foundation_model,
    make_gene_annotations,
    make_gene_ids,
)

from napistu_torch.foundation_models.constants import FM_DEFS
from napistu_torch.foundation_models.foundation_models import (
    FoundationModel,
    FoundationModelStore,
)
from napistu_torch.foundation_models.gene_embeddings import GeneEmbeddings


def test_foundation_model_save(tmp_path):
    """save() writes weights and metadata, omits residuals, accepts path or store."""
    fm = make_foundation_model(
        n_genes=10, embed_dim=8, n_layers=2, model_name="TestModel"
    )

    # Accepts raw path
    model_dir = tmp_path / fm.disk_name
    fm.save(model_dir)

    assert (model_dir / FM_DEFS.WEIGHTS_FILENAME).exists()
    assert (model_dir / FM_DEFS.METADATA_FILENAME).exists()
    assert not (model_dir / FM_DEFS.RESIDUALS_INDEX_FILENAME).exists()

    # Weights contain attention and static embeddings
    data = np.load(model_dir / FM_DEFS.WEIGHTS_FILENAME, allow_pickle=True)
    assert FM_DEFS.STATIC_GENE_EMBEDDINGS in data
    assert FM_DEFS.ATTENTION_WEIGHTS in data

    # Metadata contains model info but not bundled residuals
    with open(model_dir / FM_DEFS.METADATA_FILENAME) as f:
        meta = json.load(f)
    assert meta[FM_DEFS.MODEL_METADATA][FM_DEFS.MODEL_NAME] == "TestModel"
    assert meta[FM_DEFS.MODEL_METADATA][FM_DEFS.N_LAYERS] == 2
    assert FM_DEFS.GENE_ANNOTATIONS in meta


def test_foundation_model_save_load_roundtrip(tmp_path):
    """save() and load() roundtrip: weights, metadata, and store reference."""
    fm = make_foundation_model(
        n_genes=10, embed_dim=8, n_layers=2, model_name="TestModel"
    )
    fm.save(tmp_path / fm.disk_name)
    loaded = FoundationModel.load(tmp_path / fm.disk_name)

    # Metadata
    assert loaded.model_name == fm.model_name
    assert loaded.n_genes == fm.n_genes
    assert loaded.n_layers == fm.n_layers
    assert loaded.embed_dim == fm.embed_dim
    assert loaded.ordered_vocabulary == fm.ordered_vocabulary

    # Static embedding
    np.testing.assert_array_equal(
        loaded.weights.static_gene_embeddings.embedding,
        fm.weights.static_gene_embeddings.embedding,
    )

    # Attention weights
    for orig, loaded_layer in zip(
        fm.weights.attention_layers, loaded.weights.attention_layers
    ):
        assert orig.layer_idx == loaded_layer.layer_idx
        np.testing.assert_array_equal(orig.W_q, loaded_layer.W_q)
        np.testing.assert_array_equal(orig.W_k, loaded_layer.W_k)

    # Load semantics
    assert loaded.store.model_dir == tmp_path / fm.disk_name


def test_foundation_model_load_validates(tmp_path):
    """load() raises on missing core files."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="weights.npz missing"):
        FoundationModel.load(tmp_path / "empty")


def test_store_save_residuals(tmp_path):
    """store.save_residuals() writes arrays, updates index, raises on bad inputs."""
    gene_ids = make_gene_ids(10)
    fm = make_foundation_model(n_genes=10, embed_dim=8, n_layers=2, gene_ids=gene_ids)
    embeddings = _make_layer_grid_embeddings(
        n_layers=2,
        n_categories=3,
        gene_ids=gene_ids,
        embed_dim=8,
        model_name=fm.model_name,
        dataset_name="ds1",
    )

    store = FoundationModelStore(tmp_path / fm.disk_name)
    fm.save(store)
    store.save_residuals(embeddings)

    # Index reflects all categories
    assert set(store.list_categories("ds1")) == {"cluster_0", "cluster_1", "cluster_2"}

    # Arrays have correct shape for one category
    stem = store.get_stem("ds1", "cluster_0")
    arrays, metadata_records = store.load_residual_arrays(stem)
    assert set(arrays.keys()) == {"layer_0", "layer_1"}
    assert all(arr.shape == (10, 8) for arr in arrays.values())
    assert len(metadata_records) == 2

    # Error cases — missing required metadata fields
    bad_ge = GeneEmbeddings(
        embedding=np.zeros((10, 8)),
        ordered_gene_ids=gene_ids,
        gene_annotations=make_gene_annotations(gene_ids),
        dataset_name=None,
        category="cluster_0",
        layer_idx=0,
    )
    with pytest.raises(ValueError, match="no dataset_name"):
        store.save_residuals([bad_ge])

    bad_ge_no_layer = GeneEmbeddings(
        embedding=np.zeros((10, 8)),
        ordered_gene_ids=gene_ids,
        gene_annotations=make_gene_annotations(gene_ids),
        dataset_name="ds1",
        category="cluster_0",
        layer_idx=None,
    )
    with pytest.raises(ValueError, match="no layer_idx"):
        store.save_residuals([bad_ge_no_layer])


def test_foundation_model_residuals_roundtrip(tmp_path):
    """store.save_residuals() / load_category_residuals() roundtrip and error cases."""
    gene_ids = make_gene_ids(10)
    fm = make_foundation_model(n_genes=10, embed_dim=8, n_layers=2, gene_ids=gene_ids)
    embeddings = _make_layer_grid_embeddings(
        n_layers=2,
        n_categories=2,
        gene_ids=gene_ids,
        embed_dim=8,
        model_name=fm.model_name,
        dataset_name="ds1",
    )

    store = FoundationModelStore(tmp_path / fm.disk_name)
    fm.save(store)
    store.save_residuals(embeddings)
    loaded = FoundationModel.load(store)

    # Shapes, metadata, and array values for one category
    layer_embeddings = loaded.load_category_residuals("ds1", "cluster_0")
    assert set(layer_embeddings.keys()) == {0, 1}
    for layer_idx, ge in layer_embeddings.items():
        assert ge.layer_idx == layer_idx
        assert ge.embedding.shape == (10, 8)
        assert ge.model_name == fm.model_name
        assert ge.category == "cluster_0"
        assert ge.dataset_name == "ds1"

    # Arrays match originals
    original = {
        ge.layer_idx: ge.embedding for ge in embeddings if ge.category == "cluster_0"
    }
    for layer_idx, ge in layer_embeddings.items():
        np.testing.assert_array_equal(ge.embedding, original[layer_idx])

    # Error cases
    fm_no_store = make_foundation_model(n_genes=10, embed_dim=8, n_layers=2)
    with pytest.raises(ValueError, match="no store attached"):
        fm_no_store.load_category_residuals("ds1", "cluster_0")

    with pytest.raises(KeyError, match="not found"):
        loaded.load_category_residuals("ds1", "nonexistent_category")


def test_save_residuals_overwrites_same_category(tmp_path):
    """Saving the same (dataset, category) twice overwrites silently."""
    gene_ids = make_gene_ids(5)
    fm = make_foundation_model(n_genes=5, embed_dim=8, n_layers=2, gene_ids=gene_ids)
    store = FoundationModelStore(tmp_path / fm.disk_name)
    fm.save(store)

    embeddings_v1 = _make_layer_grid_embeddings(
        n_layers=2,
        n_categories=1,
        gene_ids=gene_ids,
        embed_dim=8,
        model_name=fm.model_name,
        dataset_name="ds1",
    )
    embeddings_v2 = _make_layer_grid_embeddings(
        n_layers=2,
        n_categories=1,
        gene_ids=gene_ids,
        embed_dim=8,
        model_name=fm.model_name,
        dataset_name="ds1",
        seed=99,
    )

    store.save_residuals(embeddings_v1)
    store.save_residuals(embeddings_v2)

    # Second save overwrites — only one entry in index
    assert store.list_categories("ds1") == ["cluster_0"]

    # Values reflect second save
    stem = store.get_stem("ds1", "cluster_0")
    arrays, _ = store.load_residual_arrays(stem)
    expected = {
        ge.layer_idx: ge.embedding for ge in embeddings_v2 if ge.category == "cluster_0"
    }
    for layer_key, arr in arrays.items():
        layer_idx = int(layer_key.split("_")[1])
        np.testing.assert_array_equal(arr, expected[layer_idx])


def test_save_residuals_stem_collision_raises(tmp_path):
    """Categories that sanitize to the same stem raise ValueError."""
    gene_ids = make_gene_ids(5)
    annotations = make_gene_annotations(gene_ids)
    fm = make_foundation_model(n_genes=5, embed_dim=8, n_layers=1, gene_ids=gene_ids)
    store = FoundationModelStore(tmp_path / fm.disk_name)
    fm.save(store)

    # "cluster (0)" and "cluster_0" both sanitize to "cluster_0"
    rng = np.random.default_rng(0)
    embeddings = [
        GeneEmbeddings(
            embedding=rng.standard_normal((5, 8)).astype(np.float32),
            ordered_gene_ids=gene_ids,
            gene_annotations=annotations,
            model_name=fm.model_name,
            layer_idx=0,
            dataset_name="ds1",
            category="cluster (0)",
        ),
        GeneEmbeddings(
            embedding=rng.standard_normal((5, 8)).astype(np.float32),
            ordered_gene_ids=gene_ids,
            gene_annotations=annotations,
            model_name=fm.model_name,
            layer_idx=0,
            dataset_name="ds1",
            category="cluster_0",
        ),
    ]

    with pytest.raises(ValueError, match="sanitize to stem"):
        store.save_residuals(embeddings)


def test_save_residuals_missing_dataset_name_raises(tmp_path):
    gene_ids = make_gene_ids(5)
    annotations = make_gene_annotations(gene_ids)
    fm = make_foundation_model(n_genes=5, embed_dim=8, n_layers=1, gene_ids=gene_ids)
    store = FoundationModelStore(tmp_path / fm.disk_name)
    fm.save(store)

    bad = GeneEmbeddings(
        embedding=np.zeros((5, 8), dtype=np.float32),
        ordered_gene_ids=gene_ids,
        gene_annotations=annotations,
        dataset_name=None,
        category="cluster_0",
        layer_idx=0,
    )
    with pytest.raises(ValueError, match="no dataset_name"):
        store.save_residuals([bad])


def test_save_residuals_missing_category_raises(tmp_path):
    gene_ids = make_gene_ids(5)
    annotations = make_gene_annotations(gene_ids)
    fm = make_foundation_model(n_genes=5, embed_dim=8, n_layers=1, gene_ids=gene_ids)
    store = FoundationModelStore(tmp_path / fm.disk_name)
    fm.save(store)

    bad = GeneEmbeddings(
        embedding=np.zeros((5, 8), dtype=np.float32),
        ordered_gene_ids=gene_ids,
        gene_annotations=annotations,
        dataset_name="ds1",
        category=None,
        layer_idx=0,
    )
    with pytest.raises(ValueError, match="no category"):
        store.save_residuals([bad])


def test_save_residuals_missing_layer_idx_raises(tmp_path):
    gene_ids = make_gene_ids(5)
    annotations = make_gene_annotations(gene_ids)
    fm = make_foundation_model(n_genes=5, embed_dim=8, n_layers=1, gene_ids=gene_ids)
    store = FoundationModelStore(tmp_path / fm.disk_name)
    fm.save(store)

    bad = GeneEmbeddings(
        embedding=np.zeros((5, 8), dtype=np.float32),
        ordered_gene_ids=gene_ids,
        gene_annotations=annotations,
        dataset_name="ds1",
        category="cluster_0",
        layer_idx=None,
    )
    with pytest.raises(ValueError, match="no layer_idx"):
        store.save_residuals([bad])
