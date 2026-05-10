"""Tests for ``napistu_torch.foundation_models.foundation_models``."""

import json

import numpy as np
import pytest
import yaml
from foundation_model_factories import (
    _clone_fm_with_dge,
    _make_all_layer_idx_none_dge,
    _make_layer_grid_dge,
    make_foundation_model,
    make_gene_ids,
)

from napistu_torch.foundation_models.constants import FM_DEFS
from napistu_torch.foundation_models.foundation_models import FoundationModelStore


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
    assert FM_DEFS.DATASET_GENE_EMBEDDINGS not in meta


def test_foundation_model_save_category_residuals(tmp_path):
    """save_category_residuals() writes arrays, updates index, raises on bad inputs."""
    gene_ids = make_gene_ids(10)
    fm_base = make_foundation_model(
        n_genes=10, embed_dim=8, n_layers=2, gene_ids=gene_ids
    )
    dge = _make_layer_grid_dge(
        n_layers=2,
        n_categories=3,
        gene_ids=gene_ids,
        embed_dim=8,
        model_name=fm_base.model_name,
    )
    fm = _clone_fm_with_dge(fm_base, dge)

    store = FoundationModelStore(tmp_path / fm.disk_name)
    fm.save(store)

    # Save all three categories
    for i in range(3):
        fm.save_category_residuals(store, "ds1", f"cluster_{i}")

    # Index reflects all categories
    assert set(store.list_categories("ds1")) == {"cluster_0", "cluster_1", "cluster_2"}

    # Arrays have correct shape for one category
    stem = store.get_stem("ds1", "cluster_0")
    arrays, metadata_records = store.load_residual_arrays(stem)
    assert set(arrays.keys()) == {"layer_0", "layer_1"}
    assert all(arr.shape == (10, 8) for arr in arrays.values())
    assert len(metadata_records) == 2

    # Error cases
    fm_no_dge = make_foundation_model(n_genes=10, embed_dim=8, n_layers=2)
    with pytest.raises(ValueError, match="no dataset_gene_embeddings"):
        fm_no_dge.save_category_residuals(store, "ds1", "cluster_0")

    with pytest.raises(ValueError, match="not found in model"):
        fm.save_category_residuals(store, "nonexistent_dataset", "cluster_0")

    with pytest.raises(ValueError, match="not found in dataset"):
        fm.save_category_residuals(store, "ds1", "nonexistent_category")


def test_store_lifecycle_and_index(tmp_path):
    """Initialize, register categories across two datasets, verify persistence."""
    store = FoundationModelStore(tmp_path / "scGPT")
    assert not store.is_initialized()

    store.initialize()
    assert store.is_initialized()
    assert store.residuals_dir.exists()

    # Empty index
    assert store.list_datasets() == []
    assert not store.has_category("ds1", "cluster_0")

    # Register across two datasets
    store.register_category("ds1", "adipocyte (0)", "ds1_adipocyte_0")
    store.register_category("ds1", "T cell", "ds1_T_cell")
    store.register_category("ds2", "cluster_0", "ds2_cluster_0")

    assert set(store.list_datasets()) == {"ds1", "ds2"}
    assert set(store.list_categories("ds1")) == {"adipocyte (0)", "T cell"}
    assert store.get_stem("ds1", "adipocyte (0)") == "ds1_adipocyte_0"
    assert store.has_category("ds2", "cluster_0")

    # Category names with special characters preserved exactly in yaml
    with open(store.index_path) as f:
        raw = yaml.safe_load(f)
    assert raw["datasets"]["ds1"]["adipocyte (0)"] == "ds1_adipocyte_0"

    # Reload from disk — verify persistence
    store2 = FoundationModelStore(tmp_path / "scGPT")
    assert set(store2.list_datasets()) == {"ds1", "ds2"}
    assert store2.get_stem("ds1", "T cell") == "ds1_T_cell"


def test_store_residual_array_roundtrip(tmp_path):
    """save_residual_arrays / load_residual_arrays roundtrip and error cases."""
    store = FoundationModelStore(tmp_path / "scGPT")
    store.initialize()

    arrays = {
        "layer_0": np.random.randn(10, 8).astype(np.float32),
        "layer_1": np.random.randn(10, 8).astype(np.float32),
    }
    metadata_records = [{"layer_idx": 0}, {"layer_idx": 1}]

    store.save_residual_arrays("ds1_cluster_0", arrays, metadata_records)
    loaded_arrays, loaded_meta = store.load_residual_arrays("ds1_cluster_0")

    assert set(loaded_arrays.keys()) == {"layer_0", "layer_1"}
    np.testing.assert_array_equal(loaded_arrays["layer_0"], arrays["layer_0"])
    np.testing.assert_array_equal(loaded_arrays["layer_1"], arrays["layer_1"])
    assert loaded_meta == metadata_records

    # Missing npz
    with pytest.raises(FileNotFoundError, match="Residual arrays not found"):
        store.load_residual_arrays("nonexistent_stem")

    # npz present but sidecar missing
    np.savez(store.residuals_path("orphan"), layer_0=np.zeros((5, 4)))
    with pytest.raises(FileNotFoundError, match="sidecar is missing"):
        store.load_residual_arrays("orphan")
