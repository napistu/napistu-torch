"""Test the ETL functions for capturing residual stream embeddings."""

from dataclasses import dataclass
from typing import List, Union

import numpy as np
from foundation_model_factories import make_gene_ids
from scipy.sparse import csr_matrix

from napistu_torch.foundation_models.etl import _scfoundation_select_cluster_genes


@dataclass
class _FakeAdata:
    """Minimal AnnData duck-type for testing functions that only use .X and .var_names."""

    X: Union[np.ndarray, csr_matrix]
    var_names: List[str]


def _make_cluster_adata(
    expression_matrix: np.ndarray,
    gene_ids: List[str],
) -> _FakeAdata:
    n_cells, n_genes = expression_matrix.shape
    assert len(gene_ids) == n_genes
    return _FakeAdata(
        X=csr_matrix(expression_matrix),
        var_names=gene_ids,
    )


def test_returns_top_n_genes_by_detection():
    """Selected genes are the N most frequently expressed."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 50, 100
    gene_ids = make_gene_ids(n_genes)

    expr = np.zeros((n_cells, n_genes))
    expr[:, :20] = rng.uniform(1, 5, (n_cells, 20))
    expr[:, 20:50] = rng.uniform(0, 5, (n_cells, 30)) * (
        rng.random((n_cells, 30)) > 0.7
    )

    adata = _make_cluster_adata(expr, gene_ids)
    selected, cell_mask = _scfoundation_select_cluster_genes(
        adata, n_genes=20, min_cell_nonzero=0
    )

    assert len(selected) == 20
    assert set(selected) == set(gene_ids[:20])


def test_n_genes_larger_than_available_returns_all():
    """When n_genes exceeds available genes, all genes are returned."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 10, 5
    gene_ids = make_gene_ids(n_genes)
    expr = rng.uniform(1, 5, (n_cells, n_genes))

    adata = _make_cluster_adata(expr, gene_ids)
    selected, cell_mask = _scfoundation_select_cluster_genes(
        adata, n_genes=100, min_cell_nonzero=0
    )

    assert len(selected) == n_genes


def test_cell_mask_filters_low_expression_cells():
    """Cells with fewer than min_cell_nonzero expressed genes are masked out."""
    n_cells, n_genes = 20, 50
    gene_ids = make_gene_ids(n_genes)
    rng = np.random.default_rng(1)

    expr = np.zeros((n_cells, n_genes))
    expr[:10, :30] = rng.uniform(1, 5, (10, 30))
    expr[10:, :5] = rng.uniform(1, 5, (10, 5))

    adata = _make_cluster_adata(expr, gene_ids)
    selected, cell_mask = _scfoundation_select_cluster_genes(
        adata, n_genes=30, min_cell_nonzero=20
    )

    assert cell_mask[:10].all()
    assert not cell_mask[10:].any()


def test_all_zero_expression():
    """All-zero expression — cell mask is all False."""
    n_cells, n_genes = 10, 20
    gene_ids = make_gene_ids(n_genes)
    expr = np.zeros((n_cells, n_genes))

    adata = _make_cluster_adata(expr, gene_ids)
    selected, cell_mask = _scfoundation_select_cluster_genes(
        adata, n_genes=10, min_cell_nonzero=1
    )

    assert len(selected) == 10
    assert not cell_mask.any()


def test_sparse_and_dense_inputs_match():
    """Sparse and dense expression matrices produce identical results."""
    rng = np.random.default_rng(7)
    n_cells, n_genes = 30, 40
    gene_ids = make_gene_ids(n_genes)
    expr = rng.uniform(0, 3, (n_cells, n_genes)) * (
        rng.random((n_cells, n_genes)) > 0.5
    )

    adata_sparse = _make_cluster_adata(expr, gene_ids)
    adata_dense = _FakeAdata(X=expr, var_names=gene_ids)

    selected_sparse, mask_sparse = _scfoundation_select_cluster_genes(
        adata_sparse, n_genes=15, min_cell_nonzero=5
    )
    selected_dense, mask_dense = _scfoundation_select_cluster_genes(
        adata_dense, n_genes=15, min_cell_nonzero=5
    )

    assert selected_sparse == selected_dense
    np.testing.assert_array_equal(mask_sparse, mask_dense)
