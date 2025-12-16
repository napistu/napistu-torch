import numpy as np
import pytest

from napistu_torch.visualization.constants import (
    CLUSTERING_DISTANCE_METRICS,
    CLUSTERING_LINKS,
    HEATMAP_AXIS,
)
from napistu_torch.visualization.heatmaps import hierarchical_cluster


def test_hierarchical_cluster():
    """Test hierarchical clustering with different axis options."""
    # Create test data with clear structure
    data = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])

    # Test clustering rows only
    row_order, col_order, row_linkage, col_linkage = hierarchical_cluster(
        data, axis=HEATMAP_AXIS.ROWS
    )

    assert row_order is not None
    assert col_order is None
    assert row_linkage is not None
    assert col_linkage is None
    assert isinstance(row_order, np.ndarray)
    assert len(row_order) == data.shape[0]
    assert row_linkage.shape[1] == 4  # Linkage matrix has 4 columns

    # Test clustering columns only
    row_order, col_order, row_linkage, col_linkage = hierarchical_cluster(
        data, axis=HEATMAP_AXIS.COLUMNS
    )

    assert row_order is None
    assert col_order is not None
    assert row_linkage is None
    assert col_linkage is not None
    assert isinstance(col_order, np.ndarray)
    assert len(col_order) == data.shape[1]
    assert col_linkage.shape[1] == 4

    # Test clustering both
    row_order, col_order, row_linkage, col_linkage = hierarchical_cluster(
        data, axis=HEATMAP_AXIS.BOTH
    )

    assert row_order is not None
    assert col_order is not None
    assert row_linkage is not None
    assert col_linkage is not None
    assert len(row_order) == data.shape[0]
    assert len(col_order) == data.shape[1]

    # Test no clustering
    row_order, col_order, row_linkage, col_linkage = hierarchical_cluster(
        data, axis=HEATMAP_AXIS.NONE
    )

    assert row_order is None
    assert col_order is None
    assert row_linkage is None
    assert col_linkage is None


def test_hierarchical_cluster_different_methods():
    """Test hierarchical clustering with different linkage methods."""
    data = np.array([[1.0, 2.0], [2.0, 3.0], [5.0, 6.0]])

    for method in [
        CLUSTERING_LINKS.AVERAGE,
        CLUSTERING_LINKS.COMPLETE,
        CLUSTERING_LINKS.SINGLE,
    ]:
        row_order, _, row_linkage, _ = hierarchical_cluster(
            data, axis=HEATMAP_AXIS.ROWS, method=method
        )

        assert row_order is not None
        assert row_linkage is not None
        assert len(row_order) == data.shape[0]


def test_hierarchical_cluster_different_metrics():
    """Test hierarchical clustering with different distance metrics."""
    data = np.array([[1.0, 2.0], [2.0, 3.0], [5.0, 6.0]])

    for metric in [
        CLUSTERING_DISTANCE_METRICS.EUCLIDEAN,
        CLUSTERING_DISTANCE_METRICS.COSINE,
        CLUSTERING_DISTANCE_METRICS.CORRELATION,
    ]:
        row_order, _, row_linkage, _ = hierarchical_cluster(
            data, axis=HEATMAP_AXIS.ROWS, metric=metric
        )

        assert row_order is not None
        assert row_linkage is not None
        assert len(row_order) == data.shape[0]


def test_hierarchical_cluster_invalid_inputs():
    """Test hierarchical clustering with invalid inputs."""
    data = np.array([[1.0, 2.0], [2.0, 3.0]])

    with pytest.raises(ValueError, match="Invalid axis"):
        hierarchical_cluster(data, axis="invalid_axis")

    with pytest.raises(ValueError, match="Invalid method"):
        hierarchical_cluster(data, method="invalid_method")

    with pytest.raises(ValueError, match="Invalid metric"):
        hierarchical_cluster(data, metric="invalid_metric")
