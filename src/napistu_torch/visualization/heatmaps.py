"""Hierarchical clustering and heatmap visualization functions."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from napistu_torch.visualization.constants import (
    CLUSTERING_DISTANCE_METRICS,
    CLUSTERING_LINKS,
    HEATMAP_AXIS,
    HEATMAP_KWARGS,
    VALID_CLUSTERING_DISTANCE_METRICS,
    VALID_CLUSTERING_LINKS,
    VALID_HEATMAP_AXIS,
)


def hierarchical_cluster(
    data: np.ndarray,
    axis: str = HEATMAP_AXIS.ROWS,
    method: str = CLUSTERING_LINKS.AVERAGE,
    metric: str = CLUSTERING_DISTANCE_METRICS.EUCLIDEAN,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Perform hierarchical clustering and return reordered indices and labels.

    Parameters
    ----------
    data : np.ndarray
        2D array to cluster
    axis : str
        One of {'rows', 'columns', 'both', 'none'}
        - 'rows': cluster rows only
        - 'columns': cluster columns only
        - 'both': cluster both rows and columns
        - 'none': no clustering
    method : str
        Linkage method for scipy.cluster.hierarchy.linkage
        Options: 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'
    metric : str
        Distance metric for scipy.spatial.distance.pdist
        Options: 'euclidean', 'correlation', 'cosine', etc.

    Returns
    -------
    row_order : np.ndarray or None
        Reordered row indices, or None if rows not clustered
    col_order : np.ndarray or None
        Reordered column indices, or None if columns not clustered
    row_linkage : np.ndarray or None
        Linkage matrix for rows, or None if rows not clustered
    col_linkage : np.ndarray or None
        Linkage matrix for columns, or None if columns not clustered
    """
    row_order = None
    col_order = None
    row_linkage = None
    col_linkage = None

    if axis not in VALID_HEATMAP_AXIS:
        raise ValueError(f"Invalid axis: {axis}. Valid axes are: {VALID_HEATMAP_AXIS}")
    if method not in VALID_CLUSTERING_LINKS:
        raise ValueError(
            f"Invalid method: {method}. Valid methods are: {VALID_CLUSTERING_LINKS}"
        )
    if metric not in VALID_CLUSTERING_DISTANCE_METRICS:
        raise ValueError(
            f"Invalid metric: {metric}. Valid metrics are: {VALID_CLUSTERING_DISTANCE_METRICS}"
        )

    if axis == HEATMAP_AXIS.NONE:
        return row_order, col_order, row_linkage, col_linkage

    # Cluster rows
    if axis in [HEATMAP_AXIS.ROWS, HEATMAP_AXIS.BOTH]:
        # Compute pairwise distances between rows
        row_distances = pdist(data, metric=metric)
        row_linkage = linkage(row_distances, method=method)
        row_order = leaves_list(row_linkage)

    # Cluster columns
    if axis in [HEATMAP_AXIS.COLUMNS, HEATMAP_AXIS.BOTH]:
        # Compute pairwise distances between columns (transpose)
        col_distances = pdist(data.T, metric=metric)
        col_linkage = linkage(col_distances, method=method)
        col_order = leaves_list(col_linkage)

    return row_order, col_order, row_linkage, col_linkage


def plot_heatmap(
    data: np.ndarray,
    row_labels: list,
    column_labels: list | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple = (10, 8),
    cmap: str = "Blues",
    fmt: str = ".3f",
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    cbar_label: str | None = None,
    mask_upper_triangle: bool = False,
    square: bool = False,
    annot: bool = True,
    cluster: str = HEATMAP_AXIS.NONE,
    cluster_method: str = CLUSTERING_LINKS.AVERAGE,
    cluster_metric: str = CLUSTERING_DISTANCE_METRICS.EUCLIDEAN,
):
    """
    Plot a heatmap with flexible labeling, masking, and clustering options.

    Parameters
    ----------
    data : np.ndarray
        2D array to plot
    row_labels : list
        Labels for rows (y-axis)
    column_labels : list, optional
        Labels for columns (x-axis). If None, uses row_labels.
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    fmt : str
        Format string for annotations
    vmin : float, optional
        Minimum value for colorbar
    vmax : float, optional
        Maximum value for colorbar
    center : float, optional
        Value to center the colormap at
    cbar_label : str, optional
        Label for colorbar
    mask_upper_triangle : bool
        If True, mask upper triangle (for symmetric matrices)
    square : bool
        If True, force square cells
    annot : bool
        If True, annotate cells with values
    cluster : str
        One of {'rows', 'columns', 'both', 'none'}
        Hierarchical clustering to apply
    cluster_method : str
        Linkage method for clustering ('average', 'complete', 'ward', etc.)
    cluster_metric : str
        Distance metric for clustering ('euclidean', 'correlation', 'cosine', etc.)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Convert labels to lists to handle dict_values and other non-list types
    row_labels_list = list[str](row_labels)

    # Use row_labels for columns if not provided
    if column_labels is None:
        column_labels_list = row_labels_list
    else:
        column_labels_list = list[str](column_labels)

    # Make copies to avoid modifying originals
    data_plot = data.copy()
    row_labels_plot = row_labels_list.copy()
    column_labels_plot = column_labels_list.copy()

    # Perform clustering
    row_order, col_order, _, _ = hierarchical_cluster(
        data_plot, axis=cluster, method=cluster_method, metric=cluster_metric
    )

    # Reorder data and labels based on clustering
    if row_order is not None:
        data_plot = data_plot[row_order, :]
        row_labels_plot = [row_labels_plot[i] for i in row_order]

    if col_order is not None:
        data_plot = data_plot[:, col_order]
        column_labels_plot = [column_labels_plot[i] for i in col_order]

    # Create mask if requested (apply after reordering)
    mask = None
    if mask_upper_triangle:
        mask = np.triu(np.ones_like(data_plot, dtype=bool), k=1)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Build kwargs for heatmap
    heatmap_kwargs = {
        HEATMAP_KWARGS.ANNOT: annot,
        HEATMAP_KWARGS.CMAP: cmap,
        HEATMAP_KWARGS.FMT: fmt,
        HEATMAP_KWARGS.SQUARE: square,
        HEATMAP_KWARGS.XTICKLABELS: column_labels_plot,
        HEATMAP_KWARGS.YTICKLABELS: row_labels_plot,
    }

    # Add optional parameters
    if center is not None:
        heatmap_kwargs[HEATMAP_KWARGS.CENTER] = center
    if cbar_label is not None:
        heatmap_kwargs[HEATMAP_KWARGS.CBAR_KWS] = {"label": cbar_label}
    if mask is not None:
        heatmap_kwargs[HEATMAP_KWARGS.MASK] = mask
    if vmax is not None:
        heatmap_kwargs[HEATMAP_KWARGS.VMAX] = vmax
    if vmin is not None:
        heatmap_kwargs[HEATMAP_KWARGS.VMIN] = vmin

    # Plot heatmap
    sns.heatmap(data_plot, **heatmap_kwargs)

    # Add labels and title
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title, fontsize=15, fontweight="bold", pad=20, loc="left")

    plt.tight_layout()

    return fig
