"""Hierarchical clustering and heatmap visualization functions."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from napistu_torch.utils.optional import import_seaborn, require_seaborn
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


@require_seaborn
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
    cbar: bool = True,
    mask_upper_triangle: bool = False,
    square: bool = False,
    annot: bool = True,
    cluster: str = HEATMAP_AXIS.NONE,
    cluster_method: str = CLUSTERING_LINKS.AVERAGE,
    cluster_metric: str = CLUSTERING_DISTANCE_METRICS.EUCLIDEAN,
    label_size: float | None = None,
    axis_title_size: float | None = None,
    title_size: float | None = None,
    annot_size: float | None = None,
    ax=None,
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
        Figure size (only used if ax is None)
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
    cbar : bool
        If True, show colorbar. If False, hide colorbar.
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
    label_size : float, optional
        Font size for axis labels (xlabel, ylabel)
    axis_title_size : float, optional
        Font size for tick labels (xticklabels, yticklabels)
    title_size : float, optional
        Font size for plot title
    annot_size : float, optional
        Font size for cell annotations
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    sns = import_seaborn()

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

    # Track whether we created the figure
    created_fig = ax is None

    # Create figure or use provided axis
    if created_fig:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        fig = ax.get_figure()

    # Build kwargs for heatmap
    heatmap_kwargs = {
        HEATMAP_KWARGS.ANNOT: annot,
        HEATMAP_KWARGS.CMAP: cmap,
        HEATMAP_KWARGS.FMT: fmt,
        HEATMAP_KWARGS.SQUARE: square,
        HEATMAP_KWARGS.XTICKLABELS: column_labels_plot,
        HEATMAP_KWARGS.YTICKLABELS: row_labels_plot,
        HEATMAP_KWARGS.CBAR: cbar,
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
    if annot_size is not None:
        heatmap_kwargs[HEATMAP_KWARGS.ANNOT_KWS] = {"size": annot_size}

    # Plot heatmap on the specified axis
    sns.heatmap(data_plot, ax=ax, **heatmap_kwargs)

    # Rotate x-axis tick labels to vertical
    xtick_kwargs = {"rotation": 90, "ha": "right"}
    if axis_title_size is not None:
        xtick_kwargs["fontsize"] = axis_title_size
    ax.set_xticklabels(ax.get_xticklabels(), **xtick_kwargs)

    # Set y-axis tick labels to horizontal (explicitly set rotation to 0)
    ytick_kwargs = {"rotation": 0, "ha": "right"}
    if axis_title_size is not None:
        ytick_kwargs["fontsize"] = axis_title_size
    ax.set_yticklabels(ax.get_yticklabels(), **ytick_kwargs)

    # Add labels and title to the axis
    # Explicitly set to empty string if None to suppress default labels
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_size)
    else:
        ax.set_xlabel("", fontsize=label_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_size)
    else:
        ax.set_ylabel("", fontsize=label_size)
    if title:
        title_fontsize = title_size if title_size is not None else 15
        ax.set_title(
            title, fontsize=title_fontsize, fontweight="bold", pad=20, loc="left"
        )

    # Only call tight_layout if we created the figure
    if created_fig:
        plt.tight_layout()

    return fig
