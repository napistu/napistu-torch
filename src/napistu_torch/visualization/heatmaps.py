"""
Hierarchical clustering and heatmap visualization functions.

Public Functions
----------------
hierarchical_cluster(data, axis, method, metric)
    Perform hierarchical clustering and return reordered indices and labels.
plot_heatmap(data, row_labels, column_labels, title, xlabel, ylabel, figsize, cmap, fmt, vmin, vmax, center, cbar_label, cbar, mask, mask_upper_triangle, square, annot, cluster, cluster_method, cluster_metric, tick_label_size, axis_label_size, title_size, annot_size, ax)
    Plot a heatmap with flexible labeling, masking, and clustering options.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

logger = logging.getLogger(__name__)


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
    data: np.ndarray | pd.DataFrame,
    row_labels: list | None = None,
    column_labels: list | None = None,
    title: str | None = None,
    suptitle: str | None = None,
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
    mask: np.ndarray | None = None,
    mask_upper_triangle: bool = False,
    mask_color: str | None = None,
    square: bool = False,
    annot: bool = True,
    cluster: str = HEATMAP_AXIS.NONE,
    cluster_method: str = CLUSTERING_LINKS.AVERAGE,
    cluster_metric: str = CLUSTERING_DISTANCE_METRICS.EUCLIDEAN,
    tick_label_size: float | None = None,
    axis_title_size: float | None = None,
    title_size: float = 15,
    title_fontweight: str | int = "bold",
    title_fontstyle: str = "normal",
    suptitle_size: float = 16,
    suptitle_fontweight: str | int = "bold",
    suptitle_fontstyle: str = "normal",
    annot_size: float | None = None,
    ax=None,
):
    """
    Plot a heatmap with flexible labeling, masking, and clustering options.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        2D array or DataFrame to plot. If DataFrame, row_labels and column_labels
        are extracted from index and columns if not provided.
    row_labels : list, optional
        Labels for rows (y-axis). Required if data is np.ndarray.
        If data is pd.DataFrame and row_labels is None, uses DataFrame index.
    column_labels : list, optional
        Labels for columns (x-axis). If None and data is pd.DataFrame, uses
        DataFrame columns. If None and data is np.ndarray (and square), uses row_labels.
    title : str, optional
        Plot title
    suptitle : str, optional
        Plot suptitle. Only used if ax is None.
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
    mask : np.ndarray, optional
        Boolean array of same shape as data. True values will be masked.
        If both mask and mask_upper_triangle are provided, they will be combined (OR operation).
    mask_upper_triangle : bool
        If True, mask upper triangle (for symmetric matrices)
    mask_color : str | None
        Background color for masked cells (sets ax facecolor)
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
    tick_label_size : float, optional
        Font size for tick labels (xticklabels, yticklabels)
    axis_title_size : float, optional
        Font size for axis labels (xlabel, ylabel)
    title_size : float
        Font size for plot title. Default is 15.
    title_fontweight : str | int
        Font weight for plot title. Can be numeric (100-1000) or string:
        'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman',
        'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'.
        Default is 'bold'.
    title_fontstyle : str
        Font style for plot title. Options: 'normal', 'italic', 'oblique'.
        Default is 'normal'.
    suptitle_size : float
        Font size for plot suptitle. Default is 16.
    suptitle_fontweight : str | int
        Font weight for plot suptitle. Can be numeric (100-1000) or string:
        'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman',
        'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'.
        Default is 'bold'.
    suptitle_fontstyle : str
        Font style for plot suptitle. Options: 'normal', 'italic', 'oblique'.
        Default is 'normal'.
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

    # Handle DataFrame input: convert to array and extract labels if needed
    data, row_labels_list, column_labels_list = _prepare_heatmap_data(
        data, row_labels, column_labels
    )

    # Reorder data and labels based on clustering
    data_plot, row_labels_plot, column_labels_plot, mask_plot = _reorder_for_clustering(
        data=data,
        row_labels=row_labels_list,
        column_labels=column_labels_list,
        mask=mask,
        cluster=cluster,
        cluster_method=cluster_method,
        cluster_metric=cluster_metric,
    )

    # Apply upper triangle masking if requested (combine with existing mask)
    if mask_upper_triangle:
        if data_plot.shape[0] != data_plot.shape[1]:
            raise ValueError(
                f"mask_upper_triangle requires a square matrix. "
                f"Got shape {data_plot.shape} (rows={data_plot.shape[0]}, cols={data_plot.shape[1]})."
            )
        upper_triangle_mask = np.triu(np.ones_like(data_plot, dtype=bool), k=1)
        mask_plot = (
            mask_plot | upper_triangle_mask
            if mask_plot is not None
            else upper_triangle_mask
        )

    # Track whether we created the figure
    created_fig = ax is None

    # Create figure or use provided axis
    if created_fig:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        fig = ax.get_figure()

    # Build heatmap kwargs
    heatmap_kwargs = _build_heatmap_kwargs(
        annot=annot,
        cmap=cmap,
        fmt=fmt,
        square=square,
        column_labels=column_labels_plot,
        row_labels=row_labels_plot,
        cbar=cbar,
        center=center,
        cbar_label=cbar_label,
        mask=mask_plot,
        vmax=vmax,
        vmin=vmin,
        annot_size=annot_size,
    )

    # Plot heatmap
    sns.heatmap(data_plot, ax=ax, **heatmap_kwargs)

    # Apply aesthetics
    _apply_heatmap_aesthetics(
        ax=ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        tick_label_size=tick_label_size,
        axis_title_size=axis_title_size,
        title_size=title_size,
        title_fontweight=title_fontweight,
        title_fontstyle=title_fontstyle,
        mask_color=mask_color,
    )

    # Apply suptitle if provided (handles tight_layout internally)
    _apply_suptitle(
        fig,
        suptitle,
        suptitle_size,
        suptitle_fontweight,
        suptitle_fontstyle,
        title,
        created_fig,
    )

    # Only call tight_layout if we created the figure AND no suptitle
    # (suptitle function handles layout when present)
    if created_fig and suptitle is None:
        plt.tight_layout()

    return fig


def _apply_heatmap_aesthetics(
    ax,
    xlabel: str | None,
    ylabel: str | None,
    title: str | None,
    tick_label_size: float | None,
    axis_title_size: float | None,
    title_size: float,
    title_fontweight: str | int,
    title_fontstyle: str,
    mask_color: str | None,
):
    """
    Apply aesthetic settings to heatmap axis (labels, title, tick formatting).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to apply aesthetics to
    xlabel : str | None
        X-axis label
    ylabel : str | None
        Y-axis label
    title : str | None
        Plot title
    tick_label_size : float | None
        Font size for tick labels
    axis_title_size : float | None
        Font size for axis labels
    title_size : float
        Font size for title. Default is 15.
    title_fontweight : str | int
        Font weight for title. Can be numeric (100-1000) or string:
        'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman',
        'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'.
        Default is 'bold'.
    title_fontstyle : str
        Font style for title. Options: 'normal', 'italic', 'oblique'.
        Default is 'normal'.
    mask_color : str | None
        Background color for masked cells (sets ax facecolor)
    """
    # Rotate x-axis tick labels to vertical and align properly
    xtick_kwargs = {"rotation": 90, "ha": "center", "va": "top"}
    if tick_label_size is not None:
        xtick_kwargs["fontsize"] = tick_label_size
    ax.set_xticklabels(ax.get_xticklabels(), **xtick_kwargs)

    # Set y-axis tick labels to horizontal
    ytick_kwargs = {"rotation": 0, "ha": "right"}
    if tick_label_size is not None:
        ytick_kwargs["fontsize"] = tick_label_size
    ax.set_yticklabels(ax.get_yticklabels(), **ytick_kwargs)

    # Add labels and title to the axis
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=axis_title_size)
    else:
        ax.set_xlabel("", fontsize=axis_title_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=axis_title_size)
    else:
        ax.set_ylabel("", fontsize=axis_title_size)
    if title:
        title_kwargs = {
            "fontsize": title_size,
            "fontweight": title_fontweight,
            "fontstyle": title_fontstyle,
            "pad": 20,
            "loc": "left",
        }
        ax.set_title(title, **title_kwargs)

    # Set background color for masked cells if specified
    if mask_color is not None:
        ax.set_facecolor(mask_color)


def _apply_suptitle(
    fig,
    suptitle: str | None,
    suptitle_size: float,
    suptitle_fontweight: str | int,
    suptitle_fontstyle: str,
    title: str | None,
    created_fig: bool,
) -> None:
    """
    Apply suptitle to figure if provided.
    """
    if suptitle is not None:
        if created_fig:
            suptitle_kwargs = {
                "fontsize": suptitle_size,
                "fontweight": suptitle_fontweight,
                "fontstyle": suptitle_fontstyle,
            }
            fig.suptitle(suptitle, **suptitle_kwargs)

            # Adjust layout to prevent overlap with suptitle
            # The rect parameter: [left, bottom, right, top]
            # Reserve space at top for suptitle regardless of axis title
            fig.tight_layout(rect=[0, 0, 1, 0.99])
        else:
            logger.warning(
                "Suptitle is not supported when ax is provided. Ignoring suptitle."
            )


def _build_heatmap_kwargs(
    annot: bool,
    cmap: str,
    fmt: str,
    square: bool,
    column_labels: list[str],
    row_labels: list[str],
    cbar: bool,
    center: float | None,
    cbar_label: str | None,
    mask: np.ndarray | None,
    vmax: float | None,
    vmin: float | None,
    annot_size: float | None,
) -> dict:
    """
    Build keyword arguments dictionary for seaborn heatmap.

    Returns
    -------
    dict
        Keyword arguments for sns.heatmap()
    """
    heatmap_kwargs = {
        HEATMAP_KWARGS.ANNOT: annot,
        HEATMAP_KWARGS.CMAP: cmap,
        HEATMAP_KWARGS.FMT: fmt,
        HEATMAP_KWARGS.SQUARE: square,
        HEATMAP_KWARGS.XTICKLABELS: column_labels,
        HEATMAP_KWARGS.YTICKLABELS: row_labels,
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

    return heatmap_kwargs


def _prepare_heatmap_data(
    data: np.ndarray | pd.DataFrame,
    row_labels: list | None,
    column_labels: list | None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Prepare data and labels for heatmap plotting.

    Handles DataFrame input by converting to array and extracting labels.
    Validates that labels are provided for array input.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        2D array or DataFrame to plot
    row_labels : list, optional
        Labels for rows. If None and data is DataFrame, extracted from index.
    column_labels : list, optional
        Labels for columns. If None and data is DataFrame, extracted from columns.
        If None and data is np.ndarray (and square), uses row_labels.

    Returns
    -------
    tuple
        (data_array, row_labels_list, column_labels_list)
        data_array is always np.ndarray
        row_labels_list and column_labels_list are always list[str]

    Raises
    ------
    ValueError
        If row_labels is not provided when data is a numpy array.
    """
    # Handle DataFrame input: convert to array and extract labels if needed
    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        # Extract labels from DataFrame if not provided
        if row_labels is None:
            row_labels = data.index.tolist()
        if column_labels is None:
            column_labels = data.columns.tolist()
        # Convert DataFrame to numpy array
        data = data.values

    # Validate that labels are provided for array input
    if row_labels is None:
        raise ValueError(
            "row_labels must be provided when data is a numpy array. "
            "If using a DataFrame, labels are automatically extracted from index/columns."
        )

    # Convert labels to lists to handle dict_values and other non-list types
    row_labels_list = list[str](row_labels)
    column_labels_list = (
        list[str](column_labels) if column_labels is not None else row_labels_list
    )

    # validate that the row and column labels are the same length as the data
    if len(row_labels_list) != data.shape[0]:
        raise ValueError(
            f"row_labels must be the same length as the number of rows in the data. "
            f"Got {len(row_labels_list)} labels but data has {data.shape[0]} rows."
        )
    if len(column_labels_list) != data.shape[1]:
        raise ValueError(
            f"column_labels must be the same length as the number of columns in the data. "
            f"Got {len(column_labels_list)} labels but data has {data.shape[1]} columns."
        )

    return data, row_labels_list, column_labels_list


def _reorder_for_clustering(
    data: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    mask: np.ndarray | None,
    cluster: str,
    cluster_method: str,
    cluster_metric: str,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray | None]:
    """
    Apply hierarchical clustering and reorder data, labels, and mask.

    Parameters
    ----------
    data : np.ndarray
        2D array to cluster and reorder
    row_labels : list[str]
        Row labels to reorder
    column_labels : list[str]
        Column labels to reorder
    mask : np.ndarray | None
        Optional mask to reorder
    cluster : str
        Clustering axis specification
    cluster_method : str
        Linkage method for clustering
    cluster_metric : str
        Distance metric for clustering

    Returns
    -------
    tuple
        (reordered_data, reordered_row_labels, reordered_column_labels, reordered_mask)
    """
    # Make copies to avoid modifying originals
    data_plot = data.copy()
    row_labels_plot = row_labels.copy()
    column_labels_plot = column_labels.copy()
    mask_plot = mask.copy() if mask is not None else None

    # Perform clustering
    row_order, col_order, _, _ = hierarchical_cluster(
        data_plot, axis=cluster, method=cluster_method, metric=cluster_metric
    )

    # Reorder data, labels, and mask based on clustering
    if row_order is not None:
        data_plot = data_plot[row_order, :]
        row_labels_plot = [row_labels_plot[i] for i in row_order]
        if mask_plot is not None:
            mask_plot = mask_plot[row_order, :]

    if col_order is not None:
        data_plot = data_plot[:, col_order]
        column_labels_plot = [column_labels_plot[i] for i in col_order]
        if mask_plot is not None:
            mask_plot = mask_plot[:, col_order]

    return data_plot, row_labels_plot, column_labels_plot, mask_plot
