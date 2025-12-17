"""Basic metrics like train loss and test/val AUC."""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from napistu_torch.ml.constants import METRIC_SUMMARIES


def plot_auc_only(
    summaries: Dict[str, Dict[str, Any]],
    display_names: List[str],
    figsize: Tuple[int, int] = (10, 6),
    test_auc_attribute: str = METRIC_SUMMARIES.TEST_AUC,
    val_auc_attribute: str = METRIC_SUMMARIES.VAL_AUC,
    **kwargs,  # Pass through ylim, bar_width, etc.
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot only test/val AUC comparison.

    Parameters
    ----------
    summaries : Dict[str, Dict[str, Any]]
        Dictionary mapping model names to their summary metrics.
        Each summary must contain 'test_auc' and 'val_auc'.
    display_names : List[str]
        Clean display names for models (must match order of summaries.keys())
    figsize : Tuple[int, int]
        Figure size as (width, height)
    test_auc_attribute : str
        Attribute name for test AUC in summaries
    val_auc_attribute : str
        Attribute name for validation AUC in summaries
    **kwargs : dict
        Additional keyword arguments to pass to _plot_test_val_auc

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axis objects

    Examples
    --------
    >>> summaries = {
    ...     'model1': {'test_auc': 0.75, 'val_auc': 0.74},
    ...     'model2': {'test_auc': 0.78, 'val_auc': 0.77}
    ... }
    >>> display_names = ['Model 1', 'Model 2']
    >>> fig, ax = plot_auc_only(summaries, display_names)
    >>> plt.show()
    """
    test_aucs = _extract_metric(summaries, test_auc_attribute)
    val_aucs = _extract_metric(summaries, val_auc_attribute)

    fig, ax = plt.subplots(figsize=figsize)
    _plot_test_val_auc(ax, display_names, test_aucs, val_aucs, **kwargs)
    plt.tight_layout()

    return fig, ax


def plot_model_comparison(
    summaries: Dict[str, Dict[str, Any]],
    display_names: List[str],
    figsize: Tuple[int, int] = (16, 6),
    train_loss_attribute: str = METRIC_SUMMARIES.TRAIN_LOSS,
    test_auc_attribute: str = METRIC_SUMMARIES.TEST_AUC,
    val_auc_attribute: str = METRIC_SUMMARIES.VAL_AUC,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Create comparison plots for model training loss and test/val AUC.

    Parameters
    ----------
    summaries : Dict[str, Dict[str, Any]]
        Dictionary mapping model names to their summary metrics.
        Each summary must contain 'train_loss', 'test_auc', and 'val_auc'.
    display_names : List[str]
        Clean display names for models (must match order of summaries.keys())
    figsize : Tuple[int, int]
        Figure size as (width, height)
    train_loss_attribute : str
        Attribute name for train loss in summaries
    test_auc_attribute : str
        Attribute name for test AUC in summaries
    val_auc_attribute : str
        Attribute name for validation AUC in summaries

    Returns
    -------
    Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]
        Figure and axes tuple (ax1 for train loss, ax2 for AUC)

    Examples
    --------
    >>> summaries = {
    ...     'model1': {'train_loss': 1.5, 'test_auc': 0.75, 'val_auc': 0.74},
    ...     'model2': {'train_loss': 1.2, 'test_auc': 0.78, 'val_auc': 0.77}
    ... }
    >>> display_names = ['Model 1', 'Model 2']
    >>> fig, (ax1, ax2) = plot_model_comparison(summaries, display_names)
    >>> plt.show()
    """

    train_losses = _extract_metric(summaries, train_loss_attribute)
    test_aucs = _extract_metric(summaries, test_auc_attribute)
    val_aucs = _extract_metric(summaries, val_auc_attribute)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Train Loss (with automatic ylim)
    _plot_train_loss(ax1, display_names, train_losses)

    # Plot 2: Test and Val AUC (with automatic ylim)
    _plot_test_val_auc(ax2, display_names, test_aucs, val_aucs)

    plt.tight_layout()

    return fig, (ax1, ax2)


def _plot_train_loss(
    ax: plt.Axes,
    display_names: List[str],
    train_losses: List[float],
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot training loss as a bar chart.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to plot on
    display_names : List[str]
        Model names for x-axis labels
    train_losses : List[float]
        Training loss values for each model
    ylim : Optional[Tuple[float, float]]
        Y-axis limits as (min, max). If None, calculated automatically
    """
    x_pos = np.arange(len(display_names))
    bars = ax.bar(
        x_pos,
        train_losses,
        color="steelblue",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_xlabel("Model", fontsize=13, fontweight="bold")
    ax.set_ylabel("Train Loss", fontsize=13, fontweight="bold")
    ax.set_title("Training Loss by Model", fontsize=15, fontweight="bold", pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Auto-calculate ylim if not provided
    if ylim is None:
        valid_losses = [loss for loss in train_losses if loss is not None]
        if valid_losses:
            min_loss = min(valid_losses)
            max_loss = max(valid_losses)
            loss_range = max_loss - min_loss
            ylim = (min_loss - 0.1 * loss_range, max_loss + 0.1 * loss_range)

    if ylim is not None:
        ax.set_ylim(ylim)

    # Add value labels on bars
    y_offset = 0.002 if ylim else 0.01
    for bar, val in zip(bars, train_losses):
        if val is not None:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + y_offset,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )


def _plot_test_val_auc(
    ax: plt.Axes,
    display_names: List[str],
    test_aucs: List[float],
    val_aucs: List[float],
    ylim: Optional[Tuple[float, float]] = None,
    bar_width: float = 0.35,
) -> None:
    """Plot test and validation AUC as grouped bar chart.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to plot on
    display_names : List[str]
        Model names for x-axis labels
    test_aucs : List[float]
        Test AUC values for each model
    val_aucs : List[float]
        Validation AUC values for each model
    ylim : Optional[Tuple[float, float]]
        Y-axis limits as (min, max). If None, calculated automatically
    bar_width : float
        Width of each bar in the grouped bar chart
    """
    x_pos = np.arange(len(display_names))

    bars_test = ax.bar(
        x_pos - bar_width / 2,
        test_aucs,
        bar_width,
        label="Test AUC",
        color="coral",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    bars_val = ax.bar(
        x_pos + bar_width / 2,
        val_aucs,
        bar_width,
        label="Val AUC",
        color="lightgreen",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_xlabel("Model", fontsize=13, fontweight="bold")
    ax.set_ylabel("AUC Score", fontsize=13, fontweight="bold")
    ax.set_title(
        "Test & Validation AUC by Model", fontsize=15, fontweight="bold", pad=20
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Auto-calculate ylim if not provided
    if ylim is None:
        all_aucs = [auc for auc in test_aucs + val_aucs if auc is not None]
        if all_aucs:
            min_auc = min(all_aucs)
            max_auc = max(all_aucs)
            auc_range = max_auc - min_auc
            ylim = (min_auc - 0.1 * auc_range, max_auc + 0.1 * auc_range)

    if ylim is not None:
        ax.set_ylim(ylim)

    # Add value labels on test bars
    y_offset = 0.001 if ylim else 0.005
    for bar, val in zip(bars_test, test_aucs):
        if val is not None:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + y_offset,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    # Add value labels on validation bars
    for bar, val in zip(bars_val, val_aucs):
        if val is not None:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + y_offset,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )


# private functions


def _extract_metric(
    summaries: Dict[str, Dict[str, Any]], metric_key: str
) -> List[Optional[float]]:
    """Extract a metric from all model summaries."""
    return [summaries[model].get(metric_key) for model in summaries.keys()]
