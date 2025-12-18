"""Advanced metrics like summaries of relation-type-specific AUCs."""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from napistu_torch.utils.labeling_utils import format_metric_label


def plot_combined_grouped_barplot(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 10),
    category: str = "relation_type",
    attribute: str = "experiment",
    value: str = "val_auc",
    value_vars: List[str] = ["val_auc", "test_auc"],
    category_order: Optional[List[str]] = None,
    attribute_order: Optional[List[str]] = None,
    category_label: Optional[str] = None,
    attribute_label: Optional[str] = None,
    value_label: str = "AUC",
    titles: Optional[List[str]] = None,
    value_lim: Optional[Tuple[float, float]] = None,
    legend_loc: str = "lower left",
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot multiple value variables side-by-side with horizontal bars.

    value : str
        Column name for values (x-axis for horizontal)
    value_vars : List[str]
        Column names for values (x-axis for horizontal)
    category_order : Optional[List[str]]
        Custom order for categories
    attribute_order : Optional[List[str]]
        Custom order for attributes (hue)
    category_label : Optional[str]
        Label for category axis (defaults to category column name)
    attribute_label : Optional[str]
        Label for legend (defaults to attribute column name)
    value_label : str
        Label for value axis
    title : Optional[str]
        Plot title
    value_lim : Optional[Tuple[float, float]]
        Limits for value axis (e.g., (0.7, 1.0))
    legend_loc : str
        Legend location

    Returns
    -------
    Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]
        Figure and axes tuple (ax1 for validation AUC, ax2 for test AUC)

    Examples
    --------
    >>> df = pd.DataFrame({
    >>>     'relation_type': ['protein_protein', 'protein_metabolite', 'metabolite_metabolite'],
    >>>     'experiment': ['experiment1', 'experiment1', 'experiment1'],
    >>>     'val_auc': [0.8, 0.7, 0.9],
    >>>     'test_auc': [0.75, 0.65, 0.85]
    >>> })
    >>> plot_combined_grouped_barplot(df, category='relation_type', attribute='experiment', value='val_auc')
    """

    if category_label is None:
        category_label = category.replace("_", " ").title()
    if attribute_label is None:
        attribute_label = attribute.replace("_", " ").title()
    if titles is None:
        titles_list = [format_metric_label(metric) for metric in value_vars]
    titles_dict = dict(zip(value_vars, titles_list))

    if value_lim is None:
        all_mins = [df[value].min() for value in value_vars]
        all_maxs = [df[value].max() for value in value_vars]
        value_lim = (min(all_mins).round(2), max(all_maxs).round(2))

    if category not in df.columns:
        raise ValueError(f"Column {category} not found in dataframe")
    if attribute not in df.columns:
        raise ValueError(f"Column {attribute} not found in dataframe")
    for value in value_vars:
        if value not in df.columns:
            raise ValueError(f"Column {value} not found in dataframe")

    fig, axes = plt.subplots(1, len(value_vars), figsize=figsize)

    for ax, metric in zip(axes, value_vars):
        plot_grouped_barplot(
            df=df,
            category=category,
            value=metric,
            attribute=attribute,
            ax=ax,
            orientation="horizontal",
            value_label=value_label,
            category_label=category_label,
            attribute_label=attribute_label,
            title=titles_dict[metric],
            value_lim=value_lim,
            legend_loc=legend_loc,
            category_order=category_order,
            attribute_order=attribute_order,
        )

    plt.tight_layout()
    return fig, axes


def plot_grouped_barplot(
    df: pd.DataFrame,
    category: str,
    value: str,
    attribute: str,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 10),
    orientation: str = "horizontal",
    value_label: str = "Value",
    category_label: Optional[str] = None,
    attribute_label: Optional[str] = None,
    title: Optional[str] = None,
    value_lim: Optional[Tuple[float, float]] = None,
    legend_loc: str = "lower right",
    category_order: Optional[List[str]] = None,
    attribute_order: Optional[List[str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a grouped barplot with flexible column mapping.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    category : str
        Column name for categories (y-axis for horizontal)
    value : str
        Column name for values (x-axis for horizontal)
    attribute : str
        Column name for grouping/hue
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates new figure
    figsize : tuple
        Figure size if creating new figure
    orientation : str
        'horizontal' or 'vertical'
    value_label : str
        Label for value axis
    category_label : str, optional
        Label for category axis (defaults to category column name)
    attribute_label : str, optional
        Label for legend (defaults to attribute column name)
    title : str, optional
        Plot title
    value_lim : tuple, optional
        Limits for value axis (e.g., (0.7, 1.0))
    legend_loc : str
        Legend location
    category_order : list, optional
        Custom order for categories
    attribute_order : list, optional
        Custom order for attributes (hue)

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axis objects

    Examples
    --------
    >>> df = pd.DataFrame({
    >>>     'relation_type': ['protein_protein', 'protein_metabolite', 'metabolite_metabolite'],
    >>>     'experiment': ['experiment1', 'experiment1', 'experiment1'],
    >>>     'val_auc': [0.8, 0.7, 0.9],
    >>>     'test_auc': [0.75, 0.65, 0.85]
    >>> })
    >>> plot_grouped_bars(df, 'relation_type', 'val_auc', 'experiment', orientation='horizontal')
    """

    # Filter out None values
    plot_df = df[df[value].notna()].copy()

    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Set default labels
    if category_label is None:
        category_label = category.replace("_", " ").title()
    if attribute_label is None:
        attribute_label = attribute.replace("_", " ").title()

    # Create barplot based on orientation
    if orientation == "horizontal":
        sns.barplot(
            data=plot_df,
            y=category,
            x=value,
            hue=attribute,
            ax=ax,
            orient="h",
            order=category_order,
            hue_order=attribute_order,
        )
        ax.set_ylabel(category_label, fontsize=12)
        ax.set_xlabel(value_label, fontsize=12)
        if value_lim:
            ax.set_xlim(value_lim)
    else:  # vertical
        sns.barplot(
            data=plot_df,
            x=category,
            y=value,
            hue=attribute,
            ax=ax,
            order=category_order,
            hue_order=attribute_order,
        )
        ax.set_xlabel(category_label, fontsize=12)
        ax.set_ylabel(value_label, fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        if value_lim:
            ax.set_ylim(value_lim)

    # Set title
    if title:
        ax.set_title(title, fontsize=14)

    # Format legend
    ax.legend(title=attribute_label, loc=legend_loc)

    return fig, ax
