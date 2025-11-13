from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.patches import Patch
from torch import Tensor

from napistu_torch.evaluation.constants import EDGE_WEIGHT_SENSITIVITY_DEFS


def compute_edge_feature_sensitivity(
    edge_encoder: nn.Module,
    edge_attr: Tensor,
    max_edges: int,
    device: Optional[Union[str, torch.device]] = None,
) -> Tensor:
    """Compute mean edge-weight gradients with respect to edge features.

    Parameters
    ----------
    edge_encoder : nn.Module
        Trained edge encoder that maps attributes to scalar weights.
    edge_attr : Tensor
        Edge attribute matrix of shape ``[num_edges, edge_dim]``.
    max_edges : int
        Number of edges to sample uniformly at random for the gradient estimate.
    device : Optional[Union[str, torch.device]], optional
        Device used for the computation. Defaults to the encoder's parameter
        device when ``None``.

    Returns
    -------
    Tensor
        One-dimensional tensor containing the mean gradient per feature with
        shape ``[edge_dim]``.

    Raises
    ------
    ValueError
        If ``edge_attr`` is empty or ``max_edges`` is not positive.
    """
    if edge_attr is None or edge_attr.numel() == 0:
        raise ValueError("edge_attr is required to compute gradients.")
    if max_edges <= 0:
        raise ValueError("max_edges must be a positive integer.")

    device = _resolve_device(edge_encoder, device)
    edge_encoder = edge_encoder.to(device)
    edge_encoder.eval()

    edge_attr = edge_attr.to(device)
    total_edges = edge_attr.shape[0]
    if max_edges < total_edges:
        indices = torch.randperm(total_edges, device=device)[:max_edges]
        edge_attr = edge_attr[indices]

    edge_attr = edge_attr.clone().detach().requires_grad_(True)

    # Scalar weights learned for each sampled edge
    outputs = edge_encoder(edge_attr).view(-1)
    # Per-edge gradients of the encoder weights w.r.t. edge attributes
    gradients = torch.autograd.grad(outputs.sum(), edge_attr, retain_graph=False)[0]

    return gradients.detach().cpu().mean(dim=0)


def format_edge_feature_sensitivity(
    edge_feature_sensitivities: Tensor, napistu_data
) -> pd.DataFrame:
    """Format edge feature sensitivities into a DataFrame.

    Parameters
    ----------
    edge_feature_sensitivities : Tensor
        One-dimensional tensor containing the mean gradient per feature with
        shape ``[edge_dim]``.
    napistu_data : NapistuData
        NapistuData object containing edge feature names.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature_name, sensitivity, absolute_sensitivity.
    """
    return (
        pd.DataFrame(
            {
                EDGE_WEIGHT_SENSITIVITY_DEFS.FEATURE_NAME: napistu_data.get_edge_feature_names(),
                EDGE_WEIGHT_SENSITIVITY_DEFS.SENSITIVITY: edge_feature_sensitivities.numpy(),
            }
        )
        .assign(
            **{
                EDGE_WEIGHT_SENSITIVITY_DEFS.ABSOLUTE_SENSITIVITY: lambda x: x[
                    EDGE_WEIGHT_SENSITIVITY_DEFS.SENSITIVITY
                ].abs()
            }
        )
        .sort_values(EDGE_WEIGHT_SENSITIVITY_DEFS.ABSOLUTE_SENSITIVITY, ascending=False)
    )


def plot_edge_feature_sensitivity(
    formatted_feature_sensitivities,
    top_n=20,
    figsize=(16, 8),
    truncate_names=50,
):
    """
    Create a signed waterfall plot for feature sensitivities.

    Parameters:
    -----------
    formatted_feature_sensitivities : pd.DataFrame
        DataFrame with columns: feature_name, sensitivity, absolute_sensitivity
    top_n : int
        Number of top features to display (by absolute sensitivity)
    figsize : tuple
        Figure size (width, height)
    truncate_names : int or None
        Maximum length for feature names. None = no truncation
    """
    # Ensure we have the right columns
    REQUIRED_COLS = [
        EDGE_WEIGHT_SENSITIVITY_DEFS.FEATURE_NAME,
        EDGE_WEIGHT_SENSITIVITY_DEFS.SENSITIVITY,
        EDGE_WEIGHT_SENSITIVITY_DEFS.ABSOLUTE_SENSITIVITY,
    ]
    if not all(col in formatted_feature_sensitivities.columns for col in REQUIRED_COLS):
        raise ValueError(f"DataFrame must contain columns: {REQUIRED_COLS}")

    # Sort by absolute sensitivity and get top N
    df_sorted = formatted_feature_sensitivities.nlargest(
        top_n, EDGE_WEIGHT_SENSITIVITY_DEFS.ABSOLUTE_SENSITIVITY
    ).copy()

    # Sort by sensitivity (signed) for plotting order
    df_sorted = df_sorted.sort_values(
        EDGE_WEIGHT_SENSITIVITY_DEFS.SENSITIVITY, ascending=True
    )

    # Optionally truncate feature names
    if truncate_names:
        df_sorted["display_name"] = df_sorted[
            EDGE_WEIGHT_SENSITIVITY_DEFS.FEATURE_NAME
        ].apply(lambda x: x[:truncate_names] + "..." if len(x) > truncate_names else x)
    else:
        df_sorted["display_name"] = df_sorted[EDGE_WEIGHT_SENSITIVITY_DEFS.FEATURE_NAME]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get values
    sensitivities = df_sorted[EDGE_WEIGHT_SENSITIVITY_DEFS.SENSITIVITY].values
    feature_names = df_sorted["display_name"].values

    # Create color map: positive = blue, negative = red
    colors = ["#d62728" if x < 0 else "#1f77b4" for x in sensitivities]

    # Create bar positions
    y_pos = np.arange(len(feature_names))

    # Create horizontal bars
    bars = ax.barh(
        y_pos,
        sensitivities,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.8,
    )

    # Customize plot
    ax.set_ylabel("Feature", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sensitivity", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Top {top_n} edge features influencing learned edge weights\nBased on mean gradient per feature",
        fontsize=15,
        fontweight="bold",
        pad=20,
        loc="left",
    )

    # Set y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=9)

    # Add vertical line at x=0
    ax.axvline(x=0, color="black", linewidth=1.5, linestyle="-", alpha=0.5)

    # Add value labels on bars
    for _, (bar, val) in enumerate(zip(bars, sensitivities)):
        width = bar.get_width()
        # Position label to the right/left of bar
        offset = max(abs(sensitivities)) * 0.02
        label_x = width + (offset if width > 0 else -offset)
        ha = "left" if width > 0 else "right"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.4f}",
            ha=ha,
            va="center",
            fontsize=7.5,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                alpha=0.7,
            ),
        )

    # Add legend
    legend_elements = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="Positive Sensitivity"),
        Patch(facecolor="#d62728", edgecolor="black", label="Negative Sensitivity"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=11, framealpha=0.9)

    # Grid
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Adjust x-axis limits to accommodate labels
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    ax.set_xlim(x_min - x_range * 0.05, x_max + x_range * 0.05)

    # Tight layout
    plt.tight_layout()

    return fig, ax


# private utils


def _resolve_device(
    edge_encoder: nn.Module,
    device: Optional[Union[str, torch.device]],
) -> torch.device:
    if device is not None:
        return torch.device(device)
    param = next(edge_encoder.parameters(), None)
    if param is None:
        raise ValueError("edge_encoder must have parameters to infer device.")
    return param.device
