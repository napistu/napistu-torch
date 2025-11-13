from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from napistu_torch.evaluation.constants import EDGE_PREDICTION_BY_STRATA_DEFS
from napistu_torch.load.constants import STRATIFICATION_DEFS
from napistu_torch.load.stratification import ensure_strata_series


def summarize_edge_predictions_by_strata(
    edge_predictions: List[torch.Tensor], edge_strata: Union[pd.DataFrame, pd.Series]
):

    edge_strata = ensure_strata_series(edge_strata)

    prediction_by_strata = _get_prediction_by_strata(edge_predictions, edge_strata)
    strata_observed_counts = _get_observed_over_expected_strata(edge_strata)

    species_strata_recovery = strata_observed_counts[
        [
            STRATIFICATION_DEFS.EDGE_STRATA,
            EDGE_PREDICTION_BY_STRATA_DEFS.OBSERVED_OVER_EXPECTED,
            EDGE_PREDICTION_BY_STRATA_DEFS.LOG2_OBSERVED_OVER_EXPECTED,
        ]
    ].merge(
        prediction_by_strata,
        left_on=STRATIFICATION_DEFS.EDGE_STRATA,
        right_index=True,
        how="left",
    )

    return species_strata_recovery


def plot_edge_predictions_by_strata(
    df,
    x_col=EDGE_PREDICTION_BY_STRATA_DEFS.LOG2_OBSERVED_OVER_EXPECTED,
    y_col=EDGE_PREDICTION_BY_STRATA_DEFS.AVERAGE_PREDICTION_PROBABILITY,
    y_lower_col=EDGE_PREDICTION_BY_STRATA_DEFS.PREDICTION_PROBABILITY_Q025,
    y_upper_col=EDGE_PREDICTION_BY_STRATA_DEFS.PREDICTION_PROBABILITY_Q975,
    count_col=EDGE_PREDICTION_BY_STRATA_DEFS.COUNT,
    figsize=(8, 6),
):
    """
    Create a scatter plot showing prediction probabilities vs. enrichment with error bars.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    x_col : str
        Column name for x-axis (log2 observed/expected)
    y_col : str
        Column name for y-axis (average prediction probability)
    y_lower_col : str
        Column name for lower bound of prediction probability (2.5th percentile)
    y_upper_col : str
        Column name for upper bound of prediction probability (97.5th percentile)
    count_col : str
        Column name for point counts (for coloring)
    figsize : tuple
        Figure size (width, height)
    """

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create color map based on log-transformed count values
    colors = np.log10(df[count_col] + 1)

    # Add vertical error bars
    for idx, row in df.iterrows():
        ax.plot(
            [row[x_col], row[x_col]],
            [row[y_lower_col], row[y_upper_col]],
            color="gray",
            alpha=0.5,
            linewidth=1.5,
            zorder=1,
        )

    # Create scatter plot
    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        s=50,
        c=colors,
        cmap="viridis",
        edgecolors="black",
        linewidth=0.5,
        zorder=2,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(
        "log₁₀(count)", rotation=270, labelpad=20, fontsize=11, fontweight="bold"
    )

    # Labels and title
    ax.set_xlabel("Log₂ observed over expected", fontsize=13, fontweight="bold")
    ax.set_ylabel(
        "Prediction probability\nmean, 2.5-97.5% quantiles",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_title(
        "Edge prediction by entity type pair",
        fontsize=15,
        fontweight="bold",
        pad=20,
        loc="left",
    )

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()

    return fig, ax


# utils


def _get_prediction_by_strata(
    edge_predictions: List[torch.Tensor], edge_strata: Union[pd.DataFrame, pd.Series]
):

    edge_strata = ensure_strata_series(edge_strata)

    n_predictions = len(edge_predictions[0])
    n_strata = len(edge_strata)

    if n_predictions != n_strata:
        raise ValueError(
            f"Number of predictions ({n_predictions}) does not match number of strata ({n_strata})"
        )

    tensor_np = edge_predictions[0].cpu().numpy()

    # Create a DataFrame or Series with the tensor values and strata index
    df = pd.DataFrame(
        {"value": tensor_np, STRATIFICATION_DEFS.EDGE_STRATA: edge_strata}
    )

    # Group by strata and calculate summary statistics
    grouped = df.groupby(STRATIFICATION_DEFS.EDGE_STRATA)["value"]
    strata_summary = pd.DataFrame(
        {
            EDGE_PREDICTION_BY_STRATA_DEFS.AVERAGE_PREDICTION_PROBABILITY: grouped.mean(),
            EDGE_PREDICTION_BY_STRATA_DEFS.PREDICTION_PROBABILITY_Q025: grouped.quantile(
                0.025
            ),
            EDGE_PREDICTION_BY_STRATA_DEFS.PREDICTION_PROBABILITY_Q975: grouped.quantile(
                0.975
            ),
        }
    )

    strata_counts = edge_strata.value_counts().to_frame()

    return strata_summary.join(strata_counts).sort_values(
        EDGE_PREDICTION_BY_STRATA_DEFS.AVERAGE_PREDICTION_PROBABILITY, ascending=False
    )


def _get_observed_over_expected_strata(
    edge_strata: Union[pd.DataFrame, pd.Series],
) -> pd.DataFrame:

    edge_strata = ensure_strata_series(edge_strata)

    strata_observed_counts = edge_strata.value_counts().reset_index()
    strata_observed_counts[EDGE_PREDICTION_BY_STRATA_DEFS.FROM_ATTRIBUTE] = (
        strata_observed_counts[STRATIFICATION_DEFS.EDGE_STRATA]
        .str.split(STRATIFICATION_DEFS.FROM_TO_SEPARATOR)
        .str[0]
    )
    strata_observed_counts[EDGE_PREDICTION_BY_STRATA_DEFS.TO_ATTRIBUTE] = (
        strata_observed_counts[STRATIFICATION_DEFS.EDGE_STRATA]
        .str.split(STRATIFICATION_DEFS.FROM_TO_SEPARATOR)
        .str[1]
    )

    strata_observed_counts = strata_observed_counts.merge(
        strata_observed_counts.groupby(EDGE_PREDICTION_BY_STRATA_DEFS.FROM_ATTRIBUTE)[
            EDGE_PREDICTION_BY_STRATA_DEFS.COUNT
        ]
        .sum()
        .rename(EDGE_PREDICTION_BY_STRATA_DEFS.FROM_ATTRIBUTE_COUNT),
        left_on=EDGE_PREDICTION_BY_STRATA_DEFS.FROM_ATTRIBUTE,
        right_index=True,
    ).merge(
        strata_observed_counts.groupby(EDGE_PREDICTION_BY_STRATA_DEFS.TO_ATTRIBUTE)[
            EDGE_PREDICTION_BY_STRATA_DEFS.COUNT
        ]
        .sum()
        .rename(EDGE_PREDICTION_BY_STRATA_DEFS.TO_ATTRIBUTE_COUNT),
        left_on=EDGE_PREDICTION_BY_STRATA_DEFS.TO_ATTRIBUTE,
        right_index=True,
    )

    strata_observed_counts[EDGE_PREDICTION_BY_STRATA_DEFS.EXPECTED_COUNT] = [
        from_attr_count
        * to_attr_count
        / strata_observed_counts[EDGE_PREDICTION_BY_STRATA_DEFS.COUNT].sum()
        for from_attr_count, to_attr_count in zip(
            strata_observed_counts[EDGE_PREDICTION_BY_STRATA_DEFS.FROM_ATTRIBUTE_COUNT],
            strata_observed_counts[EDGE_PREDICTION_BY_STRATA_DEFS.TO_ATTRIBUTE_COUNT],
        )
    ]

    strata_observed_counts[EDGE_PREDICTION_BY_STRATA_DEFS.OBSERVED_OVER_EXPECTED] = (
        strata_observed_counts[EDGE_PREDICTION_BY_STRATA_DEFS.COUNT]
        / strata_observed_counts[EDGE_PREDICTION_BY_STRATA_DEFS.EXPECTED_COUNT]
    )

    strata_observed_counts[
        EDGE_PREDICTION_BY_STRATA_DEFS.LOG2_OBSERVED_OVER_EXPECTED
    ] = np.log2(
        strata_observed_counts[EDGE_PREDICTION_BY_STRATA_DEFS.OBSERVED_OVER_EXPECTED]
    )

    return strata_observed_counts
