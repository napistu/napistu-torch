from typing import List, Union

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
        ]
    ].merge(
        prediction_by_strata,
        left_on=STRATIFICATION_DEFS.EDGE_STRATA,
        right_index=True,
        how="left",
    )

    return species_strata_recovery


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

    # Group by strata and calculate mean
    strata_means = (
        df.groupby(STRATIFICATION_DEFS.EDGE_STRATA)["value"]
        .mean()
        .rename(EDGE_PREDICTION_BY_STRATA_DEFS.AVERAGE_PREDICTION_PROBABILITY)
    ).to_frame()

    strata_counts = edge_strata.value_counts().to_frame()

    return strata_means.join(strata_counts).sort_values(
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

    return strata_observed_counts
