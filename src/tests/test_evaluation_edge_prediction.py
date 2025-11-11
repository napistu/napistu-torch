import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal

from napistu_torch.evaluation.constants import EDGE_PREDICTION_BY_STRATA_DEFS
from napistu_torch.evaluation.edge_prediction import (
    summarize_edge_predictions_by_strata,
)
from napistu_torch.load.constants import STRATIFICATION_DEFS


@pytest.fixture
def edge_strata_df():
    return pd.DataFrame(
        {
            STRATIFICATION_DEFS.EDGE_STRATA: [
                "A -> X",
                "A -> X",
                "A -> Y",
                "B -> X",
                "B -> Y",
            ]
        }
    )


@pytest.fixture
def edge_predictions():
    return [torch.tensor([0.9, 0.8, 0.4, 0.3, 0.2], dtype=torch.float32)]


def test_summarize_edge_predictions_by_strata(edge_predictions, edge_strata_df):
    summary = summarize_edge_predictions_by_strata(edge_predictions, edge_strata_df)

    expected = pd.DataFrame(
        {
            STRATIFICATION_DEFS.EDGE_STRATA: [
                "A -> X",
                "A -> Y",
                "B -> X",
                "B -> Y",
            ],
            EDGE_PREDICTION_BY_STRATA_DEFS.OBSERVED_OVER_EXPECTED: [
                1.111111,
                0.833333,
                0.833333,
                1.25,
            ],
            EDGE_PREDICTION_BY_STRATA_DEFS.AVERAGE_PREDICTION_PROBABILITY: [
                0.85,
                0.40,
                0.30,
                0.20,
            ],
            EDGE_PREDICTION_BY_STRATA_DEFS.COUNT: [2, 1, 1, 1],
        }
    )

    assert_frame_equal(
        summary.reset_index(drop=True),
        expected,
        check_exact=False,
        check_dtype=False,
        atol=1e-6,
    )
