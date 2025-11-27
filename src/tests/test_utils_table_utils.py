"""Tests for table formatting utilities."""

import pandas as pd

from napistu_torch.utils.constants import METRIC_VALUE_TABLE
from napistu_torch.utils.table_utils import format_metrics_as_markdown


def test_format_metrics_as_markdown():
    """Test markdown formatting."""
    df = pd.DataFrame(
        {
            METRIC_VALUE_TABLE.METRIC: ["Validation AUC"],
            METRIC_VALUE_TABLE.VALUE: [0.8923],
        }
    )
    result = format_metrics_as_markdown(df)
    assert "| Validation AUC | 0.8923 |" in result
