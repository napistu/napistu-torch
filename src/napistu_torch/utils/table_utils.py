"""Utilities for creating and formatting tables."""

import pandas as pd

from napistu_torch.utils.constants import METRIC_VALUE_TABLE


def format_metrics_as_markdown(df: pd.DataFrame) -> str:
    """
    Format a metrics DataFrame as a markdown table.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['metric', 'value']

    Returns
    -------
    str
        Formatted markdown table

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'metric': ['Validation AUC', 'Test AUC'],
    ...     'value': [0.8923, 0.8856]
    ... })
    >>> print(format_metrics_as_markdown(df))
    | Metric | Value |
    |--------|-------|
    | Validation AUC | 0.8923 |
    | Test AUC | 0.8856 |
    """

    if df.columns.tolist() != [METRIC_VALUE_TABLE.METRIC, METRIC_VALUE_TABLE.VALUE]:
        raise ValueError("DataFrame must have columns ['metric', 'value']")

    # Format values
    def fmt(val):
        if pd.isna(val) or val is None:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    df = df.copy()
    df[METRIC_VALUE_TABLE.VALUE] = df[METRIC_VALUE_TABLE.VALUE].apply(fmt)

    # Build markdown table manually
    lines = ["| Metric | Value |", "|--------|-------|"]
    for _, row in df.iterrows():
        lines.append(
            f"| {row[METRIC_VALUE_TABLE.METRIC]} | {row[METRIC_VALUE_TABLE.VALUE]} |"
        )
    return "\n".join(lines) + "\n"
