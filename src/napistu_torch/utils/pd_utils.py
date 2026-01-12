"""
Utilities for Pandas operations.

Public Functions
----------------
calculate_ranks(df, value_col, by_absolute_value=True, grouping_vars=None)
    Compute integer ranks for values in a DataFrame, ranking within groups.

"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd


def calculate_ranks(
    df: pd.DataFrame,
    value_col: str,
    by_absolute_value: bool = True,
    grouping_vars: Optional[Union[str, List[str]]] = None,
) -> pd.Series:
    """
    Compute integer ranks for values in a DataFrame, ranking within groups.

    Since all entries are already top N, ranks them directly based on values
    within each group. Rank 1 = highest value, rank 2 = second highest, etc.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing values to rank
    value_col : str
        Name of the column containing values to rank
    by_absolute_value : bool, optional
        If True, rank by absolute value (default: True).
        If False, rank by raw value.
    grouping_vars : str or List[str], optional
        Column name(s) to group by when calculating ranks. If None, ranks globally.
        If a single string, ranks within each value of that column.
        If a list of strings, ranks within each combination of those columns.
        Example: ['model'] or ['model', 'layer'] (default: None)

    Returns
    -------
    pd.Series
        Series of integer ranks with same index as df.
        Rank 1 = highest value, rank 2 = second highest, etc.
        Ranks are calculated within each group if grouping_vars is provided.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'model': ['A', 'A', 'B', 'B'],
    ...     'attention': [0.9, 0.8, 0.7, 0.6]
    ... })
    >>> ranks = calculate_ranks(df, 'attention', grouping_vars='model')
    >>> # Ranks within each model: A gets [1, 2], B gets [1, 2]
    """
    if by_absolute_value:
        values_to_rank = df[value_col].abs()
    else:
        values_to_rank = df[value_col]

    # Rank in descending order (highest = rank 1)
    # Use method='first' to handle ties deterministically
    if grouping_vars is not None:
        if isinstance(grouping_vars, str):
            grouping_vars = [grouping_vars]
        # Convert column names to Series for groupby
        groupby_series = [df[col] for col in grouping_vars]
        ranks = (
            values_to_rank.groupby(groupby_series)
            .rank(method="first", ascending=False)
            .astype(np.int64)
        )
    else:
        ranks = values_to_rank.rank(method="first", ascending=False).astype(np.int64)

    return ranks
