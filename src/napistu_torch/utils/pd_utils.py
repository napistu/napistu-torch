"""
Utilities for Pandas operations.

Public Functions
----------------
calculate_ranks(df, value_col, by_absolute_value=True, grouping_vars=None)
    Compute integer ranks for values in a DataFrame, ranking within groups.
reorder_multindex_by_categorical_and_numeric(multindex, categorical_order, categorical_level, numeric_level)
    Reorder MultiIndex by categorical order (from reference) then by numeric value.

"""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


def reorder_multindex_by_categorical_and_numeric(
    multindex: pd.MultiIndex,
    categorical_order: List,
    categorical_level: int = 0,
    numeric_level: int = 1,
) -> pd.MultiIndex:
    """
    Reorder MultiIndex by categorical order (from reference) then by numeric value.

    This function takes a MultiIndex and reorders it to match a desired categorical
    ordering, then sorts by numeric values within each categorical group.

    Parameters
    ----------
    multindex : pd.MultiIndex
        MultiIndex to reorder
    categorical_order : List
        Desired order for categorical values. All values in this list must be present
        in the MultiIndex at categorical_level. If some are missing, a warning is logged.
        If extra values are present in the MultiIndex that aren't in categorical_order,
        a ValueError is raised.
    categorical_level : int, optional
        Level index for categorical variable in MultiIndex (default: 0)
    numeric_level : int, optional
        Level index for numeric variable in MultiIndex (default: 1)

    Returns
    -------
    pd.MultiIndex
        Reordered MultiIndex

    Raises
    ------
    ValueError
        If the MultiIndex contains categorical values not in categorical_order

    Examples
    --------
    >>> import pandas as pd
    >>> # MultiIndex to reorder
    >>> idx = pd.MultiIndex.from_tuples([
    ...     ('model_B', 2), ('model_A', 1), ('model_A', 0), ('model_B', 0)
    ... ], names=['model', 'layer'])
    >>> # Desired categorical order
    >>> categorical_order = ['model_A', 'model_B']
    >>> # Reorder
    >>> idx_reordered = reorder_multindex_by_categorical_and_numeric(
    ...     idx, categorical_order, categorical_level=0, numeric_level=1
    ... )
    >>> # Result: ('model_A', 0), ('model_A', 1), ('model_B', 0), ('model_B', 2)
    """
    # Get actual categorical values from MultiIndex
    actual_categorical = multindex.get_level_values(categorical_level).unique().tolist()

    # Check for missing values (warn)
    missing = set(categorical_order) - set(actual_categorical)
    if missing:
        logger.warning(
            f"MultiIndex is missing expected categorical values at level {categorical_level}: {missing}"
        )

    # Check for extra values (error)
    extra = set(actual_categorical) - set(categorical_order)
    if extra:
        raise ValueError(
            f"MultiIndex contains unexpected categorical values at level {categorical_level} "
            f"that are not in categorical_order: {extra}. "
            f"Expected values: {categorical_order}"
        )

    # Create mapping for categorical order
    categorical_order_map = {
        cat: idx
        for idx, cat in enumerate(categorical_order)
        if cat in actual_categorical
    }

    # Sort tuples: first by categorical order, then by numeric value
    tuples = multindex.tolist()
    tuples_sorted = sorted(
        tuples,
        key=lambda x: (
            categorical_order_map.get(x[categorical_level], len(categorical_order)),
            x[numeric_level],
        ),
    )

    return pd.MultiIndex.from_tuples(tuples_sorted, names=multindex.names)
