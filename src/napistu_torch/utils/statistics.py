"""
Utilities for calculating statistics.

Public Functions
----------------
calculate_rank_shift(df, n_genes, grouping_vars="layer", alternative="greater", test_method="wilcoxon")
    Calculate rank percentile shift for a given DataFrame.
compare_top_k_union_ranks(top_k_union, grouping_vars, defining_vars, top_k, max_rank, rank_col, test_method="wilcoxon", alternative="two-sided")
    Calculate the rank agreement between the top-k attention pairs of a given partition and all other partitions.
"""

from typing import List, Optional, Union

import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon

from napistu_torch.utils.constants import (
    RANK_SHIFT_SUMMARIES,
    RANK_SHIFT_TESTS,
    STATISTICAL_TESTS,
)


def calculate_rank_shift(
    df: pd.DataFrame,
    rank_col: str,
    max_rank: int,
    grouping_vars: Optional[Union[str, List[str]]] = None,
    alternative: str = "two-sided",
    test_method: str = STATISTICAL_TESTS.WILCOXON_RANKSUM,
) -> pd.DataFrame:
    """
    Calculate rank shift statistics for a given DataFrame.

    For each group (e.g., layer), calculates the rank shift statistics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing rank column and grouping variable.
    rank_col : str
        Column name containing ranks.
    max_rank : int
        Maximum possible rank.
    grouping_var : str, optional
        Column name(s) to group by when calculating ranks. If None, ranks globally.
        If a single string, ranks within each value of that column.
        If a list of strings, ranks within each combination of those columns.
        Example: ['model'] or ['model', 'layer'] (default: None)
        Tests are performed separately for each unique value.
    alternative : str, optional
        Alternative hypothesis for the test (default: "greater").
        - "greater": selected queries have higher quantiles than 0.5
        - "less": selected queries have lower quantiles than 0.5
        - "two-sided": selected queries differ from 0.5
    test_method : str, optional
        Statistical test to use (default: "wilcoxon").
        - "wilcoxon": Wilcoxon signed-rank test (non-parametric)
        - "ttest": One-sample t-test (parametric)

    Returns
    -------
    pd.DataFrame
        Results with columns:
        - <grouping_var> : Grouping variable value
        - n_queries : Number of queries in this group
        - mean_quantile : Mean quantile (0.5 = null expectation)
        - median_quantile : Median quantile
        - statistic : Test statistic
        - p_value : P-value for the test

    Examples
    --------
    >>> # Test if top-k queries occupy high quantiles within each layer
    >>> results = calculate_rank_shift(
    ...     df=top_k_union,
    ...     rank_col="rank",
    ...     max_rank=len(common_ids)**2,
    ...     grouping_var="layer"
    ... )
    >>> # Significant layers have p_value < 0.05
    >>> significant = results[results['p_value'] < 0.05]

    >>> # Compare across models instead of layers
    >>> results = calculate_rank_shift(
    ...     multi_model_edges,
    ...     n_genes=len(common_ids),
    ...     grouping_var="model"
    ... )
    """

    # Validate inputs
    if rank_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{rank_col}' column")
    if any(df[rank_col] > max_rank):
        raise ValueError(f"Rank values must be less than or equal to {max_rank}")
    if any(df[rank_col] < 1):
        raise ValueError("Rank values must be greater than or equal to 1")
    if grouping_vars is not None:
        if isinstance(grouping_vars, str):
            grouping_vars = [grouping_vars]
        for col in grouping_vars:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")
    if test_method not in [
        STATISTICAL_TESTS.WILCOXON_RANKSUM,
        STATISTICAL_TESTS.ONE_SAMPLE_TTEST,
    ]:
        raise ValueError(f"test_method must be one of {RANK_SHIFT_TESTS}")

    if grouping_vars is None:
        df = df.copy()
        df["_dummy_group"] = 0
        grouping_vars = ["_dummy_group"]

    results = (
        df.groupby(grouping_vars, group_keys=True)
        .apply(
            _compute_rank_shift_for_group,
            rank_col=rank_col,
            max_rank=max_rank,
            alternative=alternative,
            test_method=test_method,
            include_groups=False,  # Don't duplicate index as columns
        )
        .reset_index()
    )

    # Drop dummy column if we added it
    if "_dummy_group" in results.columns:
        results = results.drop(columns=["_dummy_group"])

    return results


def compare_top_k_union_ranks(
    top_k_union: pd.DataFrame,
    grouping_vars: list,
    defining_vars: list,
    top_k: int,
    max_rank: int,
    rank_col: str,
    test_method: str = STATISTICAL_TESTS.WILCOXON_RANKSUM,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """
    Calculate the rank agreement between the top-k attention pairs of a given partition and all other partitions.

    Parameters
    ----------
    top_k_union : pd.DataFrame
        The top-k attention pairs of a given partition.
    grouping_vars : list
        The variables to group by.
    defining_vars : list
        The variables to define the top-k attention pairs.
    top_k : int
        The number of top-k attention pairs considered (this should match the value used to create top_k_union)
    max_rank : int
        The maximum rank considered.
    rank_col : str
        The column name of the ranks.
    test_method : str, optional
        Statistical test to use (default: "wilcoxon").
        - "wilcoxon": Wilcoxon signed-rank test (non-parametric)
        - "ttest": One-sample t-test (parametric)
    alternative : str, optional
        Alternative hypothesis for the test (default: "two-sided").
        - "greater": selected queries have higher quantiles than 0.5
        - "less": selected queries have lower quantiles than 0.5
        - "two-sided": selected queries differ from 0.5

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the rank agreement between the top-k attention pairs of a given partition and all other partitions.
    """
    partitions = top_k_union[grouping_vars].drop_duplicates().reset_index(drop=True)
    eval_renaming_map = {var: f"eval_{var}" for var in grouping_vars}
    query_cols = [f"query_{var}" for var in grouping_vars]

    results = []
    for i in range(partitions.shape[0]):
        topk_partition = partitions.iloc[[i]]  # Keep as DataFrame

        topk_partition_df = top_k_union.merge(
            topk_partition, on=grouping_vars, how="inner"
        ).query(f"{rank_col} <= @top_k")

        # Get all OTHER partitions filtered to the same gene pairs
        other_partitions = partitions.drop(i).reset_index(drop=True)

        other_partitions_df = top_k_union.merge(
            other_partitions, on=grouping_vars, how="inner"
        ).merge(topk_partition_df[defining_vars], on=defining_vars, how="inner")

        result = (
            calculate_rank_shift(
                other_partitions_df,
                rank_col=rank_col,
                alternative=alternative,
                test_method=test_method,
                max_rank=max_rank,
                grouping_vars=grouping_vars,
            )
            .rename(columns=eval_renaming_map)
            .assign(
                **{f"query_{var}": topk_partition[var].iloc[0] for var in grouping_vars}
            )
            .pipe(
                lambda df: df[
                    query_cols + [col for col in df.columns if col not in query_cols]
                ]
            )
        )

        results.append(result)

    return pd.concat(results, ignore_index=True)


def _compute_rank_shift_for_group(
    group_df: pd.DataFrame,
    rank_col: str,
    max_rank: int,
    alternative: str,
    test_method: str,
) -> pd.Series:
    """Compute rank shift statistics for a single group."""
    # Convert ranks to quantiles (0-1 scale)
    quantiles = group_df[rank_col] / max_rank

    # Perform test (null: quantiles centered at 0.5)
    if test_method == STATISTICAL_TESTS.WILCOXON_RANKSUM:
        stat, p_value = wilcoxon(quantiles - 0.5, alternative=alternative)
    elif test_method == STATISTICAL_TESTS.ONE_SAMPLE_TTEST:
        stat, p_value = ttest_1samp(quantiles, 0.5, alternative=alternative)
    else:
        raise ValueError(f"Invalid test method: {test_method}")

    return pd.Series(
        {
            RANK_SHIFT_SUMMARIES.MEAN_QUANTILE: quantiles.mean(),
            RANK_SHIFT_SUMMARIES.MEDIAN_QUANTILE: quantiles.median(),
            RANK_SHIFT_SUMMARIES.MIN_QUANTILE: quantiles.min(),
            RANK_SHIFT_SUMMARIES.MAX_QUANTILE: quantiles.max(),
            RANK_SHIFT_SUMMARIES.STATISTIC: stat,
            RANK_SHIFT_SUMMARIES.P_VALUE: p_value,
        }
    )
