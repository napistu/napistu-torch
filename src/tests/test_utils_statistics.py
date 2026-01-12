import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_1samp, wilcoxon

from napistu_torch.utils.constants import (
    RANK_SHIFT_SUMMARIES,
    RANK_SHIFT_TESTS,
    STATISTICAL_TESTS,
)
from napistu_torch.utils.statistics import (
    _compute_rank_shift_for_group,
    calculate_rank_shift,
)


# Tests for _compute_rank_shift_for_group (private function) - <50 lines
def test_compute_rank_shift_for_group_basic():
    """Test _compute_rank_shift_for_group with basic input."""
    group_df = pd.DataFrame({"rank": [10, 20, 30, 40, 50]})
    result = _compute_rank_shift_for_group(
        group_df, "rank", 100, "greater", STATISTICAL_TESTS.WILCOXON_RANKSUM
    )
    assert isinstance(result, pd.Series)
    assert result[RANK_SHIFT_SUMMARIES.MEAN_QUANTILE] == 0.3
    assert result[RANK_SHIFT_SUMMARIES.MEDIAN_QUANTILE] == 0.3
    assert (
        RANK_SHIFT_SUMMARIES.STATISTIC in result
        and RANK_SHIFT_SUMMARIES.P_VALUE in result
    )
    assert 0 <= result[RANK_SHIFT_SUMMARIES.P_VALUE] <= 1


def test_compute_rank_shift_for_group_test_methods():
    """Test _compute_rank_shift_for_group with both test methods."""
    group_df = pd.DataFrame({"rank": [10, 20, 30, 40, 50]})
    quantiles = group_df["rank"] / 100

    # Test Wilcoxon
    result_w = _compute_rank_shift_for_group(
        group_df, "rank", 100, "greater", STATISTICAL_TESTS.WILCOXON_RANKSUM
    )
    expected_stat, expected_p = wilcoxon(quantiles - 0.5, alternative="greater")
    assert np.isclose(result_w[RANK_SHIFT_SUMMARIES.STATISTIC], expected_stat)
    assert np.isclose(result_w[RANK_SHIFT_SUMMARIES.P_VALUE], expected_p)

    # Test t-test
    result_t = _compute_rank_shift_for_group(
        group_df, "rank", 100, "two-sided", STATISTICAL_TESTS.ONE_SAMPLE_TTEST
    )
    expected_stat, expected_p = ttest_1samp(quantiles, 0.5, alternative="two-sided")
    assert np.isclose(result_t[RANK_SHIFT_SUMMARIES.STATISTIC], expected_stat)
    assert np.isclose(result_t[RANK_SHIFT_SUMMARIES.P_VALUE], expected_p)


def test_compute_rank_shift_for_group_alternatives():
    """Test _compute_rank_shift_for_group with different alternatives."""
    # Low ranks = low quantiles
    result_less = _compute_rank_shift_for_group(
        pd.DataFrame({"rank": [1, 2, 3, 4, 5]}),
        "rank",
        100,
        "less",
        STATISTICAL_TESTS.WILCOXON_RANKSUM,
    )
    assert result_less[RANK_SHIFT_SUMMARIES.MEAN_QUANTILE] < 0.5
    assert result_less[RANK_SHIFT_SUMMARIES.P_VALUE] < 0.05

    # High ranks = high quantiles
    result_greater = _compute_rank_shift_for_group(
        pd.DataFrame({"rank": [96, 97, 98, 99, 100]}),
        "rank",
        100,
        "greater",
        STATISTICAL_TESTS.WILCOXON_RANKSUM,
    )
    assert result_greater[RANK_SHIFT_SUMMARIES.MEAN_QUANTILE] > 0.5
    assert result_greater[RANK_SHIFT_SUMMARIES.P_VALUE] < 0.05


# Tests for calculate_rank_shift (public function) - <50 lines
def test_calculate_rank_shift_no_grouping():
    """Test calculate_rank_shift without grouping."""
    df = pd.DataFrame({"rank": [10, 20, 30, 40, 50]})
    result = calculate_rank_shift(df, "rank", 100, grouping_vars=None)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    expected_cols = [
        RANK_SHIFT_SUMMARIES.MEAN_QUANTILE,
        RANK_SHIFT_SUMMARIES.MEDIAN_QUANTILE,
        RANK_SHIFT_SUMMARIES.MIN_QUANTILE,
        RANK_SHIFT_SUMMARIES.MAX_QUANTILE,
        RANK_SHIFT_SUMMARIES.STATISTIC,
        RANK_SHIFT_SUMMARIES.P_VALUE,
    ]
    assert all(col in result.columns for col in expected_cols)
    assert result[RANK_SHIFT_SUMMARIES.MEAN_QUANTILE].iloc[0] == 0.3


def test_calculate_rank_shift_grouping():
    """Test calculate_rank_shift with single and multiple grouping."""
    # Single grouping
    df1 = pd.DataFrame({"layer": [0, 0, 1, 1], "rank": [10, 20, 30, 40]})
    result1 = calculate_rank_shift(df1, "rank", 100, grouping_vars="layer")
    assert len(result1) == 2 and "layer" in result1.columns
    assert set(result1["layer"]) == {0, 1}

    # Multiple grouping
    df2 = pd.DataFrame(
        {"model": ["A", "A", "B", "B"], "layer": [0, 1, 0, 1], "rank": [10, 20, 30, 40]}
    )
    result2 = calculate_rank_shift(df2, "rank", 100, grouping_vars=["model", "layer"])
    assert len(result2) == 4
    assert "model" in result2.columns and "layer" in result2.columns


def test_calculate_rank_shift_validation():
    """Test calculate_rank_shift input validation."""
    with pytest.raises(ValueError, match="DataFrame must contain 'rank' column"):
        calculate_rank_shift(pd.DataFrame({"value": [1, 2]}), "rank", 100)
    with pytest.raises(ValueError, match="Rank values must be less than or equal to"):
        calculate_rank_shift(pd.DataFrame({"rank": [10, 101]}), "rank", 100)
    with pytest.raises(
        ValueError, match="Rank values must be greater than or equal to 1"
    ):
        calculate_rank_shift(pd.DataFrame({"rank": [0, 10]}), "rank", 100)
    with pytest.raises(ValueError, match="DataFrame must contain 'layer' column"):
        calculate_rank_shift(
            pd.DataFrame({"rank": [10]}), "rank", 100, grouping_vars="layer"
        )
    with pytest.raises(ValueError, match="test_method must be one of"):
        calculate_rank_shift(
            pd.DataFrame({"rank": [10]}), "rank", 100, test_method="invalid"
        )
    # Verify valid test methods work
    assert STATISTICAL_TESTS.WILCOXON_RANKSUM in RANK_SHIFT_TESTS
    assert STATISTICAL_TESTS.ONE_SAMPLE_TTEST in RANK_SHIFT_TESTS


def test_calculate_rank_shift_alternatives_and_methods():
    """Test calculate_rank_shift with different alternatives and test methods."""
    df = pd.DataFrame({"rank": [10, 20, 30, 40, 50]})
    result_g = calculate_rank_shift(df, "rank", 100, alternative="greater")
    result_l = calculate_rank_shift(df, "rank", 100, alternative="less")
    result_t = calculate_rank_shift(df, "rank", 100, alternative="two-sided")
    assert (
        result_g[RANK_SHIFT_SUMMARIES.MEAN_QUANTILE].iloc[0]
        == result_l[RANK_SHIFT_SUMMARIES.MEAN_QUANTILE].iloc[0]
    )
    assert all(
        RANK_SHIFT_SUMMARIES.P_VALUE in r.columns
        for r in [result_g, result_l, result_t]
    )

    result_w = calculate_rank_shift(
        df, "rank", 100, test_method=STATISTICAL_TESTS.WILCOXON_RANKSUM
    )
    result_tt = calculate_rank_shift(
        df, "rank", 100, test_method=STATISTICAL_TESTS.ONE_SAMPLE_TTEST
    )
    assert (
        result_w[RANK_SHIFT_SUMMARIES.MEAN_QUANTILE].iloc[0]
        == result_tt[RANK_SHIFT_SUMMARIES.MEAN_QUANTILE].iloc[0]
    )
