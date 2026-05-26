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
    compare_top_k_union_ranks,
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


def _tiny_top_k_union_with_optional_extra_partition(include_extra_partition: bool):
    """
    Build a minimal fake top-k-union-style table matching attention_patterns.compare().

    Partitions use ``model`` and ``layer``. Query partition (A, 0) has two edges whose
    global ``attention_rank`` is within top_k=2.

    Optionally add a disjoint third model ``C`` with edges unrelated to query top-k —
    analogous to stuffing an extra foundation model into a multi-model compare run.
    """
    max_rank = 100
    top_k = 2

    noise_pair = ("h1", "h2")

    rows: list = []
    # Core edges whose (A, layer 0) attention_rank is ≤ top_k; other layers give mid ranks.
    rows.extend(
        [
            {
                "model": "A",
                "layer": 0,
                "from_gene": "g1",
                "to_gene": "g2",
                "attention_rank": 1,
            },
            {
                "model": "A",
                "layer": 0,
                "from_gene": "g3",
                "to_gene": "g4",
                "attention_rank": 2,
            },
            {
                "model": "A",
                "layer": 1,
                "from_gene": "g1",
                "to_gene": "g2",
                "attention_rank": 85,
            },
            {
                "model": "A",
                "layer": 1,
                "from_gene": "g3",
                "to_gene": "g4",
                "attention_rank": 86,
            },
            {
                "model": "B",
                "layer": 0,
                "from_gene": "g1",
                "to_gene": "g2",
                "attention_rank": 40,
            },
            {
                "model": "B",
                "layer": 0,
                "from_gene": "g3",
                "to_gene": "g4",
                "attention_rank": 41,
            },
            {
                "model": "B",
                "layer": 1,
                "from_gene": "g1",
                "to_gene": "g2",
                "attention_rank": 50,
            },
            {
                "model": "B",
                "layer": 1,
                "from_gene": "g3",
                "to_gene": "g4",
                "attention_rank": 51,
            },
        ]
    )

    if include_extra_partition:
        # Third model participates in the union via edges that never intersect the
        # query partition's top-k gene pairs — still requires full-matrix rows under
        # real extraction; here only C's own layers are listed against its edge.
        rows.extend(
            [
                {
                    "model": "C",
                    "layer": 0,
                    "from_gene": noise_pair[0],
                    "to_gene": noise_pair[1],
                    "attention_rank": 1,
                },
                {
                    "model": "C",
                    "layer": 1,
                    "from_gene": noise_pair[0],
                    "to_gene": noise_pair[1],
                    "attention_rank": 70,
                },
                # Rows for noise edge on legacy models so union is symmetric (mirrors pipeline)
                {
                    "model": "A",
                    "layer": 0,
                    "from_gene": noise_pair[0],
                    "to_gene": noise_pair[1],
                    "attention_rank": 99,
                },
                {
                    "model": "A",
                    "layer": 1,
                    "from_gene": noise_pair[0],
                    "to_gene": noise_pair[1],
                    "attention_rank": 92,
                },
                {
                    "model": "B",
                    "layer": 0,
                    "from_gene": noise_pair[0],
                    "to_gene": noise_pair[1],
                    "attention_rank": 93,
                },
                {
                    "model": "B",
                    "layer": 1,
                    "from_gene": noise_pair[0],
                    "to_gene": noise_pair[1],
                    "attention_rank": 94,
                },
            ]
        )

    return pd.DataFrame(rows), max_rank, top_k


def test_compare_top_k_union_ranks_marginal_invariance_under_extra_partitions():
    """
    Rank-agreement summaries for a fixed query partition should not change when disjoint
    extra edges / models are appended to ``top_k_union`` (provided max_rank/top_k unchanged).

    This reproduces one failure mode reported for multi-model compares: pairwise
    statistics should remain well-defined margins of one another; if they drift when
    an unrelated model enters the union, something upstream merged or normed incorrectly.

    xfailing this assertion would localize a regression to compare_top_k_union_ranks /
    preprocessing rather than embedding alignment differences.
    """
    df_small, mr, tk = _tiny_top_k_union_with_optional_extra_partition(
        include_extra_partition=False
    )
    df_big, mr2, tk2 = _tiny_top_k_union_with_optional_extra_partition(
        include_extra_partition=True
    )
    assert (mr, tk) == (mr2, tk2)

    grouping = ["model", "layer"]
    defining = ["from_gene", "to_gene"]

    res_small = compare_top_k_union_ranks(
        df_small,
        grouping_vars=grouping,
        defining_vars=defining,
        top_k=tk,
        max_rank=mr,
        rank_col="attention_rank",
    )
    res_big = compare_top_k_union_ranks(
        df_big,
        grouping_vars=grouping,
        defining_vars=defining,
        top_k=tk,
        max_rank=mr,
        rank_col="attention_rank",
    )

    # Restrict to summaries that involve only models A,B (ignore any row involving C).
    key_cols = ["query_model", "query_layer", "eval_model", "eval_layer"]
    res_big_ab = res_big[
        (~res_big["eval_model"].eq("C")) & (~res_big["query_model"].eq("C"))
    ].sort_values(key_cols)
    res_small_s = res_small.sort_values(key_cols)

    cols_cmp = (
        key_cols
        + [RANK_SHIFT_SUMMARIES.MEAN_QUANTILE, RANK_SHIFT_SUMMARIES.MEDIAN_QUANTILE]
        + [
            RANK_SHIFT_SUMMARIES.MIN_QUANTILE,
            RANK_SHIFT_SUMMARIES.MAX_QUANTILE,
        ]
    )
    pd.testing.assert_frame_equal(
        res_big_ab.reset_index(drop=True)[cols_cmp],
        res_small_s.reset_index(drop=True)[cols_cmp],
        check_dtype=False,
    )


def test_compare_top_k_union_ranks_empty_top_k_union_raises():
    """Empty grouping partitions must fail fast instead of ambiguous concat."""
    empty_union = pd.DataFrame(
        {
            "layer": pd.Series(dtype=int),
            "from_gene": pd.Series(dtype=str),
            "to_gene": pd.Series(dtype=str),
            "attention_rank": pd.Series(dtype=float),
            "attention": pd.Series(dtype=float),
        }
    )
    with pytest.raises(ValueError, match="no partitions"):
        compare_top_k_union_ranks(
            empty_union,
            grouping_vars=["layer"],
            defining_vars=["from_gene", "to_gene"],
            top_k=10,
            max_rank=100,
            rank_col="attention_rank",
        )
