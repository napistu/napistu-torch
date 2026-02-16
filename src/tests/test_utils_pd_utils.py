import numpy as np
import pandas as pd
import pytest

from napistu_torch.utils.pd_utils import (
    calculate_ranks,
    filter_and_reorder_df,
    reorder_multindex_by_categorical_and_numeric,
)


def test_calculate_ranks():
    """Test calculate_ranks with various scenarios."""
    # Test 1: Basic ranking without grouping
    df1 = pd.DataFrame({"value": [0.9, 0.8, 0.7, 0.6, 0.5]})
    ranks1 = calculate_ranks(df1, "value", by_absolute_value=False)
    assert ranks1.dtype == np.int64
    assert list(ranks1) == [1, 2, 3, 4, 5]

    # Test 2: Ranking with single grouping variable
    df2 = pd.DataFrame(
        {"model": ["A", "A", "B", "B"], "attention": [0.9, 0.8, 0.7, 0.6]}
    )
    ranks2 = calculate_ranks(df2, "attention", grouping_vars="model")
    assert ranks2.iloc[0] == 1  # A, 0.9
    assert ranks2.iloc[1] == 2  # A, 0.8
    assert ranks2.iloc[2] == 1  # B, 0.7
    assert ranks2.iloc[3] == 2  # B, 0.6

    # Test 3: Ranking by absolute value with negative values
    df3 = pd.DataFrame({"value": [-0.9, 0.8, -0.7, 0.6]})
    ranks3_abs = calculate_ranks(df3, "value", by_absolute_value=True)
    ranks3_raw = calculate_ranks(df3, "value", by_absolute_value=False)
    assert ranks3_abs.iloc[0] == 1  # -0.9 has highest absolute value
    assert ranks3_raw.iloc[1] == 1  # 0.8 is highest raw value

    # Test 4: Multiple grouping variables
    df4 = pd.DataFrame(
        {
            "model": ["A", "A", "B", "B"],
            "layer": [0, 1, 0, 1],
            "attention": [0.9, 0.8, 0.7, 0.6],
        }
    )
    ranks4 = calculate_ranks(df4, "attention", grouping_vars=["model", "layer"])
    assert ranks4.iloc[0] == 1  # A, 0, 0.9
    assert ranks4.iloc[1] == 1  # A, 1, 0.8
    assert ranks4.iloc[2] == 1  # B, 0, 0.7
    assert ranks4.iloc[3] == 1  # B, 1, 0.6


def test_filter_and_reorder_df():
    """Basic tests for filter_and_reorder_df."""
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "val": [1, 2, 3],
        }
    )
    out = filter_and_reorder_df(df, target_ids=["c", "a"], id_column="id")
    assert out["id"].tolist() == ["c", "a"]
    assert out["val"].tolist() == [3, 1]
    assert len(out) == 2

    # target_ids not in df are skipped
    out2 = filter_and_reorder_df(df, target_ids=["c", "x", "a"], id_column="id")
    assert out2["id"].tolist() == ["c", "a"]

    # missing id_column raises
    with pytest.raises(ValueError) as e:
        filter_and_reorder_df(df, target_ids=["a"], id_column="missing")
    assert "not found" in str(e.value).lower()

    # no matches raises
    with pytest.raises(ValueError) as e:
        filter_and_reorder_df(df, target_ids=["x", "y"], id_column="id")
    assert "no target_ids found" in str(e.value).lower()

    # duplicates in id_column among matched rows raise
    df_dup = pd.DataFrame({"id": ["a", "a", "b"], "val": [1, 2, 3]})
    with pytest.raises(ValueError) as e:
        filter_and_reorder_df(df_dup, target_ids=["a", "b"], id_column="id")
    assert "duplicate" in str(e.value).lower()


def test_reorder_multindex_success():
    """Test successful MultiIndex reordering."""
    idx = pd.MultiIndex.from_tuples(
        [("B", 2), ("A", 1), ("A", 0), ("B", 0)], names=["model", "layer"]
    )
    categorical_order = ["A", "B"]
    result = reorder_multindex_by_categorical_and_numeric(
        idx, categorical_order, categorical_level=0, numeric_level=1
    )
    expected = pd.MultiIndex.from_tuples(
        [("A", 0), ("A", 1), ("B", 0), ("B", 2)], names=["model", "layer"]
    )
    assert result.equals(expected)


def test_reorder_multindex_invalid():
    """Test invalid MultiIndex with extra categorical values."""
    idx = pd.MultiIndex.from_tuples(
        [("A", 0), ("B", 1), ("C", 0)], names=["model", "layer"]
    )
    categorical_order = ["A", "B"]
    try:
        reorder_multindex_by_categorical_and_numeric(
            idx, categorical_order, categorical_level=0, numeric_level=1
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "C" in str(e)
