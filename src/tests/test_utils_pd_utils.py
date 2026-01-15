import numpy as np
import pandas as pd

from napistu_torch.utils.pd_utils import (
    calculate_ranks,
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
