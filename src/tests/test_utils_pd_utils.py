import numpy as np
import pandas as pd

from napistu_torch.utils.pd_utils import calculate_ranks


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
