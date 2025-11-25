"""Utilities for NapistuData objects."""

from typing import Any, Dict

import pandas as pd

from napistu_torch.constants import NAPISTU_DATA, NAPISTU_DATA_SUMMARIES, PYG


def format_summary(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Format NapistuData summary into a clean table for display.

    Parameters
    ----------
    data : Dict[str, Any]
        Summary dictionary from NapistuData.summary()

    Returns
    -------
    pd.DataFrame
        Formatted summary table
    """
    summary_data = [
        ["Name", data[NAPISTU_DATA.NAME]],
        ["", ""],  # Spacing
        ["Nodes", f"{data[PYG.NUM_NODES]:,}"],
        ["Edges", f"{data[PYG.NUM_EDGES]:,}"],
        ["", ""],  # Spacing
        ["Node Features", f"{data[PYG.NUM_NODE_FEATURES]}"],
        ["Edge Features", f"{data[PYG.NUM_EDGE_FEATURES]}"],
    ]

    # Add splitting strategy if present
    if NAPISTU_DATA.SPLITTING_STRATEGY in data:
        summary_data.extend(
            [
                ["", ""],
                ["Splitting Strategy", data[NAPISTU_DATA.SPLITTING_STRATEGY]],
            ]
        )

    # Add relation information if present
    if NAPISTU_DATA_SUMMARIES.NUM_UNIQUE_RELATIONS in data:
        summary_data.append(
            ["Unique Relations", f"{data[NAPISTU_DATA_SUMMARIES.NUM_UNIQUE_RELATIONS]}"]
        )

    # Add mask statistics if present
    if any(
        k in data
        for k in [
            NAPISTU_DATA_SUMMARIES.NUM_TRAIN_EDGES,
            NAPISTU_DATA_SUMMARIES.NUM_VAL_EDGES,
            NAPISTU_DATA_SUMMARIES.NUM_TEST_EDGES,
        ]
    ):
        summary_data.append(["", ""])  # Spacing
        if NAPISTU_DATA_SUMMARIES.NUM_TRAIN_EDGES in data:
            summary_data.append(
                ["Train Edges", f"{data[NAPISTU_DATA_SUMMARIES.NUM_TRAIN_EDGES]:,}"]
            )
        if NAPISTU_DATA_SUMMARIES.NUM_VAL_EDGES in data:
            summary_data.append(
                ["Val Edges", f"{data[NAPISTU_DATA_SUMMARIES.NUM_VAL_EDGES]:,}"]
            )
        if NAPISTU_DATA_SUMMARIES.NUM_TEST_EDGES in data:
            summary_data.append(
                ["Test Edges", f"{data[NAPISTU_DATA_SUMMARIES.NUM_TEST_EDGES]:,}"]
            )

    # Add optional attributes - one per row with checkmarks
    summary_data.append(["", ""])  # Spacing

    optional_attrs = [
        (PYG.EDGE_WEIGHT, NAPISTU_DATA_SUMMARIES.HAS_EDGE_WEIGHTS),
        (
            NAPISTU_DATA.VERTEX_FEATURE_NAMES,
            NAPISTU_DATA_SUMMARIES.HAS_VERTEX_FEATURE_NAMES,
        ),
        (
            NAPISTU_DATA.EDGE_FEATURE_NAMES,
            NAPISTU_DATA_SUMMARIES.HAS_EDGE_FEATURE_NAMES,
        ),
        (NAPISTU_DATA.NG_VERTEX_NAMES, NAPISTU_DATA_SUMMARIES.HAS_NG_VERTEX_NAMES),
        (NAPISTU_DATA.NG_EDGE_NAMES, NAPISTU_DATA_SUMMARIES.HAS_NG_EDGE_NAMES),
    ]

    for attr_name, flag_key in optional_attrs:
        has_attr = data.get(flag_key, False)
        value = "✓" if has_attr else "✗"
        summary_data.append([f"  {attr_name}", value])

    # Create DataFrame
    df = pd.DataFrame(summary_data, columns=["Metric", "Value"])

    return df
