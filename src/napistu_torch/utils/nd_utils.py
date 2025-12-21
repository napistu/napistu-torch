"""Utilities for NapistuData objects."""

from hashlib import sha256
from typing import Any, Dict, Optional

import pandas as pd
from torch import Tensor

from napistu_torch.constants import NAPISTU_DATA, NAPISTU_DATA_SUMMARIES, PYG


def add_optional_attr(summary_dict: Dict[str, Any], attr_name: str, value: Any) -> None:
    """Add an attribute to NapistuData summary if it is not None."""
    if value is not None:
        summary_dict[attr_name] = value


def compute_mask_hashes(
    train_mask: Optional[Tensor] = None,
    val_mask: Optional[Tensor] = None,
    test_mask: Optional[Tensor] = None,
) -> Dict[str, Optional[str]]:
    """
    Compute deterministic hashes of train/val/test masks.

    Uses SHA256 hash of the mask tensor bytes for reproducible comparison.
    Returns None for missing masks.

    Parameters
    ----------
    train_mask : Optional[torch.Tensor]
        Training mask tensor
    val_mask : Optional[torch.Tensor]
        Validation mask tensor
    test_mask : Optional[torch.Tensor]
        Test mask tensor

    Returns
    -------
    Dict[str, Optional[str]]
        Dictionary with keys 'train_mask_hash', 'val_mask_hash', 'test_mask_hash'
        Values are SHA256 hex strings or None if mask not provided

    Examples
    --------
    >>> hashes = compute_mask_hashes(train_mask=data.train_mask)
    >>> print(hashes['train_mask_hash'][:16])  # First 16 chars
    'a1b2c3d4e5f6g7h8'
    """
    hash_dict = {}

    masks = {
        NAPISTU_DATA.TRAIN_MASK: train_mask,
        NAPISTU_DATA.VAL_MASK: val_mask,
        NAPISTU_DATA.TEST_MASK: test_mask,
    }

    for mask_name, mask in masks.items():
        if mask is not None:
            # Convert to numpy for consistent byte representation
            mask_bytes = mask.cpu().numpy().tobytes()
            hash_hex = sha256(mask_bytes).hexdigest()
            hash_dict[f"{mask_name}_hash"] = hash_hex
        else:
            hash_dict[f"{mask_name}_hash"] = None

    return hash_dict


def format_summary(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Format NapistuData summary into a clean table for display.

    Parameters
    ----------
    data : Dict[str, Any]
        Summary dictionary from NapistuData.get_summary("detailed")

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
