"""
Utilities for comparing and validating NapistuData compatibility.

This module provides functions for validating that two NapistuData objects
are compatible for training/inference, including feature alignment, split
consistency, and structural compatibility checks.
"""

import logging
from typing import Any, Dict, List, Optional

from napistu_torch.constants import (
    MASK_TO_HASH,
    NAPISTU_DATA,
)
from napistu_torch.data.constants import DEFAULT_SAME_DATA_ALLOW_MISSING_KEYS

logger = logging.getLogger(__name__)


def validate_same_data(
    current_summary: Dict[str, Any],
    reference_summary: Dict[str, Any],
    allow_missing_keys: Optional[List[str]] = None,
    verbose: bool = False,
) -> None:
    """
    Validate that data summaries from reference and current data are compatible.

    Performs comprehensive validation including:
    - Structural attributes (num_nodes, num_edges, num_features)
    - Feature names and ordering (vertex and edge)
    - Feature aliases and canonical mappings
    - Relation type labels
    - Train/val/test split consistency (warnings only)

    Parameters
    ----------
    current_summary : Dict[str, Any]
        Data summary from current NapistuData (e.g., inference data)
    reference_summary : Dict[str, Any]
        Data summary from reference (e.g., checkpoint training data)
    allow_missing_keys : Optional[List[str]]
        Keys that are allowed to be missing in either summary.
        If present in both, values must still match.
        Defaults to [num_edge_features, num_unique_relations, num_unique_classes,
        train_mask_hash, val_mask_hash, test_mask_hash]
    verbose : bool
        Whether to print verbose output

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If summaries are incompatible in any way

    Examples
    --------
    >>> current_summary = napistu_data.get_summary("validation")
    >>> reference_summary = checkpoint.get_data_summary()
    >>> validate_same_data(current_summary, reference_summary)
    """
    if allow_missing_keys is None:
        allow_missing_keys = DEFAULT_SAME_DATA_ALLOW_MISSING_KEYS

    # 1. Validate keys presence
    _validate_keys(current_summary, reference_summary, allow_missing_keys)

    # 2. Validate structural attributes (numbers match)
    _validate_structural_attributes(current_summary, reference_summary)

    # 3. Validate feature names (order matters!)
    _validate_feature_names(
        current_summary.get(NAPISTU_DATA.VERTEX_FEATURE_NAMES),
        reference_summary.get(NAPISTU_DATA.VERTEX_FEATURE_NAMES),
        "vertex",
        verbose,
    )

    _validate_feature_names(
        current_summary.get(NAPISTU_DATA.EDGE_FEATURE_NAMES),
        reference_summary.get(NAPISTU_DATA.EDGE_FEATURE_NAMES),
        "edge",
        verbose,
    )

    # 4. Validate aliases
    _validate_feature_aliases(
        current_summary.get(NAPISTU_DATA.VERTEX_FEATURE_NAME_ALIASES),
        reference_summary.get(NAPISTU_DATA.VERTEX_FEATURE_NAME_ALIASES),
        reference_summary.get(NAPISTU_DATA.VERTEX_FEATURE_NAMES),
        "vertex",
    )

    _validate_feature_aliases(
        current_summary.get(NAPISTU_DATA.EDGE_FEATURE_NAME_ALIASES),
        reference_summary.get(NAPISTU_DATA.EDGE_FEATURE_NAME_ALIASES),
        reference_summary.get(NAPISTU_DATA.EDGE_FEATURE_NAMES),
        "edge",
    )

    # 5. Validate relation type labels
    _validate_relation_labels(
        current_summary.get(NAPISTU_DATA.RELATION_TYPE_LABELS),
        reference_summary.get(NAPISTU_DATA.RELATION_TYPE_LABELS),
    )

    # 6. Validate mask hashes (warnings instead of errors)
    _validate_mask_hashes(current_summary, reference_summary)

    if verbose:
        logger.info("✓ Data validation passed")

    return None


# private functions


def _is_comparable_value(val1: Any, val2: Any) -> bool:
    """
    Check if two values should be compared for equality.

    Only compares simple scalar types (int, float, str, bool).
    Skips lists, dicts, None, and other complex types.

    Parameters
    ----------
    val1 : Any
        First value
    val2 : Any
        Second value

    Returns
    -------
    bool
        True if values should be compared, False otherwise
    """
    # Skip if either is None
    if val1 is None or val2 is None:
        return False

    # Skip if either is a collection type
    if isinstance(val1, (list, dict)) or isinstance(val2, (list, dict)):
        return False

    # Only compare simple scalar types
    return isinstance(val1, (int, float, str, bool)) and isinstance(
        val2, (int, float, str, bool)
    )


def _validate_feature_aliases(
    current_aliases: Optional[Dict[str, str]],
    reference_aliases: Optional[Dict[str, str]],
    reference_feature_names: Optional[List[str]],
    feature_type: str,
    verbose: bool = False,
) -> None:
    """
    Validate feature aliases match and reference valid canonical features.

    Parameters
    ----------
    current_aliases : Optional[Dict[str, str]]
        Aliases from current data
    reference_aliases : Optional[Dict[str, str]]
        Aliases from reference
    reference_feature_names : Optional[List[str]]
        Feature names from reference (for validating canonical references)
    feature_type : str
        Type of features ('vertex' or 'edge') for error messages
    verbose : bool
        Whether to print verbose output

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If aliases don't match or reference invalid canonical features
    """
    # Both None is okay (no deduplication)
    if current_aliases is None and reference_aliases is None:
        return

    # One has aliases, one doesn't - incompatible
    if current_aliases is None or reference_aliases is None:
        raise ValueError(
            f"{feature_type.capitalize()} feature aliases mismatch: "
            f"current={'present' if current_aliases else 'missing'}, "
            f"reference={'present' if reference_aliases else 'missing'}"
        )

    # Check keys match (aliased feature names)
    curr_keys = set(current_aliases.keys())
    ref_keys = set(reference_aliases.keys())

    if curr_keys != ref_keys:
        missing_in_reference = curr_keys - ref_keys
        extra_in_reference = ref_keys - curr_keys
        raise ValueError(
            f"{feature_type.capitalize()} feature alias keys mismatch:\n"
            f"  Missing in reference: {sorted(missing_in_reference) if missing_in_reference else 'none'}\n"
            f"  Extra in reference: {sorted(extra_in_reference) if extra_in_reference else 'none'}\n"
            f"\nThis indicates different feature deduplication between reference and current data."
        )

    # Check values match (canonical feature names)
    mismatches = []
    for alias_name in ref_keys:
        curr_canonical = current_aliases[alias_name]
        ref_canonical = reference_aliases[alias_name]

        if curr_canonical != ref_canonical:
            mismatches.append(
                f"  '{alias_name}': current='{curr_canonical}', reference='{ref_canonical}'"
            )

        # Verify canonical feature exists in feature names
        if (
            reference_feature_names is not None
            and ref_canonical not in reference_feature_names
        ):
            raise ValueError(
                f"{feature_type.capitalize()} alias '{alias_name}' references "
                f"canonical feature '{ref_canonical}' which doesn't exist in feature names.\n"
                f"This indicates corrupted alias mapping in reference data."
            )

    if mismatches:
        raise ValueError(
            f"{feature_type.capitalize()} aliases point to different canonical features:\n"
            + "\n".join(mismatches)
        )

    if verbose:
        logger.info(
            f"✓ {feature_type.capitalize()} feature aliases validated ({len(ref_keys)} aliases)"
        )

    return None


def _validate_feature_names(
    current_names: Optional[List[str]],
    reference_names: Optional[List[str]],
    feature_type: str,
    verbose: bool = False,
) -> None:
    """
    Validate feature names match exactly (identical order).

    Parameters
    ----------
    reference_names : Optional[List[str]]
        Feature names from reference
    current_names : Optional[List[str]]
        Feature names from current data
    feature_type : str
        Type of features ('vertex' or 'edge') for error messages
    verbose : bool
        Whether to print verbose output

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If feature names don't match exactly or are in different order
    """
    # Both None is okay
    if current_names is None and reference_names is None:
        logger.warning(f"No {feature_type} feature names in current or reference data")
        return

    # One None, one not - incompatible
    if current_names is None or reference_names is None:
        raise ValueError(
            f"{feature_type.capitalize()} feature names mismatch: "
            f"current={'present' if current_names else 'missing'}, "
            f"reference={'present' if reference_names else 'missing'}"
        )

    # Different lengths
    if len(current_names) != len(reference_names):
        raise ValueError(
            f"{feature_type.capitalize()} feature count mismatch: "
            f"current has {len(current_names)}, reference has {len(reference_names)}"
        )

    # Check for identical ordering (strict)
    if current_names != reference_names:
        # Find first mismatch for helpful error message
        for i, (curr, ref) in enumerate(zip(current_names, reference_names)):
            if ref != curr:
                raise ValueError(
                    f"{feature_type.capitalize()} feature names are not identically ordered.\n"
                    f"First mismatch at index {i}:\n"
                    f"  Current: {curr}\n"
                    f"  Reference: {ref}\n"
                    f"\nThis usually indicates:\n"
                    f"  1. Features were created in different order\n"
                    f"  2. Different encoding configuration was used\n"
                    f"  3. Data preprocessing changed between reference and current data\n"
                    f"\nFull current features ({len(current_names)}): {current_names[:10]}{'...' if len(current_names) > 10 else ''}\n"
                    f"Full reference features ({len(reference_names)}): {reference_names[:10]}{'...' if len(reference_names) > 10 else ''}\n"
                )

    if verbose:
        logger.info(
            f"✓ {feature_type.capitalize()} feature names validated ({len(reference_names)} features)"
        )

    return None


def _validate_keys(
    current_summary: Dict[str, Any],
    reference_summary: Dict[str, Any],
    allow_missing_keys: List[str],
) -> None:
    """
    Validate that required keys are present in both summaries.

    Parameters
    ----------
    current_summary : Dict[str, Any]
        Data summary from current NapistuData
    reference_summary : Dict[str, Any]
        Data summary from reference
    allow_missing_keys : List[str]
        Keys allowed to be missing in either summary

    Raises
    ------
    ValueError
        If required keys are missing
    """
    current_keys = set(current_summary.keys())
    reference_keys = set(reference_summary.keys())
    allow_missing_keys_set = set(allow_missing_keys)

    key_union = current_keys | reference_keys
    key_intersection = current_keys & reference_keys
    key_difference = key_union - key_intersection
    key_difference_without_allow_missing = key_difference - allow_missing_keys_set

    if key_difference_without_allow_missing:
        missing_in_current = key_difference_without_allow_missing & reference_keys
        missing_in_reference = key_difference_without_allow_missing & current_keys

        msg_parts = ["Data summary mismatch:"]
        if missing_in_current:
            msg_parts.append(f"  Missing in current data: {sorted(missing_in_current)}")
        if missing_in_reference:
            msg_parts.append(
                f"  Missing in reference data: {sorted(missing_in_reference)}"
            )

        raise ValueError("\n".join(msg_parts))


def _validate_structural_attributes(
    current_summary: Dict[str, Any],
    reference_summary: Dict[str, Any],
) -> None:
    """
    Validate that structural numeric attributes match.

    Checks that core attributes like num_nodes, num_edges, etc. have
    identical values in both summaries.

    Parameters
    ----------
    current_summary : Dict[str, Any]
        Data summary from current NapistuData
    reference_summary : Dict[str, Any]
        Data summary from reference

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any structural attributes don't match
    """
    # Get keys present in both
    common_keys = set(current_summary.keys()) & set(reference_summary.keys())

    # Check for value mismatches
    mismatches = []
    for key in common_keys:
        curr_val = current_summary[key]
        ref_val = reference_summary[key]

        # Skip non-comparable types (lists, dicts, None, etc.)
        if not _is_comparable_value(curr_val, ref_val):
            continue

        if curr_val != ref_val:
            mismatches.append(f"  {key}: current={curr_val}, reference={ref_val}")

    if mismatches:
        raise ValueError(
            "Data summary structural attributes don't match:\n" + "\n".join(mismatches)
        )

    return None


def _validate_relation_labels(
    reference_labels: Optional[List[str]],
    current_labels: Optional[List[str]],
) -> None:
    """
    Validate relation type labels match exactly (identical order).

    Parameters
    ----------
    reference_labels : Optional[List[str]]
        Relation labels from reference
    current_labels : Optional[List[str]]
        Relation labels from current data

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If relation labels don't match exactly or are in different order
    """
    # Both None is okay (no relation types)
    if reference_labels is None and current_labels is None:
        return

    # One has labels, one doesn't - warn but don't fail
    if reference_labels is None or current_labels is None:
        logger.warning(
            f"Relation type labels mismatch: "
            f"reference={'present' if reference_labels else 'missing'}, "
            f"current={'present' if current_labels else 'missing'}"
        )
        return

    # Different lengths
    if len(reference_labels) != len(current_labels):
        raise ValueError(
            f"Relation type count mismatch: "
            f"reference has {len(reference_labels)}, current has {len(current_labels)}"
        )

    # Check for identical ordering (strict)
    if reference_labels != current_labels:
        # Find first mismatch for helpful error message
        for i, (ref, curr) in enumerate(zip(reference_labels, current_labels)):
            if ref != curr:
                raise ValueError(
                    f"Relation type labels are not identically ordered.\n"
                    f"First mismatch at index {i}:\n"
                    f"  Reference: {ref}\n"
                    f"  Current: {curr}\n"
                    f"\nThis will cause incorrect predictions for relation-aware heads.\n"
                    f"Full reference labels: {reference_labels}\n"
                    f"Full current labels: {current_labels}"
                )

    logger.info(f"✓ Relation type labels validated ({len(reference_labels)} types)")


def _validate_mask_hashes(
    current_summary: Dict[str, Any],
    reference_summary: Dict[str, Any],
    verbose: bool = False,
) -> None:
    """
    Validate train/val/test mask hashes match (warns on mismatch).

    This checks whether the data splits are identical between reference
    and current data. Mismatches are logged as warnings rather than errors
    since different splits may be intentional for evaluation.

    Parameters
    ----------
    current_summary : Dict[str, Any]
        Data summary from current NapistuData
    reference_summary : Dict[str, Any]
        Data summary from reference
    verbose : bool
        Whether to print verbose output

    Returns
    -------
    None
    """
    mask_types = [
        NAPISTU_DATA.TRAIN_MASK,
        NAPISTU_DATA.VAL_MASK,
        NAPISTU_DATA.TEST_MASK,
    ]
    for mask_type in mask_types:
        hash_key = MASK_TO_HASH[mask_type]
        curr_hash = current_summary.get(hash_key)
        ref_hash = reference_summary.get(hash_key)

        # Both None is okay (mask not present)
        if curr_hash is None and ref_hash is None:
            logger.warning(
                f"⚠ {mask_type} hash key is missing in both reference and current data"
            )
            continue

        # One None, one not - different presence
        if curr_hash is None or ref_hash is None:
            logger.warning(
                f"⚠ {mask_type} presence mismatch: "
                f"current={'present' if curr_hash else 'missing'}, "
                f"reference={'present' if ref_hash else 'missing'}"
            )
            continue

        # Both present - check hash
        if curr_hash != ref_hash:
            logger.warning(
                f"⚠ {mask_type} has changed between reference and current data.\n"
                f"  This means train/val/test splits are different.\n"
                f"  Current hash: {curr_hash[:16]}...\n"
                f"  Reference hash: {ref_hash[:16]}...\n"
                f"  This may affect evaluation metrics but not model compatibility."
            )
        elif verbose:
            logger.info(f"✓ {mask_type} validated (identical split)")
