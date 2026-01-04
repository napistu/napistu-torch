"""General utilities for the Python standard library."""

from typing import List, Union


def normalize_and_validate_indices(
    indices: Union[int, List[int], tuple, range],
    max_value: int,
    param_name: str = "indices",
) -> List[int]:
    """
    Normalize indices to a list and validate they are integers in valid range.

    Parameters
    ----------
    indices : int, List[int], tuple, or range
        Indices to normalize and validate. Can be a single integer, list, tuple, or range.
    max_value : int
        Maximum valid index value (exclusive). Valid range is [0, max_value).
    param_name : str, optional
        Name of the parameter for error messages (default: "indices")

    Returns
    -------
    List[int]
        Normalized list of validated indices

    Raises
    ------
    ValueError
        If indices are invalid (wrong type, not integers, out of range, or duplicates)
    """
    # Handle single integer input
    if isinstance(indices, int):
        indices = [indices]
    # Explicit type check for allowed types
    elif not isinstance(indices, (list, tuple, range)):
        raise ValueError(
            f"{param_name} must be an int, list, tuple, or range, got {type(indices).__name__}"
        )

    # Normalize to list
    indices = list(indices)

    if len(indices) == 0:
        raise ValueError(f"{param_name} cannot be empty")

    # Validate all are integers
    if not all(isinstance(i, int) for i in indices):
        raise ValueError(f"{param_name} must be a list of integers")

    # Validate all are in valid range
    invalid = [idx for idx in indices if idx < 0 or idx >= max_value]
    if invalid:
        raise ValueError(
            f"{param_name} contains invalid values: {invalid}. "
            f"All indices must be in range [0, {max_value})"
        )

    # Check for duplicates
    if len(indices) != len(set(indices)):
        duplicates = [idx for idx in set(indices) if indices.count(idx) > 1]
        raise ValueError(f"{param_name} contains duplicates: {duplicates}")

    return indices


def shortest_common_prefix(names: List[str], min_length: int = 3) -> str:
    """
    Find shortest common prefix respecting word boundaries.

    Parameters
    ----------
    names : List[str]
        Feature names to find common prefix for
    min_length : int, default=3
        Minimum acceptable prefix length

    Returns
    -------
    str
        Shortest common prefix, or alphabetically first name if prefix too short

    Examples
    --------
    >>> shortest_common_prefix(['is_string_x', 'is_string_y'])
    'is_string'
    >>> shortest_common_prefix(['is_omnipath_kinase', 'is_omnipath_phosphatase'])
    'is_omnipath'
    >>> shortest_common_prefix(['is_a', 'is_b'])  # Too short
    'is_a'
    """
    if len(names) == 1:
        return names[0]

    # Find character-by-character common prefix
    prefix = []
    for chars in zip(*names):
        if len(set(chars)) == 1:
            prefix.append(chars[0])
        else:
            break

    prefix_str = "".join(prefix)

    # Trim to last complete word (respect underscore boundaries)
    # Only trim if prefix ends in an underscore
    if prefix_str.endswith("_"):
        prefix_str = prefix_str.rstrip("_")

    # Enforce minimum length - fall back to first alphabetically if too short
    if len(prefix_str) < min_length:
        return sorted(names)[0]

    return prefix_str


class CorruptionError(ValueError):
    """Raised when MPS memory corruption is detected."""

    pass
