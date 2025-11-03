"""General utilities for the Python standard library."""

from typing import List


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
