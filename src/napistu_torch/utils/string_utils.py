"""
String utilities.

Public Functions
----------------
sanitize_filename(s)
    Sanitize a string for use as a filename component.
"""

import re


def sanitize_filename(s: str) -> str:
    """Sanitize a string for safe use as a filename component.

    Removes characters that are problematic in filenames, collapses
    whitespace to underscores, and strips leading/trailing separators.
    The original string is preserved in the residuals index — this is
    only used to construct the on-disk filename stem.

    Parameters
    ----------
    s
        Input string (e.g., a category name like 'adipocyte (0)').

    Returns
    -------
    str
        Filesystem-safe string (e.g., 'adipocyte_0').
    """
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s.strip("-_")
