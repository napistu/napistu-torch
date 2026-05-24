"""Tests for string utility functions."""

import pytest

from napistu_torch.utils.string_utils import sanitize_filename


@pytest.mark.parametrize(
    "value,expected",
    [
        ("adipocyte (0)", "adipocyte_0"),
        ("simple", "simple"),
        ("cluster  A  B", "cluster_A_B"),
        ("my-dataset", "my-dataset"),
        ("path/to\\name", "pathtoname"),
        ("___leading___", "leading"),
        ("--trim-edges--", "trim-edges"),
        ("α_cell", "α_cell"),
        ("", ""),
    ],
)
def test_sanitize_filename(value, expected):
    assert sanitize_filename(value) == expected
