"""Tests for base utility functions."""

import pytest

from napistu_torch.utils.base_utils import shortest_common_prefix


@pytest.mark.parametrize(
    "names,expected",
    [
        (["is_string_x", "is_string_y"], "is_string"),
        (["is_omnipath_kinase", "is_omnipath_phosphatase"], "is_omnipath"),
        (["is_a", "is_b"], "is_a"),
        (["single_name"], "single_name"),
        (["exact_match", "exact_match"], "exact_match"),
    ],
)
def test_shortest_common_prefix(names, expected):
    """Test shortest_common_prefix with various input cases."""
    result = shortest_common_prefix(names)
    assert result == expected
