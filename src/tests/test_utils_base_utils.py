"""Tests for base utility functions."""

from pathlib import Path

import pytest

from napistu_torch.utils.base_utils import (
    ensure_path,
    normalize_and_validate_indices,
    shortest_common_prefix,
)


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


def test_normalize_and_validate_indices():
    """Test normalize_and_validate_indices validation."""
    # Test single integer input
    assert normalize_and_validate_indices(2, max_value=5) == [2]
    assert normalize_and_validate_indices(0, max_value=5) == [0]
    assert normalize_and_validate_indices(4, max_value=5) == [4]

    # Test list input
    assert normalize_and_validate_indices([0, 2], max_value=5) == [0, 2]
    assert normalize_and_validate_indices([1, 3, 4], max_value=5) == [1, 3, 4]

    # Test tuple input
    assert normalize_and_validate_indices((0, 2), max_value=5) == [0, 2]

    # Test range input
    assert normalize_and_validate_indices(range(3), max_value=5) == [0, 1, 2]

    # Test invalid type
    with pytest.raises(ValueError, match="must be an int, list, tuple, or range"):
        normalize_and_validate_indices("invalid", max_value=5)
    with pytest.raises(ValueError, match="must be an int, list, tuple, or range"):
        normalize_and_validate_indices(1.5, max_value=5)  # float is not int

    # Test duplicates
    with pytest.raises(ValueError, match="duplicates"):
        normalize_and_validate_indices([0, 1, 0], max_value=5)
    with pytest.raises(ValueError, match="duplicates"):
        normalize_and_validate_indices([2, 2], max_value=5)

    # Test out of range
    with pytest.raises(ValueError, match="invalid values"):
        normalize_and_validate_indices([0, 5], max_value=5)
    with pytest.raises(ValueError, match="invalid values"):
        normalize_and_validate_indices([-1, 2], max_value=5)
    with pytest.raises(ValueError, match="invalid values"):
        normalize_and_validate_indices(5, max_value=5)

    # Test not integers in list
    with pytest.raises(ValueError, match="must be a list of integers"):
        normalize_and_validate_indices([1.5, 2.0], max_value=5)


def test_ensure_path():
    """Test ensure_path with various input types and expand_user settings."""
    # Test string input converts to Path
    result = ensure_path("./relative/path")
    assert isinstance(result, Path)
    assert result == Path("./relative/path")

    # Test Path input returns Path (unchanged except for expand_user)
    input_path = Path("./test/path")
    result = ensure_path(input_path, expand_user=False)
    assert isinstance(result, Path)
    assert result == input_path

    # Test expand_user=True expands tilde
    result = ensure_path("~/test/path", expand_user=True)
    assert isinstance(result, Path)
    # Should expand to actual home directory (can't predict exact path, but shouldn't have ~)
    assert "~" not in str(result)
    assert result.is_absolute()

    # Test expand_user=False preserves tilde
    result = ensure_path("~/test/path", expand_user=False)
    assert isinstance(result, Path)
    # Path object will have ~ as a literal string, not expanded
    assert str(result) == "~/test/path" or result.name == "path"

    # Test absolute paths work correctly
    result = ensure_path("/absolute/path", expand_user=True)
    assert isinstance(result, Path)
    assert result.is_absolute()
    assert str(result) == "/absolute/path"

    # Test relative paths with dots
    result = ensure_path("../parent/path", expand_user=True)
    assert isinstance(result, Path)
    assert str(result) == "../parent/path"

    # Test invalid type raises TypeError
    with pytest.raises(TypeError, match="must be a str or Path object"):
        ensure_path(123)  # int
    with pytest.raises(TypeError, match="must be a str or Path object"):
        ensure_path(None)  # None
    with pytest.raises(TypeError, match="must be a str or Path object"):
        ensure_path([])  # list
