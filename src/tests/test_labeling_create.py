import numpy as np
import pandas as pd
import pytest
import torch

from napistu_torch.labeling.constants import TASK_TYPES
from napistu_torch.labeling.create import encode_labels


@pytest.fixture
def valid_classification_cases():
    """Test cases for valid classification inputs."""
    return [
        {
            "input": pd.Series(["A", "B", "A", None, "C", "B"]),
            "missing_value": None,
            "expected_encoded": torch.tensor([0, 1, 0, -1, 2, 1], dtype=torch.long),
            "expected_lookup": {0: "A", 1: "B", 2: "C"},
            "description": "string labels with default missing value",
        },
        {
            "input": pd.Series([1, 2, 1, None, 3, 2]),
            "missing_value": None,
            "expected_encoded": torch.tensor([0, 1, 0, -1, 2, 1], dtype=torch.long),
            "expected_lookup": {0: 1, 1: 2, 2: 3},
            "description": "integer labels with default missing value",
        },
        {
            "input": pd.Series([1.0, 2.0, 1.0, None, 3.0, 2.0]),
            "missing_value": None,
            "expected_encoded": torch.tensor([0, 1, 0, -1, 2, 1], dtype=torch.long),
            "expected_lookup": {0: 1.0, 1: 2.0, 2: 3.0},
            "description": "float labels with default missing value",
        },
        {
            "input": pd.Series(["A", "B", "A", None, "C"], dtype="category"),
            "missing_value": None,
            "expected_encoded": torch.tensor([0, 1, 0, -1, 2], dtype=torch.long),
            "expected_lookup": {0: "A", 1: "B", 2: "C"},
            "description": "categorical labels with default missing value",
        },
        {
            "input": pd.Series(["A", "B", "A", None, "C"]),
            "missing_value": 999,
            "expected_encoded": torch.tensor([0, 1, 0, 999, 2], dtype=torch.long),
            "expected_lookup": {0: "A", 1: "B", 2: "C"},
            "description": "string labels with custom missing value",
        },
        {
            "input": pd.Series(["A", 1, "A", None, 2.0, "B"]),
            "missing_value": None,
            "expected_encoded": torch.tensor([0, 1, 0, -1, 2, 3], dtype=torch.long),
            "expected_lookup": {0: "A", 1: 1, 2: 2.0, 3: "B"},
            "description": "mixed type labels with default missing value",
        },
        {
            "input": pd.Series([], dtype=object),
            "missing_value": None,
            "expected_encoded": torch.tensor([], dtype=torch.long),
            "expected_lookup": {},
            "description": "empty series",
        },
        {
            "input": pd.Series(["protein", "metabolite", "protein", None, "drug"]),
            "missing_value": None,
            "expected_encoded": torch.tensor([2, 1, 2, -1, 0], dtype=torch.long),
            "expected_lookup": {0: "drug", 1: "metabolite", 2: "protein"},
            "description": "biological labels sorted alphabetically",
        },
    ]


@pytest.fixture
def valid_regression_cases():
    """Test cases for valid regression inputs."""
    return [
        {
            "input": pd.Series([1.5, 2.3, np.nan, 4.1, 2.3]),
            "missing_value": None,
            "expected_encoded": torch.tensor(
                [1.5, 2.3, float("nan"), 4.1, 2.3], dtype=torch.float32
            ),
            "description": "float labels with NaN missing values",
        },
        {
            "input": pd.Series([1, 2, np.nan, 4, 2]),
            "missing_value": None,
            "expected_encoded": torch.tensor(
                [1.0, 2.0, float("nan"), 4.0, 2.0], dtype=torch.float32
            ),
            "description": "integer labels with NaN missing values",
        },
        {
            "input": pd.Series([1.5, 2.3, np.nan, 4.1]),
            "missing_value": -999.0,
            "expected_encoded": torch.tensor(
                [1.5, 2.3, -999.0, 4.1], dtype=torch.float32
            ),
            "description": "float labels with custom missing value",
        },
        {
            "input": pd.Series([], dtype=float),
            "missing_value": None,
            "expected_encoded": torch.tensor([], dtype=torch.float32),
            "description": "empty series",
        },
        {
            "input": pd.Series([0.1, 0.5, np.nan, 0.8, 0.5]),
            "missing_value": None,
            "expected_encoded": torch.tensor(
                [0.1, 0.5, float("nan"), 0.8, 0.5], dtype=torch.float32
            ),
            "description": "decimal values with NaN",
        },
    ]


@pytest.fixture
def invalid_classification_cases():
    """Test cases for invalid classification inputs."""
    return [
        {
            "input": pd.Series(["A", "B", "A"]),
            "task_type": "invalid_task",
            "missing_value": None,
            "expected_error": ValueError,
            "expected_message": "task_type must be one of",
            "description": "invalid task type",
        }
    ]


@pytest.fixture
def invalid_regression_cases():
    """Test cases for invalid regression inputs."""
    return [
        {
            "input": pd.Series(["A", "B", "A"]),
            "missing_value": None,
            "expected_error": ValueError,
            "expected_message": "Continuous labels must be numeric",
            "description": "non-numeric data for regression",
        },
        {
            "input": pd.Series(["protein", "metabolite", "drug"]),
            "missing_value": None,
            "expected_error": ValueError,
            "expected_message": "Continuous labels must be numeric",
            "description": "string data for regression",
        },
    ]


def test_valid_classification_encoding(valid_classification_cases):
    """Test valid classification encoding cases."""
    for test_case in valid_classification_cases:
        encoded, lookup = encode_labels(
            test_case["input"],
            task_type=TASK_TYPES.CLASSIFICATION,
            missing_value=test_case["missing_value"],
        )

        assert torch.equal(
            encoded, test_case["expected_encoded"]
        ), f"Failed for: {test_case['description']}"
        assert (
            lookup == test_case["expected_lookup"]
        ), f"Failed lookup for: {test_case['description']}"


def test_valid_regression_encoding(valid_regression_cases):
    """Test valid regression encoding cases."""
    for test_case in valid_regression_cases:
        encoded = encode_labels(
            test_case["input"],
            task_type=TASK_TYPES.REGRESSION,
            missing_value=test_case["missing_value"],
        )

        if torch.isnan(test_case["expected_encoded"]).any():
            # Handle NaN values specially
            non_nan_mask = ~torch.isnan(test_case["expected_encoded"])
            assert torch.equal(
                encoded[non_nan_mask], test_case["expected_encoded"][non_nan_mask]
            ), f"Failed non-NaN values for: {test_case['description']}"
            assert torch.isnan(
                encoded[torch.isnan(test_case["expected_encoded"])]
            ).all(), f"Failed NaN values for: {test_case['description']}"
        else:
            assert torch.equal(
                encoded, test_case["expected_encoded"]
            ), f"Failed for: {test_case['description']}"


def test_invalid_classification_encoding(invalid_classification_cases):
    """Test invalid classification encoding cases."""
    for test_case in invalid_classification_cases:
        with pytest.raises(
            test_case["expected_error"], match=test_case["expected_message"]
        ):
            encode_labels(
                test_case["input"],
                task_type=test_case["task_type"],
                missing_value=test_case["missing_value"],
            )


def test_invalid_regression_encoding(invalid_regression_cases):
    """Test invalid regression encoding cases."""
    for test_case in invalid_regression_cases:
        with pytest.raises(
            test_case["expected_error"], match=test_case["expected_message"]
        ):
            encode_labels(
                test_case["input"],
                task_type=TASK_TYPES.REGRESSION,
                missing_value=test_case["missing_value"],
            )
