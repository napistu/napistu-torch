"""Tests for SparseContScaler in sklearn workflows"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from napistu_torch.load.encoders import SparseContScaler


def test_sparse_cont_scaler_basic():
    """Test basic fit_transform with single sparse column."""
    df = pd.DataFrame({"kinetic_constant": [1.0, 2.0, np.nan, 4.0, np.nan]})
    scaler = SparseContScaler()
    encoded = scaler.fit_transform(df)

    assert encoded.shape == (5, 2)  # indicator + value
    expected_indicator = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(encoded[:, 0], expected_indicator)
    assert encoded[2, 1] == 0.0  # missing value is 0


def test_sparse_cont_scaler_multiple_columns():
    """Test with multiple sparse columns."""
    df = pd.DataFrame(
        {"col1": [1.0, 2.0, np.nan, 4.0], "col2": [10.0, np.nan, 30.0, 40.0]}
    )
    scaler = SparseContScaler()
    encoded = scaler.fit_transform(df)

    assert encoded.shape == (4, 4)  # 2 columns per input
    np.testing.assert_array_equal(encoded[:, 0], [1.0, 1.0, 0.0, 1.0])  # col1 indicator
    np.testing.assert_array_equal(encoded[:, 2], [1.0, 0.0, 1.0, 1.0])  # col2 indicator


def test_sparse_cont_scaler_numpy_array():
    """Test with numpy array input."""
    arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
    scaler = SparseContScaler()
    encoded = scaler.fit_transform(arr)

    assert encoded.shape == (5, 2)
    assert 0 in scaler.scalers_  # uses index 0 for single array


def test_sparse_cont_scaler_all_missing():
    """Test with all missing values."""
    df = pd.DataFrame({"value": [np.nan, np.nan, np.nan]})
    scaler = SparseContScaler()
    encoded = scaler.fit_transform(df)

    assert np.all(encoded == 0.0)  # all zeros


def test_sparse_cont_scaler_no_missing():
    """Test with no missing values - should standardize."""
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = SparseContScaler()
    encoded = scaler.fit_transform(df)

    assert np.all(encoded[:, 0] == 1.0)  # all present
    assert np.abs(encoded[:, 1].mean()) < 1e-10  # standardized mean ~0
    assert np.abs(encoded[:, 1].std() - 1.0) < 1e-10  # standardized std ~1


def test_sparse_cont_scaler_train_test_consistency():
    """Test that test data is scaled using training statistics."""
    df_train = pd.DataFrame({"value": [1.0, 2.0, np.nan, 4.0, 5.0]})
    df_test = pd.DataFrame({"value": [3.0, np.nan, 6.0]})

    scaler = SparseContScaler()
    scaler.fit(df_train)
    encoded_test = scaler.transform(df_test)

    # Check scaling uses train statistics
    train_scaler = scaler.scalers_["value"]
    expected_scaled_3 = (3.0 - train_scaler.mean_) / train_scaler.scale_
    assert np.abs(encoded_test[0, 1] - expected_scaled_3) < 1e-10


def test_sparse_cont_scaler_in_pipeline():
    """Test SparseContScaler in sklearn Pipeline."""
    df = pd.DataFrame({"sparse_feature": [1.0, 2.0, np.nan, 4.0, np.nan, 6.0]})

    pipeline = Pipeline([("sparse_scaler", SparseContScaler())])
    encoded = pipeline.fit_transform(df)

    assert encoded.shape == (6, 2)
    expected_indicator = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    np.testing.assert_array_equal(encoded[:, 0], expected_indicator)


def test_sparse_cont_scaler_in_column_transformer():
    """Test SparseContScaler in ColumnTransformer with mixed feature types."""
    df = pd.DataFrame(
        {
            "sparse_feature": [1.0, np.nan, 3.0, 4.0],
            "dense_feature": [10.0, 20.0, 30.0, 40.0],
        }
    )

    preprocessor = ColumnTransformer(
        [
            ("sparse", SparseContScaler(), ["sparse_feature"]),
            ("dense", StandardScaler(), ["dense_feature"]),
        ]
    )

    encoded = preprocessor.fit_transform(df)

    # 2 from sparse (indicator + value) + 1 from dense
    assert encoded.shape == (4, 3)
    expected_indicator = np.array([1.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(encoded[:, 0], expected_indicator)


def test_sparse_cont_scaler_only_uses_present_values():
    """Test that scaling statistics are computed only from non-missing values."""
    df = pd.DataFrame({"value": [10.0, 20.0, np.nan, 30.0, np.nan, 40.0]})

    scaler = SparseContScaler()
    scaler.fit(df)

    fitted_scaler = scaler.scalers_["value"]
    expected_mean = np.mean([10.0, 20.0, 30.0, 40.0])
    expected_std = np.std([10.0, 20.0, 30.0, 40.0], ddof=0)

    assert np.abs(fitted_scaler.mean_[0] - expected_mean) < 1e-10
    assert np.abs(fitted_scaler.scale_[0] - expected_std) < 1e-10


def test_sparse_cont_scaler_transform_before_fit():
    """Test that transform before fit raises error."""
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    scaler = SparseContScaler()

    with pytest.raises(KeyError):
        scaler.transform(df)


def test_sparse_cont_scaler_get_feature_names_out():
    """Test get_feature_names_out method."""
    scaler = SparseContScaler()

    # Single column
    output = scaler.get_feature_names_out(["sparse_feature"])
    expected = np.array(["is_sparse_feature", "value_sparse_feature"], dtype=object)
    np.testing.assert_array_equal(output, expected)

    # Multiple columns
    output = scaler.get_feature_names_out(["kinetic_constant", "rate"])
    expected = np.array(
        ["is_kinetic_constant", "value_kinetic_constant", "is_rate", "value_rate"],
        dtype=object,
    )
    np.testing.assert_array_equal(output, expected)

    # Error when no input_features and not fitted
    with pytest.raises((ValueError, AttributeError)):
        scaler.get_feature_names_out()
