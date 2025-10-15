"""Tests for encoding functions (config_to_column_transformer, encode_dataframe, etc.)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from napistu_torch.load.constants import (
    ENCODING_MANAGER,
    ENCODING_MANAGER_TABLE,
    ENCODINGS,
)
from napistu_torch.load.encoding import (
    _get_feature_names,
    auto_encode,
    config_to_column_transformer,
    encode_dataframe,
)


def test_config_to_column_transformer(valid_encoding_config):
    """Test conversion of config to ColumnTransformer."""
    preprocessor = config_to_column_transformer(valid_encoding_config)

    # Check it's a ColumnTransformer
    assert isinstance(preprocessor, ColumnTransformer)

    # Check transformers were added correctly
    assert len(preprocessor.transformers) == 2

    # Check transformer names and types
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert set(transformer_names) == {ENCODINGS.CATEGORICAL, ENCODINGS.NUMERIC}


def test_config_to_column_transformer_invalid_config():
    """Test that invalid config raises error in ColumnTransformer creation."""
    invalid_config = {
        "bad": {
            ENCODING_MANAGER.COLUMNS: ["a", "b"],
            ENCODING_MANAGER.TRANSFORMER: OneHotEncoder(),
        },
        "worse": {
            ENCODING_MANAGER.COLUMNS: ["b", "c"],  # 'b' conflicts
            ENCODING_MANAGER.TRANSFORMER: StandardScaler(),
        },
    }

    with pytest.raises(ValueError, match="Column conflicts"):
        config_to_column_transformer(invalid_config)


def test_get_feature_names_unfitted():
    """Test that _get_feature_names raises error for unfitted preprocessor."""

    config = {"cat": {"columns": ["node_type"], "transformer": OneHotEncoder()}}
    preprocessor = config_to_column_transformer(config)

    with pytest.raises(ValueError, match="ColumnTransformer must be fitted first"):
        _get_feature_names(preprocessor)


def test_get_feature_names_fitted(valid_encoding_config, simple_raw_graph_df):
    """Test _get_feature_names with fitted ColumnTransformer."""

    preprocessor = config_to_column_transformer(valid_encoding_config)

    # Fit the preprocessor
    preprocessor.fit(simple_raw_graph_df)

    # Get feature names
    feature_names = _get_feature_names(preprocessor)

    # Check we get a list of strings
    assert isinstance(feature_names, list)
    assert all(isinstance(name, str) for name in feature_names)

    # Should have features for both categorical and numeric transforms
    cat_features = [name for name in feature_names if name.startswith("categorical__")]
    num_features = [name for name in feature_names if name.startswith("numeric__")]

    assert len(cat_features) > 0  # OneHotEncoder creates multiple features
    assert len(num_features) == 2  # StandardScaler preserves column count


def test_encode_dataframe_basic(valid_encoding_config, simple_raw_graph_df):
    """Test basic functionality of encode_dataframe with default config only."""
    # Encode the DataFrame
    encoded_array, feature_names = encode_dataframe(
        simple_raw_graph_df, valid_encoding_config
    )

    # Check return types
    assert isinstance(encoded_array, np.ndarray)
    assert isinstance(feature_names, list)

    # Check array properties
    assert encoded_array.shape[0] == simple_raw_graph_df.shape[0]  # Same number of rows
    assert encoded_array.shape[1] == len(feature_names)  # Columns match feature names

    # Check feature names format (should follow sklearn convention)
    assert all(
        "__" in name for name in feature_names
    ), f"Feature names should contain '__': {feature_names}"

    # Check we have both categorical and numeric features
    cat_features = [name for name in feature_names if name.startswith("categorical__")]
    num_features = [name for name in feature_names if name.startswith("numeric__")]

    assert len(cat_features) > 0, f"Expected categorical features, got: {cat_features}"
    assert (
        len(num_features) == 2
    ), f"Expected 2 numeric features, got {len(num_features)}: {num_features}"


def test_encode_dataframe_with_overrides(
    valid_encoding_config, override_encoding_config, simple_raw_graph_df
):
    """Test encode_dataframe with override configuration."""
    # Encode with overrides
    encoded_array, feature_names = encode_dataframe(
        simple_raw_graph_df,
        valid_encoding_config,
        override_encoding_config,
        verbose=True,
    )

    # Check return types
    assert isinstance(encoded_array, np.ndarray)
    assert isinstance(feature_names, list)

    # Check array properties
    assert encoded_array.shape[0] == simple_raw_graph_df.shape[0]  # Same number of rows
    assert encoded_array.shape[1] == len(feature_names)  # Columns match feature names

    # Check feature names format
    assert all(
        "__" in name for name in feature_names
    ), f"Feature names should contain '__': {feature_names}"

    # Should have features from all three transforms: categorical, numeric, embeddings
    cat_features = [name for name in feature_names if name.startswith("categorical__")]
    num_features = [name for name in feature_names if name.startswith("numeric__")]
    emb_features = [name for name in feature_names if name.startswith("embeddings__")]

    assert len(cat_features) > 0, f"Expected categorical features, got: {cat_features}"
    assert (
        len(num_features) == 2
    ), f"Expected 2 numeric features, got {len(num_features)}: {num_features}"
    assert (
        len(emb_features) > 0
    ), f"Expected embedding features, got: {emb_features}"  # OneHotEncoder creates multiple features


def test_encode_dataframe_missing_columns():
    """Test that encode_dataframe raises error for missing columns."""
    config = {
        "categorical": {
            ENCODING_MANAGER.COLUMNS: ["missing_column"],
            ENCODING_MANAGER.TRANSFORMER: OneHotEncoder(sparse_output=False),
        }
    }

    df = pd.DataFrame({"existing_column": [1, 2, 3]})

    with pytest.raises(KeyError, match="Missing columns in DataFrame"):
        encode_dataframe(df, config)


def test_encode_dataframe_empty_dataframe():
    """Test that encode_dataframe raises error for empty DataFrame."""
    config = {
        "categorical": {
            ENCODING_MANAGER.COLUMNS: ["col1"],
            ENCODING_MANAGER.TRANSFORMER: OneHotEncoder(sparse_output=False),
        }
    }

    # Empty DataFrame with required column
    df = pd.DataFrame({"col1": []})

    with pytest.raises(ValueError, match="Cannot fit encoders on empty DataFrame"):
        encode_dataframe(df, config)


def test_auto_encode_mixed_types():
    """Test auto_encode correctly assigns encodings to different column types."""
    # Create DataFrame with various column types
    test_df = pd.DataFrame(
        {
            "binary_col": [0, 1, 0, 1, 0, 1],  # Binary (0/1 no NaN)
            "binary_with_nan_col": [
                0,
                1,
                np.nan,
                1,
                0,
                np.nan,
            ],  # Binary with NaN (becomes categorical)
            "boolean_col": [True, False, True, False, True, False],  # Boolean dtype
            "categorical_col": ["A", "B", "A", "C", "B", "A"],  # Categorical (object)
            "categorical_with_nan_col": [
                "X",
                "Y",
                np.nan,
                "Z",
                "Y",
                np.nan,
            ],  # Categorical with NaN
            "numeric_col": [1.5, 2.3, 4.1, 5.2, 3.8, 2.9],  # Numeric (no NaN)
            "sparse_numeric_col": [
                1.0,
                np.nan,
                np.nan,
                4.0,
                np.nan,
                np.nan,
            ],  # Sparse numeric (>50% NaN)
            "preencoded_col": [10, 20, 30, 40, 50, 60],  # Already in existing config
        }
    )

    # Existing encodings that already handle "preencoded_col"
    existing_encodings = {ENCODINGS.NUMERIC: ["preencoded_col"]}

    # Run auto_encode
    result_manager = auto_encode(test_df, existing_encodings)

    # Get the encoding table
    table = result_manager.get_encoding_table()

    # Define expected column -> encoding mappings
    expected_encodings = {
        "binary_col": ENCODINGS.BINARY,
        "binary_with_nan_col": ENCODINGS.CATEGORICAL,
        "boolean_col": ENCODINGS.BINARY,
        "categorical_col": ENCODINGS.CATEGORICAL,
        "categorical_with_nan_col": ENCODINGS.CATEGORICAL,
        "numeric_col": ENCODINGS.NUMERIC,
        "sparse_numeric_col": ENCODINGS.SPARSE_NUMERIC,
        "preencoded_col": ENCODINGS.NUMERIC,
    }

    # Check each column is assigned to the correct encoding type
    for column, expected_encoding in expected_encodings.items():
        rows = table[table[ENCODING_MANAGER_TABLE.COLUMN] == column]
        assert len(rows) == 1, f"Expected 1 row for {column}, got {len(rows)}"
        assert rows.iloc[0][ENCODING_MANAGER_TABLE.TRANSFORM_NAME] == expected_encoding
