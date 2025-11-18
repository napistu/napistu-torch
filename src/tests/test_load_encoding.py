"""Tests for encoding functions (config_to_column_transformer, encode_dataframe, etc.)."""

import numpy as np
import pandas as pd
import pytest
from napistu.network.constants import NAPISTU_GRAPH_VERTICES
from napistu.ontologies.constants import SPECIES_TYPES
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
    compose_encoding_configs,
    config_to_column_transformer,
    deduplicate_features,
    encode_dataframe,
    expand_deduplicated_features,
)
from napistu_torch.load.encoding_manager import EncodingManager


def test_compose_encoding_configs(valid_encoding_config, override_encoding_config):
    """Test compose_encoding_configs helper function."""
    # Test with no overrides (should return defaults)
    result = compose_encoding_configs(valid_encoding_config)
    assert isinstance(result, EncodingManager)
    assert set(result.config_.keys()) == {ENCODINGS.CATEGORICAL, ENCODINGS.NUMERIC}

    # Test with overrides (should merge configs)
    result = compose_encoding_configs(valid_encoding_config, override_encoding_config)
    assert isinstance(result, EncodingManager)
    assert set(result.config_.keys()) == {
        ENCODINGS.CATEGORICAL,
        ENCODINGS.NUMERIC,
        "embeddings",
    }

    # Verify override took precedence for categorical transform
    assert (
        "node_type" in result.config_[ENCODINGS.CATEGORICAL][ENCODING_MANAGER.COLUMNS]
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
    encoded_array, feature_names, feature_aliases = encode_dataframe(
        simple_raw_graph_df, valid_encoding_config, deduplicate=False
    )

    # Check return types
    assert isinstance(encoded_array, np.ndarray)
    assert isinstance(feature_names, list)
    assert isinstance(feature_aliases, dict)

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

    # Check feature_aliases is empty when no deduplication needed
    assert feature_aliases == {}


def test_encode_dataframe_with_overrides(
    valid_encoding_config, override_encoding_config, simple_raw_graph_df
):
    """Test encode_dataframe with override configuration."""
    # Encode with overrides
    encoded_array, feature_names, feature_aliases = encode_dataframe(
        simple_raw_graph_df,
        valid_encoding_config,
        override_encoding_config,
        verbose=True,
        deduplicate=False,
    )

    # Check return types
    assert isinstance(encoded_array, np.ndarray)
    assert isinstance(feature_names, list)
    assert isinstance(feature_aliases, dict)

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

    # Check feature_aliases is empty when no deduplication needed
    assert feature_aliases == {}


def test_encode_dataframe_with_deduplication():
    """Test encode_dataframe with duplicate features that require deduplication."""
    # Create DataFrame with duplicate columns that will produce identical encoded features
    df = pd.DataFrame(
        {
            "cat_col_x": ["A", "B", "A", "B"],
            "cat_col_y": ["A", "B", "A", "B"],  # Same values as cat_col_x
            "num_col": [1.0, 2.0, 3.0, 4.0],
        }
    )

    # Config that will create identical one-hot encoded columns
    config = {
        "categorical": {
            ENCODING_MANAGER.COLUMNS: ["cat_col_x", "cat_col_y"],
            ENCODING_MANAGER.TRANSFORMER: OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            ),
        },
        "numeric": {
            ENCODING_MANAGER.COLUMNS: ["num_col"],
            ENCODING_MANAGER.TRANSFORMER: StandardScaler(),
        },
    }

    encoded_array, feature_names, feature_aliases = encode_dataframe(df, config)

    # Check return types
    assert isinstance(encoded_array, np.ndarray)
    assert isinstance(feature_names, list)
    assert isinstance(feature_aliases, dict)

    # Check that duplicates were removed (should have fewer columns than if not deduplicated)
    # OneHotEncoder on 2 identical columns creates 2 identical one-hot feature sets
    # After deduplication, should only keep one canonical name
    assert encoded_array.shape[0] == df.shape[0]

    # Should have categorical features (deduplicated) + numeric features
    cat_features = [name for name in feature_names if name.startswith("categorical__")]
    num_features = [name for name in feature_names if name.startswith("numeric__")]

    assert len(cat_features) > 0
    assert len(num_features) == 1

    # Check that feature_aliases contains mappings for deduplicated features
    # The canonical name should use the shortest common prefix
    assert len(feature_aliases) > 0

    # Check that aliases point to canonical names that exist in feature_names
    for alias_name, canonical_name in feature_aliases.items():
        assert canonical_name in feature_names
        # Both names should come from the same transform (categorical)
        assert alias_name.split("__")[0] == canonical_name.split("__")[0]


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
        "binary_with_nan_col": ENCODINGS.SPARSE_CATEGORICAL,
        "boolean_col": ENCODINGS.BINARY,
        "categorical_col": ENCODINGS.CATEGORICAL,
        "categorical_with_nan_col": ENCODINGS.SPARSE_CATEGORICAL,
        "numeric_col": ENCODINGS.NUMERIC,
        "sparse_numeric_col": ENCODINGS.SPARSE_NUMERIC,
        "preencoded_col": ENCODINGS.NUMERIC,
    }

    # Check each column is assigned to the correct encoding type
    for column, expected_encoding in expected_encodings.items():
        rows = table[table[ENCODING_MANAGER_TABLE.COLUMN] == column]
        assert len(rows) == 1, f"Expected 1 row for {column}, got {len(rows)}"
        assert rows.iloc[0][ENCODING_MANAGER_TABLE.TRANSFORM_NAME] == expected_encoding


def test_deduplicate_features():
    """Test deduplicate_features with standalone array."""
    # Array with duplicate columns (columns 0 and 1 are identical)
    array = np.array([[1, 1, 0, 2], [2, 2, 1, 3], [3, 3, 0, 1]])
    names = ["is_string_x", "is_string_y", "value_weight", "unique_feature"]

    pruned, canonical, aliases = deduplicate_features(array, names)

    # Check array shape - should have removed 1 duplicate column
    assert pruned.shape == (3, 3)
    assert len(canonical) == 3

    # Check canonical names - duplicates should use common prefix
    assert "is_string" in canonical
    assert "value_weight" in canonical
    assert "unique_feature" in canonical

    # Check alias mapping
    assert "is_string_y" in aliases
    assert aliases["is_string_y"] == "is_string"

    # Verify pruned array has correct values (first duplicate kept)
    np.testing.assert_array_equal(pruned[:, canonical.index("is_string")], array[:, 0])
    np.testing.assert_array_equal(
        pruned[:, canonical.index("value_weight")], array[:, 2]
    )
    np.testing.assert_array_equal(
        pruned[:, canonical.index("unique_feature")], array[:, 3]
    )


def test_deduplicate_features_duplicate_names():
    """Test that deduplicate_features raises error for duplicate feature names."""
    array = np.array([[1, 1, 0], [0, 0, 1]])
    names = ["feature_a", "feature_a", "feature_b"]  # Duplicate name

    with pytest.raises(ValueError, match="feature_names contains duplicates"):
        deduplicate_features(array, names)


def test_expand_deduplicated_features():
    """Test expand_deduplicated_features with standalone array."""
    # Simulate deduplicated array (after removing duplicate columns)
    deduplicated = np.array([[1, 0, 2], [2, 1, 3], [3, 0, 1]])
    canonical_names = ["is_string", "value_weight", "unique_feature"]
    aliases = {"is_string_y": "is_string"}  # One alias pointing to canonical

    expanded, expanded_names = expand_deduplicated_features(
        deduplicated, canonical_names, aliases
    )

    # Check array shape - should have added back the aliased column
    assert expanded.shape == (3, 4)  # Original had 4 columns
    assert len(expanded_names) == 4

    # Check that canonical names are preserved
    assert "is_string" in expanded_names
    assert "value_weight" in expanded_names
    assert "unique_feature" in expanded_names

    # Check that alias was added
    assert "is_string_y" in expanded_names

    # Check that aliased column is a duplicate of canonical column
    canonical_idx = expanded_names.index("is_string")
    alias_idx = expanded_names.index("is_string_y")
    np.testing.assert_array_equal(expanded[:, canonical_idx], expanded[:, alias_idx])


def test_sparse_categorical_includes_all_categories():
    """Test that sparse categorical encoding includes all categories, including the first alphabetically."""
    # Create DataFrame with sparse categorical column that includes the first alphabetical category
    # This tests that drop=None is used (not drop="first")
    df = pd.DataFrame(
        {
            NAPISTU_GRAPH_VERTICES.SPECIES_TYPE: [
                SPECIES_TYPES.COMPLEX,  # First alphabetically - should NOT be dropped
                SPECIES_TYPES.DRUG,
                SPECIES_TYPES.METABOLITE,
                None,  # Missing value
                SPECIES_TYPES.OTHER,
                SPECIES_TYPES.PROTEIN,
                SPECIES_TYPES.REGULATORY_RNA,
            ]
        }
    )

    # Use DEFAULT_ENCODERS which should have drop=None for SPARSE_CATEGORICAL
    from napistu_torch.load.encoders import DEFAULT_ENCODERS

    config = {
        ENCODINGS.SPARSE_CATEGORICAL: {
            ENCODING_MANAGER.COLUMNS: [NAPISTU_GRAPH_VERTICES.SPECIES_TYPE],
            ENCODING_MANAGER.TRANSFORMER: DEFAULT_ENCODERS[
                ENCODINGS.SPARSE_CATEGORICAL
            ],
        }
    }

    encoded_array, feature_names, _ = encode_dataframe(df, config, deduplicate=False)

    # Extract category names from feature names
    # Feature names format: "sparse_categorical__species_type_<category>"
    category_features = [
        name
        for name in feature_names
        if name.startswith("sparse_categorical__species_type_")
    ]
    categories = [name.split("__species_type_")[1] for name in category_features]

    # Verify all non-null categories are included
    expected_categories = {
        SPECIES_TYPES.COMPLEX,
        SPECIES_TYPES.DRUG,
        SPECIES_TYPES.METABOLITE,
        SPECIES_TYPES.OTHER,
        SPECIES_TYPES.PROTEIN,
        SPECIES_TYPES.REGULATORY_RNA,
    }
    actual_categories = set(categories)

    # Check that all expected categories are present
    missing_categories = expected_categories - actual_categories
    assert (
        not missing_categories
    ), f"Missing categories in sparse categorical encoding: {missing_categories}. Got: {actual_categories}"

    # Verify "complex" (first alphabetically) is included
    assert SPECIES_TYPES.COMPLEX in actual_categories, (
        f"First alphabetical category '{SPECIES_TYPES.COMPLEX}' should be included. "
        f"Got categories: {actual_categories}"
    )

    # Verify we have the expected number of categories (6 non-null + potentially nan)
    assert len(category_features) >= len(
        expected_categories
    ), f"Expected at least {len(expected_categories)} category features, got {len(category_features)}"
