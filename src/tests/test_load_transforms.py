import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from napistu_torch.load.constants import TRANSFORM_TABLE, TRANSFORMATION

# Import classes and functions from the main module
from napistu_torch.load.transforms import (
    EncodingManager,
    _get_feature_names,
    config_to_column_transformer,
    encode_dataframe,
)


@pytest.fixture
def valid_config():
    """Valid configuration without conflicts."""
    return {
        "categorical": {
            TRANSFORMATION.COLUMNS: ["node_type", "species_type"],
            TRANSFORMATION.TRANSFORMER: OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            ),
        },
        "numerical": {
            TRANSFORMATION.COLUMNS: ["weight", "score"],
            TRANSFORMATION.TRANSFORMER: StandardScaler(),
        },
    }


@pytest.fixture
def invalid_config():
    """Invalid configuration with column conflicts."""
    return {
        "cat": {
            TRANSFORMATION.COLUMNS: ["node_type", "weight"],
            TRANSFORMATION.TRANSFORMER: OneHotEncoder(),
        },
        "num": {
            TRANSFORMATION.COLUMNS: ["weight", "score"],  # 'weight' conflict
            TRANSFORMATION.TRANSFORMER: StandardScaler(),
        },
    }


@pytest.fixture
def override_config():
    """Override configuration for composition tests."""
    return {
        "categorical": {
            TRANSFORMATION.COLUMNS: [
                "node_type",
                "species_type",  # Use existing column instead of "new_col"
            ],  # node_type conflicts with base
            TRANSFORMATION.TRANSFORMER: OneHotEncoder(
                sparse_output=False
            ),  # Different transformer than base
        },
        "embeddings": {
            TRANSFORMATION.COLUMNS: ["source_col"],
            TRANSFORMATION.TRANSFORMER: OneHotEncoder(
                sparse_output=False
            ),  # Use real transformer
        },
    }


@pytest.fixture
def sample_data():
    """Sample data for testing transformations."""
    return pd.DataFrame(
        {
            "node_type": ["species", "reaction", "species", "reaction"],
            "species_type": ["gene", None, "metabolite", None],
            "weight": [1.0, 2.0, 3.0, 4.0],
            "score": [0.1, 0.2, 0.3, 0.4],
            "source_col": ["src1", "src2", "src3", "src4"],
        }
    )


def test_validate_config_valid(valid_config):
    """Test validation of a valid configuration."""
    # Use EncodingManager instead of validate_config function
    manager = EncodingManager(valid_config)
    table = manager.get_transform_table()

    # Check that we get a DataFrame
    assert isinstance(table, pd.DataFrame)

    # Check expected columns
    assert set(table.columns) == {
        TRANSFORM_TABLE.TRANSFORM_NAME,
        TRANSFORM_TABLE.COLUMN,
        TRANSFORM_TABLE.TRANSFORMER_TYPE,
    }

    # Check expected rows
    assert len(table) == 4  # 2 categorical + 2 numerical columns
    assert set(table[TRANSFORM_TABLE.COLUMN]) == {
        "node_type",
        "species_type",
        "weight",
        "score",
    }
    assert set(table[TRANSFORM_TABLE.TRANSFORM_NAME]) == {"categorical", "numerical"}


def test_validate_config_invalid(invalid_config):
    """Test validation of an invalid configuration with conflicts."""
    with pytest.raises(ValueError, match="Column conflicts"):
        EncodingManager(invalid_config)


def test_compose_configs_merge_strategy(valid_config, override_config):
    """Test config composition with merge strategy."""
    # Use EncodingManager.compose instead of compose_configs function
    base_manager = EncodingManager(valid_config)
    override_manager = EncodingManager(override_config)
    composed = base_manager.compose(override_manager)

    # Check that composition succeeded
    assert isinstance(composed, EncodingManager)

    # Check expected transforms
    assert set(composed.config_.keys()) == {"categorical", "numerical", "embeddings"}

    # Check categorical merge: should have species_type from base + node_type from override
    categorical_columns = set(composed.config_["categorical"][TRANSFORMATION.COLUMNS])
    assert "species_type" in categorical_columns  # Preserved from base
    assert "node_type" in categorical_columns  # From override (wins conflict)

    # Check numerical preserved unchanged (no conflicts)
    assert composed.config_["numerical"][TRANSFORMATION.COLUMNS] == ["weight", "score"]

    # Check new embeddings transform added
    assert composed.config_["embeddings"][TRANSFORMATION.COLUMNS] == ["source_col"]


def test_compose_configs_verbose_logging(valid_config, override_config, caplog):
    """Test verbose logging in config composition."""
    import logging

    base_manager = EncodingManager(valid_config)
    override_manager = EncodingManager(override_config)

    with caplog.at_level(logging.INFO):
        base_manager.compose(override_manager, verbose=True)

    # Check that conflict logging occurred
    assert "Cross-config conflicts detected" in caplog.text
    assert "node_type" in caplog.text


def test_compose_configs_no_conflicts():
    """Test config composition with no cross-config conflicts."""
    base = {
        "categorical": {
            TRANSFORMATION.COLUMNS: ["node_type"],
            TRANSFORMATION.TRANSFORMER: OneHotEncoder(),
        }
    }

    override = {
        "numerical": {
            TRANSFORMATION.COLUMNS: ["weight"],
            TRANSFORMATION.TRANSFORMER: StandardScaler(),
        }
    }

    base_manager = EncodingManager(base)
    override_manager = EncodingManager(override)
    composed = base_manager.compose(override_manager)

    # Should have both transforms with no changes
    assert set(composed.config_.keys()) == {"categorical", "numerical"}
    assert composed.config_["categorical"][TRANSFORMATION.COLUMNS] == ["node_type"]
    assert composed.config_["numerical"][TRANSFORMATION.COLUMNS] == ["weight"]


def test_empty_config():
    """Test validation of empty configuration."""
    empty_config = {}

    # Should validate successfully but return empty DataFrame
    manager = EncodingManager(empty_config)
    table = manager.get_transform_table()
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 0


def test_invalid_transformer():
    """Test validation with invalid transformer object."""
    invalid_transformer_config = {
        "bad_transform": {
            TRANSFORMATION.COLUMNS: ["node_type"],
            TRANSFORMATION.TRANSFORMER: "not_a_transformer",  # String without fit/transform methods
        }
    }

    with pytest.raises(ValueError, match="transformer must have fit/transform methods"):
        EncodingManager(invalid_transformer_config)


def test_empty_columns_list():
    """Test validation with empty columns list."""
    empty_columns_config = {
        "bad_transform": {
            TRANSFORMATION.COLUMNS: [],  # Empty list
            TRANSFORMATION.TRANSFORMER: OneHotEncoder(),
        }
    }

    with pytest.raises(ValueError):
        EncodingManager(empty_columns_config)


def test_passthrough_transformer(valid_config):
    """Test that 'passthrough' is accepted as a valid transformer."""
    passthrough_config = {
        "passthrough_transform": {
            TRANSFORMATION.COLUMNS: ["node_type"],
            TRANSFORMATION.TRANSFORMER: TRANSFORMATION.PASSTHROUGH,
        }
    }

    # Should validate successfully
    manager = EncodingManager(passthrough_config)
    table = manager.get_transform_table()
    assert len(table) == 1
    assert table.iloc[0][TRANSFORM_TABLE.TRANSFORMER_TYPE] == TRANSFORMATION.PASSTHROUGH


def test_config_to_column_transformer(valid_config):
    """Test conversion of config to ColumnTransformer."""
    preprocessor = config_to_column_transformer(valid_config)

    # Check it's a ColumnTransformer
    assert isinstance(preprocessor, ColumnTransformer)

    # Check transformers were added correctly
    assert len(preprocessor.transformers) == 2

    # Check transformer names and types
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert set(transformer_names) == {"categorical", "numerical"}


def test_config_to_column_transformer_invalid_config():
    """Test that invalid config raises error in ColumnTransformer creation."""
    invalid_config = {
        "bad": {
            TRANSFORMATION.COLUMNS: ["a", "b"],
            TRANSFORMATION.TRANSFORMER: OneHotEncoder(),
        },
        "worse": {
            TRANSFORMATION.COLUMNS: ["b", "c"],  # 'b' conflicts
            TRANSFORMATION.TRANSFORMER: StandardScaler(),
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


def test_get_feature_names_fitted(valid_config, sample_data):
    """Test _get_feature_names with fitted ColumnTransformer."""

    preprocessor = config_to_column_transformer(valid_config)

    # Fit the preprocessor
    preprocessor.fit(sample_data)

    # Get feature names
    feature_names = _get_feature_names(preprocessor)

    # Check we get a list of strings
    assert isinstance(feature_names, list)
    assert all(isinstance(name, str) for name in feature_names)

    # Should have features for both categorical and numerical transforms
    cat_features = [name for name in feature_names if name.startswith("categorical__")]
    num_features = [name for name in feature_names if name.startswith("numerical__")]

    assert len(cat_features) > 0  # OneHotEncoder creates multiple features
    assert len(num_features) == 2  # StandardScaler preserves column count


def test_encode_dataframe_basic(valid_config, sample_data):
    """Test basic functionality of encode_dataframe with default config only."""
    # Encode the DataFrame
    encoded_array, feature_names = encode_dataframe(sample_data, valid_config)

    # Check return types
    assert isinstance(encoded_array, np.ndarray)
    assert isinstance(feature_names, list)

    # Check array properties
    assert encoded_array.shape[0] == sample_data.shape[0]  # Same number of rows
    assert encoded_array.shape[1] == len(feature_names)  # Columns match feature names

    # Check feature names format (should follow sklearn convention)
    assert all(
        "__" in name for name in feature_names
    ), f"Feature names should contain '__': {feature_names}"

    # Check we have both categorical and numerical features
    cat_features = [name for name in feature_names if name.startswith("categorical__")]
    num_features = [name for name in feature_names if name.startswith("numerical__")]

    assert len(cat_features) > 0, f"Expected categorical features, got: {cat_features}"
    assert (
        len(num_features) == 2
    ), f"Expected 2 numerical features, got {len(num_features)}: {num_features}"


def test_encode_dataframe_with_overrides(valid_config, override_config, sample_data):
    """Test encode_dataframe with override configuration."""
    # Encode with overrides
    encoded_array, feature_names = encode_dataframe(
        sample_data, valid_config, override_config, verbose=True
    )

    # Check return types
    assert isinstance(encoded_array, np.ndarray)
    assert isinstance(feature_names, list)

    # Check array properties
    assert encoded_array.shape[0] == sample_data.shape[0]  # Same number of rows
    assert encoded_array.shape[1] == len(feature_names)  # Columns match feature names

    # Check feature names format
    assert all(
        "__" in name for name in feature_names
    ), f"Feature names should contain '__': {feature_names}"

    # Should have features from all three transforms: categorical, numerical, embeddings
    cat_features = [name for name in feature_names if name.startswith("categorical__")]
    num_features = [name for name in feature_names if name.startswith("numerical__")]
    emb_features = [name for name in feature_names if name.startswith("embeddings__")]

    assert len(cat_features) > 0, f"Expected categorical features, got: {cat_features}"
    assert (
        len(num_features) == 2
    ), f"Expected 2 numerical features, got {len(num_features)}: {num_features}"
    assert (
        len(emb_features) > 0
    ), f"Expected embedding features, got: {emb_features}"  # OneHotEncoder creates multiple features


def test_encode_dataframe_missing_columns():
    """Test that encode_dataframe raises error for missing columns."""
    config = {
        "categorical": {
            TRANSFORMATION.COLUMNS: ["missing_column"],
            TRANSFORMATION.TRANSFORMER: OneHotEncoder(sparse_output=False),
        }
    }

    df = pd.DataFrame({"existing_column": [1, 2, 3]})

    with pytest.raises(KeyError, match="Missing columns in DataFrame"):
        encode_dataframe(df, config)


def test_encode_dataframe_empty_dataframe():
    """Test that encode_dataframe raises error for empty DataFrame."""
    config = {
        "categorical": {
            TRANSFORMATION.COLUMNS: ["col1"],
            TRANSFORMATION.TRANSFORMER: OneHotEncoder(sparse_output=False),
        }
    }

    # Empty DataFrame with required column
    df = pd.DataFrame({"col1": []})

    with pytest.raises(ValueError, match="Cannot encode empty DataFrame"):
        encode_dataframe(df, config)
