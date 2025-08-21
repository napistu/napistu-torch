import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Import functions from the main module
from napistu_torch.load import transforms
from napistu_torch.load.constants import TRANSFORM_TABLE, TRANSFORMATION
from napistu_torch.load.transforms import validate_config


class MockEmbedder:
    """Mock transformer class for testing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


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
                "new_col",
            ],  # node_type conflicts with base
            TRANSFORMATION.TRANSFORMER: LabelEncoder(),  # Different transformer than base
        },
        "embeddings": {
            TRANSFORMATION.COLUMNS: ["source_col"],
            TRANSFORMATION.TRANSFORMER: MockEmbedder(),  # Use actual transformer object
        },
    }


@pytest.fixture
def sample_data():
    """Sample data for testing transformations."""
    return pd.DataFrame(
        {
            "node_type": ["protein", "reaction", "protein", "reaction"],
            "species_type": ["human", "mouse", "human", "mouse"],
            "weight": [1.0, 2.0, 3.0, 4.0],
            "score": [0.1, 0.2, 0.3, 0.4],
            "source_col": ["src1", "src2", "src3", "src4"],
        }
    )


def test_validate_config_valid(valid_config):
    """Test validation of a valid configuration."""
    table = validate_config(valid_config)

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
        validate_config(invalid_config)


def test_compose_configs_merge_strategy(valid_config, override_config):
    """Test config composition with merge strategy."""
    composed = transforms.compose_configs(valid_config, override_config)

    # Check that composition succeeded
    assert isinstance(composed, dict)

    # Check expected transforms
    assert set(composed.keys()) == {"categorical", "numerical", "embeddings"}

    # Check categorical merge: should have species_type from base + node_type,new_col from override
    categorical_columns = set(composed["categorical"][TRANSFORMATION.COLUMNS])
    assert "species_type" in categorical_columns  # Preserved from base
    assert "node_type" in categorical_columns  # From override (wins conflict)
    assert "new_col" in categorical_columns  # From override

    # Check numerical preserved unchanged (no conflicts)
    assert composed["numerical"][TRANSFORMATION.COLUMNS] == ["weight", "score"]

    # Check new embeddings transform added
    assert composed["embeddings"][TRANSFORMATION.COLUMNS] == ["source_col"]


def test_compose_configs_verbose_logging(valid_config, override_config, caplog):
    """Test verbose logging in config composition."""
    import logging

    with caplog.at_level(logging.INFO):
        transforms.compose_configs(valid_config, override_config, verbose=True)

    # Check that conflict logging occurred
    assert "Cross-config conflicts detected" in caplog.text
    assert "node_type" in caplog.text

    # Check that final transformations were logged
    assert "Final composed transformations" in caplog.text
    assert "categorical" in caplog.text
    assert "numerical" in caplog.text
    assert "embeddings" in caplog.text


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

    composed = transforms.compose_configs(base, override)

    # Should have both transforms with no changes
    assert set(composed.keys()) == {"categorical", "numerical"}
    assert composed["categorical"][TRANSFORMATION.COLUMNS] == ["node_type"]
    assert composed["numerical"][TRANSFORMATION.COLUMNS] == ["weight"]


def test_empty_config():
    """Test validation of empty configuration."""
    empty_config = {}

    # Should validate successfully but return empty DataFrame
    table = validate_config(empty_config)
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
        validate_config(invalid_transformer_config)


def test_empty_columns_list():
    """Test validation with empty columns list."""
    empty_columns_config = {
        "bad_transform": {
            TRANSFORMATION.COLUMNS: [],  # Empty list
            TRANSFORMATION.TRANSFORMER: OneHotEncoder(),
        }
    }

    with pytest.raises(ValueError):
        validate_config(empty_columns_config)


def test_passthrough_transformer(valid_config):
    """Test that 'passthrough' is accepted as a valid transformer."""
    passthrough_config = {
        "passthrough_transform": {
            TRANSFORMATION.COLUMNS: ["node_type"],
            TRANSFORMATION.TRANSFORMER: TRANSFORMATION.PASSTHROUGH,
        }
    }

    # Should validate successfully
    table = validate_config(passthrough_config)
    assert len(table) == 1
    assert table.iloc[0][TRANSFORM_TABLE.TRANSFORMER_TYPE] == TRANSFORMATION.PASSTHROUGH


def test_config_to_column_transformer(valid_config):
    """Test conversion of config to ColumnTransformer."""
    preprocessor = transforms.config_to_column_transformer(valid_config)

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
        "bad": {"columns": ["a", "b"], "transformer": OneHotEncoder()},
        "worse": {
            "columns": ["b", "c"],  # 'b' conflicts
            "transformer": StandardScaler(),
        },
    }

    with pytest.raises(ValueError, match="Column conflicts"):
        transforms.config_to_column_transformer(invalid_config)


def test_get_feature_names_unfitted():
    """Test that get_feature_names raises error for unfitted preprocessor."""
    config = {"cat": {"columns": ["node_type"], "transformer": OneHotEncoder()}}

    preprocessor = transforms.config_to_column_transformer(config)

    with pytest.raises(ValueError, match="ColumnTransformer must be fitted first"):
        transforms.get_feature_names(preprocessor)


def test_get_feature_names_fitted(valid_config, sample_data):
    """Test get_feature_names with fitted ColumnTransformer."""
    preprocessor = transforms.config_to_column_transformer(valid_config)

    # Fit the preprocessor
    preprocessor.fit(sample_data)

    # Get feature names
    feature_names = transforms.get_feature_names(preprocessor)

    # Check we get a list of strings
    assert isinstance(feature_names, list)
    assert all(isinstance(name, str) for name in feature_names)

    # Should have features for both categorical and numerical transforms
    cat_features = [name for name in feature_names if name.startswith("categorical__")]
    num_features = [name for name in feature_names if name.startswith("numerical__")]

    assert len(cat_features) > 0  # OneHotEncoder creates multiple features
    assert len(num_features) == 2  # StandardScaler preserves column count
