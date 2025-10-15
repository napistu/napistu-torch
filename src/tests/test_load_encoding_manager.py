"""Tests for EncodingManager class and related functionality."""

import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from napistu_torch.load.constants import ENCODING_MANAGER, ENCODING_MANAGER_TABLE
from napistu_torch.load.encoding_manager import EncodingManager


def test_validate_config_valid(valid_encoding_config):
    """Test validation of a valid configuration."""
    # Use EncodingManager instead of validate_config function
    manager = EncodingManager(valid_encoding_config)
    table = manager.get_encoding_table()

    # Check that we get a DataFrame
    assert isinstance(table, pd.DataFrame)

    # Check expected columns
    assert set(table.columns) == {
        ENCODING_MANAGER_TABLE.TRANSFORM_NAME,
        ENCODING_MANAGER_TABLE.COLUMN,
        ENCODING_MANAGER_TABLE.TRANSFORMER_TYPE,
    }

    # Check expected rows
    assert len(table) == 4  # 2 categorical + 2 numerical columns
    assert set(table[ENCODING_MANAGER_TABLE.COLUMN]) == {
        "node_type",
        "species_type",
        "weight",
        "score",
    }
    assert set(table[ENCODING_MANAGER_TABLE.TRANSFORM_NAME]) == {
        "categorical",
        "numerical",
    }


def test_validate_config_invalid(invalid_encoding_config):
    """Test validation of an invalid configuration with conflicts."""
    with pytest.raises(ValueError, match="Column conflicts"):
        EncodingManager(invalid_encoding_config)


def test_compose_configs_merge_strategy(
    valid_encoding_config, override_encoding_config
):
    """Test config composition with merge strategy."""
    # Use EncodingManager.compose instead of compose_configs function
    base_manager = EncodingManager(valid_encoding_config)
    override_manager = EncodingManager(override_encoding_config)
    composed = base_manager.compose(override_manager)

    # Check that composition succeeded
    assert isinstance(composed, EncodingManager)

    # Check expected transforms
    assert set(composed.config_.keys()) == {"categorical", "numerical", "embeddings"}

    # Check categorical merge: should have species_type from base + node_type from override
    categorical_columns = set(composed.config_["categorical"][ENCODING_MANAGER.COLUMNS])
    assert "species_type" in categorical_columns  # Preserved from base
    assert "node_type" in categorical_columns  # From override (wins conflict)

    # Check numerical preserved unchanged (no conflicts)
    assert composed.config_["numerical"][ENCODING_MANAGER.COLUMNS] == [
        "weight",
        "score",
    ]

    # Check new embeddings transform added
    assert composed.config_["embeddings"][ENCODING_MANAGER.COLUMNS] == ["source_col"]


def test_compose_configs_verbose_logging(
    valid_encoding_config, override_encoding_config, caplog
):
    """Test verbose logging in config composition."""
    import logging

    base_manager = EncodingManager(valid_encoding_config)
    override_manager = EncodingManager(override_encoding_config)

    with caplog.at_level(logging.INFO):
        base_manager.compose(override_manager, verbose=True)

    # Check that conflict logging occurred
    assert "Cross-config conflicts detected" in caplog.text
    assert "node_type" in caplog.text


def test_compose_configs_no_conflicts():
    """Test config composition with no cross-config conflicts."""
    base = {
        "categorical": {
            ENCODING_MANAGER.COLUMNS: ["node_type"],
            ENCODING_MANAGER.TRANSFORMER: OneHotEncoder(),
        }
    }

    override = {
        "numerical": {
            ENCODING_MANAGER.COLUMNS: ["weight"],
            ENCODING_MANAGER.TRANSFORMER: StandardScaler(),
        }
    }

    base_manager = EncodingManager(base)
    override_manager = EncodingManager(override)
    composed = base_manager.compose(override_manager)

    # Should have both transforms with no changes
    assert set(composed.config_.keys()) == {"categorical", "numerical"}
    assert composed.config_["categorical"][ENCODING_MANAGER.COLUMNS] == ["node_type"]
    assert composed.config_["numerical"][ENCODING_MANAGER.COLUMNS] == ["weight"]


def test_empty_config():
    """Test validation of empty configuration."""
    empty_config = {}

    # Should validate successfully but return empty DataFrame
    manager = EncodingManager(empty_config)
    table = manager.get_encoding_table()
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 0


def test_invalid_transformer():
    """Test validation with invalid transformer object."""
    invalid_transformer_config = {
        "bad_transform": {
            ENCODING_MANAGER.COLUMNS: ["node_type"],
            ENCODING_MANAGER.TRANSFORMER: "not_a_transformer",  # String without fit/transform methods
        }
    }

    with pytest.raises(ValueError, match="transformer must have fit/transform methods"):
        EncodingManager(invalid_transformer_config)


def test_empty_columns_list():
    """Test validation with empty columns list."""
    empty_columns_config = {
        "bad_transform": {
            ENCODING_MANAGER.COLUMNS: [],  # Empty list
            ENCODING_MANAGER.TRANSFORMER: OneHotEncoder(),
        }
    }

    with pytest.raises(ValueError):
        EncodingManager(empty_columns_config)


def test_passthrough_transformer(valid_encoding_config):
    """Test that 'passthrough' is accepted as a valid transformer."""
    passthrough_config = {
        "passthrough_transform": {
            ENCODING_MANAGER.COLUMNS: ["node_type"],
            ENCODING_MANAGER.TRANSFORMER: ENCODING_MANAGER.PASSTHROUGH,
        }
    }

    # Should validate successfully
    manager = EncodingManager(passthrough_config)
    table = manager.get_encoding_table()
    assert len(table) == 1
    assert (
        table.iloc[0][ENCODING_MANAGER_TABLE.TRANSFORMER_TYPE]
        == ENCODING_MANAGER.PASSTHROUGH
    )


def test_encoding_manager_init_simple_format():
    """Test EncodingManager.__init__() with simple format using encoders parameter."""
    simple_spec = {
        "categorical": {"node_type", "species_type"},
        "numerical": {"weight"},
    }

    encoders = {
        "categorical": OneHotEncoder(sparse_output=False),
        "numerical": StandardScaler(),
    }

    # Create manager using simple format
    manager = EncodingManager(simple_spec, encoders=encoders)

    # Check it's an EncodingManager
    assert isinstance(manager, EncodingManager)

    # Check the config was converted to complex format
    assert "categorical" in manager.config_
    assert "numerical" in manager.config_

    # Check columns are sorted lists
    assert manager.config_["categorical"]["columns"] == ["node_type", "species_type"]
    assert manager.config_["numerical"]["columns"] == ["weight"]

    # Check transformers are assigned
    assert isinstance(manager.config_["categorical"]["transformer"], OneHotEncoder)
    assert isinstance(manager.config_["numerical"]["transformer"], StandardScaler)


def test_encoding_manager_init_complex_format():
    """Test EncodingManager.__init__() with complex format (backward compatibility)."""
    complex_config = {
        "numerical": {"columns": ["value"], "transformer": StandardScaler()}
    }

    # Create manager using complex format (no encoders)
    manager = EncodingManager(complex_config)

    assert isinstance(manager, EncodingManager)
    assert manager.config_ == complex_config


def test_encoding_manager_ensure():
    """Test EncodingManager.ensure() with complex dict, simple dict, EncodingManager, and invalid types."""
    # Test complex format dict conversion
    complex_config = {
        "numerical": {"columns": ["value"], "transformer": StandardScaler()}
    }
    result_from_complex = EncodingManager.ensure(complex_config)
    assert isinstance(result_from_complex, EncodingManager)
    assert result_from_complex.config_ == complex_config

    # Test simple format dict conversion with encoders
    simple_spec = {"categorical": {"col1", "col2"}, "numerical": {"value"}}
    encoders = {
        "categorical": OneHotEncoder(sparse_output=False),
        "numerical": StandardScaler(),
    }
    result_from_simple = EncodingManager.ensure(simple_spec, encoders=encoders)
    assert isinstance(result_from_simple, EncodingManager)
    assert "categorical" in result_from_simple.config_
    assert "numerical" in result_from_simple.config_
    assert result_from_simple.config_["categorical"]["columns"] == ["col1", "col2"]
    assert result_from_simple.config_["numerical"]["columns"] == ["value"]

    # Test EncodingManager passthrough (returns same object)
    manager = EncodingManager(complex_config)
    result_from_manager = EncodingManager.ensure(manager)
    assert result_from_manager is manager

    # Test that encoders are ignored when config is already an EncodingManager
    result_from_manager_with_encoders = EncodingManager.ensure(
        manager, encoders=encoders
    )
    assert result_from_manager_with_encoders is manager

    # Test invalid types raise error
    with pytest.raises(ValueError, match="config must be a dict or an EncodingManager"):
        EncodingManager.ensure("invalid_string")

    with pytest.raises(ValueError, match="config must be a dict or an EncodingManager"):
        EncodingManager.ensure(123)


def test_encoding_manager_simple_format_with_multiple_columns():
    """Test EncodingManager with simple spec and multiple columns."""
    encoding_spec = {
        "categorical": {"node_type", "species_type"},
        "numerical": {"weight", "score"},
    }

    encoders = {
        "categorical": OneHotEncoder(sparse_output=False),
        "numerical": StandardScaler(),
    }

    # Create manager from simple spec (either via __init__ or ensure)
    manager = EncodingManager(encoding_spec, encoders=encoders)

    # Check it's an EncodingManager
    assert isinstance(manager, EncodingManager)

    # Check the config has the right structure
    assert "categorical" in manager.config_
    assert "numerical" in manager.config_

    # Check columns are sorted lists
    assert manager.config_["categorical"]["columns"] == ["node_type", "species_type"]
    assert manager.config_["numerical"]["columns"] == ["score", "weight"]

    # Check transformers are assigned
    assert isinstance(manager.config_["categorical"]["transformer"], OneHotEncoder)
    assert isinstance(manager.config_["numerical"]["transformer"], StandardScaler)

    # Check transform table
    table = manager.get_encoding_table()
    assert len(table) == 4  # 2 categorical + 2 numerical


def test_encoding_manager_simple_format_unknown_type():
    """Test that unknown encoding type raises error in simple format."""
    encoding_spec = {"unknown_type": {"col1"}}

    encoders = {"categorical": OneHotEncoder()}

    with pytest.raises(ValueError, match="Unknown encoding type: unknown_type"):
        EncodingManager(encoding_spec, encoders=encoders)


def test_encoding_manager_ensure_simple_format_unknown_type():
    """Test that ensure also raises error for unknown encoding type."""
    encoding_spec = {"unknown_type": {"col1"}}

    encoders = {"categorical": OneHotEncoder()}

    with pytest.raises(ValueError, match="Unknown encoding type: unknown_type"):
        EncodingManager.ensure(encoding_spec, encoders=encoders)
