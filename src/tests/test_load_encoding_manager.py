"""Tests for EncodingManager class and related functionality."""

import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from napistu_torch.load.constants import (
    ENCODING_MANAGER,
    ENCODING_MANAGER_TABLE,
    ENCODINGS,
)
from napistu_torch.load.encoding_manager import EncodingManager


@pytest.fixture
def invalid_encoding_config():
    """Invalid encoding configuration with column conflicts."""
    return {
        ENCODINGS.CATEGORICAL: {
            ENCODING_MANAGER.COLUMNS: ["node_type", "weight"],
            ENCODING_MANAGER.TRANSFORMER: OneHotEncoder(),
        },
        ENCODINGS.NUMERIC: {
            ENCODING_MANAGER.COLUMNS: ["weight", "score"],  # 'weight' conflict
            ENCODING_MANAGER.TRANSFORMER: StandardScaler(),
        },
    }


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
        ENCODINGS.CATEGORICAL,
        ENCODINGS.NUMERIC,
    }


def test_encoding_config_format_equivalence(
    valid_encoding_config, valid_simple_encoding_config
):
    """Test that complex and simple format configs produce equivalent EncodingManager objects."""
    from napistu_torch.load.constants import ENCODINGS

    # Create encoders dict matching the config
    encoding_transformers = {
        ENCODINGS.CATEGORICAL: OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        ),
        ENCODINGS.NUMERIC: StandardScaler(),
    }

    # Create EncodingManager from complex format
    manager_complex = EncodingManager(valid_encoding_config)

    # Create EncodingManager from simple format
    manager_simple = EncodingManager(
        valid_simple_encoding_config, encoders=encoding_transformers
    )

    # Get encoding tables from both
    table_complex = manager_complex.get_encoding_table()
    table_simple = manager_simple.get_encoding_table()

    # Sort both tables by all columns for consistent comparison
    table_complex_sorted = table_complex.sort_values(
        by=[ENCODING_MANAGER_TABLE.TRANSFORM_NAME, ENCODING_MANAGER_TABLE.COLUMN]
    ).reset_index(drop=True)
    table_simple_sorted = table_simple.sort_values(
        by=[ENCODING_MANAGER_TABLE.TRANSFORM_NAME, ENCODING_MANAGER_TABLE.COLUMN]
    ).reset_index(drop=True)

    # Tables should be identical after sorting
    pd.testing.assert_frame_equal(table_complex_sorted, table_simple_sorted)


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
    assert set(composed.config_.keys()) == {
        ENCODINGS.CATEGORICAL,
        ENCODINGS.NUMERIC,
        "embeddings",
    }

    # Check categorical merge: should have species_type from base + node_type from override
    categorical_columns = set(
        composed.config_[ENCODINGS.CATEGORICAL][ENCODING_MANAGER.COLUMNS]
    )
    assert "species_type" in categorical_columns  # Preserved from base
    assert "node_type" in categorical_columns  # From override (wins conflict)

    # Check numeric preserved unchanged (no conflicts)
    assert composed.config_[ENCODINGS.NUMERIC][ENCODING_MANAGER.COLUMNS] == [
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
        ENCODINGS.CATEGORICAL: {
            ENCODING_MANAGER.COLUMNS: ["node_type"],
            ENCODING_MANAGER.TRANSFORMER: OneHotEncoder(),
        }
    }

    override = {
        ENCODINGS.NUMERIC: {
            ENCODING_MANAGER.COLUMNS: ["weight"],
            ENCODING_MANAGER.TRANSFORMER: StandardScaler(),
        }
    }

    base_manager = EncodingManager(base)
    override_manager = EncodingManager(override)
    composed = base_manager.compose(override_manager)

    # Should have both transforms with no changes
    assert set(composed.config_.keys()) == {ENCODINGS.CATEGORICAL, ENCODINGS.NUMERIC}
    assert composed.config_[ENCODINGS.CATEGORICAL][ENCODING_MANAGER.COLUMNS] == [
        "node_type"
    ]
    assert composed.config_[ENCODINGS.NUMERIC][ENCODING_MANAGER.COLUMNS] == ["weight"]


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
        ENCODINGS.CATEGORICAL: {"node_type", "species_type"},
        ENCODINGS.NUMERIC: {"weight"},
    }

    encoders = {
        ENCODINGS.CATEGORICAL: OneHotEncoder(sparse_output=False),
        ENCODINGS.NUMERIC: StandardScaler(),
    }

    # Create manager using simple format
    manager = EncodingManager(simple_spec, encoders=encoders)

    # Check it's an EncodingManager
    assert isinstance(manager, EncodingManager)

    # Check the config was converted to complex format
    assert ENCODINGS.CATEGORICAL in manager.config_
    assert ENCODINGS.NUMERIC in manager.config_

    # Check columns are sorted lists
    assert manager.config_[ENCODINGS.CATEGORICAL]["columns"] == [
        "node_type",
        "species_type",
    ]
    assert manager.config_[ENCODINGS.NUMERIC]["columns"] == ["weight"]

    # Check transformers are assigned
    assert isinstance(
        manager.config_[ENCODINGS.CATEGORICAL]["transformer"], OneHotEncoder
    )
    assert isinstance(manager.config_[ENCODINGS.NUMERIC]["transformer"], StandardScaler)


def test_encoding_manager_init_complex_format():
    """Test EncodingManager.__init__() with complex format (backward compatibility)."""
    complex_config = {
        ENCODINGS.NUMERIC: {
            ENCODING_MANAGER.COLUMNS: ["value"],
            ENCODING_MANAGER.TRANSFORMER: StandardScaler(),
        }
    }

    # Create manager using complex format (no encoders)
    manager = EncodingManager(complex_config)

    assert isinstance(manager, EncodingManager)
    assert manager.config_ == complex_config


def test_encoding_manager_ensure():
    """Test EncodingManager.ensure() with complex dict, simple dict, EncodingManager, and invalid types."""
    # Test complex format dict conversion
    complex_config = {
        ENCODINGS.NUMERIC: {
            ENCODING_MANAGER.COLUMNS: ["value"],
            ENCODING_MANAGER.TRANSFORMER: StandardScaler(),
        }
    }
    result_from_complex = EncodingManager.ensure(complex_config)
    assert isinstance(result_from_complex, EncodingManager)
    assert result_from_complex.config_ == complex_config

    # Test simple format dict conversion with encoders
    simple_spec = {
        ENCODINGS.CATEGORICAL: {"col1", "col2"},
        ENCODINGS.NUMERIC: {"value"},
    }
    encoders = {
        ENCODINGS.CATEGORICAL: OneHotEncoder(sparse_output=False),
        ENCODINGS.NUMERIC: StandardScaler(),
    }
    result_from_simple = EncodingManager.ensure(simple_spec, encoders=encoders)
    assert isinstance(result_from_simple, EncodingManager)
    assert ENCODINGS.CATEGORICAL in result_from_simple.config_
    assert ENCODINGS.NUMERIC in result_from_simple.config_
    assert result_from_simple.config_[ENCODINGS.CATEGORICAL][
        ENCODING_MANAGER.COLUMNS
    ] == ["col1", "col2"]
    assert result_from_simple.config_[ENCODINGS.NUMERIC][ENCODING_MANAGER.COLUMNS] == [
        "value"
    ]

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
        ENCODINGS.CATEGORICAL: {"node_type", "species_type"},
        ENCODINGS.NUMERIC: {"weight", "score"},
    }

    encoders = {
        ENCODINGS.CATEGORICAL: OneHotEncoder(sparse_output=False),
        ENCODINGS.NUMERIC: StandardScaler(),
    }

    # Create manager from simple spec (either via __init__ or ensure)
    manager = EncodingManager(encoding_spec, encoders=encoders)

    # Check it's an EncodingManager
    assert isinstance(manager, EncodingManager)

    # Check the config has the right structure
    assert ENCODINGS.CATEGORICAL in manager.config_
    assert ENCODINGS.NUMERIC in manager.config_

    # Check columns are sorted lists
    assert manager.config_[ENCODINGS.CATEGORICAL][ENCODING_MANAGER.COLUMNS] == [
        "node_type",
        "species_type",
    ]
    assert manager.config_[ENCODINGS.NUMERIC][ENCODING_MANAGER.COLUMNS] == [
        "score",
        "weight",
    ]

    # Check transformers are assigned
    assert isinstance(
        manager.config_[ENCODINGS.CATEGORICAL][ENCODING_MANAGER.TRANSFORMER],
        OneHotEncoder,
    )
    assert isinstance(
        manager.config_[ENCODINGS.NUMERIC][ENCODING_MANAGER.TRANSFORMER], StandardScaler
    )

    # Check transform table
    table = manager.get_encoding_table()
    assert len(table) == 4  # 2 categorical + 2 numeric


def test_encoding_manager_simple_format_unknown_type():
    """Test that unknown encoding type raises error in simple format."""
    encoding_spec = {"unknown_type": {"col1"}}

    encoders = {ENCODINGS.CATEGORICAL: OneHotEncoder()}

    with pytest.raises(ValueError, match="Unknown encoding type: unknown_type"):
        EncodingManager(encoding_spec, encoders=encoders)


def test_encoding_manager_ensure_simple_format_unknown_type():
    """Test that ensure also raises error for unknown encoding type."""
    encoding_spec = {"unknown_type": {"col1"}}

    encoders = {ENCODINGS.CATEGORICAL: OneHotEncoder()}

    with pytest.raises(ValueError, match="Unknown encoding type: unknown_type"):
        EncodingManager.ensure(encoding_spec, encoders=encoders)
