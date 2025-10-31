"""Configuration management for DataFrame encoding transformations."""

import logging
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationError,
    field_validator,
    model_validator,
)

from napistu_torch.load.constants import (
    ENCODING_MANAGER,
    ENCODING_MANAGER_TABLE,
)

logger = logging.getLogger(__name__)


ENCODING_CONFIG_FORMAT = SimpleNamespace(
    SIMPLE="simple",
    COMPLEX="complex",
)


class EncodingManager:
    """Configuration manager for DataFrame encoding transformations.

    This class manages encoding configurations, validates them, and provides
    utilities for inspecting and composing configurations.

    Parameters
    ----------
    config : Dict[str, Dict] or Dict[str, set]
        Encoding configuration dictionary. Supports two formats:

        Complex format (when encoders=None):
            Each key is a transform name and each value is a dict with
            'columns' and 'transformer' keys.
            Example: {
                'categorical': {
                    'columns': ['col1', 'col2'],
                    'transformer': OneHotEncoder()
                },
                'numerical': {
                    'columns': ['col3'],
                    'transformer': StandardScaler()
                }
            }

        Simple format (when encoders is provided):
            Each key is an encoding type and each value is a set/list of column names.
            Example: {
                'categorical': {'col1', 'col2'},
                'numerical': {'col3'}
            }

    encoders : Dict[str, Any], optional
        Mapping from encoding type to transformer instance. Only used with
        simple format. If provided, config is treated as simple format and
        converted to complex format internally.
        Example: {
            'categorical': OneHotEncoder(),
            'numerical': StandardScaler()
        }

    Attributes
    ----------
    config_ : Dict[str, Dict]
        The validated configuration dictionary (always in complex format).

    Methods
    -------
    compose(override_config, verbose=False)
        Compose this configuration with another configuration using merge strategy.
    ensure(config, encoders=None)
        Class method to ensure config is an EncodingManager instance.
        Supports both simple and complex dict formats via encoders parameter.
    get_config()
        Get the encoding configuration dictionary.
    get_encoding_table()
        Get a summary table of all configured transformations.
    log_summary()
        Log a summary of all configured transformations.
    validate(config)
        Validate a configuration dictionary.

    Private Methods
    ---------------
    _create_encoding_table(config)
        Create transform table from validated config.

    Raises
    ------
    ValueError
        If the configuration is invalid or has column conflicts.

    Examples
    --------
    Complex format:

    >>> from sklearn.preprocessing import OneHotEncoder, StandardScaler
    >>>
    >>> config_dict = {
    ...     'categorical': {
    ...         'columns': ['category'],
    ...         'transformer': OneHotEncoder(sparse_output=False)
    ...     },
    ...     'numerical': {
    ...         'columns': ['value'],
    ...         'transformer': StandardScaler()
    ...     }
    ... }
    >>>
    >>> config = EncodingManager(config_dict)
    >>> config.log_summary()
    >>> print(config.get_encoding_table())

    Simple format:

    >>> simple_spec = {
    ...     'categorical': {'category'},
    ...     'numerical': {'value'}
    ... }
    >>> encoders = {
    ...     'categorical': OneHotEncoder(sparse_output=False),
    ...     'numerical': StandardScaler()
    ... }
    >>> config = EncodingManager(simple_spec, encoders=encoders)
    >>> print(config.get_encoding_table())
    """

    def __init__(
        self,
        config: Union[Dict[str, Dict], Dict[str, set]],
        encoders: Optional[Dict[str, Any]] = None,
    ):
        # If encoders provided, convert simple format to complex format
        if encoders is not None:
            config = self._convert_simple_to_complex(config, encoders)

        self.config_ = self.validate(config)

    @staticmethod
    def _convert_simple_to_complex(
        simple_spec: Dict[str, set], encoders: Dict[str, Any]
    ) -> Dict[str, Dict]:
        """Convert simple spec format to complex format.

        Parameters
        ----------
        simple_spec : Dict[str, set]
            Mapping from encoding type to set of column names.
        encoders : Dict[str, Any]
            Mapping from encoding type to transformer instance.

        Returns
        -------
        Dict[str, Dict]
            Complex format configuration.
        """
        complex_config = {}

        for encoding_type, columns in simple_spec.items():
            if encoding_type not in encoders:
                raise ValueError(f"Unknown encoding type: {encoding_type}")

            # Convert set to sorted list for consistent ordering
            column_list = sorted(list(columns))

            complex_config[encoding_type] = {
                ENCODING_MANAGER.COLUMNS: column_list,
                ENCODING_MANAGER.TRANSFORMER: encoders[encoding_type],
            }

        return complex_config

    def compose(
        self,
        override_config: "EncodingConfig",
        verbose: bool = False,
    ) -> "EncodingConfig":
        """Compose this configuration with another configuration using merge strategy.

        Merges configs at the transform level. For cross-config column conflicts,
        the override config takes precedence while preserving non-conflicted
        columns from this (base) config.

        Parameters
        ----------
        override_config : EncodingConfig
            Configuration to merge in, taking precedence over this config.
        verbose : bool, default=False
            If True, log detailed information about conflicts and final transformations.

        Returns
        -------
        EncodingConfig
            New EncodingConfig instance with the composed configuration.

        Examples
        --------
        >>> base = EncodingConfig({'num': {'columns': ['a', 'b'], 'transformer': StandardScaler()}})
        >>> override = EncodingConfig({'cat': {'columns': ['c'], 'transformer': OneHotEncoder()}})
        >>> composed = base.compose(override)
        >>> print(composed)  # EncodingConfig(transforms=2, columns=3)
        """
        # Both configs are already validated since they're EncodingConfig instances

        # Create transform tables for conflict detection
        base_table = self.get_encoding_table()
        override_table = override_config.get_encoding_table()

        # Find cross-config conflicts
        cross_conflicts = _find_cross_config_conflicts(base_table, override_table)

        if verbose and cross_conflicts:
            logger.info("Cross-config conflicts detected:")
            for column, details in cross_conflicts.items():
                logger.info(
                    f"  Column '{column}': base transforms {details[ENCODING_MANAGER.BASE]} -> override transforms {details[ENCODING_MANAGER.OVERRIDE]}"
                )
        elif verbose:
            logger.info("No cross-config conflicts detected")

        # Merge configs
        composed_dict = _merge_configs(
            self.config_, override_config.config_, cross_conflicts
        )

        # Return new EncodingConfig instance (validation happens in __init__)
        return EncodingManager(composed_dict)

    @classmethod
    def ensure(
        cls,
        config: Union[dict, "EncodingManager"],
        encoders: Optional[Dict[str, Any]] = None,
    ) -> "EncodingManager":
        """
        Ensure that config is an EncodingManager object.

        If config is a dict, it will be converted to an EncodingManager.
        If it's already an EncodingManager, it will be returned as-is.

        Parameters
        ----------
        config : Union[dict, EncodingManager]
            Either a dict (simple or complex format) or an EncodingManager object.
        encoders : Dict[str, Any], optional
            Mapping from encoding type to transformer instance. Only used when
            config is a dict in simple format. Ignored if config is already an
            EncodingManager.

        Returns
        -------
        EncodingManager
            The EncodingManager object

        Raises
        ------
        ValueError
            If config is neither a dict nor an EncodingManager

        Examples
        --------
        Complex format dict:

        >>> config = EncodingManager.ensure({
        ...     "foo": {"columns": ["bar"], "transformer": StandardScaler()}
        ... })
        >>> isinstance(config, EncodingManager)
        True

        Simple format dict:

        >>> config = EncodingManager.ensure(
        ...     {"categorical": {"col1", "col2"}},
        ...     encoders={"categorical": OneHotEncoder()}
        ... )
        >>> isinstance(config, EncodingManager)
        True

        EncodingManager passthrough:

        >>> manager = EncodingManager({"foo": {"columns": ["bar"], "transformer": StandardScaler()}})
        >>> result = EncodingManager.ensure(manager)
        >>> result is manager
        True
        """
        if isinstance(config, dict):
            # Detect config format and validate
            config_format = detect_config_format(config)

            # Only pass encoders if config is in simple format
            if config_format == ENCODING_CONFIG_FORMAT.COMPLEX:
                return cls(config, encoders=None)
            elif config_format == ENCODING_CONFIG_FORMAT.SIMPLE:
                return cls(config, encoders=encoders)
            else:
                raise ValueError(f"Invalid config format: {config_format}")
        elif isinstance(config, cls):
            return config
        else:
            raise ValueError(
                f"config must be a dict or an EncodingManager object, got {type(config)}"
            )

    def get_config(self) -> Dict[str, Dict]:
        """Get the encoding configuration dictionary.

        Returns
        -------
        Dict[str, Dict]
            The validated configuration dictionary in complex format.
        """
        return self.config_

    def get_encoding_table(self) -> pd.DataFrame:
        """Get a summary table of all configured transformations.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'transform_name', 'column', and 'transformer_type'
            showing which columns are assigned to which transformers.

        Examples
        --------
        >>> config = EncodingConfig(config_dict)
        >>> table = config.get_encoding_table()
        >>> print(table)
           transform_name    column transformer_type
        0     categorical      col1    OneHotEncoder
        1     categorical      col2    OneHotEncoder
        2       numerical      col3   StandardScaler
        """
        # Convert config to TransformConfig objects for validation
        validated_config = {}
        for name, config in self.config_.items():
            validated_config[name] = TransformConfig(**config)

        return self._create_encoding_table(validated_config)

    def log_summary(self) -> None:
        """Log a summary of all configured transformations.

        Logs one message per transformation showing the transformer type
        and the columns it will transform.

        Examples
        --------
        >>> config = EncodingConfig(config_dict)
        >>> config.log_summary()
        INFO:__main__:categorical (OneHotEncoder): ['col1', 'col2']
        INFO:__main__:numerical (StandardScaler): ['col3']
        """
        for transform_name, transform_config in self.config_.items():
            transformer = transform_config[ENCODING_MANAGER.TRANSFORMER]
            columns = transform_config[ENCODING_MANAGER.COLUMNS]
            columns_str = ", ".join(columns)

            transformer_type = (
                type(transformer).__name__
                if transformer != ENCODING_MANAGER.PASSTHROUGH
                else ENCODING_MANAGER.PASSTHROUGH
            )

            logger.info(f"{transform_name} ({transformer_type}): {columns_str}")

    def validate(self, config: Dict[str, Dict]) -> Dict[str, Dict]:
        """Validate a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Dict]
            Configuration dictionary to validate.

        Returns
        -------
        Dict[str, Dict]
            The validated configuration dictionary (same as input if valid).

        Raises
        ------
        ValueError
            If configuration structure is invalid or column conflicts exist.

        Examples
        --------
        >>> config_mgr = EncodingConfig({})
        >>> validated = config_mgr.validate(config_dict)
        """
        try:
            # Validate each transform config using the original Pydantic logic
            validated_transforms = {}
            for name, transform_config in config.items():
                # Validate transform structure
                if not isinstance(transform_config, dict):
                    raise ValueError(f"Transform '{name}' must be a dictionary")

                if ENCODING_MANAGER.COLUMNS not in transform_config:
                    raise ValueError(f"Transform '{name}' missing 'columns' key")

                if ENCODING_MANAGER.TRANSFORMER not in transform_config:
                    raise ValueError(f"Transform '{name}' missing 'transformer' key")

                columns = transform_config[ENCODING_MANAGER.COLUMNS]
                transformer = transform_config[ENCODING_MANAGER.TRANSFORMER]

                # Validate columns
                if not isinstance(columns, list) or len(columns) == 0:
                    raise ValueError(
                        f"Transform '{name}': columns must be a non-empty list"
                    )

                for col in columns:
                    if not isinstance(col, str) or not col.strip():
                        raise ValueError(
                            f"Transform '{name}': all columns must be non-empty strings"
                        )

                # Validate transformer
                if not (
                    hasattr(transformer, ENCODING_MANAGER.FIT)
                    or hasattr(transformer, ENCODING_MANAGER.TRANSFORM)
                    or transformer == ENCODING_MANAGER.PASSTHROUGH
                ):
                    raise ValueError(
                        f"Transform '{name}': transformer must have fit/transform methods or be 'passthrough'"
                    )

                validated_transforms[name] = transform_config

            # Check for column conflicts across transforms
            column_to_transforms = defaultdict(list)
            for transform_name, transform_config in validated_transforms.items():
                for column in transform_config[ENCODING_MANAGER.COLUMNS]:
                    column_to_transforms[column].append(transform_name)

            conflicts = {
                col: transforms
                for col, transforms in column_to_transforms.items()
                if len(transforms) > 1
            }

            if conflicts:
                conflict_details = [
                    f"'{col}': {transforms}" for col, transforms in conflicts.items()
                ]
                raise ValueError(f"Column conflicts: {'; '.join(conflict_details)}")

        except ValueError as e:
            raise ValueError(f"Config validation failed: {e}")

        return config

    def __getattr__(self, name):
        """Delegate dict methods to the underlying config dictionary."""
        if hasattr(self.config_, name):
            attr = getattr(self.config_, name)
            if callable(attr):
                return attr
            return attr
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __repr__(self) -> str:
        """Return string representation of the configuration."""
        n_transforms = len(self.config_)
        total_columns = sum(
            len(config.get(ENCODING_MANAGER.COLUMNS, []))
            for config in self.config_.values()
        )
        return f"EncodingConfig(transforms={n_transforms}, columns={total_columns})"

    def _create_encoding_table(
        self, config: Dict[str, "TransformConfig"]
    ) -> pd.DataFrame:
        """Create transform table from validated config.

        Parameters
        ----------
        config : Dict[str, TransformConfig]
            Dictionary mapping transform names to TransformConfig objects.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'transform_name', 'column', and 'transformer_type'.
        """
        rows = []
        for transform_name, transform_config in config.items():
            transformer_type = (
                type(transform_config.transformer).__name__
                if transform_config.transformer != ENCODING_MANAGER.PASSTHROUGH
                else ENCODING_MANAGER.PASSTHROUGH
            )

            for column in transform_config.columns:
                rows.append(
                    {
                        ENCODING_MANAGER_TABLE.TRANSFORM_NAME: transform_name,
                        ENCODING_MANAGER_TABLE.COLUMN: column,
                        ENCODING_MANAGER_TABLE.TRANSFORMER_TYPE: transformer_type,
                    }
                )

        return pd.DataFrame(rows)


class TransformConfig(BaseModel):
    """Configuration for a single transformation.

    Parameters
    ----------
    columns : List[str]
        Column names to transform. Must be non-empty strings.
    transformer : Any
        sklearn transformer object or 'passthrough'.
    """

    columns: list[str] = Field(..., min_length=1)
    transformer: Any = Field(...)

    @field_validator(ENCODING_MANAGER.COLUMNS)
    @classmethod
    def validate_columns(cls, v):
        for col in v:
            if not isinstance(col, str) or not col.strip():
                raise ValueError("all columns must be non-empty strings")
        return v

    @field_validator(ENCODING_MANAGER.TRANSFORMER)
    @classmethod
    def validate_transformer(cls, v):
        if not (
            hasattr(v, ENCODING_MANAGER.FIT)
            or hasattr(v, ENCODING_MANAGER.TRANSFORM)
            or v == ENCODING_MANAGER.PASSTHROUGH
        ):
            raise ValueError(
                'transformer must have fit/transform methods or be "passthrough"'
            )
        return v

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EncodingConfig(RootModel[Dict[str, TransformConfig]]):
    """Complete encoding configuration with conflict validation.

    Parameters
    ----------
    root : Dict[str, TransformConfig]
        Dictionary mapping transform names to their configurations.
    """

    @model_validator(mode="after")
    def check_no_column_conflicts(self):
        """Ensure no column appears in multiple transforms."""
        root_dict = self.root

        column_to_transforms = defaultdict(list)
        for transform_name, transform_config in root_dict.items():
            for column in transform_config.columns:
                column_to_transforms[column].append(transform_name)

        conflicts = {
            col: transforms
            for col, transforms in column_to_transforms.items()
            if len(transforms) > 1
        }

        if conflicts:
            conflict_details = [
                f"'{col}': {transforms}" for col, transforms in conflicts.items()
            ]
            raise ValueError(f"Column conflicts: {'; '.join(conflict_details)}")

        return self


class SimpleEncodingConfig(RootModel[Dict[str, Union[List[str], set]]]):
    """Simple encoding configuration format validator.

    Validates that each value is a list or set of column names (strings).

    Parameters
    ----------
    root : Dict[str, Union[List[str], set]]
        Dictionary mapping transform names to column name collections.
    """

    @model_validator(mode="after")
    def validate_all_values_are_column_collections(self):
        """Ensure all values are lists or sets of strings."""
        for transform_name, columns in self.root.items():
            if not isinstance(columns, (list, set)):
                raise ValueError(
                    f"Transform '{transform_name}': value must be a list or set of column names, got {type(columns)}"
                )

            if not columns:
                raise ValueError(
                    f"Transform '{transform_name}': column collection cannot be empty"
                )

            for col in columns:
                if not isinstance(col, str):
                    raise ValueError(
                        f"Transform '{transform_name}': all column names must be strings, got {type(col)}"
                    )

        return self


def detect_config_format(config: Dict) -> str:
    """Detect whether a config dict is in simple or complex format.

    Parameters
    ----------
    config : Dict
        Configuration dictionary to analyze.

    Returns
    -------
    str
        ENCODING_CONFIG_FORMAT.SIMPLE or ENCODING_CONFIG_FORMAT.COMPLEX

    Raises
    ------
    ValueError
        If config doesn't match either format specification.

    Examples
    --------
    >>> detect_config_format({'categorical': ['col1', 'col2']})
    'simple'

    >>> detect_config_format({'categorical': {'columns': ['col1'], 'transformer': OneHotEncoder()}})
    'complex'
    """
    if not config:
        # Empty config is valid for both formats, treat as complex
        return ENCODING_CONFIG_FORMAT.COMPLEX

    # Try validating as complex format first (more specific)
    try:
        EncodingConfig(root=config)
        return ENCODING_CONFIG_FORMAT.COMPLEX
    except ValidationError:
        pass

    # Try validating as simple format
    try:
        SimpleEncodingConfig(root=config)
        return ENCODING_CONFIG_FORMAT.SIMPLE
    except ValidationError:
        pass

    # If neither format is valid, provide helpful error
    raise ValueError(
        f"Config does not match simple or complex format. "
        f"Simple format: Dict[str, List[str]] (transform -> columns). "
        f"Complex format: Dict[str, Dict] with 'columns' and 'transformer' keys. "
        f"Got: {config}"
    )


# private utils


def _find_cross_config_conflicts(
    base_table: pd.DataFrame, override_table: pd.DataFrame
) -> Dict[str, Dict]:
    """Find columns that appear in both config tables."""
    if base_table.empty or override_table.empty:
        return {}

    base_columns = set(base_table[ENCODING_MANAGER_TABLE.COLUMN])
    override_columns = set(override_table[ENCODING_MANAGER_TABLE.COLUMN])
    conflicted_columns = base_columns & override_columns

    conflicts = {}
    for column in conflicted_columns:
        base_transforms = base_table[
            base_table[ENCODING_MANAGER_TABLE.COLUMN] == column
        ][ENCODING_MANAGER_TABLE.TRANSFORM_NAME].tolist()
        override_transforms = override_table[
            override_table[ENCODING_MANAGER_TABLE.COLUMN] == column
        ][ENCODING_MANAGER_TABLE.TRANSFORM_NAME].tolist()

        conflicts[column] = {
            ENCODING_MANAGER.BASE: base_transforms,
            ENCODING_MANAGER.OVERRIDE: override_transforms,
        }

    return conflicts


def _merge_configs(
    base_config: Dict, override_config: Dict, cross_conflicts: Dict
) -> Dict:
    """Merge configs with merge strategy."""
    composed = base_config.copy()
    conflicted_columns = set(cross_conflicts.keys())

    for transform_name, transform_config in override_config.items():
        if transform_name in composed:
            # Merge column lists
            base_columns = set(composed[transform_name][ENCODING_MANAGER.COLUMNS])
            override_columns = set(transform_config[ENCODING_MANAGER.COLUMNS])

            # Remove conflicts from base (override wins)
            base_columns -= conflicted_columns
            merged_columns = list(base_columns | override_columns)

            composed[transform_name] = {
                ENCODING_MANAGER.COLUMNS: merged_columns,
                ENCODING_MANAGER.TRANSFORMER: transform_config[
                    ENCODING_MANAGER.TRANSFORMER
                ],
            }
        else:
            composed[transform_name] = transform_config

    return composed
