import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, RootModel, field_validator, model_validator
from sklearn.compose import ColumnTransformer

from napistu_torch.load.constants import TRANSFORM_TABLE, TRANSFORMATION

logger = logging.getLogger(__name__)


class EncodingManager:
    """Configuration manager for DataFrame encoding transformations.

    This class manages encoding configurations, validates them, and provides
    utilities for inspecting and composing configurations.

    Parameters
    ----------
    config : Dict[str, Dict]
        Encoding configuration dictionary. Each key is a transform name
        and each value is a dict with 'columns' and 'transformer' keys.
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

    Attributes
    ----------
    config_ : Dict[str, Dict]
        The validated configuration dictionary.

    Methods
    -------
    compose(override_config, verbose=False)
        Compose this configuration with another configuration using merge strategy.
    get_transform_table()
        Get a summary table of all configured transformations.
    log_summary()
        Log a summary of all configured transformations.
    validate(config)
        Validate a configuration dictionary.

    Private Methods
    ---------------
    _create_transform_table(config)
        Create transform table from validated config.

    Raises
    ------
    ValueError
        If the configuration is invalid or has column conflicts.

    Examples
    --------
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
    >>> config = EncodingConfig(config_dict)
    >>> config.log_summary()
    >>> print(config.get_transform_table())
    """

    def __init__(self, config: Dict[str, Dict]):
        self.config_ = self.validate(config)

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
        base_table = self.get_transform_table()
        override_table = override_config.get_transform_table()

        # Find cross-config conflicts
        cross_conflicts = _find_cross_config_conflicts(base_table, override_table)

        if verbose and cross_conflicts:
            logger.info("Cross-config conflicts detected:")
            for column, details in cross_conflicts.items():
                logger.info(
                    f"  Column '{column}': base transforms {details[TRANSFORMATION.BASE]} -> override transforms {details[TRANSFORMATION.OVERRIDE]}"
                )
        elif verbose:
            logger.info("No cross-config conflicts detected")

        # Merge configs
        composed_dict = _merge_configs(
            self.config_, override_config.config_, cross_conflicts
        )

        # Return new EncodingConfig instance (validation happens in __init__)
        return EncodingManager(composed_dict)

    def get_transform_table(self) -> pd.DataFrame:
        """Get a summary table of all configured transformations.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'transform_name', 'column', and 'transformer_type'
            showing which columns are assigned to which transformers.

        Examples
        --------
        >>> config = EncodingConfig(config_dict)
        >>> table = config.get_transform_table()
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

        return self._create_transform_table(validated_config)

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
            transformer = transform_config[TRANSFORMATION.TRANSFORMER]
            columns = transform_config[TRANSFORMATION.COLUMNS]

            transformer_type = (
                type(transformer).__name__
                if transformer != TRANSFORMATION.PASSTHROUGH
                else TRANSFORMATION.PASSTHROUGH
            )

            logger.info(f"{transform_name} ({transformer_type}): {columns}")

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

                if TRANSFORMATION.COLUMNS not in transform_config:
                    raise ValueError(f"Transform '{name}' missing 'columns' key")

                if TRANSFORMATION.TRANSFORMER not in transform_config:
                    raise ValueError(f"Transform '{name}' missing 'transformer' key")

                columns = transform_config[TRANSFORMATION.COLUMNS]
                transformer = transform_config[TRANSFORMATION.TRANSFORMER]

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
                    hasattr(transformer, TRANSFORMATION.FIT)
                    or hasattr(transformer, TRANSFORMATION.TRANSFORM)
                    or transformer == TRANSFORMATION.PASSTHROUGH
                ):
                    raise ValueError(
                        f"Transform '{name}': transformer must have fit/transform methods or be 'passthrough'"
                    )

                validated_transforms[name] = transform_config

            # Check for column conflicts across transforms
            column_to_transforms = defaultdict(list)
            for transform_name, transform_config in validated_transforms.items():
                for column in transform_config[TRANSFORMATION.COLUMNS]:
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
            len(config.get(TRANSFORMATION.COLUMNS, []))
            for config in self.config_.values()
        )
        return f"EncodingConfig(transforms={n_transforms}, columns={total_columns})"

    def _create_transform_table(
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
                if transform_config.transformer != TRANSFORMATION.PASSTHROUGH
                else TRANSFORMATION.PASSTHROUGH
            )

            for column in transform_config.columns:
                rows.append(
                    {
                        TRANSFORM_TABLE.TRANSFORM_NAME: transform_name,
                        TRANSFORM_TABLE.COLUMN: column,
                        TRANSFORM_TABLE.TRANSFORMER_TYPE: transformer_type,
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

    columns: List[str] = Field(..., min_length=1)
    transformer: Any = Field(...)

    @field_validator(TRANSFORMATION.COLUMNS)
    @classmethod
    def validate_columns(cls, v):
        for col in v:
            if not isinstance(col, str) or not col.strip():
                raise ValueError("all columns must be non-empty strings")
        return v

    @field_validator(TRANSFORMATION.TRANSFORMER)
    @classmethod
    def validate_transformer(cls, v):
        if not (
            hasattr(v, TRANSFORMATION.FIT)
            or hasattr(v, TRANSFORMATION.TRANSFORM)
            or v == TRANSFORMATION.PASSTHROUGH
        ):
            raise ValueError(
                'transformer must have fit/transform methods or be "passthrough"'
            )
        return v

    model_config = {"arbitrary_types_allowed": True}


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


def config_to_column_transformer(
    encoding_config: Union[Dict[str, Dict], EncodingConfig],
) -> ColumnTransformer:
    """Convert validated config dict to sklearn ColumnTransformer.

    Parameters
    ----------
    encoding_config : Union[Dict[str, Dict], EncodingConfig]
        Configuration dictionary (will be validated first).

    Returns
    -------
    ColumnTransformer
        sklearn ColumnTransformer ready for fit/transform.

    Raises
    ------
    ValueError
        If config is invalid.

    Examples
    --------
    >>> config = {
    ...     'categorical': {
    ...         'columns': ['node_type', 'species_type'],
    ...         'transformer': OneHotEncoder(handle_unknown='ignore')
    ...     },
    ...     'numerical': {
    ...         'columns': ['weight', 'score'],
    ...         'transformer': StandardScaler()
    ...     }
    ... }
    >>> preprocessor = config_to_column_transformer(config)
    >>> # Equivalent to:
    >>> # ColumnTransformer([
    >>> #     ('categorical', OneHotEncoder(handle_unknown='ignore'), ['node_type', 'species_type']),
    >>> #     ('numerical', StandardScaler(), ['weight', 'score'])
    >>> # ])
    """
    # Validate config first

    if isinstance(encoding_config, dict):
        encoding_config = EncodingManager(encoding_config)

    if not isinstance(encoding_config, EncodingManager):
        raise ValueError(
            "encoding_config must be a dictionary or an EncodingManager instance"
        )

    # Build transformers list for ColumnTransformer
    transformers = []
    for transform_name, transform_config in encoding_config.items():
        transformer = transform_config["transformer"]
        columns = transform_config["columns"]

        transformers.append((transform_name, transformer, columns))

    return ColumnTransformer(transformers, remainder="drop")


def encode_dataframe(
    df: pd.DataFrame,
    encoding_defaults: Union[Dict[str, Dict], EncodingManager],
    encoding_overrides: Optional[Union[Dict[str, Dict], EncodingManager]] = None,
    verbose: bool = False,
) -> tuple[np.ndarray, List[str]]:
    """Encode a DataFrame using sklearn transformers with configurable encoding rules.

    This function applies a series of transformations to a DataFrame based on
    encoding configurations. It supports both default encoding rules and optional
    overrides that can modify or extend the default behavior.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to be encoded. Must contain all columns specified in
        the encoding configurations.
    encoding_defaults : Dict[str, Dict]
        Base encoding configuration dictionary. Each key is a transform name
        and each value is a dict with 'columns' and 'transformer' keys.
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
    encoding_overrides : Optional[Dict[str, Dict]], default=None
        Optional override configuration that will be merged with encoding_defaults.
        For column conflicts, the override configuration takes precedence.
        If None, only encoding_defaults will be used.
    verbose : bool, default=False
        If True, log detailed information about config composition and conflicts.

    Returns
    -------
    tuple[np.ndarray, List[str]]
        A tuple containing:
        - encoded_array : np.ndarray
            Transformed numpy array with encoded features. The number of columns
            may differ from the input due to transformations like OneHotEncoder.
        - feature_names : List[str]
            List of feature names corresponding to the columns in encoded_array.
            Names follow sklearn's convention: 'transform_name__column_name'.

    Raises
    ------
    ValueError
        If encoding configurations are invalid, have column conflicts, or if
        required columns are missing from the input DataFrame.
    KeyError
        If the input DataFrame is missing columns specified in the encoding config.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.preprocessing import OneHotEncoder, StandardScaler
    >>>
    >>> # Sample data
    >>> df = pd.DataFrame({
    ...     'category': ['A', 'B', 'A', 'C'],
    ...     'value': [1.0, 2.0, 3.0, 4.0]
    ... })
    >>>
    >>> # Encoding configuration
    >>> defaults = {
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
    >>> # Encode the DataFrame
    >>> encoded_array, feature_names = encode_dataframe(df, defaults)
    >>> print(f"Encoded shape: {encoded_array.shape}")
    >>> print(f"Feature names: {feature_names}")
    """

    if isinstance(encoding_defaults, dict):
        encoding_defaults = EncodingManager(encoding_defaults)
    if isinstance(encoding_overrides, dict):
        encoding_overrides = EncodingManager(encoding_overrides)

    if encoding_overrides is None:
        config = encoding_defaults
    else:
        config = encoding_defaults.compose(encoding_overrides, verbose=verbose)

    if verbose:
        config.log_summary()

    preprocessor = config_to_column_transformer(config)

    # Check for missing columns before fitting
    required_columns = set()
    for transform_config in config.values():
        required_columns.update(transform_config.get("columns", []))

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(
            f"Missing columns in DataFrame: {list(missing_columns)}. Available columns: {list(df.columns)}"
        )

    # Check for empty DataFrame
    if len(df) == 0:
        raise ValueError(
            "Cannot encode empty DataFrame. DataFrame must contain at least one row."
        )

    encoded_array = preprocessor.fit_transform(df)
    feature_names = _get_feature_names(preprocessor)

    # Return numpy array directly for PyTorch compatibility
    return encoded_array, feature_names


# private


def _find_cross_config_conflicts(
    base_table: pd.DataFrame, override_table: pd.DataFrame
) -> Dict[str, Dict]:
    """Find columns that appear in both config tables."""
    if base_table.empty or override_table.empty:
        return {}

    base_columns = set(base_table[TRANSFORM_TABLE.COLUMN])
    override_columns = set(override_table[TRANSFORM_TABLE.COLUMN])
    conflicted_columns = base_columns & override_columns

    conflicts = {}
    for column in conflicted_columns:
        base_transforms = base_table[base_table[TRANSFORM_TABLE.COLUMN] == column][
            TRANSFORM_TABLE.TRANSFORM_NAME
        ].tolist()
        override_transforms = override_table[
            override_table[TRANSFORM_TABLE.COLUMN] == column
        ][TRANSFORM_TABLE.TRANSFORM_NAME].tolist()

        conflicts[column] = {
            TRANSFORMATION.BASE: base_transforms,
            TRANSFORMATION.OVERRIDE: override_transforms,
        }

    return conflicts


def _get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Get feature names from fitted ColumnTransformer using sklearn's standard method.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted ColumnTransformer instance.

    Returns
    -------
    List[str]
        List of feature names in the same order as transform output columns.

    Examples
    --------
    >>> preprocessor = config_to_column_transformer(config)
    >>> preprocessor.fit(data)  # Must fit first!
    >>> feature_names = _get_feature_names(preprocessor)
    >>> # ['cat__node_type_A', 'cat__node_type_B', 'num__weight']
    """
    if not hasattr(preprocessor, "transformers_"):
        raise ValueError("ColumnTransformer must be fitted first")

    # Use sklearn's built-in method (available since sklearn 1.0+)
    return preprocessor.get_feature_names_out().tolist()


def _merge_configs(
    base_config: Dict, override_config: Dict, cross_conflicts: Dict
) -> Dict:
    """Merge configs with merge strategy."""
    composed = base_config.copy()
    conflicted_columns = set(cross_conflicts.keys())

    for transform_name, transform_config in override_config.items():
        if transform_name in composed:
            # Merge column lists
            base_columns = set(composed[transform_name][TRANSFORMATION.COLUMNS])
            override_columns = set(transform_config[TRANSFORMATION.COLUMNS])

            # Remove conflicts from base (override wins)
            base_columns -= conflicted_columns
            merged_columns = list(base_columns | override_columns)

            composed[transform_name] = {
                TRANSFORMATION.COLUMNS: merged_columns,
                TRANSFORMATION.TRANSFORMER: transform_config[
                    TRANSFORMATION.TRANSFORMER
                ],
            }
        else:
            composed[transform_name] = transform_config

    return composed
