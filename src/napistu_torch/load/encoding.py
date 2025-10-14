import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, RootModel, field_validator, model_validator
from sklearn.compose import ColumnTransformer

from napistu_torch.load.encoders import DEFAULT_ENCODERS
from napistu_torch.load.constants import (
    ENCODING_MANAGER,
    ENCODING_MANAGER_TABLE,
    ENCODINGS,
    NEVER_ENCODE,
)

logger = logging.getLogger(__name__)


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
        encoders: Optional[Dict[str, Any]] = None
    ):
        # If encoders provided, convert simple format to complex format
        if encoders is not None:
            config = self._convert_simple_to_complex(config, encoders)
        
        self.config_ = self.validate(config)
    
    @staticmethod
    def _convert_simple_to_complex(
        simple_spec: Dict[str, set],
        encoders: Dict[str, Any]
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
        encoders: Optional[Dict[str, Any]] = None
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
            return cls(config, encoders=encoders)
        elif isinstance(config, cls):
            return config
        else:
            raise ValueError(
                f"config must be a dict or an EncodingManager object, got {type(config)}"
            )

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

    columns: List[str] = Field(..., min_length=1)
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


def auto_encode(graph_df: pd.DataFrame, existing_encodings: Union[Dict, EncodingManager], encoders: Dict = DEFAULT_ENCODERS) -> EncodingManager:

    """
    Select appropriate encodings for each column in a graph dataframe (either the vertex_df or edge_df)

    Parameters
    ----------
    graph_df : pd.DataFrame
        The dataframe to select encodings for.
    existing_encodings : Union[Dict, EncodingManager]
        The existing encodings to use. This could be VERTEX_DEFAULT_TRANSFORMS or EDGE_DEFAULT_TRANSFORMS
        or any modified version of these.
    encoders : Dict, default=ENCODERS
        The encoders to use. These will be used to map from column encoding classes to the encoders themselves. If existing_encodings is a dict, then it must be passed in the 'simple' format which is a lookup from encoder keys to the columns using that encoder.

    Returns
    -------
    EncodingManager
        A new EncodingManager with the selected encodings.
    """

    # accounted for variables
    columns = set(graph_df.columns.tolist())

    encoding_manager = EncodingManager.ensure(existing_encodings, encoders)
    existing_encoding_columns = set(encoding_manager.get_encoding_table()[ENCODING_MANAGER_TABLE.COLUMN].tolist())

    unencoded_columns = columns - existing_encoding_columns - NEVER_ENCODE

    select_encodings = graph_df.loc[:, list(unencoded_columns)].apply(classify_encoding)

    # If this is a Series showing dtypes (like df.dtypes)
    new_encodings = select_encodings.groupby(select_encodings).groups
    new_encodings = {k: set(v) for k, v in new_encodings.items()}

    new_encoding_manager = EncodingManager(new_encodings, encoders)

    # combine existing and new encodings
    return encoding_manager.compose(new_encoding_manager)


def classify_encoding(series: pd.Series, max_categories: int = 50) -> Optional[str]:
    """
    Classify the encoding type for a pandas Series.
    
    Parameters
    ----------
    series : pd.Series
        The column to classify
    max_categories : int, default=50
        Maximum number of unique values for categorical encoding.
        If exceeded, logs a warning and returns None.
    
    Returns
    -------
    Optional[str]
        One of: 'binary', 'categorical', 'numeric', 'numeric_sparse', or None
        Returns None for constant variables or high-cardinality features.
    
    Examples
    --------
    >>> classify_encoding(pd.Series([0, 1, 0, 1]))
    'binary'
    >>> classify_encoding(pd.Series([0, 1, np.nan]))
    'categorical'
    >>> classify_encoding(pd.Series([1.5, 2.3, 4.1]))
    'numeric'
    >>> classify_encoding(pd.Series([1.5, np.nan, 4.1]))
    'numeric_sparse'
    >>> classify_encoding(pd.Series([5, 5, 5, 5]))  # Constant
    None
    """
    # Drop NaN for initial analysis
    non_null = series.dropna()
    has_missing = len(non_null) < len(series)
    
    # Handle empty or all-NaN series
    if len(non_null) == 0:
        logger.warning(f"Series '{series.name}' is empty or all NaN")
        return None
    
    # Get unique values (excluding NaN)
    unique_values = non_null.unique()
    n_unique = len(unique_values)
    
    # Check for constant variable (only 1 unique value, no NaNs)
    if n_unique == 1 and not has_missing:
        logger.warning(f"Series '{series.name}' has only 1 unique value ({unique_values[0]}), no variance")
        return None
    
    # Check if numeric dtype
    is_numeric = pd.api.types.is_numeric_dtype(series)
    
    if is_numeric:
        # Check for binary/boolean (only 0 and 1, no missing values)
        if not has_missing and n_unique <= 2 and set(unique_values).issubset({0, 1}):
            return ENCODINGS.BINARY
        
        # Check if values are only 0 and 1 but has missing (treat as categorical)
        if has_missing and n_unique <= 2 and set(unique_values).issubset({0, 1}):
            return ENCODINGS.CATEGORICAL
        
        # Numeric continuous values
        if has_missing:
            return ENCODINGS.SPARSE_NUMERIC
        else:
            return ENCODINGS.NUMERIC
    
    else:
        # Non-numeric data: categorical or boolean strings
        # Check for True/False strings
        if not has_missing and n_unique <= 2:
            str_values = set(str(v).lower() for v in unique_values)
            if str_values.issubset({'true', 'false', '0', '1'}):
                return ENCODINGS.BINARY
        
        # Categorical
        if n_unique > max_categories:
            logger.warning(f"Series '{series.name}' has {n_unique} unique values, exceeding max_categories={max_categories}")
            return None
        
        return ENCODINGS.CATEGORICAL


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
        transformer = transform_config[ENCODING_MANAGER.TRANSFORMER]
        columns = transform_config[ENCODING_MANAGER.COLUMNS]

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
        required_columns.update(transform_config.get(ENCODING_MANAGER.COLUMNS, []))

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

    base_columns = set(base_table[ENCODING_MANAGER_TABLE.COLUMN])
    override_columns = set(override_table[ENCODING_MANAGER_TABLE.COLUMN])
    conflicted_columns = base_columns & override_columns

    conflicts = {}
    for column in conflicted_columns:
        base_transforms = base_table[base_table[ENCODING_MANAGER_TABLE.COLUMN] == column][
            ENCODING_MANAGER_TABLE.TRANSFORM_NAME
        ].tolist()
        override_transforms = override_table[
            override_table[ENCODING_MANAGER_TABLE.COLUMN] == column
        ][ENCODING_MANAGER_TABLE.TRANSFORM_NAME].tolist()

        conflicts[column] = {
            ENCODING_MANAGER.BASE: base_transforms,
            ENCODING_MANAGER.OVERRIDE: override_transforms,
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
