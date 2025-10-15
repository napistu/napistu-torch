import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from napistu_torch.load.constants import (
    ENCODING_MANAGER,
    ENCODING_MANAGER_TABLE,
    ENCODINGS,
    NEVER_ENCODE,
)
from napistu_torch.load.encoders import DEFAULT_ENCODERS
from napistu_torch.load.encoding_manager import (
    EncodingConfig,
    EncodingManager,
)

logger = logging.getLogger(__name__)


def auto_encode(
    graph_df: pd.DataFrame,
    existing_encodings: Union[Dict, EncodingManager],
    encoders: Dict = DEFAULT_ENCODERS,
) -> EncodingManager:
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
    existing_encoding_columns = set(
        encoding_manager.get_encoding_table()[ENCODING_MANAGER_TABLE.COLUMN].tolist()
    )

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
        logger.warning(
            f"Series '{series.name}' has only 1 unique value ({unique_values[0]}), no variance"
        )
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
            if str_values.issubset({"true", "false", "0", "1"}):
                return ENCODINGS.BINARY

        # Categorical
        if n_unique > max_categories:
            logger.warning(
                f"Series '{series.name}' has {n_unique} unique values, exceeding max_categories={max_categories}"
            )
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
