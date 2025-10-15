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
    encoders: Dict = DEFAULT_ENCODERS,
    verbose: bool = False,
) -> tuple[np.ndarray, List[str]]:
    """Encode a DataFrame using sklearn transformers with configurable encoding rules.

    This is a convenience function that combines fitting and transforming in one step.
    For more control (e.g., fitting on training data and transforming test data),
    use fit_encoders() and transform_dataframe() separately.

    This function applies a series of transformations to a DataFrame based on
    encoding configurations. It supports both default encoding rules and optional
    overrides that can modify or extend the default behavior.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to be encoded. Must contain all columns specified in
        the encoding configurations.
    encoding_defaults : Union[Dict[str, Dict], EncodingManager]
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
    encoding_overrides : Optional[Union[Dict[str, Dict], EncodingManager]], default=None
        Optional override configuration that will be merged with encoding_defaults.
        For column conflicts, the override configuration takes precedence.
        If None, only encoding_defaults will be used.
    encoders : Dict, default=ENCODERS
        The encoders to use. If encoding_defaults or encoding_overrides are dicts, then these will be used to map from column encoding classes to the encoders themselves. If existing_encodings is a dict, then it must be passed in the 'simple' format which is a lookup from encoder keys to the columns using that encoder.
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
    >>> # Encode the DataFrame (fit and transform in one step)
    >>> encoded_array, feature_names = encode_dataframe(df, defaults)
    >>> print(f"Encoded shape: {encoded_array.shape}")
    >>> print(f"Feature names: {feature_names}")
    >>>
    >>> # For train/test split, use the two-step approach:
    >>> fitted_transformer = fit_encoders(train_df, defaults)
    >>> train_encoded, train_features = transform_dataframe(train_df, fitted_transformer)
    >>> test_encoded, test_features = transform_dataframe(test_df, fitted_transformer)
    """
    # Fit encoders on the provided DataFrame
    fitted_transformer = fit_encoders(
        df, encoding_defaults, encoding_overrides, encoders, verbose
    )

    # Transform using the fitted transformer
    return transform_dataframe(df, fitted_transformer)


def fit_encoders(
    df: pd.DataFrame,
    encoding_defaults: Union[Dict[str, Dict], EncodingManager],
    encoding_overrides: Optional[Union[Dict[str, Dict], EncodingManager]] = None,
    encoders: Dict = DEFAULT_ENCODERS,
    verbose: bool = False,
) -> ColumnTransformer:
    """Fit encoding transformers on a DataFrame.

    This function creates and fits a ColumnTransformer based on encoding
    configurations. The fitted transformer can then be used to transform
    this DataFrame or other DataFrames with the same schema.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to fit encoders on. Must contain all columns specified
        in the encoding configurations.
    encoding_defaults : Union[Dict[str, Dict], EncodingManager]
        Base encoding configuration. Each key is a transform name and each value
        is a dict with 'columns' and 'transformer' keys.
    encoding_overrides : Optional[Union[Dict[str, Dict], EncodingManager]], default=None
        Optional override configuration that will be merged with encoding_defaults.
        For column conflicts, the override configuration takes precedence.
    verbose : bool, default=False
        If True, log detailed information about config composition and conflicts.

    Returns
    -------
    ColumnTransformer
        Fitted sklearn ColumnTransformer ready to transform data.

    Raises
    ------
    ValueError
        If encoding configurations are invalid or if the DataFrame is empty.
    KeyError
        If the input DataFrame is missing columns specified in the encoding config.

    Examples
    --------
    >>> # Fit encoders on training data
    >>> fitted_transformer = fit_encoders(train_df, encoding_defaults)
    >>>
    >>> # Use the fitted transformer on train and test data
    >>> train_encoded, train_features = transform_dataframe(train_df, fitted_transformer)
    >>> test_encoded, test_features = transform_dataframe(test_df, fitted_transformer)
    """
    # Ensure configs are EncodingManager instances
    encoding_defaults = EncodingManager.ensure(encoding_defaults, encoders)
    if encoding_overrides is not None:
        encoding_overrides = EncodingManager.ensure(encoding_overrides, encoders)

    # Compose configurations
    if encoding_overrides is None:
        config = encoding_defaults
    else:
        config = encoding_defaults.compose(encoding_overrides, verbose=verbose)

    if verbose:
        config.log_summary()

    # Create ColumnTransformer from config
    preprocessor = config_to_column_transformer(config)

    # Check for missing columns before fitting
    required_columns = set()
    for transform_config in config.values():
        required_columns.update(transform_config.get(ENCODING_MANAGER.COLUMNS, []))

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(
            f"Missing columns in DataFrame: {list(missing_columns)}. "
            f"Available columns: {list(df.columns)}"
        )

    # Check for empty DataFrame
    if len(df) == 0:
        raise ValueError(
            "Cannot fit encoders on empty DataFrame. DataFrame must contain at least one row."
        )

    # Fit the transformer
    preprocessor.fit(df)

    return preprocessor


def transform_dataframe(
    df: pd.DataFrame,
    fitted_transformer: ColumnTransformer,
) -> tuple[np.ndarray, List[str]]:
    """Transform a DataFrame using a fitted ColumnTransformer.

    This function applies pre-fitted transformations to a DataFrame. The
    transformer must have been fitted previously using fit_encoders() or
    by calling .fit() directly.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to transform. Must contain all columns that the
        transformer expects.
    fitted_transformer : ColumnTransformer
        A fitted sklearn ColumnTransformer instance.

    Returns
    -------
    tuple[np.ndarray, List[str]]
        A tuple containing:
        - encoded_array : np.ndarray
            Transformed numpy array with encoded features.
        - feature_names : List[str]
            List of feature names corresponding to columns in encoded_array.

    Raises
    ------
    ValueError
        If the transformer is not fitted or if the DataFrame is empty.
    KeyError
        If the DataFrame is missing columns required by the transformer.

    Examples
    --------
    >>> # Fit on training data
    >>> fitted_transformer = fit_encoders(train_df, encoding_config)
    >>>
    >>> # Transform multiple DataFrames with same fitted transformer
    >>> train_encoded, train_features = transform_dataframe(train_df, fitted_transformer)
    >>> test_encoded, test_features = transform_dataframe(test_df, fitted_transformer)
    >>> val_encoded, val_features = transform_dataframe(val_df, fitted_transformer)
    """
    # Check that transformer is fitted
    if not hasattr(fitted_transformer, "transformers_"):
        raise ValueError(
            "ColumnTransformer must be fitted before transforming. "
            "Use fit_encoders() to fit the transformer first."
        )

    # Check for empty DataFrame
    if len(df) == 0:
        raise ValueError(
            "Cannot transform empty DataFrame. DataFrame must contain at least one row."
        )

    # Extract required columns from fitted transformer
    required_columns = set()
    for name, _, columns in fitted_transformer.transformers_:
        if name != "remainder":  # Skip the remainder transformer
            required_columns.update(columns)

    # Check for missing columns
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(
            f"Missing columns in DataFrame: {list(missing_columns)}. "
            f"Available columns: {list(df.columns)}"
        )

    # Transform the data
    encoded_array = fitted_transformer.transform(df)
    feature_names = _get_feature_names(fitted_transformer)

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
