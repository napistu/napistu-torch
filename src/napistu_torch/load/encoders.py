"""Custom transformations for NapistuGraph vertex and edge features"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from napistu_torch.load.constants import ENCODING_MANAGER, ENCODINGS


def encode_sparse_continuous(values, scaler=None, fit=True):
    """
    Encode sparse continuous features as indicator + standardized value pairs.

    Each sparse continuous feature is encoded as two columns:
    1. Binary indicator (1 if present, 0 if missing/NaN)
    2. Standardized value (scaled if present, 0 if missing)

    The standardization is computed only from non-missing values to avoid
    zeros skewing the distribution statistics.

    Parameters
    ----------
    values : array-like of shape (n_samples,)
        Feature values with NaN indicating missing entries.
    scaler : StandardScaler or None, optional
        Pre-fitted scaler. If None and fit=True, a new scaler is created
        and fitted. If None and fit=False, values are not scaled.
    fit : bool, default=True
        Whether to fit a new scaler on the non-missing values. Ignored if
        scaler is provided.

    Returns
    -------
    encoded : ndarray of shape (n_samples, 2)
        Array with columns [indicator, standardized_value].
    scaler : StandardScaler or None
        The fitted scaler (newly created if fit=True, otherwise the input
        scaler). Returns None if no scaling was performed.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import StandardScaler
    >>>
    >>> # Training data
    >>> train_vals = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
    >>> encoded_train, scaler = encode_sparse_continuous(train_vals, fit=True)
    >>> print(encoded_train)
    [[1.0, -1.134...],
     [1.0, -0.378...],
     [0.0,  0.0],
     [1.0,  1.512...],
     [0.0,  0.0]]
    >>>
    >>> # Test data with same scaler
    >>> test_vals = np.array([3.0, np.nan, 5.0])
    >>> encoded_test, _ = encode_sparse_continuous(test_vals, scaler=scaler, fit=False)

    Notes
    -----
    Missing values are encoded as [0, 0] while present values are encoded as
    [1, (value - mean) / std] where mean and std are computed only from the
    non-missing values in the training data.
    """
    values = np.asarray(values).flatten()
    mask = ~np.isnan(values)

    # Create binary indicator
    indicator = mask.astype(float).reshape(-1, 1)

    # Handle scaling
    if scaler is None and fit:
        # Fit new scaler on non-missing values only
        present_values = values[mask].reshape(-1, 1)
        if len(present_values) > 0:
            scaler = StandardScaler()
            scaler.fit(present_values)

    # Create value column
    if scaler is not None and hasattr(scaler, "mean_"):
        # Scale present values, set missing to 0
        filled = values.copy()
        filled[~mask] = 0.0
        scaled = np.zeros_like(values)
        scaled[mask] = scaler.transform(filled[mask].reshape(-1, 1)).flatten()
        value_col = scaled.reshape(-1, 1)
    else:
        # No scaling - just zero out missing values
        value_col = np.where(mask, values, 0.0).reshape(-1, 1)

    # Concatenate indicator and value
    encoded = np.hstack([indicator, value_col])

    return encoded, scaler


class SparseContScaler(BaseEstimator, TransformerMixin):
    """
    Wrapper for encode_sparse_continuous to use in NapistuFeatureEncoder.

    Standardizes continuous features with missing values by encoding each
    feature as two columns: [indicator, scaled_value].

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Configuration with sparse continuous features
    >>> config = {
    ...     'sparse_num': {
    ...         'columns': ['kinetic_constant'],
    ...         'transformer': SparseContScaler()
    ...     }
    ... }
    >>>
    >>> train_df = pd.DataFrame({
    ...     'kinetic_constant': [1.0, 2.0, np.nan, 4.0, np.nan]
    ... })
    >>>
    >>> encoder = NapistuFeatureEncoder(config)
    >>> features = encoder.fit_transform(train_df)
    >>> print(features.shape)  # (5, 2) - indicator + value for 1 column
    """

    def __init__(self):
        self.scalers_ = {}

    def fit(self, X):
        """
        Fit a scaler for each column in X.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data with NaN indicating missing values.

        Returns
        -------
        self : SparseContScaler
        """
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                _, scaler = encode_sparse_continuous(X[col].values, fit=True)
                self.scalers_[col] = scaler
        else:
            # Assume single column
            _, self.scalers_[0] = encode_sparse_continuous(X.flatten(), fit=True)

        return self

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Each input feature becomes two output features:
        - is_{feature_name}: Binary indicator (1 if present, 0 if missing)
        - value_{feature_name}: Standardized value (scaled if present, 0 if missing)

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names. If None, uses feature_names_in_ from fit().

        Returns
        -------
        feature_names_out : ndarray of str
            Output feature names.

        Examples
        --------
        >>> scaler = SparseContScaler()
        >>> df = pd.DataFrame({'kinetic_constant': [1.0, 2.0, np.nan]})
        >>> scaler.fit(df)
        >>> scaler.get_feature_names_out()
        array(['is_kinetic_constant', 'value_kinetic_constant'], dtype=object)
        """
        if input_features is None:
            input_features = self.feature_names_in_

        if input_features is None:
            raise ValueError("input_features must be provided if fit() was not called")

        # Create output feature names
        output_names = []
        for feature in input_features:
            output_names.append(f"is_{feature}")
            output_names.append(f"value_{feature}")

        return np.array(output_names, dtype=object)

    def transform(self, X):
        """
        Transform X using fitted scalers.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input data with same columns as fit().

        Returns
        -------
        encoded : np.ndarray
            Transformed features with shape (n_samples, 2 * n_columns).
        """
        if isinstance(X, pd.DataFrame):
            encoded_parts = []
            for col in X.columns:
                encoded, _ = encode_sparse_continuous(
                    X[col].values, scaler=self.scalers_[col], fit=False
                )
                encoded_parts.append(encoded)
            return np.hstack(encoded_parts)
        else:
            # Assume single column
            encoded, _ = encode_sparse_continuous(
                X.flatten(), scaler=self.scalers_[0], fit=False
            )
            return encoded


DEFAULT_ENCODERS = {
    ENCODINGS.CATEGORICAL: OneHotEncoder(sparse_output=False, drop="if_binary"),
    ENCODINGS.NUMERIC: StandardScaler(),
    ENCODINGS.SPARSE_CATEGORICAL: OneHotEncoder(
        handle_unknown="ignore", drop=None, sparse_output=False
    ),
    ENCODINGS.SPARSE_NUMERIC: SparseContScaler(),
    ENCODINGS.BINARY: ENCODING_MANAGER.PASSTHROUGH,
}
