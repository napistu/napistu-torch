from typing import Dict, Tuple, Union

import pandas as pd
import torch
from napistu.network.ng_core import NapistuGraph

from napistu_torch.labeling.constants import (
    LABEL_TYPE,
    TASK_TYPES,
    VALID_LABEL_TYPES,
    VALID_TASK_TYPES,
)
from napistu_torch.labeling.labeling_strategy import (
    LABEL_INFORMED_FEATURIZATION,
    LabelingStrategy,
)


def create_vertex_labels(
    napistu_graph: NapistuGraph,
    label_type: Union[str, LabelingStrategy] = LABEL_TYPE.SPECIES_TYPE,
    label_informed_featurization: Dict[
        str, LabelingStrategy
    ] = LABEL_INFORMED_FEATURIZATION,
):
    """
    Create vertex labels for single-label predictions tasks

    Parameters
    ----------
    napistu_graph: NapistuGraph
        A network-based representation of the SBML_dfs
    label_type: Union[str, LabelingStrategy]
        Either a string indicating the type of labels to generate (which will lookup a strategy from LABEL_INFORMED_FEATURIZATION) or a LabelingStrategy

        THe supported strings with pre-configured strategies are:
        - species_type: protein, metabolite, drug, etc.
        - node_type: protein, metabolite, drug, etc.

    label_informed_featurization: Dict[str, LabelingStrategy]
        A dictionary of LabelingStrategy objects for each label type.

    Returns
    -------
    labels: pd.Series
        A Series with labels as values and vertex names as an index
    featurization_behavior: Dict
        Changes to the featurization to ensure that label-based attributes are not
        trivially included in the vertex features

    """

    if not isinstance(napistu_graph, NapistuGraph):
        raise ValueError("napistu_graph must be a NapistuGraph")
    if label_type not in VALID_LABEL_TYPES:
        raise ValueError(
            f"label_type was {label_type} and must be one of {', '.join(VALID_LABEL_TYPES)}"
        )
    if label_type not in label_informed_featurization.keys():
        raise ValueError(
            f"The `label_type`, {label_type} is missing from the keys in `label_informed_featurization`"
        )

    labels = napistu_graph.get_vertex_series(label_type)

    # extract the label attribute
    featurization_behavior = label_informed_featurization[label_type]

    return labels, featurization_behavior


def encode_labels(
    labels: pd.Series,
    task_type: str = TASK_TYPES.CLASSIFICATION,
    missing_value: Union[int, float] = None,
) -> Union[Tuple[torch.Tensor, Dict[int, any]], torch.Tensor]:
    """
    Prepare labels for PyTorch Geometric based on task type.

    This is a convenience wrapper that calls either prepare_discrete_labels
    or prepare_continuous_labels based on the task type.

    Parameters
    ----------
    labels : pd.Series
        Series containing labels
    task_type : {'classification', 'regression'}, optional
        Type of task (default: 'classification')
    missing_value : int or float, optional
        Value to use for missing entries.
        Defaults: -1 for classification, nan for regression

    Returns
    -------
    For classification:
        encoded : torch.Tensor
            Integer-encoded labels
        lookup : Dict[int, any]
            Mapping from integers to original labels

    For regression:
        torch.Tensor
            Continuous labels as float tensor

    Examples
    --------
    >>> # Classification
    >>> labels = pd.Series(['A', 'B', 'A', None])
    >>> encoded, lookup = encode_labels(labels, task_type='classification')

    >>> # Regression
    >>> labels = pd.Series([1.5, 2.3, np.nan, 4.1])
    >>> tensor = encode_labels(labels, task_type='regression')
    """
    if task_type == TASK_TYPES.CLASSIFICATION:
        if missing_value is None:
            missing_value = -1
        return _prepare_discrete_labels(labels, missing_value=missing_value)

    elif task_type == TASK_TYPES.REGRESSION:
        if missing_value is None:
            missing_value = float("nan")
        return _prepare_continuous_labels(labels, missing_value=missing_value)

    else:
        raise ValueError(
            f"task_type must be one of {VALID_TASK_TYPES}. Got: {task_type}"
        )


def _prepare_discrete_labels(
    labels: pd.Series, missing_value: int = -1
) -> Tuple[torch.Tensor, Dict[int, Union[str, int, float]]]:
    """
    Convert a pandas Series of discrete/categorical labels to integer encoding.

    Supports:
    - String/object dtype
    - Categorical dtype
    - Integer dtype
    - Float dtype (treated as discrete categories)

    Parameters
    ----------
    labels : pd.Series
        Series containing discrete labels (can include NaN/None/pd.NA)
    missing_value : int, optional
        Integer to use for missing values (default: -1)

    Returns
    -------
    encoded : torch.Tensor
        Integer-encoded labels as a tensor (dtype=torch.long)
    lookup : Dict[int, any]
        Mapping from integer codes to original label values

    Raises
    ------
    ValueError
        If the Series dtype is not supported

    Examples
    --------
    >>> labels = pd.Series(['A', 'B', 'A', None, 'C'])
    >>> encoded, lookup = prepare_discrete_labels(labels)
    >>> print(encoded)
    tensor([0, 1, 0, -1, 2])
    """
    # Check dtype and handle accordingly
    dtype = labels.dtype

    # Handle categorical dtype
    if pd.api.types.is_categorical_dtype(dtype):
        labels = labels.astype(object)

    # Check if dtype is supported
    elif not (
        pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_integer_dtype(dtype)
        or pd.api.types.is_float_dtype(dtype)
    ):
        raise ValueError(
            f"Unsupported dtype: {dtype}. "
            f"Supported dtypes are: object, string, category, integer, and float."
        )

    # For numeric dtypes, replace NaN with None for uniform handling
    if pd.api.types.is_numeric_dtype(dtype):
        labels = labels.copy()
        labels = labels.where(pd.notna(labels), None)

    # Get unique non-null values
    unique_labels = labels.dropna().unique()

    # Sort unique labels if possible for consistency
    try:
        unique_labels = sorted(unique_labels)
    except TypeError:
        # Can't sort (mixed types or non-comparable), keep original order
        pass

    # Create mapping from label to integer
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    # Create reverse lookup (int to label)
    lookup = {idx: label for label, idx in label_to_int.items()}

    # Encode the series
    encoded = labels.map(label_to_int)
    encoded = encoded.fillna(missing_value).astype(int)

    # Convert to torch tensor
    encoded_tensor = torch.tensor(encoded.values, dtype=torch.long)

    return encoded_tensor, lookup


def _prepare_continuous_labels(
    labels: pd.Series,
    missing_value: float = float("nan"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert a pandas Series of continuous labels to a float tensor for regression.

    Parameters
    ----------
    labels : pd.Series
        Series containing continuous numeric values (can include NaN)
    missing_value : float, optional
        Value to use for missing entries (default: nan)
    dtype : torch.dtype, optional
        Torch dtype for the output tensor (default: torch.float32)

    Returns
    -------
    torch.Tensor
        Continuous labels as a tensor

    Raises
    ------
    ValueError
        If the Series dtype is not numeric

    Examples
    --------
    >>> labels = pd.Series([1.5, 2.3, np.nan, 4.1])
    >>> tensor = prepare_continuous_labels(labels)
    >>> print(tensor)
    tensor([1.5000, 2.3000,    nan, 4.1000])
    """
    if not pd.api.types.is_numeric_dtype(labels.dtype):
        raise ValueError(
            f"Continuous labels must be numeric. Got dtype: {labels.dtype}"
        )

    # Fill NaN with missing_value
    labels_filled = labels.fillna(missing_value)

    # Convert to torch tensor
    tensor = torch.tensor(labels_filled.values, dtype=dtype)

    return tensor
