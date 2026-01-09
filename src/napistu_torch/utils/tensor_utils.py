"""Torch-accelerated versions of matrix operations."""

from typing import Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torch import device as torch_device

from napistu_torch.utils.constants import CORRELATION_METHODS
from napistu_torch.utils.torch_utils import ensure_device, memory_manager


def compute_confusion_matrix(
    predictions: Tensor | np.ndarray,
    true_labels: Tensor | np.ndarray,
    normalize: str | None = None,
) -> np.ndarray:
    """
    Compute confusion matrix from prediction scores and true labels.

    Takes prediction scores/logits for each class and compares the argmax
    predictions against true class labels.

    Parameters
    ----------
    predictions : torch.Tensor or np.ndarray
        Shape [n_samples, n_classes] - scores/probabilities/logits for each class.
        Predicted class is determined by argmax over the class dimension.
    true_labels : torch.Tensor or np.ndarray
        Shape [n_samples] - integer indices of true classes (0 to n_classes-1)
    normalize : str, optional
        Normalization mode for confusion matrix:
        - 'true': normalize over true labels (rows sum to 1)
          Shows recall-like metrics: proportion of each true class predicted as each class
        - 'pred': normalize over predicted labels (columns sum to 1)
          Shows precision-like metrics: proportion of each predicted class from each true class
        - 'all': normalize over all samples (entire matrix sums to 1)
          Shows overall proportion of samples in each true/pred combination
        - None: no normalization, returns raw counts

    Returns
    -------
    cm : np.ndarray
        Confusion matrix of shape [n_classes, n_classes]
        - Rows represent true classes
        - Columns represent predicted classes
        - cm[i, j] is the count (or proportion) of samples with true class i
          predicted as class j

    Examples
    --------
    >>> predictions = torch.tensor([[0.8, 0.1, 0.1],
    ...                             [0.2, 0.7, 0.1],
    ...                             [0.1, 0.2, 0.7]])
    >>> true_labels = torch.tensor([0, 1, 2])
    >>> cm = compute_confusion_matrix(predictions, true_labels)
    >>> print(cm)  # Perfect predictions: identity matrix
    [[1 0 0]
     [0 1 0]
     [0 0 1]]

    >>> # Example with misclassifications
    >>> predictions = torch.tensor([[0.3, 0.6, 0.1],  # True: 0, Pred: 1
    ...                             [0.2, 0.7, 0.1],  # True: 1, Pred: 1
    ...                             [0.4, 0.5, 0.1]]) # True: 2, Pred: 1
    >>> true_labels = torch.tensor([0, 1, 2])
    >>> cm = compute_confusion_matrix(predictions, true_labels, normalize='true')
    >>> print(cm)  # All predictions went to class 1
    [[0. 1. 0.]
     [0. 1. 0.]
     [0. 1. 0.]]

    Notes
    -----
    This function is useful for multi-class classification evaluation where you have
    model outputs (logits, probabilities, or any scores) and want to assess how well
    the argmax predictions match the true labels.

    The diagonal elements represent correct predictions, while off-diagonal elements
    represent misclassifications.
    """
    # Convert to numpy if torch tensor
    if isinstance(predictions, Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(true_labels, Tensor):
        true_labels = true_labels.cpu().numpy()

    # Get predicted labels (argmax over class dimension)
    predicted_labels = predictions.argmax(axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, normalize=normalize)

    return cm


def compute_correlation_matrix(
    data: Tensor | np.ndarray,
    method: str = CORRELATION_METHODS.SPEARMAN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise correlation matrix between columns of a data matrix.

    Calculates correlation coefficients between all pairs of columns (features)
    in the input data. Useful for analyzing relationships between features,
    identifying redundant features, or understanding feature similarity patterns.

    Parameters
    ----------
    data : torch.Tensor or np.ndarray
        Shape [n_samples, n_features] - data matrix where each column represents
        a feature/variable and each row represents a sample/observation
    method : str, optional
        Correlation method to use:
        - 'spearman' (default): Spearman rank correlation (robust to monotonic relationships)
        - 'pearson': Pearson correlation (measures linear relationships)

    Returns
    -------
    correlation_matrix : np.ndarray
        Shape [n_features, n_features] - symmetric correlation matrix where
        element [i, j] is the correlation coefficient between feature i and feature j
        - Diagonal elements are always 1.0 (perfect self-correlation)
        - Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation)
        - Matrix is symmetric: correlation_matrix[i, j] == correlation_matrix[j, i]
    p_values : np.ndarray
        Shape [n_features, n_features] - matrix of p-values for testing non-correlation
        - Element [i, j] is the p-value for the null hypothesis that features i and j
          are uncorrelated
        - Diagonal p-values are 0.0 (self-correlation is always significant)
        - Small p-values (e.g., < 0.05) indicate statistically significant correlations

    Examples
    --------
    >>> # Example: correlation between prediction scores for different classes
    >>> scores = torch.tensor([[0.8, 0.5, 0.3],
    ...                        [0.7, 0.6, 0.2],
    ...                        [0.9, 0.4, 0.1],
    ...                        [0.6, 0.7, 0.3]])
    >>> corr_matrix, p_vals = compute_correlation_matrix(scores)
    >>> print(corr_matrix.shape)  # (3, 3) - one row/col per feature

    >>> # High positive correlation means features tend to increase/decrease together
    >>> # High negative correlation means features have opposite trends
    >>> # Near-zero correlation means features are independent

    >>> # Example: check if two features are highly correlated (redundant)
    >>> if corr_matrix[0, 1] > 0.9 and p_vals[0, 1] < 0.05:
    ...     print("Features 0 and 1 are highly correlated")

    Notes
    -----
    **Spearman vs Pearson:**
    - Use Spearman (default) for:
      - Monotonic but non-linear relationships
      - Data with outliers (more robust)
      - Ordinal data or ranks
    - Use Pearson for:
      - Linear relationships
      - Normally distributed data
      - When you want to detect only linear associations

    The correlation matrix is symmetric and can be visualized using plot_heatmap()
    with mask_upper_triangle=True to avoid redundant display.
    """
    # Convert to numpy if torch tensor
    if isinstance(data, Tensor):
        data = data.cpu().numpy()

    n_features = data.shape[1]

    # Initialize matrices
    correlation_matrix = np.zeros((n_features, n_features))
    p_values = np.zeros((n_features, n_features))

    # Compute pairwise correlations
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                # Perfect self-correlation
                correlation_matrix[i, j] = 1.0
                p_values[i, j] = 0.0
            else:
                if method == CORRELATION_METHODS.SPEARMAN:
                    corr, pval = spearmanr(data[:, i], data[:, j])
                elif method == CORRELATION_METHODS.PEARSON:
                    corr, pval = pearsonr(data[:, i], data[:, j])
                else:
                    raise ValueError(
                        f"Unknown method: {method}. Use {CORRELATION_METHODS.SPEARMAN} or {CORRELATION_METHODS.PEARSON}"
                    )

                correlation_matrix[i, j] = corr
                p_values[i, j] = pval

    return correlation_matrix, p_values


def compute_cosine_distances_torch(
    tensor_like: Union[np.ndarray, Tensor],
    device: Optional[Union[str, torch_device]] = None,
) -> np.ndarray:
    """
    Compute cosine distance matrix using PyTorch with proper memory management

    Parameters
    ----------
    tensor_like : Union[np.ndarray, torch.Tensor]
        The tensor to compute the cosine distances for
    device : Optional[Union[str, torch_device]]
        The device to use for the computation. If None, the device will be automatically selected.

    Returns
    -------
    cosine_dist : np.ndarray
        The cosine distance matrix
    """

    device = ensure_device(device, allow_autoselect=True)
    with memory_manager(device):
        # convert the embedding to a tensor and move it to the device
        if isinstance(tensor_like, np.ndarray):
            tensor = torch.tensor(tensor_like, dtype=torch.float32, device=device)
        else:
            tensor = tensor_like.to(device)

        # normalize the embeddings
        embeddings_norm = torch.nn.functional.normalize(tensor, p=2, dim=1)

        # compute the cosine similarity matrix
        cosine_sim = torch.mm(embeddings_norm, embeddings_norm.t())

        # convert to cosine distance
        cosine_dist = 1 - cosine_sim

        # move back to the cpu and convert to numpy
        result = cosine_dist.cpu().numpy()

        return result


def compute_effective_dimensionality(vectors: Tensor) -> np.ndarray:
    """
    Compute effective dimensionality (inverse participation ratio) for each vector.

    Measures how many dimensions a vector "uses".
    - If all components equal: eff_dim ≈ n (fully distributed)
    - If one component dominates: eff_dim ≈ 1 (maximally sparse)

    Formula: (sum of squares)^2 / (sum of fourth powers)

    Parameters
    ----------
    vectors : torch.Tensor
        Shape [num_vectors, embedding_dim]

    Returns
    -------
    np.ndarray
        Effective dimensionality for each vector
    """
    vec_sq = vectors**2
    sum_sq = vec_sq.sum(dim=1)
    sum_fourth = (vec_sq**2).sum(dim=1)

    # Avoid division by zero
    eff_dim = torch.where(
        sum_fourth > 0, sum_sq**2 / sum_fourth, torch.zeros_like(sum_sq)
    )

    return eff_dim.numpy()


def compute_max_abs_over_z(
    attention_3d: Tensor,
    return_indices: bool = False,
) -> Tensor:
    """
    Find maximum absolute attention values across layers (z dimension), preserving sign.

    For each feature pair (i, j), finds the layer with max |attention| and
    returns that attention value with its original sign.

    Parameters
    ----------
    attention_3d : torch.Tensor
        Attention scores of shape (n_x, n_y, n_z)
    return_indices : bool, optional
        If True, also return the layer indices where max occurred

    Returns
    -------
    torch.Tensor
        Max absolute attention with sign preserved, shape (n_x, n_y)
    torch.Tensor (optional)
        Layer indices where max occurred, shape (n_x, n_y)
    """

    # check for valid shape
    if attention_3d.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {attention_3d.shape}")

    n_x, n_y, _ = attention_3d.shape

    if n_x != n_y:
        raise ValueError(f"Expected square tensor, got shape {attention_3d.shape}")

    # Find layer with max absolute value for each feature pair (across z dimension)
    max_abs_indices = torch.abs(attention_3d).argmax(dim=2)  # (n_x, n_y)

    # Create index grids for feature dimensions
    feature_i_idx = torch.arange(n_x).unsqueeze(1).expand(n_x, n_y)
    feature_j_idx = torch.arange(n_y).unsqueeze(0).expand(n_x, n_y)

    # Index into attention_3d to get signed values
    max_abs_signed = attention_3d[feature_i_idx, feature_j_idx, max_abs_indices]

    if return_indices:
        return max_abs_signed, max_abs_indices
    return max_abs_signed


def compute_max_over_z(
    attention_3d: Tensor,
    return_indices: bool = False,
) -> Tensor:
    """
    Find maximum attention values across layers (z dimension) without taking absolute value.

    For each feature pair (i, j), finds the layer with max attention value and
    returns that attention value.

    Parameters
    ----------
    attention_3d : torch.Tensor
        Attention scores of shape (n_x, n_y, n_z)
    return_indices : bool, optional
        If True, also return the layer indices where max occurred

    Returns
    -------
    torch.Tensor
        Max attention, shape (n_x, n_y)
    torch.Tensor (optional)
        Layer indices where max occurred, shape (n_x, n_y)
    """
    # check for valid shape
    if attention_3d.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {attention_3d.shape}")

    n_x, n_y, _ = attention_3d.shape

    if n_x != n_y:
        raise ValueError(f"Expected square tensor, got shape {attention_3d.shape}")

    # Find layer with max value for each feature pair (across z dimension)
    max_indices = attention_3d.argmax(dim=2)  # (n_x, n_y)

    # Create index grids for feature dimensions
    feature_i_idx = torch.arange(n_x).unsqueeze(1).expand(n_x, n_y)
    feature_j_idx = torch.arange(n_y).unsqueeze(0).expand(n_x, n_y)

    # Index into attention_3d to get max values
    max_values = attention_3d[feature_i_idx, feature_j_idx, max_indices]

    if return_indices:
        return max_values, max_indices
    return max_values


def compute_spearman_correlation_torch(
    x: Union[np.ndarray, Tensor],
    y: Union[np.ndarray, Tensor],
    device: Optional[Union[str, torch_device]] = None,
) -> float:
    """
    Compute Spearman correlation using PyTorch with proper memory management

    Parameters
    ----------
    x : array-like
        First vector (numpy array or similar)
    y : array-like
        Second vector (numpy array or similar)
    device : Optional[Union[str, torch_device]]
        The device to use for the computation. If None, the device will be automatically selected.

    Returns
    -------
    rho : float
        Spearman correlation coefficient
    """

    device = ensure_device(device, allow_autoselect=True)
    with memory_manager(device):
        # Convert to torch tensors if needed
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float().to(device)
        else:
            x_tensor = x.to(device) if hasattr(x, "to") else x

        if isinstance(y, np.ndarray):
            y_tensor = torch.from_numpy(y).float().to(device)
        else:
            y_tensor = y.to(device) if hasattr(y, "to") else y

        # Convert values to ranks
        x_rank = torch.argsort(torch.argsort(x_tensor)).float()
        y_rank = torch.argsort(torch.argsort(y_tensor)).float()

        # Calculate Pearson correlation on ranks
        x_centered = x_rank - x_rank.mean()
        y_centered = y_rank - y_rank.mean()

        correlation = (x_centered * y_centered).sum() / (
            torch.sqrt((x_centered**2).sum()) * torch.sqrt((y_centered**2).sum())
        )

        result = correlation.item()

        return result


def find_top_k(
    tensor: torch.Tensor,
    k: int,
    by_absolute_value: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract top-k values and indices from a 2D tensor.

    Flattens the tensor and finds the k largest values (optionally by absolute value),
    then converts the flattened indices back to 2D row/column indices. When there are
    tied values at the k-th position, includes ALL values >= the k-th value, which may
    result in more than k results.

    Parameters
    ----------
    tensor : torch.Tensor
        2D tensor of shape (n_rows, n_cols)
    k : int
        Number of top values to extract
    by_absolute_value : bool, optional
        If True, rank by absolute value but return original signed values (default: True).
        If False, rank by raw values (largest values).

    Returns
    -------
    row_indices : torch.Tensor
        Row indices of top-k values, shape (k,) or larger if ties exist
    col_indices : torch.Tensor
        Column indices of top-k values, shape (k,) or larger if ties exist
    values : torch.Tensor
        Top-k values with original sign preserved, shape (k,) or larger if ties exist

    Notes
    -----
    When ties exist at the k-th position, this function includes ALL tied values,
    which means the returned tensors may have length > k. This ensures no high-scoring
    edges are randomly excluded.

    Examples
    --------
    >>> # Get top 100 by absolute value
    >>> attention = torch.randn(1000, 1000)
    >>> row_idx, col_idx, values = find_top_k(attention, k=100)
    >>>
    >>> # With ties, may return more than k
    >>> tensor = torch.tensor([[1.0, 5.0], [5.0, 2.0]])
    >>> row_idx, col_idx, values = find_top_k(tensor, k=2)
    >>> len(values)  # May be 3 if both 5.0 values are in top-2
    3
    """
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tensor.shape}")

    # Get values to rank by
    if by_absolute_value:
        ranking_tensor = torch.abs(tensor)
    else:
        ranking_tensor = tensor

    # Get the k-th largest value (the threshold)
    k_actual = min(k, ranking_tensor.numel())
    kth_value = torch.topk(ranking_tensor.flatten(), k_actual).values[-1]

    # Find ALL positions where value >= kth_value (work on 2D tensor directly!)
    mask = ranking_tensor >= kth_value
    row_indices, col_indices = torch.where(mask)

    # Get original signed values at those positions
    values = tensor[row_indices, col_indices]

    # Sort by absolute value descending to maintain top-k ordering
    if by_absolute_value:
        sort_idx = torch.argsort(torch.abs(values), descending=True)
    else:
        sort_idx = torch.argsort(values, descending=True)

    sorted_row_indices = row_indices[sort_idx]
    sorted_col_indices = col_indices[sort_idx]
    sorted_values = values[sort_idx]

    del row_indices, col_indices, values, sort_idx

    return sorted_row_indices, sorted_col_indices, sorted_values


def validate_tensor_for_nan_inf(
    tensor: Tensor,
    name: str,
) -> None:
    """
    Validate tensor for NaN/Inf values and raise informative error if found.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to validate
    name : str
        Name of the tensor for error messages

    Raises
    ------
    ValueError
        If NaN or Inf values are found in the tensor
    """
    if tensor is None:
        return

    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)

    if nan_mask.any() or inf_mask.any():
        n_nan = nan_mask.sum().item()
        n_inf = inf_mask.sum().item()
        total = tensor.numel()

        error_msg = (
            f"Found {n_nan} NaN and {n_inf} Inf values in {name}. "
            f"Total elements: {total}, NaN rate: {n_nan/total:.2%}, Inf rate: {n_inf/total:.2%}."
        )

        # Add statistics about the tensor
        if not nan_mask.all() and not inf_mask.all():
            valid_values = tensor[~(nan_mask | inf_mask)]
            if len(valid_values) > 0:
                error_msg += (
                    f" Valid values: min={valid_values.min().item():.4f}, "
                    f"max={valid_values.max().item():.4f}, "
                    f"mean={valid_values.mean().item():.4f}."
                )

        raise ValueError(error_msg)
