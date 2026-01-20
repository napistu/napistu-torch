"""
Torch-accelerated versions of matrix operations.

Public Functions
----------------
compute_confusion_matrix(predictions, true_labels, normalize=None)
    Compute confusion matrix from prediction scores and true labels.
compute_correlation(x, y, method='spearman', device=None)
    Compute correlation between two vectors using PyTorch with proper memory management.
compute_correlation_matrix(data, method='spearman', device=None)
    Compute pairwise correlation matrix between columns using optimized tensor operations.
compute_correlation_matrix_numpy(data, method='spearman')
    Compute pairwise correlation matrix using scipy (for testing/comparison or if p-values are needed).
compute_cosine_distances_torch(tensor_like, device=None)
    Compute cosine distance matrix using PyTorch with proper memory management.
compute_effective_dimensionality(vectors)
    Compute effective dimensionality (inverse participation ratio) for each vector.
compute_max_abs_over_z(attention_3d, return_indices=False)
    Find maximum absolute attention values across layers (z dimension), preserving sign.
compute_max_over_z(attention_3d, return_indices=False)
    Find maximum attention values across layers (z dimension) without taking absolute value.
compute_tensor_ranks(tensor, by_absolute_value=True)
    Compute rank matrix for a tensor.
find_top_k(tensor, k, by_absolute_value=True)
    Extract top-k values and indices from a 2D tensor.
validate_tensor_for_nan_inf(tensor, name)
    Validate tensor for NaN/Inf values and raise informative error if found.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torch import device as torch_device

from napistu_torch.utils.constants import CORRELATION_METHODS
from napistu_torch.utils.torch_utils import (
    cleanup_tensors,
    ensure_device,
    memory_manager,
)

logger = logging.getLogger(__name__)


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


def compute_correlation(
    x: Union[np.ndarray, Tensor],
    y: Union[np.ndarray, Tensor],
    method: str = CORRELATION_METHODS.SPEARMAN,
    device: Optional[Union[str, torch_device]] = None,
    verbose: bool = False,
) -> float:
    """
    Compute correlation between two vectors using PyTorch with proper memory management.

    Parameters
    ----------
    x : array-like
        First vector (numpy array or torch.Tensor)
    y : array-like
        Second vector (numpy array or torch.Tensor)
    method : str, optional
        Correlation method to use:
        - 'spearman' (default): Spearman rank correlation (robust to monotonic relationships)
        - 'pearson': Pearson correlation (measures linear relationships)
    device : Optional[Union[str, torch_device]]
        The device to use for the computation. If None, the device will be automatically selected.
    verbose : bool, optional
        If True, print verbose output.

    Returns
    -------
    correlation : float
        Correlation coefficient between x and y
        - Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation)
        - 0 indicates no correlation

    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
    >>> corr = compute_correlation(x, y, method='pearson')
    >>> print(corr)  # Should be close to 1.0 (perfect positive correlation)
    """
    device = ensure_device(device, allow_autoselect=True)

    with memory_manager(device):
        # Convert to torch tensors if needed
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float().to(device)
        else:
            x_tensor = x.to(device).float() if hasattr(x, "to") else x

        if isinstance(y, np.ndarray):
            y_tensor = torch.from_numpy(y).float().to(device)
        else:
            y_tensor = y.to(device).float() if hasattr(y, "to") else y

        # Check if inputs are multi-dimensional and suggest alternative
        if x_tensor.ndim > 1 or y_tensor.ndim > 1:
            error_parts = []
            if x_tensor.ndim > 1:
                error_parts.append(f"x has shape {x_tensor.shape} (expected 1D vector)")
            if y_tensor.ndim > 1:
                error_parts.append(f"y has shape {y_tensor.shape} (expected 1D vector)")

            error_msg = "Expected 1D vectors, but got: " + ", ".join(error_parts) + ". "
            error_msg += (
                "If you want to compute a correlation matrix between columns, use "
            )
            error_msg += f"compute_correlation_matrix(data, method='{method}') instead."

            raise ValueError(error_msg)

        # Check that vectors have the same length
        if len(x_tensor) != len(y_tensor):
            raise ValueError(
                f"Vectors must have the same length. Got x: {len(x_tensor)}, y: {len(y_tensor)}"
            )

        if verbose:
            logger.info(
                f"Computing {method.capitalize()} correlation between length {len(x_tensor)} vectors on device {device}"
            )

        # For Spearman, convert to ranks first
        if method == CORRELATION_METHODS.SPEARMAN:
            # Use the utility function for consistent tie handling
            # Stack x and y as columns, compute ranks, then extract
            data = torch.stack([x_tensor, y_tensor], dim=1)
            ranks = _compute_spearman_ranks(data, device=device)
            x_data = ranks[:, 0]
            y_data = ranks[:, 1]
        elif method == CORRELATION_METHODS.PEARSON:
            x_data = x_tensor
            y_data = y_tensor
        else:
            raise ValueError(
                f"Unknown method: {method}. Use {CORRELATION_METHODS.SPEARMAN} or {CORRELATION_METHODS.PEARSON}"
            )

        # Calculate Pearson correlation
        x_centered = x_data - x_data.mean()
        y_centered = y_data - y_data.mean()

        # Compute correlation: cov(x,y) / (std(x) * std(y))
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered**2).sum()) * torch.sqrt(
            (y_centered**2).sum()
        )

        # Handle edge case: if denominator is zero (constant vectors)
        if denominator == 0:
            correlation = torch.tensor(0.0, device=device)
        else:
            correlation = numerator / denominator

        result = correlation.item()

        return result


def compute_correlation_matrix(
    data: Tensor | np.ndarray,
    method: str = CORRELATION_METHODS.SPEARMAN,
    device: Optional[Union[str, torch_device]] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute pairwise correlation matrix between columns of a data matrix.

    Calculates correlation coefficients between all pairs of columns (features)
    in the input data using optimized tensor operations. Much faster than
    scipy-based implementations, especially for large matrices.

    Parameters
    ----------
    data : torch.Tensor or np.ndarray
        Shape [n_samples, n_features] - data matrix where each column represents
        a feature/variable and each row represents a sample/observation
    method : str, optional
        Correlation method to use:
        - 'spearman' (default): Spearman rank correlation (robust to monotonic relationships)
        - 'pearson': Pearson correlation (measures linear relationships)
    device : Optional[Union[str, torch_device]]
        The device to use for the computation. If None, the device will be automatically selected.
    verbose : bool, optional
        If True, print verbose output.

    Returns
    -------
    correlation_matrix : np.ndarray
        Shape [n_features, n_features] - symmetric correlation matrix where
        element [i, j] is the correlation coefficient between feature i and feature j
        - Diagonal elements are always 1.0 (perfect self-correlation)
        - Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation)
        - Matrix is symmetric: correlation_matrix[i, j] == correlation_matrix[j, i]

    Examples
    --------
    >>> # Example: correlation between prediction scores for different classes
    >>> scores = torch.tensor([[0.8, 0.5, 0.3],
    ...                        [0.7, 0.6, 0.2],
    ...                        [0.9, 0.4, 0.1],
    ...                        [0.6, 0.7, 0.3]])
    >>> corr_matrix = compute_correlation_matrix(scores)
    >>> print(corr_matrix.shape)  # (3, 3) - one row/col per feature

    >>> # High positive correlation means features tend to increase/decrease together
    >>> # High negative correlation means features have opposite trends
    >>> # Near-zero correlation means features are independent

    >>> # Example: check if two features are highly correlated (redundant)
    >>> if corr_matrix[0, 1] > 0.9:
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

    **Performance:**
    This implementation uses optimized tensor operations instead of loops,
    making it significantly faster than scipy-based approaches for large matrices.
    """
    device = ensure_device(device, allow_autoselect=True)

    with memory_manager(device):
        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data_tensor = torch.from_numpy(data).float().to(device)
        else:
            data_tensor = data.float().to(device)

        n_samples = data_tensor.shape[0]

        if verbose:
            n_features = data_tensor.shape[1]
            logger.info(
                f"Computing {method.capitalize()} correlation for {n_samples} samples and {n_features} features on device {device}"
            )

        # For Spearman, convert to ranks first using utility function
        if method == CORRELATION_METHODS.SPEARMAN:
            data_for_corr = _compute_spearman_ranks(data_tensor, device=device)
        elif method == CORRELATION_METHODS.PEARSON:
            data_for_corr = data_tensor
        else:
            raise ValueError(
                f"Unknown method: {method}. Use {CORRELATION_METHODS.SPEARMAN} or {CORRELATION_METHODS.PEARSON}"
            )

        # Center the data (subtract mean of each column)
        data_centered = data_for_corr - data_for_corr.mean(dim=0, keepdim=True)

        # Compute standard deviations for each column
        std_devs = data_centered.std(dim=0, unbiased=True)

        # Avoid division by zero (handle constant columns)
        std_devs = torch.clamp(std_devs, min=1e-8)

        # Compute covariance matrix: (X^T @ X) / (n-1)
        # Shape: [n_features, n_features]
        cov_matrix = (data_centered.T @ data_centered) / (n_samples - 1)

        # Compute correlation matrix: cov / (std_i * std_j)
        # Outer product of standard deviations
        std_outer = std_devs.unsqueeze(1) @ std_devs.unsqueeze(0)
        correlation_matrix = cov_matrix / std_outer

        # Set diagonal to 1.0 (perfect self-correlation)
        correlation_matrix.fill_diagonal_(1.0)

        # Convert back to numpy
        correlation_matrix_np = correlation_matrix.cpu().numpy()

        # Clean up
        cleanup_tensors(
            data_tensor,
            data_for_corr,
            data_centered,
            std_devs,
            cov_matrix,
            std_outer,
            correlation_matrix,
        )

    return correlation_matrix_np


def compute_correlation_matrix_numpy(
    data: Tensor | np.ndarray,
    method: str = CORRELATION_METHODS.SPEARMAN,
) -> np.ndarray:
    """
    Compute pairwise correlation matrix between columns using scipy (for testing).

    This is a reference implementation using scipy that can be used for comparison
    and testing against the optimized tensor-based implementation.

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

    Notes
    -----
    This function uses scipy's pairwise correlation functions in a loop, which is
    slower than the optimized tensor-based implementation but useful for testing
    and validation purposes.
    """
    # Convert to numpy if torch tensor
    if isinstance(data, Tensor):
        data = data.cpu().numpy()

    n_features = data.shape[1]

    # Initialize matrix
    correlation_matrix = np.zeros((n_features, n_features))

    # Compute pairwise correlations
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                # Perfect self-correlation
                correlation_matrix[i, j] = 1.0
            else:
                if method == CORRELATION_METHODS.SPEARMAN:
                    corr, _ = spearmanr(data[:, i], data[:, j])
                elif method == CORRELATION_METHODS.PEARSON:
                    corr, _ = pearsonr(data[:, i], data[:, j])
                else:
                    raise ValueError(
                        f"Unknown method: {method}. Use {CORRELATION_METHODS.SPEARMAN} or {CORRELATION_METHODS.PEARSON}"
                    )

                correlation_matrix[i, j] = corr

    return correlation_matrix


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


def compute_tensor_ranks(
    tensor: Tensor,
    by_absolute_value: bool = False,
    ascending: bool = False,
    handle_ties: bool = False,
) -> Tensor:
    """
    Compute rank matrix for a tensor.

    Converts a tensor to a rank matrix where each value is replaced by its
    integer rank (1 = highest, 2 = second highest, etc.). This allows O(1) rank
    lookup via indexing instead of O(log n) searchsorted per query.

    Parameters
    ----------
    tensor : Tensor
        Tensor of any shape to compute ranks for
    by_absolute_value : bool, optional
        If True, rank by absolute value (default: False).
        If False, rank by raw value.
    ascending : bool, optional
        If True, rank in ascending order (1 = smallest) (default: False).
        If False, rank in descending order (1 = largest).
    handle_ties : bool, optional
        If True, tied values receive average rank (default: False).
        If False, tied values receive arbitrary sequential ranks.
        Required for Spearman correlation.

    Returns
    -------
    Tensor
        Rank tensor of same shape as input.
        - If handle_ties=False: dtype=int64
        - If handle_ties=True: dtype=float32 (for average ranks)

    Examples
    --------
    >>> # Compute rank matrix for attention tensor
    >>> attention = torch.randn(100, 100)
    >>> rank_matrix = compute_tensor_ranks(attention)
    >>> # Extract ranks via indexing (very fast!)
    >>> edge_ranks = rank_matrix[from_idx, to_idx]
    >>>
    >>> # Spearman correlation ranks with tie handling
    >>> data = torch.tensor([1.0, 3.0, 2.0, 3.0, 5.0])
    >>> ranks = compute_tensor_ranks(data, ascending=True, handle_ties=True)
    >>> # Result: [1.0, 3.5, 2.0, 3.5, 5.0] - tied 3.0s get average rank
    """
    device = tensor.device

    # Flatten tensor
    if by_absolute_value:
        values = torch.abs(tensor).flatten()
    else:
        values = tensor.flatten()

    n_elements = values.shape[0]

    # Get argsort indices
    sorted_indices = torch.argsort(values, descending=(not ascending))

    if handle_ties:
        # Use float32 for average ranks
        ranks_flat = torch.zeros(n_elements, dtype=torch.float32, device=device)
        ranks_flat[sorted_indices] = torch.arange(
            1, n_elements + 1, dtype=torch.float32, device=device
        )

        # Check if there are any ties
        unique_vals, inverse_indices = torch.unique(values, return_inverse=True)

        if len(unique_vals) < n_elements:  # Ties exist
            # Compute average rank for each unique value
            rank_sums = torch.zeros(
                len(unique_vals), device=device, dtype=torch.float32
            )
            rank_counts = torch.zeros(
                len(unique_vals), device=device, dtype=torch.float32
            )

            rank_sums.scatter_add_(0, inverse_indices, ranks_flat)
            rank_counts.scatter_add_(0, inverse_indices, torch.ones_like(ranks_flat))

            avg_ranks = rank_sums / rank_counts
            ranks_flat = avg_ranks[inverse_indices]

            cleanup_tensors(
                rank_sums, rank_counts, avg_ranks, unique_vals, inverse_indices
            )
    else:
        # Use int64 for sequential ranks (original behavior)
        ranks_flat = torch.zeros(n_elements, dtype=torch.int64, device=device)
        ranks_flat[sorted_indices] = torch.arange(
            1, n_elements + 1, dtype=torch.int64, device=device
        )

    cleanup_tensors(values, sorted_indices)

    # Reshape back to original shape
    rank_tensor = ranks_flat.reshape(tensor.shape)
    cleanup_tensors(ranks_flat)

    return rank_tensor


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

    cleanup_tensors(row_indices, col_indices, values, sort_idx)

    return sorted_row_indices, sorted_col_indices, sorted_values


def compute_tensor_ranks_for_indices(
    tensor: Tensor,
    indices: tuple[Tensor, ...],
    by_absolute_value: bool = True,
) -> Tensor:
    """
    Compute ranks for specific indices in a tensor without building full rank matrix.

    This is a memory-efficient alternative to `compute_tensor_ranks` when you only
    need ranks for a subset of indices. Instead of building a full rank matrix, it
    computes ranks on-the-fly using binary search on a sorted array.

    Parameters
    ----------
    tensor : Tensor
        Tensor of any shape to compute ranks for
    indices : tuple[Tensor, ...]
        Tuple of index tensors for multi-dimensional indexing.
        For 2D tensor, use (row_indices, col_indices).
        Each tensor should be 1D and have the same length.
    by_absolute_value : bool, optional
        If True, rank by absolute value (default: True).
        If False, rank by raw value.

    Returns
    -------
    Tensor
        1D tensor of ranks (dtype=int64) for the specified indices.
        Rank 1 = highest value, rank 2 = second highest, etc.
        Length equals the length of the index tensors.

    Examples
    --------
    >>> # Compute ranks for specific edges only (memory-efficient!)
    >>> attention = torch.randn(100, 100)
    >>> from_idx = torch.tensor([0, 5, 10])
    >>> to_idx = torch.tensor([1, 6, 11])
    >>> edge_ranks = compute_tensor_ranks_for_indices(
    ...     attention, (from_idx, to_idx)
    ... )
    >>> # edge_ranks[i] = rank of attention[from_idx[i], to_idx[i]]
    """

    # Flatten tensor and get values at indices
    if by_absolute_value:
        values = torch.abs(tensor).flatten()
    else:
        values = tensor.flatten()

    # Convert multi-dimensional indices to flat indices
    # For 2D: flat_idx = row * n_cols + col
    if len(indices) == 2:
        row_idx, col_idx = indices
        n_cols = tensor.shape[1]
        flat_indices = row_idx * n_cols + col_idx
    elif len(indices) == 1:
        flat_indices = indices[0]
    else:
        # For higher dimensions, use ravel_multi_index equivalent
        # For now, support up to 2D explicitly, raise error for higher
        raise NotImplementedError(
            f"Multi-dimensional indexing with {len(indices)} dimensions not yet supported. "
            "Currently supports 1D and 2D tensors."
        )

    # Get values at the specified indices
    query_values = values[flat_indices]

    # Sort all values in descending order for searchsorted
    sorted_values, _ = torch.sort(values, descending=True)

    # Free original values array
    cleanup_tensors(values)

    # Use searchsorted to find rank positions
    # sorted_values is descending: [10, 8, 5, 3, 1] where rank 1 = highest (10)
    # For query value 5, we want rank 3 (it's the 3rd highest)
    # searchsorted assumes ascending order, so flip to ascending: [1, 3, 5, 8, 10]
    # Find rightmost insertion point: searchsorted([1,3,5,8,10], 5, right=True) = 3
    # Convert to descending position: len - insertion_point = 5 - 3 = 2
    # Rank = descending_position + 1 = 2 + 1 = 3 ✓
    sorted_values_asc = torch.flip(sorted_values, dims=[0])
    insertion_points = torch.searchsorted(sorted_values_asc, query_values, right=True)
    ranks = len(sorted_values) - insertion_points + 1

    # Free intermediate tensors
    cleanup_tensors(sorted_values, sorted_values_asc, insertion_points)

    return ranks


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


def _compute_spearman_ranks(
    data_tensor: Tensor,
    device: Optional[Union[str, torch_device]] = None,
) -> Tensor:
    """
    Compute Spearman ranks for each column of a data matrix with tie handling.

    Converts data to ranks where tied values receive the average rank of their group.
    This is used for Spearman correlation computation.

    Parameters
    ----------
    data_tensor : torch.Tensor
        Shape [n_samples, n_features] - data matrix where each column represents
        a feature/variable and each row represents a sample/observation
    device : Optional[Union[str, torch_device]]
        The device to use for the computation. If None, uses the device of data_tensor.

    Returns
    -------
    ranks : torch.Tensor
        Shape [n_samples, n_features] - rank matrix where each column contains
        the ranks of the corresponding column in data_tensor, with ties receiving
        average ranks (matching scipy's behavior)

    Examples
    --------
    >>> data = torch.tensor([[1.0, 5.0, 3.0],
    ...                      [2.0, 5.0, 1.0],
    ...                      [2.0, 3.0, 2.0]])
    >>> ranks = compute_spearman_ranks_torch(data)
    >>> # Column 0: [1.0, 2.0, 2.0] -> ranks [1.0, 2.5, 2.5] (ties get average)
    >>> # Column 1: [5.0, 5.0, 3.0] -> ranks [2.5, 2.5, 1.0]
    >>> # Column 2: [3.0, 1.0, 2.0] -> ranks [3.0, 1.0, 2.0]
    """
    if device is None:
        device = data_tensor.device
    else:
        device = ensure_device(device, allow_autoselect=False)
        data_tensor = data_tensor.to(device)

    n_samples, n_features = data_tensor.shape
    ranks = torch.zeros((n_samples, n_features), dtype=torch.float32, device=device)

    # Compute ranks for each column
    for col_idx in range(n_features):
        ranks[:, col_idx] = compute_tensor_ranks(
            data_tensor[:, col_idx],
            by_absolute_value=False,
            ascending=True,
            handle_ties=True,
        )

    return ranks
