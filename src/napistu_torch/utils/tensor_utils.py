"""Torch-accelerated versions of matrix operations."""

from typing import Union

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix

from napistu_torch.utils.constants import CORRELATION_METHODS
from napistu_torch.utils.torch_utils import ensure_device, memory_manager


def compute_confusion_matrix(
    predictions: torch.Tensor | np.ndarray,
    true_labels: torch.Tensor | np.ndarray,
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
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()

    # Get predicted labels (argmax over class dimension)
    predicted_labels = predictions.argmax(axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, normalize=normalize)

    return cm


def compute_correlation_matrix(
    data: torch.Tensor | np.ndarray,
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
    if isinstance(data, torch.Tensor):
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
    tensor_like: Union[np.ndarray, torch.Tensor], device: torch.device
) -> np.ndarray:
    """
    Compute cosine distance matrix using PyTorch with proper memory management

    Parameters
    ----------
    tensor_like : Union[np.ndarray, torch.Tensor]
        The tensor to compute the cosine distances for
    device : torch.device
        The device to use for the computation

    Returns
    -------
    cosine_dist : np.ndarray
        The cosine distance matrix
    """

    device = ensure_device(device)
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


def compute_spearman_correlation_torch(
    x: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    device: torch.device,
) -> float:
    """
    Compute Spearman correlation using PyTorch with proper memory management

    Parameters
    ----------
    x : array-like
        First vector (numpy array or similar)
    y : array-like
        Second vector (numpy array or similar)
    device : torch.device
        The device to use for the computation

    Returns
    -------
    rho : float
        Spearman correlation coefficient
    """

    device = ensure_device(device)
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


def validate_tensor_for_nan_inf(
    tensor: torch.Tensor,
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
