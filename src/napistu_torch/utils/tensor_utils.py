"""Torch-accelerated versions of matrix operations."""

from typing import Union

import numpy as np
import torch

from napistu_torch.utils.torch_utils import ensure_device, memory_manager


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
