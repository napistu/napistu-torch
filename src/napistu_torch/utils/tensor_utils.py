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
