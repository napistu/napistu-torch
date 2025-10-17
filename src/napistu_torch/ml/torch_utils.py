"""
Utility functions for PyTorch operations.
"""

import gc
import logging
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)

# memory management utilities


@contextmanager
def memory_manager(device: torch.device = torch.device("cpu")):
    """
    Context manager for general memory management.

    This context manager ensures proper cleanup by:
    1. Clearing device cache before and after operations
    2. Forcing garbage collection

    Parameters
    ----------
    device : torch.device
        The device to manage memory for

    Usage:
        with memory_manager(device):
            # Your operations here
            pass
    """
    # Clear cache before starting
    if device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        yield
    finally:
        # Clear cache after operations
        if device.type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()


def cleanup_tensors(*tensors) -> None:
    """
    Explicitly clean up one or more tensors and free their memory.

    Parameters
    ----------
    *tensors : torch.Tensor
        One or more tensors to clean up
    """
    for tensor in tensors:
        if tensor is not None:
            del tensor


# torch utils


def select_device(mps_valid: bool = True):
    """
    Selects the device to use for the model.
    If MPS is available and mps_valid is True, use MPS.
    If CUDA is available, use CUDA.
    Otherwise, use CPU.

    Parameters
    ----------
    mps_valid : bool
        Whether to use MPS if available.

    Returns
    -------
    device : torch.device
        The device to use for the model.
    """

    if mps_valid and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
