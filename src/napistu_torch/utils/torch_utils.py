"""Utility functions for managing torch devices and memory."""

import gc
import logging
from contextlib import contextmanager
from typing import Union

import torch

from napistu_torch.ml.constants import DEVICE

logger = logging.getLogger(__name__)

# memory management utilities


@contextmanager
def memory_manager(device: torch.device = torch.device(DEVICE.CPU)):
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


def ensure_device(device: Union[str, torch.device]) -> torch.device:
    """
    Ensure the device is a torch.device.

    Parameters
    ----------
    device : Union[str, torch.device]
        The device to ensure
    """

    if isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, torch.device):
        return device
    else:
        raise ValueError(
            f"Invalid device: {device} value, must be a string or torch.device"
        )


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
        return torch.device(DEVICE.MPS)
    elif torch.cuda.is_available():
        return torch.device(DEVICE.GPU)
    else:
        return torch.device(DEVICE.CPU)
