"""Utility functions for managing torch devices and memory."""

import gc
from contextlib import contextmanager
from typing import Optional, Union

from torch import backends, cuda, mps
from torch import device as torch_device

from napistu_torch.ml.constants import DEVICE

# memory management utilities


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


def empty_cache(device: torch_device) -> None:
    """
    Empty the cache for a given device. If the device is not MPS or GPU, do nothing.

    Parameters
    ----------
    device : torch.device
        The device to empty the cache for
    """
    if device.type == DEVICE.MPS and backends.mps.is_available():
        mps.empty_cache()
    elif device.type == DEVICE.GPU and cuda.is_available():
        cuda.empty_cache()

    return None


@contextmanager
def memory_manager(device: torch_device = torch_device(DEVICE.CPU)):
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
    empty_cache(device)

    try:
        yield
    finally:
        # Clear cache after operations
        empty_cache(device)
        # Force garbage collection
        gc.collect()


# torch utils


def ensure_device(
    device: Optional[Union[str, torch_device]], allow_autoselect: bool = False
) -> torch_device:
    """
    Ensure the device is a torch.device.

    Parameters
    ----------
    device : Union[str, torch.device]
        The device to ensure
    allow_autoselect : bool
        Whether to allow automatic selection of the device if the device is not specified
    """

    if device is None:
        if allow_autoselect:
            return select_device()
        else:
            raise ValueError("An explicit device is required but was not specified")

    if isinstance(device, str):
        return torch_device(device)
    elif isinstance(device, torch_device):
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

    if mps_valid and backends.mps.is_available():
        return torch_device(DEVICE.MPS)
    elif cuda.is_available():
        return torch_device(DEVICE.GPU)
    else:
        return torch_device(DEVICE.CPU)
