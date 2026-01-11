"""
Utility functions for managing torch devices and memory.

Public Functions
----------------
cleanup_tensors(tensors)
    Clean up tensors and empty cache for all unique devices they were on.
delete_tensors(tensors)
    Delete one or more tensors without emptying cache.
empty_cache(device)
    Empty the cache for a given device.
ensure_device(device, allow_autoselect=False)
    Ensure the device is a torch.device.
memory_manager(device)
    Context manager for general memory management.
select_device(mps_valid=True)
    Select the device to use for the model.
"""

import gc
from contextlib import contextmanager
from typing import Optional, Union

from torch import backends, cuda, mps
from torch import device as torch_device

from napistu_torch.ml.constants import DEVICE


def cleanup_tensors(*tensors) -> None:
    """
    Clean up tensors and empty cache for all unique devices they were on.

    Deletes the provided tensors and then calls empty_cache() for each unique
    device that the tensors were on. This ensures GPU/MPS memory is freed immediately.
    Non-tensor objects (e.g., DataFrames) are simply deleted without cache clearing.

    Parameters
    ----------
    *tensors : torch.Tensor or any object
        One or more tensors to clean up. Devices are automatically detected
        from the tensors themselves. Non-tensor objects are deleted but don't
        trigger cache clearing.

    Examples
    --------
    >>> # Clean up tensors on GPU
    >>> cleanup_tensors(attention, rank_tensor, edge_attentions)
    >>> # Clean up tensors on different devices
    >>> cleanup_tensors(tensor1, tensor2)  # Automatically handles both devices
    >>> # Non-tensors are handled gracefully
    >>> cleanup_tensors(tensor, df)  # DataFrame is deleted but doesn't affect cache
    """
    # Collect unique devices before deleting tensors
    devices = set()
    for tensor in tensors:
        if tensor is not None:
            # Only collect device if object has device attribute (i.e., is a tensor)
            if hasattr(tensor, "device"):
                devices.add(tensor.device)

    # Delete tensors (and any other objects)
    delete_tensors(*tensors)

    # Empty cache for each unique device
    for device in devices:
        empty_cache(device)


def delete_tensors(*tensors) -> None:
    """
    Delete one or more tensors without emptying cache.

    Parameters
    ----------
    *tensors : torch.Tensor
        One or more tensors to delete
    """
    for tensor in tensors:
        if tensor is not None:
            del tensor


def empty_cache(device: Union[str, torch_device]) -> None:
    """
    Empty the cache for a given device. If the device is not MPS or GPU, do nothing.

    Parameters
    ----------
    device : str or torch.device
        The device to empty the cache for. Can be a string like 'cuda:0' or 'mps',
        or a torch.device object.
    """
    # Normalize to torch.device if string
    if isinstance(device, str):
        device = torch_device(device)

    if device.type == DEVICE.MPS and backends.mps.is_available():
        mps.empty_cache()
    elif device.type == DEVICE.GPU and cuda.is_available():
        cuda.empty_cache()

    return None


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
