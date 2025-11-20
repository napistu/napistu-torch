"""Test utility functions."""

import torch


def assert_tensors_equal(tensor1, tensor2, msg=None):
    """
    Assert that two tensors are equal, handling NaN values correctly.

    This function properly compares tensors that may contain NaN values.
    Unlike torch.equal(), this function treats NaN == NaN as True.

    Parameters
    ----------
    tensor1 : torch.Tensor
        First tensor to compare
    tensor2 : torch.Tensor
        Second tensor to compare
    msg : str, optional
        Optional message to include in assertion error

    Raises
    ------
    AssertionError
        If tensors are not equal (shape mismatch, non-NaN values differ, or NaN positions differ)
    """
    if tensor1 is None and tensor2 is None:
        return
    if tensor1 is None or tensor2 is None:
        raise AssertionError(
            f"One tensor is None and the other is not. tensor1 is None: {tensor1 is None}, "
            f"tensor2 is None: {tensor2 is None}. {msg or ''}"
        )

    # Check shapes match
    if tensor1.shape != tensor2.shape:
        raise AssertionError(
            f"Tensor shapes don't match: {tensor1.shape} vs {tensor2.shape}. {msg or ''}"
        )

    # Check if either tensor has NaN values
    has_nan1 = torch.isnan(tensor1).any()
    has_nan2 = torch.isnan(tensor2).any()

    if has_nan1 or has_nan2:
        # Compare NaN positions - they should match
        nan_mask1 = torch.isnan(tensor1)
        nan_mask2 = torch.isnan(tensor2)

        if not torch.equal(nan_mask1, nan_mask2):
            raise AssertionError(
                f"NaN positions don't match between tensors. {msg or ''}"
            )

        # Compare non-NaN values
        non_nan_mask = ~nan_mask1
        if non_nan_mask.any():
            if not torch.equal(tensor1[non_nan_mask], tensor2[non_nan_mask]):
                raise AssertionError(
                    f"Non-NaN values don't match between tensors. {msg or ''}"
                )
    else:
        # No NaN values, use standard comparison
        if not torch.equal(tensor1, tensor2):
            raise AssertionError(f"Tensors are not equal. {msg or ''}")
