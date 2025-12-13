"""Binary classification loss computation utilities."""

from typing import Optional

import torch
import torch.nn as nn

from napistu_torch.utils.tensor_utils import validate_tensor_for_nan_inf


def compute_simple_bce_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    """
    Compute simple binary cross-entropy loss for positive and negative samples.

    Uses default reduction='mean' which automatically reduces the loss.

    Parameters
    ----------
    pos_scores : torch.Tensor
        Predicted logits for positive samples [num_pos_samples]
    neg_scores : torch.Tensor
        Predicted logits for negative samples [num_neg_samples]

    Returns
    -------
    torch.Tensor
        Combined loss (scalar)
    """
    # Use default reduction='mean' for automatic reduction
    loss_fn = nn.BCEWithLogitsLoss()

    pos_loss = loss_fn(pos_scores, torch.ones_like(pos_scores))
    neg_loss = loss_fn(neg_scores, torch.zeros_like(neg_scores))

    loss = pos_loss + neg_loss

    # Validate loss before returning (prevents NaN gradients from corrupting encoder)
    validate_tensor_for_nan_inf(loss, name="loss")

    return loss


def compute_weighted_bce_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    pos_weights: Optional[torch.Tensor] = None,
    neg_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute weighted binary cross-entropy loss for positive and negative samples.

    Parameters
    ----------
    pos_scores : torch.Tensor
        Predicted logits for positive samples [num_pos_samples]
    neg_scores : torch.Tensor
        Predicted logits for negative samples [num_neg_samples]
    pos_weights : torch.Tensor, optional
        Weights for positive samples [num_pos_samples]. If None, uniform weighting.
    neg_weights : torch.Tensor, optional
        Weights for negative samples [num_neg_samples]. If None, uniform weighting.

    Returns
    -------
    torch.Tensor
        Combined weighted loss (scalar)
    """
    # Use reduction='none' to return per-sample losses for manual weighting
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # Compute unreduced loss
    pos_loss_unreduced = loss_fn(pos_scores, torch.ones_like(pos_scores))
    neg_loss_unreduced = loss_fn(neg_scores, torch.zeros_like(neg_scores))

    # Apply weights and reduce manually
    if pos_weights is not None:
        pos_loss = (pos_loss_unreduced * pos_weights).mean()
    else:
        pos_loss = pos_loss_unreduced.mean()

    if neg_weights is not None:
        neg_loss = (neg_loss_unreduced * neg_weights).mean()
    else:
        neg_loss = neg_loss_unreduced.mean()

    loss = pos_loss + neg_loss

    # Validate loss before returning (prevents NaN gradients from corrupting encoder)
    validate_tensor_for_nan_inf(loss, name="loss")

    return loss
