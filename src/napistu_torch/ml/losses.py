"""Binary classification loss computation utilities."""

import torch
import torch.nn as nn

from napistu_torch.utils.tensor_utils import validate_tensor_for_nan_inf


def compute_bce_loss(
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

    validate_tensor_for_nan_inf(loss, name="loss")
    return loss


def compute_margin_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """
    Compute margin-based ranking loss for RotatE/TransE.

    Loss = mean(max(0, margin - pos_score + neg_score))

    This encourages positive edges to score at least 'margin' points
    higher than negative edges.

    Parameters
    ----------
    pos_scores : torch.Tensor
        Scores for positive edges [num_pos]
    neg_scores : torch.Tensor
        Scores for negative edges [num_neg]
    margin : float
        Margin value (typically 9.0 for RotatE, 1.0 for TransE)

    Returns
    -------
    torch.Tensor
        Scalar loss value

    Notes
    -----
    For RotatE/TransE, scores are negative distances:
    - Better edges have scores closer to 0 (small distance)
    - Worse edges have scores more negative (large distance)

    The loss penalizes when: pos_score < neg_score + margin
    (i.e., when positive edge isn't scoring enough better than negative)

    Assumes pos_scores and neg_scores are paired element-wise (same length).
    """
    loss = torch.clamp(margin - pos_scores + neg_scores, min=0).mean()

    validate_tensor_for_nan_inf(loss, name="loss")
    return loss


def compute_weighted_bce_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    pos_weights: torch.Tensor,
    neg_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute weighted binary cross-entropy loss for positive and negative samples.

    Parameters
    ----------
    pos_scores : torch.Tensor
        Predicted logits for positive samples [num_pos_samples]
    neg_scores : torch.Tensor
        Predicted logits for negative samples [num_neg_samples]
    pos_weights : torch.Tensor
        Weights for positive samples [num_pos_samples]
    neg_weights : torch.Tensor
        Weights for negative samples [num_neg_samples]

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
    pos_loss = (pos_loss_unreduced * pos_weights).mean()
    neg_loss = (neg_loss_unreduced * neg_weights).mean()

    loss = pos_loss + neg_loss

    validate_tensor_for_nan_inf(loss, name="loss")
    return loss


def compute_weighted_margin_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float,
    pos_weights: torch.Tensor,
    neg_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute weighted margin-based ranking loss for RotatE/TransE.

    Loss = mean((pos_weights * neg_weights) * max(0, margin - pos_score + neg_score))

    This encourages positive edges to score at least 'margin' points
    higher than negative edges, with per-sample weighting.

    Parameters
    ----------
    pos_scores : torch.Tensor
        Scores for positive edges [num_pos_samples]
    neg_scores : torch.Tensor
        Scores for negative edges [num_neg_samples]
    margin : float
        Margin value (typically 9.0 for RotatE, 1.0 for TransE)
    pos_weights : torch.Tensor
        Weights for positive samples [num_pos_samples]
    neg_weights : torch.Tensor
        Weights for negative samples [num_neg_samples]

    Returns
    -------
    torch.Tensor
        Combined weighted loss (scalar)

    Notes
    -----
    For RotatE/TransE, scores are negative distances:
    - Better edges have scores closer to 0 (small distance)
    - Worse edges have scores more negative (large distance)

    The loss penalizes when: pos_score < neg_score + margin
    (i.e., when positive edge isn't scoring enough better than negative)

    Assumes pos_scores and neg_scores are paired element-wise (same length).
    Scores are sorted by their weights before pairing to ensure similarly weighted
    samples are compared together.
    """
    # Sort scores by weights to pair similarly weighted samples
    # Sort positive scores and weights together by weights (descending)
    pos_sort_idx = torch.argsort(pos_weights, descending=True)
    pos_scores = pos_scores[pos_sort_idx]
    pos_weights = pos_weights[pos_sort_idx]

    # Sort negative scores and weights together by weights (descending)
    neg_sort_idx = torch.argsort(neg_weights, descending=True)
    neg_scores = neg_scores[neg_sort_idx]
    neg_weights = neg_weights[neg_sort_idx]

    # Compute unreduced margin loss (per-sample)
    # margin - pos_score + neg_score penalizes when pos_score < neg_score + margin
    margin_loss_unreduced = torch.clamp(margin - pos_scores + neg_scores, min=0)

    # Apply weights: multiply element-wise and take mean
    combined_weights = pos_weights * neg_weights
    loss = (margin_loss_unreduced * combined_weights).mean()

    validate_tensor_for_nan_inf(loss, name="loss")
    return loss
