"""Utility functions supporting a subset of heads.

This module provides utility functions for computing distances and probabilities
used by various prediction heads, particularly relation-aware heads like RotatE.

Public Functions
----------------
compute_rotate_distance(head_embeddings, tail_embeddings, relation_phase, eps=1e-10)
    Compute RotatE distance in complex space.
normalized_distances_to_probs(scores)
    Convert distances between softmax-normalized vectors to probabilities.
validate_symmetric_relation_indices(symmetric_relation_indices, num_relations)
    Validate that symmetric relation indices are properly configured.
"""

from typing import List, Union

from torch import Tensor, chunk, cos, sin, sqrt

from napistu_torch.utils.base_utils import normalize_and_validate_indices


def compute_rotate_distance(
    head_embeddings: Tensor,
    tail_embeddings: Tensor,
    relation_phase: Tensor,
    eps: float = 1e-10,
) -> Tensor:
    """
    Compute RotatE distance in complex space.

    Models relations as rotations: h ⊙ r ≈ t
    Distance measures how well the rotation transforms h to t.

    Parameters
    ----------
    head_embeddings : torch.Tensor
        Source node embeddings [num_edges, embedding_dim]
        Must be normalized and have even dimension
    tail_embeddings : torch.Tensor
        Target node embeddings [num_edges, embedding_dim]
        Must be normalized and have even dimension
    relation_phase : torch.Tensor
        Rotation phase angles [num_edges, embedding_dim/2]
        Angles in radians for complex rotation
    eps : float, optional
        Small constant for numerical stability, by default 1e-10

    Returns
    -------
    torch.Tensor
        Distance in [0, 2] for normalized embeddings [num_edges]

    Notes
    -----
    The computation follows RotatE (Sun et al. 2019):
    1. Split embeddings into real/imaginary parts
    2. Convert phase to complex rotation: r = cos(θ) + i*sin(θ)
    3. Complex multiply: h ⊙ r
    4. Compute L2 distance: ||h ⊙ r - t||

    References
    ----------
    Sun et al. "RotatE: Knowledge Graph Embedding by Relational Rotation
    in Complex Space" ICLR 2019.
    """
    # Split into real and imaginary parts for complex space
    # [num_edges, embedding_dim] → [num_edges, embedding_dim/2] each
    head_re, head_im = chunk(head_embeddings, 2, dim=-1)
    tail_re, tail_im = chunk(tail_embeddings, 2, dim=-1)

    # Convert phase to rotation in complex space
    # r = cos(θ) + i*sin(θ)
    rel_re = cos(relation_phase)
    rel_im = sin(relation_phase)

    # Complex multiplication: (h_re + i*h_im) * (r_re + i*r_im)
    # Real part: h_re * r_re - h_im * r_im
    # Imag part: h_re * r_im + h_im * r_re
    re_score = head_re * rel_re - head_im * rel_im
    im_score = head_re * rel_im + head_im * rel_re

    # Distance between rotated head and tail in complex space
    # ||h⊙r - t|| in ℂ^(d/2)
    re_diff = re_score - tail_re
    im_diff = im_score - tail_im

    # L2 distance with numerical stability
    # sqrt(|re_diff|² + |im_diff|²) averaged over dimensions
    distance = sqrt(re_diff**2 + im_diff**2 + eps).mean(dim=-1)

    return distance


def normalized_distances_to_probs(scores: Tensor) -> Tensor:
    """
    Convert distances between softmax-normalized vectors to probabilities.

    Parameters
    ----------
    scores : torch.Tensor
        Raw RotatE scores (negative distances [-2, 0])

    Returns
    -------
    torch.Tensor
        Probabilities in [0, 1]
    """
    # Linear mapping: prob = (score + 2) / 2
    # Maps [-2, 0] → [0, 1]
    return (scores + 2.0) / 2.0


def validate_symmetric_relation_indices(
    symmetric_relation_indices: Union[List[int], tuple, range],
    num_relations: int,
) -> None:
    """
    Validate symmetric relation indices.

    Parameters
    ----------
    symmetric_relation_indices : List[int]
        Indices to validate
    num_relations : int
        Total number of relations

    Raises
    ------
    ValueError
        If indices are invalid (duplicates, out of range, or all/none symmetric)
    """

    # Check that indices are not empty
    if not symmetric_relation_indices:
        raise ValueError(
            "symmetric_relation_indices cannot be empty. "
            "At least one symmetric relation required. "
            "Use RotatE head if all relations are asymmetric."
        )

    # Normalize and validate indices using utility function
    normalized_indices = normalize_and_validate_indices(
        indices=symmetric_relation_indices,
        max_value=num_relations,
        param_name="symmetric_relation_indices",
    )

    # Check we have both symmetric and asymmetric relations
    if len(normalized_indices) >= num_relations:
        raise ValueError(
            f"All {num_relations} relations are symmetric. "
            "ConditionalRotateHead requires both symmetric and asymmetric relations. "
            "Use DotProduct or DistMult head instead."
        )

    return None
