"""Custom metrics for model evaluation."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from napistu_torch.labels.apply import decode_labels
from napistu_torch.labels.labeling_manager import LabelingManager
from napistu_torch.ml.constants import METRICS, RELATION_WEIGHTED_AUC_DEFS

logger = logging.getLogger(__name__)


class RelationWeightedAUC:
    """
    Compute per-relation AUC and weighted average AUC.

    Computes:
    1. Overall AUC (all samples pooled, unweighted)
    2. Per-relation AUCs (one for each relation type)
    3. Relation-weighted AUC (weighted by loss_weight × validation_count)

    Parameters
    ----------
    loss_weights : torch.Tensor
        Pre-computed loss weights from training [num_relations]
    loss_weight_alpha : float
        Alpha parameter used for loss weighting (for logging/reference)
    relation_manager : LabelingManager, optional
        Manager for decoding relation type indices to human-readable names

    Public Methods
    --------------
    compute(y_true, y_pred, relation_type)
        Compute overall, per-relation, and weighted AUCs.

    Examples
    --------
    >>> # During validation
    >>> rw_auc = RelationWeightedAUC(
    ...     loss_weights=task._relation_weights,
    ...     loss_weight_alpha=task.loss_weight_alpha,
    ...     relation_manager=data.relation_manager
    ... )
    >>> metrics = rw_auc.compute(y_true, y_pred, relation_type)
    >>> print(metrics)
    {
        'auc': 0.85,
        'auc_relation_weighted': 0.82,
        'auc_catalysis': 0.78,
        'auc_interaction': 0.88,
        'auc_inhibition': 0.81
    }
    """

    def __init__(
        self,
        loss_weights: torch.Tensor,
        loss_weight_alpha: float,
        relation_manager: Optional[LabelingManager] = None,
    ):
        self.loss_weights = loss_weights
        self.loss_weight_alpha = loss_weight_alpha
        self.relation_manager = relation_manager

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        relation_type: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute overall, per-relation, and weighted AUCs.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels [num_samples]
        y_pred : np.ndarray
            Predicted probabilities [num_samples]
        relation_type : torch.Tensor
            Relation type index for each sample [num_samples]

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'auc': Overall AUC (all samples pooled)
            - 'auc_relation_weighted': Weighted average of per-relation AUCs
            - 'auc_{relation_name}': Per-relation AUC for each relation type
        """

        # Check for NaN/Inf values in predictions
        nan_inf_mask = np.isnan(y_pred) | np.isinf(y_pred)
        if nan_inf_mask.any():
            n_nan = np.isnan(y_pred).sum()
            n_inf = np.isinf(y_pred).sum()
            raise ValueError(
                f"Found {n_nan} NaN and {n_inf} Inf values in predictions. "
            )

        results = {}

        # Overall AUC (unweighted, all samples pooled)
        results[METRICS.AUC] = roc_auc_score(y_true, y_pred)

        # Per-relation AUCs and counts
        relation_type_np = relation_type.cpu().numpy()
        unique_relations = np.unique(relation_type_np)

        # Check for pathological relations (missing classes) upfront
        _log_pathological_labels(
            y_true, relation_type_np, unique_relations, self.relation_manager
        )

        per_relation_aucs, per_relation_counts, per_relation_results = (
            _compute_per_relation_aucs(
                y_true, y_pred, relation_type, unique_relations, self.relation_manager
            )
        )

        # Add per-relation AUCs to results
        results.update(per_relation_results)

        # Weighted AUC: adjusted_weights = loss_weights × val_counts
        per_relation_aucs = np.array(per_relation_aucs)
        per_relation_counts = np.array(per_relation_counts)

        # Get corresponding loss weights for these relations
        loss_weights_np = self.loss_weights[unique_relations].cpu().numpy()

        # Compute adjusted weights: (1/freq^alpha) × validation_count
        adjusted_weights = loss_weights_np * per_relation_counts

        # Weighted average
        results[RELATION_WEIGHTED_AUC_DEFS.RELATION_WEIGHTED_AUC] = (
            adjusted_weights * per_relation_aucs
        ).sum() / adjusted_weights.sum()

        logger.debug(
            f"Relation-weighted AUC computed: "
            f"overall={results[METRICS.AUC]:.3f}, "
            f"weighted={results[RELATION_WEIGHTED_AUC_DEFS.RELATION_WEIGHTED_AUC]:.3f}"
        )

        return results


def _compute_per_relation_aucs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    relation_type: torch.Tensor,
    unique_relations: np.ndarray,
    relation_manager: Optional[LabelingManager] = None,
) -> Tuple[List[float], List[int], Dict[str, float]]:
    """
    Compute per-relation AUCs for each relation type.

    Assumes all relations have been validated to have both classes present.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels [num_samples]
    y_pred : np.ndarray
        Predicted probabilities [num_samples]
    relation_type : torch.Tensor
        Relation type index for each sample [num_samples]
    unique_relations : np.ndarray
        Unique relation type indices (in sorted order)
    relation_manager : LabelingManager, optional
        Manager for decoding relation type indices to human-readable names

    Returns
    -------
    per_relation_aucs : List[float]
        AUC for each relation type (in order of unique_relations)
    per_relation_counts : List[int]
        Number of samples for each relation type (in order of unique_relations)
    per_relation_results : Dict[str, float]
        Dictionary mapping 'auc_{relation_name}' to AUC value for each relation
    """
    relation_type_np = relation_type.cpu().numpy()

    per_relation_aucs = []
    per_relation_counts = []
    per_relation_results = {}

    for rel_idx in unique_relations:
        mask = relation_type_np == rel_idx
        rel_y_true = y_true[mask]
        rel_y_pred = y_pred[mask]

        # Compute per-relation AUC (validation already done upfront)
        rel_auc = roc_auc_score(rel_y_true, rel_y_pred)
        per_relation_aucs.append(rel_auc)
        per_relation_counts.append(mask.sum())

        # Get human-readable relation name
        if relation_manager is not None:
            rel_name = decode_labels(torch.tensor([rel_idx]), relation_manager)[0]
        else:
            rel_name = str(rel_idx)

        # Store per-relation AUC
        per_relation_results[
            RELATION_WEIGHTED_AUC_DEFS.RELATION_AUC_TEMPLATE.format(
                relation_name=rel_name
            )
        ] = rel_auc

    return per_relation_aucs, per_relation_counts, per_relation_results


def _log_pathological_labels(
    y_true: np.ndarray,
    relation_type_np: np.ndarray,
    unique_relations: np.ndarray,
    relation_manager: Optional[LabelingManager] = None,
) -> None:
    """
    Check for relations missing expected classes and raise informative error.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels [num_samples]
    relation_type_np : np.ndarray
        Relation type indices [num_samples]
    unique_relations : np.ndarray
        Unique relation type indices
    relation_manager : LabelingManager, optional
        Manager for decoding relation type indices to human-readable names

    Raises
    ------
    ValueError
        If any relation type is missing both positive and negative samples
    """
    expected_labels = {0.0, 1.0}  # Binary classification expects both classes
    pathological_relations = []

    for rel_idx in unique_relations:
        mask = relation_type_np == rel_idx
        rel_y_true = y_true[mask]
        present_labels_raw = np.unique(rel_y_true)
        # Convert numpy types to native Python types for cleaner error messages
        present_labels = {float(label) for label in present_labels_raw}
        missing_labels = expected_labels - present_labels

        if missing_labels:
            if relation_manager is not None:
                rel_name = decode_labels(torch.tensor([rel_idx]), relation_manager)[0]
            else:
                rel_name = str(rel_idx)

            # Convert to native Python types
            label_counts = {
                float(label): int((rel_y_true == label).sum())
                for label in present_labels_raw
            }
            pathological_relations.append(
                {
                    "name": rel_name,
                    "index": int(rel_idx),
                    "present_labels": present_labels,
                    "missing_labels": missing_labels,
                    "label_counts": label_counts,
                    "total_samples": int(mask.sum()),
                }
            )

    if pathological_relations:
        # Build error message
        error_parts = [
            "Cannot compute AUC: some relation types are missing expected classes."
        ]
        for rel_info in pathological_relations:
            error_parts.append(
                f"  - Relation '{rel_info['name']}' (index {rel_info['index']}): "
                f"missing labels {rel_info['missing_labels']}, "
                f"present labels {rel_info['label_counts']} "
                f"(total samples: {rel_info['total_samples']})"
            )
        error_parts.append(
            "\nThis can happen when a relation type has insufficient samples in a "
            "particular train/val/test split, even after merging rare strata. "
            "Consider increasing min_relation_count to ensure that all classes are present in each split."
        )
        raise ValueError("\n".join(error_parts))
