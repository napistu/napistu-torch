"""Tests for custom metrics."""

import numpy as np
import torch

from napistu_torch.labels.labeling_manager import LabelingManager
from napistu_torch.load.constants import STRATIFICATION_DEFS
from napistu_torch.ml.constants import METRICS, RELATION_WEIGHTED_AUC_DEFS
from napistu_torch.ml.metrics import RelationWeightedAUC


def test_relation_weighted_auc():
    """Test RelationWeightedAUC with 3 relations, 50 positive and 50 negative examples each."""
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create 3 relations with 50 positive and 50 negative examples each
    num_relations = 3
    num_samples_per_relation = 100  # 50 positive + 50 negative
    total_samples = num_relations * num_samples_per_relation

    # Create relation types: [0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ..., 2]
    relation_type = torch.repeat_interleave(
        torch.arange(num_relations), num_samples_per_relation
    )

    # Create y_true: first 50 of each relation are positive (1), last 50 are negative (0)
    y_true = np.zeros(total_samples, dtype=np.float32)
    for rel_idx in range(num_relations):
        start_idx = rel_idx * num_samples_per_relation
        y_true[start_idx : start_idx + 50] = 1.0

    # Create y_pred with different AUCs for each relation to test weighting
    # Relation 0: AUC ~0.9 (good separation)
    # Relation 1: AUC ~0.7 (moderate separation)
    # Relation 2: AUC ~0.5 (poor separation, random)
    y_pred = np.zeros(total_samples, dtype=np.float32)

    for rel_idx in range(num_relations):
        start_idx = rel_idx * num_samples_per_relation

        if rel_idx == 0:
            # Relation 0: Good separation
            y_pred[start_idx : start_idx + 50] = np.random.beta(
                8, 2, 50
            )  # High scores for positives
            y_pred[start_idx + 50 : start_idx + 100] = np.random.beta(
                2, 8, 50
            )  # Low scores for negatives
        elif rel_idx == 1:
            # Relation 1: Moderate separation
            y_pred[start_idx : start_idx + 50] = np.random.beta(
                5, 5, 50
            )  # Medium-high scores for positives
            y_pred[start_idx + 50 : start_idx + 100] = np.random.beta(
                3, 7, 50
            )  # Medium-low scores for negatives
        else:
            # Relation 2: Poor separation (random)
            y_pred[start_idx : start_idx + 50] = np.random.beta(
                3, 3, 50
            )  # Random scores for positives
            y_pred[start_idx + 50 : start_idx + 100] = np.random.beta(
                3, 3, 50
            )  # Random scores for negatives

    # Create loss weights: [1.0, 2.0, 4.0] (relation 2 gets highest weight)
    # These simulate weights from get_relation_weights with different frequencies
    loss_weights = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)
    loss_weight_alpha = 0.5

    # Create RelationWeightedAUC instance
    rw_auc = RelationWeightedAUC(
        loss_weights=loss_weights,
        loss_weight_alpha=loss_weight_alpha,
        relation_manager=None,  # No relation manager for this test
    )

    # Compute metrics
    results = rw_auc.compute(y_true, y_pred, relation_type)
    relation_names = {
        i: RELATION_WEIGHTED_AUC_DEFS.RELATION_AUC_TEMPLATE.format(relation_name=i)
        for i in range(num_relations)
    }

    # Verify results structure
    assert METRICS.AUC in results
    assert RELATION_WEIGHTED_AUC_DEFS.RELATION_WEIGHTED_AUC in results
    for name in relation_names.values():
        assert name in results

    # Verify overall AUC is computed
    assert isinstance(results[METRICS.AUC], float)
    assert 0.0 <= results[METRICS.AUC] <= 1.0

    # Verify per-relation AUCs
    for relation_name in relation_names.values():
        assert isinstance(results[relation_name], float)
        assert 0.0 <= results[relation_name] <= 1.0

    # Verify relation 0 has highest AUC (good separation)
    assert results[relation_names[0]] > results[relation_names[1]]
    assert results[relation_names[0]] > results[relation_names[2]]

    # Verify relation 1 has higher AUC than relation 2 (moderate vs poor)
    assert results[relation_names[1]] > results[relation_names[2]]

    # Verify weighted AUC is computed
    assert isinstance(results[RELATION_WEIGHTED_AUC_DEFS.RELATION_WEIGHTED_AUC], float)
    assert 0.0 <= results[RELATION_WEIGHTED_AUC_DEFS.RELATION_WEIGHTED_AUC] <= 1.0

    # Verify weighted AUC calculation manually
    # Each relation has 100 samples, so counts are [100, 100, 100]
    per_relation_counts = np.array([100, 100, 100])
    loss_weights_np = loss_weights.numpy()
    adjusted_weights = loss_weights_np * per_relation_counts  # [100, 200, 400]

    per_relation_aucs = np.array(
        [
            results[relation_names[0]],
            results[relation_names[1]],
            results[relation_names[2]],
        ]
    )

    expected_weighted_auc = (
        adjusted_weights * per_relation_aucs
    ).sum() / adjusted_weights.sum()

    assert np.allclose(
        results[RELATION_WEIGHTED_AUC_DEFS.RELATION_WEIGHTED_AUC],
        expected_weighted_auc,
        rtol=1e-5,
    )

    # Verify that weighted AUC is different from overall AUC (due to weighting)
    # Weighted AUC should be closer to relation 2's AUC since it has highest weight
    # (though relation 2 has lowest AUC, so weighted AUC should be lower than overall)
    assert (
        results[RELATION_WEIGHTED_AUC_DEFS.RELATION_WEIGHTED_AUC]
        != results[METRICS.AUC]
    )


def test_relation_weighted_auc_with_relation_manager():
    """Test RelationWeightedAUC with a relation manager for human-readable names."""

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create 3 relations with 50 positive and 50 negative examples each
    num_relations = 3
    num_samples_per_relation = 100

    # Create relation types
    relation_type = torch.repeat_interleave(
        torch.arange(num_relations), num_samples_per_relation
    )

    # Create simple y_true and y_pred
    y_true = np.zeros(num_relations * num_samples_per_relation, dtype=np.float32)
    for rel_idx in range(num_relations):
        start_idx = rel_idx * num_samples_per_relation
        y_true[start_idx : start_idx + 50] = 1.0

    y_pred = np.random.rand(num_relations * num_samples_per_relation).astype(np.float32)

    # Create loss weights
    loss_weights = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)
    loss_weight_alpha = 0.5

    # Create relation manager with custom names (label_names must be a dict)
    relation_names = {0: "catalysis", 1: "interaction", 2: "inhibition"}
    relation_manager = LabelingManager(
        label_attribute=STRATIFICATION_DEFS.EDGE_STRATA,
        exclude_vertex_attributes=[],
        augment_summary_types=[],
        label_names=relation_names,
    )

    # Create RelationWeightedAUC instance with relation manager
    rw_auc = RelationWeightedAUC(
        loss_weights=loss_weights,
        loss_weight_alpha=loss_weight_alpha,
        relation_manager=relation_manager,
    )

    # Compute metrics
    results = rw_auc.compute(y_true, y_pred, relation_type)

    # Verify results use human-readable names
    assert METRICS.AUC in results
    assert RELATION_WEIGHTED_AUC_DEFS.RELATION_WEIGHTED_AUC in results
    relation_names = {
        RELATION_WEIGHTED_AUC_DEFS.RELATION_AUC_TEMPLATE.format(relation_name=name)
        for name in relation_names.values()
    }

    for name in relation_names:
        assert name in results

    # Verify old numeric keys are not present
    assert "auc_0" not in results
    assert "auc_1" not in results
    assert "auc_2" not in results
