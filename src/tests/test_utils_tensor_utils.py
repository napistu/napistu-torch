import numpy as np
import pytest
import torch

from napistu_torch.ml.constants import DEVICE
from napistu_torch.utils.constants import CORRELATION_METHODS
from napistu_torch.utils.tensor_utils import (
    compute_confusion_matrix,
    compute_correlation_matrix,
    compute_cosine_distances_torch,
    compute_max_abs_over_z,
    compute_max_over_z,
    compute_spearman_correlation_torch,
    compute_tensor_ranks,
    compute_tensor_ranks_for_indices,
    find_top_k,
)

TENSOR_3D = torch.tensor(
    [
        # Position [0,0] across layers: [1.0, 0.5, 2.5]
        # Position [0,1] across layers: [-2.0, 1.0, 0.3]
        [[1.0, 0.5, 2.5], [-2.0, 1.0, 0.3]],
        # Position [1,0] across layers: [0.5, -3.0, 1.0]
        # Position [1,1] across layers: [1.5, 0.8, -4.0]
        [[0.5, -3.0, 1.0], [1.5, 0.8, -4.0]],
    ]
)


def test_compute_cosine_distances_torch_basic_properties():
    device = torch.device(DEVICE.CPU)
    embeddings = torch.tensor(
        [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )

    distances = compute_cosine_distances_torch(embeddings, device)

    assert isinstance(distances, np.ndarray)
    assert distances.shape == (3, 3)
    assert np.allclose(np.diag(distances), 0.0, atol=1e-6)
    assert np.allclose(distances, distances.T, atol=1e-6)


def test_cosine_distances_spearman_correlation_agreement():
    torch.manual_seed(42)
    device = torch.device(DEVICE.CPU)

    base = torch.randn(6, 3)
    expanded = torch.cat(
        [base, torch.randn(6, 2) * 0.05 + base.mean(dim=1, keepdim=True)], dim=1
    )

    dist_a_full = compute_cosine_distances_torch(base, device)
    dist_b_full = compute_cosine_distances_torch(expanded, device)
    iu = np.triu_indices(base.size(0), k=1)
    dist_a = dist_a_full[iu]
    dist_b = dist_b_full[iu]

    rho = compute_spearman_correlation_torch(dist_a, dist_b, device)

    assert isinstance(rho, float)
    assert rho > 0.8


def test_compute_confusion_matrix():
    """Test confusion matrix with perfect predictions and misclassifications."""
    # Test perfect predictions
    predictions = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    true_labels = torch.tensor([0, 1, 2])

    cm = compute_confusion_matrix(predictions, true_labels)

    assert isinstance(cm, np.ndarray)
    assert cm.shape == (3, 3)
    assert np.array_equal(cm, np.eye(3))

    # Test misclassifications
    predictions = torch.tensor([[0.3, 0.6, 0.1], [0.2, 0.7, 0.1], [0.4, 0.5, 0.1]])
    true_labels = torch.tensor([0, 1, 2])

    cm = compute_confusion_matrix(predictions, true_labels)

    assert isinstance(cm, np.ndarray)
    assert cm.shape == (3, 3)
    assert cm[0, 1] == 1
    assert cm[1, 1] == 1
    assert cm[2, 1] == 1


def test_compute_confusion_matrix_normalize():
    """Test confusion matrix normalization modes."""
    predictions = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    true_labels = torch.tensor([0, 1, 2])

    cm_true = compute_confusion_matrix(predictions, true_labels, normalize="true")
    cm_pred = compute_confusion_matrix(predictions, true_labels, normalize="pred")
    cm_all = compute_confusion_matrix(predictions, true_labels, normalize="all")

    assert np.allclose(cm_true.sum(axis=1), 1.0)
    assert np.allclose(cm_pred.sum(axis=0), 1.0)
    assert np.allclose(cm_all.sum(), 1.0)


def test_compute_confusion_matrix_numpy_input():
    """Test confusion matrix with numpy array inputs."""
    predictions = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    true_labels = np.array([0, 1, 2])

    cm = compute_confusion_matrix(predictions, true_labels)

    assert isinstance(cm, np.ndarray)
    assert cm.shape == (3, 3)


def test_compute_correlation_matrix():
    """Test correlation matrix basic properties and perfect correlation."""
    # Test basic properties
    data = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])

    corr_matrix, p_values = compute_correlation_matrix(data)

    assert isinstance(corr_matrix, np.ndarray)
    assert isinstance(p_values, np.ndarray)
    assert corr_matrix.shape == (3, 3)
    assert p_values.shape == (3, 3)
    assert np.allclose(np.diag(corr_matrix), 1.0)
    assert np.allclose(corr_matrix, corr_matrix.T)

    # Test perfect correlation
    data = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])

    corr_matrix, p_values = compute_correlation_matrix(
        data, method=CORRELATION_METHODS.PEARSON
    )

    assert np.allclose(corr_matrix[0, 1], 1.0, atol=1e-6)
    assert p_values[0, 1] < 0.05


def test_compute_correlation_matrix_spearman_vs_pearson():
    """Test that Spearman and Pearson methods both work."""
    data = torch.tensor([[1.0, 1.0], [2.0, 4.0], [3.0, 9.0], [4.0, 16.0]])

    corr_spearman, _ = compute_correlation_matrix(
        data, method=CORRELATION_METHODS.SPEARMAN
    )
    corr_pearson, _ = compute_correlation_matrix(
        data, method=CORRELATION_METHODS.PEARSON
    )

    assert isinstance(corr_spearman, np.ndarray)
    assert isinstance(corr_pearson, np.ndarray)
    assert corr_spearman.shape == (2, 2)
    assert corr_pearson.shape == (2, 2)


def test_compute_correlation_matrix_numpy_input():
    """Test correlation matrix with numpy array input."""
    data = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])

    corr_matrix, p_values = compute_correlation_matrix(data)

    assert isinstance(corr_matrix, np.ndarray)
    assert isinstance(p_values, np.ndarray)


def test_compute_max_abs_over_z():
    """Test that compute_max_abs_over_z preserves sign correctly."""
    result, indices = compute_max_abs_over_z(TENSOR_3D, return_indices=True)

    # Expected results:
    # [0,0]: max(|1.0|, |0.5|, |2.5|) = 2.5 from layer 2 → result = 2.5
    # [0,1]: max(|-2.0|, |1.0|, |0.3|) = 2.0 from layer 0 → result = -2.0
    # [1,0]: max(|0.5|, |-3.0|, |1.0|) = 3.0 from layer 1 → result = -3.0
    # [1,1]: max(|1.5|, |0.8|, |-4.0|) = 4.0 from layer 2 → result = -4.0

    expected_result = torch.tensor([[2.5, -2.0], [-3.0, -4.0]])
    expected_indices = torch.tensor([[2, 0], [1, 2]])

    assert torch.allclose(
        result, expected_result
    ), f"Expected {expected_result}, got {result}"
    assert torch.equal(
        indices, expected_indices
    ), f"Expected {expected_indices}, got {indices}"


def test_compute_max_over_z():
    """Test that compute_max_over_z finds maximum value (not absolute) correctly."""
    result, indices = compute_max_over_z(TENSOR_3D, return_indices=True)

    # Expected results (finding max without taking absolute value):
    # [0,0]: max(1.0, 0.5, 2.5) = 2.5 from layer 2 → result = 2.5
    # [0,1]: max(-2.0, 1.0, 0.3) = 1.0 from layer 1 → result = 1.0
    # [1,0]: max(0.5, -3.0, 1.0) = 1.0 from layer 2 → result = 1.0
    # [1,1]: max(1.5, 0.8, -4.0) = 1.5 from layer 0 → result = 1.5

    expected_result = torch.tensor([[2.5, 1.0], [1.0, 1.5]])
    expected_indices = torch.tensor([[2, 1], [2, 0]])

    assert torch.allclose(
        result, expected_result
    ), f"Expected {expected_result}, got {result}"
    assert torch.equal(
        indices, expected_indices
    ), f"Expected {expected_indices}, got {indices}"


def test_find_top_k():
    """Test find_top_k extracts correct top-k values and indices."""
    tensor = torch.tensor(
        [
            [1.0, -5.0, 3.0],
            [-2.0, 4.0, -1.0],
            [0.5, 2.0, -3.0],
        ]
    )

    row_idx, col_idx, values = find_top_k(tensor, k=3, by_absolute_value=True)

    # Top 3 by absolute value: |-5.0|=5, |4.0|=4, |3.0|=3, |-3.0|=3
    # Because |-3.0| ties with |3.0| at k=3, we get 4 results (includes all ties)
    assert len(row_idx) == 4
    assert len(col_idx) == 4
    assert len(values) == 4

    # Check all (row, col) pairs are unique
    pairs = set(zip(row_idx.tolist(), col_idx.tolist()))
    assert len(pairs) == 4, "All pairs should be unique"

    # Top values should be -5.0, 4.0, 3.0, -3.0 (in descending absolute value order)
    expected_abs_values = torch.tensor([5.0, 4.0, 3.0, 3.0])
    assert torch.allclose(torch.abs(values), expected_abs_values)

    # Verify the actual values (sign preserved)
    assert values[0].item() == -5.0  # Highest absolute value
    assert values[1].item() == 4.0  # Second highest
    # positions 2 and 3 are 3.0 and -3.0 (order not guaranteed for ties)
    assert set([values[2].item(), values[3].item()]) == {3.0, -3.0}


def test_find_top_k_with_ties():
    """Test find_top_k behavior when there are tied values."""
    # Create tensor with multiple identical values
    tensor = torch.tensor(
        [
            [1.0, 5.0, 3.0, 5.0],
            [5.0, 2.0, 5.0, 1.0],
            [0.5, 5.0, 4.0, 2.0],
            [5.0, 1.0, 5.0, 0.5],
        ]
    )

    # There are 7 values with |5.0|, which is the maximum
    # Top 10 should include all 7 unique positions with value 5.0,
    # plus values with next highest: 4.0, 3.0, 2.0, 2.0
    # Since 2.0 appears at k=10, we include ALL 2.0s (both of them), giving us 11 total
    row_idx, col_idx, values = find_top_k(tensor, k=10, by_absolute_value=True)

    # Check we got 11 results (10 requested, but 2.0 ties at position 10, so we get both)
    assert len(row_idx) == 11
    assert len(col_idx) == 11
    assert len(values) == 11

    # Check that all (row, col) pairs are unique
    pairs = set(zip(row_idx.tolist(), col_idx.tolist()))
    assert len(pairs) == 11, f"Expected 11 unique pairs, got {len(pairs)}"

    # Top values should be the 7 copies of 5.0, then 4.0, 3.0, then 2 copies of 2.0
    expected_value_counts = {5.0: 7, 4.0: 1, 3.0: 1, 2.0: 2}
    actual_value_counts = {}
    for v in values.tolist():
        actual_value_counts[v] = actual_value_counts.get(v, 0) + 1

    assert (
        actual_value_counts == expected_value_counts
    ), f"Expected {expected_value_counts}, got {actual_value_counts}"

    # Verify all the 5.0 positions are captured
    positions_with_5 = [(0, 1), (0, 3), (1, 0), (1, 2), (2, 1), (3, 0), (3, 2)]
    found_positions_with_5 = [
        (r.item(), c.item())
        for r, c, v in zip(row_idx, col_idx, values)
        if abs(v.item() - 5.0) < 1e-6
    ]
    assert (
        len(found_positions_with_5) == 7
    ), f"Expected 7 positions with value 5.0, got {len(found_positions_with_5)}"
    assert set(found_positions_with_5) == set(
        positions_with_5
    ), "Did not capture all unique positions with value 5.0"


@pytest.mark.parametrize("by_absolute_value", [True, False])
def test_compute_tensor_ranks(by_absolute_value):
    """Test rank computation with both absolute value and raw value ranking."""
    if by_absolute_value:
        # Matrix with negative values to test absolute value ranking
        tensor = torch.tensor([[-5.0, 2.0, -8.0], [1.0, 9.0, -3.0], [4.0, -7.0, 6.0]])
        # Expected: highest absolute value is 9.0 (rank 1), then 8.0 (rank 2), then 7.0 (rank 3)
        expected_rank_1_pos = (1, 1)  # 9.0
        expected_rank_2_pos = (0, 2)  # -8.0 (abs=8.0)
        expected_rank_3_pos = (2, 1)  # -7.0 (abs=7.0)
    else:
        # Simple 3x3 matrix with known positive values
        tensor = torch.tensor([[5.0, 2.0, 8.0], [1.0, 9.0, 3.0], [4.0, 7.0, 6.0]])
        # Expected: highest value is 9.0 (rank 1), then 8.0 (rank 2), lowest is 1.0 (rank 9)
        expected_rank_1_pos = (1, 1)  # 9.0
        expected_rank_2_pos = (0, 2)  # 8.0
        expected_rank_3_pos = None  # Not needed for raw value test

    # Compute ranks
    ranks = compute_tensor_ranks(tensor, by_absolute_value=by_absolute_value)

    # Verify shape and dtype
    assert ranks.shape == tensor.shape
    assert ranks.dtype == torch.int64

    # Check expected rankings
    assert (
        ranks[expected_rank_1_pos] == 1
    ), f"Expected rank 1 at {expected_rank_1_pos}, got {ranks[expected_rank_1_pos]}"
    assert (
        ranks[expected_rank_2_pos] == 2
    ), f"Expected rank 2 at {expected_rank_2_pos}, got {ranks[expected_rank_2_pos]}"

    if not by_absolute_value:
        # For raw value, also check lowest value
        assert ranks[1, 0] == 9, f"Expected rank 9 for value 1.0, got {ranks[1, 0]}"
    else:
        # For absolute value, check third highest
        assert (
            ranks[expected_rank_3_pos] == 3
        ), f"Expected rank 3 at {expected_rank_3_pos}, got {ranks[expected_rank_3_pos]}"

    # Verify all ranks are unique and cover 1 to numel
    unique_ranks = torch.unique(ranks)
    assert len(unique_ranks) == tensor.numel()
    assert torch.all(
        unique_ranks == torch.arange(1, tensor.numel() + 1, dtype=torch.int64)
    )

    # Test indexing works
    from_idx = torch.tensor([0, 1, 2])
    to_idx = torch.tensor([0, 1, 2])
    extracted_ranks = ranks[from_idx, to_idx]
    assert extracted_ranks.shape == (3,)
    assert torch.all(extracted_ranks >= 1)
    assert torch.all(extracted_ranks <= tensor.numel())

    # Test shape preservation with various shapes
    shapes = [(10, 10), (5, 20), (100,), (3, 4, 5)]
    for shape in shapes:
        test_tensor = torch.randn(*shape)
        test_ranks = compute_tensor_ranks(
            test_tensor, by_absolute_value=by_absolute_value
        )
        assert test_ranks.shape == shape, f"Shape mismatch for input shape {shape}"
        assert test_ranks.dtype == torch.int64


def test_compute_tensor_ranks_for_indices():
    """Test that compute_tensor_ranks_for_indices matches compute_tensor_ranks for same indices."""
    # Test 2D tensor
    tensor_2d = torch.randn(50, 50)

    # Create random indices
    n_indices = 100
    row_indices = torch.randint(0, 50, (n_indices,))
    col_indices = torch.randint(0, 50, (n_indices,))

    # Compute ranks using full matrix method
    full_ranks = compute_tensor_ranks(tensor_2d, by_absolute_value=True)
    extracted_ranks = full_ranks[row_indices, col_indices]

    # Compute ranks using indices-only method
    indices_ranks = compute_tensor_ranks_for_indices(
        tensor_2d, (row_indices, col_indices), by_absolute_value=True
    )

    # Results should match exactly
    assert torch.allclose(
        extracted_ranks.float(), indices_ranks.float()
    ), "Ranks from compute_tensor_ranks_for_indices don't match compute_tensor_ranks"

    # Test with by_absolute_value=False
    full_ranks_raw = compute_tensor_ranks(tensor_2d, by_absolute_value=False)
    extracted_ranks_raw = full_ranks_raw[row_indices, col_indices]
    indices_ranks_raw = compute_tensor_ranks_for_indices(
        tensor_2d, (row_indices, col_indices), by_absolute_value=False
    )
    assert torch.allclose(
        extracted_ranks_raw.float(), indices_ranks_raw.float()
    ), "Ranks with by_absolute_value=False don't match"

    # Test 1D tensor
    tensor_1d = torch.randn(100)
    indices_1d = torch.randint(0, 100, (50,))

    full_ranks_1d = compute_tensor_ranks(tensor_1d, by_absolute_value=True)
    extracted_ranks_1d = full_ranks_1d[indices_1d]
    indices_ranks_1d = compute_tensor_ranks_for_indices(
        tensor_1d, (indices_1d,), by_absolute_value=True
    )
    assert torch.allclose(
        extracted_ranks_1d.float(), indices_ranks_1d.float()
    ), "1D tensor ranks don't match"

    # Test edge cases: single index, all indices, duplicate indices
    single_row = torch.tensor([10])
    single_col = torch.tensor([20])
    single_rank_full = compute_tensor_ranks(tensor_2d, by_absolute_value=True)[10, 20]
    single_rank_indices = compute_tensor_ranks_for_indices(
        tensor_2d, (single_row, single_col), by_absolute_value=True
    )
    assert single_rank_full == single_rank_indices[0], "Single index rank doesn't match"
