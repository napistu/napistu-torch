import numpy as np
import torch

from napistu_torch.utils.constants import CORRELATION_METHODS
from napistu_torch.utils.tensor_utils import (
    compute_confusion_matrix,
    compute_correlation_matrix,
    compute_cosine_distances_torch,
    compute_spearman_correlation_torch,
)


def test_compute_cosine_distances_torch_basic_properties():
    device = torch.device("cpu")
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
    device = torch.device("cpu")

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
