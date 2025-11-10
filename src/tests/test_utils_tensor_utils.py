import numpy as np
import torch

from napistu_torch.utils.tensor_utils import (
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
