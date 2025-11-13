import torch
import torch.nn as nn

from napistu_torch.constants import NAPISTU_DATA
from napistu_torch.evaluation.edge_weights import (
    compute_edge_feature_sensitivity,
    format_edge_feature_sensitivity,
    plot_edge_feature_sensitivity,
)
from napistu_torch.napistu_data import NapistuData


class _LinearEdgeEncoder(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(weight.numel(), 1, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(weight.unsqueeze(0))

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.linear(edge_attr)


def test_compute_edge_feature_sensitivity_matches_linear_weights():
    # Sanity check: the aggregated sensitivities should recover the true
    # linear weights when the encoder is exactly linear.
    torch.manual_seed(0)
    weight = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
    encoder = _LinearEdgeEncoder(weight)
    edge_attr = torch.randn(10, weight.numel())

    sensitivities = compute_edge_feature_sensitivity(
        encoder,
        edge_attr,
        max_edges=5,
    )

    assert sensitivities.shape == weight.shape
    torch.testing.assert_close(sensitivities, weight, atol=1e-5, rtol=1e-5)

    # Create a minimal NapistuData object with edge feature names for formatting
    napistu_data = NapistuData(
        x=torch.zeros(1, 1),
        edge_index=torch.zeros(2, 1, dtype=torch.long),
        edge_attr=torch.zeros(1, weight.numel()),
    )
    setattr(
        napistu_data,
        NAPISTU_DATA.EDGE_FEATURE_NAMES,
        [f"feature_{i}" for i in range(weight.numel())],
    )

    # Test that formatting function runs without error
    formatted = format_edge_feature_sensitivity(sensitivities, napistu_data)

    # Test that plotting function runs without error
    plot_edge_feature_sensitivity(formatted)
