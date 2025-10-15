"""Test napistu_graph_to_pyg with all splitting strategies."""

import pytest
from torch_geometric.data import Data

from napistu_torch.load.constants import (
    SPLITTING_STRATEGIES,
    VALID_SPLITTING_STRATEGIES,
)
from napistu_torch.load.napistu_graphs import napistu_graph_to_pyg


@pytest.mark.parametrize("strategy", VALID_SPLITTING_STRATEGIES)
def test_napistu_graph_to_pyg_all_strategies(napistu_graph, strategy):
    """Test that napistu_graph_to_pyg works with each splitting strategy."""
    result = napistu_graph_to_pyg(
        napistu_graph, splitting_strategy=strategy, verbose=False
    )

    # Verify result is not None
    assert result is not None

    # For strategies that return a single Data object
    if strategy in [
        SPLITTING_STRATEGIES.NO_MASK,
        SPLITTING_STRATEGIES.EDGE_MASK,
        SPLITTING_STRATEGIES.VERTEX_MASK,
    ]:
        assert isinstance(result, Data)
        assert result.num_nodes > 0
        assert result.num_edges > 0

    # For strategies that return a dictionary of Data objects
    elif strategy in [SPLITTING_STRATEGIES.INDUCTIVE]:
        assert isinstance(result, dict)
        assert len(result) > 0
        # Check that all values are Data objects
        for data_obj in result.values():
            assert isinstance(data_obj, Data)
            assert data_obj.num_nodes > 0
            assert data_obj.num_edges > 0


def test_napistu_graph_to_pyg_invalid_strategy(napistu_graph):
    """Test that napistu_graph_to_pyg raises ValueError for invalid strategy."""
    with pytest.raises(ValueError, match="splitting_strategy must be one of"):
        napistu_graph_to_pyg(
            napistu_graph, splitting_strategy="invalid_strategy", verbose=False
        )
