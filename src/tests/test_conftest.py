"""Test that metabolism fixtures work correctly."""

from napistu.constants import SBML_DFS
from torch_geometric.data import Data


def test_sbml_dfs_fixture(sbml_dfs):
    """Test that sbml_dfs_metabolism fixture works."""
    assert sbml_dfs is not None
    assert hasattr(sbml_dfs, SBML_DFS.SPECIES)
    assert hasattr(sbml_dfs, SBML_DFS.REACTIONS)
    assert len(sbml_dfs.species) > 0
    assert len(sbml_dfs.reactions) > 0


def test_napistu_graph_fixture(napistu_graph):
    """Test that napistu_graph_metabolism fixture works."""
    assert napistu_graph is not None
    assert hasattr(napistu_graph, "vcount")
    assert hasattr(napistu_graph, "ecount")
    assert napistu_graph.vcount() > 0
    assert napistu_graph.ecount() > 0


def test_pw_index_fixture(pw_index):
    """Test that pw_index_metabolism fixture works."""
    assert pw_index is not None
    assert hasattr(pw_index, "index")
    assert len(pw_index.index) > 0


def test_napistu_data_fixture(napistu_data):
    """Test that napistu_data fixture works."""
    from napistu_torch.napistu_data import NapistuData

    assert napistu_data is not None
    assert isinstance(napistu_data, NapistuData)
    assert isinstance(napistu_data, Data)  # Should inherit from Data
    assert napistu_data.num_nodes > 0
    assert napistu_data.num_edges > 0
    assert napistu_data.num_node_features > 0
    assert napistu_data.num_edge_features > 0
