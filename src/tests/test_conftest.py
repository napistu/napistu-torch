"""Test that metabolism fixtures work correctly."""

from torch_geometric.data import Data

from napistu_torch.constants import NAPISTU_DATA
from napistu_torch.labeling.labeling_manager import LabelingManager
from napistu_torch.napistu_data import NapistuData


def test_sbml_dfs_fixture(sbml_dfs):
    """Test that sbml_dfs_metabolism fixture works."""
    assert sbml_dfs is not None
    sbml_dfs.validate()


def test_napistu_graph_fixture(napistu_graph):
    """Test that napistu_graph_metabolism fixture works."""
    assert napistu_graph is not None
    napistu_graph.validate()
    assert napistu_graph.vcount() > 0
    assert napistu_graph.ecount() > 0


def test_pw_index_fixture(pw_index):
    """Test that pw_index_metabolism fixture works."""
    assert pw_index is not None
    assert hasattr(pw_index, "index")
    assert len(pw_index.index) > 0


def test_napistu_data_fixture(napistu_data):
    """Test that napistu_data fixture works."""

    assert napistu_data is not None
    assert isinstance(napistu_data, NapistuData)
    assert isinstance(napistu_data, Data)  # Should inherit from Data
    assert napistu_data.num_nodes > 0
    assert napistu_data.num_edges > 0
    assert napistu_data.num_node_features > 0
    assert napistu_data.num_edge_features > 0


def test_supervised_napistu_data_fixture(supervised_napistu_data_package):
    """Test that supervised_napistu_data fixture works."""

    # Unpack the tuple returned by the fixture
    napistu_data, labeling_manager = supervised_napistu_data_package

    # Test the NapistuData object
    assert napistu_data is not None
    assert isinstance(napistu_data, NapistuData)
    assert isinstance(napistu_data, Data)  # Should inherit from Data
    assert napistu_data.num_nodes > 0
    assert napistu_data.num_edges > 0
    assert napistu_data.num_node_features > 0
    assert napistu_data.num_edge_features > 0
    # Supervised data should have labels
    assert hasattr(napistu_data, NAPISTU_DATA.Y)
    assert napistu_data.y is not None
    assert napistu_data.y.shape[0] == napistu_data.num_nodes

    # Test the LabelingManager
    assert labeling_manager is not None
    assert isinstance(labeling_manager, LabelingManager)


def test_unsupervised_napistu_data_fixture(unsupervised_napistu_data):
    """Test that unsupervised_napistu_data fixture works."""

    assert unsupervised_napistu_data is not None
    assert isinstance(unsupervised_napistu_data, NapistuData)
    assert isinstance(unsupervised_napistu_data, Data)  # Should inherit from Data
    assert unsupervised_napistu_data.num_nodes > 0
    assert unsupervised_napistu_data.num_edges > 0
    assert unsupervised_napistu_data.num_node_features > 0
    assert unsupervised_napistu_data.num_edge_features > 0
    # Unsupervised data should not have labels
    assert (
        not hasattr(unsupervised_napistu_data, NAPISTU_DATA.Y)
        or unsupervised_napistu_data.y is None
    )
