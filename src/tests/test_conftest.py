"""Test that metabolism fixtures work correctly."""

from torch_geometric.data import Data

from napistu_torch.constants import NAPISTU_DATA
from napistu_torch.labeling.constants import LABEL_TYPE
from napistu_torch.labeling.labeling_manager import LabelingManager
from napistu_torch.load.constants import SPLITTING_STRATEGIES
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


def test_supervised_napistu_data_fixture(supervised_napistu_data):
    """Test that supervised_napistu_data fixture works."""

    # Test the NapistuData object
    assert supervised_napistu_data is not None
    assert isinstance(supervised_napistu_data, NapistuData)
    assert isinstance(supervised_napistu_data, Data)  # Should inherit from Data
    assert supervised_napistu_data.num_nodes > 0
    assert supervised_napistu_data.num_edges > 0
    assert supervised_napistu_data.num_node_features > 0
    assert supervised_napistu_data.num_edge_features > 0
    # Supervised data should have labels
    assert hasattr(supervised_napistu_data, NAPISTU_DATA.Y)
    assert supervised_napistu_data.y is not None
    assert supervised_napistu_data.y.shape[0] == supervised_napistu_data.num_nodes
    assert supervised_napistu_data.labeling_manager is not None
    assert isinstance(supervised_napistu_data.labeling_manager, LabelingManager)
    assert (
        supervised_napistu_data.labeling_manager.label_attribute
        == LABEL_TYPE.SPECIES_TYPE
    )
    assert supervised_napistu_data.name is not None
    assert (
        supervised_napistu_data.splitting_strategy == SPLITTING_STRATEGIES.VERTEX_MASK
    )


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
    # Check if labeling_manager exists, and if so, it should be None
    if hasattr(unsupervised_napistu_data, NAPISTU_DATA.LABELING_MANAGER):
        assert unsupervised_napistu_data.labeling_manager is None
    assert unsupervised_napistu_data.name is not None
    assert unsupervised_napistu_data.splitting_strategy == SPLITTING_STRATEGIES.NO_MASK
