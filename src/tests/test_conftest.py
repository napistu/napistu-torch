"""Test that metabolism fixtures work correctly."""

import torch
from torch_geometric.data import Data

from napistu_torch.constants import NAPISTU_DATA
from napistu_torch.labels.constants import LABEL_TYPE
from napistu_torch.labels.labeling_manager import LabelingManager
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


def test_species_type_prediction_napistu_data_fixture(
    species_type_prediction_napistu_data,
):
    """Test that vertex_labeled_napistu_data fixture works."""

    # Test the NapistuData object
    assert species_type_prediction_napistu_data is not None
    assert isinstance(species_type_prediction_napistu_data, NapistuData)
    assert isinstance(
        species_type_prediction_napistu_data, Data
    )  # Should inherit from Data
    assert species_type_prediction_napistu_data.num_nodes > 0
    assert species_type_prediction_napistu_data.num_edges > 0
    assert species_type_prediction_napistu_data.num_node_features > 0
    assert species_type_prediction_napistu_data.num_edge_features > 0
    # Supervised data should have labels
    assert hasattr(species_type_prediction_napistu_data, NAPISTU_DATA.Y)
    assert species_type_prediction_napistu_data.y is not None
    assert (
        species_type_prediction_napistu_data.y.shape[0]
        == species_type_prediction_napistu_data.num_nodes
    )
    assert species_type_prediction_napistu_data.labeling_manager is not None
    assert isinstance(
        species_type_prediction_napistu_data.labeling_manager, LabelingManager
    )
    assert (
        species_type_prediction_napistu_data.labeling_manager.label_attribute
        == LABEL_TYPE.SPECIES_TYPE
    )
    assert species_type_prediction_napistu_data.name is not None
    assert (
        species_type_prediction_napistu_data.splitting_strategy
        == SPLITTING_STRATEGIES.VERTEX_MASK
    )


def test_unlabeled_napistu_data_fixture(unlabeled_napistu_data):
    """Test that unlabeled_napistu_data fixture works."""

    assert unlabeled_napistu_data is not None
    assert isinstance(unlabeled_napistu_data, NapistuData)
    assert isinstance(unlabeled_napistu_data, Data)  # Should inherit from Data
    assert unlabeled_napistu_data.num_nodes > 0
    assert unlabeled_napistu_data.num_edges > 0
    assert unlabeled_napistu_data.num_node_features > 0
    assert unlabeled_napistu_data.num_edge_features > 0
    # Unlabeled data should not have labels
    assert (
        not hasattr(unlabeled_napistu_data, NAPISTU_DATA.Y)
        or unlabeled_napistu_data.y is None
    )
    # Check if labeling_manager exists, and if so, it should be None
    if hasattr(unlabeled_napistu_data, NAPISTU_DATA.LABELING_MANAGER):
        assert unlabeled_napistu_data.labeling_manager is None
    assert unlabeled_napistu_data.name is not None
    assert unlabeled_napistu_data.splitting_strategy == SPLITTING_STRATEGIES.NO_MASK


def test_edge_prediction_with_sbo_relations_fixture(edge_prediction_with_sbo_relations):
    """Test that edge_prediction_with_sbo_relations fixture works."""
    data = edge_prediction_with_sbo_relations

    # Basic NapistuData checks
    assert data is not None
    assert isinstance(data, NapistuData)
    assert isinstance(data, Data)
    assert data.num_nodes > 0
    assert data.num_edges > 0
    assert data.num_node_features > 0
    assert data.num_edge_features > 0

    # Should have edge mask splitting strategy
    assert data.splitting_strategy == SPLITTING_STRATEGIES.EDGE_MASK

    # Should have relations attribute
    assert hasattr(data, NAPISTU_DATA.RELATION_TYPE)
    assert data.relation_type is not None
    assert isinstance(data.relation_type, torch.Tensor)
    assert data.relation_type.shape[0] == data.num_edges

    # Should have relation_manager attribute
    assert hasattr(data, NAPISTU_DATA.RELATION_MANAGER)
    assert data.relation_manager is not None
    assert isinstance(data.relation_manager, LabelingManager)
    assert data.relation_manager.label_names is not None
    assert len(data.relation_manager.label_names) > 0
