"""Test napistu_graph_to_napistu_data with all splitting strategies."""

import logging

import pytest
import torch
from napistu.network.constants import NAPISTU_GRAPH
from torch_geometric.data import Data

from napistu_torch.labels.constants import (
    LABEL_TYPE,
    LABELING,
)
from napistu_torch.labels.labeling_manager import LabelingManager
from napistu_torch.load.constants import (
    SPLITTING_STRATEGIES,
    VALID_SPLITTING_STRATEGIES,
)
from napistu_torch.load.napistu_graphs import (
    _ignore_graph_attributes,
    _name_napistu_data,
    napistu_graph_to_napistu_data,
)
from napistu_torch.ml.constants import (
    SPLIT_TO_MASK,
    TRAINING,
)


@pytest.mark.parametrize("strategy", VALID_SPLITTING_STRATEGIES)
def test_napistu_graph_to_napistu_data_all_strategies(napistu_graph, strategy):
    """Test that napistu_graph_to_napistu_data works with each splitting strategy."""
    result = napistu_graph_to_napistu_data(
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


def test_napistu_graph_to_napistu_data_invalid_strategy(napistu_graph):
    """Test that napistu_graph_to_napistu_data raises ValueError for invalid strategy."""
    with pytest.raises(ValueError, match="splitting_strategy must be one of"):
        napistu_graph_to_napistu_data(
            napistu_graph, splitting_strategy="invalid_strategy", verbose=False
        )


def test_vertex_mask_with_zero_val_size(augmented_napistu_graph):
    """Test vertex masking with val_size=0 runs without error and creates blank val_mask."""
    # Create PyG data with vertex masking and val_size=0
    data = napistu_graph_to_napistu_data(
        augmented_napistu_graph,
        splitting_strategy=SPLITTING_STRATEGIES.VERTEX_MASK,
        train_size=0.8,
        test_size=0.2,
        val_size=0.0,
        verbose=False,
    )

    # Verify we get a Data object with masks
    assert isinstance(data, Data)
    for mask_type in [TRAINING.TRAIN, TRAINING.TEST, TRAINING.VALIDATION]:
        mask_name = SPLIT_TO_MASK[mask_type]
        assert hasattr(data, mask_name)

    # Verify val_mask is all zeros (blank mask)
    val_mask_name = SPLIT_TO_MASK[TRAINING.VALIDATION]
    val_mask = getattr(data, val_mask_name)
    assert torch.all(~val_mask)
    assert val_mask.sum().item() == 0


def test_no_mask_ignores_unused_params_with_warnings(napistu_graph, caplog):
    """Test that no_mask strategy ignores unused parameters and warns about them."""
    # Set up logging to capture warnings
    caplog.set_level(logging.WARNING)

    # Test with both splitting parameters and completely unexpected parameters
    data = napistu_graph_to_napistu_data(
        napistu_graph,
        splitting_strategy=SPLITTING_STRATEGIES.NO_MASK,
        train_size=0.8,  # Splitting param - should be ignored
        test_size=0.1,  # Splitting param - should be ignored
        val_size=0.1,  # Splitting param - should be ignored
        unexpected_param="should_be_ignored",  # Unexpected param - should be ignored
        another_unexpected_param=42,  # Another unexpected param - should be ignored
        verbose=False,
    )

    # Verify we get a Data object
    assert isinstance(data, Data)
    assert data.num_nodes > 0
    assert data.num_edges > 0

    # Verify no masks are present (no_mask strategy doesn't create masks)
    for mask_type in [TRAINING.TRAIN, TRAINING.TEST, TRAINING.VALIDATION]:
        mask_name = SPLIT_TO_MASK[mask_type]
        assert not hasattr(data, mask_name)

    # Verify warning was logged about ignored parameters
    assert len(caplog.records) > 0
    warning_messages = [record.message for record in caplog.records]

    # Check that we got a warning about ignored parameters
    ignored_params_warning = any(
        "parameters were ignored" in msg and SPLITTING_STRATEGIES.NO_MASK in msg
        for msg in warning_messages
    )
    assert (
        ignored_params_warning
    ), f"Expected warning about ignored parameters, got: {warning_messages}"

    # Verify the warning mentions the ignored parameters
    warning_msg = next(
        msg for msg in warning_messages if "parameters were ignored" in msg
    )
    assert (
        "train_size" in warning_msg
        or "test_size" in warning_msg
        or "val_size" in warning_msg
    )
    assert "unexpected_param" in warning_msg


def test_name_napistu_data():
    """Test _name_napistu_data validates docstring examples."""

    # Create a mock labeling manager
    labeling_manager = LabelingManager(
        label_attribute=LABEL_TYPE.SPECIES_TYPE,
        exclude_vertex_attributes=[LABEL_TYPE.SPECIES_TYPE],
        augment_summary_types=[],
    )

    # Test supervised data with vertex masking (from docstring example)
    supervised_name = _name_napistu_data(
        splitting_strategy=SPLITTING_STRATEGIES.VERTEX_MASK,
        labels=torch.tensor([0, 1, 2]),  # Mock labels for supervised case
        labeling_manager=labeling_manager,
    )
    # LABEL_TYPE.SPECIES_TYPE = "species_type", SPLITTING_STRATEGIES.VERTEX_MASK = "vertex_mask"
    expected_supervised = "_".join(
        [
            LABELING.LABELED,
            LABEL_TYPE.SPECIES_TYPE,
            SPLITTING_STRATEGIES.VERTEX_MASK,
        ]
    )
    assert supervised_name == expected_supervised

    # Test unsupervised data with no masking (from docstring example)
    unsupervised_name = _name_napistu_data(
        splitting_strategy=SPLITTING_STRATEGIES.NO_MASK,
        labels=None,  # No labels for unsupervised case
        labeling_manager=None,
    )
    # "no_mask" is dropped from the name
    expected_unsupervised = LABELING.UNLABELED
    assert unsupervised_name == expected_unsupervised


def test_ignore_graph_attributes(napistu_graph):
    """Test that _ignore_graph_attributes removes specified edge attributes."""
    # Add a test attribute
    test_attr = "test_ignore_attr"
    napistu_graph.es[test_attr] = [1.0] * napistu_graph.ecount()
    assert test_attr in napistu_graph.es.attributes()

    # Remove it
    _ignore_graph_attributes(napistu_graph, {NAPISTU_GRAPH.EDGES: [test_attr]})

    # Verify it's gone
    assert test_attr not in napistu_graph.es.attributes()
