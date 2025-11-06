"""Tests for napistu_torch lightning workflows module."""

import pytest

from napistu_torch.constants import EXPERIMENT_CONFIG
from napistu_torch.lightning.constants import EXPERIMENT_DICT
from napistu_torch.lightning.workflows import ExperimentDict


@pytest.mark.skip_on_windows
def test_experiment_dict_fixture(experiment_dict):
    """Test that the experiment_dict fixture is properly structured and validated."""
    # Validate the experiment_dict structure using Pydantic
    validated = ExperimentDict(
        data_module=experiment_dict[EXPERIMENT_DICT.DATA_MODULE],
        model=experiment_dict[EXPERIMENT_DICT.MODEL],
        trainer=experiment_dict[EXPERIMENT_DICT.TRAINER],
        run_manifest=experiment_dict[EXPERIMENT_DICT.RUN_MANIFEST],
        wandb_logger=experiment_dict[EXPERIMENT_DICT.WANDB_LOGGER],
    )

    # Verify all components are present
    assert EXPERIMENT_DICT.DATA_MODULE in experiment_dict
    assert EXPERIMENT_DICT.MODEL in experiment_dict
    assert EXPERIMENT_DICT.TRAINER in experiment_dict
    assert EXPERIMENT_DICT.RUN_MANIFEST in experiment_dict
    assert EXPERIMENT_DICT.WANDB_LOGGER in experiment_dict

    # Verify the validated object has the correct types
    assert validated.data_module is not None
    assert validated.model is not None
    assert validated.trainer is not None
    assert validated.run_manifest is not None
    # wandb_logger is None when wandb is disabled (which is the case in tests)
    assert validated.wandb_logger is None


@pytest.mark.skip_on_windows
def test_experiment_dict_run_manifest(experiment_dict):
    """Test that run_manifest contains expected information."""
    run_manifest = experiment_dict[EXPERIMENT_DICT.RUN_MANIFEST]

    # Verify run_manifest has the expected attributes
    assert run_manifest.experiment_name == "test_experiment"
    assert run_manifest.experiment_config is not None
    assert EXPERIMENT_CONFIG.TASK in run_manifest.experiment_config
    assert EXPERIMENT_CONFIG.MODEL in run_manifest.experiment_config
    assert EXPERIMENT_CONFIG.TRAINING in run_manifest.experiment_config
    assert run_manifest.created_at is not None


@pytest.mark.skip_on_windows
def test_experiment_dict_data_module(experiment_dict):
    """Test that data_module is properly configured."""
    data_module = experiment_dict[EXPERIMENT_DICT.DATA_MODULE]

    # Verify data module has the expected attributes
    assert hasattr(data_module, "num_node_features")
    assert hasattr(data_module, "num_edge_features")
    assert data_module.num_node_features > 0
    assert data_module.num_edge_features >= 0


@pytest.mark.skip_on_windows
def test_experiment_dict_model(experiment_dict):
    """Test that model is properly configured."""
    model = experiment_dict[EXPERIMENT_DICT.MODEL]

    # Verify model has the expected attributes
    assert hasattr(model, "task")
    assert hasattr(model, "config")
    assert model.task is not None


@pytest.mark.skip_on_windows
def test_experiment_dict_trainer(experiment_dict):
    """Test that trainer is properly configured."""
    trainer = experiment_dict[EXPERIMENT_DICT.TRAINER]

    # Verify trainer has the expected attributes
    assert hasattr(trainer, "config")
    assert hasattr(trainer, "_trainer")
    assert trainer.config is not None
