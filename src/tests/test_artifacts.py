"""Tests for artifact registry and creation functions."""

import pandas as pd

from napistu_torch.constants import ARTIFACT_TYPES
from napistu_torch.load.artifacts import (
    DEFAULT_ARTIFACT_REGISTRY,
    create_artifact,
    get_artifact_info,
    list_available_artifacts,
    validate_artifact_registry,
)


def test_create_all_default_artifacts(sbml_dfs, napistu_graph):
    """Test creating each default artifact and verify basic properties."""

    # Get list of available artifacts
    available_artifacts = list_available_artifacts()
    assert len(available_artifacts) > 0, "Should have at least one artifact"

    # Create each artifact and verify basic properties
    for artifact_name in available_artifacts:
        # Get artifact info first
        info = get_artifact_info(artifact_name)
        assert info.name == artifact_name
        assert info.artifact_type in [
            ARTIFACT_TYPES.NAPISTU_DATA,
            ARTIFACT_TYPES.VERTEX_TENSOR,
            ARTIFACT_TYPES.PANDAS_DFS,
        ]

        # Create the artifact
        artifact = create_artifact(artifact_name, sbml_dfs, napistu_graph)

        # Verify artifact type matches expected type
        if info.artifact_type == ARTIFACT_TYPES.NAPISTU_DATA:
            from napistu_torch.napistu_data import NapistuData

            assert isinstance(artifact, NapistuData)
        elif info.artifact_type == ARTIFACT_TYPES.VERTEX_TENSOR:
            from napistu_torch.vertex_tensor import VertexTensor

            assert isinstance(artifact, VertexTensor)
        elif info.artifact_type == ARTIFACT_TYPES.PANDAS_DFS:
            assert isinstance(artifact, pd.DataFrame)


def test_validate_default_artifact_registry():
    """Test that the default artifact registry passes validation."""

    # This should not raise any exceptions
    validate_artifact_registry(DEFAULT_ARTIFACT_REGISTRY)

    # Verify registry has expected properties
    assert len(DEFAULT_ARTIFACT_REGISTRY) > 0, "Registry should not be empty"

    # Verify all keys match definition names
    for key, definition in DEFAULT_ARTIFACT_REGISTRY.items():
        assert (
            key == definition.name
        ), f"Key '{key}' should match definition name '{definition.name}'"

    # Verify no duplicate names
    names = [defn.name for defn in DEFAULT_ARTIFACT_REGISTRY.values()]
    assert len(names) == len(set(names)), "Should have no duplicate artifact names"
