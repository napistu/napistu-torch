"""
Artifact registry for predefined NapistuData and VertexTensor objects.

This module defines all standard artifacts that can be created from SBML_dfs
and NapistuGraph objects. Each artifact has a creation function that encapsulates
the ETL logic.

To add a new artifact:
1. Create a creation function (e.g., create_my_artifact)
2. Add an ArtifactDefinition to _ARTIFACTS list
3. The registry will be built automatically
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Callable, List, Union

import pandas as pd
from napistu.network.ng_core import NapistuGraph
from napistu.sbml_dfs_core import SBML_dfs
from pydantic import BaseModel, field_validator

from napistu_torch.constants import (
    ARTIFACT_TYPES,
    VALID_ARTIFACT_TYPES,
)
from napistu_torch.evaluation.constants import STRATIFY_BY
from napistu_torch.evaluation.pathways import (
    get_comprehensive_source_membership,
)
from napistu_torch.evaluation.stratification import create_composite_edge_strata
from napistu_torch.labeling.constants import LABEL_TYPE
from napistu_torch.load.constants import (
    ARTIFACT_DEFS,
    SPLITTING_STRATEGIES,
)
from napistu_torch.load.napistu_graphs import (
    construct_supervised_pyg_data,
    construct_unsupervised_pyg_data,
)
from napistu_torch.napistu_data import NapistuData
from napistu_torch.vertex_tensor import VertexTensor

logger = logging.getLogger(__name__)


class ArtifactDefinition(BaseModel):
    """Metadata for a predefined artifact."""

    name: str
    artifact_type: str
    creation_func: Callable
    description: str = ""

    @field_validator(ARTIFACT_DEFS.NAME)
    @classmethod
    def validate_name(cls, v):
        """Validate artifact name format."""
        if " " in v:
            raise ValueError("Artifact names cannot contain spaces")
        if not v:
            raise ValueError("Artifact name cannot be empty")
        return v

    @field_validator(ARTIFACT_DEFS.ARTIFACT_TYPE)
    @classmethod
    def validate_artifact_type(cls, v):
        """Validate artifact type."""
        if v not in VALID_ARTIFACT_TYPES:
            raise ValueError(f"Invalid artifact type: {v}")
        return v

    class Config:
        arbitrary_types_allowed = True  # Needed for Callable


def create_artifact(
    name: str,
    sbml_dfs: SBML_dfs,
    napistu_graph: NapistuGraph,
    artifact_registry: dict[str, ArtifactDefinition] = None,
) -> Union[NapistuData, VertexTensor, pd.DataFrame]:
    """
    Create an artifact by name using the registry.

    Parameters
    ----------
    name : str
        Name of artifact to create
    sbml_dfs : SBML_dfs
        SBML data structure
    napistu_graph : NapistuGraph
        Napistu graph
    artifact_registry : dict[str, ArtifactDefinition]
        Artifact registry to use. If not provided, the default registry will be used.

    Returns
    -------
    Union[NapistuData, VertexTensor]
        The created artifact

    Raises
    ------
    KeyError
        If artifact name not in registry

    Examples
    --------
    >>> sbml_dfs = SBML_dfs.from_pickle("data.pkl")
    >>> napistu_graph = NapistuGraph.from_pickle("graph.pkl")
    >>> artifact = create_artifact("unsupervised", sbml_dfs, napistu_graph)
    """
    if artifact_registry is None:
        artifact_registry = DEFAULT_ARTIFACT_REGISTRY

    if name not in artifact_registry:
        available = list(artifact_registry.keys())
        raise KeyError(f"Unknown artifact: '{name}'. Available artifacts: {available}")

    definition = artifact_registry[name]
    logger.info(f"Creating artifact '{name}': {definition.description}")

    # Build arguments dict based on what the creation function actually expects
    func_params = definition.creation_func.__code__.co_varnames
    args_dict = {}

    if "sbml_dfs" in func_params:
        args_dict["sbml_dfs"] = sbml_dfs
    if "napistu_graph" in func_params:
        args_dict["napistu_graph"] = napistu_graph

    return definition.creation_func(**args_dict)


def get_artifact_info(
    name: str, artifact_registry: dict[str, ArtifactDefinition] = None
) -> ArtifactDefinition:
    """
    Get information about an artifact.

    Parameters
    ----------
    name : str
        Artifact name
    artifact_registry : dict[str, ArtifactDefinition]
        Artifact registry to use. If not provided, the default registry will be used.

    Returns
    -------
    ArtifactDefinition
        Artifact metadata including type and description

    Raises
    ------
    KeyError
        If artifact not in registry

    Examples
    --------
    >>> info = get_artifact_info("unsupervised")
    >>> print(info.artifact_type)
    'napistu_data'
    >>> print(info.description)
    'Unsupervised learning data without masking'
    """
    if artifact_registry is None:
        artifact_registry = DEFAULT_ARTIFACT_REGISTRY

    if name not in artifact_registry:
        available = list(artifact_registry.keys())
        raise KeyError(f"Unknown artifact: '{name}'. Available artifacts: {available}")
    return artifact_registry[name]


def list_available_artifacts(
    artifact_registry: dict[str, ArtifactDefinition] = None,
) -> List[str]:
    """
    List all available artifact names in the registry.

    Parameters
    ----------
    artifact_registry : dict[str, ArtifactDefinition]
        Artifact registry to use. If not provided, the default registry will be used.

    Returns
    -------
    List[str]
        Sorted list of artifact names

    Examples
    --------
    >>> artifacts = list_available_artifacts()
    >>> print(artifacts)
    ['comprehensive_pathway_memberships', 'edge_prediction', 'supervised_species_type', 'unsupervised']
    """
    if artifact_registry is None:
        artifact_registry = DEFAULT_ARTIFACT_REGISTRY

    return sorted(artifact_registry.keys())


def validate_artifact_registry(
    artifact_registry: dict[str, ArtifactDefinition],
) -> None:
    """
    Validate the artifact registry.

    Ensures:
    - Registry is not empty
    - Registry keys match definition names
    - No duplicate names

    Raises
    ------
    ValueError
        If validation fails
    """
    if not artifact_registry:
        raise ValueError("Artifact registry cannot be empty")

    # Check that keys match definition names
    mismatches = [
        (key, defn.name) for key, defn in artifact_registry.items() if key != defn.name
    ]
    if mismatches:
        details = ", ".join([f"key='{k}' vs name='{n}'" for k, n in mismatches])
        raise ValueError(
            f"Registry key/name mismatches found: {details}. "
            f"This indicates a bug in registry construction."
        )

    # Check for duplicate names (should be impossible if keys match names, but be defensive)
    names = [defn.name for defn in artifact_registry.values()]
    if len(names) != len(set(names)):
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        raise ValueError(f"Duplicate artifact names found: {duplicates}")

    logger.debug(
        f"Artifact registry validated: {len(artifact_registry)} artifacts registered"
    )


# artifact creation functions and other private functions


def _create_unsupervised_data(
    sbml_dfs: SBML_dfs, napistu_graph: NapistuGraph
) -> NapistuData:
    """
    Create unsupervised data with no masking.

    Parameters
    ----------
    sbml_dfs : SBML_dfs
        SBML data structure
    napistu_graph : NapistuGraph
        Napistu graph

    Returns
    -------
    NapistuData
        Unsupervised data suitable for full-graph training
    """
    return construct_unsupervised_pyg_data(
        sbml_dfs,
        napistu_graph,
        splitting_strategy=SPLITTING_STRATEGIES.NO_MASK,
    )


def _create_edge_prediction_data(
    sbml_dfs: SBML_dfs, napistu_graph: NapistuGraph
) -> NapistuData:
    """
    Create edge prediction data with edge masking.

    Parameters
    ----------
    sbml_dfs : SBML_dfs
        SBML data structure
    napistu_graph : NapistuGraph
        Napistu graph

    Returns
    -------
    NapistuData
        Edge prediction data with train/val/test edge masks
    """
    return construct_unsupervised_pyg_data(
        sbml_dfs,
        napistu_graph,
        splitting_strategy=SPLITTING_STRATEGIES.EDGE_MASK,
    )


def _create_supervised_species_type_data(
    sbml_dfs: SBML_dfs, napistu_graph: NapistuGraph
) -> NapistuData:
    """
    Create supervised data for species type classification.

    Parameters
    ----------
    sbml_dfs : SBML_dfs
        SBML data structure
    napistu_graph : NapistuGraph
        Napistu graph

    Returns
    -------
    NapistuData
        Supervised node classification data with species type labels
    """
    return construct_supervised_pyg_data(
        sbml_dfs,
        napistu_graph,
        splitting_strategy=SPLITTING_STRATEGIES.VERTEX_MASK,
        label_type=LABEL_TYPE.SPECIES_TYPE,
    )


def _create_comprehensive_pathway_memberships(
    sbml_dfs: SBML_dfs, napistu_graph: NapistuGraph
) -> VertexTensor:
    """
    Create comprehensive source membership tensor.

    Parameters
    ----------
    sbml_dfs : SBML_dfs
        SBML data structure
    napistu_graph : NapistuGraph
        Napistu graph

    Returns
    -------
    VertexTensor
        Comprehensive pathway membership features for all vertices
    """
    return get_comprehensive_source_membership(napistu_graph, sbml_dfs)


def _create_edge_strata_by_node_species_type(
    napistu_graph: NapistuGraph,
) -> pd.DataFrame:
    """
    Create edge strata.

    Parameters
    ----------
    napistu_graph : NapistuGraph
        Napistu graph

    Returns
    -------
    pd.DataFrame
        Edge strata
    """
    return create_composite_edge_strata(
        napistu_graph, stratify_by=STRATIFY_BY.NODE_SPECIES_TYPE
    ).to_frame(name="edge_strata")


def _create_edge_strata_by_node_type(napistu_graph: NapistuGraph) -> pd.DataFrame:
    """
    Create edge strata.

    Parameters
    ----------
    napistu_graph : NapistuGraph
        Napistu graph

    Returns
    -------
    pd.DataFrame
        Edge strata
    """
    return create_composite_edge_strata(
        napistu_graph, stratify_by=STRATIFY_BY.NODE_TYPE
    ).to_frame(name="edge_strata")


# artifact registry

# Define artifacts as a list (single source of truth for names)
DEFAULT_ARTIFACTS = [
    ArtifactDefinition(
        name="unsupervised",
        artifact_type=ARTIFACT_TYPES.NAPISTU_DATA,
        creation_func=_create_unsupervised_data,
        description="Unsupervised learning data without masking",
    ),
    ArtifactDefinition(
        name="edge_prediction",
        artifact_type=ARTIFACT_TYPES.NAPISTU_DATA,
        creation_func=_create_edge_prediction_data,
        description="Edge prediction task with edge masking",
    ),
    ArtifactDefinition(
        name="supervised_species_type",
        artifact_type=ARTIFACT_TYPES.NAPISTU_DATA,
        creation_func=_create_supervised_species_type_data,
        description="Node classification for species types with vertex masking",
    ),
    ArtifactDefinition(
        name="comprehensive_pathway_memberships",
        artifact_type=ARTIFACT_TYPES.VERTEX_TENSOR,
        creation_func=_create_comprehensive_pathway_memberships,
        description="Comprehensive pathway membership features",
    ),
    ArtifactDefinition(
        name="edge_strata_by_node_species_type",
        artifact_type=ARTIFACT_TYPES.PANDAS_DFS,
        creation_func=_create_edge_strata_by_node_species_type,
        description="Edge strata by node + species type",
    ),
    ArtifactDefinition(
        name="edge_strata_by_node_type",
        artifact_type=ARTIFACT_TYPES.PANDAS_DFS,
        creation_func=_create_edge_strata_by_node_type,
        description="Edge strata by node type",
    ),
]

# Build registry dict from list (automatic, no duplication possible)
DEFAULT_ARTIFACT_REGISTRY: dict[str, ArtifactDefinition] = {
    artifact.name: artifact for artifact in DEFAULT_ARTIFACTS
}
