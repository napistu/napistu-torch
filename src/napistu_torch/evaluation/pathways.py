from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from napistu.ingestion.constants import DEFAULT_PRIORITIZED_PATHWAYS
from napistu.network.constants import (
    ADDING_ENTITY_DATA_DEFS,
    NAPISTU_GRAPH_VERTICES,
    VERTEX_SBML_DFS_SUMMARIES,
)
from napistu.network.ng_core import NapistuGraph
from napistu.sbml_dfs_core import SBML_dfs

from napistu_torch.evaluation.constants import (
    EVALUATION_TENSOR_DESCRIPTIONS,
    EVALUATION_TENSORS,
    PATHWAY_SIMILARITY_DEFS,
)
from napistu_torch.utils.torch_utils import ensure_device, memory_manager
from napistu_torch.vertex_tensor import VertexTensor


def calculate_pathway_similarities(
    embedding_matrix: torch.Tensor,
    pathway_assignments: torch.Tensor,
    pathway_names: list[str],
    filtering_mask: Optional[torch.Tensor] = None,
    priority_pathways: list[str] = DEFAULT_PRIORITIZED_PATHWAYS,
    device: torch.device = None,
) -> dict[str, float]:
    """
    Calculate pathway similarity based on average within-category cosine similarity.

    Parameters
    ----------
    embedding_matrix: torch.Tensor
        (I, K) tensor of feature embeddings
    pathway_assignments: torch.Tensor
        (I, C) binary tensor of category memberships
    pathway_names: list[str]
        list of category/pathway names corresponding to per_category_sim
    filtering_mask:
        optional (I,) tensor of boolean mask to filter embeddings and assignments

    Returns
    -------
    dict mapping pathway names to their similarities:
    """

    if filtering_mask is None:
        # Create 1D mask for rows only
        filtering_mask = torch.ones(pathway_assignments.shape[0], dtype=torch.bool)
    else:
        if embedding_matrix.shape[0] != filtering_mask.shape[0]:
            raise ValueError(
                "embedding_matrix and filtering_mask must have the same number of rows"
            )

    if embedding_matrix.shape[0] != pathway_assignments.shape[0]:
        raise ValueError(
            "embedding_matrix and pathway_assignments must have the same number of rows"
        )

    per_cat_sim, overall_sim = _within_category_similarity(
        embedding_matrix[filtering_mask],
        pathway_assignments[filtering_mask],
        device=device,
    )

    pathway_similarities = _rollup_pathway_similarities(
        per_category_sim=per_cat_sim,
        pathway_names=pathway_names,
        priority_pathways=priority_pathways,
    )

    pathway_similarities[PATHWAY_SIMILARITY_DEFS.OVERALL] = overall_sim

    return pathway_similarities


def get_comprehensive_source_membership(
    napistu_graph: NapistuGraph, sbml_dfs: SBML_dfs
) -> VertexTensor:
    """
    Get the comprehensive source membership for a given NapistuGraph and SBML_dfs.

    Parameters
    ----------
    napistu_graph: NapistuGraph
        NapistuGraph object to add the comprehensive source membership from.
    sbml_dfs: SBML_dfs
        SBML_dfs object containing vertex source information to add to the NapistuGraph.

    Returns
    -------
    VertexTensor
        VertexTensor object containing the comprehensive source membership.
    """

    # add all source information to the graph
    working_napistu_graph = napistu_graph.copy()
    working_napistu_graph.add_sbml_dfs_summaries(
        sbml_dfs,
        summary_types=[VERTEX_SBML_DFS_SUMMARIES.SOURCES],
        priority_pathways=None,  # include all pathways including all of the fine-grained Reactome ones
        add_name_prefixes=False,
        mode=ADDING_ENTITY_DATA_DEFS.FRESH,
        overwrite=True,
        binarize=True,
    )

    binary_pathway_memberships = (
        working_napistu_graph.get_vertex_dataframe().select_dtypes(include=[int])
    )

    ng_vertex_names = working_napistu_graph.get_vertex_series(
        NAPISTU_GRAPH_VERTICES.NAME
    )
    feature_names = binary_pathway_memberships.columns.tolist()

    return VertexTensor(
        data=torch.Tensor(binary_pathway_memberships.values),
        feature_names=feature_names,
        vertex_names=ng_vertex_names,
        name=EVALUATION_TENSORS.COMPREHENSIVE_PATHWAY_MEMBERSHIPS,
        description=EVALUATION_TENSOR_DESCRIPTIONS[
            EVALUATION_TENSORS.COMPREHENSIVE_PATHWAY_MEMBERSHIPS
        ],
    )


# private functions


def _rollup_pathway_similarities(
    per_category_sim: np.ndarray,
    pathway_names: list[str],
    priority_pathways: list[str] = DEFAULT_PRIORITIZED_PATHWAYS,
) -> dict[str, float]:
    """
    Rollup per-category similarities into individual priority pathways and aggregated others.

    Parameters
    ----------
    per_category_sim: numpy.ndarray
        (C,) array of average cosine similarities per category
    pathway_names: list
        category/pathway names corresponding to per_category_sim
    priority_pathways: list
        priority pathway names (these pathways will be preserved as their own entries)

    Returns
    -------
    dict mapping pathway names to their similarities:
        - Each priority pathway gets its own entry
        - All other pathways are averaged into a single 'other' entry
    """
    priority_set = set(priority_pathways)

    result = {}
    other_sims = []

    for name, sim in zip(pathway_names, per_category_sim):
        if name in priority_set:
            result[name] = sim
        else:
            other_sims.append(sim)

    # Add aggregated 'other' category
    if other_sims:
        result[PATHWAY_SIMILARITY_DEFS.OTHER] = np.mean(other_sims)

    return result


def _within_category_similarity(
    embeddings: torch.Tensor,
    categories: torch.Tensor,
    device: torch.device = None,
) -> tuple[np.ndarray, float]:
    """
    Calculate average within-category cosine similarity (excluding self-pairs).

    Parameters
    ----------
    embeddings: torch.Tensor
        (I, K) tensor of feature embeddings
    categories: torch.Tensor
        (I, C) binary tensor of category memberships
    device: torch.device
        device to use (if None, uses embeddings.device)

    Returns
    -------
    tuple[np.ndarray, float]
        per_category_sim: (C,) numpy array of average cosine similarities per category
        overall_sim: float
            average across all categories
    """
    if device is None:
        device = embeddings.device

    device = ensure_device(device)
    with memory_manager(device):
        # Ensure inputs are on the correct device
        embeddings = embeddings.to(device)
        categories = categories.to(device)

        # Normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)

        # Compute all pairwise cosine similarities
        similarity_matrix = embeddings_norm @ embeddings_norm.T

        # Compute category similarity sums
        category_similarity_sums = categories.T @ similarity_matrix @ categories
        within_sums = torch.diag(category_similarity_sums)

        # Subtract diagonal entries (self-similarities)
        members_per_category = categories.sum(dim=0)
        within_sums = within_sums - members_per_category

        # Divide by N*(N-1) pairs (excluding self-pairs)
        num_pairs = members_per_category * (members_per_category - 1)

        per_category_sim = within_sums / num_pairs.clamp(min=1)
        overall_sim = within_sums.sum() / num_pairs.sum().clamp(min=1)

        # Convert to numpy
        per_category_sim = per_category_sim.cpu().numpy()
        overall_sim = (
            overall_sim.item()
        )  # .item() converts scalar tensor to Python float

        return per_category_sim, overall_sim
