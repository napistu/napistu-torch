import torch
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
)
from napistu_torch.vertex_tensor import VertexTensor


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
    )

    vertex_pathway_memberships = (
        working_napistu_graph.get_vertex_dataframe().select_dtypes(include=[int])
    )

    ng_vertex_names = working_napistu_graph.get_vertex_series(
        NAPISTU_GRAPH_VERTICES.NAME
    )
    feature_names = vertex_pathway_memberships.columns.tolist()

    return VertexTensor(
        data=torch.Tensor(vertex_pathway_memberships.values),
        feature_names=feature_names,
        vertex_names=ng_vertex_names,
        name=EVALUATION_TENSORS.COMPREHENSIVE_PATHWAY_MEMBERSHIPS,
        description=EVALUATION_TENSOR_DESCRIPTIONS[
            EVALUATION_TENSORS.COMPREHENSIVE_PATHWAY_MEMBERSHIPS
        ],
    )
