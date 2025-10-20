import torch
from napistu.network.constants import (
    ADDING_ENTITY_DATA_DEFS,
    NAPISTU_GRAPH_VERTICES,
    VERTEX_SBML_DFS_SUMMARIES,
)
from napistu.network.ng_core import NapistuGraph
from napistu.sbml_dfs_core import SBML_dfs

from napistu_torch.evaluation.constants import EVALUATION_DATA


def get_comprehensive_source_membership(
    napistu_graph: NapistuGraph, sbml_dfs: SBML_dfs
) -> dict:

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

    return {
        EVALUATION_DATA.DATA: torch.Tensor(vertex_pathway_memberships.values),
        EVALUATION_DATA.FEATURE_NAMES: feature_names,
        EVALUATION_DATA.VERTEX_NAMES: ng_vertex_names,
    }
