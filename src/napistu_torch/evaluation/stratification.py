import numpy as np
import pandas as pd
from napistu.network.constants import IGRAPH_DEFS, NAPISTU_GRAPH_VERTICES
from napistu.network.ng_core import NapistuGraph

from napistu_torch.evaluation.constants import (
    STRATIFY_BY,
    VALID_STRATIFY_BY,
)


def create_composite_edge_strata(
    napistu_graph: NapistuGraph, stratify_by: str = STRATIFY_BY.NODE_SPECIES_TYPE
) -> pd.Series:
    """
    Create a composite edge attribute by concatenating the endpoint attributes.

    Parameters
    ----------
    napistu_graph : NapistuGraph
        A NapistuGraph object.
    stratify_by : str
        The attribute(s) to stratify by. Must be one of the following:
        - STRATIFY_BY.NODE_SPECIES_TYPE - species and node type
        - STRATIFY_BY.NODE_TYPE - node type (species and reactions)

    Returns
    -------
    pd.Series
        A series with the composite edge attribute.
    """

    if stratify_by == STRATIFY_BY.NODE_SPECIES_TYPE:
        endpoint_attributes = [
            NAPISTU_GRAPH_VERTICES.NODE_TYPE,
            NAPISTU_GRAPH_VERTICES.SPECIES_TYPE,
        ]
    elif stratify_by == STRATIFY_BY.NODE_TYPE:
        endpoint_attributes = [NAPISTU_GRAPH_VERTICES.NODE_TYPE]
    else:
        raise ValueError(
            f"Invalid stratify_by value: {stratify_by}. Must be one of: {VALID_STRATIFY_BY}"
        )

    df = napistu_graph.get_edge_endpoint_attributes(endpoint_attributes)

    if stratify_by == STRATIFY_BY.NODE_SPECIES_TYPE:
        source_part = np.where(
            df[NAPISTU_GRAPH_VERTICES.SPECIES_TYPE][IGRAPH_DEFS.SOURCE].notna(),
            df[NAPISTU_GRAPH_VERTICES.NODE_TYPE][IGRAPH_DEFS.SOURCE]
            + " ("
            + df[NAPISTU_GRAPH_VERTICES.SPECIES_TYPE][IGRAPH_DEFS.SOURCE]
            + ")",
            df[NAPISTU_GRAPH_VERTICES.NODE_TYPE][IGRAPH_DEFS.SOURCE],
        )
        target_part = np.where(
            df[NAPISTU_GRAPH_VERTICES.SPECIES_TYPE][IGRAPH_DEFS.TARGET].notna(),
            df[NAPISTU_GRAPH_VERTICES.NODE_TYPE][IGRAPH_DEFS.TARGET]
            + " ("
            + df[NAPISTU_GRAPH_VERTICES.SPECIES_TYPE][IGRAPH_DEFS.TARGET]
            + ")",
            df[NAPISTU_GRAPH_VERTICES.NODE_TYPE][IGRAPH_DEFS.TARGET],
        )
    elif stratify_by == STRATIFY_BY.NODE_TYPE:
        source_part = df[NAPISTU_GRAPH_VERTICES.NODE_TYPE][IGRAPH_DEFS.SOURCE]
        target_part = df[NAPISTU_GRAPH_VERTICES.NODE_TYPE][IGRAPH_DEFS.TARGET]
    else:
        raise ValueError(
            f"Invalid stratify_by value: {stratify_by}. Must be one of: {VALID_STRATIFY_BY}"
        )

    edge_strata = (
        pd.Series(source_part, index=df.index)
        + " -> "
        + pd.Series(target_part, index=df.index)
    )

    return edge_strata
