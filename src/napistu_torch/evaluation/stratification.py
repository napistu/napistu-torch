import numpy as np
import pandas as pd
from napistu.network.constants import IGRAPH_DEFS, NAPISTU_GRAPH_VERTICES
from napistu.network.ng_core import NapistuGraph


def create_composite_edge_strata(napistu_graph: NapistuGraph):
    """
    Create a composite edge attribute by concatenating the endpoint attributes.

    Parameters
    ----------
    napistu_graph : NapistuGraph
        A NapistuGraph object.

    Returns
    -------
    pd.Series
        A series with the composite edge attribute.
    """

    df = napistu_graph.get_edge_endpoint_attributes(
        [NAPISTU_GRAPH_VERTICES.NODE_TYPE, NAPISTU_GRAPH_VERTICES.SPECIES_TYPE]
    )

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

    edge_strata = (
        pd.Series(source_part, index=df.index)
        + " -> "
        + pd.Series(target_part, index=df.index)
    )

    return edge_strata
