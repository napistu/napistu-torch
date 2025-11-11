import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from napistu.network.constants import (
    IGRAPH_DEFS,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_VERTICES,
)
from napistu.network.ng_core import NapistuGraph

from napistu_torch.constants import NAPISTU_DATA
from napistu_torch.load.constants import (
    STRATIFICATION_DEFS,
    STRATIFY_BY,
    VALID_STRATIFY_BY,
)
from napistu_torch.napistu_data import NapistuData

logger = logging.getLogger(__name__)


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
        + STRATIFICATION_DEFS.FROM_TO_SEPARATOR
        + pd.Series(target_part, index=df.index)
    )

    return edge_strata


def ensure_strata_series(
    edge_strata: Optional[Union[pd.Series, pd.DataFrame]],
) -> Optional[pd.Series]:
    """
    Ensure edge_strata is a pandas Series.

    Converts DataFrame with single column "edge_strata" back to Series,
    or returns Series as-is if already a Series.

    Parameters
    ----------
    edge_strata : pd.Series or pd.DataFrame or None
        Edge strata data. If DataFrame, must have single column named "edge_strata".

    Returns
    -------
    pd.Series
        Edge strata as a Series. If None, returns None.

    Raises
    ------
    ValueError
        If edge_strata is DataFrame but doesn't have exactly one column named "edge_strata".
    """
    if edge_strata is None:
        return None
    elif isinstance(edge_strata, pd.Series):
        return edge_strata
    elif isinstance(edge_strata, pd.DataFrame):
        if (
            edge_strata.shape[1] == 1
            and STRATIFICATION_DEFS.EDGE_STRATA in edge_strata.columns
        ):
            return edge_strata[STRATIFICATION_DEFS.EDGE_STRATA]
        else:
            raise ValueError(
                f"DataFrame must have exactly one column named '{STRATIFICATION_DEFS.EDGE_STRATA}', "
                f"got columns: {list(edge_strata.columns)}"
            )
    else:
        raise TypeError(f"Expected pd.Series or pd.DataFrame, got {type(edge_strata)}")


def validate_edge_strata_alignment(
    napistu_data: NapistuData, edge_strata: pd.Series
) -> None:
    """Verify edge_strata from->to aligns with NapistuData edge names."""

    if (
        not hasattr(napistu_data, NAPISTU_DATA.NG_EDGE_NAMES)
        or getattr(napistu_data, NAPISTU_DATA.NG_EDGE_NAMES) is None
    ):
        logger.warning(
            "NapistuData edge names not found. Edge strata validation skipped."
        )
        return None

    edge_names = napistu_data.ng_edge_names

    from_match = (
        edge_names[NAPISTU_GRAPH_EDGES.FROM].values
        == edge_strata.index.get_level_values(NAPISTU_GRAPH_EDGES.FROM).values
    ).all()
    to_match = (
        edge_names[NAPISTU_GRAPH_EDGES.TO].values
        == edge_strata.index.get_level_values(NAPISTU_GRAPH_EDGES.TO).values
    ).all()

    if not (from_match and to_match):
        raise ValueError("Edge strata misalignment with NapistuData edge names")

    return None
