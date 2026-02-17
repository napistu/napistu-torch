"""
Utilities for working with Napistu data structures.

Public Functions
----------------
map_identifiers_to_vertex_names :
    Map identifiers to vertex names.
"""

import logging
from typing import List

import pandas as pd
from napistu.constants import BQB, IDENTIFIERS, ONTOLOGIES, SBML_DFS
from napistu.identifiers import _check_species_identifiers_table
from napistu.network.constants import NAPISTU_GRAPH_VERTICES
from napistu.ontologies.standardization import create_uri_url

from napistu_torch.utils.constants import EXPECTED_NAME_TO_SID_MAP_COLUMNS

logger = logging.getLogger(__name__)


def map_identifiers_to_vertex_names(
    id_list: List[str],
    species_identifiers: pd.DataFrame,
    name_to_sid_map: pd.DataFrame,
    ontology: str = ONTOLOGIES.ENSEMBL_GENE,
) -> pd.DataFrame:
    f"""
    Get an identifier to Napistu ID Map

    This function takes a list of systematic identifiers and a species identifiers table and returns a map of systematic identifiers to Napistu IDs.

    Parameters
    ----------
    id_list : List[str]
        A list of molecular species identifiers.
    species_identifiers : pd.DataFrame
        A species identifiers table.
    name_to_sid_map : pd.DataFrame
        A name to sid map.
    ontology : str
        The ontology of the identifiers in `id_list`. Default is "ensembl_gene".

    Returns
    -------
    id_map: pd.DataFrame
        A pd.DataFrame with columns:
        - {ontology}: a column named after the `ontology` parameter containing the members of id_list which are found among the systematic identifiers in `species_identifiers`
        - s_id: the Napistu molecular species ID
        - name: the Napistu graph vertex name
    """

    # validate inputs

    try:
        _ = [create_uri_url(ontology, x, strict=True) for x in id_list]
    except Exception as e:
        logger.error(
            f"Some identifiers in the id_list were not the {ontology} format, error: {e}"
        )
        raise e

    # validate the species identifiers table
    _check_species_identifiers_table(species_identifiers)

    # either a DF or Series
    if isinstance(name_to_sid_map, pd.Series):
        name_to_sid_map = name_to_sid_map.to_frame()

    if not set(EXPECTED_NAME_TO_SID_MAP_COLUMNS).issubset(name_to_sid_map.columns):
        raise ValueError(
            f"name_to_sid_map must contain the following variables either in the columns or as the index: {EXPECTED_NAME_TO_SID_MAP_COLUMNS}"
        )

    id_map = (
        species_identifiers.query(f"{IDENTIFIERS.ONTOLOGY} == @ontology")
        .query(f"{IDENTIFIERS.IDENTIFIER} in @id_list")
        .query(f"{IDENTIFIERS.BQB} != '{BQB.HAS_PART}'")
        # add vertex IDs
        .merge(name_to_sid_map, on=SBML_DFS.S_ID, how="inner")[
            [IDENTIFIERS.IDENTIFIER, SBML_DFS.S_ID, NAPISTU_GRAPH_VERTICES.NAME]
        ]
        .rename(columns={IDENTIFIERS.IDENTIFIER: ontology})
    )

    if id_map.shape[0] == 0:
        logger.warning(
            f"No {ontology} identifiers found in the species identifiers table"
        )
    n_missing = len(id_list) - len(id_map[ontology].unique())
    if n_missing > 0:
        missing_str = f"{(n_missing / len(id_list)) * 100:.1f}%"
        logger.info(
            f"{n_missing} ({missing_str}) identifiers were missing from the {ontology} identifiers in the species identifiers table. These ids will be missing in the output."
        )

    return id_map
