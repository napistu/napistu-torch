"""Tests for evaluation stratification functions."""

import pandas as pd
from napistu.network.constants import NAPISTU_GRAPH_NODE_TYPES, NAPISTU_GRAPH_VERTICES

from napistu_torch.load.constants import STRATIFICATION_DEFS, STRATIFY_BY
from napistu_torch.load.stratification import create_composite_edge_strata


def test_create_composite_edge_strata(napistu_graph):
    """Test create_composite_edge_strata returns correct index and expected levels."""

    # Get the composite edge strata
    edge_strata = create_composite_edge_strata(napistu_graph)

    # Verify it's a pandas Series
    assert isinstance(edge_strata, pd.Series)

    # Verify the index matches the original dataframe
    original_df = napistu_graph.get_edge_endpoint_attributes(
        [NAPISTU_GRAPH_VERTICES.NODE_TYPE, NAPISTU_GRAPH_VERTICES.SPECIES_TYPE]
    )
    assert edge_strata.index.equals(original_df.index)

    # Verify we have the expected number of unique levels (6 based on test output)
    assert edge_strata.nunique() == 6

    # Get the unique levels
    unique_levels = set(edge_strata.unique())

    # Verify all levels contain the arrow separator
    assert all(" -> " in level for level in unique_levels)

    # Check for specific expected levels based on the test output
    expected_levels = {
        "reaction -> species (metabolite)",
        "reaction -> species (complex)",
        "reaction -> species (protein)",
        "species (complex) -> reaction",
        "species (metabolite) -> reaction",
        "species (protein) -> reaction",
    }

    assert (
        unique_levels == expected_levels
    ), f"Expected levels {expected_levels}, got {unique_levels}"


def test_create_composite_edge_strata_edge_sbo_terms(napistu_graph):
    """Test create_composite_edge_strata with edge_sbo_terms option."""
    # Get edge strata by SBO terms
    edge_strata = create_composite_edge_strata(
        napistu_graph, stratify_by=STRATIFY_BY.EDGE_SBO_TERMS
    )

    # Verify it's a pandas Series
    assert isinstance(edge_strata, pd.Series)

    # Verify the index matches the edges in the graph
    assert len(edge_strata) == len(napistu_graph.es)

    # Verify all levels contain the arrow separator
    unique_levels = set(edge_strata.unique())
    assert all(
        STRATIFICATION_DEFS.FROM_TO_SEPARATOR in level for level in unique_levels
    )

    # Verify that the values are SBO term names (not SBO codes)
    # SBO term names should be lowercase strings like "reactant", "product", "catalyst", etc.
    # or "reaction" for missing values
    from napistu.constants import MINI_SBO_TO_NAME

    valid_sbo_names = set(MINI_SBO_TO_NAME.values()) | {
        NAPISTU_GRAPH_NODE_TYPES.REACTION
    }
    for level in unique_levels:
        parts = level.split(STRATIFICATION_DEFS.FROM_TO_SEPARATOR)
        assert (
            len(parts) == 2
        ), f"Level should have two parts separated by '{STRATIFICATION_DEFS.FROM_TO_SEPARATOR}'"
        upstream, downstream = parts
        assert (
            upstream in valid_sbo_names
        ), f"Upstream part '{upstream}' is not a valid SBO name"
        assert (
            downstream in valid_sbo_names
        ), f"Downstream part '{downstream}' is not a valid SBO name"
