from types import SimpleNamespace

from napistu.constants import SBML_DFS
from napistu.network.constants import (
    IGRAPH_DEFS,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_VERTICES,
)

# artifact defs

ARTIFACT_DEFS = SimpleNamespace(
    NAME="name",
    ARTIFACT_TYPE="artifact_type",
    CREATION_FUNC="creation_func",
    DESCRIPTION="description",
)

# transformation defs

ENCODING_MANAGER = SimpleNamespace(
    COLUMNS="columns",
    TRANSFORMER="transformer",
    # attributes
    FIT="fit",
    TRANSFORM="transform",
    PASSTHROUGH="passthrough",
    # merges
    BASE="base",
    OVERRIDE="override",
)

ENCODING_MANAGER_TABLE = SimpleNamespace(
    TRANSFORM_NAME="transform_name",
    COLUMN="column",
    TRANSFORMER_TYPE="transformer_type",
)

# encodings

ENCODINGS = SimpleNamespace(
    CATEGORICAL="categorical",
    NUMERIC="numeric",
    SPARSE_CATEGORICAL="sparse_categorical",
    SPARSE_NUMERIC="sparse_numeric",
    BINARY="binary",
)

NEVER_ENCODE = {
    SBML_DFS.SC_ID,
    SBML_DFS.S_ID,
    SBML_DFS.C_ID,
    SBML_DFS.R_ID,
    IGRAPH_DEFS.INDEX,
    IGRAPH_DEFS.NAME,
    IGRAPH_DEFS.SOURCE,
    IGRAPH_DEFS.TARGET,
    NAPISTU_GRAPH_VERTICES.NODE_NAME,
    NAPISTU_GRAPH_EDGES.FROM,
    NAPISTU_GRAPH_EDGES.TO,
}

# Node configuration
VERTEX_DEFAULT_TRANSFORMS = {
    ENCODINGS.CATEGORICAL: {
        NAPISTU_GRAPH_VERTICES.NODE_TYPE,
    },
    ENCODINGS.SPARSE_CATEGORICAL: {
        NAPISTU_GRAPH_VERTICES.SPECIES_TYPE,
    },
}

# Edge configuration
EDGE_DEFAULT_TRANSFORMS = {
    ENCODINGS.CATEGORICAL: {
        NAPISTU_GRAPH_EDGES.DIRECTION,
        NAPISTU_GRAPH_EDGES.SBO_TERM,
    },
    ENCODINGS.NUMERIC: {
        NAPISTU_GRAPH_EDGES.STOICHIOMETRY,
        NAPISTU_GRAPH_EDGES.WEIGHT,
        NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHT,
    },
    ENCODINGS.BINARY: {
        NAPISTU_GRAPH_EDGES.R_ISREVERSIBLE,
    },
}

# splitting strategies

SPLITTING_STRATEGIES = SimpleNamespace(
    EDGE_MASK="edge_mask",
    VERTEX_MASK="vertex_mask",
    NO_MASK="no_mask",
    INDUCTIVE="inductive",
)

VALID_SPLITTING_STRATEGIES = list(SPLITTING_STRATEGIES.__dict__.values())
