from __future__ import annotations

from types import SimpleNamespace

NAPISTU_DATA = SimpleNamespace(
    EDGE_ATTR="edge_attr",
    EDGE_FEATURE_NAMES="edge_feature_names",
    EDGE_INDEX="edge_index",
    EDGE_WEIGHT="edge_weight",
    NG_EDGE_NAMES="ng_edge_names",
    NG_VERTEX_NAMES="ng_vertex_names",
    VERTEX_FEATURE_NAMES="vertex_feature_names",
    X="x",
    Y="y",
    NAME="name",
    SPLITTING_STRATEGY="splitting_strategy",
    LABELING_MANAGER="labeling_manager",
)

NAPISTU_DATA_DEFAULT_NAME = "default"

VERTEX_TENSOR = SimpleNamespace(
    DATA="data",
    FEATURE_NAMES="feature_names",
    VERTEX_NAMES="vertex_names",
    TENSOR_NAME="tensor_name",
    DESCRIPTION="description",
)

# defs in the json/config
NAPISTU_DATA_STORE = SimpleNamespace(
    # top-level categories
    NAPISTU_RAW="napistu_raw",
    NAPISTU_DATA="napistu_data",
    VERTEX_TENSORS="vertex_tensors",
    # attributes
    SBML_DFS="sbml_dfs",
    NAPISTU_GRAPH="napistu_graph",
    OVERWRITE="overwrite",
    # metadata
    LAST_MODIFIED="last_modified",
    CREATED="created",
    LABELS="labels",
    SPLITTING_STRATEGY=NAPISTU_DATA.SPLITTING_STRATEGY,
    LABELING_MANAGER=NAPISTU_DATA.LABELING_MANAGER,
    FILENAME="filename",
    PT_TEMPLATE="{name}.pt",
    TENSOR_NAME=VERTEX_TENSOR.TENSOR_NAME,
    DESCRIPTION=VERTEX_TENSOR.DESCRIPTION,
)

NAPISTU_DATA_STORE_STRUCTURE = SimpleNamespace(
    REGISTRY_FILE="registry.json",
    # file directories
    NAPISTU_RAW=NAPISTU_DATA_STORE.NAPISTU_RAW,
    NAPISTU_DATA=NAPISTU_DATA_STORE.NAPISTU_DATA,
    VERTEX_TENSORS=NAPISTU_DATA_STORE.VERTEX_TENSORS,
)
