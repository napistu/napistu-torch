from types import SimpleNamespace

from napistu.network.constants import NAPISTU_GRAPH_VERTICES

LABEL_TYPE = SimpleNamespace(
    SPECIES_TYPE=NAPISTU_GRAPH_VERTICES.SPECIES_TYPE,
    NODE_TYPE=NAPISTU_GRAPH_VERTICES.NODE_TYPE,
)

VALID_LABEL_TYPES = list(LABEL_TYPE.__dict__.values())

TASK_TYPES = SimpleNamespace(CLASSIFICATION="classification", REGRESSION="regression")

VALID_TASK_TYPES = list(TASK_TYPES.__dict__.values())

LABELING = SimpleNamespace(
    LABELED="labeled",
    UNLABELED="unlabeled",
)
