from types import SimpleNamespace

TASKS = SimpleNamespace(
    EDGE_PREDICTION="edge_prediction",
    NETWORK_EMBEDDING="network_embedding",
    NODE_CLASSIFICATION="node_classification",
)

VALID_TASKS = list(TASKS.__dict__.values())


SUPERVISION = SimpleNamespace(
    SUPERVISED="supervised",
    UNSUPERVISED="unsupervised",
)
