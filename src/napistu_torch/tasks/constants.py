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


NEGATIVE_SAMPLING_STRATEGIES = SimpleNamespace(
    UNIFORM="uniform",
    DEGREE_WEIGHTED="degree_weighted",
)

VALID_NEGATIVE_SAMPLING_STRATEGIES = list(
    NEGATIVE_SAMPLING_STRATEGIES.__dict__.values()
)
