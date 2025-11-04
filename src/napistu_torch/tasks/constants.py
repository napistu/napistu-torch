from types import SimpleNamespace

# constants are imported into napistu_torch.constants so don't declare imports here to avoid circular imports

TASKS = SimpleNamespace(
    EDGE_PREDICTION="edge_prediction",
    NETWORK_EMBEDDING="network_embedding",
    NODE_CLASSIFICATION="node_classification",
)

VALID_TASKS = list(TASKS.__dict__.values())

SUPERVISION = SimpleNamespace(
    SELF_SUPERVISED="self_supervised",
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

EDGE_PREDICTION_BATCH = SimpleNamespace(
    X="x",
    SUPERVISION_EDGES="supervision_edges",
    POS_EDGES="pos_edges",
    NEG_EDGES="neg_edges",
    EDGE_DATA="edge_data",
)
