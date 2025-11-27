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
    POS_RELATION_TYPE="pos_relation_type",
    NEG_RELATION_TYPE="neg_relation_type",
)


TASK_DESCRIPTIONS = {
    TASKS.EDGE_PREDICTION: """
This model performs **edge prediction** on biological pathway networks. Given node embeddings, 
the model predicts the likelihood of edges (interactions) between biological entities such as 
genes, proteins, and metabolites. This is useful for:

- Discovering novel biological interactions
- Validating experimentally observed interactions
- Completing incomplete pathway databases
- Predicting functional relationships between genes/proteins

The model learns to score potential edges based on learned embeddings of source and target nodes, 
optionally incorporating relation types for relation-aware prediction.
""".strip(),
    TASKS.NODE_CLASSIFICATION: """
This model performs **node classification** on biological pathway networks. Given node features 
and graph structure, the model predicts categorical labels for nodes (biological entities). 
This is useful for:

- Predicting gene/protein functions
- Classifying entities into biological pathways or processes
- Identifying disease-associated genes
- Annotating uncharacterized proteins

The model uses graph neural networks to aggregate information from neighboring nodes to make 
predictions about each node's class label.
""".strip(),
    TASKS.NETWORK_EMBEDDING: """
This model performs **network embedding** on biological pathway networks. Given a graph structure, 
the model learns low-dimensional representations of nodes (biological entities) that capture 
the network's topology and functional relationships. This is useful for:

- Discovering hidden patterns in biological networks
- Clustering related genes/proteins
- Visualizing biological networks
- Understanding functional modules
""".strip(),
}
